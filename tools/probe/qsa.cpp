// xenithium-probe qsa: QSA decode on real weights, both cards (heads split).
// xenithium-probe qsa --model DIR [--layers 3,23,47] [--iters I] [--kv-bytes 1|2] [--context N]
// Correctness (per requested layer): a history of ~2,400 tokens (past the 2,048-token budget) on
// the chain input (each step's x = that layer's attention-mixer output on the identity-block chain
// for 4 token ids, as probe gdn), mostly 4-row steps with some speculative ones; the last steps and
// the ones crossing the budget are checked teacher-forced: q / iq vs norm + rope of the device's
// projections, the K / V rows written vs the reference, scores vs the reference from the device's
// iq / kc, selection == the reference top-k of the device's keys, attn vs softmax over the
// device's selection and cache, out vs the sum of the per-card o_proj references; both cards
// bit-identical.
// Timing: every full-attention layer once per step, chained, at contexts 8K / 262K / 1M (or
// --context N): a cache of that length per layer filled with random K / V / compressed keys, the
// step at its end. Floor per layer per card = weights + the selected K / V of one row + every
// compressed key.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "xenithium/arch/qwen38/arch.hpp"
#include "xenithium/arch/qwen38/hc.hpp"
#include "xenithium/arch/qwen38/head.hpp"
#include "xenithium/arch/qwen38/qsa.hpp"
#include "xenithium/kernels/residual/hc.hpp"
#include "xenithium/kernels/pack.hpp"
#include "xenithium/kernels/attention/qsa.hpp"
#include "xenithium/load/dense_store.hpp"
#include "xenithium/model/arch.hpp"
#include "xenithium/model/bulk_read.hpp"
#include "xenithium/model/checkpoint.hpp"
#include "xenithium/model/config.hpp"
#include "xenithium/rt/fabric.hpp"

#include "warmup.hpp"
#include "kv.hpp"

namespace {
namespace m = xenithium::model;
namespace k = xenithium::kernels;
namespace q38 = xenithium::arch::qwen38;
using bf16 = sycl::ext::oneapi::bfloat16;

double ev_us(const sycl::event &e) {
  const auto a = e.get_profiling_info<sycl::info::event_profiling::command_start>();
  const auto b = e.get_profiling_info<sycl::info::event_profiling::command_end>();
  return static_cast<double>(b - a) * 1e-3;
}
double median(std::vector<double> v) {
  if (v.empty()) return 0;
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}
float bfr(float v) { return k::bf16_value(k::bf16_bits(v)); }

struct BoundCmp {
  double scaled = 0, w_dev = 0, w_ref = 0;
  size_t off = 0, n = 0;
  void add_row(const float *dev, const double *ref, int len) {
    double ss = 0;
    for (int i = 0; i < len; ++i) ss += ref[i] * ref[i];
    const double rms = std::sqrt(ss / len);
    for (int i = 0; i < len; ++i) {
      ++n;
      off += dev[i] != bfr(static_cast<float>(ref[i]));
      const double sc = std::fabs(dev[i] - ref[i]) / std::max({std::fabs(ref[i]), rms, 1e-30});
      if (!(sc <= scaled)) {
        scaled = sc;
        w_dev = dev[i];
        w_ref = ref[i];
      }
    }
  }
  void add_rowf(const float *dev, const float *ref, int len) {
    std::vector<double> r(ref, ref + len);
    add_row(dev, r.data(), len);
  }
  bool ok(double bound, double frac) const { return scaled <= bound && off <= frac * n; }
  std::string text() const {
    char b[160];
    std::snprintf(b, sizeof b, "worst %.2e of max(|ref|, rms) (dev %.6g ref %.6g), %zu / %zu differ (%.3f %%)", scaled, w_dev, w_ref, off, n,
                  100.0 * off / std::max<size_t>(1, n));
    return b;
  }
};

template <typename T>
std::vector<T> rd(sycl::queue &q, const T *p, size_t n) {
  std::vector<T> v(n);
  q.memcpy(v.data(), p, n * sizeof(T)).wait();
  return v;
}
std::vector<float> rd16(sycl::queue &q, const bf16 *p, size_t n) {
  const auto b = rd(q, reinterpret_cast<const uint16_t *>(p), n);
  std::vector<float> v(n);
  for (size_t i = 0; i < n; ++i) v[i] = k::bf16_value(b[i]);
  return v;
}

// Host reference pieces of one layer: norms, and each card's dequantised o_proj columns.
struct RefLayer {
  std::vector<float> qn, kn, iqn, ikn;
  std::vector<float> o[2];  // [hidden][o_k] per card
};
RefLayer ref_layer(const m::Checkpoint &ck, const q38::Config &cfg, int L, int threads) {
  const std::string P = std::string(q38::kPrefix) + "layers." + std::to_string(L) + ".self_attn.";
  std::vector<const m::TensorInfo *> ts;
  for (const char *t : {"q_norm.weight", "k_norm.weight", "indexer.q_layernorm.weight", "indexer.k_layernorm.weight", "o_proj.weight"})
    ts.push_back(&ck.at(P + t));
  m::BulkOptions bo;
  bo.threads = threads;
  const auto buf = m::bulk_read(ck, ts, bo);
  auto vals = [&](const char *t) {
    const auto &ti = ck.at(P + t);
    const uint16_t *u = buf.as<uint16_t>(ti);
    std::vector<float> v(ti.numel());
    for (size_t i = 0; i < v.size(); ++i) v[i] = k::bf16_value(u[i]);
    return v;
  };
  RefLayer r;
  r.qn = vals("q_norm.weight");
  r.kn = vals("k_norm.weight");
  r.iqn = vals("indexer.q_layernorm.weight");
  r.ikn = vals("indexer.k_layernorm.weight");
  const int H = cfg.hidden, QD = cfg.q_heads * cfg.head_dim, ok = QD / 2;
  const uint16_t *o = buf.as<uint16_t>(ck.at(P + "o_proj.weight"));
  for (int c = 0; c < 2; ++c) {
    std::vector<uint16_t> sl(static_cast<size_t>(H) * ok);
    for (int n = 0; n < H; ++n) std::memcpy(&sl[static_cast<size_t>(n) * ok], o + static_cast<size_t>(n) * QD + c * ok, ok * 2ull);
    std::vector<int8_t> codes(sl.size());
    std::vector<float> sc(H);
    k::quantize_i8_rows(sl.data(), H, ok, codes.data(), sc.data(), 1);
    r.o[c].resize(sl.size());
    for (int n = 0; n < H; ++n)
      for (int j = 0; j < ok; ++j) r.o[c][static_cast<size_t>(n) * ok + j] = codes[static_cast<size_t>(n) * ok + j] * sc[n];
  }
  return r;
}

// One card's cache + scratch for one layer.
struct Dev {
  k::QsaCache c;
  k::QsaScratch sc;
  std::vector<void *> mem;
};
Dev make_dev(xenithium::Fabric &fab, int card, const k::QsaShape &s, const k::QsaWeights &w, int cap, bool scratch) {
  Dev d;
  auto dev = [&](size_t bytes) {
    void *p = fab.alloc(card, std::max<size_t>(bytes, 64));
    d.mem.push_back(p);
    return p;
  };
  const size_t P = static_cast<size_t>(cap) / k::kQsaPage;
  d.c.k = static_cast<uint8_t *>(dev(P * k::qsa_page_kv_bytes(s)));
  d.c.v = static_cast<uint8_t *>(dev(P * k::qsa_page_v_bytes(s)));
  d.c.ks = static_cast<float *>(dev(P * k::qsa_page_scale_floats(s) * 4));
  d.c.vs = static_cast<float *>(dev(P * k::qsa_page_vscale_floats(s) * 4));
  d.c.kc = static_cast<bf16 *>(dev(P * k::qsa_page_kc_elems() * 2));
  d.c.ring = static_cast<bf16 *>(dev(k::kQsaRing * 128 * 2));
  std::vector<int32_t> pages(P);
  for (size_t i = 0; i < P; ++i) pages[i] = static_cast<int32_t>(i);
  int32_t *pg = static_cast<int32_t *>(dev(P * 4));
  fab.compute(card).memcpy(pg, pages.data(), P * 4).wait();
  d.c.pages = pg;
  d.c.max_tokens = cap;
  if (!scratch) return d;
  auto &sc = d.sc;
  sc.y = static_cast<float *>(dev(4ull * w.qkv.n * 4));
  sc.yi = static_cast<float *>(dev(4ull * w.idx.n * 4));
  sc.q = static_cast<bf16 *>(dev(4ull * s.hq() * 256 * 2));
  sc.iq = static_cast<bf16 *>(dev(4ull * 512 * 2));
  sc.nblk = static_cast<int32_t *>(dev(64));
  sc.score = static_cast<uint32_t *>(dev(k::qsa_score_words(cap) * 4));
  sc.cand_key = static_cast<uint32_t *>(dev(4ull * 16 * 512 * 4));
  sc.cand_id = static_cast<int32_t *>(dev(4ull * 16 * 512 * 4));
  sc.cand_n = static_cast<int32_t *>(dev(4096));
  sc.sel = static_cast<int32_t *>(dev(4ull * 512 * 4));
  sc.sel_n = static_cast<int32_t *>(dev(64));
  sc.part = static_cast<float *>(dev(k::qsa_part_floats(s) * 4));
  sc.attn = static_cast<bf16 *>(dev(4ull * s.o_k() * 2));
  sc.partial = static_cast<float *>(dev(4ull * s.hidden * 4));
  fab.compute(card).memset(sc.attn, 0, 4ull * s.o_k() * 2).wait();
  return d;
}
void free_dev(xenithium::Fabric &fab, Dev &d) {
  for (void *p : d.mem) fab.release(p);
  d.mem.clear();
}

}  // namespace

int cmd_qsa_run(const std::string &model, const std::string &layers_arg, int iters, int threads, int kv_bytes, int64_t context, bool verify) {
  const auto mc = m::ModelConfig::load(model);
  const auto arch = m::open_arch(mc);
  const q38::Config cfg = q38::Config::parse(mc);
  const m::Checkpoint ck = m::Checkpoint::open(model);
  std::vector<int> layers;
  {
    std::stringstream ss(layers_arg);
    std::string t;
    while (std::getline(ss, t, ','))
      if (!t.empty()) layers.push_back(std::stoi(t));
  }
  std::sort(layers.begin(), layers.end());
  for (int L : layers)
    if (L < 0 || L >= cfg.layers || !cfg.full_attention[L]) {
      std::fprintf(stderr, "qsa: layer %d is not a full-attention layer\n", L);
      return 2;
    }
  const k::QsaKv kv = probe_kv(kv_bytes);
  const bool i8 = k::qsa_kv_scaled(kv), v4 = k::qsa_kv_v4(kv);  // int8 K (V int8, or int4 with --kv-v4 1)
  const std::string P = q38::kPrefix;
  std::vector<m::DenseSpec> plan;
  for (const auto &sp : arch->dense_plan(ck, m::DeployParams{}))
    if (sp.comp == m::Component::hyper_connection || sp.comp == m::Component::attention || sp.name == P + "embed_tokens.weight" ||
        sp.name == "lm_head.weight")
      plan.push_back(sp);
  const auto loader = xenithium::l0::Loader::open();
  xenithium::Fabric fab(loader, {0, 1});
  xenithium::load::DenseStore dense(fab);
  dense.load(ck, plan, threads);
  const auto &cb = dense.stats().card_bytes[0];
  std::printf("qsa: %zu dense tensors loaded in %.2f s (attention %.3f GiB per card, heads split), KV %s\n", plan.size(), dense.stats().total_s,
              (cb.count(m::Component::attention) ? cb.at(m::Component::attention) : 0) / 1073741824.0, i8 ? "int8 + scale" : "bf16");
  q38::Head head(fab, dense, cfg);
  const q38::HcSet hs = q38::HcSet::from(dense, cfg, 2);
  const q38::QsaSet qs = q38::QsaSet::from(dense, cfg, 2, kv);
  const auto &sh = hs.shape;
  const int rows4 = q38::Head::kRows, H = sh.hidden;

  // Chain inputs: 64 steps of 4 token ids, the attention-mixer output of each requested layer.
  constexpr int kX = 64;
  bf16 *outA[2], *outB[2];
  std::vector<std::vector<bf16 *>> xin[2];  // [card][layer idx][step]
  for (int c = 0; c < 2; ++c) {
    outA[c] = static_cast<bf16 *>(fab.alloc(c, static_cast<size_t>(rows4) * H * 2));
    outB[c] = static_cast<bf16 *>(fab.alloc(c, static_cast<size_t>(rows4) * H * 2));
    xin[c].resize(layers.size());
    for (auto &v : xin[c])
      for (int s = 0; s < kX; ++s) v.push_back(static_cast<bf16 *>(fab.alloc(c, static_cast<size_t>(rows4) * H * 2)));
  }
  for (int s = 0; s < kX; ++s) {
    int32_t ids[rows4];
    for (int r = 0; r < rows4; ++r) ids[r] = static_cast<int32_t>((1009 * (s * 4 + r) + 13) % 120000);
    head.set_ids(ids, rows4);
    head.embed(rows4);
    for (int l = 0, li = 0; l <= layers.back(); ++l)
      for (int part = 0; part < 2; ++part) {
        for (int c = 0; c < 2; ++c) {
          auto &q = fab.compute(c);
          auto &st = head.state(c);
          const bf16 *blk = part == 0 ? (l > 0 ? outB[c] : nullptr) : outA[c];
          if (part == 0 && l == hs.ple_layer && blk) {
            k::hc_combine(q, sh, rows4, st, blk);
            blk = nullptr;
          }
          k::hc_mixer(q, sh, rows4, st, blk, part == 0 ? hs.attn[c][l] : hs.mlp[c][l], head.scratch(c), part == 0 ? outA[c] : outB[c]);
          if (part == 0 && li < static_cast<int>(layers.size()) && layers[li] == l) q.memcpy(xin[c][li][s], outA[c], static_cast<size_t>(rows4) * H * 2);
        }
        if (part == 0 && li < static_cast<int>(layers.size()) && layers[li] == l) ++li;
      }
    for (int c = 0; c < 2; ++c) fab.compute(c).wait_and_throw();
  }

  k::PairWire wire[2];
  bf16 *out[2], *rope[2], *rope_chk[2];
  int32_t *pos[2];
  constexpr int kCapCheck = 4096;
  const int64_t rope_n = std::max<int64_t>(kCapCheck, context > 0 ? context : k::kQsaMaxTokens);
  // Timing table: the load's static YaRN for the longest context timed (qsa_yarn_for: the smallest factor in steps of
  // 0.01 reaching it); the correctness runs use the unscaled table (the default).
  const k::QsaYarn yarn = k::qsa_yarn_for(rope_n, cfg.max_positions);
  const auto rope_h = k::qsa_rope_table(rope_n, 64, cfg.rope_theta, yarn);
  const auto rope_tab = k::qsa_rope_table(kCapCheck, 64, cfg.rope_theta, k::QsaYarn{});
  for (int c = 0; c < 2; ++c) {
    wire[c].inbox = static_cast<uint32_t *>(fab.alloc_mailbox(c, k::pair_wire_words(H) * 4));
    wire[c].error = static_cast<uint32_t *>(fab.alloc(c, 4096));
    out[c] = static_cast<bf16 *>(fab.alloc(c, 4ull * H * 2));
    pos[c] = static_cast<int32_t *>(fab.alloc(c, 64));
    rope[c] = static_cast<bf16 *>(fab.alloc(c, rope_h.size() * 2));
    fab.compute(c).memcpy(rope[c], rope_h.data(), rope_h.size() * 2).wait();
    rope_chk[c] = static_cast<bf16 *>(fab.alloc(c, rope_tab.size() * 2));
    fab.compute(c).memcpy(rope_chk[c], rope_tab.data(), rope_tab.size() * 2).wait();
  }
  for (int c = 0; c < 2; ++c) {
    wire[c].peer_inbox = wire[1 - c].inbox;
    k::pair_wire_init(fab.compute(c), H, wire[c].inbox).wait();
    fab.compute(c).memset(wire[c].error, 0, 4).wait();
  }
  int rc = 0;
  auto check = [&](bool ok, const std::string &what) {
    std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
    if (!ok) rc = 1;
  };
  int wire_parity = 0;
  char what[600];

  // ---- correctness on the chain input ----
  struct Step {
    int pos, rows, acc;
  };
  std::vector<Step> steps;
  for (int p = 0, i = 0; p < 2400; ++i) {
    const int rows = i % 9 == 4 ? 2 : 4, acc = i % 7 == 3 ? 1 : rows;
    steps.push_back({p, rows, acc});
    p += acc;
  }
  for (size_t li = 0; li < layers.size() && verify; ++li) {
    const int L = layers[li];
    const RefLayer rl = ref_layer(ck, cfg, L, threads);
    Dev d[2] = {make_dev(fab, 0, qs.shape_of(0), qs.w[0][L], kCapCheck, true), make_dev(fab, 1, qs.shape_of(1), qs.w[1][L], kCapCheck, true)};
    BoundCmp cq, ciq, ckv, ca, co;
    double serr = 0;
    bool sel_ok = true, same = true;
    int checked = 0;
    for (size_t i = 0; i < steps.size(); ++i) {
      const auto &st = steps[i];
      for (int c = 0; c < 2; ++c) {
        fab.compute(c).memcpy(pos[c], &st.pos, 4);
        k::qsa_decode(fab.compute(c), qs.shape_of(c), st.rows, xin[c][li][i % kX], qs.w[c][L], d[c].c, rope_chk[c], k::QsaStep{pos[c], st.pos + st.rows},
                      d[c].sc, &wire[c], wire_parity, out[c]);
      }
      wire_parity ^= 1;
      for (int c = 0; c < 2; ++c) fab.compute(c).wait_and_throw();
      const bool chk = i + 3 >= steps.size() || (st.pos >= 2040 && st.pos <= 2052);
      if (!chk) continue;
      ++checked;
      const auto o0 = rd16(fab.compute(0), out[0], 4ull * H), o1 = rd16(fab.compute(1), out[1], 4ull * H);
      same &= std::equal(o0.begin(), o0.begin() + static_cast<size_t>(st.rows) * H, o1.begin());
      std::vector<double> oref(static_cast<size_t>(st.rows) * H, 0.0);
      for (int c = 0; c < 2; ++c) {
        auto &q = fab.compute(c);
        const auto s = qs.shape_of(c);
        const auto &w = qs.w[c][L];
        const int ldy = w.qkv.n, ldi = w.idx.n, hq = s.hq(), D = 256;
        const auto y = rd(q, d[c].sc.y, 4ull * ldy), yi = rd(q, d[c].sc.yi, 4ull * ldi);
        const auto qd = rd16(q, d[c].sc.q, 4ull * hq * D), iqd = rd16(q, d[c].sc.iq, 4ull * 512);
        const auto attn = rd16(q, d[c].sc.attn, 4ull * s.o_k());
        const auto keys = rd(q, d[c].sc.score, k::qsa_score_words(kCapCheck));
        const auto sel = rd(q, d[c].sc.sel, 4ull * 512);
        const auto seln = rd(q, d[c].sc.sel_n, 4);
        const int stride = static_cast<int>(k::qsa_score_words(kCapCheck) / 4);
        const size_t NP = kCapCheck / 64;
        const auto kp = rd(q, d[c].c.k, NP * k::qsa_page_kv_bytes(s)), vp = rd(q, d[c].c.v, NP * k::qsa_page_v_bytes(s));
        const auto ksp = rd(q, d[c].c.ks, NP * 64), vsp = rd(q, d[c].c.vs, NP * k::qsa_page_vscale_floats(s));
        const auto kcp = rd16(q, d[c].c.kc, NP * k::qsa_page_kc_elems());
        auto kvv = [&](const std::vector<uint8_t> &pool, const std::vector<float> &sp, int tok, int dd) {
          if (i8) return static_cast<float>(static_cast<int8_t>(pool[static_cast<size_t>(tok) * D + dd])) * sp[tok];
          uint16_t b;
          std::memcpy(&b, &pool[(static_cast<size_t>(tok) * D + dd) * 2], 2);
          return k::bf16_value(b);
        };
        auto vv4 = [&](int tok, int dd) {  // V: int8-v4's int4 codes, two per byte (dim 2j low)
          if (!v4) return kvv(vp, vsp, tok, dd);
          const int byte = vp[static_cast<size_t>(tok) * (D / 2) + dd / 2], nib = dd & 1 ? byte >> 4 : byte & 0xF;
          uint16_t sb;  // bf16 scales, one per 32 dims
          std::memcpy(&sb, reinterpret_cast<const uint8_t *>(vsp.data()) + (static_cast<size_t>(tok) * k::kQsaV4Scales + dd / k::kQsaV4Group) * 2, 2);
          return static_cast<float>(nib >= 8 ? nib - 16 : nib) * k::bf16_value(sb);
        };
        for (int r = 0; r < st.rows; ++r) {
          const int p = st.pos + r;
          const uint16_t *rr = &rope_tab[static_cast<size_t>(p) * 64];
          std::vector<float> v(D), o(D);
          for (int h = 0; h < hq; ++h) {
            for (int j = 0; j < D; ++j) v[j] = bfr(y[static_cast<size_t>(r) * ldy + h * 2 * D + j]);
            k::qsa_ref_norm_rope(v.data(), rl.qn.data(), D, rr, s.eps, o.data());
            cq.add_rowf(&qd[(static_cast<size_t>(r) * hq + h) * D], o.data(), D);
          }
          for (int h = 0; h < 4; ++h) {
            for (int j = 0; j < 128; ++j) v[j] = bfr(yi[static_cast<size_t>(r) * ldi + h * 128 + j]);
            k::qsa_ref_norm_rope(v.data(), rl.iqn.data(), 128, rr, s.eps, o.data());
            ciq.add_rowf(&iqd[(static_cast<size_t>(r) * 4 + h) * 128], o.data(), 128);
          }
          {  // this row's K as stored
            for (int j = 0; j < D; ++j) v[j] = bfr(y[static_cast<size_t>(r) * ldy + s.q_n() + j]);
            k::qsa_ref_norm_rope(v.data(), rl.kn.data(), D, rr, s.eps, o.data());
            std::vector<float> kref(o), kdev(D);
            if (i8) {
              std::vector<int8_t> cd(D);
              float scl;
              k::qsa_ref_i8(o.data(), D, cd.data(), &scl);
              for (int j = 0; j < D; ++j) kref[j] = static_cast<float>(cd[j]) * scl;
            }
            for (int j = 0; j < D; ++j) kdev[j] = kvv(kp, ksp, p, j);
            ckv.add_rowf(kdev.data(), kref.data(), D);
          }
          const int nblk = (p + 1) / 4;
          for (int b = 0; b < nblk; ++b) {
            float dv;
            std::memcpy(&dv, &keys[static_cast<size_t>(r) * stride + b], 4);
            const float *kcb = &kcp[static_cast<size_t>(b) * 128];
            const double ref = k::qsa_ref_score(&iqd[static_cast<size_t>(r) * 512], kcb);
            double mag = 0;
            for (int j = 0; j < 512; ++j) mag += std::fabs(static_cast<double>(iqd[static_cast<size_t>(r) * 512 + j]) * kcb[j % 128]);
            serr = std::max(serr, std::fabs(dv - ref) / (mag / std::sqrt(128.0) + 1e-30));
          }
          const auto want = k::qsa_ref_select(&keys[static_cast<size_t>(r) * stride], nblk, 512);
          const std::vector<int> got(sel.begin() + r * 512, sel.begin() + r * 512 + seln[r]);
          sel_ok &= got == want;
          const auto toks = k::qsa_ref_tokens(got, p);
          std::vector<float> K(toks.size() * D), V(toks.size() * D);
          for (size_t t = 0; t < toks.size(); ++t)
            for (int j = 0; j < D; ++j) {
              K[t * D + j] = kvv(kp, ksp, toks[t], j);
              V[t * D + j] = vv4(toks[t], j);
            }
          for (int h = 0; h < hq; ++h) {
            std::vector<double> oo(D);
            k::qsa_ref_attend(&qd[(static_cast<size_t>(r) * hq + h) * D], static_cast<int>(toks.size()), K.data(), V.data(), oo.data());
            std::vector<float> gate(D), ga(D);
            for (int j = 0; j < D; ++j) gate[j] = bfr(y[static_cast<size_t>(r) * ldy + h * 2 * D + D + j]);
            k::qsa_ref_gate(oo.data(), gate.data(), ga.data());
            ca.add_rowf(&attn[(static_cast<size_t>(r) * hq + h) * D], ga.data(), D);
          }
          for (int n = 0; n < H; ++n) {
            double acc = 0;
            for (int j = 0; j < s.o_k(); ++j) acc += static_cast<double>(attn[static_cast<size_t>(r) * s.o_k() + j]) * rl.o[c][static_cast<size_t>(n) * s.o_k() + j];
            oref[static_cast<size_t>(r) * H + n] += acc;
          }
        }
      }
      for (int r = 0; r < st.rows; ++r) co.add_row(&o0[static_cast<size_t>(r) * H], &oref[static_cast<size_t>(r) * H], H);
    }
    std::printf("\nlayer %d (%zu steps to position %d, rows 1-4, %d checked, chain input):\n", L, steps.size(), steps.back().pos + steps.back().rows,
                checked);
    std::snprintf(what, sizeof what, "layer %d q vs norm + rope of the device's projections: %s", L, cq.text().c_str());
    check(cq.ok(1.0 / 128, 0.01), what);
    std::snprintf(what, sizeof what, "layer %d iq: %s", L, ciq.text().c_str());
    check(ciq.ok(1.0 / 128, 0.01), what);
    std::snprintf(what, sizeof what, "layer %d K rows as stored vs reference: %s", L, ckv.text().c_str());
    check(ckv.ok(i8 ? 1.0 / 16 : 1.0 / 128, i8 ? 1.0 : 0.01), what);  // int8 x scale is no bf16 value: the bound decides
    std::snprintf(what, sizeof what, "layer %d scores: worst error %.2e of sum |products|; selection == reference top-k of the device's keys", L, serr);
    check(serr <= 1e-6 && sel_ok, what);
    std::snprintf(what, sizeof what, "layer %d attn (gated) vs reference over the device's selection: %s", L, ca.text().c_str());
    check(ca.ok(1.0 / 64, 0.005), what);
    std::snprintf(what, sizeof what, "layer %d out vs the sum of the per-card o_proj references: %s; both cards bit-identical", L, co.text().c_str());
    check(co.ok(1.0 / 128, 0.01) && same, what);
    for (int c = 0; c < 2; ++c) free_dev(fab, d[c]);
  }

  // ---- timing: every full-attention layer once per step, chained ----
  const int NL = static_cast<int>(qs.layers.size());
  const auto s0 = qs.shape_of(0);
  const auto &w0 = qs.w[0][qs.layers[0]];
  const double wbytes = static_cast<double>(w0.qkv.bytes() + w0.idx.bytes() + w0.o.bytes());
  const double floor_bw = 605.9e9;
  std::vector<int64_t> ctxs = context > 0 ? std::vector<int64_t>{context} : std::vector<int64_t>{8192, 262144, 1048576};
  for (const int64_t ctx : ctxs) {
    const int cap = static_cast<int>((ctx + 63) / 64 * 64);
    if (cap > rope_n) {
      std::printf("context %lld: no rotary table that long\n", static_cast<long long>(ctx));
      continue;
    }
    std::vector<Dev> dv[2];
    Dev scr[2];
    for (int c = 0; c < 2; ++c) {
      scr[c] = make_dev(fab, c, qs.shape_of(c), qs.w[c][qs.layers[0]], cap, true);
      for (int i = 0; i < NL; ++i) dv[c].push_back(make_dev(fab, c, qs.shape_of(c), qs.w[c][qs.layers[i]], cap, false));
    }
    {  // random history: K / V codes (no NaN codes), unit-ish scales, compressed keys
      const size_t NP = cap / 64;
      std::vector<uint8_t> kvh(NP * k::qsa_page_kv_bytes(s0));
      std::vector<float> scl(NP * 64);
      std::vector<uint16_t> kch(NP * k::qsa_page_kc_elems());
      uint64_t x = 88172645463325252ull;
      auto nx = [&]() {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        return x;
      };
      for (size_t i = 0; i < kvh.size(); i += 8) {
        uint64_t v = nx();
        for (int j = 0; j < 8 && i + j < kvh.size(); ++j) {
          uint8_t b = static_cast<uint8_t>(v >> (8 * j));
          kvh[i + j] = i8 ? b : static_cast<uint8_t>(j & 1 ? (b & 0xBF) | 0x3C : b);
        }
      }
      for (auto &v : scl) v = 1.f / 64;
      std::normal_distribution<float> nd(0.f, 1.f);
      std::mt19937 rng(3);
      for (auto &v : kch) v = k::bf16_bits(nd(rng));
      for (int c = 0; c < 2; ++c)
        for (auto &dd : dv[c]) {
          auto &q = fab.compute(c);
          q.memcpy(dd.c.k, kvh.data(), kvh.size());
          q.memcpy(dd.c.v, kvh.data(), NP * k::qsa_page_v_bytes(s0));  // int8-v4: half of it (any byte = two codes)
          q.memcpy(dd.c.ks, scl.data(), NP * 64 * 4);
          if (v4) {  // bf16 scales 1 / 64
            std::vector<uint16_t> vs4(NP * k::qsa_page_vscale_floats(s0) * 2, k::bf16_bits(1.f / 64));
            q.memcpy(dd.c.vs, vs4.data(), vs4.size() * 2).wait();
          } else {
            q.memcpy(dd.c.vs, scl.data(), NP * 64 * 4);
          }
          q.memcpy(dd.c.kc, kch.data(), kch.size() * 2).wait();
        }
    }
    const double kv_row = 2051.0 * (256 * s0.kv_elem() + s0.v_row_bytes() + (i8 ? 8 : 0)) * s0.hkv();
    const double kc_bytes = static_cast<double>(cap) / 4 * 256;
    const double bytes = wbytes + kv_row + kc_bytes;
    const double floor_us = bytes / floor_bw * 1e6;
    std::printf("\ncontext %d: floor per layer per card %.2f MB (weights %.2f, selected K/V of one row %.2f, compressed keys %.2f) -> %.1f us; "
                "x%d layers %.0f us\n",
                cap, bytes / 1e6, wbytes / 1e6, kv_row / 1e6, kc_bytes / 1e6, floor_us, NL, floor_us * NL);
    std::printf("yarn %.2f; rows |   qkv    idx   prep  score select smerge   attn  merge      o reduce (us, card 0 median) | GPU per call: card 0  card 1 | x floor\n", cap > cfg.max_positions ? yarn.factor : 1.0);
    for (const int R : {1, 4}) {
      const int p0 = cap - R;
      for (int c = 0; c < 2; ++c) fab.compute(c).memcpy(pos[c], &p0, 4).wait();
      std::vector<double> kt[10], gspan[2];
      const int reps = std::max(4, iters / 10);
      xenithium::probe::warm_for([&] {  // >= 100 ms of the same chain: clocks up before timing
        for (int i = 0; i < NL; ++i)
          for (int c = 0; c < 2; ++c) {
            k::qsa_decode(fab.compute(c), qs.shape_of(c), R, xin[c][0][0], qs.w[c][qs.layers[i]], dv[c][i].c, rope[c], k::QsaStep{pos[c], cap},
                          scr[c].sc, &wire[c], wire_parity, out[c]);
            if (c == 1) wire_parity ^= 1;
          }
        for (int c = 0; c < 2; ++c) fab.compute(c).wait_and_throw();
      });
      for (int rep = 0; rep < reps + 2; ++rep) {
        std::vector<k::QsaEvents> ev[2];
        for (int i = 0; i < NL; ++i)
          for (int c = 0; c < 2; ++c) {
            k::QsaScratch sc = scr[c].sc;
            ev[c].push_back(k::qsa_decode(fab.compute(c), qs.shape_of(c), R, xin[c][0][rep % kX], qs.w[c][qs.layers[i]], dv[c][i].c, rope[c],
                                          k::QsaStep{pos[c], cap}, sc, &wire[c], wire_parity, out[c]));
            if (c == 1) wire_parity ^= 1;
          }
        for (int c = 0; c < 2; ++c) fab.compute(c).wait_and_throw();
        if (rep < 2) continue;
        for (int i = 2; i < NL; ++i) {
          const auto &e = ev[0][i];
          const sycl::event *es[10] = {&e.qkv, &e.idx, &e.prep, &e.score, &e.select, &e.merge_sel, &e.attn, &e.merge, &e.o, &e.reduce};
          for (int j = 0; j < 10; ++j) kt[j].push_back(j == 5 && e.merge_sel == e.select ? 0.0 : ev_us(*es[j]));  // one select launch: no merge
        }
        for (int c = 0; c < 2; ++c) {
          const auto a = ev[c][1].reduce.get_profiling_info<sycl::info::event_profiling::command_end>();
          const auto b = ev[c][NL - 1].reduce.get_profiling_info<sycl::info::event_profiling::command_end>();
          gspan[c].push_back(static_cast<double>(b - a) * 1e-3 / (NL - 2));
        }
      }
      std::printf("        %4d | %6.2f %6.2f %6.2f %6.2f %6.2f %6.2f %6.2f %6.2f %6.2f %6.2f                    | %20.1f %7.1f | %.2f\n", R, median(kt[0]),
                  median(kt[1]), median(kt[2]), median(kt[3]), median(kt[4]), median(kt[5]), median(kt[6]), median(kt[7]), median(kt[8]),
                  median(kt[9]), median(gspan[0]), median(gspan[1]), std::max(median(gspan[0]), median(gspan[1])) / floor_us);
    }
    for (int c = 0; c < 2; ++c) {
      free_dev(fab, scr[c]);
      for (auto &dd : dv[c]) free_dev(fab, dd);
    }
  }
  uint32_t err[2];
  for (int c = 0; c < 2; ++c) fab.compute(c).memcpy(&err[c], wire[c].error, 4).wait();
  check(!err[0] && !err[1], "qsa mailbox: no timeout");
  return rc;
}
