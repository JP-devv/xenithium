// xenithium-probe gqa35: the qwen35 gated full-attention decode block on real weights, both cards.
// xenithium-probe gqa35 --model DIR [--layers 3,31,63] [--iters I] [--kv-bytes 1|2] [--context N]
// Loads every full-attention layer (dense plan, heads split: q_proj | k | v as one W4 matrix per card, o_proj's k half
// as W4, q / k norms replicated). Correctness per requested layer: a history of ~4,200 tokens (past 4,096: the context
// cut goes from 1 to 2 tiles per split) of ~N(0, 1) bf16 inputs, steps of 1-4 rows with accepts 1-4; on the first
// steps, across 4,096 and at the end, teacher-forced: the W4 qkv vs the dequantised host slices (once), q vs norm +
// rope of the device's projections, the K / V rows as stored, attn vs softmax over the whole context as stored + the
// gate, out vs the sum of the per-card o_proj references from the device's attn; both cards bit-identical.
// Timing: every full-attention layer once per step, chained, at contexts 8K / 32K / 128K / 262K (or --context N): a
// cache of that length per layer filled with random K / V, the step at its end, 1 and 4 rows. Floor per layer per
// card = the W4 weights + the whole K / V of the context (+ int8 scales).
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "probe.hpp"
#include "qwen35_host.hpp"
#include "warmup.hpp"
#include "kv.hpp"
#include "xenithium/arch/qwen35/arch.hpp"
#include "xenithium/arch/qwen35/attn.hpp"
#include "xenithium/kernels/attention/gqa.hpp"
#include "xenithium/kernels/pack.hpp"
#include "xenithium/load/dense_store.hpp"
#include "xenithium/model/arch.hpp"
#include "xenithium/model/bulk_read.hpp"
#include "xenithium/model/checkpoint.hpp"
#include "xenithium/model/config.hpp"
#include "xenithium/rt/fabric.hpp"

namespace {
namespace m = xenithium::model;
namespace k = xenithium::kernels;
namespace q35 = xenithium::arch::qwen35;
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
  bool ok(double bound, double frac) const { return n > 0 && scaled <= bound && off <= frac * n; }
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

// Card c's reference pieces of layer L: the W4 host slices dequantised ([n][k] floats), the norms as values.
struct RefLayer {
  std::vector<float> qkv[2], o[2], qn, kn;
};
RefLayer ref_layer(const m::Checkpoint &ck, const q35::Config &cfg, const std::vector<m::DenseSpec> &plan, int L, const q35::AttnSet &as) {
  const std::string P = std::string(q35::kPrefix) + "layers." + std::to_string(L) + ".self_attn.";
  const m::DenseSpec *sq = nullptr, *so = nullptr;
  for (const auto &sp : plan) {
    if (sp.name == cfg.packed_name(P + "q_proj")) sq = &sp;
    if (sp.name == cfg.packed_name(P + "o_proj")) so = &sp;
  }
  if (!sq || !so) throw std::runtime_error("gqa35: layer " + std::to_string(L) + " not in the plan");
  const auto srcs_q = w4_sources(ck, *sq), srcs_o = w4_sources(ck, *so);
  const auto buf_q = m::bulk_read(ck, srcs_q), buf_o = m::bulk_read(ck, srcs_o);
  RefLayer r;
  for (int c = 0; c < 2; ++c) {
    const auto s = as.shape_of(c);
    const HostW4 hq = host_w4_slice(ck, *sq, buf_q, srcs_q, c, 2, s.qkv_n(), s.hidden);
    const HostW4 ho = host_w4_slice(ck, *so, buf_o, srcs_o, c, 2, s.hidden, s.o_k());
    if (hq.rows != s.qkv_n() || ho.rows != s.hidden) throw std::runtime_error("gqa35: host slice rows");
    r.qkv[c] = deq_w4(hq, s.qkv_n(), s.hidden);
    r.o[c] = deq_w4(ho, s.hidden, s.o_k());
  }
  std::vector<const m::TensorInfo *> ts = {&ck.at(P + "q_norm.weight"), &ck.at(P + "k_norm.weight")};
  const auto buf = m::bulk_read(ck, ts);
  auto vals = [&](const char *t) {
    const auto &ti = ck.at(P + t);
    const uint16_t *u = buf.as<uint16_t>(ti);
    std::vector<float> v(ti.numel());
    for (size_t i = 0; i < v.size(); ++i) v[i] = k::bf16_value(u[i]);
    return v;
  };
  r.qn = vals("q_norm.weight");
  r.kn = vals("k_norm.weight");
  return r;
}

// One card's cache + scratch for one layer.
struct Dev {
  k::GqaCache c;
  k::GqaScratch sc;
  std::vector<void *> mem;
};
Dev make_dev(xenithium::Fabric &fab, int card, const k::GqaShape &s, int cap, bool scratch) {
  Dev d;
  auto dev = [&](size_t bytes) {
    void *p = fab.alloc(card, std::max<size_t>(bytes, 4096));
    d.mem.push_back(p);
    return p;
  };
  const size_t P = static_cast<size_t>(cap) / k::kQsaPage;
  d.c.k = static_cast<uint8_t *>(dev(P * k::gqa_page_kv_bytes(s)));
  d.c.v = static_cast<uint8_t *>(dev(P * k::gqa_page_v_bytes(s)));
  d.c.ks = static_cast<float *>(dev(P * k::gqa_page_scale_floats(s) * 4));
  d.c.vs = static_cast<float *>(dev(P * k::gqa_page_vscale_floats(s) * 4));
  std::vector<int32_t> pages(P);
  for (size_t i = 0; i < P; ++i) pages[i] = static_cast<int32_t>(i);
  auto *pg = static_cast<int32_t *>(dev(P * 4));
  fab.compute(card).memcpy(pg, pages.data(), P * 4).wait();
  d.c.pages = pg;
  d.c.max_tokens = cap;
  if (!scratch) return d;
  auto &sc = d.sc;
  sc.y = static_cast<float *>(dev(8ull * s.qkv_n() * 4));
  sc.q = static_cast<bf16 *>(dev(8ull * s.hq() * 256 * 2));
  sc.part = static_cast<float *>(dev(k::gqa_part_floats(s) * 4));
  sc.attn = static_cast<bf16 *>(dev(8ull * s.o_k() * 2));
  sc.partial = static_cast<float *>(dev(8ull * s.hidden * 4));
  fab.compute(card).memset(sc.attn, 0, 8ull * s.o_k() * 2).wait();
  return d;
}
void free_dev(xenithium::Fabric &fab, Dev &d) {
  for (void *p : d.mem) fab.release(p);
  d.mem.clear();
}
float kv_at(const k::GqaShape &s, const std::vector<uint8_t> &pool, const std::vector<float> &sc, int t, int kvh, int dd, bool isv = false) {
  const size_t slot = (static_cast<size_t>(t / 64) * s.hkv() + kvh) * 64 + t % 64;
  if (isv && s.kv == k::QsaKv::i8v4) {  // A nibble code x its group's bf16 scale
    const int b = pool[slot * 128 + dd / 2], c = (dd & 1) ? (b >> 4) - ((b & 128) >> 3) : (b & 15) - ((b & 8) << 1);
    uint16_t sb;
    std::memcpy(&sb, reinterpret_cast<const uint8_t *>(sc.data()) + (slot * 8 + dd / 32) * 2, 2);
    return static_cast<float>(c) * k::bf16_value(sb);
  }
  if (s.kv != k::QsaKv::bf16) return static_cast<float>(static_cast<int8_t>(pool[slot * 256 + dd])) * sc[slot];
  uint16_t b;
  std::memcpy(&b, &pool[(slot * 256 + dd) * 2], 2);
  return k::bf16_value(b);
}

}  // namespace

int cmd_gqa35(const std::string &model, const std::string &layers_arg, int iters, int threads, int kv_bytes, int64_t context) {
  const auto mc = m::ModelConfig::load(model);
  const auto arch = m::open_arch(mc);
  if (arch->id() != "qwen35") {
    std::fprintf(stderr, "gqa35: %s is arch %s, not qwen35\n", model.c_str(), std::string(arch->id()).c_str());
    return 2;
  }
  const q35::Config cfg = q35::Config::parse(mc);
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
      std::fprintf(stderr, "gqa35: layer %d is not a full-attention layer\n", L);
      return 2;
    }
  const k::QsaKv kv = probe_kv(kv_bytes);
  const bool i8 = kv != k::QsaKv::bf16, v4 = kv == k::QsaKv::i8v4;  // K int8
  std::vector<m::DenseSpec> plan;
  for (const auto &sp : arch->dense_plan(ck, m::DeployParams{}))
    if (sp.comp == m::Component::attention) plan.push_back(sp);
  const auto loader = xenithium::l0::Loader::open();
  xenithium::Fabric fab(loader, {0, 1});
  xenithium::load::DenseStore dense(fab);
  dense.load(ck, plan, threads);
  const auto &cb = dense.stats().card_bytes[0];
  std::printf("gqa35: %zu dense tensors (every full-attention layer) loaded in %.2f s (%.3f GiB per card, heads split), KV %s\n", plan.size(),
              dense.stats().total_s, (cb.count(m::Component::attention) ? cb.at(m::Component::attention) : 0) / 1073741824.0,
              v4 ? "int8 K + int4 V (int8-v4)" : i8 ? "int8 + scale" : "bf16");
  q35::AttnSet as = q35::AttnSet::from(dense, cfg, 2, kv);
  // The decode attention numerics (--dec-attn, the server's names)
  const bool dec_exact = ::g_dec_attn == "exact";
  as.shape.dec_qk = ::g_dec_attn.rfind("int8", 0) == 0 ? 1 : ::g_dec_attn == "fast" && i8 ? 2 : 0;
  as.shape.dec_p_lo = ::g_dec_attn == "exact" || ::g_dec_attn == "int8";
  if (::g_dec_attn == "phi") as.shape.dec_qk = 0, as.shape.dec_p_lo = false;  // P hi only on the exact q.k (A/B)
  if (::g_dec_attn == "fast-i8") as.shape.dec_qk = 2, as.shape.dec_p_lo = false;
  if (!dec_exact) std::printf("gqa35: decode attention %s (dec_qk %d, P %s)\n", ::g_dec_attn.c_str(), as.shape.dec_qk, as.shape.dec_p_lo ? "hi + lo" : "hi");
  const int H = cfg.hidden;
  const auto s0 = as.shape_of(0);
  std::printf("gqa35: %d q / %d kv heads (per card %d / %d, group %d), head_dim %d, %zu attention layers\n", cfg.q_heads, cfg.kv_heads, s0.hq(), s0.hkv(),
              s0.grp(), cfg.head_dim, as.layers.size());

  // Inputs: 64 x 8 rows of ~N(0, 1) bf16, the same on both cards.
  constexpr int kX = 64;
  std::vector<std::vector<uint16_t>> xh(kX);
  std::vector<bf16 *> xin[2];
  {
    std::mt19937 rng(21);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (int s = 0; s < kX; ++s) {
      xh[s].resize(8ull * H);
      for (auto &v : xh[s]) v = k::bf16_bits(nd(rng));
      for (int c = 0; c < 2; ++c) {
        auto *d = static_cast<bf16 *>(fab.alloc(c, xh[s].size() * 2));
        fab.compute(c).memcpy(d, xh[s].data(), xh[s].size() * 2).wait();
        xin[c].push_back(d);
      }
    }
  }
  k::PairWire wire[2];
  bf16 *out[2], *rope[2];
  int32_t *pos[2];
  const int64_t rope_n = std::max<int64_t>(8192, context > 0 ? context : 262144);
  if (rope_n > k::kQsaMaxTokens) {
    std::fprintf(stderr, "gqa35: context %lld above the kernels' 1,048,576 tokens\n", static_cast<long long>(rope_n));
    return 2;
  }
  // past the model's positions: the load's static YaRN (qsa_yarn_for), as the server builds it
  const auto rope_h = k::qsa_rope_table(rope_n, 64, cfg.rope_theta, k::qsa_yarn_for(rope_n, cfg.max_positions));
  for (int c = 0; c < 2; ++c) {
    wire[c].inbox = static_cast<uint32_t *>(fab.alloc_mailbox(c, k::pair_wire_words(H) * 4));
    wire[c].error = static_cast<uint32_t *>(fab.alloc(c, 4096));
    out[c] = static_cast<bf16 *>(fab.alloc(c, 8ull * H * 2));
    pos[c] = static_cast<int32_t *>(fab.alloc(c, 4096));
    rope[c] = static_cast<bf16 *>(fab.alloc(c, rope_h.size() * 2));
    fab.compute(c).memcpy(rope[c], rope_h.data(), rope_h.size() * 2).wait();
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

  // ---- correctness ----
  constexpr int kCapCheck = 4608, kEnd = 4200;
  struct Step {
    int pos, rows, acc;
  };
  std::vector<Step> steps;
  for (int p = 0, i = 0; p < kEnd; ++i) {
    const int rows = i % 9 == 4 ? 2 : (i % 13 == 6 ? 1 : 4), acc = i % 7 == 3 ? 1 : rows;
    steps.push_back({p, rows, acc});
    p += acc;
  }
  for (int L : layers) {
    const RefLayer rl = ref_layer(ck, cfg, plan, L, as);
    Dev d[2] = {make_dev(fab, 0, as.shape_of(0), kCapCheck, true), make_dev(fab, 1, as.shape_of(1), kCapCheck, true)};
    BoundCmp cp, cq, ckv, ca, co;
    bool same = true;
    int checked = 0;
    for (size_t i = 0; i < steps.size(); ++i) {
      const auto &st = steps[i];
      for (int c = 0; c < 2; ++c) {
        fab.compute(c).memcpy(pos[c], &st.pos, 4);
        k::gqa_decode(fab.compute(c), as.shape_of(c), st.rows, xin[c][i % kX], as.w[c][L], d[c].c, rope[c], k::GqaStep{pos[c], st.pos + st.rows},
                      d[c].sc, &wire[c], wire_parity, out[c]);
      }
      wire_parity ^= 1;
      for (int c = 0; c < 2; ++c) fab.compute(c).wait_and_throw();
      const bool chk = i < 2 || i + 2 >= steps.size() || (st.pos >= 4090 && st.pos <= 4100);
      if (!chk) continue;
      ++checked;
      const auto o0 = rd16(fab.compute(0), out[0], static_cast<size_t>(st.rows) * H), o1 = rd16(fab.compute(1), out[1], static_cast<size_t>(st.rows) * H);
      same &= o0 == o1;
      std::vector<double> oref(static_cast<size_t>(st.rows) * H, 0.0);
      for (int c = 0; c < 2; ++c) {
        auto &q = fab.compute(c);
        const auto s = as.shape_of(c);
        const int ldy = s.qkv_n(), hq = s.hq(), hkv = s.hkv(), G = s.grp(), D = 256;
        const auto y = rd(q, d[c].sc.y, static_cast<size_t>(st.rows) * ldy);
        const auto qd = rd16(q, d[c].sc.q, static_cast<size_t>(st.rows) * hq * D);
        const auto attn = rd16(q, d[c].sc.attn, static_cast<size_t>(st.rows) * s.o_k());
        const size_t NP = kCapCheck / 64;
        const auto kp = rd(q, d[c].c.k, NP * k::gqa_page_kv_bytes(s)), vp = rd(q, d[c].c.v, NP * k::gqa_page_v_bytes(s));
        std::vector<float> ksp(NP * k::gqa_page_scale_floats(s), 1.f), vsp(NP * k::gqa_page_vscale_floats(s), 1.f);
        if (i8) ksp = rd(q, d[c].c.ks, ksp.size()), vsp = rd(q, d[c].c.vs, vsp.size());
        for (int r = 0; r < st.rows; ++r) {
          const int p = st.pos + r;
          const uint16_t *rr = &rope_h[static_cast<size_t>(p) * 64];
          std::vector<float> v(D), o(D);
          if (i == 0) {  // the W4 qkv vs the dequantised host slice from x
            const auto &xr = xh[i % kX];
            std::vector<double> ref(ldy);
            for (int n = 0; n < ldy; ++n) {
              double a = 0;
              const float *wr = &rl.qkv[c][static_cast<size_t>(n) * H];
              for (int j = 0; j < H; ++j) a += static_cast<double>(k::bf16_value(xr[static_cast<size_t>(r) * H + j])) * wr[j];
              ref[n] = a;
            }
            cp.add_row(&y[static_cast<size_t>(r) * ldy], ref.data(), ldy);
          }
          for (int h = 0; h < hq; ++h) {
            for (int j = 0; j < D; ++j) v[j] = bfr(y[static_cast<size_t>(r) * ldy + h * 2 * D + j]);
            k::qsa_ref_norm_rope(v.data(), rl.qn.data(), D, rr, s.eps, o.data());
            cq.add_rowf(&qd[(static_cast<size_t>(r) * hq + h) * D], o.data(), D);
          }
          for (int kh = 0; kh < hkv; ++kh) {
            for (int j = 0; j < D; ++j) v[j] = bfr(y[static_cast<size_t>(r) * ldy + s.q_n() + kh * D + j]);
            k::qsa_ref_norm_rope(v.data(), rl.kn.data(), D, rr, s.eps, o.data());
            std::vector<float> vv(D);
            for (int j = 0; j < D; ++j) vv[j] = bfr(y[static_cast<size_t>(r) * ldy + s.q_n() + (hkv + kh) * D + j]);
            for (int which = 0; which < 2; ++which) {
              std::vector<float> want(which ? vv : o), dv(D);
              if (v4 && which) {
                std::vector<int8_t> cd(D);
                std::vector<uint16_t> sg(D / 32);
                k::qsa_ref_v4(want.data(), cd.data(), sg.data());
                for (int j = 0; j < D; ++j) want[j] = static_cast<float>(cd[j]) * k::bf16_value(sg[j / 32]);
              } else if (i8) {
                std::vector<int8_t> cd(D);
                float scl;
                k::qsa_ref_i8(want.data(), D, cd.data(), &scl);
                for (int j = 0; j < D; ++j) want[j] = static_cast<float>(cd[j]) * scl;
              }
              for (int j = 0; j < D; ++j) dv[j] = kv_at(s, which ? vp : kp, which ? vsp : ksp, p, kh, j, which != 0);
              ckv.add_rowf(dv.data(), want.data(), D);
            }
          }
          for (int kh = 0; kh < hkv; ++kh) {
            const int nt = p + 1;
            std::vector<float> K(static_cast<size_t>(nt) * D), V(K.size());
            for (int t = 0; t < nt; ++t)
              for (int j = 0; j < D; ++j) {
                K[static_cast<size_t>(t) * D + j] = kv_at(s, kp, ksp, t, kh, j);
                V[static_cast<size_t>(t) * D + j] = kv_at(s, vp, vsp, t, kh, j, true);
              }
            for (int hh = 0; hh < G; ++hh) {
              const int h = kh * G + hh;
              std::vector<double> oo(D);
              k::qsa_ref_attend(&qd[(static_cast<size_t>(r) * hq + h) * D], nt, K.data(), V.data(), oo.data());
              std::vector<float> gate(D), ga(D);
              for (int j = 0; j < D; ++j) gate[j] = bfr(y[static_cast<size_t>(r) * ldy + h * 2 * D + D + j]);
              k::qsa_ref_gate(oo.data(), gate.data(), ga.data());
              ca.add_rowf(&attn[(static_cast<size_t>(r) * hq + h) * D], ga.data(), D);
            }
          }
          for (int n = 0; n < H; ++n) {
            double acc = 0;
            const float *wr = &rl.o[c][static_cast<size_t>(n) * s.o_k()];
            for (int j = 0; j < s.o_k(); ++j) acc += static_cast<double>(attn[static_cast<size_t>(r) * s.o_k() + j]) * wr[j];
            oref[static_cast<size_t>(r) * H + n] += acc;
          }
        }
      }
      for (int r = 0; r < st.rows; ++r) co.add_row(&o0[static_cast<size_t>(r) * H], &oref[static_cast<size_t>(r) * H], H);
    }
    std::printf("\nlayer %d (%zu steps to position %d, rows 1-4, %d checked, N(0, 1) input, %d splits at the end):\n", L, steps.size(),
                steps.back().pos + steps.back().rows, checked, k::gqa_splits(steps.back().pos + steps.back().rows - 1));
    std::snprintf(what, sizeof what, "layer %d qkv (W4) vs the dequantised host slices, step 0: %s", L, cp.text().c_str());
    check(cp.ok(1.0 / 128, 1.0), what);  // fp32 output: the bound decides, the bf16 differ count is moot
    std::snprintf(what, sizeof what, "layer %d q vs norm + rope of the device's projections: %s", L, cq.text().c_str());
    check(cq.ok(1.0 / 128, 0.01), what);
    std::snprintf(what, sizeof what, "layer %d K / V rows as stored vs reference: %s", L, ckv.text().c_str());
    check(ckv.ok(i8 ? 1.0 / 16 : 1.0 / 128, i8 ? 1.0 : 0.01), what);
    std::snprintf(what, sizeof what, "layer %d attn (gated) vs softmax over the whole context as stored: %s", L, ca.text().c_str());
    check(dec_exact ? ca.ok(1.0 / 64, 0.005) : ca.ok(1.0 / 8, 1.0), what);
    std::snprintf(what, sizeof what, "layer %d out vs the sum of the per-card o_proj references: %s; both cards bit-identical", L, co.text().c_str());
    check((dec_exact ? co.ok(1.0 / 128, 0.01) : co.ok(1.0 / 16, 1.0)) && same, what);
    for (int c = 0; c < 2; ++c) free_dev(fab, d[c]);
  }

  // ---- timing: every full-attention layer once per step, chained ----
  const int NL = static_cast<int>(as.layers.size());
  const auto &w0 = as.w[0][as.layers[0]];
  const double wbytes = static_cast<double>(w0.qkv.bytes() + w0.o.bytes());
  const double floor_bw = 604e9;
  std::vector<int64_t> ctxs = context > 0 ? std::vector<int64_t>{context} : std::vector<int64_t>{8192, 32768, 131072, 262144};
  for (const int64_t ctx : ctxs) {
    const int cap = static_cast<int>((ctx + 63) / 64 * 64);
    std::vector<Dev> dv[2];
    Dev scr[2];
    for (int c = 0; c < 2; ++c) {
      scr[c] = make_dev(fab, c, as.shape_of(c), 64, true);
      for (int i = 0; i < NL; ++i) dv[c].push_back(make_dev(fab, c, as.shape_of(c), cap, false));
    }
    {  // random history: K / V codes (finite bf16), unit-ish scales
      const size_t NP = cap / 64;
      std::vector<uint8_t> kvh(NP * k::gqa_page_kv_bytes(s0));
      std::vector<float> scl(NP * k::gqa_page_scale_floats(s0), 1.f / 64);
      std::vector<uint16_t> vsb(NP * k::gqa_page_vscale_floats(s0) * 2, k::bf16_bits(1.f / 8));  // int8-v4: bf16 scales
      uint64_t x = 88172645463325252ull;
      for (size_t i = 0; i < kvh.size(); i += 8) {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        for (int j = 0; j < 8 && i + j < kvh.size(); ++j) {
          const uint8_t b = static_cast<uint8_t>(x >> (8 * j));
          kvh[i + j] = i8 ? b : static_cast<uint8_t>(j & 1 ? (b & 0xBF) | 0x3C : b);  // bf16 high bytes: |v| in [2^-7, 2)
        }
      }
      for (int c = 0; c < 2; ++c)
        for (auto &dd : dv[c]) {
          auto &q = fab.compute(c);
          q.memcpy(dd.c.k, kvh.data(), kvh.size());
          q.memcpy(dd.c.v, kvh.data(), NP * k::gqa_page_v_bytes(s0));  // int8-v4: half of K's bytes, any nibbles
          q.memcpy(dd.c.ks, scl.data(), scl.size() * 4);
          if (v4) q.memcpy(dd.c.vs, vsb.data(), vsb.size() * 2);
          else q.memcpy(dd.c.vs, scl.data(), scl.size() * 4);
          q.wait();
        }
    }
    const double kv_bytes_ctx = static_cast<double>(cap) * s0.hkv() * (256.0 * s0.kv_elem() + (i8 ? 4 : 0) + s0.v_row_bytes() + (i8 ? s0.v_scale_bytes() : 0));
    const double bytes = wbytes + kv_bytes_ctx, floor_us = bytes / floor_bw * 1e6;
    std::printf("\ncontext %d: floor per layer per card %.2f MB (W4 weights %.2f, K / V of the context %.2f) -> %.1f us at %.0f GB/s; x%d layers %.0f us\n",
                cap, bytes / 1e6, wbytes / 1e6, kv_bytes_ctx / 1e6, floor_us, floor_bw / 1e9, NL, floor_us * NL);
    std::printf("rows splits group |    qkv   prep   attn  merge      o reduce (us, card 0 median) | GPU per layer: card 0  card 1 | x floor | attn GB/s\n");
    // : the attention per row count: old = the per-row kernel, RMxGRF = one K / V
    // pass per group of RM rows at that register file, auto = gqa_attn_group's pick
    struct Var {
      int rows, grf;
      const char *name;
      int pf = -1;  // The L2 prefetch (-1 = gqa_attn_group's pick)
    };
    const std::vector<Var> vars = {{-1, 0, "old"}, {1, 128, "1x128"}, {1, 256, "1x256"}, {2, 128, "2x128"}, {2, 256, "2x256"},
                                   {4, 128, "4x128"}, {4, 256, "4x256"}, {8, 256, "8x256"}, {0, 0, "auto"},
                                   {1, 128, "1x128p", 1}, {1, 256, "1x256p", 1}, {2, 256, "2x256n", 0}, {4, 256, "4x256n", 0}, {0, 0, "auto-n", 0}};
    for (const int R : {1, 2, 3, 4, 8})
     for (const Var &var : vars) {
      if (var.rows > 1 && var.rows > 2 * R) continue;  // a group far wider than the step
      if (var.rows == 8 && R < 8) continue;
      if (v4 && var.rows < 0) continue;  // The per-row kernel takes int8 | bf16
      if (!dec_exact && (var.rows < 0 || (var.grf == 128 && var.rows != 1) || var.rows == 8)) continue;  // the numerics' instances
      if (R == 8 && var.rows != 0 && (var.rows < 2 || var.grf == 128)) continue;  // an 8-row verify (DFlash depth 7): 4 x 2, 2 x 4, 8, auto
      const int p0 = cap - R;
      for (int c = 0; c < 2; ++c) fab.compute(c).memcpy(pos[c], &p0, 4).wait();
      std::vector<double> kt[6], gspan[2];
      const int reps = std::max(4, iters / 10);
      auto chain = [&](int xi, std::vector<k::GqaEvents> *ev) {
        for (int i = 0; i < NL; ++i)
          for (int c = 0; c < 2; ++c) {
            k::GqaStep st{pos[c], cap};
            st.attn_rows = var.rows;
            st.attn_grf = var.grf;
            st.attn_pf = var.pf;
            auto e = k::gqa_decode(fab.compute(c), as.shape_of(c), R, xin[c][xi % kX], as.w[c][as.layers[i]], dv[c][i].c, rope[c], st, scr[c].sc,
                                   &wire[c], wire_parity, out[c]);
            if (ev) ev[c].push_back(e);
            if (c == 1) wire_parity ^= 1;
          }
        for (int c = 0; c < 2; ++c) fab.compute(c).wait_and_throw();
      };
      xenithium::probe::warm_for([&] { chain(0, nullptr); });
      for (int rep = 0; rep < reps + 2; ++rep) {
        std::vector<k::GqaEvents> ev[2];
        chain(rep, ev);
        if (rep < 2) continue;
        for (int i = 2; i < NL; ++i) {
          const auto &e = ev[0][i];
          const sycl::event *es[6] = {&e.qkv, &e.prep, &e.attn, &e.merge, &e.o, &e.reduce};
          for (int j = 0; j < 6; ++j) kt[j].push_back(ev_us(*es[j]));
        }
        for (int c = 0; c < 2; ++c) {
          const auto a = ev[c][1].reduce.get_profiling_info<sycl::info::event_profiling::command_end>();
          const auto b = ev[c][NL - 1].reduce.get_profiling_info<sycl::info::event_profiling::command_end>();
          gspan[c].push_back(static_cast<double>(b - a) * 1e-3 / (NL - 2));
        }
      }
      const double g = std::max(median(gspan[0]), median(gspan[1]));
      std::printf("%4d %6d %5s | %6.2f %6.2f %6.2f %6.2f %6.2f %6.2f                    | %20.1f %7.1f | %7.2f | %9.0f\n", R, k::gqa_splits(cap - 1), var.name,
                  median(kt[0]), median(kt[1]), median(kt[2]), median(kt[3]), median(kt[4]), median(kt[5]), median(gspan[0]), median(gspan[1]), g / floor_us,
                  kv_bytes_ctx / (median(kt[2]) * 1e-6) / 1e9);
    }
    for (int c = 0; c < 2; ++c) {
      free_dev(fab, scr[c]);
      for (auto &dd : dv[c]) free_dev(fab, dd);
    }
  }
  uint32_t err[2];
  for (int c = 0; c < 2; ++c) fab.compute(c).memcpy(&err[c], wire[c].error, 4).wait();
  check(!err[0] && !err[1], "gqa35 mailbox: no timeout");
  for (int c = 0; c < 2; ++c)
    for (auto *p : xin[c]) fab.release(p);
  std::printf("gqa35: %s\n", rc ? "FAIL" : "PASS");
  return rc;
}
