// xenithium-probe model35 --dflash-check N: the DFlash 2 drafter on the cards against a host reference
// of the checkpoint's drafter (C++, double sums, the reference implementation's bf16 rounding points), on the features the
// target produced on the cards. Two cycles: the first drafts after a prefill of N prompt tokens (context = the prefill's
// features), the second after one 8-row verify of those drafts (context + the verify's kept rows from the step's features).
// The host mirrors the device's weights (int8 per row per card slice, or bf16), so the gate isolates the kernels:
// every block row's final hidden (cosine, worst relative row error), the selector's hidden projection, the context K / V in
// the ring, and the drafts = the host's walk over the device's candidates.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <functional>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

#include "xenithium/arch/qwen35/dflash.hpp"
#include "xenithium/arch/qwen35/model.hpp"
#include "xenithium/kernels/drafter/dflash.hpp"
#include "xenithium/kernels/pack.hpp"
#include "xenithium/kernels/residual/rmsnorm.hpp"
#include "xenithium/model/checkpoint.hpp"

#include "probe.hpp"

namespace q35 = xenithium::arch::qwen35;
namespace k = xenithium::kernels;

namespace {

void par(int n, const std::function<void(int, int)> &fn) {
  const int T = std::max(1, std::min<int>(static_cast<int>(std::thread::hardware_concurrency()), 32));
  std::vector<std::thread> th;
  for (int t = 0; t < T; ++t) th.emplace_back([&, t] { fn(n * t / T, n * (t + 1) / T); });
  for (auto &x : th) x.join();
}

std::vector<uint16_t> read_tensor(const xenithium::model::Checkpoint &ck, const std::string &name, int64_t row0 = 0, int64_t rows = -1) {
  const auto &ti = ck.at(name);
  const int64_t d0 = ti.shape.empty() ? 1 : ti.shape[0], rb = static_cast<int64_t>(ti.bytes) / d0;
  if (rows < 0) rows = d0 - row0;
  std::vector<uint16_t> v(static_cast<size_t>(rows * rb / 2));
  const int fd = ::open(ck.shards()[ti.shard].path.c_str(), O_RDONLY);
  size_t got = 0;
  const size_t want = v.size() * 2;
  while (fd >= 0 && got < want) {
    const ssize_t r = ::pread(fd, reinterpret_cast<char *>(v.data()) + got, want - got, static_cast<off_t>(ti.offset + row0 * rb + got));
    if (r <= 0) break;
    got += static_cast<size_t>(r);
  }
  if (fd >= 0) ::close(fd);
  if (got != want) throw std::runtime_error("dflash-check: short read of " + name);
  return v;
}

float f(uint16_t b) { return k::bf16_value(b); }
uint16_t b16(double v) { return k::bf16_bits(static_cast<float>(v)); }

// A matrix as the device holds it: n x kk floats (int8 codes x row scale per card slice, or the bf16 values).
struct W {
  std::vector<float> w;
  int n = 0, kk = 0;
};
// split_cols: o / down / fc (each of `cards` cards quantises its 1 / cards of k per row; TP1: the whole row)
W device_w(const std::vector<uint16_t> &raw, int n, int kk, int bytes, bool split_cols, int cards) {
  W m;
  m.n = n;
  m.kk = kk;
  m.w.resize(static_cast<size_t>(n) * kk);
  if (bytes == 2) {
    for (size_t i = 0; i < raw.size(); ++i) m.w[i] = f(raw[i]);
    return m;
  }
  const int parts = split_cols ? cards : 1, kp = kk / parts;
  par(n, [&](int r0, int r1) {
    std::vector<uint16_t> row(static_cast<size_t>(kp));
    std::vector<int8_t> codes(static_cast<size_t>(kp));
    std::vector<uint32_t> words(static_cast<size_t>(kp) / 8 + 1);
    std::vector<uint16_t> gs(static_cast<size_t>(kp) / k::kW4Group + 1);
    for (int r = r0; r < r1; ++r)
      for (int p = 0; p < parts; ++p) {
        std::memcpy(row.data(), &raw[static_cast<size_t>(r) * kk + p * kp], static_cast<size_t>(kp) * 2);
        if (bytes == 4) {  // int4 by RTN per 128-k group, as the load quantises it (DenseSpec:Store:w4r)
          k::quantize_w4_row(row.data(), kp, words.data(), gs.data());
          for (int i = 0; i < kp; ++i) {
            _Float16 h;
            std::memcpy(&h, &gs[static_cast<size_t>(i / k::kW4Group)], 2);
            const int code = static_cast<int>((words[static_cast<size_t>(i / 8)] >> (4 * (i % 8))) & 15u) - 8;
            m.w[static_cast<size_t>(r) * kk + p * kp + i] = static_cast<float>(code) * static_cast<float>(h);
          }
          continue;
        }
        float sc = 1.f;
        k::quantize_i8_rows(row.data(), 1, kp, codes.data(), &sc, 1);
        for (int i = 0; i < kp; ++i) m.w[static_cast<size_t>(r) * kk + p * kp + i] = codes[static_cast<size_t>(i)] * sc;
      }
  });
  return m;
}

// y[m][n] = bf16(x[m]. w[n]) (double sums), x as bf16 bits
std::vector<uint16_t> lin(const std::vector<uint16_t> &x, int m, const W &w) {
  std::vector<uint16_t> y(static_cast<size_t>(m) * w.n);
  std::vector<float> xf(x.size());
  for (size_t i = 0; i < x.size(); ++i) xf[i] = f(x[i]);
  par(w.n, [&](int n0, int n1) {
    for (int j = n0; j < n1; ++j) {
      const float *wr = &w.w[static_cast<size_t>(j) * w.kk];
      for (int r = 0; r < m; ++r) {
        const float *xr = &xf[static_cast<size_t>(r) * w.kk];
        double a = 0;
        for (int i = 0; i < w.kk; ++i) a += static_cast<double>(xr[i]) * wr[i];
        y[static_cast<size_t>(r) * w.n + j] = b16(a);
      }
    }
  });
  return y;
}

struct Rope {
  std::vector<float> inv;
  explicit Rope(double theta) : inv(64) {
    for (int i = 0; i < 64; ++i) inv[static_cast<size_t>(i)] = static_cast<float>(1.0 / std::pow(theta, static_cast<double>(2 * i) / 128));
  }
  // one head: Qwen3RMSNorm (w null: none) then rope at pos (pos < 0: none); in / out bf16 bits
  void head(const uint16_t *x, const uint16_t *w, int pos, uint16_t *out) const {
    float o[128];
    double ss = 0;
    for (int d = 0; d < 128; ++d) ss += static_cast<double>(f(x[d])) * f(x[d]);
    const float rr = 1.f / std::sqrt(static_cast<float>(ss / 128) + 1e-6f);
    for (int d = 0; d < 128; ++d) o[d] = w ? f(b16(static_cast<double>(f(w[d])) * f(b16(static_cast<double>(f(x[d])) * rr)))) : f(x[d]);
    for (int d = 0; d < 128; ++d) {
      if (pos < 0) {
        out[d] = b16(o[d]);
        continue;
      }
      const float a = static_cast<float>(pos) * inv[static_cast<size_t>(d % 64)];
      const float c = f(b16(std::cos(static_cast<double>(a)))), s = f(b16(std::sin(static_cast<double>(a))));
      const float rot = d < 64 ? -o[d + 64] : o[d - 64];
      out[d] = b16(static_cast<double>(f(b16(static_cast<double>(o[d]) * c))) + f(b16(static_cast<double>(rot) * s)));
    }
  }
};

struct HostDrafter {
  xenithium::arch::DFlashConfig cfg;
  int bytes = 1;
  W fc, sel;
  std::vector<uint16_t> hnorm, fnorm;
  struct L {
    W q, k, v, o, ap, mp, g, u, dn;
    std::vector<uint16_t> in_ln, post_ln, qn, kn, ab, mb;
  };
  std::vector<L> layers;
  std::vector<uint16_t> pred, succ;  // codebooks
};

HostDrafter load_host(const std::string &dir, int big, int C) {
  HostDrafter h;
  h.cfg = xenithium::arch::DFlashConfig::load(dir);
  h.bytes = big;
  const int bytes = big == 4 ? 1 : big;  // int4 (DFlashOptions:dense_bytes 4): q, o, gate, up, down; the rest int8 per row
  const auto ck = xenithium::model::Checkpoint::open(dir);
  const int H = h.cfg.hidden, F = static_cast<int>(h.cfg.target_layers.size()) * H, D = 128, I = h.cfg.inter;
  h.fc = device_w(read_tensor(ck, "fc.weight"), H, F, bytes, true, C);
  h.sel = device_w(read_tensor(ck, "candidate_selector.hidden_projection.weight"), h.cfg.sel_rank, H, bytes, false, C);
  h.hnorm = read_tensor(ck, "hidden_norm.weight");
  h.fnorm = read_tensor(ck, "norm.weight");
  for (int l = 0; l < h.cfg.layers; ++l) {
    const std::string P = "layers." + std::to_string(l) + ".";
    HostDrafter::L y;
    y.q = device_w(read_tensor(ck, P + "self_attn.q_proj.weight"), h.cfg.q_heads * D, H, big, false, C);
    y.k = device_w(read_tensor(ck, P + "self_attn.k_proj.weight"), h.cfg.kv_heads * D, H, bytes, false, C);
    y.v = device_w(read_tensor(ck, P + "self_attn.v_proj.weight"), h.cfg.kv_heads * D, H, bytes, false, C);
    y.o = device_w(read_tensor(ck, P + "self_attn.o_proj.weight"), H, h.cfg.q_heads * D, big, true, C);
    y.ap = device_w(read_tensor(ck, P + "attention_conv.kernel_projection.weight"), 4 * H / h.cfg.conv_group, H, bytes, false, C);
    y.mp = device_w(read_tensor(ck, P + "mlp_conv.kernel_projection.weight"), 4 * H / h.cfg.conv_group, H, bytes, false, C);
    y.g = device_w(read_tensor(ck, P + "mlp.gate_proj.weight"), I, H, big, false, C);
    y.u = device_w(read_tensor(ck, P + "mlp.up_proj.weight"), I, H, big, false, C);
    y.dn = device_w(read_tensor(ck, P + "mlp.down_proj.weight"), H, I, big, true, C);
    y.in_ln = read_tensor(ck, P + "input_layernorm.weight");
    y.post_ln = read_tensor(ck, P + "post_attention_layernorm.weight");
    y.qn = read_tensor(ck, P + "self_attn.q_norm.weight");
    y.kn = read_tensor(ck, P + "self_attn.k_norm.weight");
    y.ab = read_tensor(ck, P + "attention_conv.base_kernel");
    y.mb = read_tensor(ck, P + "mlp_conv.base_kernel");
    h.layers.push_back(std::move(y));
  }
  h.pred = read_tensor(ck, "candidate_selector.predecessor_codebook");
  h.succ = read_tensor(ck, "candidate_selector.successor_codebook");
  return h;
}

// The drafter's context K / V of every position from its features (bf16 [n][5 hidden]): [layer][pos][kv heads][128] each.
struct Ctx {
  std::vector<std::vector<uint16_t>> k, v;
};
Ctx context(const HostDrafter &h, const std::vector<uint16_t> &feat, int n, const Rope &rope) {
  const int H = h.cfg.hidden, hkv = h.cfg.kv_heads, D = 128;
  auto t = lin(feat, n, h.fc);
  std::vector<uint16_t> tn(t.size());
  k::rms_norm_plain_ref(n, H, t.data(), nullptr, h.hnorm.data(), static_cast<float>(h.cfg.eps), tn.data());
  Ctx c;
  for (const auto &y : h.layers) {
    const auto kk = lin(tn, n, y.k), vv = lin(tn, n, y.v);
    std::vector<uint16_t> ko(kk.size());
    for (int p = 0; p < n; ++p)
      for (int hh = 0; hh < hkv; ++hh) rope.head(&kk[(static_cast<size_t>(p) * hkv + hh) * D], y.kn.data(), p, &ko[(static_cast<size_t>(p) * hkv + hh) * D]);
    c.k.push_back(ko);
    c.v.push_back(vv);
  }
  return c;
}

// One block of B rows at s (ids), context positions [0, s): the final hidden [B][hidden] and the hidden projection.
std::vector<uint16_t> block(const HostDrafter &h, const Ctx &c, const std::vector<uint16_t> &x0, int B, int s, const Rope &rope, std::vector<uint16_t> &hp) {
  const int H = h.cfg.hidden, hq = h.cfg.q_heads, hkv = h.cfg.kv_heads, D = 128, grp = hq / hkv, G = H / h.cfg.conv_group, win = h.cfg.window;
  const float eps = static_cast<float>(h.cfg.eps);
  std::vector<uint16_t> res = x0, xn(res.size()), add;
  auto conv = [&](const std::vector<uint16_t> &x, const std::vector<uint16_t> &dyn, int which, const std::vector<uint16_t> &base) {
    std::vector<float> df(dyn.size());
    for (size_t i = 0; i < dyn.size(); ++i) df[i] = f(dyn[i]);
    std::vector<uint16_t> o(x.size());
    k::df_conv_ref(B, B, H, h.cfg.conv_group, x.data(), df.data(), 4 * G, which, base.data() + static_cast<size_t>(which) * 2 * H, o.data());
    return o;
  };
  for (size_t l = 0; l < h.layers.size(); ++l) {
    const auto &y = h.layers[l];
    k::rms_norm_plain_ref(B, H, res.data(), l > 0 ? add.data() : nullptr, y.in_ln.data(), eps, xn.data());
    const auto dyn = lin(xn, B, y.ap);
    const auto xc = conv(xn, dyn, 0, y.ab);
    const auto q = lin(xc, B, y.q), kk = lin(xc, B, y.k), vv = lin(xc, B, y.v);
    std::vector<uint16_t> qo(q.size()), ko(kk.size());
    for (int j = 0; j < B; ++j) {
      for (int hh = 0; hh < hq; ++hh) rope.head(&q[(static_cast<size_t>(j) * hq + hh) * D], y.qn.data(), s + j, &qo[(static_cast<size_t>(j) * hq + hh) * D]);
      for (int hh = 0; hh < hkv; ++hh) rope.head(&kk[(static_cast<size_t>(j) * hkv + hh) * D], y.kn.data(), s + j, &ko[(static_cast<size_t>(j) * hkv + hh) * D]);
    }
    std::vector<uint16_t> at(static_cast<size_t>(B) * hq * D);
    par(B * hq, [&](int a0, int a1) {
      for (int a = a0; a < a1; ++a) {
        const int j = a / hq, hh = a % hq, kvh = hh / grp;
        const uint16_t *qr = &qo[(static_cast<size_t>(j) * hq + hh) * D];
        std::vector<const uint16_t *> ks, vs;
        for (int p = std::max(0, s + j - win + 1); p < s; ++p) {
          ks.push_back(&c.k[l][(static_cast<size_t>(p) * hkv + kvh) * D]);
          vs.push_back(&c.v[l][(static_cast<size_t>(p) * hkv + kvh) * D]);
        }
        for (int b = 0; b < B; ++b) {
          ks.push_back(&ko[(static_cast<size_t>(b) * hkv + kvh) * D]);
          vs.push_back(&vv[(static_cast<size_t>(b) * hkv + kvh) * D]);
        }
        std::vector<double> sc(ks.size());
        double mx = -1e300;
        for (size_t i = 0; i < ks.size(); ++i) {
          double d = 0;
          for (int e = 0; e < D; ++e) d += static_cast<double>(f(qr[e])) * f(ks[i][e]);
          sc[i] = d / std::sqrt(128.0);
          mx = std::max(mx, sc[i]);
        }
        double l2 = 0;
        for (auto &v : sc) l2 += (v = std::exp(v - mx));
        for (int e = 0; e < D; ++e) {
          double o = 0;
          for (size_t i = 0; i < ks.size(); ++i) o += sc[i] * f(vs[i][e]);
          at[(static_cast<size_t>(j) * hq + hh) * D + e] = b16(o / l2);
        }
      }
    });
    const auto o = lin(at, B, y.o);
    const auto oc = conv(o, dyn, 1, y.ab);
    k::rms_norm_plain_ref(B, H, res.data(), oc.data(), y.post_ln.data(), eps, xn.data());
    const auto dyn2 = lin(xn, B, y.mp);
    const auto xc2 = conv(xn, dyn2, 0, y.mb);
    const auto g = lin(xc2, B, y.g), u = lin(xc2, B, y.u);
    std::vector<uint16_t> hh(g.size());
    for (size_t i = 0; i < g.size(); ++i) {
      const double gv = f(g[i]);
      hh[i] = b16(static_cast<double>(f(b16(gv / (1.0 + std::exp(-gv))))) * f(u[i]));
    }
    const auto m = lin(hh, B, y.dn);
    add = conv(m, dyn2, 1, y.mb);
  }
  std::vector<uint16_t> out(res.size());
  k::rms_norm_plain_ref(B, H, res.data(), add.data(), h.fnorm.data(), eps, out.data());
  hp = lin(out, B, h.sel);
  return out;
}

struct RowCmp {
  double cos_min = 1, rel_max = 0;
};
RowCmp cmp_rows(const std::vector<uint16_t> &a, const std::vector<uint16_t> &b, int rows, int n, int r0 = 0) {
  RowCmp c;
  for (int r = r0; r < rows; ++r) {
    double ab = 0, aa = 0, bb = 0, dd = 0;
    for (int i = 0; i < n; ++i) {
      const double x = f(a[static_cast<size_t>(r) * n + i]), y = f(b[static_cast<size_t>(r) * n + i]);
      ab += x * y, aa += x * x, bb += y * y, dd += (x - y) * (x - y);
    }
    c.cos_min = std::min(c.cos_min, ab / std::sqrt(std::max(aa * bb, 1e-300)));
    c.rel_max = std::max(c.rel_max, std::sqrt(dd / std::max(bb, 1e-300)));
  }
  return c;
}

}  // namespace

// The gate (see the header). prompt: the first N tokens of the first token file; check: the probe's PASS / FAIL printer.
// The gate runs on any target through DfCheckTarget (probe.hpp); qwen35's adapter below, qwen38's in dflash38.cpp.
void dflash_check(DfCheckTarget &m, const std::string &target_dir, const std::vector<int32_t> &prompt_in, int n, int dense_bytes,
                  const std::function<void(bool, const std::string &)> &check) {
  xenithium::arch::DFlash *df = m.dflash();
  if (!df) throw std::runtime_error("dflash-check: no DFlash drafter loaded (--dflash DIR)");
  n = std::min<int>(n, static_cast<int>(prompt_in.size()));
  n = std::min(n, m.prefill_chunk());
  std::vector<int32_t> prompt(prompt_in.begin(), prompt_in.begin() + n);
  const auto &cfg = df->config();
  const int C = m.cards();  // TP1 / TP2: the host mirrors each card's slices
  const int H = cfg.hidden, B = cfg.block, F = df->feat_cols(), hkvc = cfg.kv_heads / C;
  std::printf("\ndflash-check: host reference of %s (%s weights mirrored), %d prompt tokens, block %d\n", df->dir().c_str(), dense_bytes == 2 ? "bf16" : dense_bytes == 4 ? "int4 (q, o, gate, up, down) / int8" : "int8",
              n, B);
  const auto t0 = std::chrono::steady_clock::now();
  const HostDrafter h = load_host(df->dir(), dense_bytes, C);
  const Rope rope(cfg.theta);
  std::printf("dflash-check: host weights in %.1f s\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
  // the target's embedding rows of the tokens the blocks use
  const auto tck = xenithium::model::Checkpoint::open(target_dir);
  auto embed = [&](int32_t id) { return read_tensor(tck, m.embed_name(), id, 1); };
  const auto mask = embed(static_cast<int32_t>(cfg.mask_id));
  auto x0_of = [&](int32_t anchor) {
    std::vector<uint16_t> x = embed(anchor);
    for (int r = 1; r < B; ++r) x.insert(x.end(), mask.begin(), mask.end());
    return x;
  };
  auto feats = [&](bool pf, int rows) {  // every card's columns -> [rows][5 hidden]
    std::vector<uint16_t> out(static_cast<size_t>(rows) * C * F);
    for (int c = 0; c < C; ++c) {
      std::vector<uint16_t> part(static_cast<size_t>(rows) * F);
      const auto *src = pf ? df->prefill_features(c) : df->step_features(c);
      // the probe's Fabric queues: read through the model's card queue
      m.read_device(c, src, part.data(), part.size() * 2);
      for (int r = 0; r < rows; ++r) std::memcpy(&out[(static_cast<size_t>(r) * C + c) * F], &part[static_cast<size_t>(r) * F], static_cast<size_t>(F) * 2);
    }
    return out;
  };
  const double cos_gate = dense_bytes == 2 ? 0.9999 : 0.9995;
  const std::vector<int64_t> eos = m.eos();  // the specials' 64-aligned floor (the draft vocabulary keeps them)
  const int eos_floor = static_cast<int>(eos.empty() ? 0 : *std::min_element(eos.begin(), eos.end()) / 64 * 64);
  // ---- cycle 1: prefill, the first drafts ----
  m.reset();
  m.prefill(prompt.data(), n);
  const int lr = (n - 1) % 4;
  const int32_t first = m.picks(lr + 1)[static_cast<size_t>(lr)];
  auto pf_feat = feats(true, n);
  const auto drafts1 = m.draft_after_prefill(first, B - 1);
  const auto dev_h1 = df->read_hidden(B);
  const auto dev_k1 = df->read_keys(B);
  const auto dev_hp1 = df->read_hp(B);
  Ctx ctx = context(h, pf_feat, n, rope);
  std::vector<uint16_t> hp1;
  const auto ref_h1 = block(h, ctx, x0_of(first), B, n, rope, hp1);
  const RowCmp c1 = cmp_rows(dev_h1, ref_h1, B, H, 1);
  char line[400];
  std::snprintf(line, sizeof line, "dflash-check cycle 1 (after a prefill of %d): block rows 1-%d final hidden vs host: min cosine %.6f, worst row rel err %.2e", n, B - 1,
                c1.cos_min, c1.rel_max);
  check(c1.cos_min >= cos_gate, line);
  {
    std::vector<uint16_t> hpd(dev_hp1.size());
    for (size_t i = 0; i < hpd.size(); ++i) hpd[i] = k::bf16_bits(dev_hp1[i]);
    // the projection on its own: the host's of the device's final hidden (the end-to-end one carries the hidden's error)
    const auto hp_own = lin(dev_h1, B, h.sel);
    const RowCmp ch = cmp_rows(hpd, hp_own, B, cfg.sel_rank, 1), ce = cmp_rows(hpd, hp1, B, cfg.sel_rank, 1);
    std::snprintf(line, sizeof line, "dflash-check cycle 1: selector hidden projection vs host of the device's hidden: min cosine %.6f (end to end %.6f)", ch.cos_min,
                  ce.cos_min);
    check(ch.cos_min >= 0.99999, line);
    std::vector<int32_t> walk(static_cast<size_t>(B - 1));
    k::df_select_ref(B, dev_k1.data(), dev_hp1.data(), cfg.sel_rank, h.pred.data(), h.succ.data(), first, df->draft_vocab() ? m.mtp_vocab() : 0,
                     eos_floor, walk.data());
    // the confidence cut (DFlashOptions:conf_cut, one sequence) keeps a prefix of the walk
    const bool prefix1 = !drafts1.empty() && drafts1.size() <= walk.size() && std::equal(drafts1.begin(), drafts1.end(), walk.begin());
    check(prefix1, "dflash-check cycle 1: the drafts == the host walk over the device's candidates and projection (" + std::to_string(drafts1.size()) + " of " +
                       std::to_string(walk.size()) + " kept by the confidence cut)");
    if (!prefix1) {  // the two lists, for the diagnosis
      std::printf("  device:");
      for (const int32_t t : drafts1) std::printf(" %d", t);
      std::printf("\n  host:  ");
      for (const int32_t t : walk) std::printf(" %d", t);
      std::printf("\n");
    }
  }
  // ring K / V of the prompt's last positions (layer 0 and the last layer, card 0's heads)
  {
    double kc = 1, vc = 1;
    for (int l : {0, cfg.layers - 1}) {
      const auto rk = df->read_ring(0, l, m.slot(), false), rv = df->read_ring(0, l, m.slot(), true);
      std::vector<uint16_t> a, b, av, bv;
      for (int p = std::max(0, n - 64); p < n; ++p)
        for (int hh = 0; hh < hkvc; ++hh) {
          const size_t ro = (static_cast<size_t>(p % df->ring_tokens()) * hkvc + hh) * 128, co = (static_cast<size_t>(p) * cfg.kv_heads + hh) * 128;
          a.insert(a.end(), rk.begin() + static_cast<std::ptrdiff_t>(ro), rk.begin() + static_cast<std::ptrdiff_t>(ro + 128));
          b.insert(b.end(), ctx.k[static_cast<size_t>(l)].begin() + static_cast<std::ptrdiff_t>(co), ctx.k[static_cast<size_t>(l)].begin() + static_cast<std::ptrdiff_t>(co + 128));
          av.insert(av.end(), rv.begin() + static_cast<std::ptrdiff_t>(ro), rv.begin() + static_cast<std::ptrdiff_t>(ro + 128));
          bv.insert(bv.end(), ctx.v[static_cast<size_t>(l)].begin() + static_cast<std::ptrdiff_t>(co), ctx.v[static_cast<size_t>(l)].begin() + static_cast<std::ptrdiff_t>(co + 128));
        }
      kc = std::min(kc, cmp_rows(a, b, static_cast<int>(a.size() / 128), 128).cos_min);
      vc = std::min(vc, cmp_rows(av, bv, static_cast<int>(av.size() / 128), 128).cos_min);
    }
    std::snprintf(line, sizeof line, "dflash-check cycle 1: context K / V in the ring (prefill path, layers 0 and %d) vs host: min cosine %.6f / %.6f", cfg.layers - 1, kc, vc);
    check(kc >= cos_gate && vc >= cos_gate, line);
  }
  // ---- cycle 2: verify the drafts (8 rows), accept the greedy prefix, draft again ----
  std::vector<int32_t> rows_t{first};
  rows_t.insert(rows_t.end(), drafts1.begin(), drafts1.end());
  const int R = static_cast<int>(rows_t.size());
  m.step(rows_t.data(), R);
  const auto pk = m.picks(R);
  int keep = 1;
  while (keep < R && rows_t[static_cast<size_t>(keep)] == pk[static_cast<size_t>(keep - 1)]) ++keep;
  m.accept(keep);
  const auto st_feat = feats(false, R);
  const auto drafts2 = m.draft(pk.data(), R, keep, B - 1);
  const auto dev_h2 = df->read_hidden(B);
  const auto dev_k2 = df->read_keys(B);
  const auto dev_hp2 = df->read_hp(B);
  // host context: the prompt + the verify's kept rows (positions n.. n + keep - 1)
  {
    std::vector<uint16_t> kept(st_feat.begin(), st_feat.begin() + static_cast<std::ptrdiff_t>(keep) * C * F);
    pf_feat.insert(pf_feat.end(), kept.begin(), kept.end());
  }
  ctx = context(h, pf_feat, n + keep, rope);
  std::vector<uint16_t> hp2;
  const int32_t anchor2 = pk[static_cast<size_t>(keep - 1)];
  const auto ref_h2 = block(h, ctx, x0_of(anchor2), B, n + keep, rope, hp2);
  const RowCmp c2 = cmp_rows(dev_h2, ref_h2, B, H, 1);
  std::snprintf(line, sizeof line, "dflash-check cycle 2 (an 8-row verify kept %d rows): block rows final hidden vs host: min cosine %.6f, worst row rel err %.2e", keep,
                c2.cos_min, c2.rel_max);
  check(c2.cos_min >= cos_gate, line);
  {
    std::vector<int32_t> walk(static_cast<size_t>(B - 1));
    k::df_select_ref(B, dev_k2.data(), dev_hp2.data(), cfg.sel_rank, h.pred.data(), h.succ.data(), anchor2, df->draft_vocab() ? m.mtp_vocab() : 0,
                     eos_floor, walk.data());
    // the confidence cut (DFlashOptions:conf_cut, one sequence) keeps a prefix of the walk
    const bool prefix2 = !drafts2.empty() && drafts2.size() <= walk.size() && std::equal(drafts2.begin(), drafts2.end(), walk.begin());
    check(prefix2, "dflash-check cycle 2: the drafts == the host walk over the device's candidates and projection (" + std::to_string(drafts2.size()) + " of " +
                       std::to_string(walk.size()) + " kept by the confidence cut)");
    if (!prefix2) {  // the two lists, for the diagnosis
      std::printf("  device:");
      for (const int32_t t : drafts2) std::printf(" %d", t);
      std::printf("\n  host:  ");
      for (const int32_t t : walk) std::printf(" %d", t);
      std::printf("\n");
    }
    // the verify rows' context K in the ring (decode path: the step's features)
    const auto rk = df->read_ring(0, 0, m.slot(), false);
    std::vector<uint16_t> a, b;
    for (int p = n; p < n + keep; ++p)
      for (int hh = 0; hh < hkvc; ++hh) {
        const size_t ro = (static_cast<size_t>(p % df->ring_tokens()) * hkvc + hh) * 128, co = (static_cast<size_t>(p) * cfg.kv_heads + hh) * 128;
        a.insert(a.end(), rk.begin() + static_cast<std::ptrdiff_t>(ro), rk.begin() + static_cast<std::ptrdiff_t>(ro + 128));
        b.insert(b.end(), ctx.k[0].begin() + static_cast<std::ptrdiff_t>(co), ctx.k[0].begin() + static_cast<std::ptrdiff_t>(co + 128));
      }
    const double kc = cmp_rows(a, b, static_cast<int>(a.size() / 128), 128).cos_min;
    std::snprintf(line, sizeof line, "dflash-check cycle 2: context K of the verify's kept rows (decode features, layer 0) vs host: min cosine %.6f", kc);
    check(kc >= cos_gate, line);
  }
  std::printf("dflash-check: drafts after the prefill %d accepted of %d; host reference %.1f s in all\n", keep - 1, B - 1,
              std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
}

// qwen35's side of the gate (the 27B; the call of onwards).
namespace {
struct Q35Target final : DfCheckTarget {
  q35::Model &m;
  explicit Q35Target(q35::Model &mm) : m(mm) {}
  xenithium::arch::DFlash *dflash() override { return m.dflash(); }
  int cards() override { return m.cards(); }
  int prefill_chunk() override { return m.prefill_chunk(); }
  int slot() override { return m.slot(); }
  int mtp_vocab() override { return m.options().mtp_vocab; }
  std::vector<int64_t> eos() override { return m.config().eos; }
  std::string embed_name() override { return std::string(q35::kPrefix) + "embed_tokens.weight"; }
  void reset() override { m.reset(); }
  void prefill(const int32_t *ids, int n) override { m.prefill(ids, n); }
  std::vector<int32_t> picks(int rows) override { return m.picks(0, rows); }
  std::vector<int32_t> draft_after_prefill(int32_t first, int depth) override { return m.draft_after_prefill(first, depth); }
  void step(const int32_t *rows, int n) override { m.step(rows, n); }
  void accept(int keep) override { m.accept(keep); }
  std::vector<int32_t> draft(const int32_t *picks, int rows, int keep, int depth) override { return m.draft(picks, rows, keep, depth); }
  void read_device(int c, const void *src, void *dst, size_t bytes) override { m.read_device(c, src, dst, bytes); }
};
}  // namespace

void dflash_check(q35::Model &m, const std::string &target_dir, const std::vector<int32_t> &prompt, int n, int dense_bytes,
                  const std::function<void(bool, const std::string &)> &check) {
  Q35Target t(m);
  dflash_check(t, target_dir, prompt, n, dense_bytes, check);
}
