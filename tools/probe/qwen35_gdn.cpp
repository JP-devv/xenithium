// xenithium-probe gdn35: the qwen35 Gated DeltaNet decode block on real weights, both cards.
// xenithium-probe gdn35 --model DIR [--layers 0,1,2,62] [--iters I]
// Loads the requested linear-attention layers (dense plan, heads split: in_proj_qkv | z as one W4 matrix, out_proj's k
// half as W4, a / b / conv / A_log / dt_bias / norm raw bf16). Inputs are ~N(0, 1) bf16 rows (what a normed residual
// looks like), the same on both cards. Per layer, a plan of steps with 1-4 rows and accept counts 0-4:
// y (core) vs the reference teacher-forced at the device's projections, the committed state, out vs the sum of the
// per-card references from the device's y, both cards bit-identical; the reference weights are the host slices
// dequantised ((q - 8) x scale), so the projection GEMMs are gated by w4load and here only the core + out + reduce.
// Then the timing: one call per loaded layer chained like a step, plain decode (1 row, accept 1), 4 / 4 and 4 / 1,
// against the read floor.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "qwen35_host.hpp"
#include "xenithium/arch/qwen35/arch.hpp"
#include "xenithium/arch/qwen35/gdn.hpp"
#include "xenithium/kernels/linear_attention/gdn.hpp"
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
  double scaled = 0, w_dev = 0, w_ref = 0, w_rms = 0;
  size_t off = 0, n = 0;
  void add_row(const float *dev, const double *ref, int len) {
    double ss = 0;
    for (int i = 0; i < len; ++i) ss += ref[i] * ref[i];
    const double rms = std::sqrt(ss / len);
    for (int i = 0; i < len; ++i) {
      ++n;
      off += dev[i] != bfr(static_cast<float>(ref[i]));
      const double sc = std::fabs(dev[i] - ref[i]) / std::max({std::fabs(ref[i]), rms, 1e-30});
      if (sc > scaled) {
        scaled = sc;
        w_dev = dev[i];
        w_ref = ref[i];
        w_rms = rms;
      }
    }
  }
  std::string text() const {
    char b[192];
    std::snprintf(b, sizeof b, "worst %.2e of max(|ref|, rms) (dev %.6g ref %.6g rms %.3g), %zu / %zu differ (%.3f %%)", scaled, w_dev, w_ref, w_rms,
                  off, n, 100.0 * off / std::max<size_t>(1, n));
    return b;
  }
};

// Card c's reference weights for layer L: the W4 host slices dequantised, the bf16 vectors as values.
k::GdnRefWeights ref_weights(const m::Checkpoint &ck, const q35::Config &cfg, const std::vector<m::DenseSpec> &plan, int L, int c, int cards,
                             const k::GdnShape &s) {
  const std::string P = std::string(q35::kPrefix) + "layers." + std::to_string(L) + ".linear_attn.";
  const m::DenseSpec *sq = nullptr, *so = nullptr;
  for (const auto &sp : plan) {
    if (sp.name == cfg.packed_name(P + "in_proj_qkv")) sq = &sp;
    if (sp.name == cfg.packed_name(P + "out_proj")) so = &sp;
  }
  if (!sq || !so) throw std::runtime_error("gdn35: layer " + std::to_string(L) + " not in the plan");
  const auto srcs_q = w4_sources(ck, *sq), srcs_o = w4_sources(ck, *so);
  const auto buf_q = m::bulk_read(ck, srcs_q), buf_o = m::bulk_read(ck, srcs_o);
  const HostW4 hq = host_w4_slice(ck, *sq, buf_q, srcs_q, c, cards, s.qkvz_n(), s.hidden);
  const HostW4 ho = host_w4_slice(ck, *so, buf_o, srcs_o, c, cards, s.hidden, s.val_w());
  if (hq.rows != s.qkvz_n() || ho.rows != s.hidden) throw std::runtime_error("gdn35: host slice rows");
  k::GdnRefWeights rw;
  rw.qkvz = deq_w4(hq, s.qkvz_n(), s.hidden);
  rw.out = deq_w4(ho, s.hidden, s.val_w());
  std::vector<const m::TensorInfo *> ts;
  for (const char *t : {"in_proj_a.weight", "in_proj_b.weight", "conv1d.weight", "A_log", "dt_bias", "norm.weight"}) ts.push_back(&ck.at(P + t));
  const auto buf = m::bulk_read(ck, ts);
  auto u16 = [&](const char *t) { return buf.as<uint16_t>(ck.at(P + t)); };
  const int H = s.hidden, D = s.k_dim, KH = s.kh(), VH = s.vh(), KWa = s.k_heads * D, kh0 = c * KH, vh0 = c * VH;
  auto take = [&](const uint16_t *t, int r0, int n, int cols, std::vector<uint16_t> &dst) {
    dst.insert(dst.end(), t + static_cast<size_t>(r0) * cols, t + static_cast<size_t>(r0 + n) * cols);
  };
  auto vals = [](const std::vector<uint16_t> &w) {
    std::vector<float> v(w.size());
    for (size_t i = 0; i < w.size(); ++i) v[i] = k::bf16_value(w[i]);
    return v;
  };
  std::vector<uint16_t> ab, conv;
  take(u16("in_proj_a.weight"), vh0, VH, H, ab);
  take(u16("in_proj_b.weight"), vh0, VH, H, ab);
  rw.ab = vals(ab);
  take(u16("conv1d.weight"), kh0 * D, KH * D, 4, conv);
  take(u16("conv1d.weight"), KWa + kh0 * D, KH * D, 4, conv);
  take(u16("conv1d.weight"), 2 * KWa + vh0 * D, VH * D, 4, conv);
  rw.conv = vals(conv);
  std::vector<uint16_t> al(u16("A_log") + vh0, u16("A_log") + vh0 + VH), dt(u16("dt_bias") + vh0, u16("dt_bias") + vh0 + VH),
      nw(u16("norm.weight"), u16("norm.weight") + D);
  rw.a_log = vals(al);
  rw.dt_bias = vals(dt);
  rw.norm = vals(nw);
  return rw;
}

struct Dev {  // one card's state for one layer
  k::GdnState st;
  int32_t *accept = nullptr;
};

}  // namespace

int cmd_gdn35(const std::string &model, const std::string &layers_arg, int iters, int threads) {
  const auto mc = m::ModelConfig::load(model);
  const auto arch = m::open_arch(mc);
  if (arch->id() != "qwen35") {
    std::fprintf(stderr, "gdn35: %s is arch %s, not qwen35\n", model.c_str(), std::string(arch->id()).c_str());
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
    if (L < 0 || L >= cfg.layers || cfg.full_attention[L]) {
      std::fprintf(stderr, "gdn35: layer %d is not a linear-attention layer\n", L);
      return 2;
    }
  std::vector<m::DenseSpec> plan;
  for (const auto &sp : arch->dense_plan(ck, m::DeployParams{}))
    if (sp.comp == m::Component::linear_attention && std::find(layers.begin(), layers.end(), sp.layer) != layers.end()) plan.push_back(sp);
  const auto loader = xenithium::l0::Loader::open();
  xenithium::Fabric fab(loader, {0, 1});
  xenithium::load::DenseStore dense(fab);
  dense.load(ck, plan, threads);
  const auto &cb = dense.stats().card_bytes[0];
  std::printf("gdn35: %zu dense tensors of %zu layer(s) loaded in %.2f s (%.3f GiB per card, heads split; output gate %s)\n", plan.size(), layers.size(),
              dense.stats().total_s, (cb.count(m::Component::linear_attention) ? cb.at(m::Component::linear_attention) : 0) / 1073741824.0,
              cfg.output_gate.c_str());
  const q35::GdnSet gs = q35::GdnSet::from(dense, cfg, 2, layers);
  const int rows4 = k::kGdnRows, H = cfg.hidden;

  struct Step {
    int rows, accept;
  };
  const std::vector<Step> steps = {{1, 1}, {4, 2}, {4, 4}, {1, 1}, {3, 3}, {2, 2}, {1, 1}, {4, 1}, {1, 0}, {2, 2}};
  const int S = static_cast<int>(steps.size());
  // Inputs per step: 4 rows x H of ~N(0, 1) bf16, the same on both cards.
  std::vector<bf16 *> xin[2];
  {
    std::mt19937 rng(21);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (int s = 0; s < S; ++s) {
      std::vector<bf16> xb(static_cast<size_t>(rows4) * H);
      for (auto &v : xb) v = bf16(nd(rng));
      for (int c = 0; c < 2; ++c) {
        auto *d = static_cast<bf16 *>(fab.alloc(c, xb.size() * 2));
        fab.compute(c).memcpy(d, xb.data(), xb.size() * 2).wait();
        xin[c].push_back(d);
      }
    }
  }
  k::GdnScratch scr[2];
  k::PairWire wire[2];
  bf16 *out[2];
  for (int c = 0; c < 2; ++c) {
    const auto s = gs.shape_of(c);
    scr[c].core = static_cast<bf16 *>(fab.alloc(c, 8ull * s.val_w() * 2));
    scr[c].partial = static_cast<float *>(fab.alloc(c, 8ull * H * 4));
    wire[c].inbox = static_cast<uint32_t *>(fab.alloc_mailbox(c, k::pair_wire_words(H) * 4));
    wire[c].error = static_cast<uint32_t *>(fab.alloc(c, 4096));
    out[c] = static_cast<bf16 *>(fab.alloc(c, 8ull * H * 2));
    fab.compute(c).memset(scr[c].core, 0, 8ull * s.val_w() * 2).wait();
  }
  for (int c = 0; c < 2; ++c) {
    wire[c].peer_inbox = wire[1 - c].inbox;
    k::pair_wire_init(fab.compute(c), H, wire[c].inbox).wait();
    fab.compute(c).memset(wire[c].error, 0, 4).wait();
  }
  auto make_dev = [&](int c, const k::GdnWeights &w) {
    const auto s = gs.shape_of(c);
    Dev d;
    d.st.S = static_cast<float *>(fab.alloc(c, k::gdn_S_elems(s) * 4));
    for (int p = 0; p < 2; ++p) {
      d.st.win[p] = static_cast<bf16 *>(fab.alloc(c, k::gdn_win_elems(s) * 2));
      d.st.y[p] = static_cast<float *>(fab.alloc(c, 4ull * w.qkvz.n * 4));
      d.st.ab[p] = static_cast<float *>(fab.alloc(c, 4ull * s.ab_n() * 4));
    }
    d.accept = static_cast<int32_t *>(fab.alloc(c, 64));
    d.st.accept = d.accept;
    k::gdn_state_reset(fab.compute(c), s, d.st).wait();
    fab.compute(c).memset(d.accept, 0, 4).wait();
    return d;
  };
  auto free_dev = [&](Dev &d) {
    for (void *p : {(void *)d.st.S, (void *)d.st.win[0], (void *)d.st.win[1], (void *)d.st.y[0], (void *)d.st.y[1], (void *)d.st.ab[0],
                    (void *)d.st.ab[1], (void *)d.accept})
      fab.release(p);
  };
  int rc = 0;
  auto check = [&](bool ok, const std::string &what) {
    std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
    if (!ok) rc = 1;
  };
  int wire_parity = 0;
  char what[512];

  for (size_t li = 0; li < layers.size(); ++li) {
    const int L = layers[li];
    Dev d[2] = {make_dev(0, gs.w[0][L]), make_dev(1, gs.w[1][L])};
    k::GdnRefWeights rw[2] = {ref_weights(ck, cfg, plan, L, 0, 2, gs.shape_of(0)), ref_weights(ck, cfg, plan, L, 1, 2, gs.shape_of(1))};
    k::GdnRefState rs[2];
    for (int c = 0; c < 2; ++c) rs[c].reset(gs.shape_of(c));
    BoundCmp yc, oc, pc;
    bool same = true;
    double s_err = 0, s_max = 0;
    struct SStep {
      double err, mx;
      size_t at;
    };
    std::vector<SStep> s_steps[2];  // per card, per step: max |dS| (before the step's commit), max |S|, where
    for (int s = 0; s < S; ++s) {
      const int R = steps[s].rows, prev = s ? steps[s - 1].accept : 0;
      for (int c = 0; c < 2; ++c) {
        fab.compute(c).memcpy(d[c].accept, &prev, 4).wait();
        k::gdn_decode(fab.compute(c), gs.shape_of(c), R, xin[c][s], gs.w[c][L], d[c].st, s, scr[c], &wire[c], wire_parity, out[c]);
      }
      wire_parity ^= 1;
      for (int c = 0; c < 2; ++c) fab.compute(c).wait_and_throw();
      std::vector<uint16_t> ob[2];
      for (int c = 0; c < 2; ++c) {
        ob[c].resize(static_cast<size_t>(R) * H);
        fab.compute(c).memcpy(ob[c].data(), out[c], ob[c].size() * 2).wait();
      }
      same &= ob[0] == ob[1];
      std::vector<double> ref(static_cast<size_t>(R) * H, 0.0), part(H);
      std::vector<k::GdnRefState> after[2];
      std::vector<uint16_t> xb(static_cast<size_t>(rows4) * H);
      fab.compute(0).memcpy(xb.data(), xin[0][s], xb.size() * 2).wait();
      for (int c = 0; c < 2; ++c) {
        const auto sc = gs.shape_of(c);
        const auto &w = gs.w[c][L];
        const int VW = sc.val_w();
        std::vector<float> proj(4ull * w.qkvz.n), ab(4ull * sc.ab_n());
        std::vector<uint16_t> yb(static_cast<size_t>(R) * VW);
        fab.compute(c).memcpy(proj.data(), d[c].st.y[s & 1], proj.size() * 4);
        fab.compute(c).memcpy(ab.data(), d[c].st.ab[s & 1], ab.size() * 4);
        fab.compute(c).memcpy(yb.data(), scr[c].core, yb.size() * 2).wait();
        {  // committed S now = reference before this step's commit (replayed rows of the previous step)
          std::vector<float> Sd(k::gdn_S_elems(sc));
          fab.compute(c).memcpy(Sd.data(), d[c].st.S, Sd.size() * 4).wait();
          double e = 0, mx = 0;
          size_t at = 0;
          for (size_t j = 0; j < Sd.size(); ++j) {
            const double dd = std::fabs(Sd[j] - rs[c].S[j]);
            if (dd > e) e = dd, at = j;
            mx = std::max(mx, std::fabs(rs[c].S[j]));
          }
          s_steps[c].push_back({e, mx, at});
          if (s == S - 1) {
            s_err = std::max(s_err, e);
            s_max = std::max(s_max, mx);
          }
        }
        k::GdnRefState run = rs[c];
        for (int r = 0; r < R; ++r) {
          std::vector<float> pq(sc.qkvz_n()), pa(sc.ab_n()), o(VW), y(VW), yd(VW);
          for (int n = 0; n < sc.qkvz_n(); ++n) pq[n] = bfr(proj[static_cast<size_t>(r) * w.qkvz.n + n]);
          for (int n = 0; n < sc.ab_n(); ++n) pa[n] = bfr(ab[static_cast<size_t>(r) * sc.ab_n() + n]);
          if (s == 0 && c == 0) {  // the W4 projections (and the a / b dots) vs the reference from x: once per layer
            std::vector<float> xr(H), rq(sc.qkvz_n()), ra(sc.ab_n());
            for (int n = 0; n < H; ++n) xr[n] = k::bf16_value(xb[static_cast<size_t>(r) * H + n]);
            k::gdn_ref_proj(sc, rw[c], xr.data(), rq.data(), ra.data());
            std::vector<double> rqd(rq.begin(), rq.end()), rad(ra.begin(), ra.end());
            pc.add_row(pq.data(), rqd.data(), sc.qkvz_n());
            pc.add_row(pa.data(), rad.data(), sc.ab_n());
          }
          k::gdn_ref_core(sc, rw[c], run, pq.data(), pa.data(), o.data(), y.data());
          after[c].push_back(run);
          for (int n = 0; n < VW; ++n) yd[n] = k::bf16_value(yb[static_cast<size_t>(r) * VW + n]);
          for (int h = 0; h < sc.vh(); ++h) {
            std::vector<double> yr(y.begin() + h * 128, y.begin() + (h + 1) * 128);
            yc.add_row(&yd[h * 128], yr.data(), 128);
          }
          k::gdn_ref_out(sc, rw[c], yd.data(), part.data());
          for (int n = 0; n < H; ++n) ref[static_cast<size_t>(r) * H + n] += part[n];
        }
        if (steps[s].accept > 0) rs[c] = after[c][steps[s].accept - 1];
      }
      for (int r = 0; r < R; ++r) {
        std::vector<float> a(H);
        for (int n = 0; n < H; ++n) a[n] = k::bf16_value(ob[0][static_cast<size_t>(r) * H + n]);
        oc.add_row(a.data(), &ref[static_cast<size_t>(r) * H], H);
      }
    }
    std::printf("\nlayer %d (%d steps, rows 1-4, accept 0-4, N(0, 1) input):\n", L, S);
    std::snprintf(what, sizeof what, "layer %d projections q|k|v|z (W4) and a|b (bf16 dots) vs the dequantised reference, step 0 card 0: %s", L, pc.text().c_str());
    check(pc.scaled <= 1.0 / 128 && pc.off * 100 <= pc.n, what);
    std::snprintf(what, sizeof what, "layer %d y (conv, gates, recurrence, gated norm x silu(z); both cards) vs reference from the device's projections: %s", L,
                  yc.text().c_str());
    check(yc.scaled <= 1.0 / 64 && yc.off * 200 <= yc.n, what);  // 2 ulps: y is rounded after o (see test_gdn)
    // Gate 1e-3 of max |S| (probe gdn: 1e-4): the reference re-rounds the conv output to bf16 from its own fp32 bits, and
    // one k element landing on the other side of a rounding boundary (1 in ~10^4 elements) moves that token's d k^T update
    // in the three value heads of its key head by up to 2^-8 of |d| - a rank-1 |dS| up to ~5e-4 of max |S| (layer 62,
    // Not a kernel bug: plain 1-row chains agree to 2e-7.
    std::snprintf(what, sizeof what, "layer %d committed S vs reference: max |dS| %.2e of max |S| %.2e (gate 1e-3)", L, s_err, s_max);
    check(s_err <= 1e-3 * s_max && s_max > 0, what);
    {  // the state error per step (card 0): where it enters and whether it grows
      std::string line = "       |dS| per step (card 0, before the step's commit; head.v.k of the worst):";
      for (size_t i = 0; i < s_steps[0].size(); ++i) {
        const auto &e = s_steps[0][i];
        char b[64];
        std::snprintf(b, sizeof b, " %.1e@%zu.%zu.%zu", e.err, e.at / 16384, e.at / 128 % 128, e.at % 128);
        line += b;
      }
      std::printf("%s\n", line.c_str());
    }
    std::snprintf(what, sizeof what, "layer %d out (W4 out_proj + mailbox sum) vs the sum of the per-card references from the device's y: %s", L, oc.text().c_str());
    check(oc.scaled <= 1.0 / 128 && oc.off * 100 <= oc.n, what);
    std::snprintf(what, sizeof what, "layer %d both cards bit-identical output every step", L);
    check(same, what);
    for (int c = 0; c < 2; ++c) free_dev(d[c]);
  }

  // ---- timing: every loaded layer once per step, chained ----
  const auto s0 = gs.shape_of(0);
  const auto &w0 = gs.w[0][gs.layers[0]];
  const double ab_bytes = 2.0 * s0.vh() * s0.hidden * 2;
  const double bytes = static_cast<double>(w0.qkvz4.bytes() + w0.out4.bytes()) + ab_bytes + 2.0 * k::gdn_S_elems(s0) * 4 +
                       static_cast<double>(s0.conv_ch()) * 4 * 2;
  const double floor_bw = 604e9, floor_us = bytes / floor_bw * 1e6;
  const int NL = static_cast<int>(gs.layers.size());
  std::printf("\nfloor per layer per card: %.2f MB (qkvz W4 %.2f, ab %.2f, out W4 %.2f, state read+write %.2f) -> %.1f us at %.0f GB/s; x48 layers %.0f us\n",
              bytes / 1e6, w0.qkvz4.bytes() / 1e6, ab_bytes / 1e6, w0.out4.bytes() / 1e6, 2.0 * k::gdn_S_elems(s0) * 4 / 1e6, floor_us, floor_bw / 1e9,
              floor_us * 48);
  std::vector<Dev> dv[2];
  for (int c = 0; c < 2; ++c)
    for (int l : gs.layers) dv[c].push_back(make_dev(c, gs.w[c][l]));
  std::printf("rows accept |  qkvz   core    out reduce (us, card 0 median) | GPU per call, steady: card 0  card 1 | x floor\n");
  int parity = 0;
  const int skip = std::min(NL - 1, 1);  // the span starts after this many layers (a 48-layer chain skips 4 in probe gdn)
  for (const auto cfgp : {std::pair<int, int>{1, 1}, {4, 4}, {4, 1}}) {
    const int R = cfgp.first, A = cfgp.second;
    for (int c = 0; c < 2; ++c)
      for (auto &dd : dv[c]) fab.compute(c).memcpy(dd.accept, &A, 4).wait();
    std::vector<double> kt[4], gspan[2];
    const int reps = std::max(4, iters / 10);
    for (int rep = 0; rep < reps + 2; ++rep) {
      std::vector<k::GdnEvents> ev[2];
      for (int i = 0; i < NL; ++i)
        for (int c = 0; c < 2; ++c) {
          ev[c].push_back(k::gdn_decode(fab.compute(c), gs.shape_of(c), R, xin[c][0], gs.w[c][gs.layers[i]], dv[c][i].st, parity, scr[c], &wire[c],
                                        wire_parity, out[c]));
          if (c == 1) wire_parity ^= 1;
        }
      parity ^= 1;
      for (int c = 0; c < 2; ++c) fab.compute(c).wait_and_throw();
      if (rep < 2) continue;
      for (int i = skip; i < NL; ++i) {
        const auto &e = ev[0][i];
        kt[0].push_back(ev_us(e.qkvz));
        kt[1].push_back(ev_us(e.core));
        kt[2].push_back(ev_us(e.out));
        kt[3].push_back(ev_us(e.reduce));
      }
      for (int c = 0; c < 2; ++c) {
        const auto a = ev[c][0].qkvz.get_profiling_info<sycl::info::event_profiling::command_start>();
        const auto b = ev[c][NL - 1].reduce.get_profiling_info<sycl::info::event_profiling::command_end>();
        gspan[c].push_back(static_cast<double>(b - a) * 1e-3 / NL);
      }
    }
    std::printf("%4d %6d | %6.2f %6.2f %6.2f %6.2f                    | %27.1f %7.1f | %.2f\n", R, A, median(kt[0]), median(kt[1]), median(kt[2]),
                median(kt[3]), median(gspan[0]), median(gspan[1]), std::max(median(gspan[0]), median(gspan[1])) / floor_us);
  }
  std::printf("(GPU per call = the chain of %d loaded layer(s) from the first qkvz start to the last reduce end, divided by %d)\n", NL, NL);
  uint32_t err[2];
  for (int c = 0; c < 2; ++c) fab.compute(c).memcpy(&err[c], wire[c].error, 4).wait();
  check(!err[0] && !err[1], "gdn35 mailbox: no timeout");
  for (int c = 0; c < 2; ++c)
    for (auto &dd : dv[c]) free_dev(dd);
  for (int c = 0; c < 2; ++c)
    for (auto *p : xin[c]) fab.release(p);
  std::printf("gdn35: %s\n", rc ? "FAIL" : "PASS");
  return rc;
}
