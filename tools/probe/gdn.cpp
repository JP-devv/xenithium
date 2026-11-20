// xenithium-probe gdn: the Gated DeltaNet decode block on real weights, both cards.
// xenithium-probe gdn --model DIR [--layers 0,1,2,44] [--iters I]
// Loads the hyper-connections, embedding, lm_head and every linear-attention layer (dense plan,
// heads split). The input of layer L at each step is that layer's attention-mixer output on the
// identity-block chain (embed -> mixers, blocks = identity, as probe hc) for that step's token
// ids: real weights on a real-weight data path, not a model run ("chain input").
// Per layer, a plan of steps with 1-4 rows and accept counts 0-4:
// y (core) vs a reference teacher-forced at the device's projections, the committed state,
// out vs the sum of the per-card references from the device's y, both cards bit-identical.
// Then the timing: one call per linear-attention layer chained like a step (no host wait), for
// plain decode (1 row, accept 1) and 4 rows with 4 replayed, against the read floor.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

#include "xenithium/arch/qwen38/arch.hpp"
#include "xenithium/arch/qwen38/gdn.hpp"
#include "xenithium/arch/qwen38/hc.hpp"
#include "xenithium/arch/qwen38/head.hpp"
#include "xenithium/kernels/linear_attention/gdn.hpp"
#include "xenithium/kernels/residual/hc.hpp"
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

// Card c's reference weights for layer L, from the checkpoint (int8 rows quantised as the loader does).
k::GdnRefWeights ref_weights(const m::Checkpoint &ck, const q38::Config &cfg, int L, int c, int cards, int threads) {
  const std::string P = std::string(q38::kPrefix) + "layers." + std::to_string(L) + ".linear_attn.";
  std::vector<const m::TensorInfo *> ts;
  for (const char *t : {"in_proj_qkv.weight", "in_proj_z.weight", "in_proj_a.weight", "in_proj_b.weight", "out_proj.weight", "conv1d.weight",
                        "A_log", "dt_bias", "norm.weight"})
    ts.push_back(&ck.at(P + t));
  m::BulkOptions bo;
  bo.threads = threads;
  const auto buf = m::bulk_read(ck, ts, bo);
  auto u16 = [&](const char *t) { return buf.as<uint16_t>(ck.at(P + t)); };
  const int H = cfg.hidden, D = cfg.lin_k_dim, KH = cfg.lin_k_heads / cards, VH = cfg.lin_v_heads / cards;
  const int KWa = cfg.lin_k_heads * D, VWa = cfg.lin_v_heads * D, kh0 = c * KH, vh0 = c * VH;
  std::vector<std::pair<const uint16_t *, int>> qkvz_rows;  // (tensor, first row) blocks of rows
  std::vector<uint16_t> rows;
  auto take = [&](const uint16_t *t, int r0, int n, int cols, std::vector<uint16_t> &dst) {
    dst.insert(dst.end(), t + static_cast<size_t>(r0) * cols, t + static_cast<size_t>(r0 + n) * cols);
  };
  take(u16("in_proj_qkv.weight"), kh0 * D, KH * D, H, rows);
  take(u16("in_proj_qkv.weight"), KWa + kh0 * D, KH * D, H, rows);
  take(u16("in_proj_qkv.weight"), 2 * KWa + vh0 * D, VH * D, H, rows);
  take(u16("in_proj_z.weight"), vh0 * D, VH * D, H, rows);
  auto deq = [](const std::vector<uint16_t> &w, int n, int kk) {
    std::vector<int8_t> codes(w.size());
    std::vector<float> sc(n);
    k::quantize_i8_rows(w.data(), n, kk, codes.data(), sc.data(), 1);
    std::vector<float> d(w.size());
    for (int r = 0; r < n; ++r)
      for (int j = 0; j < kk; ++j) d[static_cast<size_t>(r) * kk + j] = codes[static_cast<size_t>(r) * kk + j] * sc[r];
    return d;
  };
  auto vals = [](const std::vector<uint16_t> &w) {
    std::vector<float> v(w.size());
    for (size_t i = 0; i < w.size(); ++i) v[i] = k::bf16_value(w[i]);
    return v;
  };
  k::GdnRefWeights rw;
  rw.qkvz = deq(rows, static_cast<int>(rows.size() / H), H);
  std::vector<uint16_t> ab, conv, out(static_cast<size_t>(H) * VH * D);
  take(u16("in_proj_a.weight"), vh0, VH, H, ab);
  take(u16("in_proj_b.weight"), vh0, VH, H, ab);
  rw.ab = vals(ab);
  const uint16_t *o = u16("out_proj.weight");
  for (int n = 0; n < H; ++n) std::memcpy(&out[static_cast<size_t>(n) * VH * D], o + static_cast<size_t>(n) * VWa + vh0 * D, VH * D * 2ull);
  rw.out = deq(out, H, VH * D);
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

int cmd_gdn_run(const std::string &model, const std::string &layers_arg, int iters, int threads) {
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
    if (L < 0 || L >= cfg.layers || cfg.full_attention[L]) {
      std::fprintf(stderr, "gdn: layer %d is not a linear-attention layer\n", L);
      return 2;
    }
  const std::string P = q38::kPrefix;
  std::vector<m::DenseSpec> plan;
  for (const auto &sp : arch->dense_plan(ck, m::DeployParams{}))
    if (sp.comp == m::Component::hyper_connection || sp.comp == m::Component::linear_attention || sp.name == P + "embed_tokens.weight" ||
        sp.name == "lm_head.weight")
      plan.push_back(sp);
  const auto loader = xenithium::l0::Loader::open();
  xenithium::Fabric fab(loader, {0, 1});
  xenithium::load::DenseStore dense(fab);
  dense.load(ck, plan, threads);
  const auto &cb = dense.stats().card_bytes[0];
  std::printf("gdn: %zu dense tensors loaded in %.2f s (linear attention %.3f GiB per card, heads split)\n", plan.size(), dense.stats().total_s,
              (cb.count(m::Component::linear_attention) ? cb.at(m::Component::linear_attention) : 0) / 1073741824.0);
  q38::Head head(fab, dense, cfg);
  const q38::HcSet hs = q38::HcSet::from(dense, cfg, 2);
  const q38::GdnSet gs = q38::GdnSet::from(dense, cfg, 2);
  const auto &sh = hs.shape;
  const int rows4 = q38::Head::kRows, H = sh.hidden;

  // Steps: (rows, accept of this step's rows by the next step) and each step's token ids.
  struct Step {
    int rows, accept;
  };
  const std::vector<Step> steps = {{1, 1}, {4, 2}, {4, 4}, {1, 1}, {3, 3}, {2, 2}, {1, 1}, {4, 1}, {1, 0}, {2, 2}};
  const int S = static_cast<int>(steps.size());
  // Inputs: the attention-mixer output of each requested layer for each step's 4 token ids.
  bf16 *outA[2], *outB[2];
  std::vector<std::vector<bf16 *>> xin[2];  // [card][layer idx][step]
  for (int c = 0; c < 2; ++c) {
    outA[c] = static_cast<bf16 *>(fab.alloc(c, static_cast<size_t>(rows4) * H * 2));
    outB[c] = static_cast<bf16 *>(fab.alloc(c, static_cast<size_t>(rows4) * H * 2));
    xin[c].resize(layers.size());
    for (auto &v : xin[c])
      for (int s = 0; s < S; ++s) v.push_back(static_cast<bf16 *>(fab.alloc(c, static_cast<size_t>(rows4) * H * 2)));
  }
  for (int s = 0; s < S; ++s) {
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

  // Per-card scratch and wire.
  k::GdnScratch scr[2];
  k::PairWire wire[2];
  bf16 *out[2];
  for (int c = 0; c < 2; ++c) {
    const auto s = gs.shape_of(c);
    scr[c].core = static_cast<bf16 *>(fab.alloc(c, 4ull * s.val_w() * 2));
    scr[c].partial = static_cast<float *>(fab.alloc(c, 4ull * H * 4));
    wire[c].inbox = static_cast<uint32_t *>(fab.alloc_mailbox(c, k::pair_wire_words(H) * 4));
    wire[c].error = static_cast<uint32_t *>(fab.alloc(c, 4096));
    out[c] = static_cast<bf16 *>(fab.alloc(c, 4ull * H * 2));
    fab.compute(c).memset(scr[c].core, 0, 4ull * s.val_w() * 2).wait();
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
    k::GdnRefWeights rw[2] = {ref_weights(ck, cfg, L, 0, 2, threads), ref_weights(ck, cfg, L, 1, 2, threads)};
    k::GdnRefState rs[2];
    for (int c = 0; c < 2; ++c) rs[c].reset(gs.shape_of(c));
    BoundCmp yc, oc;
    bool same = true;
    double s_err = 0, s_max = 0;
    for (int s = 0; s < S; ++s) {
      const int R = steps[s].rows, prev = s ? steps[s - 1].accept : 0;
      for (int c = 0; c < 2; ++c) {
        fab.compute(c).memcpy(d[c].accept, &prev, 4).wait();
        k::gdn_decode(fab.compute(c), gs.shape_of(c), R, xin[c][li][s], gs.w[c][L], d[c].st, s, scr[c], &wire[c], wire_parity, out[c]);
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
      for (int c = 0; c < 2; ++c) {
        const auto sc = gs.shape_of(c);
        const auto &w = gs.w[c][L];
        const int VW = sc.val_w();
        std::vector<float> proj(4ull * w.qkvz.n), ab(4ull * sc.ab_n());
        std::vector<uint16_t> yb(static_cast<size_t>(R) * VW);
        fab.compute(c).memcpy(proj.data(), d[c].st.y[s & 1], proj.size() * 4);
        fab.compute(c).memcpy(ab.data(), d[c].st.ab[s & 1], ab.size() * 4);
        fab.compute(c).memcpy(yb.data(), scr[c].core, yb.size() * 2).wait();
        if (s == S - 1) {  // committed S now = reference before the last commit
          std::vector<float> Sd(k::gdn_S_elems(sc));
          fab.compute(c).memcpy(Sd.data(), d[c].st.S, Sd.size() * 4).wait();
          for (size_t j = 0; j < Sd.size(); ++j) {
            s_err = std::max(s_err, std::fabs(Sd[j] - rs[c].S[j]));
            s_max = std::max(s_max, std::fabs(rs[c].S[j]));
          }
        }
        k::GdnRefState run = rs[c];
        for (int r = 0; r < R; ++r) {
          std::vector<float> pq(sc.qkvz_n()), pa(sc.ab_n()), o(VW), y(VW), yd(VW);
          for (int n = 0; n < sc.qkvz_n(); ++n) pq[n] = bfr(proj[static_cast<size_t>(r) * w.qkvz.n + n]);
          for (int n = 0; n < sc.ab_n(); ++n) pa[n] = bfr(ab[static_cast<size_t>(r) * sc.ab_n() + n]);
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
    std::printf("\nlayer %d (%d steps, rows 1-4, accept 0-4, chain input):\n", L, S);
    std::snprintf(what, sizeof what, "layer %d y (conv, gates, recurrence, gated norm; both cards) vs reference from the device's projections: %s", L,
                  yc.text().c_str());
    check(yc.scaled <= 1.0 / 64 && yc.off * 200 <= yc.n, what);  // 2 ulps: y is rounded after o (see test_gdn)
    std::snprintf(what, sizeof what, "layer %d committed S vs reference: max |dS| %.2e of max |S| %.2e", L, s_err, s_max);
    check(s_err <= 1e-4 * s_max && s_max > 0, what);
    std::snprintf(what, sizeof what, "layer %d out vs the sum of the per-card references from the device's y: %s", L, oc.text().c_str());
    check(oc.scaled <= 1.0 / 128 && oc.off * 100 <= oc.n, what);
    std::snprintf(what, sizeof what, "layer %d both cards bit-identical output every step", L);
    check(same, what);
    for (int c = 0; c < 2; ++c) free_dev(d[c]);
  }

  // ---- timing: every linear-attention layer once per step, chained ----
  const auto s0 = gs.shape_of(0);
  const auto &w0 = gs.w[0][gs.layers[0]];
  const double ab_bytes = 2.0 * s0.vh() * s0.hidden * 2;
  const double bytes = static_cast<double>(w0.qkvz.bytes() + w0.out.bytes()) + ab_bytes + 2.0 * k::gdn_S_elems(s0) * 4 +
                       static_cast<double>(s0.conv_ch()) * 4 * 2;
  const double floor_bw = 605.9e9, floor_us = bytes / floor_bw * 1e6;
  const int NL = static_cast<int>(gs.layers.size());
  std::printf("\nfloor per layer per card: %.2f MB (qkvz %.2f, ab %.2f, out %.2f, state read+write %.2f) -> %.1f us at %.1f GB/s; x%d layers %.0f us\n",
              bytes / 1e6, w0.qkvz.bytes() / 1e6, ab_bytes / 1e6, w0.out.bytes() / 1e6, 2.0 * k::gdn_S_elems(s0) * 4 / 1e6, floor_us,
              floor_bw / 1e9, NL, floor_us * NL);
  std::vector<Dev> dv[2];
  for (int c = 0; c < 2; ++c)
    for (int l : gs.layers) dv[c].push_back(make_dev(c, gs.w[c][l]));
  std::printf("rows accept |  qkvz   core    out reduce (us, card 0 median) | GPU per call, steady: card 0  card 1 | x floor\n");
  int parity = 0;
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
          ev[c].push_back(k::gdn_decode(fab.compute(c), gs.shape_of(c), R, xin[c][0][0], gs.w[c][gs.layers[i]], dv[c][i].st, parity, scr[c],
                                        &wire[c], wire_parity, out[c]));
          if (c == 1) wire_parity ^= 1;
        }
      parity ^= 1;
      for (int c = 0; c < 2; ++c) fab.compute(c).wait_and_throw();
      if (rep < 2) continue;
      for (int i = 4; i < NL; ++i) {
        const auto &e = ev[0][i];
        kt[0].push_back(ev_us(e.qkvz));
        kt[1].push_back(ev_us(e.core));
        kt[2].push_back(ev_us(e.out));
        kt[3].push_back(ev_us(e.reduce));
      }
      for (int c = 0; c < 2; ++c) {
        const auto a = ev[c][3].reduce.get_profiling_info<sycl::info::event_profiling::command_end>();
        const auto b = ev[c][NL - 1].reduce.get_profiling_info<sycl::info::event_profiling::command_end>();
        gspan[c].push_back(static_cast<double>(b - a) * 1e-3 / (NL - 4));
      }
    }
    std::printf("%4d %6d | %6.2f %6.2f %6.2f %6.2f                    | %27.1f %7.1f | %.2f\n", R, A, median(kt[0]), median(kt[1]),
                median(kt[2]), median(kt[3]), median(gspan[0]), median(gspan[1]),
                std::max(median(gspan[0]), median(gspan[1])) / floor_us);
  }
  uint32_t err[2];
  for (int c = 0; c < 2; ++c) fab.compute(c).memcpy(&err[c], wire[c].error, 4).wait();
  check(!err[0] && !err[1], "gdn mailbox: no timeout");
  for (int c = 0; c < 2; ++c)
    for (auto &dd : dv[c]) free_dev(dd);
  return rc;
}
