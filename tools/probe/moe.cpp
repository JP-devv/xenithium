// xenithium-probe moe: the MoE decode block on real weights, both cards.
// Loads the hyper-connections, embedding, lm_head, routers and shared experts (dense plan) and
// the routed experts of the named layers (all 512, so every route is hot; one layer at a time,
// 0.65 GB per card). The MoE input of layer L is that layer's MLP mixer output on the
// identity-block chain (embed -> mixers, blocks = identity, as probe hc), so routing comes
// from real weights on a real-weight data path, not from a model run: label "chain routing".
// Per layer: routing vs a host reference, h vs the reference (<= 4 ulps, <= 0.5 % off), the
// output vs the reference fed the device's h, both cards bit-identical, cold experts (pinned
// host, read over PCIe) bit-identical to hot, then the timing table against the read floor.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

#include "warmup.hpp"
#include "xenithium/arch/qwen38/arch.hpp"
#include "xenithium/arch/qwen38/experts.hpp"
#include "xenithium/arch/qwen38/hc.hpp"
#include "xenithium/arch/qwen38/head.hpp"
#include "xenithium/arch/qwen38/moe.hpp"
#include "xenithium/kernels/residual/hc.hpp"
#include "xenithium/kernels/moe/moe.hpp"
#include "xenithium/kernels/pack.hpp"
#include "xenithium/kernels/rows.hpp"
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
double pct(std::vector<double> v, double p) {
  if (v.empty()) return 0;
  std::sort(v.begin(), v.end());
  return v[std::min(v.size() - 1, static_cast<size_t>(p * static_cast<double>(v.size())))];
}

std::vector<float> f32_of_bf16(const uint16_t *p, size_t n) {
  std::vector<float> v(n);
  for (size_t i = 0; i < n; ++i) v[i] = k::bf16_value(p[i]);
  return v;
}
std::vector<float> deq_i8(const uint16_t *w, int n, int kk) {
  std::vector<int8_t> codes(static_cast<size_t>(n) * kk);
  std::vector<float> sc(n);
  k::quantize_i8_rows(w, n, kk, codes.data(), sc.data(), 1);
  std::vector<float> d(codes.size());
  for (int r = 0; r < n; ++r)
    for (int c = 0; c < kk; ++c) d[static_cast<size_t>(r) * kk + c] = codes[static_cast<size_t>(r) * kk + c] * sc[r];
  return d;
}
// GPTQ qweight [k/8][n] + fp16 scales [k/128][n] -> w[k][n]
std::vector<float> deq_w4(const uint32_t *qw, const uint16_t *sc, int n, int kk) {
  std::vector<float> d(static_cast<size_t>(kk) * n);
  for (int r = 0; r < kk; ++r)
    for (int c = 0; c < n; ++c) {
      const int q = static_cast<int>((qw[static_cast<size_t>(r / 8) * n + c] >> (4 * (r % 8))) & 15u);
      sycl::half h;
      std::memcpy(&h, &sc[static_cast<size_t>(r / 128) * n + c], 2);
      d[static_cast<size_t>(r) * n + c] = static_cast<float>(q - 8) * static_cast<float>(h);
    }
  return d;
}

struct Cmp {
  double scaled = 0;
  size_t off = 0, n = 0;
};
Cmp compare(const std::vector<uint16_t> &a, const std::vector<float> &ref, size_t row) {
  Cmp c;
  c.n = a.size();
  for (size_t r0 = 0; r0 < a.size(); r0 += row) {
    double ss = 0;
    for (size_t i = r0; i < r0 + row; ++i) ss += static_cast<double>(ref[i]) * ref[i];
    const double rms = std::sqrt(ss / static_cast<double>(row));
    for (size_t i = r0; i < r0 + row; ++i) {
      const double d = std::fabs(k::bf16_value(a[i]) - ref[i]);
      c.off += k::bf16_value(a[i]) != ref[i];
      c.scaled = std::max(c.scaled, d / std::max({static_cast<double>(std::fabs(ref[i])), rms, 1e-30}));
    }
  }
  return c;
}

}  // namespace

int cmd_moe_run(const std::string &model, const std::string &layers_arg, int iters, int threads) {
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
  const std::string P = "model.language_model.";
  std::vector<m::DenseSpec> plan;
  for (const auto &sp : arch->dense_plan(ck, m::DeployParams{}))
    if (sp.comp == m::Component::hyper_connection || sp.comp == m::Component::router || sp.comp == m::Component::shared_expert ||
        sp.name == P + "embed_tokens.weight" || sp.name == "lm_head.weight")
      plan.push_back(sp);
  const auto loader = xenithium::l0::Loader::open();
  xenithium::Fabric fab(loader, {0, 1});
  xenithium::load::DenseStore dense(fab);
  dense.load(ck, plan, threads);
  std::printf("moe: %zu dense tensors loaded in %.2f s (router + shared expert %.3f GiB per card, replicated)\n", plan.size(), dense.stats().total_s,
              ((dense.stats().card_bytes[0].count(m::Component::router) ? dense.stats().card_bytes[0].at(m::Component::router) : 0) +
               (dense.stats().card_bytes[0].count(m::Component::shared_expert) ? dense.stats().card_bytes[0].at(m::Component::shared_expert) : 0)) /
                  1073741824.0);
  q38::Head head(fab, dense, cfg);
  const q38::HcSet hs = q38::HcSet::from(dense, cfg, 2);
  const q38::MoeSet ms = q38::MoeSet::from(dense, cfg, 2);
  const auto &sh = hs.shape;
  const int rows = q38::Head::kRows, H = sh.hidden;
  const int32_t ids[rows] = {11, 1000, 50000, 248043};
  head.set_ids(ids, rows);

  // MoE inputs: the MLP mixer output of each requested layer on the identity-block chain.
  bf16 *outA[2], *outB[2];
  std::vector<bf16 *> xin[2];
  for (int c = 0; c < 2; ++c) {
    outA[c] = static_cast<bf16 *>(fab.alloc(c, static_cast<size_t>(rows) * H * 2));
    outB[c] = static_cast<bf16 *>(fab.alloc(c, static_cast<size_t>(rows) * H * 2));
    for (size_t i = 0; i < layers.size(); ++i) xin[c].push_back(static_cast<bf16 *>(fab.alloc(c, static_cast<size_t>(rows) * H * 2)));
  }
  head.embed(rows);
  for (int l = 0, li = 0; l <= layers.back(); ++l)
    for (int part = 0; part < 2; ++part) {
      for (int c = 0; c < 2; ++c) {
        auto &q = fab.compute(c);
        auto &st = head.state(c);
        const bf16 *blk = part == 0 ? (l > 0 ? outB[c] : nullptr) : outA[c];
        if (part == 0 && l == hs.ple_layer && blk) {
          k::hc_combine(q, sh, rows, st, blk);
          blk = nullptr;
        }
        k::hc_mixer(q, sh, rows, st, blk, part == 0 ? hs.attn[c][l] : hs.mlp[c][l], head.scratch(c), part == 0 ? outA[c] : outB[c]);
        if (part == 1 && li < static_cast<int>(layers.size()) && layers[li] == l) q.memcpy(xin[c][li], outB[c], static_cast<size_t>(rows) * H * 2);
      }
      if (part == 1 && li < static_cast<int>(layers.size()) && layers[li] == l) ++li;
    }
  for (int c = 0; c < 2; ++c) fab.compute(c).wait_and_throw();

  // Per-card MoE scratch and wire.
  k::MoeScratch scr[2];
  k::MoeWire wire[2];
  bf16 *out[2];
  for (int c = 0; c < 2; ++c) {
    const auto s = ms.shape_of(c);
    scr[c].logits = static_cast<float *>(fab.alloc(c, 4ull * ms.w[c][0].router.n * 4));
    scr[c].route = static_cast<k::MoeRoute *>(fab.alloc(c, sizeof(k::MoeRoute)));
    scr[c].h = static_cast<bf16 *>(fab.alloc(c, k::moe_h_elems(s) * 2));
    scr[c].partial = static_cast<float *>(fab.alloc(c, k::moe_partial_elems(s) * 4));
    wire[c].inbox = static_cast<uint32_t *>(fab.alloc_mailbox(c, k::moe_wire_words(s) * 4));
    wire[c].error = static_cast<uint32_t *>(fab.alloc(c, 4096));
    out[c] = static_cast<bf16 *>(fab.alloc(c, static_cast<size_t>(rows) * H * 2));
  }
  for (int c = 0; c < 2; ++c) {
    wire[c].peer_inbox = wire[1 - c].inbox;
    k::moe_wire_init(fab.compute(c), ms.shape_of(c), wire[c].inbox).wait();
    fab.compute(c).memset(wire[c].error, 0, 4).wait();
  }

  const q38::ExpertLayout lay = q38::ExpertLayout::of(cfg);
  const int E = cfg.experts, per = E / 2, I = cfg.moe_inter, SI = cfg.shared_inter;
  const double floor_bw = 605.9e9;
  int rc = 0;
  auto check = [&](bool ok, const std::string &what) {
    std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
    if (!ok) rc = 1;
  };
  uint8_t *vram[2] = {static_cast<uint8_t *>(fab.alloc(0, per * lay.stride)), static_cast<uint8_t *>(fab.alloc(1, per * lay.stride))};
  uint8_t *pinned = static_cast<uint8_t *>(fab.alloc_host(per * lay.stride));  // card 1's experts, cold copy
  const uint8_t **table_d[2] = {static_cast<const uint8_t **>(fab.alloc(0, per * 8)), static_cast<const uint8_t **>(fab.alloc(1, per * 8))};
  std::vector<uint8_t> host(per * lay.stride);

  for (size_t li = 0; li < layers.size(); ++li) {
    const int L = layers[li];
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<const m::TensorInfo *> ts;
    for (int e = 0; e < E; ++e)
      for (const auto *t : q38::expert_tensors(ck, L, e)) ts.push_back(t);
    const std::string LP = P + "layers." + std::to_string(L) + ".mlp.";
    for (const char *nm : {"gate.weight", "shared_expert_gate.weight", "shared_expert.gate_proj.weight", "shared_expert.up_proj.weight",
                           "shared_expert.down_proj.weight"})
      ts.push_back(&ck.at(LP + nm));
    m::BulkOptions bo;
    bo.threads = threads;
    const auto buf = m::bulk_read(ck, ts, bo);
    for (int c = 0; c < 2; ++c) {
      for (int i = 0; i < per; ++i) q38::pack_expert(lay, ck, buf, L, c * per + i, host.data() + static_cast<size_t>(i) * lay.stride);
      fab.compute(c).memcpy(vram[c], host.data(), host.size()).wait();
      if (c == 1) std::memcpy(pinned, host.data(), host.size());
    }
    std::vector<const uint8_t *> tab(per);
    for (int c = 0; c < 2; ++c) {
      for (int i = 0; i < per; ++i) tab[i] = vram[c] + static_cast<size_t>(i) * lay.stride;
      fab.compute(c).memcpy(table_d[c], tab.data(), per * 8).wait();
    }
    std::printf("\nlayer %d: experts read + packed + uploaded in %.2f s (all %d hot)\n", L,
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), E);
    k::MoeWeights w[2];
    for (int c = 0; c < 2; ++c) {
      w[c] = ms.w[c][L];
      w[c].blobs = table_d[c];
    }

    // Reference weights.
    k::MoeRefWeights rw;
    rw.router = f32_of_bf16(buf.as<uint16_t>(ck.at(LP + "gate.weight")), static_cast<size_t>(E) * H);
    rw.shared_gate = f32_of_bf16(buf.as<uint16_t>(ck.at(LP + "shared_expert_gate.weight")), H);
    rw.s_gate = deq_i8(buf.as<uint16_t>(ck.at(LP + "shared_expert.gate_proj.weight")), SI, H);
    rw.s_up = deq_i8(buf.as<uint16_t>(ck.at(LP + "shared_expert.up_proj.weight")), SI, H);
    rw.s_down = deq_i8(buf.as<uint16_t>(ck.at(LP + "shared_expert.down_proj.weight")), H, SI);
    std::vector<uint16_t> xb(static_cast<size_t>(rows) * H);
    fab.compute(0).memcpy(xb.data(), xin[0][li], xb.size() * 2).wait();
    const auto x = f32_of_bf16(xb.data(), xb.size());
    const auto s2 = ms.shape_of(0);
    k::MoeRefOut ro;
    k::moe_route_ref(s2, rows, x.data(), rw, ro);
    std::vector<k::MoeRefExpert> store(E);
    rw.experts.assign(E, nullptr);
    for (int e : ro.ids) {
      if (rw.experts[e]) continue;
      auto g = [&](const char *proj, const char *part) { return &ck.at(q38::expert_name(L, e, proj, part)); };
      store[e].gate = deq_w4(buf.as<uint32_t>(*g("gate_proj", "qweight")), buf.as<uint16_t>(*g("gate_proj", "scales")), I, H);
      store[e].up = deq_w4(buf.as<uint32_t>(*g("up_proj", "qweight")), buf.as<uint16_t>(*g("up_proj", "scales")), I, H);
      store[e].down = deq_w4(buf.as<uint32_t>(*g("down_proj", "qweight")), buf.as<uint16_t>(*g("down_proj", "scales")), H, I);
      rw.experts[e] = &store[e];
    }

    auto run = [&](int parity, int ks, k::MoeEvents *ev, int dks = 0) {
      for (int c = 0; c < 2; ++c) {
        const auto e = k::moe_decode(fab.compute(c), ms.shape_of(c), rows, xin[c][li], w[c], scr[c], &wire[c], parity, out[c], ks, dks);
        if (ev) ev[c] = e;
      }
    };
    run(0, 0, nullptr);
    for (int c = 0; c < 2; ++c) fab.compute(c).wait_and_throw();
    std::vector<uint16_t> o[2];
    k::MoeRoute rt[2];
    k::MoeRefH dh;
    const int HW = std::max(I, SI);
    for (int c = 0; c < 2; ++c) {
      o[c].resize(static_cast<size_t>(rows) * H);
      fab.compute(c).memcpy(o[c].data(), out[c], o[c].size() * 2);
      fab.compute(c).memcpy(&rt[c], scr[c].route, sizeof(k::MoeRoute)).wait();
      std::vector<uint16_t> hb(k::moe_h_elems(ms.shape_of(c)));
      fab.compute(c).memcpy(hb.data(), scr[c].h, hb.size() * 2).wait();
      auto vec = [&](int slot, int r, int n0, int n1, int n) {
        std::vector<float> v(n, 0.f);
        // h is [slot][row_shape(rows)][HW]: 8 rows per slot since Head:kRows = 8 (this read assumed 4)
        for (int i = n0; i < n1; ++i) v[i] = k::bf16_value(hb[(static_cast<size_t>(slot) * k::row_shape(rows) + r) * HW + i]);
        return v;
      };
      for (int u = 0; u < rt[c].count; ++u)
        for (int r = 0; r < rows; ++r)
          if ((rt[c].mask[u] >> r) & 1u) dh[{r, rt[c].expert[u]}] = vec(u, r, 0, I, I);
      for (int r = 0; r < rows; ++r) dh[{r, -1 - c}] = vec(s2.slots(), r, c * SI / 2, (c + 1) * SI / 2, SI);
    }
    bool route_ok = true;
    int near_tie = 0;
    for (int r = 0; r < rows; ++r)
      for (int j = 0; j < cfg.topk; ++j) {
        route_ok &= rt[0].ids[r][j] == ro.ids[static_cast<size_t>(r) * cfg.topk + j] && rt[1].ids[r][j] == rt[0].ids[r][j];
        route_ok &= std::fabs(rt[0].topw[r][j] - ro.topw[static_cast<size_t>(r) * cfg.topk + j]) <= 1e-5f;
      }
    // Is the k-th / (k+1)-th boundary a tie in bf16 (the reference's order then decides)?
    for (int r = 0; r < rows; ++r) {
      std::vector<float> lg(ro.logits.begin() + static_cast<size_t>(r) * E, ro.logits.begin() + static_cast<size_t>(r + 1) * E);
      std::sort(lg.rbegin(), lg.rend());
      near_tie += lg[cfg.topk - 1] == lg[cfg.topk];
    }
    char what[256];
    std::snprintf(what, sizeof what, "layer %d routing: top-%d ids on both cards == host reference, weights within 1e-5 (%d of %d rows tie at the k-th logit)", L,
                  cfg.topk, near_tie, rows);
    check(route_ok, what);
    k::MoeRefOut rof;
    if (route_ok) {
      k::moe_ref(s2, rows, x.data(), rw, ro);
      size_t hn = 0, hoff = 0;
      int hulp = 0;
      bool keys = dh.size() == ro.h.size();
      for (const auto &[key, v] : ro.h) {
        const auto it = dh.find(key);
        if (it == dh.end()) {
          keys = false;
          continue;
        }
        for (size_t i = 0; i < v.size(); ++i) {
          ++hn;
          const int a = k::bf16_bits(it->second[i]), b = k::bf16_bits(v[i]);
          const int d = std::abs(((a & 0x8000) ? -(a & 0x7fff) : a) - ((b & 0x8000) ? -(b & 0x7fff) : b));
          hoff += d != 0;
          hulp = std::max(hulp, d);
        }
      }
      std::snprintf(what, sizeof what, "layer %d h (int4 gate/up -> silu * up, both cards) vs reference: %zu / %zu differ (%.3f %%), worst %d ulp", L, hoff, hn,
                    100.0 * hoff / std::max<size_t>(1, hn), hulp);
      check(keys && hulp <= 4 && hoff * 200 <= hn, what);
      k::moe_ref(s2, rows, x.data(), rw, rof, &dh);
      const Cmp c = compare(o[0], rof.out, H), cf = compare(o[0], ro.out, H);
      std::snprintf(what, sizeof what,
                    "layer %d out vs reference from the device's h: worst %.2e of max(|ref|, rms), %zu / %zu differ (free-running: %.2e, %zu differ)", L,
                    c.scaled, c.off, c.n, cf.scaled, cf.off);
      check(c.scaled <= 1.0 / 128 && c.off * 100 <= c.n && cf.scaled <= 1.0 / 128, what);
    }
    std::snprintf(what, sizeof what, "layer %d both cards bit-identical output", L);
    check(o[0] == o[1], what);
    // Cold experts: point card 1's table at the pinned host copy (read in place over PCIe).
    {
      for (int i = 0; i < per; ++i) tab[i] = pinned + static_cast<size_t>(i) * lay.stride;
      fab.compute(1).memcpy(table_d[1], tab.data(), per * 8).wait();
      run(1, 0, nullptr);
      for (int c = 0; c < 2; ++c) fab.compute(c).wait_and_throw();
      std::vector<uint16_t> oc(o[0].size());
      fab.compute(0).memcpy(oc.data(), out[0], oc.size() * 2).wait();
      std::snprintf(what, sizeof what, "layer %d card 1's experts cold (pinned host, read over PCIe): output bit-identical to hot (%d cold routes)", L,
                    rt[1].count);
      check(oc == o[0], what);
      for (int i = 0; i < per; ++i) tab[i] = vram[1] + static_cast<size_t>(i) * lay.stride;
      fab.compute(1).memcpy(table_d[1], tab.data(), per * 8).wait();
    }

    // Floor per card: its unique experts + router + its shared half + x / out.
    double bytes[2];
    for (int c = 0; c < 2; ++c) {
      const auto &wc = w[c];
      bytes[c] = rt[c].count * static_cast<double>(lay.bytes) + static_cast<double>(wc.router.bytes()) +
                 (static_cast<double>(wc.shared_gu.bytes()) + static_cast<double>(wc.shared_down.bytes())) / 2;
    }
    std::printf("layer %d routing: %d rows, unique experts card 0 %d, card 1 %d (of %d routes); floor card 0 %.1f us, card 1 %.1f us (%.1f / %.1f MB, %.1f GB/s)\n", L,
                rows, rt[0].count, rt[1].count, rows * cfg.topk, bytes[0] / floor_bw * 1e6, bytes[1] / floor_bw * 1e6, bytes[0] / 1e6, bytes[1] / 1e6,
                floor_bw / 1e9);
    const double floor_us = std::max(bytes[0], bytes[1]) / floor_bw * 1e6;
    std::printf("variant (both cards)      wall med   wall p90 | card0: router  route  gate_up   down  combine   span | card1 span | x floor\n");
    int parity = 0;
    struct V {
      int ks, dks;
    };
    for (const V v : {V{0, 0}, V{4, 1}, V{8, 1}, V{4, 2}, V{8, 4}, V{0, 0}}) {
      const int ks = v.ks, dks = v.dks;
      xenithium::probe::warm_for([&] {
        run(parity++, ks, nullptr, dks);
        for (int c = 0; c < 2; ++c) fab.compute(c).wait();
      });
      std::vector<double> wall, st[5], span[2];
      for (int i = 0; i < iters; ++i) {
        k::MoeEvents ev[2];
        const auto a = std::chrono::steady_clock::now();
        run(parity++, ks, ev, dks);
        for (int c = 0; c < 2; ++c) fab.compute(c).wait();
        wall.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - a).count());
        const sycl::event *e0[5] = {&ev[0].router, &ev[0].route, &ev[0].gate_up, &ev[0].down, &ev[0].combine};
        for (int j = 0; j < 5; ++j) st[j].push_back(ev_us(*e0[j]));
        for (int c = 0; c < 2; ++c) {
          const auto s0 = ev[c].router.get_profiling_info<sycl::info::event_profiling::command_start>();
          const auto s1 = ev[c].combine.get_profiling_info<sycl::info::event_profiling::command_end>();
          span[c].push_back(static_cast<double>(s1 - s0) * 1e-3);
        }
      }
      // The combine includes waiting for the slower card; the span of the faster card shows it.
      char nm[40];
      if (ks == 0) std::snprintf(nm, sizeof nm, "default");
      else std::snprintf(nm, sizeof nm, "gate_up ks %d, down ks %d", ks, dks);
      std::printf("%-24s  %8.1f   %8.1f | %13.2f %6.2f %8.2f %6.2f %8.2f %6.1f | %10.1f | %.2f\n", nm, median(wall), pct(wall, 0.9),
                  median(st[0]), median(st[1]), median(st[2]), median(st[3]), median(st[4]), median(span[0]), median(span[1]),
                  std::min(median(span[0]), median(span[1])) / floor_us);
    }
    // Chained: 48 calls back to back on both cards (a step's worth of MoE layers, no host wait
    // between them), so the cards pace each other through the mailbox as in a real step and the
    // host's submission skew between the cards is paid once, not per call.
    {
      constexpr int kChain = 48;
      std::vector<double> per_call, gspan[2];
      for (int rep = 0; rep < 12; ++rep) {
        std::vector<k::MoeEvents> ev0(kChain), ev1(kChain);
        const auto a = std::chrono::steady_clock::now();
        for (int i = 0; i < kChain; ++i) {
          k::MoeEvents ev[2];
          run(parity++, 0, ev);
          ev0[i] = ev[0];
          ev1[i] = ev[1];
        }
        for (int c = 0; c < 2; ++c) fab.compute(c).wait();
        if (rep < 2) continue;  // warm
        per_call.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - a).count() / kChain);
        for (int c = 0; c < 2; ++c) {
          const auto &evs = c == 0 ? ev0 : ev1;
          // Steady state: from the end of call 8's combine to the end of the last one.
          const auto s0 = evs[7].combine.get_profiling_info<sycl::info::event_profiling::command_end>();
          const auto s1 = evs[kChain - 1].combine.get_profiling_info<sycl::info::event_profiling::command_end>();
          gspan[c].push_back(static_cast<double>(s1 - s0) * 1e-3 / (kChain - 8));
        }
      }
      std::printf("chained x48 (default): host wall %.1f us per call; GPU per call (steady, calls 9-48) card 0 %.1f us, card 1 %.1f us -> x floor %.2f\n",
                  median(per_call), median(gspan[0]), median(gspan[1]), std::max(median(gspan[0]), median(gspan[1])) / floor_us);
    }
    uint32_t err[2];
    for (int c = 0; c < 2; ++c) fab.compute(c).memcpy(&err[c], wire[c].error, 4).wait();
    std::snprintf(what, sizeof what, "layer %d mailbox: no timeout during %d+ steps", L, iters * 6 + 576);
    check(err[0] == 0 && err[1] == 0, what);
    // Every split variant must give the default's routing and stay within the bound.
  }
  for (int c = 0; c < 2; ++c) {
    for (void *p : {(void *)outA[c], (void *)outB[c], (void *)scr[c].logits, (void *)scr[c].route, (void *)scr[c].h, (void *)scr[c].partial,
                    (void *)wire[c].inbox, (void *)wire[c].error, (void *)out[c], (void *)vram[c], (void *)table_d[c]})
      fab.release(p);
    for (auto *p : xin[c]) fab.release(p);
  }
  fab.release(pinned);
  return rc;
}
