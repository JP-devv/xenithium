// Gate for Flint's MoE decode block (kernels/moe/moe_aff.hpp) on synthetic weights in the checkpoint's own layout
// (affine int4 g64 codes, bf16 scales and biases with a non-integer zero point): hidden 512, 64 experts x inter 256,
// top 6, shared expert 512, rows 1 / 3 / 4 / 5 / 8. Checks:
// route the device's experts == the double reference's per row (a near-tie, score gap < 1e-6, may swap), weights
// within 1e-6 relative
// h every routed (row, expert) and the shared h vs bf16(silu(bf16 g) bf16 u) from double sums: <= 1 bf16 ulp,
// <= 1 % of elements off
// out vs the double reference teacher-forced at the device's route and h: within 1 bf16 ulp of max(|ref|, rms)
// bits run-to-run identical; every row of a multi-row step == the same row stepped alone; TP2 (each card's
// partial, no wire) partial_0 + partial_1 == the one-card result within fp32 reassociation (1e-5 of rms)
// The many-row block (moe_aff_rows) at 300 rows: route / h / out on every 7th row as above (teacher-forced at
// the device's route and h); rerun, 37 rows at 100 and 1 row at 5 == the 300-row call bit for bit; vs decode per row
// within 1 bf16 ulp of max(|out|, rms); TP2 partials vs one card.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "xenithium/kernels/gemm/mlp.hpp"
#include "xenithium/kernels/moe/moe_aff.hpp"
#include "xenithium/kernels/pack.hpp"
#include "xenithium/kernels/gemm/w4a16_gemm.hpp"
#include "xenithium/kernels/comm/wire.hpp"
#include "xenithium/rt/device.hpp"

namespace {
namespace k = xenithium::kernels;
using bf16 = sycl::ext::oneapi::bfloat16;
using MK = xenithium::l0::MemKind;

int g_failures = 0;
void expect(bool ok, const std::string &what) {
  std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
  if (!ok) ++g_failures;
}
uint16_t f2bf(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  u += 0x7fffu + ((u >> 16) & 1u);
  return static_cast<uint16_t>(u >> 16);
}
float bf2f(uint16_t b) {
  const uint32_t u = static_cast<uint32_t>(b) << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}
float ulp_bf16(float v) {
  const float a = std::fabs(v);
  return a < 1e-30f ? 1e-30f : std::ldexp(1.f, std::ilogb(a) - 7);
}

struct ExpertData {  // checkpoint-layout tensors of one expert
  std::vector<uint8_t> gw, uw, dw;
  std::vector<uint16_t> gs, gb, us, ub, ds, db;
  int inter = 0;
  k::MoeAffSrc src() const {
    return {gw.data(), uw.data(), dw.data(), gs.data(), gb.data(), us.data(), ub.data(), ds.data(), db.data(), inter};
  }
};

ExpertData make_expert(int H, int I, float sd, std::mt19937 &rng) {
  ExpertData e;
  e.inter = I;
  auto mat = [&](int n, int kk, std::vector<uint8_t> &w, std::vector<uint16_t> &s, std::vector<uint16_t> &b, float target) {
    w.resize(static_cast<size_t>(n) * kk / 2);
    for (auto &x : w) x = static_cast<uint8_t>(rng());
    s.resize(static_cast<size_t>(n) * (kk / 64)), b.resize(s.size());
    std::uniform_real_distribution<float> us(0.6f, 1.4f), uz(2.f, 13.f);
    for (size_t i = 0; i < s.size(); ++i) {
      const float sc = target * us(rng) / 4.6f;  // q in 0..15, centred by the zero point
      s[i] = f2bf(sc), b[i] = f2bf(-bf2f(s[i]) * uz(rng));
    }
  };
  mat(I, H, e.gw, e.gs, e.gb, sd);
  mat(I, H, e.uw, e.us, e.ub, sd);
  mat(H, I, e.dw, e.ds, e.db, sd);
  return e;
}

}  // namespace

int main() {
  try {
    const auto loader = xenithium::l0::Loader::open();
    if (loader.size() == 0) {
      std::printf("no device\n");
      return 1;
    }
    xenithium::Device dev = xenithium::Device::open(loader, 0);
    auto &q = dev.compute();
    std::vector<void *> mem;
    auto alloc = [&](size_t bytes) {
      void *p = dev.alloc<uint8_t>(std::max<size_t>(bytes, 4096), MK::Device);
      mem.push_back(p);
      return p;
    };
    const int H = 2048, I = 512, SI = 1024, E = 64, K = 6;  // Flint's hidden / expert / shared widths (64 of its 256 experts)
    std::mt19937 rng(404);
    std::vector<ExpertData> ex;
    for (int e = 0; e < E; ++e) ex.push_back(make_expert(H, I, 0.05f, rng));
    ExpertData shared = make_expert(H, SI, 0.05f, rng);
    std::vector<k::MoeAffRefExpert> rex;
    for (auto &e : ex) rex.push_back(k::moe_aff_dequant(H, e.src()));
    const k::MoeAffRefExpert rsh = k::moe_aff_dequant(H, shared.src());
    std::vector<uint16_t> router(static_cast<size_t>(E) * H);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (auto &v : router) v = f2bf(nd(rng) * 0.06f);
    std::vector<float> routerf(router.size());
    for (size_t i = 0; i < router.size(); ++i) routerf[i] = bf2f(router[i]);
    auto *silu = static_cast<float *>(alloc(k::kSiluTabSize * 4));
    k::swiglu_table(q, silu).wait_and_throw();

    // one card set-up per (cards, card)
    struct CardSet {
      k::MoeAffShape s;
      k::MoeAffWeights w;
      k::MoeAffScratch sc;
    };
    auto setup = [&](int cards, int card) {
      CardSet cs;
      cs.s = k::MoeAffShape{H, I, E, K, SI, cards, card};
      cs.w.blob = k::moe_aff_layout(H, cs.s.inter_c());
      cs.w.shared_lay = k::moe_aff_layout(H, cs.s.shared_c());
      std::vector<const uint8_t *> table(E);
      std::vector<uint8_t> blob(cs.w.blob.bytes);
      for (int e = 0; e < E; ++e) {
        k::moe_aff_pack(H, cards, card, ex[e].src(), cs.w.blob, blob.data());
        auto *d = static_cast<uint8_t *>(alloc(blob.size()));
        q.memcpy(d, blob.data(), blob.size()).wait();
        table[e] = d;
      }
      std::vector<uint8_t> sb(cs.w.shared_lay.bytes);
      k::moe_aff_pack(H, cards, card, shared.src(), cs.w.shared_lay, sb.data());
      auto *dsb = static_cast<uint8_t *>(alloc(sb.size()));
      q.memcpy(dsb, sb.data(), sb.size()).wait();
      cs.w.shared = dsb;
      auto *dt = static_cast<const uint8_t **>(alloc(E * sizeof(void *)));
      q.memcpy(dt, table.data(), E * sizeof(void *)).wait();
      cs.w.blobs = dt;
      std::vector<uint16_t> rp(k::dense_packed_elems(E, H));
      k::pack_dense_bf16(router.data(), E, H, rp.data());
      auto *dr = static_cast<uint16_t *>(alloc(rp.size() * 2));
      q.memcpy(dr, rp.data(), rp.size() * 2).wait();
      cs.w.router.w = dr, cs.w.router.kind = k::DenseKind::bf16, cs.w.router.n = E, cs.w.router.k = H;
      cs.w.silu_tab = silu;
      cs.sc.logits = static_cast<float *>(alloc(8ull * E * 4));
      cs.sc.route = static_cast<k::MoeRoute *>(alloc(sizeof(k::MoeRoute)));
      cs.sc.h = static_cast<bf16 *>(alloc(k::moe_aff_h_elems(cs.s) * 2));
      cs.sc.y = static_cast<float *>(alloc(k::moe_aff_y_elems(cs.s) * 4));
      cs.sc.partial = static_cast<float *>(alloc(8ull * H * 4));
      return cs;
    };
    CardSet one = setup(1, 0), tp0 = setup(2, 0), tp1 = setup(2, 1);
    auto *x = static_cast<bf16 *>(alloc(8ull * H * 2));
    auto *out = static_cast<bf16 *>(alloc(8ull * H * 2));
    std::vector<uint16_t> xall(8ull * H);
    for (auto &v : xall) v = f2bf(nd(rng));

    auto run = [&](CardSet &cs, int rows, const uint16_t *xrows) {
      std::vector<uint16_t> xs(8ull * H, 0);
      std::memcpy(xs.data(), xrows, static_cast<size_t>(rows) * H * 2);
      q.memcpy(x, xs.data(), xs.size() * 2).wait();
      k::moe_aff_decode(q, cs.s, rows, x, cs.w, cs.sc, nullptr, 0, out);
      q.wait_and_throw();
      std::vector<uint16_t> o(static_cast<size_t>(rows) * H);
      if (cs.s.cards == 1) q.memcpy(o.data(), out, o.size() * 2).wait();
      return o;
    };
    for (const int rows : {1, 3, 4, 5, 8}) {
      const auto o = run(one, rows, xall.data());
      k::MoeRoute rt;
      q.memcpy(&rt, one.sc.route, sizeof rt).wait();
      std::vector<uint16_t> hd(k::moe_aff_h_elems(one.s));
      q.memcpy(hd.data(), one.sc.h, hd.size() * 2).wait();
      const int hmax = one.s.hmax();
      size_t route_bad = 0, h_off = 0, h_bad = 0, h_n = 0;
      double worst = 0;
      for (int r = 0; r < rows; ++r) {
        std::vector<float> xf(H);
        for (int c = 0; c < H; ++c) xf[c] = bf2f(xall[static_cast<size_t>(r) * H + c]);
        int32_t ids[k::kMoeMaxTopk];
        float wts[k::kMoeMaxTopk];
        std::vector<double> lg(E);
        k::moe_aff_route_ref(xf.data(), routerf.data(), H, E, K, ids, wts, lg.data());
        // the device's choice: equal sets, or a swap at a near-tie
        std::vector<int> a(rt.ids[r], rt.ids[r] + K), b(ids, ids + K);
        std::sort(a.begin(), a.end()), std::sort(b.begin(), b.end());
        if (a != b) {
          auto sig = [&](int e) { return 1.0 / (1.0 + std::exp(-lg[e])); };
          double kth = 1e9;
          for (int t = 0; t < K; ++t) kth = std::min(kth, sig(ids[t]));
          for (int e : a)
            if (std::find(b.begin(), b.end(), e) == b.end() && std::fabs(sig(e) - kth) > 1e-6) ++route_bad;
        } else {
          for (int t = 0; t < K; ++t) {
            const int j = static_cast<int>(std::find(ids, ids + K, rt.ids[r][t]) - ids);
            if (std::fabs(rt.topw[r][t] - wts[j]) > 1e-6f * std::fabs(wts[j]) + 1e-7f) ++route_bad;
          }
        }
        // teacher forcing: the device's route; h per (row, expert) vs the reference rule, then out from the device's h
        std::vector<std::vector<float>> hf;
        std::vector<const k::MoeAffRefExpert *> pe;
        std::vector<float> pw;
        for (int t = 0; t < K; ++t) {
          const int e = rt.ids[r][t];
          int slot = -1;
          for (int j = 0; j < rt.count; ++j)
            if (rt.expert[j] == e) slot = j;
          std::vector<float> h(I);
          for (int i = 0; i < I; ++i) h[i] = bf2f(hd[(static_cast<size_t>(slot) * k::kStepRows16 + r) * hmax + i]);
          hf.push_back(h);
          pe.push_back(&rex[e]);
          pw.push_back(rt.topw[r][t]);
        }
        std::vector<float> hs(SI);
        for (int i = 0; i < SI; ++i) hs[i] = bf2f(hd[(static_cast<size_t>(rt.count) * k::kStepRows16 + r) * hmax + i]);
        hf.push_back(hs);
        // h vs the reference rule
        auto check_h = [&](const k::MoeAffRefExpert &e, const std::vector<float> &h) {
          const int II = static_cast<int>(e.gate.size() / H);
          for (int i = 0; i < II; ++i) {
            double g = 0, u = 0;
            for (int c = 0; c < H; ++c) g += static_cast<double>(xf[c]) * e.gate[static_cast<size_t>(i) * H + c], u += static_cast<double>(xf[c]) * e.up[static_cast<size_t>(i) * H + c];
            const float gb = bf2f(f2bf(static_cast<float>(g))), ub = bf2f(f2bf(static_cast<float>(u)));
            const float want = bf2f(f2bf(gb / (1.f + std::exp(-gb)) * ub));
            ++h_n;
            h_off += h[i] != want;
            h_bad += !(std::fabs(h[i] - want) <= ulp_bf16(want));
          }
        };
        for (int t = 0; t < K; ++t) check_h(*pe[t], hf[t]);
        check_h(rsh, hs);
        std::vector<double> ref(H);
        k::moe_aff_ref(xf.data(), H, K, rt.ids[r], pw.data(), pe.data(), rsh, ref.data(), &hf);
        double ss = 0;
        for (double v : ref) ss += v * v;
        const double rms = std::sqrt(ss / H);
        for (int c = 0; c < H; ++c) {
          const double e = std::fabs(bf2f(o[static_cast<size_t>(r) * H + c]) - ref[c]) / std::max(std::fabs(ref[c]), rms);
          worst = std::isfinite(e) ? std::max(worst, e) : HUGE_VAL;
        }
      }
      char b[256];
      std::snprintf(b, sizeof b, "%d row(s): route %zu bad, h %zu / %zu off by 1 ulp (%zu worse), out worst %.2e of max(|ref|, rms) (%d unique experts)", rows,
                    route_bad, h_off, h_n, h_bad, worst, rt.count);
      expect(route_bad == 0 && h_bad * 10000 <= h_n && h_off * 100 <= h_n && worst <= 1.0 / 128, b);  // (h: see the many-row check below)
      // bits: rerun; each row alone; TP2 partials
      bool same = run(one, rows, xall.data()) == o;
      for (int r = 0; r < rows && same; ++r) {
        const auto o1 = run(one, 1, &xall[static_cast<size_t>(r) * H]);
        same &= std::equal(o1.begin(), o1.end(), o.begin() + static_cast<size_t>(r) * H);
      }
      expect(same, std::to_string(rows) + " row(s): rerun and each row stepped alone bit-identical");
      run(tp0, rows, xall.data());
      std::vector<float> p0(static_cast<size_t>(rows) * H), p1(p0.size()), p(p0.size());
      q.memcpy(p0.data(), tp0.sc.partial, p0.size() * 4).wait();
      run(tp1, rows, xall.data());
      q.memcpy(p1.data(), tp1.sc.partial, p1.size() * 4).wait();
      {  // the one-card partial: out null
        std::vector<uint16_t> xs(8ull * H, 0);
        std::memcpy(xs.data(), xall.data(), static_cast<size_t>(rows) * H * 2);
        q.memcpy(x, xs.data(), xs.size() * 2).wait();
        k::moe_aff_decode(q, one.s, rows, x, one.w, one.sc, nullptr, 0, nullptr);
        q.wait_and_throw();
      }
      q.memcpy(p.data(), one.sc.partial, p.size() * 4).wait();
      double tw = 0;
      for (int r = 0; r < rows; ++r) {
        double ss = 0;
        for (int c = 0; c < H; ++c) ss += static_cast<double>(p[static_cast<size_t>(r) * H + c]) * p[static_cast<size_t>(r) * H + c];
        const double rms = std::sqrt(ss / H);
        for (int c = 0; c < H; ++c) {
          const size_t i = static_cast<size_t>(r) * H + c;
          const double e = std::fabs(static_cast<double>(p0[i]) + p1[i] - p[i]) / rms;
          tw = std::isfinite(e) ? std::max(tw, e) : HUGE_VAL;
        }
      }
      char b2[160];
      std::snprintf(b2, sizeof b2, "%d row(s): TP2 partial_0 + partial_1 vs one card: worst %.2e of rms", rows, tw);
      expect(tw <= 1e-5, b2);
    }
    // ---- the ESIMD route vs the SIMT route ----
    {
      for (const int fr : {1})
      for (const int rows : {1, 2, 3, 4, 5, 8}) {
        k::moe_aff_set_route_esimd(0);
        const auto o0 = run(one, rows, xall.data());
        k::MoeRoute r0, r1;
        std::vector<float> l0(static_cast<size_t>(rows) * E), l1(l0.size());
        q.memcpy(&r0, one.sc.route, sizeof r0);
        q.memcpy(l0.data(), one.sc.logits, l0.size() * 4).wait();
        k::moe_aff_set_route_esimd(fr);
        std::vector<uint16_t> o1;
        for (int rep = 0; rep < 3; ++rep) o1 = run(one, rows, xall.data());
        q.memcpy(&r1, one.sc.route, sizeof r1);
        q.memcpy(l1.data(), one.sc.logits, l1.size() * 4).wait();
        bool same_list = r0.count == r1.count;
        for (int u = 0; same_list && u < r0.count; ++u) {
          same_list &= r0.expert[u] == r1.expert[u] && r0.mask[u] == r1.mask[u] && r0.blob[u] == r1.blob[u];
          for (int rr = 0; rr < k::kStepRows16; ++rr) same_list &= std::fabs(r0.weight[rr][u] - r1.weight[rr][u]) <= 2e-7f * std::fabs(r0.weight[rr][u]);
        }
        double wmax = 0;
        for (int r = 0; r < rows; ++r)
          for (int t = 0; t < K; ++t) {
            same_list &= r0.ids[r][t] == r1.ids[r][t];
            wmax = std::max(wmax, static_cast<double>(std::fabs(r0.topw[r][t] - r1.topw[r][t]) / r0.topw[r][t]));
          }
        size_t od = 0, obad = 0;
        for (size_t i = 0; i < o0.size(); ++i) od += o0[i] != o1[i], obad += !(std::fabs(bf2f(o0[i]) - bf2f(o1[i])) <= ulp_bf16(bf2f(o0[i])));
        char b[256];
        std::snprintf(b, sizeof b, "%s, %d row(s): logits bit-equal %s, ids / list equal %s, topw worst rel %.1e, out %zu of %zu differ (%zu by > 1 ulp)", "ESIMD route vs SIMT", rows,
                      l0 == l1 ? "yes" : "no", same_list ? "yes" : "no", wmax, od, o0.size(), obad);
        expect(l0 == l1 && same_list && wmax <= 2e-7 && obad == 0, b);
      }
    }
    // ---- the many-row block (moe_aff_rows) ----
    {
      const int RM = 300;
      k::w4a16_pf_aff_reserve(q, RM, std::max(H, SI));
      auto rows_scratch = [&](const k::MoeAffShape &sh) {
        const auto z = k::moe_aff_rows_sizes(sh, RM);
        k::MoeAffRowsScratch sc;
        sc.rows_max = RM;
        sc.logits = static_cast<float *>(alloc(z.logits * 4));
        sc.ids = static_cast<int32_t *>(alloc(z.ids * 4));
        sc.topw = static_cast<float *>(alloc(z.ids * 4));
        sc.slot = static_cast<int32_t *>(alloc(z.ids * 4));
        sc.row = static_cast<int32_t *>(alloc(z.row * 4));
        sc.seg = static_cast<int32_t *>(alloc(z.seg * 4));
        sc.seg_n = static_cast<int32_t *>(alloc(64));
        sc.ax = static_cast<float *>(alloc(z.ax * 4));
        sc.ah = static_cast<float *>(alloc(z.ah * 4));
        sc.lds_x = (RM + 31) / 32 * 32, sc.lds_h = (RM * K + 31) / 32 * 32;
        sc.h = static_cast<bf16 *>(alloc(z.h * 2));
        sc.y = static_cast<float *>(alloc(z.y * 4));
        sc.sh = static_cast<bf16 *>(alloc(z.sh * 2));
        sc.sy = static_cast<float *>(alloc(z.sy * 4));
        sc.part = static_cast<float *>(alloc(z.part * 4));
        return sc;
      };
      const k::MoeAffRowsScratch r1 = rows_scratch(one.s), r0 = rows_scratch(tp0.s), r2 = rows_scratch(tp1.s);
      std::vector<uint16_t> xr(static_cast<size_t>(RM) * H);
      for (auto &v : xr) v = f2bf(nd(rng));
      auto *xd = static_cast<bf16 *>(alloc(xr.size() * 2));
      auto *od = static_cast<bf16 *>(alloc(xr.size() * 2));
      q.memcpy(xd, xr.data(), xr.size() * 2).wait();
      auto run_rows = [&](const CardSet &cs, const k::MoeAffRowsScratch &sc, int m, int r0w) {  // rows [r0w, r0w + m) of xr
        k::PairRowsWire pw;
        pw.split = true;  // TP2: no exchange, the partial stays in sc.part
        k::moe_aff_rows(q, cs.s, m, xd + static_cast<size_t>(r0w) * H, cs.w, sc, cs.s.cards > 1 ? &pw : nullptr, 0, od);
        q.wait_and_throw();
        std::vector<uint16_t> o(static_cast<size_t>(m) * H);
        if (cs.s.cards == 1) q.memcpy(o.data(), od, o.size() * 2).wait();
        return o;
      };
      const auto o = run_rows(one, r1, RM, 0);
      std::vector<int32_t> ids(static_cast<size_t>(RM) * K), slot(ids.size());
      std::vector<float> tw(ids.size());
      std::vector<uint16_t> hd(static_cast<size_t>(RM) * K * I), shd(static_cast<size_t>(RM) * SI);
      q.memcpy(ids.data(), r1.ids, ids.size() * 4);
      q.memcpy(slot.data(), r1.slot, slot.size() * 4);
      q.memcpy(tw.data(), r1.topw, tw.size() * 4);
      q.memcpy(hd.data(), r1.h, hd.size() * 2);
      q.memcpy(shd.data(), r1.sh, shd.size() * 2).wait();
      size_t route_bad = 0, h_off = 0, h_bad = 0, h_n = 0, h_bad_sh = 0;
      double worst = 0;
      for (int r = 0; r < RM; r += 7) {  // every 7th row against the double reference
        std::vector<float> xf(H);
        for (int c = 0; c < H; ++c) xf[c] = bf2f(xr[static_cast<size_t>(r) * H + c]);
        int32_t rid[k::kMoeMaxTopk];
        float rwt[k::kMoeMaxTopk];
        std::vector<double> lg(E);
        k::moe_aff_route_ref(xf.data(), routerf.data(), H, E, K, rid, rwt, lg.data());
        std::vector<int> a(ids.begin() + static_cast<size_t>(r) * K, ids.begin() + static_cast<size_t>(r + 1) * K), b(rid, rid + K);
        std::sort(a.begin(), a.end()), std::sort(b.begin(), b.end());
        if (a != b) {
          auto sig = [&](int e) { return 1.0 / (1.0 + std::exp(-lg[e])); };
          double kth = 1e9;
          for (int t = 0; t < K; ++t) kth = std::min(kth, sig(rid[t]));
          for (int e : a)
            if (std::find(b.begin(), b.end(), e) == b.end() && std::fabs(sig(e) - kth) > 1e-6) ++route_bad;
        } else {
          for (int t = 0; t < K; ++t) {
            const int j = static_cast<int>(std::find(rid, rid + K, ids[static_cast<size_t>(r) * K + t]) - rid);
            if (std::fabs(tw[static_cast<size_t>(r) * K + t] - rwt[j]) > 1e-6f * std::fabs(rwt[j]) + 1e-7f) ++route_bad;
          }
        }
        std::vector<std::vector<float>> hf;
        std::vector<const k::MoeAffRefExpert *> pe;
        std::vector<float> pw;
        for (int t = 0; t < K; ++t) {
          const size_t ai = static_cast<size_t>(r) * K + t;
          std::vector<float> h(I);
          for (int i = 0; i < I; ++i) h[i] = bf2f(hd[static_cast<size_t>(slot[ai]) * I + i]);
          hf.push_back(h);
          pe.push_back(&rex[ids[ai]]);
          pw.push_back(tw[ai]);
        }
        std::vector<float> hs(SI);
        for (int i = 0; i < SI; ++i) hs[i] = bf2f(shd[static_cast<size_t>(r) * SI + i]);
        hf.push_back(hs);
        auto check_h = [&](const k::MoeAffRefExpert &e, const std::vector<float> &h) {
          const int II = static_cast<int>(e.gate.size() / H);
          for (int i = 0; i < II; ++i) {
            double g = 0, u = 0;
            for (int c = 0; c < H; ++c) g += static_cast<double>(xf[c]) * e.gate[static_cast<size_t>(i) * H + c], u += static_cast<double>(xf[c]) * e.up[static_cast<size_t>(i) * H + c];
            const float gb = bf2f(f2bf(static_cast<float>(g))), ub = bf2f(f2bf(static_cast<float>(u)));
            const float want = bf2f(f2bf(gb / (1.f + std::exp(-gb)) * ub));
            ++h_n;
            h_off += h[i] != want;
            h_bad += !(std::fabs(h[i] - want) <= ulp_bf16(want));
          }
        };
        for (int t = 0; t < K; ++t) check_h(*pe[t], hf[t]);
        const size_t hb0 = h_bad;
        check_h(rsh, hs);
        h_bad_sh += h_bad - hb0;
        std::vector<double> ref(H);
        std::vector<int32_t> idr(ids.begin() + static_cast<size_t>(r) * K, ids.begin() + static_cast<size_t>(r + 1) * K);
        k::moe_aff_ref(xf.data(), H, K, idr.data(), pw.data(), pe.data(), rsh, ref.data(), &hf);
        double ss = 0;
        for (double v : ref) ss += v * v;
        const double rms = std::sqrt(ss / H);
        for (int c = 0; c < H; ++c) {
          const double e = std::fabs(bf2f(o[static_cast<size_t>(r) * H + c]) - ref[c]) / std::max(std::fabs(ref[c]), rms);
          worst = std::isfinite(e) ? std::max(worst, e) : HUGE_VAL;
        }
      }
      char b[256];
      std::snprintf(b, sizeof b, "moe_aff_rows %d rows (every 7th checked): route %zu bad, h %zu / %zu off by 1 ulp (%zu worse, %zu of them shared), out worst %.2e of max(|ref|, rms)", RM,
                    route_bad, h_off, h_n, h_bad, h_bad_sh, worst);
      // h: a g or u fp32 sum on a bf16 rounding edge moves h by more than an ulp now and then (the group order differs from
      // the double sum's): <= 1 in 10,000 elements
      expect(route_bad == 0 && h_bad * 10000 <= h_n && h_off * 100 <= h_n && worst <= 1.0 / 128, b);
      // bits: rerun; rows 100..136 alone and row 5 alone == the same rows of the 300-row call
      bool same = run_rows(one, r1, RM, 0) == o;
      const auto o37 = run_rows(one, r1, 37, 100);
      same &= std::equal(o37.begin(), o37.end(), o.begin() + static_cast<size_t>(100) * H);
      const auto o1 = run_rows(one, r1, 1, 5);
      same &= std::equal(o1.begin(), o1.end(), o.begin() + static_cast<size_t>(5) * H);
      expect(same, "moe_aff_rows: rerun, 37 rows at 100 and 1 row at 5 == the 300-row call bit for bit");
      // vs decode per row: the same routes, out within 1 bf16 ulp of max(|out|, rms) (fp32 order only)
      double dw = 0;
      size_t dn = 0;
      for (int r = 0; r < RM; r += 50) {
        const auto od1 = run(one, 1, &xr[static_cast<size_t>(r) * H]);
        for (int c = 0; c < H; ++c) dn += od1[c] != o[static_cast<size_t>(r) * H + c];
        double ss = 0;
        for (int c = 0; c < H; ++c) ss += static_cast<double>(bf2f(od1[c])) * bf2f(od1[c]);
        const double rms = std::sqrt(ss / H);
        for (int c = 0; c < H; ++c) {
          const double e = std::fabs(bf2f(od1[c]) - bf2f(o[static_cast<size_t>(r) * H + c])) / std::max<double>(std::fabs(bf2f(od1[c])), rms);
          dw = std::isfinite(e) ? std::max(dw, e) : HUGE_VAL;
        }
      }
      std::snprintf(b, sizeof b, "moe_aff_rows vs moe_aff_decode per row (6 rows): worst %.2e of max(|decode|, rms) <= 1 / 128 (%zu of %d differ)", dw, dn, 6 * H);
      expect(dw <= 1.0 / 128, b);
      // TP2: partial_0 + partial_1 vs the one-card partial (of the 300-row call again: the calls above overwrote it)
      run_rows(one, r1, RM, 0);
      run_rows(tp0, r0, RM, 0);
      run_rows(tp1, r2, RM, 0);
      std::vector<float> p0(static_cast<size_t>(RM) * H), p1(p0.size()), p(p0.size());
      q.memcpy(p0.data(), r0.part, p0.size() * 4);
      q.memcpy(p1.data(), r2.part, p1.size() * 4);
      q.memcpy(p.data(), r1.part, p.size() * 4).wait();
      double tpw = 0;
      for (int r = 0; r < RM; ++r) {
        double ss = 0;
        for (int c = 0; c < H; ++c) ss += static_cast<double>(p[static_cast<size_t>(r) * H + c]) * p[static_cast<size_t>(r) * H + c];
        const double rms = std::sqrt(ss / H);
        for (int c = 0; c < H; ++c) {
          const size_t i = static_cast<size_t>(r) * H + c;
          const double e = std::fabs(static_cast<double>(p0[i]) + p1[i] - p[i]) / rms;
          tpw = std::isfinite(e) ? std::max(tpw, e) : HUGE_VAL;
        }
      }
      {  // diagnostic split: the shared y and the routed sum apart
        std::vector<float> s0(p.size()), s1(p.size()), s2(p.size());
        q.memcpy(s0.data(), r0.sy, s0.size() * 4);
        q.memcpy(s1.data(), r2.sy, s1.size() * 4);
        q.memcpy(s2.data(), r1.sy, s2.size() * 4).wait();
        double ws = 0, wr = 0, rs = 0;
        for (size_t i = 0; i < p.size(); ++i) {
          ws = std::max(ws, std::fabs(static_cast<double>(s0[i]) + s1[i] - s2[i]));
          wr = std::max(wr, std::fabs(static_cast<double>(p0[i] - s0[i]) + (p1[i] - s1[i]) - (p[i] - s2[i])));
          rs = std::max(rs, std::fabs(static_cast<double>(s2[i])));
        }
        std::printf("  TP2 diagnostic: shared y worst abs %.3g, routed sum worst abs %.3g (max |shared y| %.3g)\n", ws, wr, rs);
      }
      std::snprintf(b, sizeof b, "moe_aff_rows TP2 partial_0 + partial_1 vs one card: worst %.2e of rms", tpw);
      expect(tpw <= 1e-5, b);
    }
    for (void *p : mem) dev.release(p);
  } catch (const std::exception &e) {
    std::printf("[FAIL] exception: %s\n", e.what());
    return 1;
  }
  std::printf("%s: %d failure(s)\n", g_failures ? "FAILED" : "OK", g_failures);
  return g_failures ? 1 : 0;
}
