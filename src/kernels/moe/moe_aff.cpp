// Flint's MoE decode block (moe_aff.hpp).
#include "xenithium/kernels/moe/moe_aff.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "xenithium/kernels/gemm/dense.hpp"
#include "xenithium/kernels/gemm/mlp.hpp"
#include "xenithium/kernels/pack.hpp"
#include "xenithium/kernels/gemm/pf_dense.hpp"
#include "xenithium/kernels/gemm/w4a16_gemm.hpp"
#include "xenithium/kernels/rows.hpp"
#include "xenithium/kernels/tile/w4_tile.hpp"
#include "comm/mailbox_esimd.hpp"

namespace xenithium::kernels {

// this MoE keeps the 16-row layout (MoeRoute is sized for qwen38's decode: kMoeRows rows x kMoeMaxTopk slots)
constexpr int kSlots16 = kStepRows16 * kMoeMaxTopk;
namespace {

namespace esimd = sycl::ext::intel::esimd;
using bf16 = sycl::ext::oneapi::bfloat16;

constexpr size_t align1k(size_t v) { return (v + 1023) / 1024 * 1024; }

// the row of a one-row mask, else -1 (scalar: sycl:popcount / ctz are not ESIMD functions)
inline int one_row(uint32_t m) {
  if (m == 0 || (m & (m - 1)) != 0) return -1;
  int r = 0;
#pragma unroll
  for (int i = 1; i < 8; ++i) r = (m >> i) & 1u ? i : r;
  return r;
}

// One 128-k block of an affine matrix (codes [k/128][n/16][1024], bf16 scales / biases [k/64][n]) into acc, for kRows
// rows of a (bf16, row stride lda) at k block g: w4a16_gemm's affine mode (two 64-k groups, s sum(q a) + b sum(a)).
template <int kRows>
inline void aff_block(esimd::simd<float, kRows * kW4Width> &acc, const uint8_t *w, const uint16_t *sc, const uint16_t *bs, int n, int tile, int g,
                      const bf16 *a, int lda) SYCL_ESIMD_FUNCTION {
  const int ntiles = n / kW4Width;
  const auto raw = w4::load_block(w + (static_cast<size_t>(g) * ntiles + tile) * kW4BlockBytes);
  esimd::simd<bf16, kRows * kW4Group> ag;
#pragma unroll
  for (int row = 0; row < kRows; ++row)
    ag.template select<kW4Group, 1>(row * kW4Group) = esimd::block_load<bf16, kW4Group>(a + static_cast<size_t>(row) * lda + g * kW4Group);
  esimd::simd<uint32_t, 2 * kW4Width> su, bu;
#pragma unroll
  for (int s = 0; s < 2; ++s) {
    su.template select<kW4Width, 1>(s * kW4Width) = esimd::block_load<uint16_t, kW4Width>(sc + static_cast<size_t>(2 * g + s) * n + tile * kW4Width);
    bu.template select<kW4Width, 1>(s * kW4Width) = esimd::block_load<uint16_t, kW4Width>(bs + static_cast<size_t>(2 * g + s) * n + tile * kW4Width);
  }
  esimd::simd<uint32_t, 2 * kW4Width> sw = su << 16, bw = bu << 16;
  w4::block_dot_aff<kRows>(acc, raw, ag, sw.template bit_cast_view<float>().read(), bw.template bit_cast_view<float>().read());
}

template <int kM, int PER>
class MoeAffRouteK;
template <int kRows, int KS>
class MoeAffGuK;
template <int kRows, int KS>
class MoeAffDnK;

// One work-group, a 16-lane sub-group per row (the qwen38 route's shape, moe.cpp): lane l keeps the keys of experts [l
// PER, l PER + PER) in registers as the fp32 logit's order-preserving bits (sigmoid is monotone: the highest logit =
// the highest score). k rounds of a sub-group max, then the lowest id at that key (32-bit: the 64-bit key's max is
// emulated); lane j keeps round j's expert and its sigmoid score, the weights are the scores over their sum (a
// sub-group sum, the same order on every row). The list: a presence table in SLM (bit r of mask[e], wt[e][r]), then an
// exclusive scan of the items' counts (ascending expert id).
template <int kM, int PER>
sycl::event route(sycl::queue &q, const MoeAffShape &s, int rows, const float *logits, const uint8_t *const *blobs, MoeRoute *rt) {
  const int E = s.experts, K = s.topk;
  constexpr int kItems = kM * 16;
  return q.submit([&](sycl::handler &h) {
    sycl::local_accessor<uint32_t, 1> mask(sycl::range<1>(static_cast<size_t>(PER) * 16), h);
    sycl::local_accessor<float, 1> wt(sycl::range<1>(static_cast<size_t>(PER) * 16 * kM), h);
    h.parallel_for<MoeAffRouteK<kM, PER>>(sycl::nd_range<1>(kItems, kItems), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
      const auto sg = it.get_sub_group();
      const int r = static_cast<int>(sg.get_group_linear_id()), lane = static_cast<int>(sg.get_local_linear_id());
      const int i = static_cast<int>(it.get_local_linear_id());
      const int P = (E + kItems - 1) / kItems;
      for (int x = 0; x < P; ++x)
        if (i * P + x < E) mask[i * P + x] = 0u;
      // keys: the fp32 logit's order-preserving bits (0 = absent); a round's winner = the max key, then the lowest
      // expert id holding it (exact fp32 ties -> the lowest id): two 32-bit sub-group reductions
      uint32_t key[PER];
      const float *lr = logits + static_cast<size_t>(r) * E + lane * PER;
#pragma unroll
      for (int x = 0; x < PER; ++x) {
        const int e = lane * PER + x;
        const uint32_t b = e < E && r < rows ? sycl::bit_cast<uint32_t>(lr[x]) : 0u;
        const uint32_t o = (b & 0x80000000u) ? ~b : (b | 0x80000000u);
        key[x] = e < E && r < rows ? o : 0u;
      }
      int my_e = 0;
      float my_s = 0.f;
      for (int j = 0; j < K; ++j) {
        uint32_t m = 0;
#pragma unroll
        for (int x = 0; x < PER; ++x) m = key[x] > m ? key[x] : m;
        m = sycl::reduce_over_group(sg, m, sycl::maximum<uint32_t>());
        int lo = 0x7fffffff;
#pragma unroll
        for (int x = PER - 1; x >= 0; --x) lo = key[x] == m ? lane * PER + x : lo;
        const int be = sycl::reduce_over_group(sg, lo, sycl::minimum<int>());
#pragma unroll
        for (int x = 0; x < PER; ++x) key[x] = lane * PER + x == be ? 0u : key[x];
        if (lane == j) {
          const float lg = sycl::bit_cast<float>((m & 0x80000000u) ? (m & 0x7fffffffu) : ~m);
          my_e = be;
          my_s = 1.f / (1.f + sycl::exp(-lg));
        }
      }
      const float z = sycl::reduce_over_group(sg, lane < K ? my_s : 0.f, sycl::plus<float>());
      const float wv = my_s / z;
      if (lane < K && r < rows) rt->ids[r][lane] = my_e, rt->topw[r][lane] = wv;
      sycl::group_barrier(it.get_group());  // mask zeroed
      if (lane < K && r < rows) {
        sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::work_group, sycl::access::address_space::local_space>(mask[my_e])
            .fetch_or(1u << r);
        wt[my_e * kM + r] = wv;
      }
      sycl::group_barrier(it.get_group());
      int cnt = 0;
      for (int x = 0; x < P; ++x) {
        const int e = i * P + x;
        if (e < E && mask[e] != 0u) ++cnt;
      }
      int u = sycl::exclusive_scan_over_group(it.get_group(), cnt, sycl::plus<int>());
      if (i == kItems - 1) rt->count = u + cnt;
      for (int x = 0; x < P; ++x) {
        const int e = i * P + x;
        if (e >= E) break;
        const uint32_t mk = mask[e];
        if (!mk) continue;
        rt->expert[u] = e;
        rt->blob[u] = blobs[e];
        rt->mask[u] = mk;
#pragma unroll
        for (int rr = 0; rr < kStepRows16; ++rr) rt->weight[rr][u] = rr < kM && ((mk >> rr) & 1u) ? wt[e * kM + rr] : 0.f;
        ++u;
      }
    });
  });
}

template <int kM>
sycl::event route_m(sycl::queue &q, const MoeAffShape &s, int rows, const float *logits, const uint8_t *const *blobs, MoeRoute *rt) {
  if (s.experts <= 16 * 16) return route<kM, 16>(q, s, rows, logits, blobs, rt);
  if (s.experts <= 32 * 16) return route<kM, 32>(q, s, rows, logits, blobs, rt);
  return route<kM, 64>(q, s, rows, logits, blobs, rt);
}

// The route as ESIMD, one work-group of KS items over the logits: item r < rows takes row r (keys as route's
// order-preserving bits in registers, K rounds of hmax then the lowest id at it (hmin); sigmoid by sycl:exp, the
// scores' sum in the SIMT sub-group reduction's xor-butterfly order), its picks into an SLM presence table (bit-or
// atomics) and weight table; then item p lists the 16-expert chunks p, p + KS... (counts scanned through SLM:
// ascending id) with scattered stores. The same list and weights as route bit for bit (test-moe-aff).
template <int KS, int PER>
class MoeAffRouteEK;
template <int KS, int PER>
sycl::event route_e(sycl::queue &q, const MoeAffShape &s, int rows, const MoeAffScratch &sc, const uint8_t *const *blobs) {
  constexpr int kEM = PER * 16;
  const int n = s.experts, E = s.experts, K = s.topk;
  const float *y = sc.logits;
  MoeRoute *rt = sc.route;
  constexpr int kMkOff = 0, kWtOff = kMkOff + kEM * 4, kCntOff = kWtOff + kEM * kStepRows16 * 4, kSlmAll = kCntOff + PER * 4;
  return q.parallel_for<MoeAffRouteEK<KS, PER>>(sycl::nd_range<1>(KS, KS), [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
    esimd::slm_init<kSlmAll>();
    constexpr auto ld_uc = esimd::properties{esimd::cache_hint_L1<esimd::cache_hint::uncached>, esimd::cache_hint_L2<esimd::cache_hint::cached>,
                                             esimd::alignment<16>};
    const int part = static_cast<int>(it.get_local_id(0));
    // the presence table: mask[e] (bit r), wt[e][r]; zeroed by chunks
    for (int c = part; c < PER; c += KS) esimd::slm_block_store<uint32_t, 16>(static_cast<uint32_t>(kMkOff + c * 64), esimd::simd<uint32_t, 16>(0u));
    esimd::barrier();
    for (int r = part; r < rows; r += KS) {
      esimd::simd<uint32_t, kEM> key;
      const esimd::simd<uint32_t, kEM> idx(0, 1);
#pragma unroll
      for (int c = 0; c < PER; ++c) {
        esimd::simd<uint32_t, 16> b = esimd::block_load<uint32_t, 16>(reinterpret_cast<const uint32_t *>(y + static_cast<size_t>(r) * n) + c * 16, ld_uc);
        esimd::simd<uint32_t, 16> o = b | 0x80000000u;
        o.merge(~b, (b & 0x80000000u) != 0u);
        key.template select<16, 1>(c * 16) = o;
      }
      key.merge(esimd::simd<uint32_t, kEM>(0u), idx >= static_cast<uint32_t>(E));
      esimd::simd<uint32_t, 16> ids(0u);
      esimd::simd<float, 16> sc16(0.f);
      for (int j = 0; j < K; ++j) {
        const uint32_t m = esimd::hmax<uint32_t>(key);
        esimd::simd<uint32_t, kEM> cand(0xffffffffu);
        cand.merge(idx, key == m);
        const uint32_t be = esimd::hmin<uint32_t>(cand);
        key.merge(esimd::simd<uint32_t, kEM>(0u), idx == be);
        const uint32_t lb = (m & 0x80000000u) ? (m & 0x7fffffffu) : ~m;
        const float lg = sycl::bit_cast<float>(lb);
        const float sv = 1.f / (1.f + sycl::exp(-lg));
        ids[j] = be;
        sc16[j] = sv;
      }
      // the scores' sum in the SIMT sub-group reduction's order (route: reduce_over_group over 16 lanes, zeros past K):
      // an xor butterfly, x[i] + x[i + 8] first
      esimd::simd<float, 8> z8 = sc16.template select<8, 1>(0).read() + sc16.template select<8, 1>(8).read();
      esimd::simd<float, 4> z4 = z8.template select<4, 1>(0).read() + z8.template select<4, 1>(4).read();
      esimd::simd<float, 2> z2 = z4.template select<2, 1>(0).read() + z4.template select<2, 1>(2).read();
      const float z = z2[0] + z2[1];
      const esimd::simd<float, 16> wv = sc16 / z;
      const esimd::simd_mask<16> live = esimd::simd<uint32_t, 16>(0, 1) < static_cast<uint32_t>(K);
      esimd::scatter<int32_t, 16>(&rt->ids[r][0], esimd::simd<uint32_t, 16>(0, 4), esimd::simd<int32_t, 16>(ids), live);
      esimd::scatter<float, 16>(&rt->topw[r][0], esimd::simd<uint32_t, 16>(0, 4), wv, live);
      esimd::slm_atomic_update<esimd::atomic_op::bit_or, uint32_t, 16>(kMkOff + ids * 4u, esimd::simd<uint32_t, 16>(1u << r), live);
      esimd::slm_scatter<float, 16>(kWtOff + (ids * kStepRows16 + static_cast<uint32_t>(r)) * 4u, wv, live);
    }
    esimd::barrier();
    // the list: item p takes the 16-expert chunks p, p + KS...: counts scanned in SLM (ascending id), scattered stores
    const esimd::simd<uint32_t, 16> lane(0, 1);
    for (int c = part; c < PER; c += KS) {
      const esimd::simd<uint32_t, 16> m16 = esimd::slm_block_load<uint32_t, 16>(static_cast<uint32_t>(kMkOff + c * 64));
      esimd::slm_scalar_store<uint32_t>(static_cast<uint32_t>(kCntOff + c * 4), esimd::cbit(esimd::pack_mask(m16 != 0u)));
    }
    esimd::barrier();
    const esimd::simd<uint32_t, PER> cnts = esimd::slm_block_load<uint32_t, PER>(static_cast<uint32_t>(kCntOff));
    for (int c = part; c < PER; c += KS) {
      const esimd::simd<uint32_t, 16> ev = lane + static_cast<uint32_t>(c * 16);
      const esimd::simd<uint32_t, 16> m16 = esimd::slm_block_load<uint32_t, 16>(static_cast<uint32_t>(kMkOff + c * 64));
      const esimd::simd_mask<16> present = m16 != 0u;
      if (!present.any()) continue;
      uint32_t u0 = 0;
      for (int cc = 0; cc < c; ++cc) u0 += cnts[cc];
      esimd::simd<uint32_t, 16> rank(0u);
      uint32_t acc = 0;
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        rank[i] = acc;
        acc += present[i] ? 1u : 0u;
      }
      const esimd::simd<uint32_t, 16> slot = rank + u0;
      esimd::scatter<int32_t, 16>(rt->expert, slot * 4u, esimd::simd<int32_t, 16>(ev), present);
      esimd::scatter<uint32_t, 16>(rt->mask, slot * 4u, m16, present);
      const esimd::simd<uint64_t, 16> bl = esimd::gather<uint64_t, 16>(reinterpret_cast<const uint64_t *>(blobs), ev * 8u);
      esimd::scatter<uint64_t, 16>(reinterpret_cast<uint64_t *>(rt->blob), slot * 8u, bl, present);
#pragma unroll
      for (int rr = 0; rr < kStepRows16; ++rr) {
        esimd::simd<float, 16> wt = esimd::slm_gather<float, 16>(kWtOff + (ev * kStepRows16 + rr) * 4u);
        wt.merge(esimd::simd<float, 16>(0.f), ((m16 >> rr) & 1u) == 0u);
        esimd::scatter<float, 16>(&rt->weight[rr][0], slot * 4u, wt, present);
      }
    }
    if (part == 0) rt->count = static_cast<int32_t>(esimd::reduce<uint32_t>(cnts, std::plus<>()));
  });
}

sycl::event route_e_any(sycl::queue &q, const MoeAffShape &s, int rows, const MoeAffScratch &sc, const uint8_t *const *blobs) {
  if (s.experts <= 16 * 16) return route_e<16, 16>(q, s, rows, sc, blobs);
  if (s.experts <= 32 * 16) return route_e<16, 32>(q, s, rows, sc, blobs);
  return route_e<16, 64>(q, s, rows, sc, blobs);
}

// gate | up of every slot (the routed experts, then the shared expert at slot `count`): work-group = 2 KS items per
// (slot, h tile): [0, KS) gate tile ht's k parts, [KS, 2 KS) up tile ht's; h = swiglu in the epilogue (live rows).
template <int kRows, int KS>
sycl::event gate_up(sycl::queue &q, const MoeAffShape &s, int live, int slots, const bf16 *x, const MoeAffWeights &w, const MoeAffScratch &sc) {
  const int H = s.hidden, groups = H / kW4Group, hmax = s.hmax();
  const int ht_r = s.inter_c() / kW4Width, ht_s = s.shared_c() / kW4Width, ht_max = std::max(ht_r, ht_s);
  constexpr int kAcc = kRows * kW4Width, kItems = 2 * KS;
  const MoeRoute *rt = sc.route;
  const uint8_t *shared = w.shared;
  const MoeAffLayout L = w.blob, LS = w.shared_lay;
  const float *tab = w.silu_tab;
  bf16 *hb = sc.h;
  return q.submit([&](sycl::handler &hd) {
    hd.parallel_for<MoeAffGuK<kRows, KS>>(
        sycl::nd_range<1>(static_cast<size_t>(slots) * ht_max * kItems, kItems), [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
          esimd::slm_init<kItems * kAcc * sizeof(float)>();
          const int g = static_cast<int>(it.get_group(0)), lid = static_cast<int>(it.get_local_id(0));
          const int slot = g / ht_max, ht = g % ht_max;
          const int count = rt->count;
          if (slot > count) return;
          const bool sh = slot == count;
          if (ht >= (sh ? ht_s : ht_r)) return;
          const uint8_t *blob = sh ? shared : rt->blob[slot];
          const MoeAffLayout &lay = sh ? LS : L;
          const int ic = lay.ic, n = 2 * ic;
          const int half = lid / KS, part = lid % KS;
          const int tile = ht + half * (ic / kW4Width);
          const int g0 = part * groups / KS, g1 = (part + 1) * groups / KS;
          const uint8_t *cw = blob + lay.gu_w;
          const auto *cs = reinterpret_cast<const uint16_t *>(blob + lay.gu_s);
          const auto *cb = reinterpret_cast<const uint16_t *>(blob + lay.gu_b);
          const uint32_t rmask = sh ? 0xffu : rt->mask[slot];
          const int one = one_row(rmask);  // a slot of one row: that row only
          if (one >= 0) {
            esimd::simd<float, kW4Width> acc(0.f);
            for (int gg = g0; gg < g1; ++gg) aff_block<1>(acc, cw, cs, cb, n, tile, gg, x + static_cast<size_t>(one) * H, H);
            esimd::slm_block_store<float, kW4Width>(static_cast<uint32_t>((lid * kAcc + one * kW4Width) * sizeof(float)), acc);
          } else {
            esimd::simd<float, kAcc> acc(0.f);
            for (int gg = g0; gg < g1; ++gg) aff_block<kRows>(acc, cw, cs, cb, n, tile, gg, x, H);
            esimd::slm_block_store<float, kAcc>(static_cast<uint32_t>(lid * kAcc * sizeof(float)), acc);
          }
          esimd::barrier();
          for (int row = lid; row < live; row += kItems) {
            if (!((rmask >> row) & 1u)) continue;
            esimd::simd<float, kW4Width> gs(0.f), us(0.f);
#pragma unroll
            for (int p = 0; p < KS; ++p) {
              gs += esimd::slm_block_load<float, kW4Width>(static_cast<uint32_t>((p * kAcc + row * kW4Width) * sizeof(float)));
              us += esimd::slm_block_load<float, kW4Width>(static_cast<uint32_t>(((KS + p) * kAcc + row * kW4Width) * sizeof(float)));
            }
            esimd::simd<bf16, kW4Width> gb = gs;
            const esimd::simd<float, kW4Width> uv = esimd::simd<bf16, kW4Width>(us);
            const esimd::simd<uint32_t, kW4Width> off = esimd::simd<uint32_t, kW4Width>(gb.template bit_cast_view<uint16_t>().read()) * 4u;
            const esimd::simd<float, kW4Width> sg = esimd::gather<float, kW4Width>(tab, off);
            const esimd::simd<bf16, kW4Width> hv = sg * uv;
            hv.copy_to(hb + (static_cast<size_t>(slot) * kStepRows16 + row) * hmax + static_cast<size_t>(ht) * kW4Width);
          }
        });
  });
}

// down of every slot: work-group = KS items (k parts) per (slot, 16-column output tile); y fp32 [slot][row][hidden]
template <int kRows, int KS>
sycl::event down(sycl::queue &q, const MoeAffShape &s, int live, int slots, const MoeAffWeights &w, const MoeAffScratch &sc) {
  const int H = s.hidden, ot = H / kW4Width, hmax = s.hmax();
  constexpr int kAcc = kRows * kW4Width;
  const MoeRoute *rt = sc.route;
  const uint8_t *shared = w.shared;
  const MoeAffLayout L = w.blob, LS = w.shared_lay;
  const bf16 *hb = sc.h;
  float *y = sc.y;
  return q.submit([&](sycl::handler &hd) {
    hd.parallel_for<MoeAffDnK<kRows, KS>>(sycl::nd_range<1>(static_cast<size_t>(slots) * ot * KS, KS), [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
      if constexpr (KS > 1) esimd::slm_init<KS * kAcc * sizeof(float)>();
      const int g = static_cast<int>(it.get_group(0)), part = static_cast<int>(it.get_local_id(0));
      const int slot = g / ot, tile = g % ot;
      const int count = rt->count;
      if (slot > count) return;
      const bool sh = slot == count;
      const uint8_t *blob = sh ? shared : rt->blob[slot];
      const MoeAffLayout &lay = sh ? LS : L;
      const int groups = lay.ic / kW4Group;
      const int g0 = part * groups / KS, g1 = (part + 1) * groups / KS;
      const uint8_t *cw = blob + lay.dn_w;
      const auto *cs = reinterpret_cast<const uint16_t *>(blob + lay.dn_s);
      const auto *cb = reinterpret_cast<const uint16_t *>(blob + lay.dn_b);
      const bf16 *a = hb + static_cast<size_t>(slot) * kStepRows16 * hmax;
      float *yo = y + static_cast<size_t>(slot) * kStepRows16 * H + static_cast<size_t>(tile) * kW4Width;
      const uint32_t rmask = sh ? 0xffu : rt->mask[slot];  // only the rows routed here are read by the combine
      const int one = one_row(rmask);
      esimd::simd<float, kAcc> acc(0.f);
      if (one >= 0) {  // a slot of one row: that row only (its h row; the other rows' acc stay 0 and are not stored)
        esimd::simd<float, kW4Width> a1(0.f);
        for (int gg = g0; gg < g1; ++gg) aff_block<1>(a1, cw, cs, cb, H, tile, gg, a + static_cast<size_t>(one) * hmax, hmax);
        acc.template select<kW4Width, 1>(one * kW4Width) = a1;
      } else {
        for (int gg = g0; gg < g1; ++gg) aff_block<kRows>(acc, cw, cs, cb, H, tile, gg, a, hmax);
      }
      if constexpr (KS == 1) {
        for (int row = 0; row < live; ++row) {
          if (!((rmask >> row) & 1u)) continue;
          const esimd::simd<float, kW4Width> o = acc.template select<kW4Width, 1>(row * kW4Width);
          o.copy_to(yo + static_cast<size_t>(row) * H);
        }
      } else {
        esimd::slm_block_store<float, kAcc>(static_cast<uint32_t>(part * kAcc * sizeof(float)), acc);
        esimd::barrier();
        for (int row = part; row < live; row += KS) {
          if (!((rmask >> row) & 1u)) continue;
          esimd::simd<float, kW4Width> sum(0.f);
#pragma unroll
          for (int p = 0; p < KS; ++p) sum += esimd::slm_block_load<float, kW4Width>(static_cast<uint32_t>((p * kAcc + row * kW4Width) * sizeof(float)));
          sum.copy_to(yo + static_cast<size_t>(row) * H);
        }
      }
    });
  });
}

// combine (+ the two-card sum): one item per (row, 64-column chunk), one item per work-group (the peer stores of one
// work-group are rate-limited by its core: wire.cpp). The row's masks / weights into registers, then the slots'
// y 8 loads at a time (dead slots re-read the last live one with a zero contribution: still ascending slot order), the
// shared y, then (WIRE) the pair_reduce mailbox exchange: out = bf16(p_self + p_peer) (fp32 wire) or bf16(bf16(p_self) +
// bf16(p_peer)) (bf16 wire, PairWire:bf16); both cards add in the same order.
constexpr int kCombCols = 64;
template <bool WIRE, bool B16>
class MoeAffCombineK;
template <bool WIRE, bool B16>
sycl::event combine(sycl::queue &q, const MoeAffShape &s, int rows, const MoeAffScratch &sc, PairWire wire, int parity, bf16 *out) {
  const int H = s.hidden, chunks = H / kCombCols;
  const MoeRoute *rt = sc.route;
  const float *y = sc.y;
  float *partial = sc.partial;
  return q.parallel_for<MoeAffCombineK<WIRE, B16>>(sycl::nd_range<1>(static_cast<size_t>(rows) * chunks, 1), [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
    const int id = static_cast<int>(it.get_global_id(0)), r = id / chunks, c0 = (id % chunks) * kCombCols;
    const int count = rt->count;
    const esimd::simd<uint32_t, kSlots16> mk = esimd::block_load<uint32_t, kSlots16>(rt->mask);
    esimd::simd<float, kSlots16> wr = esimd::block_load<float, kSlots16>(&rt->weight[r][0]);
    wr.merge(esimd::simd<float, kSlots16>(0.f), ((mk >> r) & 1u) == 0u);
    esimd::simd<float, kCombCols> acc(0.f);
    const float *yr = y + static_cast<size_t>(r) * H + c0;
    for (int u0 = 0; u0 < count; u0 += 8) {
      esimd::simd<float, kCombCols> pv[8];
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const int u = u0 + j < count ? u0 + j : count - 1;
        pv[j] = esimd::block_load<float, kCombCols>(yr + static_cast<size_t>(u) * kStepRows16 * H);
      }
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const float wv = u0 + j < count ? static_cast<float>(wr[u0 + j < kSlots16 ? u0 + j : 0]) : 0.f;
        if (wv != 0.f) acc = pv[j] * wv + acc;
      }
    }
    acc += esimd::block_load<float, kCombCols>(yr + static_cast<size_t>(count) * kStepRows16 * H);
    if constexpr (!WIRE) {
      if (!out) {  // the card's partial only (tests, probes)
        acc.copy_to(partial + static_cast<size_t>(r) * H + c0);
        return;
      }
    } else if constexpr (B16) {
      const size_t at = (static_cast<size_t>(parity) * kPairRows + r) * H + c0;
      const esimd::simd<bf16, kCombCols> mine = acc;
      mail::mail_send<kCombCols / 2>(wire.peer_inbox + at, mail::pack_bf16<kCombCols>(mine), kPairMailEmpty, mail::kNanPairWord);
      const esimd::simd<uint32_t, kCombCols / 2> got = mail::mail_take<kCombCols / 2>(wire.inbox + at, kPairMailEmpty, wire.wait_cycles, wire.error, mail::kNanPairWord);
      acc = esimd::simd<float, kCombCols>(mine) + esimd::simd<float, kCombCols>(mail::unpack_bf16<kCombCols>(got));
    } else {
      constexpr auto uc = esimd::properties{esimd::cache_hint_L1<esimd::cache_hint::uncached>, esimd::cache_hint_L2<esimd::cache_hint::uncached>,
                                            esimd::alignment<256>};
      const size_t at = (static_cast<size_t>(parity) * kPairRows + r) * H + c0;
      esimd::simd<uint32_t, kCombCols> mine = acc.template bit_cast_view<uint32_t>();
      mine.merge(esimd::simd<uint32_t, kCombCols>(0x7fc00000u), mine == kPairMailEmpty);
      esimd::block_store<uint32_t, kCombCols>(wire.peer_inbox + at, mine, uc);
      esimd::simd<uint32_t, kCombCols> got = esimd::block_load<uint32_t, kCombCols>(wire.inbox + at, uc);
      bool ok = !(got == kPairMailEmpty).any();
      if (!ok) {
        const uint64_t t0 = esimd::rdtsc();
        while (!ok) {
          esimd::fence<esimd::memory_kind::global, esimd::fence_flush_op::none, esimd::fence_scope::system>();
          got = esimd::block_load<uint32_t, kCombCols>(wire.inbox + at, uc);
          ok = !(got == kPairMailEmpty).any();
          if (!ok && esimd::rdtsc() - t0 > wire.wait_cycles) break;
        }
      }
      if (!ok) {
        wire.error[0] = 1u;
        got = 0x7fc00000u;
      }
      esimd::block_store<uint32_t, kCombCols>(wire.inbox + at, esimd::simd<uint32_t, kCombCols>(kPairMailEmpty), uc);
      acc = acc + got.template bit_cast_view<float>();
    }
    const esimd::simd<bf16, kCombCols> o = acc;
    o.copy_to(out + static_cast<size_t>(r) * H + c0);
  });
}

template <int kRows>
MoeAffEvents decode_m(sycl::queue &q, const MoeAffShape &s, int rows, const bf16 *x, const MoeAffWeights &w, const MoeAffScratch &sc, const PairWire *wire,
                      int parity, bf16 *out, int gu_ks, int dn_ks) {
  MoeAffEvents e;
  const int slots = kRows * s.topk + 1;
  e.router = dense_gemm(q, rows, x, w.router, sc.logits);
  e.route = moe_aff_route_esimd() ? route_e_any(q, s, rows, sc, w.blobs) : route_m<kRows>(q, s, rows, sc.logits, w.blobs, sc.route);
  // the k splits, one pair for every row count (a row's bits must not depend on its step's rows: test-moe-aff).
  if (gu_ks <= 0) gu_ks = 4;
  if (dn_ks <= 0) dn_ks = 2;
  switch (gu_ks) {
    case 1: e.gate_up = gate_up<kRows, 1>(q, s, rows, slots, x, w, sc); break;
    case 2: e.gate_up = gate_up<kRows, 2>(q, s, rows, slots, x, w, sc); break;
    case 8: e.gate_up = gate_up<kRows, 8>(q, s, rows, slots, x, w, sc); break;
    default: e.gate_up = gate_up<kRows, 4>(q, s, rows, slots, x, w, sc); break;
  }
  switch (dn_ks) {
    case 1: e.down = down<kRows, 1>(q, s, rows, slots, w, sc); break;
    case 4: e.down = down<kRows, 4>(q, s, rows, slots, w, sc); break;
    default: e.down = down<kRows, 2>(q, s, rows, slots, w, sc); break;
  }
  if (wire)
    e.combine = wire->bf16 ? combine<true, true>(q, s, rows, sc, *wire, parity & 1, out) : combine<true, false>(q, s, rows, sc, *wire, parity & 1, out);
  else
    e.combine = combine<false, false>(q, s, rows, sc, PairWire{}, 0, s.cards == 1 ? out : nullptr);
  e.reduce = e.combine;
  return e;
}

uint16_t f2bf(float f) {  // round to nearest even
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
float bfr(float f) { return bf2f(f2bf(f)); }

}  // namespace

namespace {
int g_route_e = 1;  // 0 = the SIMT route (moe_aff_set_route_esimd: tests)
}
bool moe_aff_route_esimd() { return g_route_e != 0; }
void moe_aff_set_route_esimd(int on) { g_route_e = on; }

MoeAffLayout moe_aff_layout(int hidden, int ic) {
  if (hidden % kW4Group || ic % kW4Group) throw std::invalid_argument("moe_aff_layout: hidden and ic must be multiples of 128");
  MoeAffLayout l;
  l.ic = ic;
  const size_t nkgu = static_cast<size_t>(2 * ic) * hidden, nkdn = static_cast<size_t>(hidden) * ic;
  l.gu_w = 0;
  l.gu_s = align1k(nkgu / 2);
  l.gu_b = l.gu_s + align1k(nkgu / kW4AffGroup * 2);
  l.dn_w = l.gu_b + align1k(nkgu / kW4AffGroup * 2);
  l.dn_s = l.dn_w + align1k(nkdn / 2);
  l.dn_b = l.dn_s + align1k(nkdn / kW4AffGroup * 2);
  l.bytes = l.dn_b + align1k(nkdn / kW4AffGroup * 2);
  return l;
}

void moe_aff_pack(int hidden, int cards, int card, const MoeAffSrc &src, const MoeAffLayout &lay, uint8_t *blob) {
  const int I = src.inter, ic = lay.ic, H = hidden, G = kW4AffGroup;
  if (ic * cards != I) throw std::invalid_argument("moe_aff_pack: the layout's ic is not inter / cards");
  const int r0 = card * ic;
  // gate | up: this card's rows of each, as ct words [2 ic][H / 8]
  std::vector<uint32_t> gu(static_cast<size_t>(2 * ic) * (H / 8));
  std::vector<uint16_t> gs(static_cast<size_t>(H / G) * 2 * ic), gb(gs.size());
  for (int part = 0; part < 2; ++part) {
    const uint8_t *wsrc = part ? src.up_w : src.gate_w;
    const uint16_t *ssrc = part ? src.up_s : src.gate_s, *bsrc = part ? src.up_b : src.gate_b;
    for (int r = 0; r < ic; ++r) {
      const int dr = part * ic + r;
      std::memcpy(&gu[static_cast<size_t>(dr) * (H / 8)], wsrc + static_cast<size_t>(r0 + r) * (H / 2), static_cast<size_t>(H / 2));
      for (int g = 0; g < H / G; ++g) {
        gs[static_cast<size_t>(g) * 2 * ic + dr] = ssrc[static_cast<size_t>(r0 + r) * (H / G) + g];
        gb[static_cast<size_t>(g) * 2 * ic + dr] = bsrc[static_cast<size_t>(r0 + r) * (H / G) + g];
      }
    }
  }
  pack_w4_ct_rows(gu.data(), 2 * ic, H, 0, 2 * ic, blob + lay.gu_w);
  std::memcpy(blob + lay.gu_s, gs.data(), gs.size() * 2);
  std::memcpy(blob + lay.gu_b, gb.data(), gb.size() * 2);
  // down: every row, this card's k columns [r0, r0 + ic)
  std::vector<uint32_t> dn(static_cast<size_t>(H) * (ic / 8));
  std::vector<uint16_t> ds(static_cast<size_t>(ic / G) * H), db(ds.size());
  for (int r = 0; r < H; ++r) {
    std::memcpy(&dn[static_cast<size_t>(r) * (ic / 8)], src.down_w + static_cast<size_t>(r) * (I / 2) + r0 / 2, static_cast<size_t>(ic / 2));
    for (int g = 0; g < ic / G; ++g) {
      ds[static_cast<size_t>(g) * H + r] = src.down_s[static_cast<size_t>(r) * (I / G) + r0 / G + g];
      db[static_cast<size_t>(g) * H + r] = src.down_b[static_cast<size_t>(r) * (I / G) + r0 / G + g];
    }
  }
  pack_w4_ct_rows(dn.data(), H, ic, 0, H, blob + lay.dn_w);
  std::memcpy(blob + lay.dn_s, ds.data(), ds.size() * 2);
  std::memcpy(blob + lay.dn_b, db.data(), db.size() * 2);
}

size_t moe_aff_h_elems(const MoeAffShape &s) { return static_cast<size_t>(kStepRows16 * s.topk + 1) * kStepRows16 * s.hmax(); }
size_t moe_aff_y_elems(const MoeAffShape &s) { return static_cast<size_t>(kStepRows16 * s.topk + 1) * kStepRows16 * s.hidden; }

MoeAffEvents moe_aff_decode(sycl::queue &q, const MoeAffShape &s, int rows, const bf16 *x, const MoeAffWeights &w, const MoeAffScratch &sc,
                            const PairWire *wire, int parity, bf16 *out, int gu_ks, int dn_ks) {
  if (rows < 1 || rows > kStepRows16) throw std::invalid_argument("moe_aff_decode: rows 1..8");
  if (s.topk < 1 || s.topk > kMoeMaxTopk || s.experts < s.topk || s.experts > 1024) throw std::invalid_argument("moe_aff_decode: experts / topk");
  if (s.inter % s.cards || s.shared_inter % s.cards || s.inter_c() % kW4Group || s.shared_c() % kW4Group || s.hidden % kW4Group)
    throw std::invalid_argument("moe_aff_decode: intermediate columns per card and hidden must be multiples of 128");
  if (w.blob.ic != s.inter_c() || w.shared_lay.ic != s.shared_c() || !w.blobs || !w.shared || !w.silu_tab)
    throw std::invalid_argument("moe_aff_decode: weights do not match the shape");
  if (!w.router.valid() || w.router.kind != DenseKind::bf16 || w.router.n != s.experts || w.router.k != s.hidden)
    throw std::invalid_argument("moe_aff_decode: the router must be bf16 [experts][hidden]");
  if (s.cards == 1 && wire) throw std::invalid_argument("moe_aff_decode: one card takes no wire");
  if (s.hidden % kCombCols) throw std::invalid_argument("moe_aff_decode: hidden % 64");
  if (row_shape(rows) == 4) return decode_m<4>(q, s, rows, x, w, sc, wire, parity, out, gu_ks, dn_ks);
  return decode_m<8>(q, s, rows, x, w, sc, wire, parity, out, gu_ks, dn_ks);
}


// ---- many rows (prefill): router, route, group, combine and the block's driver. gate_up / down are
// moe_aff_rows.cpp's (256 GRF, xenithium_kernels_xl).
namespace {
namespace xesimd = sycl::ext::intel::experimental::esimd;
constexpr int kMarSeg = kMoeAffSegRows, kMarGroupItems = 1024, kMarCombCols = 64;
inline int mar_ld(int m) { return (m + 31) / 32 * 32; }
template <int PER>
class MarRoute;
class MarGroup;
class MarCombine;
// ---- route: decode's arithmetic (moe_aff.cpp route) per row --------------------------------------------------------
template <int PER>
sycl::event rows_route(sycl::queue &q, const MoeAffShape &s, int m, const float *logits, const MoeAffRowsScratch &sc) {
  const int E = s.experts, K = s.topk;
  int32_t *ids = sc.ids;
  float *topw = sc.topw;
  return q.parallel_for<MarRoute<PER>>(sycl::nd_range<1>(static_cast<size_t>(m) * 16, 16), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
    const auto sg = it.get_sub_group();
    const int r = static_cast<int>(it.get_group(0)), lane = static_cast<int>(sg.get_local_linear_id());
    uint32_t key[PER];
    const float *lr = logits + static_cast<size_t>(r) * E + lane * PER;
#pragma unroll
    for (int x = 0; x < PER; ++x) {
      const int e = lane * PER + x;
      const uint32_t b = e < E ? sycl::bit_cast<uint32_t>(lr[x]) : 0u;
      const uint32_t o = (b & 0x80000000u) ? ~b : (b | 0x80000000u);
      key[x] = e < E ? o : 0u;
    }
    int my_e = 0;
    float my_s = 0.f;
    for (int j = 0; j < K; ++j) {
      uint32_t mx = 0;
#pragma unroll
      for (int x = 0; x < PER; ++x) mx = key[x] > mx ? key[x] : mx;
      mx = sycl::reduce_over_group(sg, mx, sycl::maximum<uint32_t>());
      int lo = 0x7fffffff;
#pragma unroll
      for (int x = PER - 1; x >= 0; --x) lo = key[x] == mx ? lane * PER + x : lo;
      const int be = sycl::reduce_over_group(sg, lo, sycl::minimum<int>());
#pragma unroll
      for (int x = 0; x < PER; ++x) key[x] = lane * PER + x == be ? 0u : key[x];
      if (lane == j) {
        const float lg = sycl::bit_cast<float>((mx & 0x80000000u) ? (mx & 0x7fffffffu) : ~mx);
        my_e = be;
        my_s = 1.f / (1.f + sycl::exp(-lg));
      }
    }
    const float z = sycl::reduce_over_group(sg, lane < K ? my_s : 0.f, sycl::plus<float>());
    if (lane < K) {
      ids[static_cast<size_t>(r) * K + lane] = my_e;
      topw[static_cast<size_t>(r) * K + lane] = my_s / z;
    }
  });
}

// ---- group by expert (moe_rows' scheme) ------------------------------------------------------------------------------
sycl::event rows_group(sycl::queue &q, const MoeAffShape &s, int m, const MoeAffRowsScratch &sc, int segr) {
  const int E = s.experts, K = s.topk, A = m * s.topk;
  const int32_t *ids = sc.ids;
  int32_t *slot = sc.slot, *row = sc.row, *seg = sc.seg, *seg_n = sc.seg_n;
  return q.submit([&](sycl::handler &h) {
    sycl::local_accessor<int32_t, 1> cnt(sycl::range<1>(kMoeMaxExperts), h);
    sycl::local_accessor<int32_t, 1> off(sycl::range<1>(kMoeMaxExperts), h);
    sycl::local_accessor<int32_t, 1> cur(sycl::range<1>(kMoeMaxExperts), h);
    h.parallel_for<MarGroup>(sycl::nd_range<1>(kMarGroupItems, kMarGroupItems), [=](sycl::nd_item<1> it) {
      const int i = static_cast<int>(it.get_local_linear_id());
      using LA = sycl::atomic_ref<int32_t, sycl::memory_order::relaxed, sycl::memory_scope::work_group, sycl::access::address_space::local_space>;
      if (i < E) cnt[i] = 0, cur[i] = 0;
      sycl::group_barrier(it.get_group());
      for (int a = i; a < A; a += kMarGroupItems) LA(cnt[ids[a]]).fetch_add(1);
      sycl::group_barrier(it.get_group());
      const int c = i < E ? cnt[i] : 0;
      const int ns = (c + segr - 1) / segr;
      const int o = sycl::exclusive_scan_over_group(it.get_group(), c, sycl::plus<int>());
      const int so = sycl::exclusive_scan_over_group(it.get_group(), ns, sycl::plus<int>());
      if (i < E) off[i] = o;
      for (int j = 0; j < ns; ++j) {
        int32_t *sp = seg + static_cast<size_t>(so + j) * 4;
        sp[0] = i;
        sp[1] = o + j * segr;
        sp[2] = sycl::min(segr, c - j * segr);
      }
      if (i == kMarGroupItems - 1) seg_n[0] = so + ns;
      sycl::group_barrier(it.get_group());
      for (int a = i; a < A; a += kMarGroupItems) {
        const int e = ids[a];
        const int p = off[e] + LA(cur[e]).fetch_add(1);
        row[p] = a / K;
        slot[a] = p;
      }
    });
  });
}

// ---- combine: the row's experts in ascending id, then the shared y -> part fp32 ----------------------------------------
sycl::event rows_combine(sycl::queue &q, const MoeAffShape &s, int base, int rows, const MoeAffRowsScratch &sc) {
  const int H = s.hidden, K = s.topk, chunks = H / kMarCombCols;
  const int32_t *ids = sc.ids, *slot = sc.slot;
  const float *topw = sc.topw, *y = sc.y, *sy = sc.sy;
  float *part = sc.part;
  return q.parallel_for<MarCombine>(sycl::nd_range<1>(static_cast<size_t>(rows) * chunks, chunks % 8 == 0 ? 8 : 1), [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
    const int id = static_cast<int>(it.get_global_id(0)), r = base + id / chunks, c0 = (id % chunks) * kMarCombCols;
    int ei[kMoeMaxTopk], pi[kMoeMaxTopk];
    float wi[kMoeMaxTopk];
    for (int j = 0; j < K; ++j) {  // insertion sort by expert id (decode's slot order)
      const size_t a = static_cast<size_t>(r) * K + j;
      const int e = ids[a], p = slot[a];
      const float wv = topw[a];
      int x = j;
      while (x > 0 && ei[x - 1] > e) {
        ei[x] = ei[x - 1], pi[x] = pi[x - 1], wi[x] = wi[x - 1];
        --x;
      }
      ei[x] = e, pi[x] = p, wi[x] = wv;
    }
    esimd::simd<float, kMarCombCols> acc(0.f);
    for (int j = 0; j < K; ++j) {
      const esimd::simd<float, kMarCombCols> v = esimd::block_load<float, kMarCombCols>(y + static_cast<size_t>(pi[j]) * H + c0);
      acc = xesimd::fma(v, esimd::simd<float, kMarCombCols>(wi[j]), acc);
    }
    acc += esimd::block_load<float, kMarCombCols>(sy + static_cast<size_t>(r) * H + c0);
    acc.copy_to(part + static_cast<size_t>(r) * H + c0);
  });
}

}  // namespace

int moe_aff_rows_max_segs(const MoeAffShape &s, int rows_max) { return s.experts + rows_max * s.topk / kMarSeg + 1; }

MoeAffRowsSizes moe_aff_rows_sizes(const MoeAffShape &s, int rows_max) {
  const size_t R = static_cast<size_t>(rows_max), A = R * s.topk;
  MoeAffRowsSizes z;
  z.logits = R * s.experts;
  z.ids = A;
  z.row = A;
  z.seg = static_cast<size_t>(moe_aff_rows_max_segs(s, rows_max)) * 4;
  z.ax = static_cast<size_t>(s.hidden / kW4AffGroup) * mar_ld(rows_max);
  z.ah = static_cast<size_t>(s.inter_c() / kW4AffGroup) * mar_ld(static_cast<int>(A));
  z.h = A * s.inter_c();
  z.y = A * s.hidden;
  z.sh = R * s.shared_c();
  z.sy = R * s.hidden;
  z.part = R * s.hidden;
  return z;
}

MoeAffEvents moe_aff_rows(sycl::queue &q, const MoeAffShape &s, int m, const bf16 *x, const MoeAffWeights &w, const MoeAffRowsScratch &sc,
                          const PairRowsWire *wire, int parity, bf16 *out) {
  if (m < 1 || m > sc.rows_max) throw std::invalid_argument("moe_aff_rows: 1..rows_max rows");
  if (s.topk < 1 || s.topk > kMoeMaxTopk || s.topk > 16 || s.experts < s.topk || s.experts > 1024 || s.experts % 16)
    throw std::invalid_argument("moe_aff_rows: experts (a multiple of 16, <= 1024) / topk (<= 16)");
  if (s.inter_c() % kW4Group || s.shared_c() % kW4Group || s.hidden % kW4Group || (s.hidden / kW4Width) % 2 || s.hidden % kMarCombCols)
    throw std::invalid_argument("moe_aff_rows: intermediate columns per card and hidden must be multiples of 128");
  if (w.blob.ic != s.inter_c() || w.shared_lay.ic != s.shared_c() || !w.blobs || !w.shared || !w.silu_tab)
    throw std::invalid_argument("moe_aff_rows: weights do not match the shape");
  if (!w.router.valid() || w.router.kind != DenseKind::bf16 || w.router.n != s.experts || w.router.k != s.hidden)
    throw std::invalid_argument("moe_aff_rows: the router must be bf16 [experts][hidden]");
  if ((s.cards > 1) != (wire != nullptr)) throw std::invalid_argument("moe_aff_rows: a wire is needed exactly with two cards");
  if (sc.lds_x < mar_ld(m) || sc.lds_h < mar_ld(m * s.topk)) throw std::invalid_argument("moe_aff_rows: row-sum strides");
  MoeAffEvents e;
  const int H = s.hidden, I = s.inter_c(), SI = s.shared_c(), A = m * s.topk;
  const int max_segs = moe_aff_rows_max_segs(s, m);
  e.router = pf_dense(q, m, w.router, x, H, PfOut{sc.logits, nullptr, s.experts});
  const int per = s.experts / 16;
  if (per <= 16) e.route = rows_route<16>(q, s, m, sc.logits, sc);
  else if (per <= 32) e.route = rows_route<32>(q, s, m, sc.logits, sc);
  else e.route = rows_route<64>(q, s, m, sc.logits, sc);
  rows_group(q, s, m, sc, kMarSeg);
  w4aff_rowsum(q, x, H, m, H, sc.ax, sc.lds_x);
  e.gate_up = moe_aff_rows_gate_up(q, s, max_segs, x, w, sc);
  w4aff_rowsum(q, sc.h, I, A, I, sc.ah, sc.lds_h);
  e.down = moe_aff_rows_down(q, s, max_segs, w, sc);
  // the shared expert on every row: its card's columns (gate | up with swiglu, then down's k slice)
  const MoeAffLayout &L = w.shared_lay;
  w4aff_gemm_pf_swiglu(q, m, x, H, w.shared + L.gu_w, reinterpret_cast<const uint16_t *>(w.shared + L.gu_s), reinterpret_cast<const uint16_t *>(w.shared + L.gu_b),
                       sc.sh, SI, SI, H);
  w4aff_gemm_pf(q, m, sc.sh, SI, w.shared + L.dn_w, reinterpret_cast<const uint16_t *>(w.shared + L.dn_s), reinterpret_cast<const uint16_t *>(w.shared + L.dn_b),
                sc.sy, H, H, SI);
  if (wire && wire->split && wire->send) {  // the peer's rows first, their copy on the side queue beside the own rows
    int lo = 0, hi = 0;
    pair_split_owner_rows(1 - s.card, m, &lo, &hi);
    if (hi > lo) rows_combine(q, s, lo, hi - lo, sc);
    e.send = pair_split_send(q, wire->send->side, s.card, m, H, sc.part, *wire->send->sp, wire->send->seq);
    pair_split_owner_rows(s.card, m, &lo, &hi);
    if (hi > lo) e.combine = rows_combine(q, s, lo, hi - lo, sc);
  } else {
    e.combine = rows_combine(q, s, 0, m, sc);
  }
  if (!wire || !wire->split) e.reduce = pair_reduce_rows(q, m, H, sc.part, wire, parity, out);
  return e;
}


// ---- host reference -------------------------------------------------------------------------------------------

MoeAffRefExpert moe_aff_dequant(int hidden, const MoeAffSrc &src) {
  const int I = src.inter, H = hidden, G = kW4AffGroup;
  MoeAffRefExpert e;
  auto deq = [&](const uint8_t *w, const uint16_t *s, const uint16_t *b, int n, int k, std::vector<float> &out) {
    out.resize(static_cast<size_t>(n) * k);
    for (int r = 0; r < n; ++r)
      for (int c = 0; c < k; ++c) {
        const uint8_t byte = w[static_cast<size_t>(r) * (k / 2) + c / 2];
        const int qv = c % 2 ? byte >> 4 : byte & 15;
        const size_t gi = static_cast<size_t>(r) * (k / G) + c / G;
        out[static_cast<size_t>(r) * k + c] = static_cast<float>(qv) * bf2f(s[gi]) + bf2f(b[gi]);
      }
  };
  deq(src.gate_w, src.gate_s, src.gate_b, I, H, e.gate);
  deq(src.up_w, src.up_s, src.up_b, I, H, e.up);
  deq(src.down_w, src.down_s, src.down_b, H, I, e.down);
  return e;
}

void moe_aff_route_ref(const float *x, const float *router, int hidden, int experts, int topk, int32_t *ids, float *weights, double *logits) {
  std::vector<double> sc(experts);
  for (int e = 0; e < experts; ++e) {
    double d = 0;
    for (int c = 0; c < hidden; ++c) d += static_cast<double>(x[c]) * router[static_cast<size_t>(e) * hidden + c];
    if (logits) logits[e] = d;
    sc[e] = 1.0 / (1.0 + std::exp(-d));
  }
  double sum = 0;
  for (int k = 0; k < topk; ++k) {
    int best = -1;
    for (int e = 0; e < experts; ++e) {
      if (std::find(ids, ids + k, e) != ids + k) continue;
      if (best < 0 || sc[e] > sc[best]) best = e;
    }
    ids[k] = best;
    sum += sc[best];
  }
  for (int k = 0; k < topk; ++k) weights[k] = static_cast<float>(sc[ids[k]] / sum);
}

void moe_aff_ref(const float *x, int hidden, int topk, const int32_t *ids, const float *weights, const MoeAffRefExpert *const *experts,
                 const MoeAffRefExpert &shared, double *out, const std::vector<std::vector<float>> *h_force) {
  const int H = hidden;
  std::fill(out, out + H, 0.0);
  auto mlp = [&](const MoeAffRefExpert &e, double wgt, int which) {
    const int I = static_cast<int>(e.gate.size() / H);
    std::vector<float> h(I);
    if (h_force) {
      h = (*h_force)[static_cast<size_t>(which)];
    } else {
      for (int i = 0; i < I; ++i) {
        double g = 0, u = 0;
        for (int c = 0; c < H; ++c) g += static_cast<double>(x[c]) * e.gate[static_cast<size_t>(i) * H + c], u += static_cast<double>(x[c]) * e.up[static_cast<size_t>(i) * H + c];
        const float gb = bfr(static_cast<float>(g)), ub = bfr(static_cast<float>(u));
        h[i] = bfr(gb / (1.f + std::exp(-gb)) * ub);
      }
    }
    for (int c = 0; c < H; ++c) {
      double y = 0;
      for (int i = 0; i < I; ++i) y += static_cast<double>(h[i]) * e.down[static_cast<size_t>(c) * I + i];
      out[c] += wgt * y;
    }
  };
  for (int k = 0; k < topk; ++k) mlp(*experts[k], weights[k], k);
  mlp(shared, 1.0, topk);
}

}  // namespace xenithium:kernels
