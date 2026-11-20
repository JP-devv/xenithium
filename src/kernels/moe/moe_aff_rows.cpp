// Flint's MoE block over many rows (prefill; moe_aff.hpp, moe_aff_rows): its gate_up and down kernels, built at 256 GRF
// (xenithium_kernels_xl; the rest of the block is moe_aff.cpp's).
// Launches per layer and card: router (pf_dense over the bf16 router), route (one 16-lane sub-group per row: decode's
// top-k arithmetic on the fp32 logits), group (one work-group: an expert-ordered list of the (row, k) assignments and
// segments of <= 32 list rows of one expert), x's row sums per 64-k group (w4aff_rowsum), gate_up (one item per
// (segment, 16-row part, 16-column h tile): gate and up of the part's gathered x rows, affine g64 folded per group
// with explicit fmas, h = swiglu by decode's silu table), h's row sums, down (one item per (segment, two 16-column
// output tiles): y fp32 per list row), the shared expert on every row (w4aff_gemm_pf_swiglu + w4aff_gemm_pf: its card's
// intermediate columns), combine (per (row, 64 columns): the row's experts in ascending id, w y + acc by fma, then + the
// shared y) -> the card's fp32 partial, then the two-card sum (pair_reduce_rows, or the caller's split sum).
// Every row's arithmetic depends only on the row (list order inside an expert is by atomics and changes no result), so
// a prompt prefilled in calls of any size gives the same rows. Built at 256 GRF (xenithium_kernels_xl).
#include <sycl/ext/intel/esimd.hpp>

#include <algorithm>
#include <stdexcept>

#include "xenithium/kernels/moe/moe_aff.hpp"
#include "xenithium/kernels/pack.hpp"
#include "xenithium/kernels/tile/w4_tile.hpp"
#include "xenithium/kernels/gemm/w4a16_gemm.hpp"

namespace xenithium::kernels {
namespace {

namespace esimd = sycl::ext::intel::esimd;
namespace xesimd = sycl::ext::intel::experimental::esimd;
namespace xmx = esimd::xmx;
using bf16 = sycl::ext::oneapi::bfloat16;

constexpr int kW = 16, kSeg = 16;  // gate_up's rows per item
constexpr int kCB = w4::kCallBytes;  // 128 bytes = one 16-k call of a 16-column tile
constexpr int kKC = w4::kKPerCall;  // 16

// raw codes (0..15) of one 16-k call -> the bf16 B operand
inline esimd::simd<bf16, 2 * kCB> raw_codes(esimd::simd<uint8_t, kW4BlockBytes> raw, int j) SYCL_ESIMD_FUNCTION {
  const esimd::simd<uint8_t, kCB> bytes = raw.template select<kCB, 1>(j * kCB);
  esimd::simd<bf16, 2 * kCB> b;
  b.template select<kCB, 2>(0) = esimd::simd<float, kCB>(bytes & 15);
  b.template select<kCB, 2>(1) = esimd::simd<float, kCB>(bytes >> 4);
  return b;
}
inline esimd::simd<float, kW> bf16_row(const uint16_t *p) SYCL_ESIMD_FUNCTION {
  esimd::simd<uint32_t, kW> u = esimd::block_load<uint16_t, kW>(p);
  u <<= 16;
  return u.template bit_cast_view<float>().read();
}
// acc[r] = fma(b, asum[r], fma(gs[r], s, acc[r])) for rows [0, R)
template <int R>
inline void fold(esimd::simd<float, R * kW> &acc, esimd::simd<float, R * kW> gs, esimd::simd<float, kW> sv, esimd::simd<float, kW> bv,
                 esimd::simd<float, R> as) SYCL_ESIMD_FUNCTION {
#pragma unroll
  for (int r = 0; r < R; ++r) {
    auto av = acc.template select<kW, 1>(r * kW);
    const esimd::simd<float, kW> t1 = xesimd::fma(gs.template select<kW, 1>(r * kW).read(), sv, av.read());
    av = xesimd::fma(bv, esimd::simd<float, kW>(static_cast<float>(as[r])), t1);
  }
}

class MarGateUp;
template <int S>
class MarDown;

// ---- gate_up -> h: one item per (segment, 16-row part, h tile); gate and up tiles of the part's rows ------------------
sycl::event gate_up(sycl::queue &q, const MoeAffShape &s, int max_segs, int parts, const bf16 *x, const MoeAffWeights &w, const MoeAffRowsScratch &sc) {
  const int H = s.hidden, I = s.inter_c(), tiles_i = I / kW, G = H / kW4Group, lds = sc.lds_x;
  const MoeAffLayout lay = w.blob;
  const uint8_t *const *blobs = w.blobs;
  const int32_t *seg = sc.seg, *seg_n = sc.seg_n, *row = sc.row;
  const float *ax = sc.ax, *tab = w.silu_tab;
  bf16 *hbuf = sc.h;
  const size_t items = static_cast<size_t>(max_segs) * parts * tiles_i;
  constexpr int kWg = 4;
  return q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for<MarGateUp>(sycl::nd_range<1>((items + kWg - 1) / kWg * kWg, kWg), [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
      const int gid = static_cast<int>(it.get_global_id(0));
      const int sid = gid / (parts * tiles_i), part = gid / tiles_i % parts, j = gid % tiles_i;
      if (sid >= seg_n[0]) return;
      const esimd::simd<int32_t, 4> sp = esimd::block_load<int32_t, 4>(seg + static_cast<size_t>(sid) * 4);
      const int e = sp[0], p0 = sp[1] + part * kSeg, n = sp[2] - part * kSeg;
      if (n <= 0) return;
      const uint8_t *blob = blobs[e];
      const uint8_t *codes = blob + lay.gu_w;
      const uint16_t *scl = reinterpret_cast<const uint16_t *>(blob + lay.gu_s), *bia = reinterpret_cast<const uint16_t *>(blob + lay.gu_b);
      int rr[kSeg];
      esimd::simd<uint32_t, kSeg> roff;
#pragma unroll
      for (int r = 0; r < kSeg; ++r) {
        rr[r] = row[p0 + (r < n ? r : n - 1)];
        roff[r] = static_cast<uint32_t>(rr[r]) * 4u;
      }
      esimd::simd<float, 8 * kW> ag0(0.f), au0(0.f), ag1(0.f), au1(0.f);
      const bool two = n > 8;
      for (int g = 0; g < G; ++g) {
        const auto rg = w4::load_block(codes + (static_cast<size_t>(g) * 2 * tiles_i + j) * kW4BlockBytes);
        const auto ru = w4::load_block(codes + (static_cast<size_t>(g) * 2 * tiles_i + tiles_i + j) * kW4BlockBytes);
        esimd::simd<bf16, 8 * kW4Group> ah0, ah1;
#pragma unroll
        for (int r = 0; r < 8; ++r) ah0.template select<kW4Group, 1>(r * kW4Group) = esimd::block_load<bf16, kW4Group>(x + static_cast<size_t>(rr[r]) * H + g * kW4Group);
        if (two)
#pragma unroll
          for (int r = 0; r < 8; ++r) ah1.template select<kW4Group, 1>(r * kW4Group) = esimd::block_load<bf16, kW4Group>(x + static_cast<size_t>(rr[8 + r]) * H + g * kW4Group);
#pragma unroll
        for (int sg = 0; sg < 2; ++sg) {  // the block's two 64-k groups
          const int gg = 2 * g + sg;
          esimd::simd<float, 8 * kW> gg0(0.f), gu0(0.f), gg1(0.f), gu1(0.f);
#pragma unroll
          for (int jj = 0; jj < 4; ++jj) {
            const int jc = 4 * sg + jj;
            const esimd::simd<bf16, 2 * kCB> bg = raw_codes(rg, jc), bu = raw_codes(ru, jc);
            esimd::simd<bf16, 8 * kKC> at;
#pragma unroll
            for (int r = 0; r < 8; ++r) at.template select<kKC, 1>(r * kKC) = ah0.template select<kKC, 1>(r * kW4Group + jc * kKC);
            gg0 = xmx::dpas<8, 8, float, float, bf16, bf16>(gg0, bg, at);
            gu0 = xmx::dpas<8, 8, float, float, bf16, bf16>(gu0, bu, at);
            if (two) {
#pragma unroll
              for (int r = 0; r < 8; ++r) at.template select<kKC, 1>(r * kKC) = ah1.template select<kKC, 1>(r * kW4Group + jc * kKC);
              gg1 = xmx::dpas<8, 8, float, float, bf16, bf16>(gg1, bg, at);
              gu1 = xmx::dpas<8, 8, float, float, bf16, bf16>(gu1, bu, at);
            }
          }
          const esimd::simd<float, kW> sgv = bf16_row(scl + static_cast<size_t>(gg) * 2 * I + j * kW);
          const esimd::simd<float, kW> bgv = bf16_row(bia + static_cast<size_t>(gg) * 2 * I + j * kW);
          const esimd::simd<float, kW> suv = bf16_row(scl + static_cast<size_t>(gg) * 2 * I + (tiles_i + j) * kW);
          const esimd::simd<float, kW> buv = bf16_row(bia + static_cast<size_t>(gg) * 2 * I + (tiles_i + j) * kW);
          esimd::simd<float, kSeg> as = esimd::gather<float, kSeg>(ax + static_cast<size_t>(gg) * lds, roff);
          fold<8>(ag0, gg0, sgv, bgv, as.template select<8, 1>(0));
          fold<8>(au0, gu0, suv, buv, as.template select<8, 1>(0));
          if (two) {
            fold<8>(ag1, gg1, sgv, bgv, as.template select<8, 1>(8));
            fold<8>(au1, gu1, suv, buv, as.template select<8, 1>(8));
          }
        }
      }
      // h = bf16(s(bf16 g) x bf16 u), s from decode's silu table
#pragma unroll
      for (int r = 0; r < kSeg; ++r) {
        if (r >= n) break;
        const esimd::simd<float, kW> gs = r < 8 ? ag0.template select<kW, 1>(r * kW).read() : ag1.template select<kW, 1>((r - 8) * kW).read();
        const esimd::simd<float, kW> us = r < 8 ? au0.template select<kW, 1>(r * kW).read() : au1.template select<kW, 1>((r - 8) * kW).read();
        esimd::simd<bf16, kW> gb = gs;
        const esimd::simd<float, kW> uv = esimd::simd<bf16, kW>(us);
        const esimd::simd<uint32_t, kW> off = esimd::simd<uint32_t, kW>(gb.template bit_cast_view<uint16_t>().read()) * 4u;
        const esimd::simd<float, kW> sv = esimd::gather<float, kW>(tab, off);
        const esimd::simd<bf16, kW> hv = sv * uv;
        hv.copy_to(hbuf + static_cast<size_t>(p0 + r) * I + j * kW);
      }
    });
  });
}


// ---- down -> y fp32: one item per (segment of <= S rows, two output tiles) ---------------------------------------------
template <int S>
sycl::event down(sycl::queue &q, const MoeAffShape &s, int max_segs, const MoeAffWeights &w, const MoeAffRowsScratch &sc) {
  constexpr int NH = S / 8;
  const int H = s.hidden, I = s.inter_c(), tiles = H / kW, pairs = tiles / 2, G = I / kW4Group, lds = sc.lds_h;
  const MoeAffLayout lay = w.blob;
  const uint8_t *const *blobs = w.blobs;
  const int32_t *seg = sc.seg, *seg_n = sc.seg_n;
  const bf16 *hbuf = sc.h;
  const float *ah = sc.ah;
  float *y = sc.y;
  constexpr int kWg = 8;
  const size_t items = static_cast<size_t>(max_segs) * pairs;
  return q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for<MarDown<S>>(sycl::nd_range<1>((items + kWg - 1) / kWg * kWg, kWg), [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
      const int gid = static_cast<int>(it.get_global_id(0));
      const int sid = gid / pairs, t0 = (gid % pairs) * 2;
      if (sid >= seg_n[0]) return;
      const esimd::simd<int32_t, 4> sp = esimd::block_load<int32_t, 4>(seg + static_cast<size_t>(sid) * 4);
      const int e = sp[0], p0 = sp[1], n = sp[2];
      const int nh = (n + 7) / 8;
      const uint8_t *blob = blobs[e];
      const uint8_t *codes = blob + lay.dn_w;
      const uint16_t *scl = reinterpret_cast<const uint16_t *>(blob + lay.dn_s), *bia = reinterpret_cast<const uint16_t *>(blob + lay.dn_b);
      esimd::simd<float, S * kW> acc0(0.f), acc1(0.f);
      for (int g = 0; g < G; ++g) {
        const auto raw0 = w4::load_block(codes + (static_cast<size_t>(g) * tiles + t0) * kW4BlockBytes);
        const auto raw1 = w4::load_block(codes + (static_cast<size_t>(g) * tiles + t0 + 1) * kW4BlockBytes);
#pragma unroll
        for (int sg = 0; sg < 2; ++sg) {
          const int gg = 2 * g + sg;
          esimd::simd<bf16, 2 * kCB> b0[4], b1[4];
#pragma unroll
          for (int jj = 0; jj < 4; ++jj) b0[jj] = raw_codes(raw0, 4 * sg + jj), b1[jj] = raw_codes(raw1, 4 * sg + jj);
          const esimd::simd<float, kW> sv0 = bf16_row(scl + static_cast<size_t>(gg) * H + t0 * kW), bv0 = bf16_row(bia + static_cast<size_t>(gg) * H + t0 * kW);
          const esimd::simd<float, kW> sv1 = bf16_row(scl + static_cast<size_t>(gg) * H + (t0 + 1) * kW),
                                       bv1 = bf16_row(bia + static_cast<size_t>(gg) * H + (t0 + 1) * kW);
#pragma unroll
          for (int hb = 0; hb < NH; ++hb) {
            if (hb >= nh) break;
            esimd::simd<bf16, 8 * kW4AffGroup> a;
#pragma unroll
            for (int r = 0; r < 8; ++r) {
              const int rr = hb * 8 + r < n ? hb * 8 + r : n - 1;
              a.template select<kW4AffGroup, 1>(r * kW4AffGroup) = esimd::block_load<bf16, kW4AffGroup>(hbuf + static_cast<size_t>(p0 + rr) * I + gg * kW4AffGroup);
            }
            esimd::simd<float, 8 * kW> s0(0.f), s1(0.f);
#pragma unroll
            for (int jj = 0; jj < 4; ++jj) {
              esimd::simd<bf16, 8 * kKC> at;
#pragma unroll
              for (int r = 0; r < 8; ++r) at.template select<kKC, 1>(r * kKC) = a.template select<kKC, 1>(r * kW4AffGroup + jj * kKC);
              s0 = xmx::dpas<8, 8, float, float, bf16, bf16>(s0, b0[jj], at);
              s1 = xmx::dpas<8, 8, float, float, bf16, bf16>(s1, b1[jj], at);
            }
            esimd::simd<float, 8> as = esimd::gather<float, 8>(ah + static_cast<size_t>(gg) * lds + p0 + hb * 8, esimd::simd<uint32_t, 8>(0, 4));
            esimd::simd<float, 8 * kW> c0 = acc0.template select<8 * kW, 1>(hb * 8 * kW), c1 = acc1.template select<8 * kW, 1>(hb * 8 * kW);
            fold<8>(c0, s0, sv0, bv0, as);
            fold<8>(c1, s1, sv1, bv1, as);
            acc0.template select<8 * kW, 1>(hb * 8 * kW) = c0;
            acc1.template select<8 * kW, 1>(hb * 8 * kW) = c1;
          }
        }
      }
#pragma unroll
      for (int r = 0; r < S; ++r) {
        if (r >= n) break;
        esimd::simd<float, 2 * kW> v;
        v.template select<kW, 1>(0) = acc0.template select<kW, 1>(r * kW);
        v.template select<kW, 1>(kW) = acc1.template select<kW, 1>(r * kW);
        v.copy_to(y + static_cast<size_t>(p0 + r) * H + t0 * kW);
      }
    });
  });
}

}  // namespace

sycl::event moe_aff_rows_gate_up(sycl::queue &q, const MoeAffShape &s, int max_segs, const bf16 *x, const MoeAffWeights &w, const MoeAffRowsScratch &sc) {
  return gate_up(q, s, max_segs, kMoeAffSegRows / kSeg, x, w, sc);
}
sycl::event moe_aff_rows_down(sycl::queue &q, const MoeAffShape &s, int max_segs, const MoeAffWeights &w, const MoeAffRowsScratch &sc) {
  return down<kMoeAffSegRows>(q, s, max_segs, w, sc);
}

}  // namespace xenithium:kernels
