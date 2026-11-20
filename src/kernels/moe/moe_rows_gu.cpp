// Xenithium · moe_rows' W4A16 gate_up and down. Their own TU because they need 256 GRF (gate_up
// spills at 128): built in xenithium_kernels_xl (-doubleGRF). gate_up is one item for both projections, down two tiles
// per item.
#include <stdexcept>

#include <sycl/ext/intel/esimd.hpp>

#include "xenithium/kernels/moe/moe_rows.hpp"
#include "xenithium/kernels/pack.hpp"
#include "xenithium/kernels/tile/w4_tile.hpp"

namespace xenithium::kernels {
namespace {

namespace esimd = sycl::ext::intel::esimd;
namespace xmx = esimd::xmx;
using bf16 = moe_bf16;

constexpr int kW = 16;
constexpr int kGroup = 128;
constexpr int kSeg = kMoeSegRows;

class MrGateUpBoth;
template <int PK>
class MrDown2;

template <int N>
inline esimd::simd<float, N> sigmoid_v(esimd::simd<float, N> x) SYCL_ESIMD_FUNCTION {
  return esimd::inv(esimd::simd<float, N>(1.f) + esimd::exp(-x));
}

// ---- 4: gate_up -> h --------------------------------------------------------------------------------
// One item per (segment, tile) computing both the gate and the up tile, so each x row is loaded
// once per group for both.
sycl::event gate_up_both(sycl::queue &q, const MoeShape &s, int max_segs, const bf16 *x, const MoeWeights &w, const MoeRowsScratch &sc) {
  const int H = s.hidden, I = s.inter_local(), tiles_i = I / kW, G = H / kGroup;
  const MoeBlobLayout lay = w.blob;
  const uint8_t *const *blobs = w.blobs;
  const int32_t *seg = sc.seg, *seg_n = sc.seg_n, *row = sc.row;
  bf16 *hbuf = sc.h;
  const size_t items = static_cast<size_t>(max_segs) * tiles_i;
  constexpr int kWg = 4;
  return q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for<MrGateUpBoth>(sycl::nd_range<1>((items + kWg - 1) / kWg * kWg, kWg), [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
      const int gid = static_cast<int>(it.get_global_id(0));
      const int sid = gid / tiles_i, j = gid % tiles_i;
      if (sid >= seg_n[0]) return;
      const esimd::simd<int32_t, 4> sp = esimd::block_load<int32_t, 4>(seg + static_cast<size_t>(sid) * 4);
      const int e = sp[0], p0 = sp[1], n = sp[2];
      const uint8_t *blob = blobs[e];
      const uint8_t *codes = blob + lay.gu_codes;
      const sycl::half *scales = reinterpret_cast<const sycl::half *>(blob + lay.gu_scales);
      int rr[kSeg];
#pragma unroll
      for (int r = 0; r < kSeg; ++r) rr[r] = row[p0 + (r < n ? r : n - 1)];
      esimd::simd<float, kSeg * kW> ag(0.f), au(0.f);
      const bool two = n > 8;
      for (int g = 0; g < G; ++g) {
        const auto rg = w4::load_block(codes + (static_cast<size_t>(g) * 2 * tiles_i + j) * kW4BlockBytes);
        const auto ru = w4::load_block(codes + (static_cast<size_t>(g) * 2 * tiles_i + tiles_i + j) * kW4BlockBytes);
        const esimd::simd<float, kW> sg = esimd::block_load<sycl::half, kW>(scales + static_cast<size_t>(g) * 2 * I + j * kW);
        const esimd::simd<float, kW> su = esimd::block_load<sycl::half, kW>(scales + static_cast<size_t>(g) * 2 * I + (tiles_i + j) * kW);
        esimd::simd<bf16, 8 * kGroup> ah0, ah1;
#pragma unroll
        for (int r = 0; r < 8; ++r) ah0.template select<kGroup, 1>(r * kGroup) = esimd::block_load<bf16, kGroup>(x + static_cast<size_t>(rr[r]) * H + g * kGroup);
        if (two)
#pragma unroll
          for (int r = 0; r < 8; ++r) ah1.template select<kGroup, 1>(r * kGroup) = esimd::block_load<bf16, kGroup>(x + static_cast<size_t>(rr[8 + r]) * H + g * kGroup);
        esimd::simd<float, 8 * kW> gg0(0.f), gu0(0.f), gg1(0.f), gu1(0.f);
#pragma unroll
        for (int jc = 0; jc < w4::kCalls; ++jc) {
          const esimd::simd<bf16, 2 * w4::kCallBytes> bg = w4::call_codes(rg, jc), bu = w4::call_codes(ru, jc);
          esimd::simd<bf16, 8 * w4::kKPerCall> at;
#pragma unroll
          for (int r = 0; r < 8; ++r) at.template select<w4::kKPerCall, 1>(r * w4::kKPerCall) = ah0.template select<w4::kKPerCall, 1>(r * kGroup + jc * w4::kKPerCall);
          gg0 = xmx::dpas<8, 8, float, float, bf16, bf16>(gg0, bg, at);
          gu0 = xmx::dpas<8, 8, float, float, bf16, bf16>(gu0, bu, at);
          if (two) {
#pragma unroll
            for (int r = 0; r < 8; ++r) at.template select<w4::kKPerCall, 1>(r * w4::kKPerCall) = ah1.template select<w4::kKPerCall, 1>(r * kGroup + jc * w4::kKPerCall);
            gg1 = xmx::dpas<8, 8, float, float, bf16, bf16>(gg1, bg, at);
            gu1 = xmx::dpas<8, 8, float, float, bf16, bf16>(gu1, bu, at);
          }
        }
        esimd::simd<float, 8 * kW> t = ag.template select<8 * kW, 1>(0);
        w4::apply_scale<8>(t, gg0, sg);
        ag.template select<8 * kW, 1>(0) = t;
        t = au.template select<8 * kW, 1>(0);
        w4::apply_scale<8>(t, gu0, su);
        au.template select<8 * kW, 1>(0) = t;
        if (two) {
          t = ag.template select<8 * kW, 1>(8 * kW);
          w4::apply_scale<8>(t, gg1, sg);
          ag.template select<8 * kW, 1>(8 * kW) = t;
          t = au.template select<8 * kW, 1>(8 * kW);
          w4::apply_scale<8>(t, gu1, su);
          au.template select<8 * kW, 1>(8 * kW) = t;
        }
      }
      const esimd::simd<float, kSeg * kW> gb = esimd::simd<bf16, kSeg * kW>(ag);
      const esimd::simd<float, kSeg * kW> ub = esimd::simd<bf16, kSeg * kW>(au);
      esimd::simd<bf16, kSeg * kW> hv = gb * sigmoid_v<kSeg * kW>(gb) * ub;
#pragma unroll
      for (int r = 0; r < kSeg; ++r) {
        if (r >= n) break;
        const esimd::simd<bf16, kW> v = hv.template select<kW, 1>(r * kW);
        v.copy_to(hbuf + static_cast<size_t>(p0 + r) * I + j * kW);
      }
    });
  });
}

// moe_rows' W4A16 down with one item per (segment, two adjacent 16-column tiles): each h row is
// loaded once per 64-k half group for both tiles (the one-tile kernel in moe_rows.cpp loads it per tile), the second 8-row
// half skipped on <= 8-row segments. Each tile's DPAS chain is the one-tile kernel's: the same bits.
// PK: the TP blob's down piece (moe_tp_piece): 64-k halves, or 32-k quarters at TP4's 160 columns per card.
template <int PK>
sycl::event down2(sycl::queue &q, const MoeShape &s, int max_segs, const MoeWeights &w, const MoeRowsScratch &sc) {
  constexpr int kPB = PK * kW4BlockBytes / kGroup, PG = kGroup / PK, kCB = w4::kCallBytes, kKC = w4::kKPerCall;
  const int H = s.hidden, I = s.inter_local(), tiles = H / kW, pairs = tiles / 2, NP = I / PK;
  const int pg0 = s.card * NP;
  const MoeBlobLayout lay = w.blob;
  const uint8_t *const *blobs = w.blobs;
  const int32_t *seg = sc.seg, *seg_n = sc.seg_n;
  const bf16 *hbuf = sc.h;
  bf16 *y = sc.y;
  constexpr int kWg = 8;
  const size_t items = static_cast<size_t>(max_segs) * pairs;
  return q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for<MrDown2<PK>>(sycl::nd_range<1>((items + kWg - 1) / kWg * kWg, kWg), [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
      const int gid = static_cast<int>(it.get_global_id(0));
      const int sid = gid / pairs, t0 = (gid % pairs) * 2;
      if (sid >= seg_n[0]) return;
      const esimd::simd<int32_t, 4> sp = esimd::block_load<int32_t, 4>(seg + static_cast<size_t>(sid) * 4);
      const int e = sp[0], p0 = sp[1], n = sp[2];
      const uint8_t *blob = blobs[e];
      const uint8_t *codes = blob + lay.dn_codes;
      const sycl::half *scales = reinterpret_cast<const sycl::half *>(blob + lay.dn_scales);
      esimd::simd<float, kSeg * kW> acc0(0.f), acc1(0.f);
      const int halves = n > 8 ? 2 : 1;
      for (int g = 0; g < NP; ++g) {
        esimd::simd<uint8_t, kPB> raw0 = esimd::block_load<uint8_t, kPB>(codes + (static_cast<size_t>(g) * tiles + t0) * kPB);
        esimd::simd<uint8_t, kPB> raw1 = esimd::block_load<uint8_t, kPB>(codes + (static_cast<size_t>(g) * tiles + t0 + 1) * kPB);
        const int srow = (pg0 + g) / PG - pg0 / PG;
        const esimd::simd<float, kW> sc0 = esimd::block_load<sycl::half, kW>(scales + static_cast<size_t>(srow) * H + t0 * kW);
        const esimd::simd<float, kW> sc1 = esimd::block_load<sycl::half, kW>(scales + static_cast<size_t>(srow) * H + (t0 + 1) * kW);
#pragma unroll
        for (int hb = 0; hb < 2; ++hb) {
          if (hb >= halves) break;
          esimd::simd<bf16, 8 * PK> a;
#pragma unroll
          for (int r = 0; r < 8; ++r) {
            const int rr = hb * 8 + r < n ? hb * 8 + r : n - 1;
            a.template select<PK, 1>(r * PK) = esimd::block_load<bf16, PK>(hbuf + static_cast<size_t>(p0 + rr) * I + g * PK);
          }
          esimd::simd<float, 8 * kW> s0(0.f), s1(0.f);
#pragma unroll
          for (int jc = 0; jc < PK / kKC; ++jc) {
            esimd::simd<bf16, 8 * kKC> at;
#pragma unroll
            for (int r = 0; r < 8; ++r) at.template select<kKC, 1>(r * kKC) = a.template select<kKC, 1>(r * PK + jc * kKC);
#pragma unroll
            for (int tt = 0; tt < 2; ++tt) {
              esimd::simd<uint8_t, kCB> bytes;
              if (tt) bytes = raw1.template select<kCB, 1>(jc * kCB);
              else bytes = raw0.template select<kCB, 1>(jc * kCB);
              const esimd::simd<int16_t, kCB> lo = esimd::simd<int16_t, kCB>(bytes & 15) - kW4Zero;
              const esimd::simd<int16_t, kCB> hi = esimd::simd<int16_t, kCB>(bytes >> 4) - kW4Zero;
              esimd::simd<bf16, 2 * kCB> b;
              b.template select<kCB, 2>(0) = esimd::simd<float, kCB>(lo);
              b.template select<kCB, 2>(1) = esimd::simd<float, kCB>(hi);
              if (tt) s1 = xmx::dpas<8, 8, float, float, bf16, bf16>(s1, b, at);
              else s0 = xmx::dpas<8, 8, float, float, bf16, bf16>(s0, b, at);
            }
          }
          esimd::simd<float, 8 * kW> ac = acc0.template select<8 * kW, 1>(hb * 8 * kW);
          w4::apply_scale<8>(ac, s0, sc0);
          acc0.template select<8 * kW, 1>(hb * 8 * kW) = ac;
          ac = acc1.template select<8 * kW, 1>(hb * 8 * kW);
          w4::apply_scale<8>(ac, s1, sc1);
          acc1.template select<8 * kW, 1>(hb * 8 * kW) = ac;
        }
      }
      constexpr auto hint = esimd::properties{esimd::cache_hint_L1<esimd::cache_hint::streaming>, esimd::cache_hint_L2<esimd::cache_hint::write_back>, esimd::alignment<32>};
#pragma unroll
      for (int r = 0; r < kSeg; ++r) {
        if (r >= n) break;
        esimd::simd<bf16, 2 * kW> v;
        v.template select<kW, 1>(0) = acc0.template select<kW, 1>(r * kW);
        v.template select<kW, 1>(kW) = acc1.template select<kW, 1>(r * kW);
        esimd::block_store<bf16, 2 * kW>(y + static_cast<size_t>(p0 + r) * H + t0 * kW, v, hint);  // y in bf16
      }
    });
  });
}


// ---- a work-group per block of <= kMoeBlkRows list rows of one expert ------------------------------------------------
// gate_up: TI threads (one per 16-column tile pair of this card's gate and up columns). Per 128-k group the block's x rows
// are staged once in SLM (double-buffered) as the DPAS A tiles [8-row block][call][8][16]; each thread loads its two int4
// tiles once, widens each call's codes once and runs four 8-row blocks x (gate, up) independent DPAS chains. Each (row,
// column) keeps gate_up_both's chain - per group the eight dpas<8, 8> calls from zero in k order, w4:apply_scale with the
// group's fp16 scale, bf16 g and u, the same swiglu - so h has the same bits.
template <int TI>
class MrGateUpBlkT;
template <int TI>
sycl::event gate_up_blk(sycl::queue &q, const MoeShape &s, int max_blks, const bf16 *x, const MoeWeights &w, const MoeRowsScratch &sc) {
  constexpr int NB = kMoeBlkRows / 8, R = kMoeBlkRows;
  constexpr int kABytes = R * kGroup * 2;
  constexpr int kMine = (R + TI - 1) / TI;
  const int H = s.hidden, I = s.inter_local(), G = H / kGroup;
  const MoeBlobLayout lay = w.blob;
  const uint8_t *const *blobs = w.blobs;
  const int32_t *blk = sc.blk, *blk_n = sc.blk_n, *row = sc.row;
  bf16 *hbuf = sc.h;
  return q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for<MrGateUpBlkT<TI>>(sycl::nd_range<1>(static_cast<size_t>(max_blks) * TI, TI), [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
      esimd::slm_init<2 * kABytes>();
      const int b = static_cast<int>(it.get_group(0)), j = static_cast<int>(it.get_local_id(0));
      if (b >= blk_n[0]) return;  // the whole work-group
      const esimd::simd<int32_t, 4> bp = esimd::block_load<int32_t, 4>(blk + static_cast<size_t>(b) * 4);
      const int e = bp[0], p0 = bp[1], n = bp[2];
      const int nrb = (n + 7) / 8;
      const uint8_t *blob = blobs[e];
      const uint8_t *codes = blob + lay.gu_codes;
      const sycl::half *scales = reinterpret_cast<const sycl::half *>(blob + lay.gu_scales);
      int rr[kMine];  // this thread's staged rows r = j, j + TI... below nrb x 8 (past n: the last row)
#pragma unroll
      for (int i = 0; i < kMine; ++i) {
        const int r = j + i * TI;
        rr[i] = r < nrb * 8 ? row[p0 + (r < n ? r : n - 1)] : -1;
      }
      auto stage = [&](int g, int buf) SYCL_ESIMD_FUNCTION {
#pragma unroll
        for (int i = 0; i < kMine; ++i) {
          const int r = j + i * TI;
          if (rr[i] < 0) break;
          esimd::simd<bf16, kGroup> a = esimd::block_load<bf16, kGroup>(x + static_cast<size_t>(rr[i]) * H + g * kGroup);
          const int rb = r / 8, rl = r % 8;
#pragma unroll
          for (int jc = 0; jc < w4::kCalls; ++jc)
            esimd::slm_block_store<bf16, w4::kKPerCall>(static_cast<uint32_t>(buf * kABytes + (((rb * w4::kCalls + jc) * 8 + rl) * w4::kKPerCall) * 2),
                                                         a.template select<w4::kKPerCall, 1>(jc * w4::kKPerCall));
        }
      };
      esimd::simd<float, NB * 8 * kW> ag(0.f), au(0.f);
      stage(0, 0);
      esimd::barrier();
      for (int g = 0; g < G; ++g) {
        if (g + 1 < G) stage(g + 1, (g + 1) & 1);
        const auto rg = w4::load_block(codes + (static_cast<size_t>(g) * 2 * TI + j) * kW4BlockBytes);
        const auto ru = w4::load_block(codes + (static_cast<size_t>(g) * 2 * TI + TI + j) * kW4BlockBytes);
        const esimd::simd<float, kW> sg = esimd::block_load<sycl::half, kW>(scales + static_cast<size_t>(g) * 2 * I + j * kW);
        const esimd::simd<float, kW> su = esimd::block_load<sycl::half, kW>(scales + static_cast<size_t>(g) * 2 * I + (TI + j) * kW);
        const uint32_t base = static_cast<uint32_t>((g & 1) * kABytes);
        esimd::simd<float, NB * 8 * kW> pg(0.f), pu(0.f);
#pragma unroll
        for (int jc = 0; jc < w4::kCalls; ++jc) {
          const esimd::simd<bf16, 2 * w4::kCallBytes> bg = w4::call_codes(rg, jc), bu = w4::call_codes(ru, jc);
#pragma unroll
          for (int rb = 0; rb < NB; ++rb) {
            if (rb >= nrb) break;
            const esimd::simd<bf16, 8 * w4::kKPerCall> at =
                esimd::slm_block_load<bf16, 8 * w4::kKPerCall>(base + static_cast<uint32_t>(((rb * w4::kCalls + jc) * 8) * w4::kKPerCall * 2));
            esimd::simd<float, 8 * kW> c = pg.template select<8 * kW, 1>(rb * 8 * kW);
            c = xmx::dpas<8, 8, float, float, bf16, bf16>(c, bg, at);
            pg.template select<8 * kW, 1>(rb * 8 * kW) = c;
            esimd::simd<float, 8 * kW> d = pu.template select<8 * kW, 1>(rb * 8 * kW);
            d = xmx::dpas<8, 8, float, float, bf16, bf16>(d, bu, at);
            pu.template select<8 * kW, 1>(rb * 8 * kW) = d;
          }
        }
#pragma unroll
        for (int rb = 0; rb < NB; ++rb) {
          if (rb >= nrb) break;
          esimd::simd<float, 8 * kW> t = ag.template select<8 * kW, 1>(rb * 8 * kW);
          w4::apply_scale<8>(t, pg.template select<8 * kW, 1>(rb * 8 * kW), sg);
          ag.template select<8 * kW, 1>(rb * 8 * kW) = t;
          t = au.template select<8 * kW, 1>(rb * 8 * kW);
          w4::apply_scale<8>(t, pu.template select<8 * kW, 1>(rb * 8 * kW), su);
          au.template select<8 * kW, 1>(rb * 8 * kW) = t;
        }
        esimd::barrier();  // group g's SLM reads are done and group g + 1 is staged
      }
#pragma unroll
      for (int rb = 0; rb < NB; ++rb) {
        if (rb >= nrb) break;
        const esimd::simd<float, 8 * kW> gb = esimd::simd<bf16, 8 * kW>(ag.template select<8 * kW, 1>(rb * 8 * kW));
        const esimd::simd<float, 8 * kW> ub = esimd::simd<bf16, 8 * kW>(au.template select<8 * kW, 1>(rb * 8 * kW));
        esimd::simd<bf16, 8 * kW> hv = gb * sigmoid_v<8 * kW>(gb) * ub;
#pragma unroll
        for (int r = 0; r < 8; ++r) {
          if (rb * 8 + r >= n) break;
          const esimd::simd<bf16, kW> v = hv.template select<kW, 1>(r * kW);
          v.copy_to(hbuf + static_cast<size_t>(p0 + rb * 8 + r) * I + j * kW);
        }
      }
    });
  });
}

// down: 16 threads per block. The block's h rows (32 x this card's k, bf16) are staged once in SLM as the DPAS A tiles
// [8-row block][k piece][call][8][16]; each thread walks the output tile pairs p = j, j + 16... and runs, per pair, two
// tiles x four 8-row blocks of independent DPAS chains, each call's codes widened once for the four blocks. Each (row,
// column) keeps down2's chain (per k piece: its calls from zero, w4:apply_scale with the piece's group scale) and its
// bf16 store: the same bits. NPMAX: the most k pieces per card (NP = I / PK at run time).
template <int PK, int NPMAX>
class MrDownBlk;
template <int PK, int NPMAX>
sycl::event down_blk(sycl::queue &q, const MoeShape &s, int max_blks, const MoeWeights &w, const MoeRowsScratch &sc) {
  constexpr int kPB = PK * kW4BlockBytes / kGroup, PG = kGroup / PK, kCB = w4::kCallBytes, kKC = w4::kKPerCall, kJC = PK / kKC;
  constexpr int NB = kMoeBlkRows / 8, R = kMoeBlkRows, T = 16;
  constexpr int kABytes = R * PK * NPMAX * 2;
  const int H = s.hidden, I = s.inter_local(), tiles = H / kW, pairs = tiles / 2, NP = I / PK;
  const int pg0 = s.card * NP;
  const MoeBlobLayout lay = w.blob;
  const uint8_t *const *blobs = w.blobs;
  const int32_t *blk = sc.blk, *blk_n = sc.blk_n;
  const bf16 *hbuf = sc.h;
  bf16 *y = sc.y;
  return q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for<MrDownBlk<PK, NPMAX>>(sycl::nd_range<1>(static_cast<size_t>(max_blks) * T, T), [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
      esimd::slm_init<kABytes>();
      const int b = static_cast<int>(it.get_group(0)), j = static_cast<int>(it.get_local_id(0));
      if (b >= blk_n[0]) return;
      const esimd::simd<int32_t, 4> bp = esimd::block_load<int32_t, 4>(blk + static_cast<size_t>(b) * 4);
      const int e = bp[0], p0 = bp[1], n = bp[2];
      const int nrb = (n + 7) / 8;
      const uint8_t *blob = blobs[e];
      const uint8_t *codes = blob + lay.dn_codes;
      const sycl::half *scales = reinterpret_cast<const sycl::half *>(blob + lay.dn_scales);
      // stage: row r (past n: the last row), piece g, call jc -> [rb][g][jc][rl][16]
      for (int r = j; r < nrb * 8; r += T) {
        const bf16 *src = hbuf + static_cast<size_t>(p0 + (r < n ? r : n - 1)) * I;
        const int rb = r / 8, rl = r % 8;
        for (int g = 0; g < NP; ++g) {
          esimd::simd<bf16, PK> a = esimd::block_load<bf16, PK>(src + g * PK);
#pragma unroll
          for (int jc = 0; jc < kJC; ++jc)
            esimd::slm_block_store<bf16, kKC>(static_cast<uint32_t>((((rb * NP + g) * kJC + jc) * 8 + rl) * kKC * 2), a.template select<kKC, 1>(jc * kKC));
        }
      }
      esimd::barrier();
      constexpr auto hint = esimd::properties{esimd::cache_hint_L1<esimd::cache_hint::streaming>, esimd::cache_hint_L2<esimd::cache_hint::write_back>, esimd::alignment<32>};
      auto widen = [&](esimd::simd<uint8_t, kPB> &raw, int jc) SYCL_ESIMD_FUNCTION {
        esimd::simd<uint8_t, kCB> bytes = raw.template select<kCB, 1>(jc * kCB);
        const esimd::simd<int16_t, kCB> lo = esimd::simd<int16_t, kCB>(bytes & 15) - kW4Zero;
        const esimd::simd<int16_t, kCB> hi = esimd::simd<int16_t, kCB>(bytes >> 4) - kW4Zero;
        esimd::simd<bf16, 2 * kCB> bb;
        bb.template select<kCB, 2>(0) = esimd::simd<float, kCB>(lo);
        bb.template select<kCB, 2>(1) = esimd::simd<float, kCB>(hi);
        return bb;
      };
      for (int pr = j; pr < pairs; pr += T) {
        const int t0 = pr * 2;
        esimd::simd<float, NB * 8 * kW> acc0(0.f), acc1(0.f);
        for (int g = 0; g < NP; ++g) {
          esimd::simd<uint8_t, kPB> raw0 = esimd::block_load<uint8_t, kPB>(codes + (static_cast<size_t>(g) * tiles + t0) * kPB);
          esimd::simd<uint8_t, kPB> raw1 = esimd::block_load<uint8_t, kPB>(codes + (static_cast<size_t>(g) * tiles + t0 + 1) * kPB);
          const int srow = (pg0 + g) / PG - pg0 / PG;
          const esimd::simd<float, kW> sc0 = esimd::block_load<sycl::half, kW>(scales + static_cast<size_t>(srow) * H + t0 * kW);
          const esimd::simd<float, kW> sc1 = esimd::block_load<sycl::half, kW>(scales + static_cast<size_t>(srow) * H + (t0 + 1) * kW);
          esimd::simd<float, NB * 8 * kW> s0(0.f), s1(0.f);
#pragma unroll
          for (int jc = 0; jc < kJC; ++jc) {
            const esimd::simd<bf16, 2 * kCB> bb0 = widen(raw0, jc), bb1 = widen(raw1, jc);
#pragma unroll
            for (int rb = 0; rb < NB; ++rb) {
              if (rb >= nrb) break;
              const esimd::simd<bf16, 8 * kKC> at = esimd::slm_block_load<bf16, 8 * kKC>(static_cast<uint32_t>((((rb * NP + g) * kJC + jc) * 8) * kKC * 2));
              esimd::simd<float, 8 * kW> c = s0.template select<8 * kW, 1>(rb * 8 * kW);
              c = xmx::dpas<8, 8, float, float, bf16, bf16>(c, bb0, at);
              s0.template select<8 * kW, 1>(rb * 8 * kW) = c;
              esimd::simd<float, 8 * kW> d = s1.template select<8 * kW, 1>(rb * 8 * kW);
              d = xmx::dpas<8, 8, float, float, bf16, bf16>(d, bb1, at);
              s1.template select<8 * kW, 1>(rb * 8 * kW) = d;
            }
          }
#pragma unroll
          for (int rb = 0; rb < NB; ++rb) {
            if (rb >= nrb) break;
            esimd::simd<float, 8 * kW> ac = acc0.template select<8 * kW, 1>(rb * 8 * kW);
            w4::apply_scale<8>(ac, s0.template select<8 * kW, 1>(rb * 8 * kW), sc0);
            acc0.template select<8 * kW, 1>(rb * 8 * kW) = ac;
            ac = acc1.template select<8 * kW, 1>(rb * 8 * kW);
            w4::apply_scale<8>(ac, s1.template select<8 * kW, 1>(rb * 8 * kW), sc1);
            acc1.template select<8 * kW, 1>(rb * 8 * kW) = ac;
          }
        }
#pragma unroll
        for (int r = 0; r < R; ++r) {
          if (r >= n) break;
          esimd::simd<bf16, 2 * kW> v;
          v.template select<kW, 1>(0) = acc0.template select<kW, 1>(r * kW);
          v.template select<kW, 1>(kW) = acc1.template select<kW, 1>(r * kW);
          esimd::block_store<bf16, 2 * kW>(y + static_cast<size_t>(p0 + r) * H + t0 * kW, v, hint);  // y in bf16
        }
      }
    });
  });
}

}  // namespace

sycl::event moe_rows_gate_up(sycl::queue &q, const MoeShape &s, int max_segs, const bf16 *x, const MoeWeights &w, const MoeRowsScratch &sc) {
  return gate_up_both(q, s, max_segs, x, w, sc);
}
sycl::event moe_rows_down2(sycl::queue &q, const MoeShape &s, int max_segs, const MoeWeights &w, const MoeRowsScratch &sc) {
  return moe_tp_piece(s.inter_local()) == 32 ? down2<32>(q, s, max_segs, w, sc) : down2<64>(q, s, max_segs, w, sc);
}

bool moe_rows_blk_shape(const MoeShape &s) {
  const int I = s.inter_local(), ti = I / kW, PK = moe_tp_piece(I);
  return I % kW == 0 && (ti == 8 || ti == 10 || ti == 16 || ti == 20) && s.hidden % 32 == 0 && s.hidden % kGroup == 0 && I % PK == 0 && I / PK <= 10;
}
sycl::event moe_rows_gate_up_blk(sycl::queue &q, const MoeShape &s, int max_blks, const bf16 *x, const MoeWeights &w, const MoeRowsScratch &sc) {
  switch (s.inter_local() / kW) {
    case 8: return gate_up_blk<8>(q, s, max_blks, x, w, sc);
    case 10: return gate_up_blk<10>(q, s, max_blks, x, w, sc);
    case 16: return gate_up_blk<16>(q, s, max_blks, x, w, sc);
    case 20: return gate_up_blk<20>(q, s, max_blks, x, w, sc);
    default: throw std::invalid_argument("moe_rows_gate_up_blk: tile pairs per card");
  }
}
sycl::event moe_rows_down_blk(sycl::queue &q, const MoeShape &s, int max_blks, const MoeWeights &w, const MoeRowsScratch &sc) {
  const int I = s.inter_local(), PK = moe_tp_piece(I), NP = I / PK;
  if (PK == 32 && NP <= 5) return down_blk<32, 5>(q, s, max_blks, w, sc);
  if (PK == 64 && NP <= 5) return down_blk<64, 5>(q, s, max_blks, w, sc);
  if (PK == 64 && NP <= 10) return down_blk<64, 10>(q, s, max_blks, w, sc);
  throw std::invalid_argument("moe_rows_down_blk: k pieces per card");
}

}  // namespace xenithium:kernels
