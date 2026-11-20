#include "xenithium/kernels/gemm/dense.hpp"

#include <algorithm>
#include <stdexcept>
#include <sycl/ext/intel/esimd.hpp>

#include "xenithium/kernels/tile/rot_quant.hpp"

namespace xenithium::kernels {
namespace {

namespace esimd = sycl::ext::intel::esimd;
namespace xmx = esimd::xmx;
using bf16 = sycl::ext::oneapi::bfloat16;

constexpr int kW = 16;                        // columns per tile = DPAS exec width
constexpr int kKCall = 16;                    // k per bf16 DPAS call
constexpr int kCallElems = kDenseCallElems;   // 256
constexpr int kGroup = 128;
constexpr int kMaxKSplit = 16, kMinItems = 512;

// One 16-k call's weights as bf16 (VNNI order).
template <DenseKind KIND>
inline esimd::simd<bf16, kCallElems> load_call(const void *w, size_t elem) SYCL_ESIMD_FUNCTION {
  if constexpr (KIND == DenseKind::bf16) {
    return esimd::block_load<bf16, kCallElems>(static_cast<const bf16 *>(w) + elem);
  } else {
    const esimd::simd<int8_t, kCallElems> c = esimd::block_load<int8_t, kCallElems>(static_cast<const int8_t *>(w) + elem);
    const esimd::simd<float, kCallElems> f = c;  // exact
    return f;
  }
}

// M: the kernel's row shape (kernels/rows.hpp: 4, 8 or 16). Every row's sum is formed alone in the
// systolic array over the same k order, so row r of an 8-row launch == row r of a 4-row one. 16 rows run
// as two 8-row halves of one work-group (launch()).
template <int M, DenseKind KIND, int KS, bool RQ, int MC>
class DenseM4;


// RQ: the launch also runs a RotQuantJob (rot_quant.hpp, 128-blocks) in extra work-groups
// after the GEMM's tiles: the decode MoE's x rotate + quantise rides in the router's launch.
template <int M, DenseKind KIND, int KS, bool RQ = false, int MC = M>
sycl::event launch(sycl::queue &q, const bf16 *xin, const DenseWeights &dw, float *yout, RotQuantJob rq = {}) {
  const int n = dw.n, k = dw.k, tiles = n / kW, groups = k / kGroup, tail = (k % kGroup) / kKCall;
  const void *w = dw.w;
  const float *scale = dw.scale;
  // The 16-row shape is two halves of 8 rows in one work-group (items [half][part]): each half runs the
  // 8-row body on its rows over the same k split, so row r's bits are its 8-row bits; the halves read the same weight
  // tiles at the same time (the second from L1). The 32-row shape is four such 8-row quarters (KS <= 16:
  // the work-group holds 4 KS items; at KS 32 dense_gemm runs two 16-row launches).
  constexpr int HS = MC > 16 ? 4 : MC > 8 ? 2 : 1;
  static_assert(MC % HS == 0 && MC / HS <= 8, "dense: 16- / 32-row shapes run 8-row halves / quarters");
  constexpr int kM = MC / HS;  // rows computed per item (y rows >= MC are not written)
  constexpr int kAcc = kM * kW;
  const size_t rq_nb = RQ ? static_cast<size_t>(rq.cols / kGroup) : 0, rq_n = RQ ? static_cast<size_t>(rq.rows) * rq_nb : 0;
  const int rq_wgs = RQ ? static_cast<int>((rq_n + KS * HS - 1) / (KS * HS)) : 0;
  return q.submit([&](sycl::handler &h) {
    h.parallel_for<DenseM4<M, KIND, KS, RQ, MC>>(sycl::nd_range<1>(static_cast<size_t>(tiles + rq_wgs) * KS * HS, KS * HS), [=](sycl::nd_item<1> it) [[intel::sycl_explicit_simd]] {
      if constexpr (KS > 1) esimd::slm_init<HS * KS * kAcc * sizeof(float)>();
      const int t = static_cast<int>(it.get_group(0)), lid = static_cast<int>(it.get_local_id(0)), half = lid / KS, part = lid % KS;
      const bf16 *x = xin + static_cast<size_t>(half) * kM * k;
      float *y = yout + static_cast<size_t>(half) * kM * n;
      const uint32_t slm0 = static_cast<uint32_t>(half * KS * kAcc * sizeof(float));
      if constexpr (RQ) {
        if (t >= tiles) {  // uniform per work-group
          const size_t id = static_cast<size_t>(t - tiles) * KS * HS + lid;
          if (id < rq_n) {
            if (rq.rot) rot_quant_item<kGroup, true>(rq.src, rq.ld, rq.dst, rq.scl, id, rq_nb);
            else rot_quant_item<kGroup, false>(rq.src, rq.ld, rq.dst, rq.scl, id, rq_nb);
          }
          return;
        }
      }
      const int g0 = part * groups / KS, g1 = (part + 1) * groups / KS;
      esimd::simd<float, kAcc> acc(0.f);
      for (int g = g0; g < g1; ++g) {
        if constexpr (KIND != DenseKind::i2) {
          // The next group's weight tiles into L1 / L2 while this group computes (no registers held).
          if (g + 1 < g1) {
            constexpr int kEB = KIND == DenseKind::bf16 ? 2 : 1, kBytes = kDenseCalls * kCallElems * kEB;
            const uint8_t *nb = static_cast<const uint8_t *>(w) + ((static_cast<size_t>(g) + 1) * tiles + t) * (kDenseCalls * kCallElems) * kEB;
#pragma unroll
            for (int j = 0; j < kBytes / 256; ++j)
              esimd::prefetch<uint32_t, 64>(reinterpret_cast<const uint32_t *>(nb + j * 256),
                                            esimd::properties{esimd::cache_hint_L1<esimd::cache_hint::cached>, esimd::cache_hint_L2<esimd::cache_hint::cached>});
          }
        }
        esimd::simd<bf16, kM * kGroup> a;
#pragma unroll
        for (int r = 0; r < kM; ++r)
          a.template select<kGroup, 1>(r * kGroup) = esimd::block_load<bf16, kGroup>(x + static_cast<size_t>(r) * k + g * kGroup);
        const size_t base = (static_cast<size_t>(g) * tiles + t) * (kDenseCalls * kCallElems);
        if constexpr (KIND == DenseKind::i2) {
          // the group's sum alone, then its 16 column scales
          esimd::simd<float, kAcc> ag(0.f);
#pragma unroll
          for (int j = 0; j < kDenseCalls; ++j) {
            esimd::simd<bf16, kM * kKCall> at;
#pragma unroll
            for (int r = 0; r < kM; ++r) at.template select<kKCall, 1>(r * kKCall) = a.template select<kKCall, 1>(r * kGroup + j * kKCall);
            const esimd::simd<uint8_t, kCallElems / 4> b =
                esimd::block_load<uint8_t, kCallElems / 4>(static_cast<const uint8_t *>(w) + (base + j * kCallElems) / 4);
            esimd::simd<float, kCallElems> f;
#pragma unroll
            for (int p = 0; p < 4; ++p) {
              const esimd::simd<uint8_t, kCallElems / 4> c = (b >> static_cast<uint8_t>(2 * p)) & static_cast<uint8_t>(3);
              f.template select<kCallElems / 4, 4>(p) = esimd::simd<float, kCallElems / 4>(c) - 1.5f;
            }
            ag = xmx::dpas<8, kM, float, float, bf16, bf16>(ag, esimd::simd<bf16, kCallElems>(f), at);
          }
          const esimd::simd<float, kW> sv = esimd::block_load<float, kW>(scale + static_cast<size_t>(g) * n + t * kW);
#pragma unroll
          for (int r = 0; r < kM; ++r) acc.template select<kW, 1>(r * kW) += ag.template select<kW, 1>(r * kW) * sv;
        } else {
#pragma unroll
          for (int j = 0; j < kDenseCalls; ++j) {
            esimd::simd<bf16, kM * kKCall> at;
#pragma unroll
            for (int r = 0; r < kM; ++r) at.template select<kKCall, 1>(r * kKCall) = a.template select<kKCall, 1>(r * kGroup + j * kKCall);
            acc = xmx::dpas<8, kM, float, float, bf16, bf16>(acc, load_call<KIND>(w, base + j * kCallElems), at);
          }
        }
      }
      if (KIND != DenseKind::i2 && part == KS - 1) {  // i2: k % 128 == 0, no tail
        const size_t base = static_cast<size_t>(groups) * tiles * kDenseCalls * kCallElems + static_cast<size_t>(t) * tail * kCallElems;
        for (int j = 0; j < tail; ++j) {
          esimd::simd<bf16, kM * kKCall> at;
#pragma unroll
          for (int r = 0; r < kM; ++r)
            at.template select<kKCall, 1>(r * kKCall) =
                esimd::block_load<bf16, kKCall>(x + static_cast<size_t>(r) * k + groups * kGroup + j * kKCall);
          acc = xmx::dpas<8, kM, float, float, bf16, bf16>(acc, load_call<KIND>(w, base + j * kCallElems), at);
        }
      }
      auto store = [&](int r, esimd::simd<float, kW> v) SYCL_ESIMD_FUNCTION {
        if constexpr (KIND == DenseKind::i8) v *= esimd::block_load<float, kW>(scale + t * kW);
        v.copy_to(y + static_cast<size_t>(r) * n + t * kW);
      };
      if constexpr (KS == 1) {
#pragma unroll
        for (int r = 0; r < kM; ++r) store(r, acc.template select<kW, 1>(r * kW));
      } else {
        esimd::slm_block_store<float, kAcc>(slm0 + static_cast<uint32_t>(part * kAcc * sizeof(float)), acc);
        esimd::barrier();
        for (int r = part; r < kM; r += KS) {  // rows dealt round-robin; parts summed in order
          esimd::simd<float, kW> sum(0.f);
#pragma unroll
          for (int p = 0; p < KS; ++p) sum += esimd::slm_block_load<float, kW>(slm0 + static_cast<uint32_t>((p * kAcc + r * kW) * sizeof(float)));
          store(r, sum);
        }
      }
    });
  });
}

// The "C" form of the 16- / 32-row shapes for bf16 and int8 weights. One item per (tile, k part);
// each 16-k call's weights are loaded and widened once for R / 8 DPAS chains of 8 rows, A by one 2D block load of 16 k x R
// rows per call (rows >= live read 0 and are not stored). A row's sum is the same chain over the same operands in the
// same k order as in its 8-row launch, the parts summed in part order: the same bits. The halves / quarters form above
// loads and widens the tile once per half or quarter.
template <DenseKind KIND, int KS, int R>
class DenseC;
template <DenseKind KIND, int KS, int R>
sycl::event launch_c(sycl::queue &q, int live, const bf16 *xin, const DenseWeights &dw, float *yout) {
  static_assert(R == 16 || R == 32 || R == 64, "dense C form: 16, 32 or 64 rows");
  // 64 rows = eight 8-row chains on each widened call (one weight read for the step),
  // A in two 32-row 2D loads (the 2D block's height limit), the k parts reduced per 32-row half (SLM).
  const int n = dw.n, k = dw.k, tiles = n / kW, groups = k / kGroup, tail = (k % kGroup) / kKCall;
  const void *w = dw.w;
  const float *scale = dw.scale;
  constexpr int NCH = R / 8;
  return q.submit([&](sycl::handler &h) {
    h.parallel_for<DenseC<KIND, KS, R>>(sycl::nd_range<1>(static_cast<size_t>(tiles) * KS, KS), [=](sycl::nd_item<1> it) [[intel::sycl_explicit_simd]] {
      if constexpr (KS > 1) esimd::slm_init<KS * (R > 32 ? 32 : R) * kW * sizeof(float)>();
      const int t = static_cast<int>(it.get_group(0)), part = static_cast<int>(it.get_local_id(0));
      const auto *x16 = reinterpret_cast<const uint16_t *>(xin);
      const int g0 = part * groups / KS, g1 = (part + 1) * groups / KS;
      esimd::simd<float, 8 * kW> c0(0.f), c1(0.f), c2(0.f), c3(0.f), c4(0.f), c5(0.f), c6(0.f), c7(0.f);
      constexpr int kLR = R > 32 ? 32 : R;  // rows of one 2D A load
      auto call = [&](size_t elem, int k0) SYCL_ESIMD_FUNCTION {
        const esimd::simd<bf16, kCallElems> b = load_call<KIND>(w, elem);
        {
          esimd::simd<uint16_t, kLR * kKCall> u = esimd::load_2d<uint16_t, kKCall, kLR>(x16, k * 2 - 1, live - 1, k * 2 - 1, k0, 0);
          esimd::simd<bf16, kLR * kKCall> at = u.template bit_cast_view<bf16>().read();
          c0 = xmx::dpas<8, 8, float, float, bf16, bf16>(c0, b, at.template select<8 * kKCall, 1>(0).read());
          c1 = xmx::dpas<8, 8, float, float, bf16, bf16>(c1, b, at.template select<8 * kKCall, 1>(8 * kKCall).read());
          if constexpr (NCH > 2) {
            c2 = xmx::dpas<8, 8, float, float, bf16, bf16>(c2, b, at.template select<8 * kKCall, 1>(16 * kKCall).read());
            c3 = xmx::dpas<8, 8, float, float, bf16, bf16>(c3, b, at.template select<8 * kKCall, 1>(24 * kKCall).read());
          }
        }
        if constexpr (NCH > 4) {
          esimd::simd<uint16_t, kLR * kKCall> u = esimd::load_2d<uint16_t, kKCall, kLR>(x16, k * 2 - 1, live - 1, k * 2 - 1, k0, 32);
          esimd::simd<bf16, kLR * kKCall> at = u.template bit_cast_view<bf16>().read();
          c4 = xmx::dpas<8, 8, float, float, bf16, bf16>(c4, b, at.template select<8 * kKCall, 1>(0).read());
          c5 = xmx::dpas<8, 8, float, float, bf16, bf16>(c5, b, at.template select<8 * kKCall, 1>(8 * kKCall).read());
          c6 = xmx::dpas<8, 8, float, float, bf16, bf16>(c6, b, at.template select<8 * kKCall, 1>(16 * kKCall).read());
          c7 = xmx::dpas<8, 8, float, float, bf16, bf16>(c7, b, at.template select<8 * kKCall, 1>(24 * kKCall).read());
        }
      };
      for (int g = g0; g < g1; ++g) {
        if (g + 1 < g1) {  // the next group's weight tiles into L1 / L2 (as launch())
          constexpr int kEB = KIND == DenseKind::bf16 ? 2 : 1, kBytes = kDenseCalls * kCallElems * kEB;
          const uint8_t *nb = static_cast<const uint8_t *>(w) + ((static_cast<size_t>(g) + 1) * tiles + t) * (kDenseCalls * kCallElems) * kEB;
#pragma unroll
          for (int j = 0; j < kBytes / 256; ++j)
            esimd::prefetch<uint32_t, 64>(reinterpret_cast<const uint32_t *>(nb + j * 256),
                                          esimd::properties{esimd::cache_hint_L1<esimd::cache_hint::cached>, esimd::cache_hint_L2<esimd::cache_hint::cached>});
        }
        const size_t base = (static_cast<size_t>(g) * tiles + t) * (kDenseCalls * kCallElems);
#pragma unroll
        for (int j = 0; j < kDenseCalls; ++j) call(base + j * kCallElems, g * kGroup + j * kKCall);
      }
      if (part == KS - 1) {
        const size_t base = static_cast<size_t>(groups) * tiles * kDenseCalls * kCallElems + static_cast<size_t>(t) * tail * kCallElems;
        for (int j = 0; j < tail; ++j) call(base + j * kCallElems, groups * kGroup + j * kKCall);
      }
      auto store = [&](int r, esimd::simd<float, kW> v) SYCL_ESIMD_FUNCTION {
        if constexpr (KIND == DenseKind::i8) v *= esimd::block_load<float, kW>(scale + t * kW);
        v.copy_to(yout + static_cast<size_t>(r) * n + t * kW);
      };
      // rows [r0, r0 + HR) from four (or two) chains: stored, or the k parts summed in part order
      constexpr int HR = R > 32 ? 32 : R, kHAcc = HR * kW;
      auto finish = [&](int r0, const esimd::simd<float, 8 * kW> &a0, const esimd::simd<float, 8 * kW> &a1, const esimd::simd<float, 8 * kW> &a2,
                        const esimd::simd<float, 8 * kW> &a3) SYCL_ESIMD_FUNCTION {
        esimd::simd<float, kHAcc> acc;
        acc.template select<8 * kW, 1>(0) = a0;
        acc.template select<8 * kW, 1>(8 * kW) = a1;
        if constexpr (HR > 16) {
          acc.template select<8 * kW, 1>(16 * kW) = a2;
          acc.template select<8 * kW, 1>(24 * kW) = a3;
        }
        if constexpr (KS == 1) {
#pragma unroll
          for (int r = 0; r < HR; ++r)
            if (r0 + r < live) store(r0 + r, acc.template select<kW, 1>(r * kW));
        } else {
          if (r0 > 0) esimd::barrier();  // the last half's sums are read: the slots are free again
          esimd::slm_block_store<float, kHAcc>(static_cast<uint32_t>(part * kHAcc * sizeof(float)), acc);
          esimd::barrier();
          for (int r = part; r0 + r < live && r < HR; r += KS) {  // rows dealt round-robin; parts summed in order
            esimd::simd<float, kW> sum(0.f);
#pragma unroll
            for (int p = 0; p < KS; ++p) sum += esimd::slm_block_load<float, kW>(static_cast<uint32_t>((p * kHAcc + r * kW) * sizeof(float)));
            store(r0 + r, sum);
          }
        }
      };
      finish(0, c0, c1, c2, c3);
      if constexpr (NCH > 4) finish(32, c4, c5, c6, c7);
    });
  });
}
template <DenseKind KIND, int R>
sycl::event dispatch_c(sycl::queue &q, int live, const bf16 *x, const DenseWeights &w, float *y, int ks) {
  if (ks >= 16) return launch_c<KIND, 16, R>(q, live, x, w, y);
  if (ks >= 8) return launch_c<KIND, 8, R>(q, live, x, w, y);
  if (ks >= 4) return launch_c<KIND, 4, R>(q, live, x, w, y);
  if (ks >= 2) return launch_c<KIND, 2, R>(q, live, x, w, y);
  return launch_c<KIND, 1, R>(q, live, x, w, y);
}
// the 2D A loads want a 64-byte aligned base (the row pitch, 2 k bytes, is a multiple of 32)
bool a2d_ok(const bf16 *x) { return reinterpret_cast<uintptr_t>(x) % 64 == 0; }

template <int M, DenseKind KIND, int MC = M>
sycl::event dispatch(sycl::queue &q, const bf16 *x, const DenseWeights &w, float *y, int ks) {
  if (ks >= 32) return launch<M, KIND, 32, false, MC>(q, x, w, y);
  if (ks >= 16) return launch<M, KIND, 16, false, MC>(q, x, w, y);
  if (ks >= 8) return launch<M, KIND, 8, false, MC>(q, x, w, y);
  if (ks >= 4) return launch<M, KIND, 4, false, MC>(q, x, w, y);
  if (ks >= 2) return launch<M, KIND, 2, false, MC>(q, x, w, y);
  return launch<M, KIND, 1, false, MC>(q, x, w, y);
}
template <int M, DenseKind KIND>
sycl::event dispatch_rows(sycl::queue &q, int rows, const bf16 *x, const DenseWeights &w, float *y, int ks) {
  // A 1-3-row step computes only its live rows (DPAS repeat count = rows; row r's sum is formed
  // alone, so its bits do not change; the y rows past `rows` are not written).
  if constexpr (M == 4) {
    if (rows == 1) return dispatch<4, KIND, 1>(q, x, w, y, ks);
    if (rows == 2) return dispatch<4, KIND, 2>(q, x, w, y, ks);
    if (rows == 3) return dispatch<4, KIND, 3>(q, x, w, y, ks);
  }
  // The 8-row shape likewise for 5-7 rows (3 x 2 MTP-1 rows, 5-7 plain streams).
  if constexpr (M == 8) {
    if (rows == 5) return dispatch<8, KIND, 5>(q, x, w, y, ks);
    if (rows == 6) return dispatch<8, KIND, 6>(q, x, w, y, ks);
    if (rows == 7) return dispatch<8, KIND, 7>(q, x, w, y, ks);
  }
  return dispatch<M, KIND>(q, x, w, y, ks);
}
// bf16 weights only (the MoE router), with the extra rotate + quantise job.
template <int M>
sycl::event dispatch_rq(sycl::queue &q, const bf16 *x, const DenseWeights &w, float *y, int ks, const RotQuantJob &rq) {
  constexpr DenseKind B = DenseKind::bf16;
  if (ks >= 16) return launch<M, B, 16, true>(q, x, w, y, rq);
  if (ks >= 8) return launch<M, B, 8, true>(q, x, w, y, rq);
  if (ks >= 4) return launch<M, B, 4, true>(q, x, w, y, rq);
  if (ks >= 2) return launch<M, B, 2, true>(q, x, w, y, rq);
  return launch<M, B, 1, true>(q, x, w, y, rq);
}

template <int M>
sycl::event run(sycl::queue &q, const bf16 *x, const DenseWeights &w, float *y, int ksplit, int rows = M) {
  if (!w.valid()) throw std::invalid_argument("dense_gemm: n % 16, k % 16, weights (and int8 scales) required");
  if (ksplit <= 0) ksplit = dense_auto_ksplit(w.n, w.k);
  ksplit = std::max(1, std::min({ksplit, 32, std::max(1, w.k / kGroup)}));
  if (w.kind == DenseKind::i2) return dispatch<M, DenseKind::i2>(q, x, w, y, ksplit);
  return w.kind == DenseKind::i8 ? dispatch_rows<M, DenseKind::i8>(q, rows, x, w, y, ksplit) : dispatch_rows<M, DenseKind::bf16>(q, rows, x, w, y, ksplit);
}

}  // namespace

// dense_gemm's KS = 1 tile body (launch<M, KIND, 1, false, 1>) for one row and one listed tile per
// work-group. Keep every expression in step with launch() above: the bits are only equal while the bodies are.
template <DenseKind KIND>
class DenseTiles;
template <DenseKind KIND>
sycl::event launch_tiles(sycl::queue &q, int rows, const bf16 *x, const DenseWeights &dw, float *y, const int32_t *list, int per_row) {
  const int n = dw.n, k = dw.k, tiles = n / kW, groups = k / kGroup, tail = (k % kGroup) / kKCall;
  const void *w = dw.w;
  const float *scale = dw.scale;
  return q.submit([&](sycl::handler &h) {
    h.parallel_for<DenseTiles<KIND>>(sycl::nd_range<2>({static_cast<size_t>(rows), static_cast<size_t>(per_row)}, {1, 1}), [=](sycl::nd_item<2> it) [[intel::sycl_explicit_simd]] {
      // a (row, listed tile) grid: the row comes from the launch, not from a division by the run-time tile count in every item
      const int r = static_cast<int>(it.get_group(0)), gid = r * per_row + static_cast<int>(it.get_group(1));
      const int t = list[gid];
      const bf16 *xr = x + static_cast<size_t>(r) * k;
      esimd::simd<float, kW> acc(0.f);
      for (int g = 0; g < groups; ++g) {
        if (g + 1 < groups) {
          constexpr int kEB = KIND == DenseKind::bf16 ? 2 : 1, kBytes = kDenseCalls * kCallElems * kEB;
          const uint8_t *nb = static_cast<const uint8_t *>(w) + ((static_cast<size_t>(g) + 1) * tiles + t) * (kDenseCalls * kCallElems) * kEB;
#pragma unroll
          for (int j = 0; j < kBytes / 256; ++j)
            esimd::prefetch<uint32_t, 64>(reinterpret_cast<const uint32_t *>(nb + j * 256),
                                          esimd::properties{esimd::cache_hint_L1<esimd::cache_hint::cached>, esimd::cache_hint_L2<esimd::cache_hint::cached>});
        }
        esimd::simd<bf16, kGroup> a = esimd::block_load<bf16, kGroup>(xr + g * kGroup);
        const size_t base = (static_cast<size_t>(g) * tiles + t) * (kDenseCalls * kCallElems);
#pragma unroll
        for (int j = 0; j < kDenseCalls; ++j) {
          esimd::simd<bf16, kKCall> at = a.template select<kKCall, 1>(j * kKCall);
          acc = xmx::dpas<8, 1, float, float, bf16, bf16>(acc, load_call<KIND>(w, base + j * kCallElems), at);
        }
      }
      {
        const size_t base = static_cast<size_t>(groups) * tiles * kDenseCalls * kCallElems + static_cast<size_t>(t) * tail * kCallElems;
        for (int j = 0; j < tail; ++j) {
          esimd::simd<bf16, kKCall> at = esimd::block_load<bf16, kKCall>(xr + groups * kGroup + j * kKCall);
          acc = xmx::dpas<8, 1, float, float, bf16, bf16>(acc, load_call<KIND>(w, base + j * kCallElems), at);
        }
      }
      if constexpr (KIND == DenseKind::i8) acc *= esimd::block_load<float, kW>(scale + t * kW);
      acc.copy_to(y + static_cast<size_t>(r) * n + t * kW);
    });
  });
}

sycl::event dense_gemm_tiles(sycl::queue &q, int rows, const bf16 *x, const DenseWeights &w, float *y, const int32_t *tiles, int per_row) {
  if (rows < 1 || rows > kMaxRows || per_row < 1) throw std::invalid_argument("dense_gemm_tiles: rows 1..32, per_row >= 1");
  if (!w.valid() || w.kind == DenseKind::i2) throw std::invalid_argument("dense_gemm_tiles: bf16 / int8 weights");
  if (dense_auto_ksplit(w.n, w.k) != 1) throw std::invalid_argument("dense_gemm_tiles: the matrix's dense_gemm splits k (not the same bits)");
  return w.kind == DenseKind::i8 ? launch_tiles<DenseKind::i8>(q, rows, x, w, y, tiles, per_row) : launch_tiles<DenseKind::bf16>(q, rows, x, w, y, tiles, per_row);
}

int dense_auto_ksplit(int n, int k) {
  const int tiles = n / kW, groups = std::max(1, k / kGroup);
  int ks = 1;
  while (ks < kMaxKSplit && tiles * ks < kMinItems && ks * 2 <= groups) ks *= 2;
  return ks;
}

sycl::event dense_gemm_m4(sycl::queue &q, const bf16 *x, const DenseWeights &w, float *y, int ksplit) { return run<kDenseRows>(q, x, w, y, ksplit); }

sycl::event dense_gemm(sycl::queue &q, int rows, const bf16 *x, const DenseWeights &w, float *y, int ksplit) {
  if (rows < 1 || rows > kMaxShape) throw std::invalid_argument("dense_gemm: rows 1.." + std::to_string(kMaxShape) + ", got " + std::to_string(rows));
  // 33-64 rows in the C form at 64 rows (one weight read); else (i2, a k split above 16, an unaligned x) two
  // launches, rows 0-31 then the rest at its own shape. Each row's bits are its 8-row bits either way.
  if (rows > kStepRows32 && w.valid() && w.kind != DenseKind::i2 && a2d_ok(x)) {
    int ks = ksplit <= 0 ? dense_auto_ksplit(w.n, w.k) : ksplit;
    ks = std::max(1, std::min({ks, 32, std::max(1, w.k / kGroup)}));
    if (ks <= 16) return w.kind == DenseKind::i8 ? dispatch_c<DenseKind::i8, 64>(q, rows, x, w, y, ks) : dispatch_c<DenseKind::bf16, 64>(q, rows, x, w, y, ks);
  }
  if (rows > kStepRows32) {
    dense_gemm(q, kStepRows32, x, w, y, ksplit);
    return dense_gemm(q, rows - kStepRows32, x + static_cast<size_t>(kStepRows32) * w.k, w, y + static_cast<size_t>(kStepRows32) * w.n, ksplit);
  }
  // 9-16 rows run the 16-row shape on 16 rows (x readable and y writable for 16 rows);
  // 17-32 rows the 32-row shape on 32 rows (four 8-row quarters per work-group), or two launches (rows 0-15, then the
  // rest) where the k split is 32 (a work-group of 128 items) or the weights are i2. Row r's bits are its 8-row bits.
  const int sh = row_shape(rows);
  // 9-32 rows of bf16 / int8 weights in the C form (one weight load and widen per call for every
  // row chain) where the k split is at most 16 and x allows the 2D loads; only the live rows of y are written
  if (sh >= 16 && w.valid() && w.kind != DenseKind::i2 && a2d_ok(x)) {
    int ks = ksplit <= 0 ? dense_auto_ksplit(w.n, w.k) : ksplit;
    ks = std::max(1, std::min({ks, 32, std::max(1, w.k / kGroup)}));
    if (ks <= 16) {
      if (w.kind == DenseKind::i8) return sh == 16 ? dispatch_c<DenseKind::i8, 16>(q, rows, x, w, y, ks) : dispatch_c<DenseKind::i8, 32>(q, rows, x, w, y, ks);
      return sh == 16 ? dispatch_c<DenseKind::bf16, 16>(q, rows, x, w, y, ks) : dispatch_c<DenseKind::bf16, 32>(q, rows, x, w, y, ks);
    }
  }
  if (sh == 32) {
    int ks = ksplit <= 0 ? dense_auto_ksplit(w.n, w.k) : ksplit;
    ks = std::max(1, std::min({ks, 32, std::max(1, w.k / kGroup)}));
    if (ks <= 16 && w.kind != DenseKind::i2) return run<32>(q, x, w, y, ksplit, rows);
    run<16>(q, x, w, y, ksplit, 16);
    return dense_gemm(q, rows - 16, x + static_cast<size_t>(16) * w.k, w, y + static_cast<size_t>(16) * w.n, ksplit);
  }
  return sh == 4 ? run<4>(q, x, w, y, ksplit, rows) : sh == 8 ? run<8>(q, x, w, y, ksplit, rows) : run<16>(q, x, w, y, ksplit, rows);
}

sycl::event dense_gemm_rq(sycl::queue &q, int rows, const bf16 *x, const DenseWeights &w, float *y, const RotQuantJob &rq, int ksplit) {
  if (rows < 1 || rows > kStepRows32) throw std::invalid_argument("dense_gemm_rq: rows 1..32");  // the MoE router (qwen38)
  if (!w.valid() || w.kind != DenseKind::bf16) throw std::invalid_argument("dense_gemm_rq: bf16 weights only");
  if (!rq.src || !rq.dst || !rq.scl || rq.rows < 1 || rq.cols % kGroup || rq.ld < rq.cols) throw std::invalid_argument("dense_gemm_rq: job");
  if (ksplit <= 0) ksplit = dense_auto_ksplit(w.n, w.k);
  ksplit = std::max(1, std::min({ksplit, 16, std::max(1, w.k / kGroup)}));
  const int sh = row_shape(rows);
  if (sh == 32) return dispatch_rq<32>(q, x, w, y, ksplit, rq);  // four 8-row quarters (KS <= 16)
  return sh == 4 ? dispatch_rq<4>(q, x, w, y, ksplit, rq) : sh == 8 ? dispatch_rq<8>(q, x, w, y, ksplit, rq) : dispatch_rq<16>(q, x, w, y, ksplit, rq);
}

}  // namespace xenithium::kernels
