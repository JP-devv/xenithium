// Xenithium · the W4A8 decode GEMM: the step's int4 GEMMs on int8 activations, opt-in
// (w4a8_rows_set; w4a16_gemm.cpp dispatches a registered matrix here, 128 GRF). Private to src/kernels/gemm.
// quant: A bf16 [live][k] -> int8 codes [live][k] + one fp32 scale per (row, 128-k group): s = amax / 127, code =
// rne(a x 127 / amax) (IEEE divisions; a group of zeros: codes 0, scale 0), the prefill W4A8's quantisation.
// The body: one item per (tile, part); per 128-k group the int4 block reordered to s4 operands in registers
// (w4a8:to_s4), A's codes by one 2D load of 32 k x R rows per call, a chain of 8 rows per DPAS (s8 x s4: 4 calls per
// group, the bf16 body's 8), the group's int32 sums exact, folded in fp32 as sum x (the weights' group scale x the row's
// group scale), groups in k order, the k split's parts summed in part order: a row's bits do not depend on the step's
// rows, the instance or the launch. Instances of 4 / 8 / 16 / 24 / 32 rows (rows past the live ones read 0 and are not
// stored).
#pragma once

#include <cstdint>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

#include "w4a16_c32.hpp"
#include "xenithium/kernels/tile/w4_tile.hpp"
#include "xenithium/kernels/tile/w4a8_tile.hpp"

namespace xenithium::kernels::a8r {

namespace esimd = sycl::ext::intel::esimd;
namespace xmx = esimd::xmx;
using bf16 = sycl::ext::oneapi::bfloat16;

inline constexpr int kScaleLd = 32;  // the scales' layout: fp32 [k / 128][32 rows]

// the instance's rows for a step of `live` rows (1..32)
constexpr int rows_of(int live) { return live <= 4 ? 4 : live <= 8 ? 8 : live <= 16 ? 16 : live <= 24 ? 24 : 32; }

class Quant;
inline sycl::event quant(sycl::queue &q, int live, const bf16 *a, int k, int8_t *codes, float *scales) {
  const int ng = k / kW4Group;
  return q.submit([&](sycl::handler &h) {
    h.parallel_for<Quant>(sycl::range<1>(static_cast<size_t>(live) * ng), [=](sycl::id<1> id) [[intel::sycl_explicit_simd]] {
      const int r = static_cast<int>(id[0]) / ng, g = static_cast<int>(id[0]) % ng;
      const size_t at = static_cast<size_t>(r) * k + static_cast<size_t>(g) * kW4Group;
      const esimd::simd<float, kW4Group> x = esimd::block_load<bf16, kW4Group>(a + at);
      const float amax = esimd::hmax<float>(esimd::abs(x));
      float inv = 0.f, sv = 0.f;
      if (amax > 0.f) inv = esimd::div_ieee(127.f, amax), sv = esimd::div_ieee(amax, 127.f);
      esimd::simd<float, kW4Group> t = esimd::rnde<float, kW4Group>(x * inv);
      t = esimd::max(esimd::min(t, 127.f), -127.f);
      const esimd::simd<int8_t, kW4Group> cd = t;
      esimd::block_store<int8_t, kW4Group>(codes + at, cd);
      scales[static_cast<size_t>(g) * kScaleLd + r] = sv;
    });
  });
}

template <int R>
inline void groups(esimd::simd<float, R * kW4Width> &acc, const uint8_t *a8, const float *as, int k, int live, const uint8_t *w, const sycl::half *sc, int n,
                   int ntiles, int tile, int g0, int g1) SYCL_ESIMD_FUNCTION {
  static_assert(R == 4 || R == 8 || R == 16 || R == 24 || R == 32, "W4A8 rows: 4 / 8 / 16 / 24 / 32 rows");
  constexpr int NC = (R + 7) / 8, RC = R >= 8 ? 8 : R;  // chains, rows per chain
  constexpr int kOp = w4a8::kOp8;
  for (int g = g0; g < g1; ++g) {
    const esimd::simd<uint8_t, kW4BlockBytes> raw = w4::load_block(w + (static_cast<size_t>(g) * ntiles + tile) * kW4BlockBytes);
    esimd::simd<uint8_t, 4 * kOp> b = w4a8::to_s4<4>(raw);
    const esimd::simd<float, kW4Width> scale = esimd::block_load<sycl::half, kW4Width>(sc + static_cast<size_t>(g) * n + tile * kW4Width);
    const esimd::simd<float, kScaleLd> sa = esimd::block_load<float, kScaleLd>(as + static_cast<size_t>(g) * kScaleLd);
    esimd::simd<int32_t, R * kW4Width> gs(0);
#pragma unroll
    for (int cc = 0; cc < 4; ++cc) {
      esimd::simd<uint8_t, R * 32> at = esimd::load_2d<uint8_t, 32, R>(a8, k - 1, live - 1, k - 1, g * kW4Group + cc * 32, 0);
      const esimd::simd<uint8_t, kOp> bc = b.template select<kOp, 1>(cc * kOp).read();
#pragma unroll
      for (int ch = 0; ch < NC; ++ch) {
        auto gv = gs.template select<RC * kW4Width, 1>(ch * 8 * kW4Width);
        const esimd::simd<int32_t, RC * kW4Width> cur = gv;
        gv = xmx::dpas<8, RC, int32_t, int32_t, uint8_t, uint8_t, xmx::dpas_argument_type::s4, xmx::dpas_argument_type::s8, RC * kW4Width, kOp, RC * 32>(
            cur, bc, esimd::simd<uint8_t, RC * 32>(at.template select<RC * 32, 1>(ch * 8 * 32).read()));
      }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
      const esimd::simd<float, kW4Width> f = esimd::simd<int32_t, kW4Width>(gs.template select<kW4Width, 1>(r * kW4Width).read());
      acc.template select<kW4Width, 1>(r * kW4Width) += f * (scale * static_cast<float>(sa[r]));
    }
  }
}

template <int KS, int R>
class Gemm;
template <int KS, int R>
sycl::event launch(sycl::queue &q, int live, const uint8_t *a8, const float *as, const uint8_t *w, const sycl::half *sc, float *c, int n, int k) {
  const int ntiles = n / kW4Width, ng = k / kW4Group;
  constexpr int kAcc = R * kW4Width;
  return q.submit([&](sycl::handler &h) {
    h.parallel_for<Gemm<KS, R>>(sycl::nd_range<1>(static_cast<size_t>(ntiles) * KS, KS), [=](sycl::nd_item<1> it) [[intel::sycl_explicit_simd]] {
      if constexpr (KS > 1) esimd::slm_init<KS * kAcc * sizeof(float)>();
      const int tile = static_cast<int>(it.get_group(0)), part = static_cast<int>(it.get_local_id(0));
      esimd::simd<float, kAcc> acc(0.f);
      groups<R>(acc, a8, as, k, live, w, sc, n, ntiles, tile, part * ng / KS, (part + 1) * ng / KS);
      c32::store<KS, R>(acc, c + static_cast<size_t>(tile) * kW4Width, n, live, part);
    });
  });
}

// The gate | up GEMM with swiglu: c32:launch_glu's work-group (items [gate / up][part]) and epilogue on this body.
template <int KS, int R>
class Glu;
template <int KS, int R>
sycl::event launch_glu(sycl::queue &q, int live, const uint8_t *a8, const float *as, const uint8_t *w, const sycl::half *sc, const float *tab, bf16 *h, int n_half,
                       int k) {
  const int n = 2 * n_half, ntiles = n / kW4Width, htiles = n_half / kW4Width;
  const int ng = k / kW4Group;
  constexpr int kAcc = R * kW4Width, kItems = 2 * KS;
  return q.submit([&](sycl::handler &hd) {
    hd.parallel_for<Glu<KS, R>>(sycl::nd_range<1>(static_cast<size_t>(htiles) * kItems, kItems), [=](sycl::nd_item<1> it) [[intel::sycl_explicit_simd]] {
      esimd::slm_init<kItems * kAcc * sizeof(float)>();
      const int ht = static_cast<int>(it.get_group(0)), lid = static_cast<int>(it.get_local_id(0));
      const int half = lid / KS, part = lid % KS, tile = ht + half * htiles;
      esimd::simd<float, kAcc> acc(0.f);
      groups<R>(acc, a8, as, k, live, w, sc, n, ntiles, tile, part * ng / KS, (part + 1) * ng / KS);
      esimd::slm_block_store<float, kAcc>(static_cast<uint32_t>(lid * kAcc * sizeof(float)), acc);
      esimd::barrier();
      for (int row = lid; row < live; row += kItems) {
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
        hv.copy_to(h + static_cast<size_t>(row) * n_half + static_cast<size_t>(ht) * kW4Width);
      }
    });
  });
}

template <int R>
sycl::event gemm_r(sycl::queue &q, int live, const uint8_t *a8, const float *as, const uint8_t *w, const sycl::half *sc, float *c, int n, int k, int ks) {
  if (ks >= 16) return launch<16, R>(q, live, a8, as, w, sc, c, n, k);
  if (ks >= 8) return launch<8, R>(q, live, a8, as, w, sc, c, n, k);
  if (ks >= 4) return launch<4, R>(q, live, a8, as, w, sc, c, n, k);
  if (ks >= 2) return launch<2, R>(q, live, a8, as, w, sc, c, n, k);
  return launch<1, R>(q, live, a8, as, w, sc, c, n, k);
}
inline sycl::event gemm(sycl::queue &q, int live, const uint8_t *a8, const float *as, const uint8_t *w, const sycl::half *sc, float *c, int n, int k, int ks) {
  switch (rows_of(live)) {
    case 4: return gemm_r<4>(q, live, a8, as, w, sc, c, n, k, ks);
    case 8: return gemm_r<8>(q, live, a8, as, w, sc, c, n, k, ks);
    case 16: return gemm_r<16>(q, live, a8, as, w, sc, c, n, k, ks);
    case 24: return gemm_r<24>(q, live, a8, as, w, sc, c, n, k, ks);
    default: return gemm_r<32>(q, live, a8, as, w, sc, c, n, k, ks);
  }
}
template <int R>
sycl::event glu_r(sycl::queue &q, int live, const uint8_t *a8, const float *as, const uint8_t *w, const sycl::half *sc, const float *tab, bf16 *h, int n_half, int k,
                  int ks) {
  if (ks >= 16) return launch_glu<16, R>(q, live, a8, as, w, sc, tab, h, n_half, k);
  if (ks >= 8) return launch_glu<8, R>(q, live, a8, as, w, sc, tab, h, n_half, k);
  if (ks >= 4) return launch_glu<4, R>(q, live, a8, as, w, sc, tab, h, n_half, k);
  if (ks >= 2) return launch_glu<2, R>(q, live, a8, as, w, sc, tab, h, n_half, k);
  return launch_glu<1, R>(q, live, a8, as, w, sc, tab, h, n_half, k);
}
inline sycl::event glu(sycl::queue &q, int live, const uint8_t *a8, const float *as, const uint8_t *w, const sycl::half *sc, const float *tab, bf16 *h, int n_half,
                       int k, int ks) {
  switch (rows_of(live)) {
    case 4: return glu_r<4>(q, live, a8, as, w, sc, tab, h, n_half, k, ks);
    case 8: return glu_r<8>(q, live, a8, as, w, sc, tab, h, n_half, k, ks);
    case 16: return glu_r<16>(q, live, a8, as, w, sc, tab, h, n_half, k, ks);
    case 24: return glu_r<24>(q, live, a8, as, w, sc, tab, h, n_half, k, ks);
    default: return glu_r<32>(q, live, a8, as, w, sc, tab, h, n_half, k, ks);
  }
}

}  // namespace xenithium:kernels:a8r
