// Xenithium · the W4A16 decode GEMM at 17-32 rows: the C32 body and its launches (w4a16_gemm.cpp, 128 GRF). Private to
// src/kernels/gemm.
// C32: one item per (tile, part); each call's B widened once for up to four 8-row DPAS chains (rows 0-7, 8-15, 16-23, 24-31),
// A by one 2D block load of 16 k x R rows (rows >= live read 0 and are not stored). Every row's group sum is its own
// chain over the same widened B and k order, the k split the 8-row one: row r's bits are its 8-row bits.
// R: the rows an instance computes, the step's live rows rounded up to 20 / 24 / 32: the third chain on 4 or 8 rows and
// no fourth one for 17-24 live rows (9-12 sequences x 2 rows, 5-6 x 4, 3 x 8). The same chains on the same operands, so
// the bits do not change (bf16 DPAS is exact at every repeat count).
#pragma once

#include <cstdint>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

#include "xenithium/kernels/tile/w4_tile.hpp"

namespace xenithium::kernels::c32 {

namespace esimd = sycl::ext::intel::esimd;
namespace xmx = esimd::xmx;
using bf16 = sycl::ext::oneapi::bfloat16;

// the instance's rows for a step of `live` rows (17..32)
constexpr int rows_of(int live) { return live <= 20 ? 20 : live <= 24 ? 24 : 32; }

template <int U, int R>
inline void groups(esimd::simd<float, R * kW4Width> &acc, const bf16 *a, int k, int live, const uint8_t *w, const sycl::half *sc, int n, int ntiles, int tile,
                   int g0, int g1) SYCL_ESIMD_FUNCTION {
  static_assert(R == 20 || R == 24 || R == 32, "C32: 20 / 24 / 32 rows");
  constexpr int kK = w4::kKPerCall;
  constexpr int R2 = R >= 24 ? 8 : 4;  // rows of the third chain
  constexpr int R3 = R == 32 ? 8 : 0;  // rows of the fourth chain (0: none)
  constexpr int S3 = R3 > 0 ? R3 : 1;
  const auto *a16 = reinterpret_cast<const uint16_t *>(a);
  for (int g = g0; g < g1; ++g) {
    const auto raw = w4::load_block(w + (static_cast<size_t>(g) * ntiles + tile) * kW4BlockBytes);
    const esimd::simd<float, kW4Width> scale = esimd::block_load<sycl::half, kW4Width>(sc + static_cast<size_t>(g) * n + tile * kW4Width);
    esimd::simd<float, 8 * kW4Width> s0(0.f), s1(0.f);
    esimd::simd<float, R2 * kW4Width> s2(0.f);
    esimd::simd<float, S3 * kW4Width> s3(0.f);
    auto call = [&](int j) SYCL_ESIMD_FUNCTION {
      const auto b = w4::call_codes(raw, j);
      esimd::simd<uint16_t, R * kK> u = esimd::load_2d<uint16_t, kK, R>(a16, k * 2 - 1, live - 1, k * 2 - 1, g * kW4Group + j * kK, 0);
      esimd::simd<bf16, R * kK> at = u.template bit_cast_view<bf16>().read();
      s0 = xmx::dpas<8, 8, float, float, bf16, bf16>(s0, b, at.template select<8 * kK, 1>(0).read());
      s1 = xmx::dpas<8, 8, float, float, bf16, bf16>(s1, b, at.template select<8 * kK, 1>(8 * kK).read());
      s2 = xmx::dpas<8, R2, float, float, bf16, bf16>(s2, b, at.template select<R2 * kK, 1>(16 * kK).read());
      if constexpr (R3 > 0) s3 = xmx::dpas<8, R3, float, float, bf16, bf16>(s3, b, at.template select<R3 * kK, 1>(24 * kK).read());
    };
#pragma unroll U
    for (int j = 0; j < w4::kCalls; ++j) call(j);
#pragma unroll
    for (int row = 0; row < 8; ++row) {
      acc.template select<kW4Width, 1>(row * kW4Width) += s0.template select<kW4Width, 1>(row * kW4Width) * scale;
      acc.template select<kW4Width, 1>((8 + row) * kW4Width) += s1.template select<kW4Width, 1>(row * kW4Width) * scale;
    }
#pragma unroll
    for (int row = 0; row < R2; ++row) acc.template select<kW4Width, 1>((16 + row) * kW4Width) += s2.template select<kW4Width, 1>(row * kW4Width) * scale;
    if constexpr (R3 > 0) {
#pragma unroll
      for (int row = 0; row < R3; ++row) acc.template select<kW4Width, 1>((24 + row) * kW4Width) += s3.template select<kW4Width, 1>(row * kW4Width) * scale;
    }
  }
}

// The live rows of a tile's KS parts (w4a16_gemm.cpp's store_rows at R rows): KS 1 the item's own accumulator, else
// summed through SLM in part order.
template <int KS, int R>
inline void store(esimd::simd<float, R * kW4Width> &acc, float *out_tile, int n, int live, int part) SYCL_ESIMD_FUNCTION {
  constexpr int kAcc = R * kW4Width;
  if constexpr (KS == 1) {
#pragma unroll
    for (int row = 0; row < R; ++row)
      if (row < live) {
        const esimd::simd<float, kW4Width> o = acc.template select<kW4Width, 1>(row * kW4Width);
        o.copy_to(out_tile + static_cast<size_t>(row) * n);
      }
  } else {
    esimd::slm_block_store<float, kAcc>(static_cast<uint32_t>(part * kAcc * sizeof(float)), acc);
    esimd::barrier();
    for (int row = part; row < live; row += KS) {
      esimd::simd<float, kW4Width> sum(0.f);
#pragma unroll
      for (int p = 0; p < KS; ++p) sum += esimd::slm_block_load<float, kW4Width>(static_cast<uint32_t>((p * kAcc + row * kW4Width) * sizeof(float)));
      sum.copy_to(out_tile + static_cast<size_t>(row) * n);
    }
  }
}

// U: the call-loop unroll (4 at 128 GRF).
template <int KS, int R>
class Gemm;
template <int KS, int U, int R>
sycl::event launch(sycl::queue &q, int live, const bf16 *a, const uint8_t *w, const sycl::half *sc, float *c, int n, int k) {
  const int ntiles = n / kW4Width, ng = k / kW4Group;
  constexpr int kAcc = R * kW4Width;
  return q.submit([&](sycl::handler &h) {
    h.parallel_for<Gemm<KS, R>>(sycl::nd_range<1>(static_cast<size_t>(ntiles) * KS, KS), [=](sycl::nd_item<1> it) [[intel::sycl_explicit_simd]] {
      if constexpr (KS > 1) esimd::slm_init<KS * kAcc * sizeof(float)>();
      const int tile = static_cast<int>(it.get_group(0)), part = static_cast<int>(it.get_local_id(0));
      esimd::simd<float, kAcc> acc(0.f);
      groups<U, R>(acc, a, k, live, w, sc, n, ntiles, tile, part * ng / KS, (part + 1) * ng / KS);
      store<KS, R>(acc, c + static_cast<size_t>(tile) * kW4Width, n, live, part);
    });
  });
}

// The gate | up GEMM with swiglu at 17-32 rows: launch_glu16's work-group (items [gate / up][part]) on the C32 body;
// the epilogue launch_glu's (h = bf16(silu(bf16 gate) x bf16 up) through the silu table).
template <int KS, int R>
class Glu;
template <int KS, int U, int R>
sycl::event launch_glu(sycl::queue &q, int live, const bf16 *a, const uint8_t *w, const sycl::half *sc, const float *tab, bf16 *h, int n_half, int k) {
  const int n = 2 * n_half, ntiles = n / kW4Width, htiles = n_half / kW4Width;
  const int ng = k / kW4Group;
  constexpr int kAcc = R * kW4Width, kItems = 2 * KS;
  return q.submit([&](sycl::handler &hd) {
    hd.parallel_for<Glu<KS, R>>(sycl::nd_range<1>(static_cast<size_t>(htiles) * kItems, kItems), [=](sycl::nd_item<1> it) [[intel::sycl_explicit_simd]] {
      esimd::slm_init<kItems * kAcc * sizeof(float)>();
      const int ht = static_cast<int>(it.get_group(0)), lid = static_cast<int>(it.get_local_id(0));
      const int half = lid / KS, part = lid % KS, tile = ht + half * htiles;
      esimd::simd<float, kAcc> acc(0.f);
      groups<U, R>(acc, a, k, live, w, sc, n, ntiles, tile, part * ng / KS, (part + 1) * ng / KS);
      esimd::slm_block_store<float, kAcc>(static_cast<uint32_t>(lid * kAcc * sizeof(float)), acc);
      esimd::barrier();
      for (int row = lid; row < live; row += kItems) {
        esimd::simd<float, kW4Width> gs, us;
        if constexpr (KS == 1) {
          gs = esimd::slm_block_load<float, kW4Width>(static_cast<uint32_t>(row * kW4Width * sizeof(float)));
          us = esimd::slm_block_load<float, kW4Width>(static_cast<uint32_t>((kAcc + row * kW4Width) * sizeof(float)));
        } else {
          gs = 0.f;
          us = 0.f;
#pragma unroll
          for (int p = 0; p < KS; ++p) {
            gs += esimd::slm_block_load<float, kW4Width>(static_cast<uint32_t>((p * kAcc + row * kW4Width) * sizeof(float)));
            us += esimd::slm_block_load<float, kW4Width>(static_cast<uint32_t>(((KS + p) * kAcc + row * kW4Width) * sizeof(float)));
          }
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

}  // namespace xenithium:kernels:c32
