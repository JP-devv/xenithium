// Xenithium · int4 GPTQ group tile, shared by the W4A16 GEMM and the MoE decode op.
// One call = one (128-k group, 16-column tile) block of the packed layout written by
// pack_w4_gptq: 1024 bytes, byte (kg/2)*16 + lane, even k in the low nibble. Each
// 128-byte slice widens (q - 8, exact in bf16) into one bf16 VNNI B operand; eight
// bf16 DPAS calls give the group's fp32 sum, before the scale.
#pragma once

#include <functional>

#include <sycl/ext/intel/esimd.hpp>

#include "xenithium/kernels/gemm/w4a16_gemm.hpp"

namespace xenithium::kernels::w4 {

namespace esimd = sycl::ext::intel::esimd;
namespace xmx = esimd::xmx;
using bf16 = sycl::ext::oneapi::bfloat16;

inline constexpr int kKPerCall = 16;  // bf16 DPAS: systolic depth 8 x 2
inline constexpr int kCalls = kW4Group / kKPerCall;  // 8 per group
inline constexpr int kCallBytes = kW4BlockBytes / kCalls;  // 128 packed bytes -> 256 bf16

// Loads the 1024-byte block at `wp` (must be 1024-aligned within a 4 KiB-aligned
// buffer, which the packed layout guarantees).
inline esimd::simd<uint8_t, kW4BlockBytes> load_block(const uint8_t *wp) SYCL_ESIMD_FUNCTION {
  esimd::simd<uint8_t, kW4BlockBytes> raw;
  raw.template select<512, 1>(0) = esimd::block_load<uint8_t, 512>(wp);
  raw.template select<512, 1>(512) = esimd::block_load<uint8_t, 512>(wp + 512);
  return raw;
}

// Call j's B operand (q - 8 as bf16, VNNI): what group_dot widens per call (for callers that share it
// between several DPAS chains).
inline esimd::simd<bf16, 2 * kCallBytes> call_codes(esimd::simd<uint8_t, kW4BlockBytes> raw, int j) SYCL_ESIMD_FUNCTION {
  const esimd::simd<uint8_t, kCallBytes> bytes = raw.template select<kCallBytes, 1>(j * kCallBytes);
  const esimd::simd<int16_t, kCallBytes> lo = esimd::simd<int16_t, kCallBytes>(bytes & 15) - kW4Zero;
  const esimd::simd<int16_t, kCallBytes> hi = esimd::simd<int16_t, kCallBytes>(bytes >> 4) - kW4Zero;
  esimd::simd<bf16, 2 * kCallBytes> b;
  b.template select<kCallBytes, 2>(0) = esimd::simd<float, kCallBytes>(lo);
  b.template select<kCallBytes, 2>(1) = esimd::simd<float, kCallBytes>(hi);
  return b;
}

// gsum (M x 16 fp32, row-major) = A (M x 128 bf16, row-major) * (q - 8) for one block.
template <int M>
// By value: ESIMD `select` needs a non-const lvalue; the copies vanish when inlined.
inline esimd::simd<float, M * kW4Width> group_dot(esimd::simd<uint8_t, kW4BlockBytes> raw,
                                                  esimd::simd<bf16, M * kW4Group> a_group)
    SYCL_ESIMD_FUNCTION {
  esimd::simd<float, M * kW4Width> gsum(0.f);
#pragma unroll
  for (int j = 0; j < kCalls; ++j) {
    const esimd::simd<uint8_t, kCallBytes> bytes = raw.template select<kCallBytes, 1>(j * kCallBytes);
    const esimd::simd<int16_t, kCallBytes> lo = esimd::simd<int16_t, kCallBytes>(bytes & 15) - kW4Zero;
    const esimd::simd<int16_t, kCallBytes> hi = esimd::simd<int16_t, kCallBytes>(bytes >> 4) - kW4Zero;
    esimd::simd<bf16, 2 * kCallBytes> b;
    b.template select<kCallBytes, 2>(0) = esimd::simd<float, kCallBytes>(lo);
    b.template select<kCallBytes, 2>(1) = esimd::simd<float, kCallBytes>(hi);
    esimd::simd<bf16, M * kKPerCall> a_tile;
#pragma unroll
    for (int row = 0; row < M; ++row) {
      a_tile.template select<kKPerCall, 1>(row * kKPerCall) =
          a_group.template select<kKPerCall, 1>(row * kW4Group + j * kKPerCall);
    }
    gsum = xmx::dpas<8, M, float, float, bf16, bf16>(gsum, b, a_tile);
  }
  return gsum;
}

// Zero points, group 32 (pack.hpp kW4ZpGroup): one block = four 32-k groups of two calls each. Group s widens q - z
// (z per column from the block's zero bytes zb: [2][16], group s in the low / high nibble of half s / 2) and its fp32
// sum takes the group's scale sc[s] (16 columns): acc[row] += sum_s gsum_s[row] x sc[s].
template <int M>
inline void block_dot_zp(esimd::simd<float, M * kW4Width> &acc, esimd::simd<uint8_t, kW4BlockBytes> raw, esimd::simd<bf16, M * kW4Group> a_group,
                         esimd::simd<uint8_t, 2 * kW4Width> zb, esimd::simd<float, 4 * kW4Width> sc) SYCL_ESIMD_FUNCTION {
#pragma unroll
  for (int s = 0; s < 4; ++s) {
    const esimd::simd<uint8_t, kW4Width> zh = zb.template select<kW4Width, 1>((s >> 1) * kW4Width);
    const esimd::simd<int16_t, kW4Width> z = (s & 1) ? esimd::simd<int16_t, kW4Width>(zh >> 4) : esimd::simd<int16_t, kW4Width>(zh & 15);
    const esimd::simd<int16_t, kCallBytes> zr = z.template replicate<kCallBytes / kW4Width>();
    esimd::simd<float, M * kW4Width> gsum(0.f);
#pragma unroll
    for (int jj = 0; jj < 2; ++jj) {
      const int j = 2 * s + jj;
      const esimd::simd<uint8_t, kCallBytes> bytes = raw.template select<kCallBytes, 1>(j * kCallBytes);
      const esimd::simd<int16_t, kCallBytes> lo = esimd::simd<int16_t, kCallBytes>(bytes & 15) - zr;
      const esimd::simd<int16_t, kCallBytes> hi = esimd::simd<int16_t, kCallBytes>(bytes >> 4) - zr;
      esimd::simd<bf16, 2 * kCallBytes> b;
      b.template select<kCallBytes, 2>(0) = esimd::simd<float, kCallBytes>(lo);
      b.template select<kCallBytes, 2>(1) = esimd::simd<float, kCallBytes>(hi);
      esimd::simd<bf16, M * kKPerCall> a_tile;
#pragma unroll
      for (int row = 0; row < M; ++row)
        a_tile.template select<kKPerCall, 1>(row * kKPerCall) = a_group.template select<kKPerCall, 1>(row * kW4Group + j * kKPerCall);
      gsum = xmx::dpas<8, M, float, float, bf16, bf16>(gsum, b, a_tile);
    }
    const esimd::simd<float, kW4Width> scale = sc.template select<kW4Width, 1>(s * kW4Width);
#pragma unroll
    for (int row = 0; row < M; ++row)
      acc.template select<kW4Width, 1>(row * kW4Width) += gsum.template select<kW4Width, 1>(row * kW4Width) * scale;
  }
}

// Affine, group 64 (pack.hpp kW4AffGroup): one block = two 64-k groups of four calls each. Group s's fp32 sum of
// q x a (q 0..15 raw, exact in bf16) takes the group's scale sc[s] and the row's fp32 sum of a over the group its bias
// bs[s] (16 columns each): acc[row] += gsum_s[row] x sc[s] + asum_s[row] x bs[s].
template <int M>
inline void block_dot_aff(esimd::simd<float, M * kW4Width> &acc, esimd::simd<uint8_t, kW4BlockBytes> raw, esimd::simd<bf16, M * kW4Group> a_group,
                          esimd::simd<float, 2 * kW4Width> sc, esimd::simd<float, 2 * kW4Width> bs) SYCL_ESIMD_FUNCTION {
#pragma unroll
  for (int s = 0; s < 2; ++s) {
    esimd::simd<float, M * kW4Width> gsum(0.f);
#pragma unroll
    for (int jj = 0; jj < 4; ++jj) {
      const int j = 4 * s + jj;
      const esimd::simd<uint8_t, kCallBytes> bytes = raw.template select<kCallBytes, 1>(j * kCallBytes);
      esimd::simd<bf16, 2 * kCallBytes> b;
      b.template select<kCallBytes, 2>(0) = esimd::simd<float, kCallBytes>(bytes & 15);
      b.template select<kCallBytes, 2>(1) = esimd::simd<float, kCallBytes>(bytes >> 4);
      esimd::simd<bf16, M * kKPerCall> a_tile;
#pragma unroll
      for (int row = 0; row < M; ++row)
        a_tile.template select<kKPerCall, 1>(row * kKPerCall) = a_group.template select<kKPerCall, 1>(row * kW4Group + j * kKPerCall);
      gsum = xmx::dpas<8, M, float, float, bf16, bf16>(gsum, b, a_tile);
    }
    const esimd::simd<float, kW4Width> scale = sc.template select<kW4Width, 1>(s * kW4Width);
    const esimd::simd<float, kW4Width> bias = bs.template select<kW4Width, 1>(s * kW4Width);
#pragma unroll
    for (int row = 0; row < M; ++row) {
      const esimd::simd<float, 64> av = a_group.template select<64, 1>(row * kW4Group + s * 64);
      const float asum = esimd::reduce<float>(av, std::plus<>());
      acc.template select<kW4Width, 1>(row * kW4Width) += gsum.template select<kW4Width, 1>(row * kW4Width) * scale + bias * asum;
    }
  }
}

// acc[row] += gsum[row] * scale (per column), the once-per-group fp16 scale.
template <int M>
inline void apply_scale(esimd::simd<float, M * kW4Width> &acc, esimd::simd<float, M * kW4Width> gsum,
                        esimd::simd<float, kW4Width> scale) SYCL_ESIMD_FUNCTION {
#pragma unroll
  for (int row = 0; row < M; ++row) {
    acc.template select<kW4Width, 1>(row * kW4Width) += gsum.template select<kW4Width, 1>(row * kW4Width) * scale;
  }
}

}  // namespace xenithium:kernels:w4
