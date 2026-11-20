// Xenithium · FP8 E4M3 weight tiles on XMX (Xe2 DPAS has no FP8 input): W8A16.
// Layout: the bf16 tile layout (bf16_tile.hpp) with one byte per element - block
// (128-k group g, 16-column tile t) is 2048 bytes at (g * tiles + t) * 2048, 8 DPAS
// calls of 16 k x 16 columns in VNNI order (pack_fp8_tiles in dense_bf16.hpp).
// Widening is exact: an E4M3 byte b shifted into fp16 position,
// h = (int16(int8(b)) << 7) & 0xBF80 (sign -> bit 15, eeee mmm -> bits 13..7)
// is the fp16 number w * 2^-8 (normal and subnormal codes alike: fp16 has 1 more
// exponent bit and 7 more mantissa bits), and fp16 -> fp32 -> bf16 keeps it exact
// (bf16 holds every value with <= 4 significant bits in [2^-17, 2^0]). The 2^8 is
// folded with the per-output-channel scale into one multiply of the fp32 sum.
#pragma once

#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

#include "xenithium/kernels/tile/bf16_tile.hpp"

namespace xenithium::kernels::fp8t {

namespace esimd = sycl::ext::intel::esimd;
namespace xmx = esimd::xmx;
using bf16 = sycl::ext::oneapi::bfloat16;

inline constexpr int kBlockBytes = kW4Group * kW4Width;  // 2048
inline constexpr float kWidenScale = 256.f;  // widen yields w * 2^-8

template <int N>
inline esimd::simd<bf16, N> widen(esimd::simd<uint8_t, N> b) SYCL_ESIMD_FUNCTION {
  const esimd::simd<int8_t, N> sb = b.template bit_cast_view<int8_t>();
  const esimd::simd<int16_t, N> s = sb;  // sign-extend
  esimd::simd<uint16_t, N> h = (esimd::simd<uint16_t, N>(s << 7)) & static_cast<uint16_t>(0xBF80);
  const esimd::simd<sycl::half, N> hv = h.template bit_cast_view<sycl::half>();
  const esimd::simd<float, N> f = hv;
  return f;
}

// gsum (M x 16 fp32, in units of 2^-8 * scale) = A (M x 128 bf16) * one FP8 block, or
// its first CALLS DPAS calls (16 k each: a partial last K group, no padding read).
template <int M, int CALLS = w4::kCalls>
inline esimd::simd<float, M * kW4Width> dot(const uint8_t *wp, esimd::simd<bf16, M * kW4Group> a_group)
    SYCL_ESIMD_FUNCTION {
  constexpr int kCalls = CALLS, kCallElems = bf16t::kCallElems;  // 256 bytes per call
  constexpr int kBytes = kCalls * kCallElems;
  esimd::simd<uint8_t, kBytes> raw;
  if constexpr (kBytes % 512 == 0) {
#pragma unroll
    for (int j = 0; j < kBytes / 512; ++j)
      raw.template select<512, 1>(j * 512) = esimd::block_load<uint8_t, 512>(wp + j * 512);
  } else {
#pragma unroll
    for (int j = 0; j < kBytes / 256; ++j)
      raw.template select<256, 1>(j * 256) = esimd::block_load<uint8_t, 256>(wp + j * 256);
  }
  esimd::simd<float, M * kW4Width> gsum(0.f);
#pragma unroll
  for (int j = 0; j < kCalls; ++j) {
    const esimd::simd<bf16, kCallElems> bj = widen<kCallElems>(raw.template select<kCallElems, 1>(j * kCallElems));
    esimd::simd<bf16, M * w4::kKPerCall> a_tile;
#pragma unroll
    for (int row = 0; row < M; ++row)
      a_tile.template select<w4::kKPerCall, 1>(row * w4::kKPerCall) =
          a_group.template select<w4::kKPerCall, 1>(row * kW4Group + j * w4::kKPerCall);
    gsum = xmx::dpas<8, M, float, float, bf16, bf16>(gsum, bj, a_tile);
  }
  return gsum;
}

}  // namespace xenithium:kernels:fp8t
