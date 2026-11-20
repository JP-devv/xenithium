// Xenithium · W4A8 tile pieces (shared with the decode MoE): the int4 expert
// codes as s4 DPAS operands and the s8 x s4 group dot.
// The expert weights are the decode bytes: each 32-k call's 256 bytes are reordered in registers from
// the bf16 layout (byte p * 16 + lane holds k 2p, 2p + 1) to the s4 operand (dword (k / 8) * 16 + lane,
// nibble k % 8 LSB first), and q ^ 8 = q - 8 as s4.
#pragma once

#include <cstdint>

#include <sycl/ext/intel/esimd.hpp>

namespace xenithium::kernels::w4a8 {

namespace esimd = sycl::ext::intel::esimd;
namespace xmx = esimd::xmx;

inline constexpr int kK8 = 32;  // k per s8 x s4 call
inline constexpr int kOp8 = 256;  // bytes: the s4 B operand (32 k x 16 columns) = the s8 A operand (8 rows x 32 k)
inline constexpr int kTileW = 16;

// One 128-k group block (1024 bytes, 4 calls) or half block (512 bytes, 2 calls) as s4 operands.
template <int CALLS>
inline esimd::simd<uint8_t, CALLS * kOp8> to_s4(esimd::simd<uint8_t, CALLS * kOp8> raw) SYCL_ESIMD_FUNCTION {
  esimd::simd<uint8_t, CALLS * kOp8> nb;
#pragma unroll
  for (int c = 0; c < CALLS; ++c)
#pragma unroll
    for (int kb = 0; kb < 4; ++kb)
#pragma unroll
      for (int qq = 0; qq < 4; ++qq) nb.template select<16, 4>(c * kOp8 + kb * 64 + qq) = raw.template select<16, 1>(c * kOp8 + (4 * kb + qq) * 16);
  esimd::simd<uint32_t, CALLS * kOp8 / 4> w = nb.template bit_cast_view<uint32_t>();
  w ^= 0x88888888u;
  return w.template bit_cast_view<uint8_t>();
}

// M rows (1..8) x CALLS * 32 k of int8 (row r at r * CALLS * 32 in `a`) against the s4 operands:
// int32 [M][16].
template <int M, int CALLS>
inline esimd::simd<int32_t, M * kTileW> dot_s8(esimd::simd<uint8_t, CALLS * kOp8> b, esimd::simd<uint8_t, M * CALLS * kK8> a) SYCL_ESIMD_FUNCTION {
  esimd::simd<int32_t, M * kTileW> gs(0);
#pragma unroll
  for (int c = 0; c < CALLS; ++c) {
    esimd::simd<uint8_t, M * kK8> at;
#pragma unroll
    for (int r = 0; r < M; ++r) at.template select<kK8, 1>(r * kK8) = a.template select<kK8, 1>(r * CALLS * kK8 + c * kK8);
    esimd::simd<uint8_t, kOp8> bc = b.template select<kOp8, 1>(c * kOp8);
    gs = xmx::dpas<8, M, int32_t, int32_t, uint8_t, uint8_t, xmx::dpas_argument_type::s4, xmx::dpas_argument_type::s8, M * kTileW, kOp8, M * kK8>(gs, bc, at);
  }
  return gs;
}

}  // namespace xenithium:kernels:w4a8
