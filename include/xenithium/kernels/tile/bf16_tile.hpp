// Xenithium · bf16 weight tiles on XMX, shared by the MoE block and the HC mixers.
// Layout (pack_bf16_tiles in moe_w4.hpp): block (128-k group g, 16-column tile t) is
// 4096 bytes, 8 DPAS calls of 16 k x 16 columns in VNNI order. A partial last group of
// `calls` x 16 k is the first calls * 512 bytes of such a block.
#pragma once

#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

#include "xenithium/kernels/tile/w4_tile.hpp"
#include "xenithium/kernels/gemm/w4a16_gemm.hpp"

namespace xenithium::kernels::bf16t {

namespace esimd = sycl::ext::intel::esimd;
namespace xmx = esimd::xmx;
using bf16 = sycl::ext::oneapi::bfloat16;

inline constexpr int kCallElems = w4::kKPerCall * kW4Width;  // 256 bf16 = 512 bytes

// gsum (M x 16 fp32) = A (M x 128 bf16, only the first CALLS*16 k used) * one packed
// bf16 block of CALLS calls (CALLS * 16 k x 16 columns).
template <int M, int CALLS = w4::kCalls>
inline esimd::simd<float, M * kW4Width> dot(const uint8_t *wp, esimd::simd<bf16, M * kW4Group> a_group)
    SYCL_ESIMD_FUNCTION {
  esimd::simd<bf16, CALLS * kCallElems> b;
#pragma unroll
  for (int j = 0; j < CALLS; ++j)
    b.template select<kCallElems, 1>(j * kCallElems) =
        esimd::block_load<bf16, kCallElems>(reinterpret_cast<const bf16 *>(wp) + j * kCallElems);
  esimd::simd<float, M * kW4Width> gsum(0.f);
#pragma unroll
  for (int j = 0; j < CALLS; ++j) {
    esimd::simd<bf16, M * w4::kKPerCall> a_tile;
#pragma unroll
    for (int row = 0; row < M; ++row)
      a_tile.template select<w4::kKPerCall, 1>(row * w4::kKPerCall) =
          a_group.template select<w4::kKPerCall, 1>(row * kW4Group + j * w4::kKPerCall);
    const esimd::simd<bf16, kCallElems> bj = b.template select<kCallElems, 1>(j * kCallElems);
    gsum = xmx::dpas<8, M, float, float, bf16, bf16>(gsum, bj, a_tile);
  }
  return gsum;
}

}  // namespace xenithium:kernels:bf16t
