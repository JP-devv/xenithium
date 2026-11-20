// Xenithium · ESIMD int8 DPAS tile primitives (the hot-kernel format).
// DPAS on Arc Pro B70 (bmg, Xe2):
// * execution width 16 is the one that computes correctly (8 silently
// produces wrong results, so the engine must never emit it);
// * A operand = M x 32 int8 per work-item, row-major;
// * B operand = one contiguous 512-byte VNNI block per work-item:
// dword (k/4)*16 + lane holds k%4..; 16 lanes = 16 columns;
// * C operand = M x 16 int32 per work-item, row-major.
#pragma once

#include <sycl/ext/intel/esimd.hpp>

#include "xenithium/kernels/gemm/int8_gemm.hpp"

namespace xenithium::dpas {

namespace esimd = sycl::ext::intel::esimd;
namespace xmx = esimd::xmx;

inline constexpr int kSystolicDepth = 8;
inline constexpr int kExecWidth = 16;  // Xe2: 16 (8 is wrong here)
inline constexpr int kOpsPerChannel = 32 / 8;  // int8: 4
inline constexpr int kKPerCall = kSystolicDepth * kOpsPerChannel;  // 32 int8 per DPAS
inline constexpr int kLaneBytes = kKPerCall;  // 32 bytes of B per lane
inline constexpr int kBlockBytes = kExecWidth * kLaneBytes;  // 512 bytes per DPAS B block

// C += A * B for one tile, one work-item. `b` is the VNNI block of 16 columns,
// `a` is M rows x 32 int8 of the activations.
template <int M>
inline esimd::simd<int32_t, M * kExecWidth> accumulate(
    const esimd::simd<int32_t, M * kExecWidth> &acc,
    const esimd::simd<int8_t, kBlockBytes> &b,
    const esimd::simd<int8_t, M * kKPerCall> &a) SYCL_ESIMD_FUNCTION {
  return xmx::dpas<kSystolicDepth, M, int32_t, int32_t, int8_t, int8_t,
                   xmx::dpas_argument_type::s8, xmx::dpas_argument_type::s8,
                   M * kExecWidth, kBlockBytes, M * kKPerCall>(acc, b, a);
}

}  // namespace xenithium:dpas