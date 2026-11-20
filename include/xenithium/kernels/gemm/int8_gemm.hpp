// Xenithium · int8 W8A8 GEMM on XMX.
//
// Weights are streamed, activations are 4 rows: the shape of a decode step.
// The weight layout the DPAS consumes is what the Python converter must emit,
// so it is defined here once and used by kernel, tests and converter test.
#pragma once

#include <cstddef>
#include <cstdint>
#include <sycl/sycl.hpp>

#include "xenithium/kernels/pack.hpp"
#include "xenithium/kernels/rows.hpp"

namespace xenithium::kernels {

inline constexpr int kGemmM4Rows = 4;  // gemm_int8_m4 / w4a16_gemm_m4: M = 4 (one decode step), fixed

struct Int8GemmProblem {
  int m = kGemmM4Rows;
  int n = 0;  // output columns (vocab rows, expert rows, ...)
  int k = 0;  // reduction depth

  bool valid() const { return m == kGemmM4Rows && n > 0 && k > 0 && (n % 16 == 0) && (k % 32 == 0); }
  size_t weights_bytes() const { return static_cast<size_t>(n) * static_cast<size_t>(k); }
  size_t output_bytes() const { return static_cast<size_t>(m) * static_cast<size_t>(n) * sizeof(int32_t); }
  double macs() const { return static_cast<double>(m) * n * k; }
};

// Split-K: decode shapes (N = 640..3072) give only 40-192 column tiles, far too
// few work-items to keep 256 EUs streaming. K is split across a work-group until
// >= kMinWorkItems are in flight, and the int32 partials are reduced in SLM, bit-exact.
inline constexpr int kMinWorkItems = 256;  // one per Xe2 EU on a B70
inline constexpr int kMaxKSplit = 16;
// Power of two in [1, kMaxKSplit] that the auto mode picks for (n, k).
int gemm_int8_auto_ksplit(int n, int k);

// C (m x n int32, row-major) = A (m x k int8) * W (n x k int8, VNNI packed).
// One work-group per 16 output columns, `ksplit` work-items along K (rounded down
// to a power of two <= 16; 0 = auto). Any k % 32 == 0 works: parts may be uneven.
sycl::event gemm_int8_m4(sycl::queue &q, const int8_t *a, const int8_t *w_vnni, int32_t *c,
                         int n, int k, int ksplit = 0);

}  // namespace xenithium::kernels