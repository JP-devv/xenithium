// Xenithium · W4A16 decode GEMMs on oneDNN's weight layout (src/kernels/gemm/w4kn_rows.cpp).
// The layout ("kn"): w [n][k / 2] bytes, int4 s4 two per byte (low nibble = even k, value q - 8 as signed 4-bit), fp16
// scales [k / 128][n] (the checkpoint's own, shared with the nib layout) - the layout oneDNN's matmul reads as weights
// {k, n} tag ba, so the prefill GEMMs run on oneDNN without a repack and the decode reads the same bytes: one stored copy.
// w4kn_repack writes it from the nib blocks (pack.hpp / w4a16_gemm.hpp) as a byte permutation with q ^ 8.
// Arithmetic per (row, column) equals w4a16_gemm_rows': per 128-k group the fp32 DPAS sum of a x (q - 8) (bf16 operands,
// the same B operand bytes), times the group's scale, groups in order per k part, parts summed in order.
#pragma once

#include <cstdint>

#include <sycl/sycl.hpp>

namespace xenithium::kernels {

// c (rows x n fp32, row stride n) = a (rows x k bf16, row stride k). W (kn layout). rows 1..8, n % 16 == 0,
// k % 128 == 0; w's base 64-byte aligned and k / 2 a multiple of 16 bytes (2D block loads). ksplit <= 0: by shape.
sycl::event w4kn_gemm_rows(sycl::queue &q, int rows, const sycl::ext::oneapi::bfloat16 *a, const uint8_t *w_kn, const sycl::half *scales, float *c, int n,
                           int k, int ksplit = 0);

// The kn layout from the nib blocks (w4a16_gemm's packed weights): dst [n][k / 2] (n * k / 2 bytes).
sycl::event w4kn_repack(sycl::queue &q, const uint8_t *w_nib, int n, int k, uint8_t *dst);

}  // namespace xenithium:kernels
