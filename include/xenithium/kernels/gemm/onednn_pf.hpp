// Xenithium · prefill GEMMs on oneDNN: W4A8 (src/kernels/gemm/onednn_pf.cpp) and the int8 dense matrices.
#pragma once

#include <cstddef>
#include <cstdint>

#include <sycl/sycl.hpp>

namespace xenithium::kernels {

// Routes the W4A8 prefill GEMMs (w4a16_gemm_pf / _swiglu with w4a16_pf_set_a8 on) whose output width (n; GLU: 2 x n_half)
// is >= min_n to oneDNN; 0 = never (the ESIMD body). Process-wide; initial value from XEN_PF_ONEDNN.
void w4a16_pf_set_onednn(int min_n);
int w4a16_pf_onednn();
// The rows per call from which a wide GEMM takes the oneDNN route (default 1,024, initial value from
// XEN_PF_ONEDNN_MIN_M; >= 1). The route's bits are not the ESIMD body's, so this is a numerics setting; values below
// 1,024 are meant for matrices whose layouts are kept (onednn_pf_keep).
void w4a16_pf_set_onednn_rows(int min_m);
int w4a16_pf_onednn_rows();
bool onednn_pf_takes(int m, int n_out);

// C (m x n fp32, row stride ldc) or, with h, h (m x n/2 bf16, stride ldh) = swiglu([gate | up]) of
// A8 (m x k s8 codes, row stride k; fp32 scales group-major [k/128][lds]). W (nib blocks + fp16 scales [k/128][n]).
// With h and gu_out, the GLU stops after the matmul: *gu_out = the bf16 [m][n] [gate | up] scratch (valid until the
// device's next call) and *tab_out = the swiglu table, for a caller that runs its own epilogue (w4a16_rows.cpp glu_q8).
sycl::event onednn_w4a8_pf(sycl::queue &q, int m, int n, int k, const int8_t *aq, const float *as, int lds, const uint8_t *w_packed, const sycl::half *scales,
                           float *c, int ldc, sycl::ext::oneapi::bfloat16 *h, int ldh, sycl::ext::oneapi::bfloat16 **gu_out = nullptr,
                           const float **tab_out = nullptr, const float *as_rows = nullptr);
// (as_rows: the activation scales already row-major [m][k / 128] (rms_norm_q8's s_rows): no transpose launch)

// Kept oneDNN layouts in memory the caller lends (the KV pool's free pages). keep(q, w, n, k,
// dst): from its next call on, onednn_w4a8_pf reads matrix (w, n, k)'s [n][k] s4 layout at dst (n x k / 2 bytes, 4 KiB
// aligned), repacked there once (at that call, or by onednn_pf_keep_fill) instead of into the per-call scratch on every
// call; the same bytes, so the same bits. dst null: the matrix is forgotten (the per-call scratch again) - the caller
// drains q before it reuses the memory. keep_fill repacks up to max_n kept matrices not filled yet (returns how many);
// keep_count: kept / of them filled; w4a16_pf_set_onednn forgets every kept matrix.
// onednn_pf_reserve: the per-call scratch slots allocated now for matrices of up to matrix_bytes (a caller that sizes
// a pool from the VRAM left wants the fallback's memory taken first, whether or not a call used it yet).
void onednn_pf_keep(sycl::queue &q, const uint8_t *w_packed, int n, int k, uint8_t *dst);
void onednn_pf_reserve(sycl::queue &q, size_t matrix_bytes);
int onednn_pf_keep_fill(sycl::queue &q, int max_n);
void onednn_pf_keep_count(sycl::queue &q, int *kept, int *filled);

// The int8 dense matrices' W8A16 many-row GEMMs (qwen38; pf_dense.hpp: its dispatch decides when) on oneDNN's
// matmul. w_packed = dense.hpp's int8 VNNI blocks (n x k, k % 16 == 0), repacked per call into oneDNN's [n][k] s8 in a
// per-device LRU of XEN_PF_DENSE_ONEDNN_SLOTS (default 4) scratch matrices (keyed by pointer: the split prefill calls one
// matrix twice per layer); the stored weights stay the decode layout. bf16 A (m x k, row stride lda) x the int8 codes
// widened to bf16 (fpmath bf16), fp32 sums, then x wscale[col] (null: the raw sums) - the engine's arithmetic, the same
// bits iff oneDNN sums k in order (the caller checks that per shape). ks > 1 (split-K, raw sums, f32 only): the k range in
// ks equal slices (k % ks == 0, slices multiples of 16) as one batched matmul, part p's raw sums at f32 + p x m x ld
// (pf_dense_splitk's layout). Exactly one of f32 / b16 (row stride ld >= n).
sycl::event onednn_i8_pf(sycl::queue &q, int m, int n, int k, const int8_t *w_packed, const float *wscale, const sycl::ext::oneapi::bfloat16 *a, int lda,
                         float *f32, sycl::ext::oneapi::bfloat16 *b16, int ld, int ks = 1);
// Frees the repack slot holding w_packed's copy, if any (pf_dense: a matrix whose check kept the engine's body).
void onednn_dense_forget(sycl::queue &q, const int8_t *w_packed);
// The repack slots per device (XEN_PF_DENSE_ONEDNN_SLOTS, default 4) and the VRAM they hold per device at the largest
// matrix's codes (pf_dense's first-call check allocates its output copy transiently, outside any budget).
int onednn_dense_slots();
size_t onednn_dense_scratch_bytes(size_t max_matrix_bytes);
// Differing 32-bit words between x and y over `parts` blocks of `rows` rows x row_bytes (row stride ld_bytes, part
// stride part_bytes); waits for q. row_bytes % 4 == 0.
size_t onednn_words_differ(sycl::queue &q, const void *x, const void *y, int parts, int rows, size_t row_bytes, size_t ld_bytes, size_t part_bytes);

}  // namespace xenithium:kernels
