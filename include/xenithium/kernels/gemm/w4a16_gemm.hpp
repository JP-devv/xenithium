// Xenithium · int4 W4A16 GEMM on XMX (the routed-expert inner loop).
// Weights are the checkpoint's GPTQ int4 (auto_round:auto_gptq, symmetric, group 128,
// stored zero 7 -> effective 8): w[k][n] = (q - 8) * scale[k / 128][n], q in 0..15.
// The kernel keeps them int4 in memory (the tiering budget assumes 4-bit experts),
// widens q - 8 to bf16 in registers (exact: |q - 8| <= 8), runs bf16 DPAS into an
// fp32 per-group sum, and applies the group's fp16 scale once per group:
// C[m][n] = sum_g scale[g][n] * sum_{k in g} A[m][k] * (q[k][n] - 8)
// Activations are bf16, output fp32.
#pragma once

#include <cstddef>
#include <cstdint>
#include <sycl/sycl.hpp>

#include "xenithium/kernels/gemm/int8_gemm.hpp"  // rows.hpp
#include "xenithium/kernels/pack.hpp"

namespace xenithium::kernels {

// Default work-items in flight for the auto k split. The split changes the fp32 order of the group sums.
inline constexpr int kW4MinWorkItems = 1024;
inline constexpr int kW4MaxKSplit = 16;

struct W4Problem {
  int n = 0, k = 0;
  bool valid() const { return n > 0 && k > 0 && n % kW4Width == 0 && k % kW4Group == 0; }
  int groups() const { return k / kW4Group; }
  size_t packed_bytes() const { return static_cast<size_t>(n) * k / 2; }
  size_t scale_count() const { return static_cast<size_t>(groups()) * n; }
};

// Scales stay in the checkpoint's own layout: fp16 [k/128][n] (16 columns of a tile
// are 32 contiguous bytes).
// C (4 x n fp32) = A (4 x k bf16) * W. `ksplit` work-items per tile share K by whole
// groups (0 = auto, >= kW4MinWorkItems in flight); partials meet in SLM.
sycl::event w4a16_gemm_m4(sycl::queue &q, const sycl::ext::oneapi::bfloat16 *a, const uint8_t *w_packed,
                          const sycl::half *scales, float *c, int n, int k, int ksplit = 0);
int w4a16_auto_ksplit(int n, int k);
// The split w4a16_auto_ksplit picks for a target of `items` work-items instead of the w4a16_set_min_items one.
int w4a16_ksplit_for(int n, int k, int items);
// A W4 matrix on the card as the GEMM reads it (load:W4Weights converts to it): blocks
// [k/128][n/16][1024] + fp16 scales [k/128][n], n padded to 16.
struct W4Mat {
  const uint8_t *w = nullptr;
  const sycl::half *scales = nullptr;
  int n = 0, k = 0;
  bool valid() const { return w && scales && n > 0 && k > 0 && n % kW4Width == 0 && k % kW4Group == 0; }
  size_t bytes() const { return static_cast<size_t>(n) * k / 2 + static_cast<size_t>(k / kW4Group) * n * 2; }
};
// The same GEMM at the step's row shape (rows.hpp): 1 / 2 rows one M = 1 / 2 launch (only the
// live rows' A is loaded), 3-4 rows M = 4; 5-8 rows one M = 8 launch reading the weights once. a's row stride k, c's n.
// Row r's bits do not depend on the row count or the mode (bf16 DPAS is exact at every repeat count). 9-16 rows in one
// launch (rows 9-12 / 13-16: a second DPAS chain of 4 / 8 rows on the same widened weights, A by 2D block loads; or two
// 8-row halves per work-group where they fit one wave of the card's threads), same bits per row; a, c then hold the live
// rows (a: a 64-byte aligned base, else two launches). Zero-point and affine matrices take 9-16 rows as two launches.
// 17-32 rows in one launch (four 8-row DPAS chains on each widened call, A by one 2D load of 32 rows; the same
// bits per row), else two launches (rows 0-15, then the rest).
sycl::event w4a16_gemm_rows(sycl::queue &q, int rows, const sycl::ext::oneapi::bfloat16 *a, const uint8_t *w_packed, const sycl::half *scales,
                            float *c, int n, int k, int ksplit = 0);
// The [gate | up] W4 GEMM with swiglu in its epilogue: w = [gate | up] with n = 2 x n_half;
// h [rows][n_half] bf16 = swiglu(gu) of w4a16_gemm_rows(rows, a, w.., n = 2 n_half, ksplit) bit for bit (same k split,
// same per-item sums; s(g) from silu_tab = kernels:swiglu_table, so h equals kernels:swiglu exactly).
// One launch and no fp32 gu round trip; only the live rows of h are written.
sycl::event w4a16_gemm_rows_swiglu(sycl::queue &q, int rows, const sycl::ext::oneapi::bfloat16 *a, const uint8_t *w_packed, const sycl::half *scales,
                                   const float *silu_tab, sycl::ext::oneapi::bfloat16 *h, int n_half, int k, int ksplit = 0);
// W4A8 decode (opt-in; it changes the numerics): a matrix registered by its packed-weight pointer runs
// both GEMMs above on int8 activations while w4a8_rows_set(true) holds: A quantised per (row, 128-k group) (s = amax /
// 127, rne; one small launch into the device's scratch), then s8 x s4 DPAS with exact int32 group sums folded in fp32
// (src/kernels/gemm/w4a8_rows.hpp). A row's bits do not depend on the step's rows. Plain symmetric g128 matrices only
// (zero-point and affine ones stay W4A16); unregistered matrices (the drafters', lm_head) stay W4A16. The scratch
// is one per device: the calls of one device come from one thread on one in-order queue (the engine's step thread).
void w4a8_rows_register(const uint8_t *w_packed);
void w4a8_rows_unregister(const uint8_t *w_packed);
void w4a8_rows_set(bool on);
bool w4a8_rows_on();
bool w4a8_rows_of(const uint8_t *w_packed);  // registered and on
// The device's scratch for A's codes [32 rows][max_k] and scales (grows; before the first W4A8 call on the device).
void w4a8_rows_reserve(sycl::queue &q, int max_k);
// codes / scales of the device's scratch for a call at k; throws std:logic_error without a reserve of >= k.
void w4a8_rows_scratch(sycl::queue &q, int k, int8_t **codes, float **scales);
// Zero points (arch llama: int4 g32 asymmetric, pack.hpp kW4ZpGroup): a matrix registered here by its packed-weight
// pointer is read by every W4 GEMM above as w = (q - z) x s per 32-k group: `scales` then points at fp16 [k/32][n] and
// `zeros` holds uint8 [k/64][n] (pack_w4_zeros). Process-wide; the loader registers each such matrix and clears it
// when the weights are released. The many-row (prefill) GEMMs refuse a registered matrix until they implement it.
void w4_set_zero_points(const uint8_t *w_packed, const uint8_t *zeros);
void w4_clear_zero_points(const uint8_t *w_packed);
const uint8_t *w4_zero_points(const uint8_t *w_packed);  // nullptr: symmetric g128
// Affine groups (arch flint: int4 g64, pack.hpp kW4AffGroup): a matrix registered here by its packed-weight pointer is
// read by the decode W4 GEMMs above as w = q x s + b per 64-k group (q 0..15 raw): `scales` then points at bf16 bits
// [k/64][n] (sycl:half-typed pointer, reinterpreted) and `biases` at bf16 [k/64][n]. Each group's term is
// s x sum(q a) + b x sum(a) in fp32 (the quantised matvec's own form). The many-row GEMMs below take it too
// (w4aff_gemm_pf).
void w4_set_affine(const uint8_t *w_packed, const uint16_t *biases);
void w4_clear_affine(const uint8_t *w_packed);
const uint16_t *w4_affine(const uint8_t *w_packed);  // nullptr: not affine

// A/B switch for 5-8 rows (process-wide): 1 = one M = 8 launch (default), 0 = two M = 4 launches.
void w4a16_rows8_mode(int on);
// The auto k split's work-item target (process-wide; <= 0 = kW4MinWorkItems; the qwen38-27b server's XEN_W4_ITEMS,
// StepOptions:w4_items).
void w4a16_set_min_items(int items);
int w4a16_min_items();
// w4a16_gemm_pf / w4a16_gemm_pf_swiglu calls of at most `rows` rows (0 = off, up to 8) run on the decode GEMMs
// (w4a16_gemm_rows / _swiglu: W4A16 numerics, decode's k split, any A8 codes of the call unused): a 1-8-row prefill
// call's GEMMs at the decode step's rate instead of the many-row bodies' fixed cost.
// Process-wide (the server: XEN_PREFILL_SHORT); matrices with zero points keep the many-row path.
void w4a16_pf_set_short_rows(int rows);
int w4a16_pf_short_rows();  // 0 when the decode GEMMs are not registered
// (libxenithium_kernels registers its decode GEMMs and the swiglu table at load; the many-row library cannot link them)
using W4ShortGemm = sycl::event (*)(sycl::queue &, int, const sycl::ext::oneapi::bfloat16 *, const uint8_t *, const sycl::half *, float *, int, int, int);
using W4ShortGlu = sycl::event (*)(sycl::queue &, int, const sycl::ext::oneapi::bfloat16 *, const uint8_t *, const sycl::half *, const float *,
                                   sycl::ext::oneapi::bfloat16 *, int, int, int);
using W4SiluTable = sycl::event (*)(sycl::queue &, float *);
void w4a16_pf_short_hooks(W4ShortGemm gemm, W4ShortGlu glu, W4SiluTable tab);
// Many rows (prefill; w4a16_rows.cpp): C (m x n fp32, row stride ldc) = A (m x k bf16, row stride
// lda: a 16-byte multiple, >= 64 bytes; 64-byte aligned base). W on the same blocks and scales. Per 128-k group the
// fp32 group sum times the group's scale, groups in order (decode's arithmetic; decode's k split sums the groups in
// another order, so the two differ by fp32 summation order only). A row's bits depend neither on m nor on the row's
// place in the call. A work-group widens a column block's weights once into SLM for its R x MB rows (R row blocks x
// 16 / R column groups of NT tiles). shape: -1 = by m (8 x 4 under 256 rows, else 32 x 2 on R 8),
// 0 = MB 16 x NT 2, 1 = 16 x 1, 2 = 8 x 4, 3 = 32 x 1, 4 = 16 x 4 (R 16), 7 = 32 x 2 (R 8), 11 = 32 x 2 (R 16); every shape
// gives the same bits.
sycl::event w4a16_gemm_pf(sycl::queue &q, int m, const sycl::ext::oneapi::bfloat16 *a, int lda, const uint8_t *w_packed, const sycl::half *scales,
                          float *c, int ldc, int n, int k, int shape = -1);
// The gate | up GEMM with swiglu in its epilogue: W = [gate | up] with n = 2 x n_half columns; out
// h [m][ldh] bf16 = bf16(g sigmoid(g) u) with g, u = bf16 of the plain kernel's fp32 columns c and n_half + c (the
// swiglu kernel's rounding points; gated bit-identical in test-kernels). Saves the fp32 [m][2 n_half] round trip and
// a launch. shape: -1 = by m (as w4a16_gemm_pf), 2 = 8 x 4, 4 = 16 x 4, 7 = 32 x 2 on R 8, other = 16 x 2 (NT tiles =
// NT / 2 gate + NT / 2 up).
sycl::event w4a16_gemm_pf_swiglu(sycl::queue &q, int m, const sycl::ext::oneapi::bfloat16 *a, int lda, const uint8_t *w_packed,
                                 const sycl::half *scales, sycl::ext::oneapi::bfloat16 *h, int ldh, int n_half, int k, int shape = -1);
// The many-row GEMMs on an affine g64 matrix (arch flint; codes in the W4 blocks, bf16 scales / biases
// [k/64][n]; w4a16_gemm_pf / _swiglu call these for a matrix registered with w4_set_affine). Exact mode (default): A's row
// sums per 64-k group first (one small launch into the device's scratch: w4a16_pf_aff_reserve before the first call, rows x
// k of the largest), then per group gs = sum q a (raw codes, 4 DPAS calls) folded as acc += gs x s + asum x b in fp32, the
// groups in order: decode's affine expression (block_dot_aff), the groups in another order than decode's k split. With
// w4a16_pf_set_dequant: the widen writes bf16(q x s + b) and the DPAS accumulate the whole k (each weight rounded to
// bf16 once; no row sums). shape: the exact mode's (-1 = by m, 0 / 2 / 4 / 7) or the dequantised mode's (12-15).
sycl::event w4aff_gemm_pf(sycl::queue &q, int m, const sycl::ext::oneapi::bfloat16 *a, int lda, const uint8_t *w, const uint16_t *scales,
                          const uint16_t *biases, float *c, int ldc, int n, int k, int shape = -1);
sycl::event w4aff_gemm_pf_swiglu(sycl::queue &q, int m, const sycl::ext::oneapi::bfloat16 *a, int lda, const uint8_t *w, const uint16_t *scales,
                                 const uint16_t *biases, sycl::ext::oneapi::bfloat16 *h, int ldh, int n_half, int k, int shape = -1);
void w4a16_pf_aff_reserve(sycl::queue &q, int max_m, int max_k);
// A's fp32 row sums per 64-k group, asum[g][lds] (lds >= m rounded up to 16; esimd:reduce over the group's 64 values):
// the exact mode's first launch, for kernels that fold the affine biases themselves (moe_aff_rows).
sycl::event w4aff_rowsum(sycl::queue &q, const sycl::ext::oneapi::bfloat16 *a, int lda, int m, int k, float *asum, int lds);
// Prefill numerics switch for both many-row GEMMs above (process-wide; qwen35
// ModelOptions:prefill_w4_dequant, server XEN_PREFILL_W4). false (default) = the exact per-group fp32 sums above. true =
// dequantised: the widen writes bf16((q - 8) x scale) and the DPAS accumulate the whole k in fp32 (each weight rounded
// to bf16 once), with no group sums, on 32 x 4 tiles. Shapes in
// this mode: -1 = by m and n, 12 = 32 x 4 R 8, 13 = 16 x 4 R 16, 14 = 32 x 2 R 8, 15 = 8 x 4 R 16 (all the same bits; a row's
// bits depend neither on m nor on its place in the call, as in the exact mode). Decode (w4a16_gemm_rows) is not affected.
void w4a16_pf_set_dequant(bool on);
bool w4a16_pf_dequant();
// W4A8 prefill (the nib blocks reordered into s4 operands while staged): both many-row GEMMs above quantise A per (row,
// 128-k group) to int8 (row_quant.hpp's rule) into the device's scratch, then s8 x s4 DPAS with int32 group sums folded
// as float(sum) x (w_scale x a_scale), groups in order. A numerics switch (the checkpoint is calibrated W4A16). Scratch
// per device: w4a16_pf_a8_reserve before the first call (rows x k of the largest call). shape (bench): -1 by m (8 x 4
// R 16 to 96 rows, else 24 x 4 R 16 HALF where its 384-row work-groups pad the call little - 384 / 768 / 1,024 / 1,536 /
// 2,048 rows, not 512 - else 32 x 2 R 8), 0 = 32 x 2 R 8, 1 = 8 x 4 R 16, 2 = 16 x 4 R 16, 3 = 32 x 2 R 16, 5 = 24 x 4
// R 16 HALF. The body is pipelined (next stage's weights in registers during the DPAS, 2 groups per barrier, 4 for
// gate|up; A one call ahead); every shape gives the same bits. The served default for qwen35 (server
// XEN_PREFILL_W4=a8).
void w4a16_pf_set_a8(bool on);
bool w4a16_pf_a8();
void w4a16_pf_set_a8_shape(int shape);
// The W4A8 GEMM body: 0 = the pipelined SLM body (default), 1 = the SLM-free body on every call (no SLM, each item's nib
// units reordered in registers + L1 prefetch), 3 = the SLM-free body on calls of at most 128 rows and the SLM body above
// (the SLM body's small shape waits at its barriers). Every choice gives the same outputs bit for bit; process-wide
// (ModelOptions:prefill_a8_body).
void w4a16_pf_set_a8_body(int body);
int w4a16_pf_a8_body();
void w4a16_pf_a8_reserve(sycl::queue &q, int max_m, int max_k);
// Matrices (by packed-weight pointer) that stay W4A16 while W4A8 is on: the per-GEMM-family / per-layer mask.
void w4a16_pf_a8_keep_clear();
void w4a16_pf_a8_keep(const uint8_t *w_packed);
// A already quantised by its producer (rms_norm_q8): codes int8 [rows][k] (pitch k) + scales
// s[g x lds + row] of the bf16 buffer src (pitch lda) on q's device. A W4A8 call whose A lies inside it (same lda and k,
// whole rows, a row offset that keeps the scales' 16-byte loads aligned) reads those codes and skips its quantiser - the
// same bits. One per device; the producer registers it after enqueuing, clear when src is rewritten otherwise.
// scales_rows (optional) = the same scales row-major [rows][k / 128]: a oneDNN call then skips its transpose.
void w4a16_pf_a8_input(sycl::queue &q, const sycl::ext::oneapi::bfloat16 *src, int lda, int rows, int k, const int8_t *codes, const float *scales, int lds,
                       const float *scales_rows = nullptr);
void w4a16_pf_a8_input_clear(sycl::queue &q);

}  // namespace xenithium:kernels
