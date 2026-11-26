// Xenithium · plain-residual RMS norm, decode rows (the dense qwen35 family).
// A position carries one residual stream of `hidden` bf16. Every norm of the family (input and
// post-attention layer norms, the final norm) is the (1 + w) affine in fp32 with one rounding to
// bf16 at the end (the model's own definition):
// res' = bf16(res + add) (only with a pending add: the block's output,
// bf16, or the cards' fp32 all-reduced partial sum)
// xn = bf16(fma(y, w, y)), y = res' * rsqrt(mean(res'^2) + eps)
// ESIMD, one work-group per row of hidden / kRmsNormCols items, each item
// holding its 128 columns in registers from the first load to the store (no second pass over res);
// the per-item sums of squares meet in SLM and every item adds the 64 slots in the same fixed order.
// Not exact vs a host loop (sum order, rsqrt): gated to 2 bf16 ulps of max(|ref|, row rms) plus
// run-to-run bit identity; the residual add is exact (one fp32 add, one rounding) and gated bit for
// bit.
#pragma once

#include <cstdint>
#include <sycl/sycl.hpp>

namespace xenithium::kernels {

using rn_bf16 = sycl::ext::oneapi::bfloat16;

inline constexpr int kRmsNormCols = 128;  // columns per item; hidden % 128 == 0, hidden <= 64 x 128

// res bf16 [rows][hidden], updated in place when add_bf16 or add_f32 is set (at most one); w [hidden];
// xn bf16 [rows][hidden]. Rows past `rows` are not touched.
// plain (arch llama, LlamaRMSNorm): xn = bf16(w x bf16(y)) (rms_norm_plain's two roundings) in place of the (1 + w) affine;
// plain 2 (arch flint, FlintRMSNorm): xn = bf16(w x y), one rounding. 0 / false = the (1 + w) affine.
sycl::event rms_norm(sycl::queue &q, int rows, int hidden, rn_bf16 *res, const rn_bf16 *add_bf16, const float *add_f32, const rn_bf16 *w,
                     float eps, rn_bf16 *xn, int plain = 0);

// The same norm also writing xn as the W4A8 prefill's activation codes - int8 [rows][hidden]
// (pitch hidden) + fp32 scales group-major s[g x lds + row], per (row, 128-k group) by the W4A8 quantiser's rule on the bf16 xn
// (s = amax / 127, q = rne(v x 127 / amax) clamped to +-127; 0 / 0 for an all-zero group): an item's 128 columns are one
// group. Equal, bit for bit, to w4a16_gemm_pf's own quantiser on xn (w4a16_pf_a8_input hands the codes to it).
struct RmsNormQ8 {
  int8_t *q = nullptr;
  float *s = nullptr;
  int lds = 0;
  float *s_rows = nullptr;  // optional: the same scales row-major [rows][hidden / 128] (oneDNN's W4A8 reads them so)
  bool codes_only = false;  // the codes and scales only: xn is not written (its only readers take the codes)
};
sycl::event rms_norm_q8(sycl::queue &q, int rows, int hidden, rn_bf16 *res, const rn_bf16 *add_bf16, const float *add_f32, const rn_bf16 *w,
                        float eps, rn_bf16 *xn, const RmsNormQ8 &q8, int plain = 0);
// s[g x lds + r] = s_rows[r x groups + g] for rows [0, rows): the group-major scales of rows whose row-major scales came
// from another card (the prefill norm on its owner's rows); pass s_rows and s offset to the first row.
sycl::event rms_q8_scales_cols(sycl::queue &q, int rows, int groups, const float *s_rows, float *s, int lds);

// (the DFlash 2 drafter, a Qwen3 model): the plain Qwen3RMSNorm, two roundings,
// res' = bf16(res + add) (with add), xn = bf16(w x bf16(res' x rsqrt(mean(res'^2) + eps))); the same kernel otherwise.
sycl::event rms_norm_plain(sycl::queue &q, int rows, int hidden, rn_bf16 *res, const rn_bf16 *add_bf16, const rn_bf16 *w, float eps, rn_bf16 *xn);
void rms_norm_plain_ref(int rows, int hidden, uint16_t *res, const uint16_t *add_bf16, const uint16_t *w, float eps, uint16_t *xn);

// ---- host reference (bf16 tensors as uint16 bits; the same rounding points, fp32 math, sum in double) ----
void rms_norm_ref(int rows, int hidden, uint16_t *res, const uint16_t *add_bf16, const float *add_f32, const uint16_t *w, float eps,
                  uint16_t *xn);

}  // namespace xenithium:kernels
