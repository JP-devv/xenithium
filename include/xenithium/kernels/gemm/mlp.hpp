// Xenithium · dense SwiGLU MLP pieces, decode rows (the dense qwen35 family).
// Per card (TP by intermediate column, C cards): the card holds gate | up rows [2 I/C][H] as one W4
// matrix (gate's slice first, then up's) and down's k slice [H][I/C].
// gu = x. Wgu^T fp32 [rows][2 I/C] (w4a16_gemm_rows)
// h = bf16(silu(bf16 g) * bf16 u) bf16 [rows][I/C] (swiglu; silu(g) = g sigmoid(g), fp32)
// part = h. Wd^T fp32 [rows][H] (w4a16_gemm_rows, this card's k half)
// out = bf16(part_0 + part_1) bf16 [rows][H] (pair_reduce through the mailbox)
// Rounding points: g, u and h bf16 (the GEMM outputs are rounded to bf16), the down
// partials fp32 until the one rounding after the sum. The residual add and the next norm are
// rms_norm with its bf16 add.
#pragma once

#include <cstdint>
#include <sycl/sycl.hpp>

namespace xenithium::kernels {

using mlp_bf16 = sycl::ext::oneapi::bfloat16;

// Rows past `rows` untouched.
sycl::event swiglu(sycl::queue &q, int rows, int n, const float *gu, mlp_bf16 *h);
// s(g) = g sigmoid(g) with swiglu's expression for every bf16 g, by its bits: tab fp32 [65,536]
// (kSiluTabSize). swiglu(g, u) == bf16(tab[bits(bf16 g)] x bf16 u) for every pair, which lets an
// ESIMD epilogue (w4a16_gemm_rows_swiglu) reproduce the SIMT kernel's bits without its library exp.
inline constexpr int kSiluTabSize = 65536;
sycl::event swiglu_table(sycl::queue &q, float *tab);

// ---- host references (fp32 math as the device, exp from libm: bounded, not bit-exact) ----
void swiglu_ref(int rows, int n, const float *gu, uint16_t *h);
// One card's MLP from x fp32 [rows][H] with the W4 slices in the GPTQ orientation (pack.hpp: qweight
// [k/8][n] words, scales fp32 [k/128][n]): gu in double rounded to bf16, swiglu_ref, down in double ->
// part [rows][H] (this card's partial sum, not rounded).
void dense_mlp_ref(int rows, int H, int ipc, const float *x, const uint32_t *gu_qweight, const float *gu_scales, const uint32_t *dn_qweight,
                   const float *dn_scales, double *part);

}  // namespace xenithium:kernels
