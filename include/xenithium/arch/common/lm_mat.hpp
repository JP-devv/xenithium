// Xenithium · an lm_head matrix on one card, as the step heads and the drafters read it (qwen35 and qwen38 share it
// through the DFlash 2 drafter).
#pragma once

#include <sycl/sycl.hpp>

#include "xenithium/kernels/gemm/dense.hpp"
#include "xenithium/kernels/gemm/w4a16_gemm.hpp"
#include "xenithium/load/dense_store.hpp"

namespace xenithium::arch {

// The lm_head (or a draft vocabulary's rows of it) on one card: int8 / bf16 dense tiles, or (
// ModelOptions::head_w4) the checkpoint's int4 g128 lm_head.qweight on the W4 GEMM. logits [rows][n] fp32 either way.
struct LmMat {
  kernels::DenseWeights d;
  kernels::W4Mat w4;
  int n = 0, k = 0;
  bool is_w4() const { return w4.w != nullptr; }
};
LmMat lm_mat(const load::DeviceTensor &t);
sycl::event lm_gemm(sycl::queue &q, int rows, const sycl::ext::oneapi::bfloat16 *x, const LmMat &w, float *logits);

}  // namespace xenithium::arch
