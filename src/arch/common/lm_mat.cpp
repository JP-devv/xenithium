#include "xenithium/arch/common/lm_mat.hpp"

namespace xenithium::arch {

LmMat lm_mat(const load::DeviceTensor &t) {
  LmMat m;
  if (t.store == model::DenseSpec::Store::w4) {  // int4
    const load::W4Weights w = t.w4();
    m.w4 = kernels::W4Mat{w.w, w.scales, w.n, w.k};
    m.n = w.n, m.k = w.k;
  } else {
    m.d = t.dense();
    m.n = m.d.n, m.k = m.d.k;
  }
  return m;
}

sycl::event lm_gemm(sycl::queue &q, int rows, const sycl::ext::oneapi::bfloat16 *x, const LmMat &w, float *logits) {
  if (w.is_w4()) return kernels::w4a16_gemm_rows(q, rows, x, w.w4.w, w.w4.scales, logits, w.w4.n, w.w4.k);
  return kernels::dense_gemm(q, rows, x, w.d, logits);
}

}  // namespace xenithium::arch
