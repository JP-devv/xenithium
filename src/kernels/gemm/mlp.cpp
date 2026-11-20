#include "xenithium/kernels/gemm/mlp.hpp"

#include <cmath>
#include <stdexcept>
#include <vector>

#include "xenithium/kernels/pack.hpp"

namespace xenithium::kernels {
namespace {

class Swiglu;
class SiluTab;

}  // namespace

sycl::event swiglu(sycl::queue &q, int rows, int n, const float *gu, mlp_bf16 *h) {
  if (rows < 1 || n < 1 || !gu || !h) throw std::invalid_argument("swiglu: shape or buffers");
  return q.parallel_for<Swiglu>(sycl::range<1>(static_cast<size_t>(rows) * n), [=](sycl::id<1> id) {
    const size_t i = id[0];
    const int r = static_cast<int>(i / n), c = static_cast<int>(i % n);
    const float g = static_cast<float>(mlp_bf16(gu[static_cast<size_t>(r) * 2 * n + c]));
    const float u = static_cast<float>(mlp_bf16(gu[static_cast<size_t>(r) * 2 * n + n + c]));
    const float s = g * (1.f / (1.f + sycl::exp(-g)));
    h[i] = mlp_bf16(s * u);
  });
}

sycl::event swiglu_table(sycl::queue &q, float *tab) {
  if (!tab) throw std::invalid_argument("swiglu_table: buffer");
  return q.parallel_for<SiluTab>(sycl::range<1>(kSiluTabSize), [=](sycl::id<1> id) {
    const float g = static_cast<float>(sycl::bit_cast<mlp_bf16>(static_cast<uint16_t>(id[0])));
    tab[id[0]] = g * (1.f / (1.f + sycl::exp(-g)));  // swiglu's s
  });
}

void swiglu_ref(int rows, int n, const float *gu, uint16_t *h) {
  for (int r = 0; r < rows; ++r)
    for (int c = 0; c < n; ++c) {
      const float g = bf16_value(bf16_bits(gu[static_cast<size_t>(r) * 2 * n + c]));
      const float u = bf16_value(bf16_bits(gu[static_cast<size_t>(r) * 2 * n + n + c]));
      const float s = g * (1.f / (1.f + std::exp(-g)));
      h[static_cast<size_t>(r) * n + c] = bf16_bits(s * u);
    }
}

void dense_mlp_ref(int rows, int H, int ipc, const float *x, const uint32_t *gu_qweight, const float *gu_scales, const uint32_t *dn_qweight,
                   const float *dn_scales, double *part) {
  std::vector<double> gu(static_cast<size_t>(rows) * 2 * ipc);
  w4a16_ref(x, gu_qweight, gu_scales, rows, 2 * ipc, H, gu.data());
  std::vector<float> guf(gu.size());
  for (size_t i = 0; i < gu.size(); ++i) guf[i] = static_cast<float>(gu[i]);  // the device's fp32 sum, then bf16 inside swiglu
  std::vector<uint16_t> hb(static_cast<size_t>(rows) * ipc);
  swiglu_ref(rows, ipc, guf.data(), hb.data());
  std::vector<float> hf(hb.size());
  for (size_t i = 0; i < hb.size(); ++i) hf[i] = bf16_value(hb[i]);
  w4a16_ref(hf.data(), dn_qweight, dn_scales, rows, H, ipc, part);
}

}  // namespace xenithium:kernels
