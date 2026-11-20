#include "xenithium/kernels/drafter/mtp.hpp"

#include <stdexcept>

namespace xenithium::kernels {
namespace {

constexpr int kFrontItems = 256;
class MtpFrontNorm;
class MtpFrontAdd;

}  // namespace

sycl::event mtp_front_norm(sycl::queue &q, int H, int S, int rows, const mtp_bf16 *e, const mtp_bf16 *h, const mtp_bf16 *norm_e,
                           const mtp_bf16 *norm_h, float eps, mtp_bf16 *en, mtp_bf16 *hn) {
  if (rows < 1 || H <= 0 || S <= 0) throw std::invalid_argument("mtp_front_norm: shape");
  // work-group 2 r: the embedding of row r; 2 r + 1: the hidden of row r (all S * H values).
  return q.parallel_for<MtpFrontNorm>(sycl::nd_range<1>(static_cast<size_t>(rows) * 2 * kFrontItems, kFrontItems), [=](sycl::nd_item<1> it) {
    const int g = static_cast<int>(it.get_group(0)), row = g / 2, lid = static_cast<int>(it.get_local_id(0));
    const bool hid = g & 1;
    const int n = hid ? S * H : H;
    const mtp_bf16 *x = hid ? h + static_cast<size_t>(row) * S * H : e + static_cast<size_t>(row) * H;
    const mtp_bf16 *w = hid ? norm_h : norm_e;
    mtp_bf16 *out = hid ? hn + static_cast<size_t>(row) * S * H : en + static_cast<size_t>(row) * H;
    float ss = 0.f;
    for (int i = lid; i < n; i += kFrontItems) {
      const float v = static_cast<float>(x[i]);
      ss += v * v;
    }
    ss = sycl::reduce_over_group(it.get_group(), ss, sycl::plus<float>());
    const float rr = 1.f / sycl::sqrt(ss / static_cast<float>(n) + eps);
    for (int i = lid; i < n; i += kFrontItems) {
      const float y = static_cast<float>(x[i]) * rr;
      out[i] = mtp_bf16(sycl::fma(y, static_cast<float>(w[i]), y));
    }
  });
}

sycl::event mtp_front_add(sycl::queue &q, int H, int S, int rows, const float *fe, const float *fh, mtp_bf16 *res) {
  if (rows < 1) throw std::invalid_argument("mtp_front_add: rows");
  return q.parallel_for<MtpFrontAdd>(sycl::range<1>(static_cast<size_t>(rows) * S * H), [=](sycl::id<1> id) {
    const size_t i = id[0];
    const int c = static_cast<int>(i % H), row = static_cast<int>(i / (static_cast<size_t>(S) * H));
    const float a = static_cast<float>(mtp_bf16(fe[static_cast<size_t>(row) * H + c]));
    const float b = static_cast<float>(mtp_bf16(fh[i]));
    res[i] = mtp_bf16(a + b);
  });
}

}  // namespace xenithium::kernels
