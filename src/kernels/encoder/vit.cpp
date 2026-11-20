// Xenithium · the vision encoder's own kernels (xenithium/kernels/encoder/vit.hpp): position-grid sampling and the
// 2-D rope. Compiled -fp-model=precise (src/kernels/CMakeLists.txt): the rounding points of the header hold on the card.
#include <sycl/ext/intel/math.hpp>
#include <sycl/sycl.hpp>

#include "xenithium/kernels/encoder/vit.hpp"

namespace xenithium::kernels {

namespace {
class VitPosAddK;
class VitRopeK;
class VitLnK;
}  // namespace

sycl::event vit_pos_add(sycl::queue &q, int n, int hidden, const vit_bf16 *table, const int32_t *idx, const vit_bf16 *w, vit_bf16 *x) {
  return q.parallel_for<VitPosAddK>(sycl::range<2>(static_cast<size_t>(n), static_cast<size_t>(hidden)), [=](sycl::id<2> id) {
    const size_t p = id[0], h = id[1];
    float s = 0.f;
    for (int k = 0; k < 4; ++k) {
      const float e = static_cast<float>(vit_bf16(static_cast<float>(table[static_cast<size_t>(idx[p * 4 + k]) * hidden + h]) * static_cast<float>(w[p * 4 + k])));
      s = k == 0 ? e : s + e;
    }
    vit_bf16 &o = x[p * hidden + h];
    o = vit_bf16(static_cast<float>(o) + static_cast<float>(vit_bf16(s)));
  });
}

sycl::event vit_rope(sycl::queue &q, int n, int heads, int hd, const float *cs, const float *sn, vit_bf16 *qkv) {
  const int half = hd / 2;
  return q.parallel_for<VitRopeK>(sycl::range<3>(static_cast<size_t>(n), static_cast<size_t>(2 * heads), static_cast<size_t>(half)), [=](sycl::id<3> id) {
    const size_t p = id[0], h = id[1], i = id[2];
    vit_bf16 *b = qkv + p * 3 * heads * hd + h * hd;  // q heads, then k heads (consecutive in [3][heads][hd])
    const float x1 = static_cast<float>(b[i]), x2 = static_cast<float>(b[i + half]);
    const float c = cs[p * half + i], s = sn[p * half + i];
    b[i] = vit_bf16(sycl::fma(x1, c, -(x2 * s)));
    b[i + half] = vit_bf16(sycl::fma(x2, c, x1 * s));
  });
}

sycl::event vit_layer_norm(sycl::queue &q, int m, int n, const vit_bf16 *x, const float *gamma, const float *beta, float eps, vit_bf16 *y) {
  constexpr int L = 16;
  return q.submit([&](sycl::handler &h) {
    sycl::local_accessor<float, 1> part(sycl::range<1>(L), h);
    h.parallel_for<VitLnK>(sycl::nd_range<1>(static_cast<size_t>(m) * L, L), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
      const size_t r = it.get_group(0);
      const int l = static_cast<int>(it.get_local_id(0));
      const vit_bf16 *xr = x + r * n;
      float s = 0.f;
      for (int c = l; c < n; c += L) s += static_cast<float>(xr[c]);
      part[l] = s;
      sycl::group_barrier(it.get_group());
      float sum = 0.f;
      for (int k = 0; k < L; ++k) sum += part[k];
      namespace imf = sycl::ext::intel::math;  // IEEE division / sqrt (the plain operators are approximate on the card)
      const float mean = imf::fdiv_rn(sum, static_cast<float>(n));
      sycl::group_barrier(it.get_group());
      s = 0.f;
      for (int c = l; c < n; c += L) {
        const float d = static_cast<float>(xr[c]) - mean;
        s += d * d;
      }
      part[l] = s;
      sycl::group_barrier(it.get_group());
      float ss = 0.f;
      for (int k = 0; k < L; ++k) ss += part[k];
      const float rstd = imf::fdiv_rn(1.f, imf::fsqrt_rn(imf::fdiv_rn(ss, static_cast<float>(n)) + eps));
      for (int c = l; c < n; c += L) y[r * n + c] = vit_bf16(sycl::fma((static_cast<float>(xr[c]) - mean) * rstd, gamma[c], beta[c]));
    });
  });
}

}  // namespace xenithium::kernels
