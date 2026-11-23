#include "xenithium/kernels/residual/rmsnorm.hpp"

#include <cmath>
#include <functional>
#include <stdexcept>

#include <sycl/ext/intel/esimd.hpp>

#include "xenithium/kernels/pack.hpp"

namespace xenithium::kernels {
namespace {

namespace esimd = sycl::ext::intel::esimd;
constexpr int kC = kRmsNormCols;
constexpr int kSlots = 64;  // most items per row (SLM sum slots)

// none: res as is; bf16 / f32: res' = bf16(res + add)
enum class Add : int { none, bf16, f32 };

// PLAIN: 0 = bf16(y x (1 + w)), 1 = bf16(w x bf16(y)) (LlamaRMSNorm / Qwen3RMSNorm), 2 = bf16(w x y) (FlintRMSNorm: the
// weight times the fp32 normed value, one rounding)
template <Add A, int PLAIN, bool Q8>
class RmsNormK;
class RmsQ8ScalesColsK;

template <Add A, int PLAIN, bool Q8 = false>
sycl::event launch(sycl::queue &q, int rows, int H, rn_bf16 *res, const rn_bf16 *add_b, const float *add_f, const rn_bf16 *w, float eps,
                   rn_bf16 *xn, RmsNormQ8 q8 = {}) {
  const int ni = H / kC;
  const float inv_h = 1.f / static_cast<float>(H);
  return q.parallel_for<RmsNormK<A, PLAIN, Q8>>(sycl::nd_range<1>(static_cast<size_t>(rows) * ni, ni), [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
    esimd::slm_init<kSlots * 4>();
    const int r = static_cast<int>(it.get_group(0)), i = static_cast<int>(it.get_local_id(0));
    const size_t off = static_cast<size_t>(r) * H + static_cast<size_t>(i) * kC;
    esimd::simd<float, kC> v = esimd::block_load<rn_bf16, kC>(res + off);
    if constexpr (A != Add::none) {
      esimd::simd<float, kC> a;
      if constexpr (A == Add::bf16) {
        a = esimd::block_load<rn_bf16, kC>(add_b + off);
      } else if constexpr (A == Add::f32) {
        a = esimd::block_load<float, kC>(add_f + off);
      }
      const esimd::simd<rn_bf16, kC> rb = v + a;
      esimd::block_store<rn_bf16, kC>(res + off, rb);
      v = rb;
    }
    const esimd::simd<float, kC> wv = esimd::block_load<rn_bf16, kC>(w + static_cast<size_t>(i) * kC);
    esimd::slm_scalar_store<float>(static_cast<uint32_t>(i) * 4, esimd::reduce<float>(v * v, std::plus<>()));
    if (i == 0)
      for (int s = ni; s < kSlots; ++s) esimd::slm_scalar_store<float>(static_cast<uint32_t>(s) * 4, 0.f);
    esimd::barrier();
    const esimd::simd<float, kSlots> ss = esimd::slm_block_load<float, kSlots>(0);
    const float m = esimd::reduce<float>(ss, std::plus<>()) * inv_h + eps;
    const float rr = esimd::div_ieee(esimd::simd<float, 1>(1.f), esimd::sqrt_ieee(esimd::simd<float, 1>(m)))[0];
    const esimd::simd<float, kC> y = v * rr;
    esimd::simd<rn_bf16, kC> o;
    if constexpr (PLAIN == 1) {  // Qwen3RMSNorm: bf16(w x bf16(y))
      const esimd::simd<float, kC> yb = esimd::simd<rn_bf16, kC>(y);
      o = yb * wv;
    } else if constexpr (PLAIN == 2) {  // FlintRMSNorm: bf16(w x y)
      o = y * wv;
    } else {
      o = y * wv + y;  // (1 + w)
    }
    if constexpr (Q8) {
      if (!q8.codes_only) esimd::block_store<rn_bf16, kC>(xn + off, o);
    } else {
      esimd::block_store<rn_bf16, kC>(xn + off, o);
    }
    if constexpr (Q8) {  // w4a16_rows.cpp quant_2d16's rule on the bf16 xn
      const esimd::simd<float, kC> v8 = o;
      const float amax = esimd::hmax<float>(esimd::abs(v8));
      esimd::simd<int8_t, kC> qv(0);
      float sv = 0.f;
      if (amax > 0.f) {
        const float inv = esimd::div_ieee(127.f, amax);
        sv = esimd::div_ieee(amax, 127.f);
        const esimd::simd<float, kC> t = esimd::rnde<float, kC>(v8 * inv);
        qv = esimd::simd<int8_t, kC>(esimd::max(esimd::min(t, 127.f), -127.f));
      }
      esimd::block_store<int8_t, kC>(q8.q + off, qv);
      q8.s[static_cast<size_t>(i) * q8.lds + r] = sv;
      if (q8.s_rows) q8.s_rows[static_cast<size_t>(r) * ni + i] = sv;
    }
  });
}

void check(int rows, int hidden) {
  if (rows < 1 || hidden < kC || hidden % kC || hidden / kC > kSlots) throw std::invalid_argument("rms_norm: rows >= 1, hidden a multiple of 128 up to 8,192");
}

}  // namespace

sycl::event rms_norm(sycl::queue &q, int rows, int hidden, rn_bf16 *res, const rn_bf16 *add_bf16, const float *add_f32, const rn_bf16 *w,
                     float eps, rn_bf16 *xn, int plain) {
  check(rows, hidden);
  if (!res || !w || !xn) throw std::invalid_argument("rms_norm: buffers");
  if (add_bf16 && add_f32) throw std::invalid_argument("rms_norm: at most one pending add");
  if (plain < 0 || plain > 2) throw std::invalid_argument("rms_norm: plain 0..2");
  if (plain == 1) {  // arch llama: bf16(w x bf16(y))
    if (add_bf16) return launch<Add::bf16, 1>(q, rows, hidden, res, add_bf16, nullptr, w, eps, xn);
    if (add_f32) return launch<Add::f32, 1>(q, rows, hidden, res, nullptr, add_f32, w, eps, xn);
    return launch<Add::none, 1>(q, rows, hidden, res, nullptr, nullptr, w, eps, xn);
  }
  if (plain == 2) {  // arch flint: bf16(w x y)
    if (add_bf16) return launch<Add::bf16, 2>(q, rows, hidden, res, add_bf16, nullptr, w, eps, xn);
    if (add_f32) return launch<Add::f32, 2>(q, rows, hidden, res, nullptr, add_f32, w, eps, xn);
    return launch<Add::none, 2>(q, rows, hidden, res, nullptr, nullptr, w, eps, xn);
  }
  if (add_bf16) return launch<Add::bf16, 0>(q, rows, hidden, res, add_bf16, nullptr, w, eps, xn);
  if (add_f32) return launch<Add::f32, 0>(q, rows, hidden, res, nullptr, add_f32, w, eps, xn);
  return launch<Add::none, 0>(q, rows, hidden, res, nullptr, nullptr, w, eps, xn);
}

sycl::event rms_norm_q8(sycl::queue &q, int rows, int hidden, rn_bf16 *res, const rn_bf16 *add_bf16, const float *add_f32, const rn_bf16 *w,
                        float eps, rn_bf16 *xn, const RmsNormQ8 &q8, int plain) {
  check(rows, hidden);
  if (!res || !w || !xn || !q8.q || !q8.s || q8.lds < rows) throw std::invalid_argument("rms_norm_q8: buffers");
  if (add_bf16 && add_f32) throw std::invalid_argument("rms_norm_q8: at most one pending add");
  if (plain == 2) throw std::invalid_argument("rms_norm_q8: the one-rounding plain norm (arch flint) has no W4A8 codes path");
  if (plain == 1) {
    if (add_bf16) return launch<Add::bf16, 1, true>(q, rows, hidden, res, add_bf16, nullptr, w, eps, xn, q8);
    if (add_f32) return launch<Add::f32, 1, true>(q, rows, hidden, res, nullptr, add_f32, w, eps, xn, q8);
    return launch<Add::none, 1, true>(q, rows, hidden, res, nullptr, nullptr, w, eps, xn, q8);
  }
  if (add_bf16) return launch<Add::bf16, 0, true>(q, rows, hidden, res, add_bf16, nullptr, w, eps, xn, q8);
  if (add_f32) return launch<Add::f32, 0, true>(q, rows, hidden, res, nullptr, add_f32, w, eps, xn, q8);
  return launch<Add::none, 0, true>(q, rows, hidden, res, nullptr, nullptr, w, eps, xn, q8);
}

sycl::event rms_q8_scales_cols(sycl::queue &q, int rows, int groups, const float *s_rows, float *s, int lds) {
  if (rows < 1 || groups < 1 || !s_rows || !s || lds < rows) throw std::invalid_argument("rms_q8_scales_cols: shape / buffers");
  return q.parallel_for<RmsQ8ScalesColsK>(sycl::range<2>(static_cast<size_t>(groups), static_cast<size_t>(rows)), [=](sycl::item<2> it) {
    const size_t g = it.get_id(0), r = it.get_id(1);
    s[g * static_cast<size_t>(lds) + r] = s_rows[r * static_cast<size_t>(groups) + g];
  });
}

sycl::event rms_norm_plain(sycl::queue &q, int rows, int hidden, rn_bf16 *res, const rn_bf16 *add_bf16, const rn_bf16 *w, float eps, rn_bf16 *xn) {
  check(rows, hidden);
  if (!res || !w || !xn) throw std::invalid_argument("rms_norm_plain: buffers");
  if (add_bf16) return launch<Add::bf16, 1>(q, rows, hidden, res, add_bf16, nullptr, w, eps, xn);
  return launch<Add::none, 1>(q, rows, hidden, res, nullptr, nullptr, w, eps, xn);
}

void rms_norm_plain_ref(int rows, int hidden, uint16_t *res, const uint16_t *add_bf16, const uint16_t *w, float eps, uint16_t *xn) {
  for (int r = 0; r < rows; ++r) {
    uint16_t *x = res + static_cast<size_t>(r) * hidden;
    double ss = 0;
    for (int i = 0; i < hidden; ++i) {
      float v = bf16_value(x[i]);
      if (add_bf16) v = bf16_value(x[i] = bf16_bits(v + bf16_value(add_bf16[static_cast<size_t>(r) * hidden + i])));
      ss += static_cast<double>(v) * v;
    }
    const float rr = 1.f / std::sqrt(static_cast<float>(ss / hidden) + eps);
    for (int i = 0; i < hidden; ++i) {
      const float y = bf16_value(bf16_bits(bf16_value(x[i]) * rr));
      xn[static_cast<size_t>(r) * hidden + i] = bf16_bits(y * bf16_value(w[i]));
    }
  }
}

void rms_norm_ref(int rows, int hidden, uint16_t *res, const uint16_t *add_bf16, const float *add_f32, const uint16_t *w, float eps,
                  uint16_t *xn) {
  for (int r = 0; r < rows; ++r) {
    uint16_t *x = res + static_cast<size_t>(r) * hidden;
    double ss = 0;
    for (int i = 0; i < hidden; ++i) {
      float v = bf16_value(x[i]);
      if (add_bf16) v = bf16_value(x[i] = bf16_bits(v + bf16_value(add_bf16[static_cast<size_t>(r) * hidden + i])));
      else if (add_f32) v = bf16_value(x[i] = bf16_bits(v + add_f32[static_cast<size_t>(r) * hidden + i]));
      ss += static_cast<double>(v) * v;
    }
    const float rr = 1.f / std::sqrt(static_cast<float>(ss / hidden) + eps);
    for (int i = 0; i < hidden; ++i) {
      const float y = bf16_value(x[i]) * rr;
      xn[static_cast<size_t>(r) * hidden + i] = bf16_bits(std::fma(y, bf16_value(w[i]), y));
    }
  }
}

}  // namespace xenithium:kernels
