// Xenithium · exp(x) for ESIMD epilogues that must match the SIMT swiglu kernel's sycl::exp (w4a16_rows.cpp).
#pragma once

#include <cmath>
#include <sycl/ext/intel/esimd.hpp>

namespace xenithium::kernels::expdev {

namespace esimd = sycl::ext::intel::esimd;

// exp(x) with the argument reduced first (Cody-Waite: x = n ln2 + r, |r| <= ln2 / 2), so the native exp only sees a
// small r and 2^n is exact: the SIMT swiglu's sycl::exp is the precise library exp, which the bare native esimd::exp
// misses by an ulp from |x| ~ 10.
template <int N>
inline esimd::simd<float, N> exp_cw(esimd::simd<float, N> x) SYCL_ESIMD_FUNCTION {
  const esimd::simd<float, N> x0 = x;
  x = esimd::max(esimd::min(x, esimd::simd<float, N>(88.f)), esimd::simd<float, N>(-103.f));
  const esimd::simd<float, N> n = esimd::rnde<float, N>(esimd::simd<float, N>(x * 1.44269504088896341f));
  const esimd::simd<float, N> r = (x - n * 0.693145751953125f) - n * 1.428606765330187e-06f;
  esimd::simd<float, N> e = esimd::exp(r);
  // 2^n in two steps (n can reach -150: past the normal range of one exponent field)
  const esimd::simd<int32_t, N> ni = n;
  const esimd::simd<int32_t, N> n1 = ni >> 1, n2 = ni - n1;
  esimd::simd<int32_t, N> b1 = (n1 + 127) << 23, b2 = (n2 + 127) << 23;
  e = e * b1.template bit_cast_view<float>().read();
  e = e * b2.template bit_cast_view<float>().read();
  e.merge(esimd::simd<float, N>(INFINITY), x0 > 88.7228394f);  // past FLT_MAX
  e.merge(esimd::simd<float, N>(0.f), x0 < -103.972084f);      // under the smallest subnormal
  return e;
}

}  // namespace xenithium::kernels::expdev
