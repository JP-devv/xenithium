// Xenithium · int8 activation quantiser body for W4A8, shared by moe_rows' own launch and
// pf_dense_pair's third body, so both give the same bits.
// Per (row, G-group) of a bf16 [rows][cols] matrix: s = amax / 127, q = rne(v * (127 / amax)) clamped
// to [-127, 127]; an all-zero group gives q = 0, s = 0. Item `id` of n = rows * (cols / G) owns one
// group; scl[id] is its scale (row-major [rows][cols / G]).
#pragma once

#include <cstddef>
#include <cstdint>

#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

namespace xenithium::kernels {

template <int G>
inline void row_quant_item(const sycl::ext::oneapi::bfloat16 *src, int cols, int8_t *dst, float *scl, size_t id, size_t n) SYCL_ESIMD_FUNCTION {
  namespace esimd = sycl::ext::intel::esimd;
  if (id >= n) return;
  const size_t gpr = static_cast<size_t>(cols / G);
  const size_t r = id / gpr, g = id % gpr;
  const esimd::simd<float, G> v = esimd::block_load<sycl::ext::oneapi::bfloat16, G>(src + r * cols + g * G);
  const float amax = esimd::hmax<float>(esimd::abs(v));
  esimd::simd<int8_t, G> qv(0);
  float sv = 0.f;
  if (amax > 0.f) {
    const float inv = esimd::div_ieee(127.f, amax);
    sv = esimd::div_ieee(amax, 127.f);
    const esimd::simd<float, G> t = esimd::rnde<float, G>(v * inv);
    qv = esimd::simd<int8_t, G>(esimd::max(esimd::min(t, 127.f), -127.f));
  }
  esimd::block_store<int8_t, G>(dst + r * cols + g * G, qv);
  scl[id] = sv;
}

// W8A8 dense prefill: the same rule over a [rows][cols] matrix whose row stride is `src_ld`
// (bf16 elements) and whose last group may be short (cols % G, a multiple of 32: the dense trunk's
// k = 320 has a 64-wide tail), NR rows x one group per item (the rows' loads issued together). Items id = rb * gpr + g
// over ceil(rows / NR) x gpr, gpr = ceil(cols / G). dst is [rows][cols] (row stride cols); the scales
// group-major, scl[g * scl_ld + r] (dense.hpp's PfA8). A full group computes exactly row_quant_item's
// expressions (same bits); a short group sees zeros past its end, which change neither the amax nor the
// codes it stores.
template <int G, int NR>
inline void row_quant_rows(const sycl::ext::oneapi::bfloat16 *src, int src_ld, int rows, int cols, int8_t *dst, float *scl, int scl_ld,
                           size_t id) SYCL_ESIMD_FUNCTION {
  namespace esimd = sycl::ext::intel::esimd;
  static_assert(G % 32 == 0, "row_quant_rows: G in 32-wide pieces");
  const size_t gpr = static_cast<size_t>((cols + G - 1) / G);
  const size_t rb = id / gpr, g = id % gpr, r0 = rb * NR;
  if (r0 >= static_cast<size_t>(rows)) return;
  const int rem = cols - static_cast<int>(g) * G;
  const int width = rem < G ? rem : G;
  esimd::simd<float, G * NR> v(0.f);
#pragma unroll
  for (int i = 0; i < NR; ++i) {
    const size_t r = r0 + i;
    if (r >= static_cast<size_t>(rows)) break;
    const sycl::ext::oneapi::bfloat16 *s = src + r * static_cast<size_t>(src_ld) + g * G;
    if (width == G) {
      v.template select<G, 1>(i * G) = esimd::block_load<sycl::ext::oneapi::bfloat16, G>(s);
    } else {
#pragma unroll
      for (int p = 0; p < G / 32; ++p)
        if (p * 32 < width) v.template select<32, 1>(i * G + p * 32) = esimd::block_load<sycl::ext::oneapi::bfloat16, 32>(s + p * 32);
    }
  }
#pragma unroll
  for (int i = 0; i < NR; ++i) {
    const size_t r = r0 + i;
    if (r >= static_cast<size_t>(rows)) break;
    const esimd::simd<float, G> vg = v.template select<G, 1>(i * G);
    const float amax = esimd::hmax<float>(esimd::abs(vg));
    esimd::simd<int8_t, G> qv(0);
    float sv = 0.f;
    if (amax > 0.f) {
      const float inv = esimd::div_ieee(127.f, amax);
      sv = esimd::div_ieee(amax, 127.f);
      const esimd::simd<float, G> t = esimd::rnde<float, G>(vg * inv);
      qv = esimd::simd<int8_t, G>(esimd::max(esimd::min(t, 127.f), -127.f));
    }
    int8_t *d = dst + r * static_cast<size_t>(cols) + g * G;
    if (width == G) {
      esimd::block_store<int8_t, G>(d, qv);
    } else {
#pragma unroll
      for (int p = 0; p < G / 32; ++p)
        if (p * 32 < width) esimd::block_store<int8_t, 32>(d + p * 32, qv.template select<32, 1>(p * 32));
    }
    scl[g * static_cast<size_t>(scl_ld) + r] = sv;
  }
}

}  // namespace xenithium:kernels
