// Xenithium · rotate + int8 quantise one N-block, shared by the decode MoE's own launch, the
// decode router's extra work-groups and the prefill quantiser, so every caller gives the same bits.
// ROT: v = bf16(H_N v) exactly as moe_hadamard_rows (stage-by-stage butterflies a + c, a - c on the same
// pairs, the 1 / sqrt(N) scale, bf16 rounding). Then row_quant.hpp's rule: s = amax / 127, q = rne(v x
// 127 / amax) clamped to +-127; an all-zero block q = 0, s = 0. Item `id` of rows x (cols / N) owns one
// block: src / dst at stride ld, scl[id] row-major [rows][cols / N].
#pragma once

#include <cstddef>
#include <cstdint>

#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

namespace xenithium::kernels {

namespace rq_detail {
namespace esimd = sycl::ext::intel::esimd;
template <int N, int Hh>
inline void hadamard_stage(esimd::simd<float, N> &v) SYCL_ESIMD_FUNCTION {
  auto mv = v.template bit_cast_view<float, N / (2 * Hh), 2 * Hh>();
  const esimd::simd<float, N / 2> a = mv.template select<N / (2 * Hh), 1, Hh, 1>(0, 0);
  const esimd::simd<float, N / 2> c = mv.template select<N / (2 * Hh), 1, Hh, 1>(0, Hh);
  mv.template select<N / (2 * Hh), 1, Hh, 1>(0, 0) = a + c;
  mv.template select<N / (2 * Hh), 1, Hh, 1>(0, Hh) = a - c;
  if constexpr (2 * Hh < N) hadamard_stage<N, 2 * Hh>(v);
}
}  // namespace rq_detail

// bf16(H_N v) in registers (v holds bf16 values as float on entry and on return).
template <int N>
inline void rot_block(sycl::ext::intel::esimd::simd<float, N> &v) SYCL_ESIMD_FUNCTION {
  namespace esimd = sycl::ext::intel::esimd;
  rq_detail::hadamard_stage<N, 1>(v);
  const float sc = N == 64 ? 0.125f : 0.08838834764831845f;  // 1 / sqrt(N)
  v *= sc;
  v = esimd::simd<float, N>(esimd::simd<sycl::ext::oneapi::bfloat16, N>(v));
}

// int8 codes + scale of one block of bf16 values (as float).
template <int N>
inline sycl::ext::intel::esimd::simd<int8_t, N> quant_block(sycl::ext::intel::esimd::simd<float, N> v, float &sv) SYCL_ESIMD_FUNCTION {
  namespace esimd = sycl::ext::intel::esimd;
  const float amax = esimd::hmax<float>(esimd::abs(v));
  esimd::simd<int8_t, N> qv(0);
  sv = 0.f;
  if (amax > 0.f) {
    const float inv = esimd::div_ieee(127.f, amax);
    sv = esimd::div_ieee(amax, 127.f);
    const esimd::simd<float, N> t = esimd::rnde<float, N>(v * inv);
    qv = esimd::simd<int8_t, N>(esimd::max(esimd::min(t, 127.f), -127.f));
  }
  return qv;
}

template <int N, bool ROT>
inline void rot_quant_item(const sycl::ext::oneapi::bfloat16 *src, int ld, int8_t *dst, float *scl, size_t id, size_t nb) SYCL_ESIMD_FUNCTION {
  namespace esimd = sycl::ext::intel::esimd;
  const size_t r = id / nb, b = id % nb;
  esimd::simd<float, N> v = esimd::block_load<sycl::ext::oneapi::bfloat16, N>(src + r * static_cast<size_t>(ld) + b * N);
  if constexpr (ROT) rot_block<N>(v);
  float sv;
  const esimd::simd<int8_t, N> qv = quant_block<N>(v, sv);
  esimd::block_store<int8_t, N>(dst + r * static_cast<size_t>(ld) + b * N, qv);
  scl[id] = sv;
}

// A rotate + quantise job riding in another launch's extra work-groups: rows x cols / N blocks.
struct RotQuantJob {
  const sycl::ext::oneapi::bfloat16 *src = nullptr;
  int8_t *dst = nullptr;
  float *scl = nullptr;
  int rows = 0, cols = 0, ld = 0;
  bool rot = false;
};

}  // namespace xenithium:kernels
