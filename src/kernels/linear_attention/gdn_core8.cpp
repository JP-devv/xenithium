// Xenithium · the GDN decode core's 8-row instances (an 8-row verify of one sequence for the DFlash 2
// drafter). In the 256-GRF library: x rows, a / b accumulators and the 16 staged rows outgrow 128 GRF.
#include "linear_attention/gdn_core.hpp"

namespace xenithium::kernels::gdn_detail {

template <bool SWISH>
sycl::event core_launch8(sycl::queue &q, const GdnShape &s, int rows, const bf16 *x, const GdnWeights &w, const GdnState &st, int parity, const float *ystep,
                         bf16 *core_out) {
  return core_launch_km<SWISH, kGdnMaxRows>(q, s, rows, x, w, st, parity, ystep, core_out);
}

template <bool SWISH>
sycl::event core_launch8_tree(sycl::queue &q, const GdnShape &s, int rows, const bf16 *x, const GdnWeights &w, const GdnState &st, int parity,
                              const float *ystep, bf16 *core_out) {
  return core_abi<SWISH, kGdnMaxRows, kGdnMaxRows, kGdnMaxRows, true>(q, s, rows, x, w, st, parity, ystep, core_out);
}

template sycl::event core_launch8_tree<true>(sycl::queue &, const GdnShape &, int, const bf16 *, const GdnWeights &, const GdnState &, int, const float *, bf16 *);
template sycl::event core_launch8_tree<false>(sycl::queue &, const GdnShape &, int, const bf16 *, const GdnWeights &, const GdnState &, int, const float *, bf16 *);
template sycl::event core_launch8<true>(sycl::queue &, const GdnShape &, int, const bf16 *, const GdnWeights &, const GdnState &, int, const float *, bf16 *);
template sycl::event core_launch8<false>(sycl::queue &, const GdnShape &, int, const bf16 *, const GdnWeights &, const GdnState &, int, const float *, bf16 *);

}  // namespace xenithium:kernels:gdn_detail
