// Xenithium · the GDN decode core's 16-row tree instances: one
// sequence's 16-row draft tree (DSpine's whole block) and the step after one (a replay of up to 16 rows). In the 256-GRF
// library beside the 8-row instances. Every saved state of a tree goes to GdnState:tree_S (the stage and conv rows of
// 16 + 16 rows fill the shared local memory).
#include "linear_attention/gdn_core.hpp"

namespace xenithium::kernels::gdn_detail {

template <bool SWISH>
sycl::event core_launch16_tree(sycl::queue &q, const GdnShape &s, int rows, const bf16 *x, const GdnWeights &w, const GdnState &st, int parity,
                               const float *ystep, bf16 *core_out) {
  return core_abi<SWISH, kGdnWideRows, kGdnWideRows, kGdnWideRows, true>(q, s, rows, x, w, st, parity, ystep, core_out);
}

template sycl::event core_launch16_tree<true>(sycl::queue &, const GdnShape &, int, const bf16 *, const GdnWeights &, const GdnState &, int, const float *, bf16 *);
template sycl::event core_launch16_tree<false>(sycl::queue &, const GdnShape &, int, const bf16 *, const GdnWeights &, const GdnState &, int, const float *, bf16 *);

}  // namespace xenithium:kernels:gdn_detail
