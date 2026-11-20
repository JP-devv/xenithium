// Xenithium · MTP drafter front: decode rows (1..4 positions) or a prefill chunk's rows (any count;
// both launches are per row / per element, so a chunk's rows equal its 4-row slices').
//
// A drafter pass at position p takes the target's multi-stream state h_p [S][hidden] (after its
// final mixer's combine) and the token after p (semantics from the drafter checkpoint's config
// and tensors; written to match tools/mtp_ref):
//   e   = bf16(gemma_rmsnorm(embed[t], pre_fc_norm_embedding))     over hidden
//   hn  = bf16(gemma_rmsnorm(h_p, pre_fc_norm_hidden))             one norm over all S x hidden
//   x_s = bf16(bf16(e . Wfe^T) + bf16(hn_s . Wfh^T))               fc_hidden shared by the streams
// gemma_rmsnorm: y = x / sqrt(mean x^2 + eps) (fp32), out = bf16(fma(y, w, y)) (the (1 + w) affine).
// The rest of the pass is the target's kernels: hc_mixer, qsa_decode, moe_decode, the final mixer,
// lm_head + pick.
//
// Launches: front_norm (one work-group per row and input), then the projections as dense GEMMs
// (the caller: e . Wfe^T once, hn . Wfh^T once per row with the row's S streams as the GEMM's 4
// rows), then front_add.
#pragma once

#include <sycl/sycl.hpp>

namespace xenithium::kernels {

using mtp_bf16 = sycl::ext::oneapi::bfloat16;

// e bf16 [rows][H] (the gathered embeddings), h bf16 rows of S*H at stride S*H; norm_e [H],
// norm_h [S*H]. en bf16 [rows][H], hn bf16 [rows][S][H].
sycl::event mtp_front_norm(sycl::queue &q, int H, int S, int rows, const mtp_bf16 *e, const mtp_bf16 *h, const mtp_bf16 *norm_e,
                           const mtp_bf16 *norm_h, float eps, mtp_bf16 *en, mtp_bf16 *hn);
// fe fp32 [rows][H] (e . Wfe^T, GEMM row stride H), fh fp32 [rows][S][H] -> res bf16 [rows][S][H].
sycl::event mtp_front_add(sycl::queue &q, int H, int S, int rows, const float *fe, const float *fh, mtp_bf16 *res);

}  // namespace xenithium::kernels
