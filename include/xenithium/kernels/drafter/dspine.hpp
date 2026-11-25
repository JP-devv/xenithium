// Xenithium · the DSpine block drafter's own kernels (`DFlashDraftModel` / `DSpineDraftModel` + the DSpine code bridge).
//
// DSpine (Tencent) drafts a block like DFlash: row 0 = the anchor (the last verified token), rows 1.. the mask token;
// five Qwen3 decoder layers whose attention is causal inside the block and reads the fc-projected target context in a
// sliding window. After every layer the code bridge passes each row's predicted feature to the next row:
//   ds_bridge_in   raw = bf16(res + add) (res updated), rms = sqrt(max(mean raw^2, FLT_MIN)), xr = bf16(raw / (rms + eps))
//   (GEMM)         p = xr . R_l^T                       (R_l [code][hidden], bf16; the engine's dense_gemm, fp32 out)
//   ds_codes       code = bf16(sqrt(code) bf16(p) / (|bf16(p)| + eps)); a segment's row 0: bf16(sqrt(code) codebook[anchor])
//   (GEMM)         packed = code . [G_q; G_p; A]^T      ([3 rank][code])
//   ds_message     msg_t = bf16(sigmoid(q_t + s_{t-1} + b) v_{t-1}) for t >= 1, msg_0 = 0 (q | s | v = bf16(packed) thirds)
//   (GEMM)         delta = msg . W_l^T                  (W_l [hidden][rank])
//   ds_bridge_out  res = bf16(raw + rms bf16(delta)) except a segment's row 0 (res = raw)
// and the next layer's input is rms_norm_plain(res, its input_layernorm) (the final norm after the last layer).
//   ds_select      the drafts (card 0), DFlash2's walk with DSpine's scores: for block row j >= 1 and the pick of row
//                  j - 1 (the anchor for j = 1), score(b) = logit_j(b) + sum_r sigmoid(q_j[r] + g[r] + bias[r]) v[r] TG[b][r]
//                  (fp32), (g | v) = TSV[pred] (bf16 [2 rank]) for a candidate pick, the last layer's packed row 0 (s | v
//                  thirds) for the anchor; q_j = the last layer's packed row j (q third). TG [n][rank] / TSV [n][2 rank]:
//                  the vocabulary tables (built at load from the target's lm_head rows and the codebook; candidate index
//                  -> table row as df_select's token map but without it: tables are indexed by the candidate index).
//                  The model defines the score as rms_j (base(b) + s(b)) with base = <raw_j x norm_w, head_b>, raw = the
//                  last layer's pre-write row. The head's logits come from the post-write row out = raw + rms delta,
//                  delta = msg W^T, so base(b) = rms_out logit(b) - rms <msg, TG[b]> (rms_out = sqrt(mean out^2 + eps),
//                  the final norm's): the kernel scores (rms_out / rms) logit(b) + <m - msg, TG[b]> = the model's
//                  score / rms, up to bf16 rounding (drafts only: verification is exact either way).
// Semantics: the checkpoint's config.json + tensors. Rows: segs segments of seg_rows rows each, segment i's row 0 at
// row i x seg_rows.
#pragma once

#include <cstdint>

#include <sycl/sycl.hpp>

#include "xenithium/kernels/drafter/dflash.hpp"
#include "xenithium/kernels/head/head.hpp"

namespace xenithium::kernels {

using ds_bf16 = sycl::ext::oneapi::bfloat16;

inline constexpr int kDsCode = 1024;  // code size (kernel constant: the checkpoint's selected_prefix_code_bridge.code_size)
inline constexpr int kDsRank = 512;   // bridge rank (kernel constant)

sycl::event ds_bridge_in(sycl::queue &q, int rows, int hidden, ds_bf16 *res, const ds_bf16 *add, float eps, float *rms, ds_bf16 *xr);
void ds_bridge_in_ref(int rows, int hidden, uint16_t *res, const uint16_t *add, float eps, float *rms, uint16_t *xr);
// p fp32 [rows][kDsCode]; anchors int32 [segs] (device); codebook bf16 [vocab][kDsCode] (device or pinned host); code out bf16.
sycl::event ds_codes(sycl::queue &q, int rows, int seg_rows, const float *p, const int32_t *anchors, const ds_bf16 *codebook, float eps, ds_bf16 *code);
void ds_codes_ref(int rows, int seg_rows, const float *p, const int32_t *anchors, const uint16_t *codebook, float eps, uint16_t *code);
// packed fp32 [rows][3 kDsRank]; bias bf16 [kDsRank]; msg bf16 [rows][kDsRank]
sycl::event ds_message(sycl::queue &q, int rows, int seg_rows, const float *packed, const ds_bf16 *bias, ds_bf16 *msg);
void ds_message_ref(int rows, int seg_rows, const float *packed, const uint16_t *bias, uint16_t *msg);
// res: raw in, out out; rms [rows]; delta fp32 [rows][hidden]
sycl::event ds_bridge_out(sycl::queue &q, int rows, int seg_rows, int hidden, ds_bf16 *res, const float *rms, const float *delta);
void ds_bridge_out_ref(int rows, int seg_rows, int hidden, uint16_t *res, const float *rms, const float *delta);
// keys [rows][16] (df_topk, best first); packed fp32 [rows][3 kDsRank] (the last layer's); tsv bf16 [n][2 kDsRank], tg bf16
// [n][kDsRank]; bias bf16 [kDsRank]; candidate index -> token = v < sub_k ? v : v - sub_k + sub_s0 (sub_k 0: identity);
// drafts int32 [segs][seg_rows - 1]; noise (optional, device [segs]): DFlash's coupled picks, inv_t 0 = argmax.
// out: the last bridge's rows bf16 [rows][hidden] (the head's input before the final norm), rms: its pre-write rms [rows],
// msg: its message bf16 [rows][kDsRank]; eps: the final norm's.
struct DsTail {
  const ds_bf16 *out = nullptr;
  int hidden = 0;
  float eps = 0;
  const float *rms = nullptr;
  const ds_bf16 *msg = nullptr;
};
sycl::event ds_select(sycl::queue &q, int segs, int seg_rows, const uint64_t *keys, const float *packed, const ds_bf16 *tsv, const ds_bf16 *tg,
                      const ds_bf16 *bias, const DsTail &tail, int sub_k, int sub_s0, int32_t *drafts, const DraftNoiseRec *noise);
// host reference of one segment (pointers at its row 0; out / msg as bf16 bits)
void ds_select_ref(int seg_rows, const uint64_t *keys, const float *packed, const uint16_t *tsv, const uint16_t *tg, const uint16_t *bias, const uint16_t *out,
                   int hidden, float eps, const float *rms, const uint16_t *msg, int sub_k, int sub_s0, int32_t *drafts, const DraftNoiseRec *noise);
// DDTree's best-first draft tree with DSpine's scores (df_tree's walk and output words,
// kDfTreeInts per segment): a node at depth d < seg_rows - 1 has block row d + 1's 16 candidates, each scored as
// ds_select scores it given the node's pick (the root: the anchor's packed row 0), as log-softmax over the 16, best
// first; then up to `budget` times the frontier's best joins (cut > 0: not below log(cut)). Greedy only (no noise).
// inv_temp: df_tree's (the scores x inv_temp before the log-softmax).
sycl::event ds_tree(sycl::queue &q, int segs, int seg_rows, const uint64_t *keys, const float *packed, const ds_bf16 *tsv, const ds_bf16 *tg, const ds_bf16 *bias,
                    const DsTail &tail, int sub_k, int sub_s0, int budget, float cut, int32_t *tree, float inv_temp = 1.0f);
void ds_tree_ref(int seg_rows, const uint64_t *keys, const float *packed, const uint16_t *tsv, const uint16_t *tg, const uint16_t *bias, const uint16_t *out, int hidden,
                 float eps, const float *rms, const uint16_t *msg, int sub_k, int sub_s0, int budget, float cut, int32_t *tree, float inv_temp = 1.0f);
// The table build's activations: dst[r][i] = bf16(bf16(src[r][i]) x w[i]) (w null: x scale); bf16 [rows][n].
sycl::event ds_scale_rows(sycl::queue &q, int rows, int n, const ds_bf16 *src, const ds_bf16 *w, float scale, ds_bf16 *dst);

}  // namespace xenithium::kernels
