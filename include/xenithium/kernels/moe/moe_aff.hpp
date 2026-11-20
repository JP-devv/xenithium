// Xenithium · Flint's MoE decode block (arch flint): router, sigmoid top-k, routed experts and the ungated
// shared expert in affine int4 group 64 (pack.hpp kW4AffGroup), weighted combine and the two-card sum, for one
// decode step of <= 8 rows (row shapes 4 / 8, rows.hpp).
// Semantics (Flint's sparse MoE block, from the model definition):
// logits = x. Wr^T in fp32 (bf16 x and router weight, fp32 sums) [experts] per row
// scores = sigmoid(logits); the top k by score (ties -> lowest expert id); weights = selected / their sum
// expert e: g, u = bf16(x. Wg^T), bf16(x. Wu^T); h = bf16(silu(g) u) (swiglu: kernels/gemm/mlp.hpp); y_e = h. Wd^T
// shared: the same MLP (inter 1024), added in full (no gate)
// out = bf16(sum_k w_k y_k + y_shared)
// Rounding points of this engine: g, u, h bf16; every y and the sums fp32 until the one final bf16.
// Tensor parallel over C cards (C = 1 or 2): card c holds columns [c I/C, (c+1) I/C) of every expert's (and the
// shared expert's) intermediate dimension: its gate / up rows and its k slice of down. Both cards run the same router
// and route; each computes its partial p_c = sum over the step's unique experts (ascending id) w y + y_shared and the
// pair sum (wire.hpp pair_reduce) gives out = bf16(p_0 + p_1) on both.
// Launches (grids fixed by (rows, shape)): router (dense_gemm, bf16 weights), route (one work-group: scores, top-k,
// weights, the unique-expert list -> MoeRoute), gate_up (per (slot, 16-column h tile): gate and up over the hidden k,
// k split over KS items, h = swiglu in the epilogue; the shared expert is the slot after the routed ones), down (per
// (slot, 16-column output tile): y = h. Wd^T fp32), combine (per (row, column): the weighted sum in slot order + the
// shared y, then the pair_reduce exchange in the same launch -> out).
#pragma once

#include <cstddef>
#include <cstdint>
#include <sycl/sycl.hpp>
#include <vector>

#include "xenithium/kernels/gemm/dense.hpp"
#include "xenithium/kernels/moe/moe.hpp"  // MoeRoute (kStepRows16: this MoE's rows)
#include "xenithium/kernels/comm/wire.hpp"

namespace xenithium::kernels {

struct MoeAffShape {
  int hidden = 0, inter = 0, experts = 0, topk = 0, shared_inter = 0;
  int cards = 1, card = 0;
  int inter_c() const { return inter / cards; }  // routed h columns on this card
  int shared_c() const { return shared_inter / cards; }  // shared h columns on this card
  int hmax() const { return inter_c() > shared_c() ? inter_c() : shared_c(); }
};

// One expert's per-card blob (byte offsets, every section 1 KiB aligned): gate | up as one W4 matrix [2 ic][hidden]
// (codes in the W4 block layout [hidden/128][2 ic/16][1024], gate rows first), its bf16 scales and biases
// [hidden/64][2 ic]; down [hidden][ic] (codes [ic/128][hidden/16][1024]), its scales and biases [ic/64][hidden].
// ic = the card's intermediate columns (inter_c, or shared_c for the shared expert); ic % 128 == 0.
struct MoeAffLayout {
  size_t gu_w = 0, gu_s = 0, gu_b = 0, dn_w = 0, dn_s = 0, dn_b = 0, bytes = 0;
  int ic = 0;
};
MoeAffLayout moe_aff_layout(int hidden, int ic);

// One expert as the checkpoint stores it (all cards' columns): gate / up U8 [inter][hidden/2] with bf16 scales and
// biases [inter][hidden/64]; down U8 [hidden][inter/2] with [hidden][inter/64].
struct MoeAffSrc {
  const uint8_t *gate_w = nullptr, *up_w = nullptr, *down_w = nullptr;
  const uint16_t *gate_s = nullptr, *gate_b = nullptr, *up_s = nullptr, *up_b = nullptr, *down_s = nullptr, *down_b = nullptr;
  int inter = 0;
};
// card's blob of an expert (or the shared expert: inter = shared_inter) into `blob` (lay.bytes)
void moe_aff_pack(int hidden, int cards, int card, const MoeAffSrc &src, const MoeAffLayout &lay, uint8_t *blob);

struct MoeAffWeights {
  DenseWeights router;  // bf16 [experts][hidden], packed by pack_dense_bf16
  const uint8_t *const *blobs = nullptr;  // device [experts]: this card's blob of each expert
  MoeAffLayout blob;  // routed experts' layout (ic = inter_c)
  const uint8_t *shared = nullptr;  // the shared expert's blob (device)
  MoeAffLayout shared_lay;  // ic = shared_c
  const float *silu_tab = nullptr;  // kernels:swiglu_table
};

struct MoeAffScratch {
  float *logits = nullptr;  // [8][experts]
  MoeRoute *route = nullptr;
  moe_bf16 *h = nullptr;  // [8 x topk + 1][8][hmax]
  float *y = nullptr;  // [8 x topk + 1][8][hidden]
  float *partial = nullptr;  // [8][hidden]: written only with two cards and no wire
};
bool moe_aff_route_esimd();  // the decode route as ESIMD (default) or the SIMT one
void moe_aff_set_route_esimd(int on);  // (tests: the two routes compared bit for bit)
size_t moe_aff_h_elems(const MoeAffShape &s);
size_t moe_aff_y_elems(const MoeAffShape &s);

struct MoeAffEvents {
  sycl::event router, route, gate_up, down, combine, reduce;  // decode: reduce == combine (one launch)
  sycl::event send;  // moe_aff_rows with PairRowsWire:send: the peer rows' copy
};

// x bf16 [shape][hidden] (row_shape(rows) rows readable, finite); out bf16 [rows][hidden]. wire: null with one card; with
// two cards and no wire, or out null, the step stops at this card's fp32 partial (sc.partial: tests, probes). The
// wire's inbox must hold hidden columns (pair_wire_words(n_max >= hidden)); parity alternates per call on both cards.
MoeAffEvents moe_aff_decode(sycl::queue &q, const MoeAffShape &s, int rows, const moe_bf16 *x, const MoeAffWeights &w, const MoeAffScratch &sc,
                            const PairWire *wire, int parity, moe_bf16 *out, int gu_ks = 0, int dn_ks = 0);

// ---- many rows (prefill; moe_aff_rows.cpp) ---------------------------------------------------------------------------
// The same block over m rows of a chunk (decode's semantics and rounding points: fp32 router logits, decode's top-k
// arithmetic per row, g / u / h bf16 with decode's silu table, y and every sum fp32 until the one final bf16): router
// (pf_dense), route, group (an expert-ordered list of the (row, k) assignments, segments of <= kMoeAffSegRows rows of one
// expert), x's row sums per 64-k group, gate_up per (segment, 16-row part, h tile), h's row sums, down per (segment, two output tiles),
// the shared expert over every row on the many-row affine GEMMs (w4aff_gemm_pf_swiglu / w4aff_gemm_pf: the reserved
// row-sum scratch of w4a16_pf_aff_reserve, rows x hidden), combine per (row, 64 columns): the row's experts in ascending
// id (fma), + the shared y -> the card's fp32 partial `part`; then the two-card sum (pair_reduce_rows on the wire, one card:
// out = bf16(part)) or, wire->split, the caller's split sum over `part` (with wire->send: the peer's rows combined first and
// their copy sent on the side queue). The affine folds are gs x s + asum x b as explicit fmas. Prefill and decode
// differ by fp32 order (the many-row GEMMs' group order; the shared y joins after the routed sum in both).
inline constexpr int kMoeAffSegRows = 32;
struct MoeAffRowsScratch {
  float *logits = nullptr;  // [rows_max][experts]
  int32_t *ids = nullptr;  // [rows_max][topk]
  float *topw = nullptr;  // [rows_max][topk]
  int32_t *slot = nullptr;  // [rows_max][topk]: the assignment's list position
  int32_t *row = nullptr;  // [rows_max x topk]: list position -> row
  int32_t *seg = nullptr;  // [moe_aff_rows_max_segs][4]: expert, first list position, rows, -
  int32_t *seg_n = nullptr;  // [16]: [0] = segments
  float *ax = nullptr;  // x's row sums [hidden / 64][lds_x]
  float *ah = nullptr;  // h's row sums [inter_c / 64][lds_h]
  int lds_x = 0, lds_h = 0;  // >= rows_max / rows_max x topk rounded up to 32
  moe_bf16 *h = nullptr;  // [rows_max x topk][inter_c]
  float *y = nullptr;  // [rows_max x topk][hidden]
  moe_bf16 *sh = nullptr;  // [rows_max][shared_c]
  float *sy = nullptr;  // [rows_max][hidden]
  float *part = nullptr;  // [rows_max][hidden]: the card's partial
  int rows_max = 0;
};
struct MoeAffRowsSizes {
  size_t logits = 0, ids = 0, row = 0, seg = 0, ax = 0, ah = 0, h = 0, y = 0, sh = 0, sy = 0, part = 0;  // elements (ids: also topw, slot)
};
int moe_aff_rows_max_segs(const MoeAffShape &s, int rows_max);
MoeAffRowsSizes moe_aff_rows_sizes(const MoeAffShape &s, int rows_max);
// x bf16 [m][hidden] (pitch hidden); out bf16 [m][hidden] (unused with wire->split). wire: null with one card.
MoeAffEvents moe_aff_rows(sycl::queue &q, const MoeAffShape &s, int m, const moe_bf16 *x, const MoeAffWeights &w, const MoeAffRowsScratch &sc,
                          const PairRowsWire *wire, int parity, moe_bf16 *out);
// Internal to moe_aff_rows (moe_aff_rows.cpp, 256 GRF): gate_up -> sc.h and down -> sc.y over the grouped segments.
sycl::event moe_aff_rows_gate_up(sycl::queue &q, const MoeAffShape &s, int max_segs, const moe_bf16 *x, const MoeAffWeights &w, const MoeAffRowsScratch &sc);
sycl::event moe_aff_rows_down(sycl::queue &q, const MoeAffShape &s, int max_segs, const MoeAffWeights &w, const MoeAffRowsScratch &sc);

// ---- host reference -------------------------------------------------------------------------------------------
// One expert dequantised (w = q s + b, all of `inter`): gate / up [inter][hidden], down [hidden][inter] (float).
struct MoeAffRefExpert {
  std::vector<float> gate, up, down;
};
MoeAffRefExpert moe_aff_dequant(int hidden, const MoeAffSrc &src);
// The route of one row: scores = sigmoid(logits) (logits in double), top k, renormalised.
void moe_aff_route_ref(const float *x, const float *router, int hidden, int experts, int topk, int32_t *ids, float *weights, double *logits = nullptr);
// The block for one row given its experts / weights (teacher-forced route): out (double) = sum w y + y_shared, with
// g, u, h rounded to bf16 as the device does (h_force: the device's h for the routed experts in `ids` order then the
// shared one, each [inter] (all cards' columns) - the down half checked from it; null = computed).
void moe_aff_ref(const float *x, int hidden, int topk, const int32_t *ids, const float *weights, const MoeAffRefExpert *const *experts,
                 const MoeAffRefExpert &shared, double *out, const std::vector<std::vector<float>> *h_force = nullptr);

}  // namespace xenithium:kernels
