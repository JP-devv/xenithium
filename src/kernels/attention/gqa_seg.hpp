// Xenithium · dense gated GQA decode: the attention stage with one K / V pass per row group.
// Private to gqa.cpp / gqa_seg.cpp (the template is instantiated at 128 GRF in the static kernels and
// at 256 GRF in the xl library; gqa_decode picks per group size: gqa_attn_group); gqa_sel.cpp's page scores use its
// bf16 DPAS helpers (dpas_rows, seg_k_operand).
//
// One work-group of 16 items per (row group, class, kv head, context split) carries every row's q heads of the group
// (NH = RM x G DPAS rows) and reads each K / V tile (one 64-token page of this kv head) once for all of them:
//   - item i = (dim quarter dq = i % 4, token quarter tq = i / 4) loads its 16 tokens x 64 dims of K and of V with one
//     2D block load each (int8: 16 x 64 bytes; bf16: two 16 x 32 blocks), widens to bf16 per 16-dim chunk, and
//     computes the q.k partial of every head over its 64 dims (4 DPAS steps accumulated in C);
//   - the four partials of a (head, 16 tokens) are summed through SLM in dq order (from 0) by the item owning the head
//     (heads i, i + 16, ...), which runs the tile's online softmax for its head in registers (scale 1/16, x the K scale
//     for int8; m' = max(m, tile max), alpha = exp(m - m'), l' = l alpha + sum p) and writes P (x the V scale) and alpha;
//   - every item rescales its o accumulators [NH][its 64 dims] by alpha and adds P (hi + lo bf16 parts) . V of its 16
//     tokens on DPAS; at the end of the split the four token quarters' o are summed through SLM in tq order and the
//     (o, max, sum) of every row active in the split go to gqa.cpp's part rows (gqa.cpp's merge finishes them);
//   - 2 barriers per tile; the item's K and V blocks of a tile are both loaded before its q.k (V's latency hides behind
//     the softmax). No L1 prefetch of the next tile.
//   - pf: the first row group of a segment prefetches the next tile's K and V blocks into L2 only (L1 uncached) before
//     the tile's first barrier, so multi-row groups, which are compute-heavy per tile, overlap the next tile's VRAM
//     reads with this tile's softmax and P.V (the second group of an 8-row segment then reads them from L2). Loads
//     only, so the bits do not depend on it; gqa_decode picks it per group size and context.
// Per row: the split cut is the row's own (tps = ceil(ntiles / 64)); rows whose tps differs from the group's first
// row run in a second work-group (class 1: rows of one sequence span <= 8 positions, so <= 2 classes). Inside a split a
// row walks exactly its own tiles and tokens: on a tile past its last one its scores are -inf, p = 0 and alpha = 1;
// 0 x V adds exact zeros (unwritten slots of a page may hold any bits: dead V rows are zeroed for bf16, dead P is 0
// after the V scale for int8). So a row's bits do not depend on the group size, the class split, the register file or
// the step's other rows (test-gqa: groups of 1 / 2 / 4 / 8 rows and 128 / 256 GRF bit-identical).
// The per-row kernel (gqa.cpp attn, GqaStep:attn_rows = -1) sums the q.k over 16 items of 16 dims and
// streams 16-byte loads: same semantics, other fp order.
// Decode attention numerics (GqaShape::dec_qk / dec_p_lo; int8 KV, the prefill fa body's QK switches on the decode
// group kernel): QK 1 = q as int8 codes per (row, q head) over all D dims (s = amax / 127, rne,
// clamp +-127: gqa_fa.cpp's rule, the amax of a head's D dims from the four dim quarters through SLM), the raw K codes
// as s8 DPAS B operands (32 dims x 16 tokens = the item's [16 tokens][8 dwords] block transposed: no widen) and q.k as
// s8 x s8 DPAS (32 dims per call: half the calls); QK 2 = q as hi + lo codes (lo = rne((q / s - hi) x 127), two DPAS
// per operand), the integer partial 127 hi + lo per (head, token, dim quarter). The four quarters' integer partials
// are summed as int32 through SLM (exact) and scaled once: S = float(sum) x (s_q / sqrt(D) [/ 127]) x the K scale.
// LO false = P as one bf16 (no lo pass: half the P.V DPAS). Every variant keeps the row-independence above (the same
// bits for every group size and register file).
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

#include "xenithium/kernels/attention/gqa.hpp"
#include "xenithium/kernels/rows.hpp"

namespace xenithium::kernels::gqa_detail {

namespace esimd = sycl::ext::intel::esimd;
namespace xmx = esimd::xmx;
using bf16 = gqa_bf16;

inline constexpr int kTile = kQsaTile, kPage = kQsaPage;
static_assert(kTile == 64 && kPage == 64, "a tile is one KV page of 64 tokens");
// head_dim D (256: qwen35, 128: llama): part rows D + 16 floats (o[D] | max, sum: gqa.cpp's layout); q.k scale 1 / sqrt(D)
template <int D>
inline constexpr float kQkScale = D == 256 ? 1.f / 16.f : D == 128 ? 0.08838834764831845f : 0.125f;

inline int cut_tps(int ntiles) { return (ntiles + kGqaSplits - 1) / kGqaSplits; }
inline int cut_nsp(int ntiles, int tps) { return (ntiles + tps - 1) / tps; }

// C [NH][16] (+)= A [NH][16 k] . B [16 k][16] in DPAS calls of <= 8 rows (rows are independent); A at offset ao of a.
template <int NH, int AN, int B = 0>
inline void dpas_rows(esimd::simd<float, NH * 16> &c, const esimd::simd<bf16, 256> &b, esimd::simd<bf16, AN> &a, int ao) SYCL_ESIMD_FUNCTION {
  if constexpr (B * 8 < NH) {
    constexpr int R = NH - B * 8 < 8 ? NH - B * 8 : 8;
    esimd::simd<float, R * 16> cc = c.template select<R * 16, 1>(B * 128);
    const esimd::simd<bf16, R * 16> aa = a.template select<R * 16, 1>(ao + B * 128);
    cc = xmx::dpas<8, R, float, float, bf16, bf16>(cc, b, aa);
    c.template select<R * 16, 1>(B * 128) = cc;
    dpas_rows<NH, AN, B + 1>(c, b, a, ao);
  }
}

// int8-v4 P . V for one 32-dim group (chunks dc0, dc0 + 1) per 8-row block: P x the tokens' group scales (fp32),
// hi / lo split, DPAS. Per element the same ops in the same order as folding all NH rows at once, but only one block's
// copies are live at a time.
template <int NH, bool LO, int N, int B = 0>
inline void v4_pv_rows(esimd::simd<float, N> &oa, esimd::simd<float, NH * 16> &pw, esimd::simd<float, 16> &sc,
                       const esimd::simd<bf16, 256> &b0, const esimd::simd<bf16, 256> &b1, int dc0) SYCL_ESIMD_FUNCTION {
  if constexpr (B * 8 < NH) {
    constexpr int R = NH - B * 8 < 8 ? NH - B * 8 : 8;
    esimd::simd<float, R * 16> pg;
#pragma unroll
    for (int r = 0; r < R; ++r) pg.template select<16, 1>(r * 16) = pw.template select<16, 1>(B * 128 + r * 16) * sc;
    const esimd::simd<bf16, R * 16> ph = pg;
    esimd::simd<bf16, R * 16> pl;
    if constexpr (LO) pl = pg - esimd::simd<float, R * 16>(ph);
#pragma unroll
    for (int d = 0; d < 2; ++d) {
      esimd::simd<float, R * 16> cc = oa.template select<R * 16, 1>((dc0 + d) * NH * 16 + B * 128);
      cc = xmx::dpas<8, R, float, float, bf16, bf16>(cc, d ? b1 : b0, ph);
      if constexpr (LO) cc = xmx::dpas<8, R, float, float, bf16, bf16>(cc, d ? b1 : b0, pl);
      oa.template select<R * 16, 1>((dc0 + d) * NH * 16 + B * 128) = cc;
    }
    v4_pv_rows<NH, LO, N, B + 1>(oa, pw, sc, b0, b1, dc0);
  }
}

// 16 tokens x 16 dims bf16 (row-major [tok][dim]) -> the q.k B operand (k = dims, n = tokens): dword pairs transposed
inline esimd::simd<bf16, 256> seg_k_operand(esimd::simd<bf16, 256> v) SYCL_ESIMD_FUNCTION {
  esimd::simd<uint32_t, 128> kd = v.template bit_cast_view<uint32_t>();
  esimd::simd<uint32_t, 128> bd;
#pragma unroll
  for (int pp = 0; pp < 8; ++pp) bd.template select<16, 1>(pp * 16) = kd.template select<16, 8>(pp);
  return bd.template bit_cast_view<bf16>();
}
// 16 tokens x 16 dims -> the P.V B operand (k = tokens, n = dims): token pairs interleaved
inline esimd::simd<bf16, 256> seg_v_operand(esimd::simd<bf16, 256> v) SYCL_ESIMD_FUNCTION {
  esimd::simd<bf16, 256> bv;
#pragma unroll
  for (int qq = 0; qq < 8; ++qq) {
    bv.template select<16, 2>(qq * 32) = v.template select<16, 1>(2 * qq * 16);
    bv.template select<16, 2>(qq * 32 + 1) = v.template select<16, 1>((2 * qq + 1) * 16);
  }
  return bv;
}

template <QsaKv KV, int G, int RM, int Lib, bool Sel, int D, int QK, bool LO, int TR>
class GqaAttnSegK;

// C [NH][16] (+)= A [NH][32 k] . B [32 k][16] on s8 x s8 DPAS in calls of <= 8 rows; A at offset ao (bytes) of a.
template <int NH, int AN, int B = 0>
inline void dpas_rows8(esimd::simd<int32_t, NH * 16> &c, const esimd::simd<int8_t, 512> &b, esimd::simd<int8_t, AN> &a, int ao) SYCL_ESIMD_FUNCTION {
  if constexpr (B * 8 < NH) {
    constexpr int R = NH - B * 8 < 8 ? NH - B * 8 : 8;
    esimd::simd<int32_t, R * 16> cc = c.template select<R * 16, 1>(B * 128);
    const esimd::simd<int8_t, R * 32> aa = a.template select<R * 32, 1>(ao + B * 256);
    cc = xmx::dpas<8, R, int32_t, int32_t, int8_t, int8_t, xmx::dpas_argument_type::s8, xmx::dpas_argument_type::s8, R * 16, 512, R * 32>(cc, b, aa);
    c.template select<R * 16, 1>(B * 128) = cc;
    dpas_rows8<NH, AN, B + 1>(c, b, a, ao);
  }
}

// The heads per pass of the final token-quarter sum: the largest divisor of NH up to 12.
constexpr int seg_hc(int nh) {
  for (int c = nh < 12 ? nh : 12; c > 1; --c)
    if (nh % c == 0) return c;
  return 1;
}

// Rows of the step = segments of seg_size(rows, seg_rows) rows (records as gqa.cpp); each segment in groups of RM
// rows. part: gqa.cpp's [rows][hkv][kGqaSplits][G][kPartStride]. ns: grid splits (gqa.cpp's). Lib: 0 = the 128-GRF
// instance (gqa.cpp), 1 = the 256-GRF one (gqa_seg.cpp): distinct kernel names.
// Sel: every row of segment g walks the pages sel[(g x hkv + kv head) x kGqaSelMax ..]
// (ascending logical pages, sel_n of them: gqa_select) as its tiles instead of 0 .. ntiles - 1; the cut follows sel_n (one
// class), a row masks its tokens past its own position as in the dense walk (a selected page past a row's last one: no
// tokens, no loads, alpha 1). sel_n 0 = the segment keeps every page: the dense walk. Instantiated in gqa_seg.cpp only
// (256 GRF).
// D (head_dim 256 or 128): an item = (dim quarter dq of D / 4 dims, token quarter tq of 16 tokens) either way; at D 128
// the final sum's 16-dim outputs of a quarter are the first two token quarters' (tq < 2).
// TR: the tree instance. Tree steps' side rows do not run their own units inside the row group's work-group (one after
// the other, on the critical path of the last split's work-groups): each side row gets a work-group of its own for
// every split that holds one of its tiles from p0 on (<= 2 splits; the split's tiles below p0 too, so its running sums
// see the same tiles in the same order), placed first in the grid; the row group's work-group leaves the row out of
// those splits. A side work-group runs the group's code with one live row whose tiles from p0 on carry its path's
// K / V in the virtual lanes: the same per-row arithmetic as its unit (a row's bits do not depend on its group).
template <QsaKv KV, int G, int RM, int Lib, bool Sel = false, int D = 256, int QK = 0, bool LO = true, int TR = 0>
sycl::event attn_seg(sycl::queue &q, const GqaShape &s, int rows, const GqaCache &c, const int32_t *rec, int rstride, int seg_rows, const bf16 *qb,
                     float *part, int ns, const int32_t *sel = nullptr, const int32_t *sel_n = nullptr, bool pf = false) {
  constexpr bool V4 = KV == QsaKv::i8v4;  // K int8, V int4 codes + a bf16 scale per 32 dims (head_dim 256)
  constexpr bool I8 = KV != QsaKv::bf16;  // K int8 with a scale per token (and V int8 unless V4)
  static_assert(!V4 || D == 256, "int8-v4 KV: head_dim 256");
  static_assert(!TR || (!V4 && !Sel && RM > 1), "the tree instance: int8 / bf16 KV, no selection, row groups (qwen38-27b round 89: any numerics)");
  constexpr int kD = D, kPartStride = D + 16;
  static_assert(D == 256 || D == 128, "head_dim 256 or 128");
  static_assert(QK == 0 || I8, "int8 q.k reads the int8 KV codes");
  static_assert(QK >= 0 && QK <= 2, "QK 0 / 1 / 2");
  constexpr int kC8 = D / 4 / 32;  // 32-dim s8 calls per item (QK > 0)
  constexpr int kI = 16, kDQ = 4, kTQ = 4, kDW = kD / kDQ, kTW = kTile / kTQ;  // 64 (32) dims, 16 tokens per item
  constexpr int kDc = kDW / 16;                                                // 16-dim chunks per item
  constexpr int kEB = I8 ? 1 : 2;                                              // KV element bytes
  constexpr int NH = RM * G;                                                   // DPAS rows: every row's q heads
  constexpr int HPI = (NH + kI - 1) / kI;                                      // heads an item owns (sums, softmax)
  constexpr int kHC = seg_hc(NH);                                              // heads per pass of the final tq sum
  static_assert(NH % kHC == 0 && NH <= 64, "at most 64 q heads per work-group");
  constexpr int kSpN = kTQ * kDQ * NH * 16;  // score partials [tq][dq][NH][16]
  constexpr int kRedN = kI * kHC * kDW;      // final sum [item][kHC][64]
  constexpr int kAOff = 0, kAN = kSpN > kRedN ? kSpN : kRedN;
  constexpr int kAlN = NH <= 16 ? 16 : (NH <= 32 ? 32 : 64);
  constexpr int kPvOff = kAOff + kAN, kAlOff = kPvOff + NH * kTile;
  constexpr int kSlm = (kAlOff + kAlN) * 4;
  const int hq = s.hq(), hkv = s.hkv();
  const int sr = seg_size(rows, seg_rows), nseg = seg_count(rows, seg_rows), nch = (sr + RM - 1) / RM;
  const uint8_t *kp = c.k, *vp = c.v;
  const float *ksp = c.ks, *vsp = c.vs;
  const int32_t *pages0 = c.pages;
  const int slot_pages = c.slot_pages, ring = c.ring_pages, win = s.window;
  // TR: the side work-groups (segment, row, kv head, its first / second split from p0) before the row groups
  const size_t nside = TR ? static_cast<size_t>(nseg) * sr * hkv * 2 : 0;
  const size_t groups = nside + static_cast<size_t>(nseg) * nch * 2 * hkv * ns;
  return q.parallel_for<GqaAttnSegK<KV, G, RM, Lib, Sel, D, QK, LO, TR>>(sycl::nd_range<1>(groups * kI, kI), [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
    esimd::slm_init<kSlm>();
    int g = static_cast<int>(it.get_group(0));
    const int i = static_cast<int>(it.get_local_id(0));
    const int dq = i % kDQ, tq = i / kDQ;
    bool sidewg = false;
    int sj = 0, se = 0, sseg = 0, skvh = 0;
    if constexpr (TR != 0) {
      if (static_cast<size_t>(g) < nside) {
        sidewg = true;
        se = g % 2;
        const int t1 = g / 2;
        skvh = t1 % hkv;
        sj = (t1 / hkv) % sr;
        sseg = t1 / hkv / sr;
      } else {
        g -= static_cast<int>(nside);
      }
    }
    // a segment's row groups (ch) are adjacent work-group ids, so the groups of one (kv head, split) run
    // together and the second reads the K / V tiles the first brought into L2
    const int ch = sidewg ? 0 : g % nch, g2 = g / nch;
    int sp = g2 % ns;
    const int kvh = sidewg ? skvh : (g2 / ns) % hkv, cls = sidewg ? 0 : (g2 / (ns * hkv)) % 2, seg = sidewg ? sseg : g2 / (ns * hkv * 2);
    int seg_lo = 0, seg_n = 0;  // any layout (seg_het: k segments of sr rows, then one-row segments)
    seg_span(seg_rows, rows, seg, seg_lo, seg_n);
    const int seg_hi = seg_lo + seg_n;
    if (sidewg && sj >= seg_n) return;
    const int r_lo = sidewg ? seg_lo + sj : seg_lo + ch * RM;
    const int r_hi = sidewg ? r_lo + 1 : (r_lo + RM < seg_hi ? r_lo + RM : seg_hi);
    if (r_lo >= r_hi) return;
    // the group's rows: tokens, tiles, class and split activity
    int ntk[RM], ntl[RM];
    bool act[RM];
    const SegRow s0 = seg_row(rec, rstride, seg_rows, r_lo);
    const int32_t *slist = Sel ? sel + static_cast<size_t>(seg * hkv + kvh) * kGqaSelMax : nullptr;
    const int nsel = Sel ? sel_n[seg * hkv + kvh] : 0;
    const bool pick = Sel && nsel > 0;  // 0: the segment keeps every page (the dense walk, bit for bit)
    // DDTree: a tree segment (record kSegTree): row j attends as row depth(j) of a chain over its ancestors.
    // vpos = its position for the cut and the masks (p0 + depth); a side row (j != depth: its path is not rows 0 .. j)
    // walks the tiles from tb (the one holding p0) in units of its own, each a physical tile with the lanes of virtual
    // positions p0 .. p0 + depth replaced by its path's K / V rows (anc: the path's rows packed 4 bits per depth). Every
    // other row (and every row on the tiles below tb) shares the group's units: the same instructions either way.
    const int32_t *srec = rstride >= kSegRec ? rec + static_cast<size_t>(seg) * rstride : nullptr;
    const uint64_t tw = srec ? rec_word(srec, kSegTree, kSegTreeHi) : 0ull;
    const bool treeg = (tw & kTreeFlag) != 0;
    const int tp0 = treeg ? srec[kSegPos] : 0, tb = tp0 / kTile;
    int vpos[RM], pdep[RM];
    bool side[RM];
#pragma unroll
    for (int p = 0; p < RM; ++p) {
      const int jr = r_lo + p < r_hi ? r_lo + p - seg_lo : 0;
      pdep[p] = jr, side[p] = false, vpos[p] = s0.pos + p;
      if (treeg) {
        pdep[p] = tree_depth(tw, jr);  // (the 64-bit path word only for side row u, below)
        vpos[p] = tp0 + pdep[p];
        side[p] = jr != pdep[p];
      }
    }
    if constexpr (TR != 0) {
      if (sidewg) {  // the side row's split: the first (se 0) / second (se 1) one holding a tile of it from p0 on
        if (!treeg || !side[0]) return;
        const int nt0 = (vpos[0] + 1 + kTile - 1) / kTile, ts = cut_tps(nt0);
        if (se && (nt0 - 1) / ts == tb / ts) return;
        sp = tb / ts + se;
      }
    }
    const int tps0 = cut_tps(pick ? nsel : (vpos[0] + 1 + kTile - 1) / kTile);
    int tps = 0, ntok_max = 0, ntl_max = 0;
    bool any = false;
    // windowed (arch flint): split sp = the one tile wt of the group with wt % kGqaWinSplits == sp (the group's tiles
    // run from its first row's lowest to its last row's last: <= kGqaWinSplits); a row is active on wt inside its window
    const bool winl = win > 0;
    if (winl && cls) return;
    int lot[RM];  // a row's lowest token (windowed; 0 otherwise)
    int wt = 0;
    if (winl) {
      const int lo0 = (s0.pos + 1 > win ? s0.pos + 1 - win : 0) / kTile;
      wt = lo0 + ((sp - lo0 % kGqaWinSplits) % kGqaWinSplits + kGqaWinSplits) % kGqaWinSplits;
      tps = 1;
    }
#pragma unroll
    for (int p = 0; p < RM; ++p) {
      const bool live = r_lo + p < r_hi;
      const int ntok = vpos[p] + 1, lo = winl && ntok > win ? ntok - win : 0;
      const int ntiles = winl ? wt + 1 : (pick ? nsel : (ntok + kTile - 1) / kTile), t = winl ? 1 : cut_tps(ntiles);
      const bool in = live && (winl ? (wt >= lo / kTile && wt <= (ntok - 1) / kTile) : ((t == tps0) == (cls == 0)));
      if (!winl && in) tps = t;
      ntk[p] = ntok;
      ntl[p] = ntiles;
      act[p] = in;
      lot[p] = lo;
    }
    if (tps == 0) return;  // no row of this class
#pragma unroll
    for (int p = 0; p < RM; ++p) {
      if (!winl) act[p] = act[p] && sp < cut_nsp(ntl[p], tps);
      if constexpr (TR != 0) {
        if (!sidewg && side[p] && sp >= tb / tps) act[p] = false;  // TR: the row's side work-group runs this split
      }
      if (act[p]) {
        any = true;
        ntok_max = ntk[p] > ntok_max ? ntk[p] : ntok_max;  // (rows ascend, tree rows need not)
        ntl_max = ntl[p] > ntl_max ? ntl[p] : ntl_max;
      }
    }
    if (!any) return;
    const int32_t *pages = pages0 + static_cast<size_t>(s0.slot) * slot_pages;
    const int tile0 = win > 0 ? wt : sp * tps, tile1 = win > 0 ? wt + 1 : (tile0 + tps < ntl_max ? tile0 + tps : ntl_max);
    const int lo_g = win > 0 ? lot[0] : 0;  // the group's lowest token (its first row's)
    auto so = [](int f) SYCL_ESIMD_FUNCTION { return static_cast<uint32_t>(f * 4); };
    // q of every row's heads for this item's 64 dims: [chunk][NH][16]
    esimd::simd<bf16, QK ? 16 : kDc * NH * 16> qa;
    // QK > 0: s8 codes [call][NH][32 dims] (hi; QK 2: lo beside), the head's scale s_q x the q.k factor per owned head
    esimd::simd<int8_t, QK ? kC8 * NH * 32 : 32> qa8, ql8;
    float qsc[HPI];
    if constexpr (QK == 0) {
#pragma unroll
      for (int p = 0; p < RM; ++p) {
        const int r = r_lo + p < r_hi ? r_lo + p : r_lo;  // dead rows: any finite q
#pragma unroll
        for (int h = 0; h < G; ++h) {
          esimd::simd<bf16, kDW> qv = esimd::block_load<bf16, kDW>(qb + (static_cast<size_t>(r) * hq + kvh * G + h) * kD + dq * kDW);
#pragma unroll
          for (int dc = 0; dc < kDc; ++dc) qa.template select<16, 1>((dc * NH + p * G + h) * 16) = qv.template select<16, 1>(dc * 16);
        }
      }
    } else {
      // the amax of each head's D dims: the four dim quarters' through SLM (items of token quarter 0 store)
      esimd::simd<float, NH * kDW> qf;
#pragma unroll
      for (int p = 0; p < RM; ++p) {
        const int r = r_lo + p < r_hi ? r_lo + p : r_lo;
#pragma unroll
        for (int h = 0; h < G; ++h) {
          const esimd::simd<bf16, kDW> qv = esimd::block_load<bf16, kDW>(qb + (static_cast<size_t>(r) * hq + kvh * G + h) * kD + dq * kDW);
          qf.template select<kDW, 1>((p * G + h) * kDW) = qv;
        }
      }
      if (tq == 0) {
#pragma unroll
        for (int h = 0; h < NH; ++h)
          esimd::slm_scalar_store<float>(static_cast<uint32_t>((dq * NH + h) * 4), esimd::hmax<float>(esimd::abs(qf.template select<kDW, 1>(h * kDW).read())));
      }
      esimd::barrier();
#pragma unroll
      for (int h = 0; h < NH; ++h) {
        float amax = 0.f;
#pragma unroll
        for (int d4 = 0; d4 < kDQ; ++d4) {
          const float a = esimd::slm_scalar_load<float>(static_cast<uint32_t>((d4 * NH + h) * 4));
          amax = a > amax ? a : amax;
        }
        float inv = 0.f, sv = 0.f;
        if (amax > 0.f) inv = esimd::div_ieee(127.f, amax), sv = esimd::div_ieee(amax, 127.f);
        const esimd::simd<float, kDW> x = qf.template select<kDW, 1>(h * kDW).read() * inv;
        esimd::simd<float, kDW> th = esimd::rnde<float, kDW>(x);
        th = esimd::max(esimd::min(th, 127.f), -127.f);
        esimd::simd<int8_t, kDW> hv = th;
#pragma unroll
        for (int cc = 0; cc < kC8; ++cc) qa8.template select<32, 1>((cc * NH + h) * 32) = hv.template select<32, 1>(cc * 32);
        if constexpr (QK == 2) {
          esimd::simd<float, kDW> tl = esimd::rnde<float, kDW>((x - th) * 127.f);
          tl = esimd::max(esimd::min(tl, 127.f), -127.f);
          esimd::simd<int8_t, kDW> lv = tl;
#pragma unroll
          for (int cc = 0; cc < kC8; ++cc) ql8.template select<32, 1>((cc * NH + h) * 32) = lv.template select<32, 1>(cc * 32);
        }
#pragma unroll
        for (int k = 0; k < HPI; ++k)
          if (h == i + 16 * k) qsc[k] = sv * (QK == 2 ? kQkScale<D> * (1.f / 127.f) : kQkScale<D>);
      }
      esimd::barrier();  // the amax words are read before the loop's partials reuse the SLM
    }
    // this item's 16 tokens x 64 dims of a tile as bf16, chunk dc = [16 tokens][16 dims] at dc * 256
    auto load_tile = [&](const uint8_t *base) SYCL_ESIMD_FUNCTION {
      esimd::simd<bf16, kTW * kDW> out;
      if constexpr (I8) {
        esimd::simd<uint8_t, kTW * kDW> raw = esimd::load_2d<uint8_t, kDW, kTW>(base, kD - 1, kPage - 1, kD - 1, dq * kDW, tq * kTW);
        auto r2 = raw.template bit_cast_view<int8_t, kTW, kDW>();
#pragma unroll
        for (int dc = 0; dc < kDc; ++dc) {
          const esimd::simd<int8_t, 256> sb = r2.template select<kTW, 1, 16, 1>(0, dc * 16);
          const esimd::simd<float, 256> f = sb;
          out.template select<256, 1>(dc * 256) = f;
        }
      } else {
#pragma unroll
        for (int hf = 0; hf < kDW / 32; ++hf) {
          esimd::simd<uint16_t, kTW * 32> raw = esimd::load_2d<uint16_t, 32, kTW>(reinterpret_cast<const uint16_t *>(base), kD * 2 - 1, kPage - 1, kD * 2 - 1, dq * kDW + hf * 32, tq * kTW);
          auto r2 = raw.template bit_cast_view<uint16_t, kTW, 32>();
#pragma unroll
          for (int d2 = 0; d2 < 2; ++d2) {
            esimd::simd<uint16_t, 256> sb = r2.template select<kTW, 1, 16, 1>(0, d2 * 16);
            out.template select<256, 1>((hf * 2 + d2) * 256) = sb.template bit_cast_view<bf16>();
          }
        }
      }
      return out;
    };
    // int8-v4: this item's 16 tokens x 64 dims of V = 32 bytes per token of a 128-byte row; nibbles widened
    // exactly (|code| <= 7) into load_tile's [dc][16 tokens][16 dims] layout (dims 2j low nibble, 2j + 1 high)
    auto load_v4 = [&](const uint8_t *base) SYCL_ESIMD_FUNCTION {
      esimd::simd<bf16, kTW * kDW> out;
      if constexpr (V4) {
        esimd::simd<uint8_t, kTW * kDW / 2> raw = esimd::load_2d<uint8_t, kDW / 2, kTW>(base, kD / 2 - 1, kPage - 1, kD / 2 - 1, dq * kDW / 2, tq * kTW);
        auto r2 = raw.template bit_cast_view<uint8_t, kTW, kDW / 2>();
#pragma unroll
        for (int dc = 0; dc < kDc; ++dc) {
          const esimd::simd<int16_t, 128> b = esimd::simd<uint8_t, 128>(r2.template select<kTW, 1, 8, 1>(0, dc * 8));
          const esimd::simd<int16_t, 128> lo = (b & 15) - ((b & 8) << 1), hi = (b >> 4) - ((b & 128) >> 3);
          out.template select<128, 2>(dc * 256) = esimd::simd<float, 128>(lo);
          out.template select<128, 2>(dc * 256 + 1) = esimd::simd<float, 128>(hi);
        }
      }
      return out;
    };
    const float kmul = kQkScale<D>;  // 1 / sqrt(D)
    esimd::simd<float, kDc * NH * 16> oa(0.f);  // [chunk][NH][16 dims]
    float m_run[HPI], l_run[HPI];
#pragma unroll
    for (int k = 0; k < HPI; ++k) m_run[k] = -INFINITY, l_run[k] = 0.f;
    const esimd::simd<int32_t, kTile> ti(0, 1);
    const esimd::simd<int32_t, 256> rowi = esimd::simd<int32_t, 256>(0, 1) >> 4;  // token row of a [16][16] chunk
    // a unit's tile lanes of virtual positions p0 + i (i <= the side row's depth) from its path's rows (phys p0 + anc_i)
    auto path_slot = [&](uint64_t anc, int i) SYCL_ESIMD_FUNCTION {
      const int ps = tp0 + static_cast<int>((anc >> (4 * i)) & 15u);
      return (static_cast<size_t>(pages[ring ? (ps / kTile) % ring : ps / kTile]) * hkv + kvh) * kPage + ps % kTile;
    };
#pragma unroll 1
    for (int si = tile0; si < tile1; ++si) {
#pragma unroll 1
    for (int u = -1; u < (treeg && !TR ? RM : 0); ++u) {  // u -1: the shared unit; u >= 0: side row u's own view (tree segments)
      const int tile = pick ? slist[si] : si;  // Sel: the si-th selected page
      const int t0 = tile * kTile;
      // TR: one unit per tile; a side work-group's row (p 0) carries its path from tb on
      const int pu = TR ? (sidewg && tile >= tb ? 0 : -1) : u;
      bool on[RM];  // the unit's rows (the group's active rows on the shared unit)
      int ntu = 0;
      bool anyu = false;
#pragma unroll
      for (int p = 0; p < RM; ++p) {
        const bool own = side[p] && tile >= tb;
        // TR: on = act (a side work-group's row on every tile; the row groups' side rows only in splits below tb). Written
        // with own, not as act alone: with `act[p] && (TR || ...)` the VC backend drops the softmax of the 4-row
        // instance (no exp in its vISA; oneAPI 2026.2).
        on[p] = act[p] && (TR ? !(own && !sidewg) : (u < 0 ? !own : (p == u && own)));
        if (on[p]) anyu = true, ntu = ntk[p] > ntu ? ntk[p] : ntu;
      }
      if (!anyu) continue;
      const int ntm = treeg ? ntu : ntok_max;  // chains: the group's last token
      const int nt = ntm - t0 < kTile ? ntm - t0 : kTile;  // Sel: <= 0 past the group's rows
      const size_t slot = (static_cast<size_t>(pages[ring ? tile % ring : tile]) * hkv + kvh) * kPage;  // the tile's first slot
      const bool mine = tq * kTW < nt;  // this item's token quarter holds tokens of the tile
      // side row u's path lanes in this tile: i in [ilo, ihi) at lane tp0 + i - t0
      int ilo = 0, ihi = 0;
      uint64_t uanc = 0;
#pragma unroll
      for (int p = 0; p < RM; ++p)
        if (p == pu) {
          int dd = 0;
          uanc = tree_anc_word(tw, r_lo + p - seg_lo, &dd);
          ilo = t0 - tp0 > 0 ? t0 - tp0 : 0;
          ihi = tp0 + pdep[p] + 1 - t0 < kTile ? pdep[p] + 1 : t0 + kTile - tp0;
        }
      // TR: a side work-group's tile from tb on: every lane's slot, its path's rows in the virtual lanes [ilo, ihi) (the
      // path rows are the segment's rows: slots p0 .. p0 + seg_n - 1, at most two pages), for the K / V rows and the
      // scales, so the loads issue together (the per-lane loop waits on each lane's loads in turn). Computed where used
      // (kept live across the q.k it spills the 4-row instance).
      auto lane_slots = [&](auto nw, int l0) SYCL_ESIMD_FUNCTION {  // lanes l0 .. l0 + N - 1
        constexpr int N = decltype(nw)::value;
        const esimd::simd<int32_t, N> ln(l0, 1);
        const esimd::simd<int32_t, N> vi = ln + (t0 - tp0);  // lane L's virtual index i (p0 + i = t0 + L)
        const esimd::simd<uint32_t, N> sh = esimd::simd<uint32_t, N>(vi & 7) * 4u;
        esimd::simd<uint32_t, N> nib = esimd::simd<uint32_t, N>(static_cast<uint32_t>(uanc)) >> sh;
        nib.merge(esimd::simd<uint32_t, N>(static_cast<uint32_t>(uanc >> 32)) >> sh, vi >= 8);
        const esimd::simd<int32_t, N> ps = esimd::simd<int32_t, N>(nib & 15u) + tp0;  // the path row's position
        const int pb = (tp0 + seg_n - 1) / kTile;
        const int32_t sa = (pages[tb] * hkv + kvh) * kPage - tb * kTile, sb = (pages[pb] * hkv + kvh) * kPage - pb * kTile;
        esimd::simd<int32_t, N> pslot = ps + sa;
        pslot.merge(ps + sb, ps >= pb * kTile);
        esimd::simd<uint32_t, N> srcv = esimd::simd<uint32_t, N>(ln) + static_cast<uint32_t>(slot);
        srcv.merge(esimd::simd<uint32_t, N>(pslot), (vi >= ilo) & (vi < ihi));
        return srcv;
      };
      // ---- q.k partials over this item's 64 dims: [NH][16 tokens] ----
      esimd::simd<bf16, kTW * kDW> vt;
      if (mine) {
        esimd::simd<float, NH * 16> spd(0.f);
        if constexpr (QK == 0) {
          esimd::simd<bf16, kTW * kDW> kt;
          // TR: a side work-group's quarter that holds lanes of its row's path is built row by row from the lanes' slots
          // (lane_slots; a lane off the path reads its own slot: the bytes load_tile gives it), each table's 16 row loads
          // issued before their use (the per-lane loop below waits on each lane's loads in turn); widening as load_tile's
          bool rowsq = false;
          if constexpr (TR != 0) {
            const int qv = tq * kTW + t0 - tp0;  // the quarter's first virtual index
            rowsq = pu >= 0 && qv + kTW > ilo && qv < ihi;
          }
          if (rowsq) {
            if constexpr (TR != 0) {
              using RawT = std::conditional_t<I8, int8_t, bf16>;
              const esimd::simd<uint32_t, kTW> sq = lane_slots(std::integral_constant<int, kTW>{}, tq * kTW);  // this quarter's rows
              auto load_rows = [&](const uint8_t *base, esimd::simd<bf16, kTW * kDW> &out) SYCL_ESIMD_FUNCTION {
                constexpr int kRB = kTW;  // rows per batch of loads (all 16: no spill once the slots are 16 lanes, not 64)
#pragma unroll
                for (int b0 = 0; b0 < kTW; b0 += kRB) {
                  esimd::simd<RawT, kDW> raw[kRB];
#pragma unroll
                  for (int r = 0; r < kRB; ++r) raw[r] = esimd::block_load<RawT, kDW>(reinterpret_cast<const RawT *>(base) + static_cast<size_t>(sq[b0 + r]) * kD + dq * kDW);
#pragma unroll
                  for (int r = 0; r < kRB; ++r) {
                    esimd::simd<bf16, kDW> w;
                    if constexpr (I8) w = esimd::simd<float, kDW>(raw[r]);
                    else w = raw[r];
#pragma unroll
                    for (int dc = 0; dc < kDc; ++dc) out.template select<16, 1>(dc * 256 + (b0 + r) * 16) = w.template select<16, 1>(dc * 16);
                  }
                }
              };
              load_rows(kp, kt);
              load_rows(vp, vt);
            }
          } else {
            kt = load_tile(kp + slot * kD * kEB);
            if constexpr (V4) vt = load_v4(vp + slot * (kD / 2));
            else vt = load_tile(vp + slot * kD * kEB);  // V issued with K: its latency hides behind the softmax
          }
          if constexpr (!V4 && TR == 0) {
            // a side row's unit: its path's K / V rows into their virtual lanes of this item's quarter (the
            // widening as load_tile's: int8 -> float -> bf16 exact, bf16 as is)
            for (int pi = ilo; pi < ihi; ++pi) {
              const int L = tp0 + pi - t0;
              if (L / kTW != tq || ((uanc >> (4 * pi)) & 15u) == static_cast<uint64_t>(pi)) continue;  // not mine / in place
              const size_t ps = path_slot(uanc, pi);
              const int tr = L % kTW;
              esimd::simd<bf16, kDW> kr, vr;
              if constexpr (I8) {
                kr = esimd::simd<float, kDW>(esimd::block_load<int8_t, kDW>(reinterpret_cast<const int8_t *>(kp) + ps * kD + dq * kDW));
                vr = esimd::simd<float, kDW>(esimd::block_load<int8_t, kDW>(reinterpret_cast<const int8_t *>(vp) + ps * kD + dq * kDW));
              } else {
                kr = esimd::block_load<bf16, kDW>(reinterpret_cast<const bf16 *>(kp) + ps * kD + dq * kDW);
                vr = esimd::block_load<bf16, kDW>(reinterpret_cast<const bf16 *>(vp) + ps * kD + dq * kDW);
              }
#pragma unroll
              for (int dc = 0; dc < kDc; ++dc) {
                kt.template select<16, 1>(dc * 256 + tr * 16) = kr.template select<16, 1>(dc * 16);
                vt.template select<16, 1>(dc * 256 + tr * 16) = vr.template select<16, 1>(dc * 16);
              }
            }
          }
#pragma unroll
          for (int dc = 0; dc < kDc; ++dc) {
            const esimd::simd<bf16, 256> bk = seg_k_operand(kt.template select<256, 1>(dc * 256));
            dpas_rows<NH, kDc * NH * 16>(spd, bk, qa, dc * NH * 16);
          }
        } else {
          // the raw K codes [16 tokens][kDW bytes]; per 32-dim call the [16][8 dwords] block transposed = the s8 B operand
          esimd::simd<uint8_t, kTW * kDW> raw;
          // TR: a side work-group's quarter on its row's path, row by row from the lanes'
          // slots as the bf16 branch above: the K codes as stored (the s8 operand's rows), V widened as load_tile's;
          // each table's 16 row loads issued before their use
          bool rowsq8 = false;
          if constexpr (TR != 0) {
            const int qv = tq * kTW + t0 - tp0;  // the quarter's first virtual index
            rowsq8 = pu >= 0 && qv + kTW > ilo && qv < ihi;
          }
          if (rowsq8) {
            if constexpr (TR != 0) {
              const esimd::simd<uint32_t, kTW> sq = lane_slots(std::integral_constant<int, kTW>{}, tq * kTW);  // this quarter's rows
              esimd::simd<uint8_t, kDW> kr[kTW];
#pragma unroll
              for (int r = 0; r < kTW; ++r) kr[r] = esimd::block_load<uint8_t, kDW>(kp + static_cast<size_t>(sq[r]) * kD + dq * kDW);
#pragma unroll
              for (int r = 0; r < kTW; ++r) raw.template select<kDW, 1>(r * kDW) = kr[r];
              esimd::simd<int8_t, kDW> vr[kTW];
#pragma unroll
              for (int r = 0; r < kTW; ++r) vr[r] = esimd::block_load<int8_t, kDW>(reinterpret_cast<const int8_t *>(vp) + static_cast<size_t>(sq[r]) * kD + dq * kDW);
#pragma unroll
              for (int r = 0; r < kTW; ++r) {
                esimd::simd<bf16, kDW> w = esimd::simd<float, kDW>(vr[r]);
#pragma unroll
                for (int dc = 0; dc < kDc; ++dc) vt.template select<16, 1>(dc * 256 + r * 16) = w.template select<16, 1>(dc * 16);
              }
            }
          } else {
            raw = esimd::load_2d<uint8_t, kDW, kTW>(kp + slot * kD, kD - 1, kPage - 1, kD - 1, dq * kDW, tq * kTW);
            if constexpr (V4) vt = load_v4(vp + slot * (kD / 2));
            else vt = load_tile(vp + slot * kD * kEB);
          }
          if constexpr (!V4 && TR == 0) {
            // a side row's unit on the int8 q.k, as the bf16 branch above: its path's K codes
            // (the raw bytes: the s8 operand's rows) and V rows (widened as load_tile's) into their virtual lanes
            for (int pi = ilo; pi < ihi; ++pi) {
              const int L = tp0 + pi - t0;
              if (L / kTW != tq || ((uanc >> (4 * pi)) & 15u) == static_cast<uint64_t>(pi)) continue;  // not mine / in place
              const size_t ps = path_slot(uanc, pi);
              const int tr = L % kTW;
              raw.template select<kDW, 1>(tr * kDW) = esimd::block_load<uint8_t, kDW>(kp + ps * kD + dq * kDW);
              esimd::simd<bf16, kDW> vr = esimd::simd<float, kDW>(esimd::block_load<int8_t, kDW>(reinterpret_cast<const int8_t *>(vp) + ps * kD + dq * kDW));
#pragma unroll
              for (int dc = 0; dc < kDc; ++dc) vt.template select<16, 1>(dc * 256 + tr * 16) = vr.template select<16, 1>(dc * 16);
            }
          }
          auto r32 = raw.template bit_cast_view<uint32_t, kTW, kDW / 4>();
          esimd::simd<int32_t, NH * 16> si(0), sl(0);
#pragma unroll
          for (int cc = 0; cc < kC8; ++cc) {
            esimd::simd<uint32_t, 128> kd = r32.template select<16, 1, 8, 1>(0, cc * 8);
            esimd::simd<uint32_t, 128> bd;
#pragma unroll
            for (int pp = 0; pp < 8; ++pp) bd.template select<16, 1>(pp * 16) = kd.template select<16, 8>(pp);
            esimd::simd<int8_t, 512> b8 = bd.template bit_cast_view<int8_t>();
            dpas_rows8<NH, kC8 * NH * 32>(si, b8, qa8, cc * NH * 32);
            if constexpr (QK == 2) dpas_rows8<NH, kC8 * NH * 32>(sl, b8, ql8, cc * NH * 32);
          }
          if constexpr (QK == 2) si = si * 127 + sl;
          spd = si.template bit_cast_view<float>();  // int32 bits through SLM: the quarters summed exactly
        }
#pragma unroll
        for (int b = 0; b < NH / 2; ++b)
          esimd::slm_block_store<float, 32>(so(kAOff + (tq * kDQ + dq) * NH * 16 + b * 32), spd.template select<32, 1>(b * 32));
        if constexpr (NH % 2)  // odd NH (grp 7 at one row): the last head's 16 partials
          esimd::slm_block_store<float, 16>(so(kAOff + (tq * kDQ + dq) * NH * 16 + (NH - 1) * 16), spd.template select<16, 1>((NH - 1) * 16));
      }
      if (pf && u < 0 && ch == 0 && !sidewg && si + 1 < tile1) {  // the next tile's K / V blocks into L2 while this one computes
        const int ntile = pick ? slist[si + 1] : si + 1;
        const size_t nslot = (static_cast<size_t>(pages[ring ? ntile % ring : ntile]) * hkv + kvh) * kPage;
        constexpr auto pfp = esimd::properties{esimd::cache_hint_L1<esimd::cache_hint::uncached>, esimd::cache_hint_L2<esimd::cache_hint::cached>};
        if constexpr (I8) {
          esimd::prefetch_2d<uint8_t, kDW, kTW>(kp + nslot * kD, kD - 1, kPage - 1, kD - 1, dq * kDW, tq * kTW, pfp);
          if constexpr (V4) esimd::prefetch_2d<uint8_t, kDW / 2, kTW>(vp + nslot * (kD / 2), kD / 2 - 1, kPage - 1, kD / 2 - 1, dq * kDW / 2, tq * kTW, pfp);
          else esimd::prefetch_2d<uint8_t, kDW, kTW>(vp + nslot * kD, kD - 1, kPage - 1, kD - 1, dq * kDW, tq * kTW, pfp);
        } else {
#pragma unroll
          for (int hf = 0; hf < kDW / 32; ++hf) {
            esimd::prefetch_2d<uint16_t, 32, kTW>(reinterpret_cast<const uint16_t *>(kp) + nslot * kD, kD * 2 - 1, kPage - 1, kD * 2 - 1, dq * kDW + hf * 32, tq * kTW, pfp);
            esimd::prefetch_2d<uint16_t, 32, kTW>(reinterpret_cast<const uint16_t *>(vp) + nslot * kD, kD * 2 - 1, kPage - 1, kD * 2 - 1, dq * kDW + hf * 32, tq * kTW, pfp);
          }
        }
      }
      esimd::barrier();  // the partials are in SLM; the previous tile's P readers are done
      // ---- per owned head: the tile's scores (dq sums), the online softmax, P and alpha ----
      esimd::simd<float, kTile> ks(1.f), vs(1.f);
      if constexpr (I8) {
        if constexpr (TR != 0) {
          if (pu >= 0) {  // TR: the side row's lanes by their slots (lane_slots): two gathers per scale table
            esimd::simd<uint32_t, kTile> bo = lane_slots(std::integral_constant<int, kTile>{}, 0) * 4u;
#pragma unroll
            for (int h2 = 0; h2 < 2; ++h2) {
              const esimd::simd<uint32_t, 32> o = bo.template select<32, 1>(h2 * 32);
              ks.template select<32, 1>(h2 * 32) = esimd::gather<float, 32>(ksp, o);
              vs.template select<32, 1>(h2 * 32) = esimd::gather<float, 32>(vsp, o);
            }
          } else {
            ks = esimd::block_load<float, kTile>(ksp + slot);
            vs = esimd::block_load<float, kTile>(vsp + slot);
          }
        } else {
          ks = esimd::block_load<float, kTile>(ksp + slot);
          if constexpr (!V4) vs = esimd::block_load<float, kTile>(vsp + slot);  // V4: the scales are folded in P . V
          if constexpr (!V4) {
            for (int pi = ilo; pi < ihi; ++pi) {  // a side row's path lanes (every item: the heads' owners read all)
              if (((uanc >> (4 * pi)) & 15u) == static_cast<uint64_t>(pi)) continue;
              const size_t ps = path_slot(uanc, pi);
              ks[tp0 + pi - t0] = ksp[ps];
              vs[tp0 + pi - t0] = vsp[ps];
            }
          }
        }
      }
#pragma unroll
      for (int k = 0; k < HPI; ++k) {
        const int h = i + 16 * k;
        if (h < NH) {
          const int p = h / G;
          int ntr = 0, nlo = 0;  // this row's tokens in the tile: [nlo, ntr) (ntr 0: past its last tile, or inactive in the split)
#pragma unroll
          for (int pp = 0; pp < RM; ++pp)
            if (pp == p) {
              ntr = on[pp] && (pick ? ntk[pp] > t0 : tile < ntl[pp]) ? (ntk[pp] - t0 < kTile ? ntk[pp] - t0 : kTile) : 0;
              nlo = lot[pp] - t0 > 0 ? lot[pp] - t0 : 0;
            }
          esimd::simd<float, kTile> pe(0.f);
          float al = 1.f;
          if (ntr > 0) {
            esimd::simd<float, kTile> x;
#pragma unroll
            for (int t4 = 0; t4 < kTQ; ++t4) {
              if constexpr (QK == 0) {
                esimd::simd<float, 16> acc(0.f);
                if (t4 * kTW < nt) {
#pragma unroll
                  for (int d4 = 0; d4 < kDQ; ++d4) acc += esimd::slm_block_load<float, 16>(so(kAOff + ((t4 * kDQ + d4) * NH + h) * 16));
                }
                x.template select<16, 1>(t4 * 16) = acc * kmul;
              } else {
                esimd::simd<int32_t, 16> acc(0);
                if (t4 * kTW < nt) {
#pragma unroll
                  for (int d4 = 0; d4 < kDQ; ++d4) acc += esimd::slm_block_load<int32_t, 16>(so(kAOff + ((t4 * kDQ + d4) * NH + h) * 16));
                }
                x.template select<16, 1>(t4 * 16) = esimd::simd<float, 16>(acc) * qsc[k];
              }
            }
            if constexpr (I8) x *= ks;
            x.merge(esimd::simd<float, kTile>(-INFINITY), (ti >= ntr) | (ti < nlo));
            const float mt = esimd::hmax<float>(x);
            const float mn = m_run[k] > mt ? m_run[k] : mt;
            const esimd::simd<float, 1> a1 = esimd::exp(esimd::simd<float, 1>(m_run[k] - mn));  // 0 on the row's first tile
            pe = esimd::exp(x - mn);
            pe.merge(esimd::simd<float, kTile>(0.f), (ti >= ntr) | (ti < nlo));
            l_run[k] = l_run[k] * static_cast<float>(a1[0]) + esimd::reduce<float>(pe, std::plus<>());
            m_run[k] = mn;
            al = a1[0];
            if constexpr (I8 && !V4) {
              pe *= vs;
              pe.merge(esimd::simd<float, kTile>(0.f), (ti >= ntr) | (ti < nlo));  // unwritten slots' scales may be any bits
            }
          }
          esimd::slm_block_store<float, kTile>(so(kPvOff + h * kTile), pe);
          esimd::slm_scalar_store<float>(so(kAlOff + h), al);
        }
      }
      esimd::barrier();  // P and alpha are in SLM; the partials' readers are done
      {
        const esimd::simd<float, kAlN> al = esimd::slm_block_load<float, kAlN>(so(kAlOff));
#pragma unroll
        for (int h = 0; h < NH; ++h) {
          const float a = al[h];
#pragma unroll
          for (int dc = 0; dc < kDc; ++dc) oa.template select<16, 1>((dc * NH + h) * 16) *= a;
        }
      }
      // ---- P . V over this item's 16 tokens for its 64 dims ----
      if (mine) {
        esimd::simd<float, NH * 16> pw;
#pragma unroll
        for (int h = 0; h < NH; ++h) pw.template select<16, 1>(h * 16) = esimd::slm_block_load<float, 16>(so(kPvOff + h * kTile + tq * kTW));
        const int live = nt - tq * kTW;  // rows of this quarter holding tokens <= the group's last position
        if constexpr (V4) {
          // P x the token's scale of each 32-dim group (fp32, before the hi / lo split: qsa.cpp's fold); this item's
          // 64 dims are groups 2 dq (chunks 0-1) and 2 dq + 1 (chunks 2-3): dword dq of each token's 8 bf16 scales
          const esimd::simd<int32_t, 16> tk(0, 1);
          const esimd::simd<uint32_t, 16> soff = (esimd::simd<uint32_t, 16>(tk) + static_cast<uint32_t>(slot + tq * kTW)) * 16u + static_cast<uint32_t>(dq * 4);
          const esimd::simd<uint32_t, 16> sw2 = esimd::gather<uint32_t, 16>(reinterpret_cast<const uint32_t *>(vsp), soff);
#pragma unroll
          for (int g2 = 0; g2 < 2; ++g2) {
            esimd::simd<uint32_t, 16> sb = g2 == 0 ? esimd::simd<uint32_t, 16>(sw2 << 16) : esimd::simd<uint32_t, 16>(sw2 & 0xFFFF0000u);
            esimd::simd<float, 16> sc = sb.template bit_cast_view<float>();
            sc.merge(esimd::simd<float, 16>(0.f), (tk >= live) | (tk < lo_g - t0 - tq * kTW));  // unwritten slots' scales: any bits
            const esimd::simd<bf16, 256> b0 = seg_v_operand(vt.template select<256, 1>(2 * g2 * 256));
            const esimd::simd<bf16, 256> b1 = seg_v_operand(vt.template select<256, 1>((2 * g2 + 1) * 256));
            v4_pv_rows<NH, LO>(oa, pw, sc, b0, b1, 2 * g2);  // per 8-row block: the fold's P copies stay 128 wide
          }
        }
        esimd::simd<bf16, NH * 16> ph = pw;
        esimd::simd<bf16, LO ? NH * 16 : 16> pl;
        if constexpr (LO) pl = pw - esimd::simd<float, NH * 16>(ph);
#pragma unroll
        for (int dc = 0; dc < (V4 ? 0 : kDc); ++dc) {
          esimd::simd<bf16, 256> v = vt.template select<256, 1>(dc * 256);
          if constexpr (!I8) v.merge(esimd::simd<bf16, 256>(0.f), (rowi >= live) | (rowi < lo_g - t0 - tq * kTW));  // unwritten slots may hold NaN bits
          const esimd::simd<bf16, 256> bv = seg_v_operand(v);
          esimd::simd<float, NH * 16> oc = oa.template select<NH * 16, 1>(dc * NH * 16);
          dpas_rows<NH, NH * 16>(oc, bv, ph, 0);
          if constexpr (LO) dpas_rows<NH, NH * 16>(oc, bv, pl, 0);
          oa.template select<NH * 16, 1>(dc * NH * 16) = oc;
        }
      }
    }  // units
    }
    // ---- the four token quarters' o summed in tq order; (o, max, sum) of the active rows to part ----
#pragma unroll
    for (int hc0 = 0; hc0 < NH; hc0 += kHC) {
      esimd::barrier();  // the loop's (or the previous pass's) SLM readers are done
#pragma unroll
      for (int hh = 0; hh < kHC; ++hh)
#pragma unroll
        for (int dc = 0; dc < kDc; ++dc)
          esimd::slm_block_store<float, 16>(so(kAOff + (i * kHC + hh) * kDW + dc * 16), oa.template select<16, 1>((dc * NH + hc0 + hh) * 16));
      esimd::barrier();
#pragma unroll
      for (int hh = 0; hh < kHC; ++hh) {
        const int h = hc0 + hh, p = h / G;
        bool on = tq * 16 < kDW;  // D 128: a quarter's 32 dims are the outputs of tq 0 and 1
#pragma unroll
        for (int pp = 0; pp < RM; ++pp)
          if (pp == p) on = on && act[pp];
        if (!on) continue;
        esimd::simd<float, 16> o(0.f);  // dims dq * 64 + tq * 16 .. + 16
#pragma unroll
        for (int t4 = 0; t4 < kTQ; ++t4) o += esimd::slm_block_load<float, 16>(so(kAOff + ((t4 * kDQ + dq) * kHC + hh) * kDW + tq * 16));
        float *pb = part + (static_cast<size_t>((r_lo + p) * hkv + kvh) * kGqaSplits + sp) * G * kPartStride;
        esimd::block_store<float, 16>(pb + static_cast<size_t>(h % G) * kPartStride + dq * kDW + tq * 16, o);
      }
    }
#pragma unroll
    for (int k = 0; k < HPI; ++k) {
      const int h = i + 16 * k;
      if (h < NH) {
        const int p = h / G;
        bool on = false;
#pragma unroll
        for (int pp = 0; pp < RM; ++pp)
          if (pp == p) on = act[pp];
        if (on) {
          float *pb = part + (static_cast<size_t>((r_lo + p) * hkv + kvh) * kGqaSplits + sp) * G * kPartStride;
          esimd::simd<float, 2> ml;
          ml[0] = m_run[k];
          ml[1] = l_run[k];
          esimd::block_store<float, 2>(pb + static_cast<size_t>(h % G) * kPartStride + kD, ml);
        }
      }
    }
  });
}

}  // namespace xenithium::kernels::gqa_detail
