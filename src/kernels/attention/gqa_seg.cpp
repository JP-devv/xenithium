// Xenithium · the group attention of the dense gated GQA decode at 256 GRF (gqa_seg.hpp).
#include "attention/gqa_seg.hpp"

#include <stdexcept>

namespace xenithium::kernels {

sycl::event gqa_attn_seg_xl(sycl::queue &q, const GqaShape &s, int rows, const GqaCache &c, const int32_t *rec, int stride, int seg_rows,
                            const gqa_bf16 *qb, float *part, int ns, int rm, bool pf, bool tree) {
  using namespace gqa_detail;
  const bool i8 = s.kv == QsaKv::i8;
  if (s.kv == QsaKv::i8v4) {  // int8-v4 KV, grp 6 at head_dim 256 (the dense 27B)
    if (s.grp() != 6 || s.head_dim != 256) throw std::invalid_argument("gqa_attn_seg_xl: int8-v4 KV is built for 6 q heads per kv head at head_dim 256");
#define XEN_V4(RM, QK, LO) return attn_seg<QsaKv::i8v4, 6, RM, 1, false, 256, QK, LO>(q, s, rows, c, rec, stride, seg_rows, qb, part, ns, nullptr, nullptr, pf);
    if (!s.dec_exact()) {
      // tree steps on int8-fast: the tree instance on 2- / 4-row groups (same as int8 KV)
      if (tree && s.dec_qk == 1 && !s.dec_p_lo && (rm == 2 || rm == 4)) {
        if (rm == 2) return attn_seg<QsaKv::i8v4, 6, 2, 1, false, 256, 1, false, 1>(q, s, rows, c, rec, stride, seg_rows, qb, part, ns, nullptr, nullptr, pf);
        return attn_seg<QsaKv::i8v4, 6, 4, 1, false, 256, 1, false, 1>(q, s, rows, c, rec, stride, seg_rows, qb, part, ns, nullptr, nullptr, pf);
      }
#define XEN_V4N(RM)                          \
  if (rm == RM) {                            \
    if (s.dec_qk == 0) XEN_V4(RM, 0, false)  \
    if (s.dec_qk == 2) XEN_V4(RM, 2, false)  \
    if (s.dec_p_lo) XEN_V4(RM, 1, true)      \
    XEN_V4(RM, 1, false)                     \
  }
      XEN_V4N(1)
      XEN_V4N(2)
      XEN_V4N(4)
#undef XEN_V4N
      throw std::invalid_argument("gqa_attn_seg_xl: the decode attention numerics run groups of 1 / 2 / 4 rows");
    }
    // exact trees on 2- / 4-row groups: the tree instance, int4 V + its bf16 scales
    if (tree && (rm == 2 || rm == 4)) {
      if (rm == 2) return attn_seg<QsaKv::i8v4, 6, 2, 1, false, 256, 0, true, 1>(q, s, rows, c, rec, stride, seg_rows, qb, part, ns, nullptr, nullptr, pf);
      return attn_seg<QsaKv::i8v4, 6, 4, 1, false, 256, 0, true, 1>(q, s, rows, c, rec, stride, seg_rows, qb, part, ns, nullptr, nullptr, pf);
    }
    if (rm == 1) XEN_V4(1, 0, true)
    if (rm == 2) XEN_V4(2, 0, true)
    if (rm == 4) XEN_V4(4, 0, true)
    if (rm == 8) XEN_V4(8, 0, true)
#undef XEN_V4
    throw std::invalid_argument("gqa_attn_seg_xl: groups of 1 / 2 / 4 / 8 rows");
  }
  if (!s.dec_exact()) {  // the decode attention numerics (grp 6, D 256; gqa_decode checked the shape)
    // tree steps on int8-fast run the tree instance on the served row groups, as the exact
    // numerics below (side rows in work-groups of their own); the other modes keep the side rows' units in the row group
    if (tree && i8 && s.dec_qk == 1 && !s.dec_p_lo && (rm == 2 || rm == 4)) {
      if (rm == 2) return attn_seg<QsaKv::i8, 6, 2, 1, false, 256, 1, false, 1>(q, s, rows, c, rec, stride, seg_rows, qb, part, ns, nullptr, nullptr, pf);
      return attn_seg<QsaKv::i8, 6, 4, 1, false, 256, 1, false, 1>(q, s, rows, c, rec, stride, seg_rows, qb, part, ns, nullptr, nullptr, pf);
    }
#define XEN_SEGN(RM)                                                                                                           \
  if (rm == RM) {                                                                                                              \
    if (!i8) return attn_seg<QsaKv::bf16, 6, RM, 1, false, 256, 0, false>(q, s, rows, c, rec, stride, seg_rows, qb, part, ns, nullptr, nullptr, pf); \
    if (s.dec_qk == 0) return attn_seg<QsaKv::i8, 6, RM, 1, false, 256, 0, false>(q, s, rows, c, rec, stride, seg_rows, qb, part, ns, nullptr, nullptr, pf); \
    if (s.dec_qk == 2) return attn_seg<QsaKv::i8, 6, RM, 1, false, 256, 2, false>(q, s, rows, c, rec, stride, seg_rows, qb, part, ns, nullptr, nullptr, pf); \
    return s.dec_p_lo ? attn_seg<QsaKv::i8, 6, RM, 1, false, 256, 1, true>(q, s, rows, c, rec, stride, seg_rows, qb, part, ns, nullptr, nullptr, pf)      \
                      : attn_seg<QsaKv::i8, 6, RM, 1, false, 256, 1, false>(q, s, rows, c, rec, stride, seg_rows, qb, part, ns, nullptr, nullptr, pf);    \
  }
    XEN_SEGN(1)
    XEN_SEGN(2)
    XEN_SEGN(4)
#undef XEN_SEGN
    throw std::invalid_argument("gqa_attn_seg_xl: the decode attention numerics run groups of 1 / 2 / 4 rows");
  }
  // tree steps on the served row groups (2 / 4 rows) run the tree instance
  // (side rows in work-groups of their own); other groups keep the side rows' units in the row group (TR 0)
  if (tree && s.grp() == 6 && s.head_dim == 256 && (rm == 2 || rm == 4)) {
    if (rm == 2) return i8 ? attn_seg<QsaKv::i8, 6, 2, 1, false, 256, 0, true, 1>(q, s, rows, c, rec, stride, seg_rows, qb, part, ns, nullptr, nullptr, pf)
                           : attn_seg<QsaKv::bf16, 6, 2, 1, false, 256, 0, true, 1>(q, s, rows, c, rec, stride, seg_rows, qb, part, ns, nullptr, nullptr, pf);
    return i8 ? attn_seg<QsaKv::i8, 6, 4, 1, false, 256, 0, true, 1>(q, s, rows, c, rec, stride, seg_rows, qb, part, ns, nullptr, nullptr, pf)
              : attn_seg<QsaKv::bf16, 6, 4, 1, false, 256, 0, true, 1>(q, s, rows, c, rec, stride, seg_rows, qb, part, ns, nullptr, nullptr, pf);
  }
#define XEN_SEG(G, RM, D)                                                                                                \
  if (s.grp() == G && rm == RM && s.head_dim == D)                                                                       \
    return i8 ? attn_seg<QsaKv::i8, G, RM, 1, false, D>(q, s, rows, c, rec, stride, seg_rows, qb, part, ns, nullptr, nullptr, pf)             \
              : attn_seg<QsaKv::bf16, G, RM, 1, false, D>(q, s, rows, c, rec, stride, seg_rows, qb, part, ns, nullptr, nullptr, pf);
  XEN_SEG(6, 1, 256)
  XEN_SEG(6, 2, 256)
  XEN_SEG(6, 4, 256)
  XEN_SEG(6, 8, 256)
  XEN_SEG(12, 1, 256)
  XEN_SEG(12, 2, 256)
  XEN_SEG(12, 4, 256)
  XEN_SEG(7, 1, 128)  // arch llama (Nanbeige4.3: 56 q / 8 kv heads x 128)
  XEN_SEG(7, 2, 128)
  XEN_SEG(7, 4, 128)
  XEN_SEG(7, 8, 128)
  XEN_SEG(8, 1, 128)  // arch flint (32 q / 4 kv heads x 128)
  XEN_SEG(8, 2, 128)
  XEN_SEG(8, 4, 128)
  XEN_SEG(8, 8, 128)
#undef XEN_SEG
  throw std::invalid_argument("gqa_attn_seg_xl: groups of 1 / 2 / 4 / 8 rows (grp 6, grp 7 at head_dim 128) or 1 / 2 / 4 (grp 12)");
}

// The page-selected instances (grp 6, the dense 27B): the exact numerics and int8-fast,
// the numerics `auto` serves on the int8 KV, in its groups of 1 / 2 / 4 rows
sycl::event gqa_attn_seg_sel_xl(sycl::queue &q, const GqaShape &s, int rows, const GqaCache &c, const int32_t *rec, int stride, int seg_rows,
                                const gqa_bf16 *qb, float *part, int ns, int rm, const int32_t *sel, const int32_t *sel_n, bool pf) {
  using namespace gqa_detail;
  const bool i8 = s.kv == QsaKv::i8;
  if (s.kv == QsaKv::i8v4) throw std::invalid_argument("gqa_attn_seg_sel_xl: page selection takes int8 | bf16 KV (not int8-v4)");
  if (s.head_dim != kQsaDim) throw std::invalid_argument("gqa_attn_seg_sel_xl: page selection is built for head_dim 256");
  if (!s.dec_exact()) {
    if (!s.dec_int8_fast() || !i8 || s.grp() != 6)
      throw std::invalid_argument("gqa_attn_seg_sel_xl: page selection runs the exact or the int8-fast decode attention numerics (grp 6, int8 KV)");
#define XEN_SEL_FAST(RM) \
  if (rm == RM) return attn_seg<QsaKv::i8, 6, RM, 1, true, 256, 1, false>(q, s, rows, c, rec, stride, seg_rows, qb, part, ns, sel, sel_n, pf);
    XEN_SEL_FAST(1)
    XEN_SEL_FAST(2)
    XEN_SEL_FAST(4)
#undef XEN_SEL_FAST
    throw std::invalid_argument("gqa_attn_seg_sel_xl: int8-fast page selection runs groups of 1 / 2 / 4 rows");
  }
#define XEN_SEG(G, RM)                                                                                                   \
  if (s.grp() == G && rm == RM)                                                                                          \
    return i8 ? attn_seg<QsaKv::i8, G, RM, 1, true>(q, s, rows, c, rec, stride, seg_rows, qb, part, ns, sel, sel_n, pf)     \
              : attn_seg<QsaKv::bf16, G, RM, 1, true>(q, s, rows, c, rec, stride, seg_rows, qb, part, ns, sel, sel_n, pf);
  XEN_SEG(6, 1)
  XEN_SEG(6, 2)
  XEN_SEG(6, 4)
  XEN_SEG(6, 8)
#undef XEN_SEG
  throw std::invalid_argument("gqa_attn_seg_sel_xl: page selection is built for 6 q heads per kv head, groups of 1 / 2 / 4 / 8 rows");
}

// The same grouped attention body and K8V4 operands as the dense path.
// Only the logical-page walk and four-token mask differ; use the requested
// native QK / P numerics, not a separate attention implementation. The
// one-row group still uses the stock tree's virtual-depth, ancestor overlay
// and causal cut when kSegTree is set. Its page mask pins the path tiles.
sycl::event gqa_attn_seg_trained_xl(sycl::queue &q, const GqaShape &s, int rows, const GqaCache &c, const GqaStep &step,
                                   const gqa_bf16 *qb, float *part, int ns) {
  using gqa_detail::attn_seg;
#define XEN_TRAINED(QK, LO) \
  return attn_seg<QsaKv::i8v4, 6, 1, 1, true, 256, QK, LO, 0, true>(q, s, rows, c, step.pos, step.stride, step.seg_rows, \
    qb, part, ns, step.trained_qsa.entries, step.trained_qsa.counts)
  if (s.dec_qk == 2) { XEN_TRAINED(2, false); }
  if (s.dec_qk == 1) {
    if (s.dec_p_lo) { XEN_TRAINED(1, true); }
    XEN_TRAINED(1, false);
  }
  if (s.dec_p_lo) { XEN_TRAINED(0, true); }
  XEN_TRAINED(0, false);
#undef XEN_TRAINED
}

}  // namespace xenithium::kernels
