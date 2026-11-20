// Xenithium · page-selected attention for the dense gated GQA (gqa.hpp): the page key bounds, the page scores
// and the selection per (unit, kv head). A unit is the rows that share one page set: a decode step's segment (the
// attention over its selection is gqa_seg.hpp's group kernel, Sel; the merge gqa.cpp's) or a prefill call's attention
// row block (the attention is gqa_fa.cpp's SEL instances).
//
// Bounds: per (physical page, kv head) the elementwise min and max over the page's written keys of the values attention
// uses (bf16 K, or int8 code x the token's scale), rounded to bf16. A write covers the page's tokens from its first one up
// to the writer's last row, so tokens a rejected speculative row left past it fall out at the next write.
// Score of page p for one q head (kSelSpread below): q . mid + spread x |q| . rad, with mid and rad the middle and
// the half-width of the page's bounds per coordinate. Per coordinate that is a weighted sum of the larger and the
// smaller of q_d kmax_d and q_d kmin_d; at spread 1 it is their larger alone, the upper bound of q . k over the page.
// The score launch computes it as a bf16 DPAS GEMM of the step's q heads against the pages' bounds (score_pages below).
// A segment scores a page by the largest over its rows' q heads of the kv head. Selection: the first kGqaSelSink pages,
// the last kGqaSelLocal pages up to the segment's last row, and the top budget - sink - local of the rest by score
// (topk_wg: exact, ties to the lower page), ascending; a context of budget pages or fewer keeps them all (count 0: the
// attention's dense walk).
#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

#include <sycl/ext/intel/esimd.hpp>

#include "xenithium/kernels/attention/gqa.hpp"
#include "xenithium/kernels/rows.hpp"
#include "attention/gqa_seg.hpp"  // the bf16 DPAS helpers (dpas_rows, seg_k_operand)
#include "attention/topk_wg.hpp"

namespace xenithium::kernels {
namespace {

using bf16 = gqa_bf16;
constexpr int kD = kQsaDim, kPage = kQsaPage;

namespace esimd = sycl::ext::intel::esimd;

class GqaBoundsK;
class GqaBoundsRangeK;
template <int G, int RB, class Units>
class GqaSelScoreK;
template <class Units>
class GqaSelectK;

// The rows that share one page set, and where their context ends.
struct SelUnit {
  int row0, rows;  // the unit's rows of the call
  int last_pos;    // its last row's position: the context is pages 0 .. last_pos / kPage
  int slot;        // the sequence's page table
};
// A decode step's units: its segments (the records, rows.hpp; any layout).
struct SegUnits {
  const int32_t *rec;
  int stride, seg_rows, rows;
  int count() const { return seg_count(rows, seg_rows); }
  bool keeps_all(int) const { return false; }
  SelUnit unit(int i) const {
    int row0 = 0, n = 0;
    seg_span(seg_rows, rows, i, row0, n);
    const SegRow first = seg_row(rec, stride, seg_rows, row0), last = seg_row(rec, stride, seg_rows, row0 + n - 1);
    return SelUnit{row0, n, last.pos, first.slot};
  }
};
// A prefill call's units: the attention's row blocks of one sequence (rows at pos0 + row; gqa_rows' one-slot cache).
struct BlockUnits {
  int pos0, block_rows, rows;
  bool dense_last;  // the last block keeps every page (GqaRowsSelect::dense_last_block)
  int count() const { return (rows + block_rows - 1) / block_rows; }
  bool keeps_all(int i) const { return dense_last && i == count() - 1; }
  SelUnit unit(int i) const {
    const int row0 = i * block_rows, n = rows - row0 < block_rows ? rows - row0 : block_rows;
    return SelUnit{row0, n, pos0 + row0 + n - 1, 0};
  }
};

// order-preserving uint32 of each float (larger float -> larger key)
inline esimd::simd<uint32_t, 16> order_keys(esimd::simd<float, 16> f) SYCL_ESIMD_FUNCTION {
  const esimd::simd<uint32_t, 16> bits = f.template bit_cast_view<uint32_t>();
  esimd::simd<uint32_t, 16> keys = bits | 0x80000000u;
  keys.merge(~bits, (bits & 0x80000000u) != 0u);
  return keys;
}

// one work-group of kD items (item = dim) per (page unit, kv head): min / max over tokens [t0, t1) of the page
inline void page_bounds(int d, const uint8_t *kp, const float *ksp, bool i8, size_t slot0, int n, bf16 *mn_out, bf16 *mx_out) {
  float mn = INFINITY, mx = -INFINITY;
  for (int t = 0; t < n; ++t) {
    const size_t e = (slot0 + static_cast<size_t>(t)) * kD + static_cast<size_t>(d);
    float v;
    if (i8) v = static_cast<float>(reinterpret_cast<const int8_t *>(kp)[e]) * ksp[slot0 + static_cast<size_t>(t)];
    else v = static_cast<float>(reinterpret_cast<const bf16 *>(kp)[e]);
    mn = sycl::fmin(mn, v);
    mx = sycl::fmax(mx, v);
  }
  mn_out[d] = bf16(mn);
  mx_out[d] = bf16(mx);
}

// The weight of the bounds' half-width in a page's score (gqa.hpp).
constexpr float kSelSpread = 0.5f;

// The score as a sum over the stored bounds. Per coordinate, of x = q_d kmax_d and y = q_d kmin_d the larger is
// q_d mid_d + |q_d| rad_d and the smaller q_d mid_d - |q_d| rad_d, so q_d mid_d + spread |q_d| rad_d =
// kScoreHi max(x, y) + kScoreLo min(x, y). Which of x, y is the larger is the sign of q_d alone, so with
//   w_max_d = q_d x (q_d >= 0 ? kScoreHi : kScoreLo),  w_min_d = q_d x (q_d >= 0 ? kScoreLo : kScoreHi)
// the score is sum_d kmax_d w_max_d + kmin_d w_min_d: linear in the bounds, a GEMM.
constexpr float kScoreHi = 0.5f * (1.f + kSelSpread), kScoreLo = 0.5f * (1.f - kSelSpread);

// What the score and select launches read and write.
struct ScoreJob {
  const bf16 *kmin, *kmax;   // the cache's bounds, [physical page][kv head][kD]
  const int32_t *pages0;     // the slots' page tables, slot_pages entries each
  int slot_pages;
  const bf16 *q;             // the call's q, [row][q head][kD]
  int hkv, hq;
  int budget;                // a unit of budget pages or fewer is not scored (the dense walk)
  const int32_t *slot_heads;  // device [slot][hkv] pages per (slot, kv head); null = budget for every head
  int pages_of(int slot, int kvh) const { return slot_heads ? slot_heads[slot * hkv + kvh] : budget; }
  int max_pages;             // pages the longest unit's context spans
  uint32_t *keys;            // order_keys of the pages' scores, [unit][kv head][key_stride]
  int key_stride;            // a multiple of kScorePages, >= max_pages
  int32_t *sel, *sel_n;      // out: the selection [unit][kv head][kGqaSelMax] and its count [unit][kv head]
};

constexpr int kScorePages = 16;  // pages per hardware thread: the n of one DPAS call
static_assert(kGqaSelPagesMax % kScorePages == 0, "a thread stores the keys of 16 pages at once; the last ones may lie past the context");

// The page scores of every (unit, kv head): one hardware thread per 16 pages. Per 16-dim chunk it loads the 16
// pages' kmax and kmin (32 bytes per page each: the bounds are read once per row block), builds w_max / w_min of a
// block of RB rows x G q heads from q, and adds both products on bf16 DPAS; a page's score is the largest over the
// heads. A unit of more than RB rows runs its row blocks one after the other; rows past its last repeat that row,
// which leaves the largest unchanged. The operands are bf16 (w is rounded: within 1e-3 of the fp32 sum's rms).
template <int G, int RB, class Units>
sycl::event score_pages(sycl::queue &q, const ScoreJob &job, const Units &units) {
  using gqa_detail::dpas_rows;
  using gqa_detail::seg_k_operand;
  constexpr int NH = RB * G;  // the DPAS rows of one row block
  const int items = (job.max_pages + kScorePages - 1) / kScorePages;
  return q.parallel_for<GqaSelScoreK<G, RB, Units>>(sycl::range<1>(static_cast<size_t>(units.count()) * job.hkv * items), [=](sycl::id<1> id) SYCL_ESIMD_KERNEL {
    const int thread = static_cast<int>(id[0]);
    const int item = thread % items, kvh = (thread / items) % job.hkv, seg = thread / (items * job.hkv);
    const SelUnit u = units.unit(seg);
    const int r0 = u.row0, r1 = u.row0 + u.rows - 1;
    const int np = u.last_pos / kPage + 1;
    if (np <= job.budget || np <= job.pages_of(u.slot, kvh) || units.keeps_all(seg) || item * kScorePages >= np) return;
    const int32_t *pages = job.pages0 + static_cast<size_t>(u.slot) * job.slot_pages;
    size_t bounds_at[kScorePages];  // each page's row of the bounds (pages past the context repeat the last one)
#pragma unroll
    for (int p = 0; p < kScorePages; ++p)
      bounds_at[p] = (static_cast<size_t>(pages[std::min(item * kScorePages + p, np - 1)]) * job.hkv + kvh) * kD;
    esimd::simd<float, 16> best(-INFINITY);
    for (int r = r0; r <= r1; r += RB) {
      size_t q_at[RB];  // the block's rows of q for this kv head
#pragma unroll
      for (int j = 0; j < RB; ++j) q_at[j] = (static_cast<size_t>(std::min(r + j, r1)) * job.hq + kvh * G) * kD;
      esimd::simd<float, NH * 16> score(0.f);  // [RB x G heads][16 pages]
      for (int chunk = 0; chunk < kD / 16; ++chunk) {
        esimd::simd<bf16, 256> mx, mn;  // [16 pages][16 dims]
#pragma unroll
        for (int p = 0; p < kScorePages; ++p) {
          mx.template select<16, 1>(p * 16) = esimd::block_load<bf16, 16>(job.kmax + bounds_at[p] + chunk * 16);
          mn.template select<16, 1>(p * 16) = esimd::block_load<bf16, 16>(job.kmin + bounds_at[p] + chunk * 16);
        }
        esimd::simd<float, NH * 16> qf;  // [RB x G heads][16 dims]
#pragma unroll
        for (int j = 0; j < RB; ++j)
#pragma unroll
          for (int h = 0; h < G; ++h)
            qf.template select<16, 1>((j * G + h) * 16) = esimd::simd<float, 16>(esimd::block_load<bf16, 16>(job.q + q_at[j] + h * kD + chunk * 16));
        const esimd::simd_mask<NH * 16> negative = qf < 0.f;
        esimd::simd<float, NH * 16> to_max(kScoreHi), to_min(kScoreLo);
        to_max.merge(esimd::simd<float, NH * 16>(kScoreLo), negative);
        to_min.merge(esimd::simd<float, NH * 16>(kScoreHi), negative);
        esimd::simd<bf16, NH * 16> w_max = qf * to_max, w_min = qf * to_min;
        dpas_rows<NH, NH * 16>(score, seg_k_operand(mx), w_max, 0);
        dpas_rows<NH, NH * 16>(score, seg_k_operand(mn), w_min, 0);
      }
#pragma unroll
      for (int h = 0; h < NH; ++h) best = esimd::max<float, 16>(best, score.template select<16, 1>(h * 16));
    }
    esimd::block_store<uint32_t, 16>(job.keys + (static_cast<size_t>(seg) * job.hkv + kvh) * job.key_stride + item * kScorePages, order_keys(best));
  });
}

// The score launch in the row block of the call's widest unit.
template <class Units>
sycl::event score_pages_for(sycl::queue &q, const ScoreJob &job, const Units &units, int grp, int widest_unit_rows) {
  if (grp != 6) throw std::invalid_argument("gqa_select: the page scores are built for 6 q heads per kv head");
  if (widest_unit_rows <= 1) return score_pages<6, 1, Units>(q, job, units);
  if (widest_unit_rows == 2) return score_pages<6, 2, Units>(q, job, units);
  return score_pages<6, 4, Units>(q, job, units);
}

// The selection of every (unit, kv head) from its pages' score keys: one work-group each. The first kGqaSelSink pages,
// the last kGqaSelLocal up to the unit's last row, and the best of the rest (topk_wg: exact, ties to the lower page),
// ascending; a unit whose context is the budget or fewer pages, or that the layout marks so, keeps them all (count 0:
// the attention's dense walk).
template <class Units>
sycl::event select_pages(sycl::queue &q, const ScoreJob &job, const Units &units) {
  const int hkv = job.hkv, common = job.budget, key_stride = job.key_stride;
  const uint32_t *keys = job.keys;
  int32_t *sel = job.sel, *sel_n = job.sel_n;
  return q.submit([&](sycl::handler &h) {
    sycl::local_accessor<uint32_t, 1> hist(sycl::range<1>(512), h);
    sycl::local_accessor<int32_t, 1> shw(sycl::range<1>(8), h);
    h.parallel_for<GqaSelectK<Units>>(sycl::nd_range<1>(static_cast<size_t>(units.count()) * hkv * topk::kSelItems, topk::kSelItems), [=](sycl::nd_item<1> it) {
      const int g = static_cast<int>(it.get_group(0)), li = static_cast<int>(it.get_local_id(0));
      const SelUnit u = units.unit(g / hkv);  // group g = (unit, kv head): its slot of keys / sel / sel_n
      const int np = u.last_pos / kPage + 1;
      int32_t *out = sel + static_cast<size_t>(g) * kGqaSelMax;
      int32_t *on = sel_n + g;
      const int budget = job.pages_of(u.slot, g % hkv);
      if (np <= common || np <= budget || units.keeps_all(g / hkv)) {  // every page: the dense walk
        if (li == 0) *on = 0;
        return;
      }
      const int mid = np - kGqaSelSink - kGqaSelLocal, K = budget - kGqaSelSink - kGqaSelLocal;
      const uint32_t *kr = keys + static_cast<size_t>(g) * key_stride + kGqaSelSink;
      topk::SelSlm m{&hist[0], &shw[0]};
      topk::topk_wg(
          it, m, mid, [](int) { return true; }, [&](int i) { return kr[i]; }, [&](int i) { return kGqaSelSink + i; }, K, out + kGqaSelSink, on);
      for (int i = li; i < kGqaSelSink; i += topk::kSelItems) out[i] = i;
      for (int i = li; i < kGqaSelLocal; i += topk::kSelItems) out[kGqaSelSink + K + i] = np - kGqaSelLocal + i;
      sycl::group_barrier(it.get_group());
      if (li == 0) *on = budget;
    });
  });
}

void check_budget(int budget, const char *who) {
  if (budget < kGqaSelSink + kGqaSelLocal + 1 || budget > kGqaSelMax) throw std::invalid_argument(std::string(who) + ": budget out of range");
}
}  // namespace

size_t gqa_page_bound_bytes(const GqaShape &s) { return static_cast<size_t>(s.hkv()) * kD * 2 * 2; }

float gqa_page_score_ref(const float *q, const float *kmin, const float *kmax) {
  float acc = 0.f;
  for (int d = 0; d < kD; ++d) {  // the device's operands: w rounded to bf16 (its DPAS sums in another fp32 order)
    const bool negative = q[d] < 0.f;
    const float w_max = static_cast<float>(bf16(q[d] * (negative ? kScoreLo : kScoreHi)));
    const float w_min = static_cast<float>(bf16(q[d] * (negative ? kScoreHi : kScoreLo)));
    acc += kmax[d] * w_max + kmin[d] * w_min;
  }
  return acc;
}

namespace {
// gqa_bounds_page's caches, by value in its kernel
struct BoundBufs {
  const uint8_t *k[kGqaMoveCaches];
  const float *ks[kGqaMoveCaches];
  const int32_t *pages[kGqaMoveCaches];  // the slot's page table
  bf16 *kmin[kGqaMoveCaches], *kmax[kGqaMoveCaches];
};
}  // namespace

sycl::event gqa_bounds_page(sycl::queue &q, const GqaShape &s, const GqaCache *caches, int n_caches, int slot, int page) {
  if (n_caches < 1 || n_caches > kGqaMoveCaches || slot < 0 || page < 0) throw std::invalid_argument("gqa_bounds_page: 1..16 caches, a slot and a page");
  BoundBufs b{};
  for (int l = 0; l < n_caches; ++l) {
    const GqaCache &c = caches[l];
    if (!c.kmin || !c.kmax) throw std::invalid_argument("gqa_bounds_page: the cache has no bounds");
    if ((page + 1) * kPage > c.max_tokens) throw std::invalid_argument("gqa_bounds_page: the page is past the cache");
    b.k[l] = c.k, b.ks[l] = c.ks, b.pages[l] = c.pages + static_cast<size_t>(slot) * c.slot_pages, b.kmin[l] = c.kmin, b.kmax[l] = c.kmax;
  }
  const int hkv = s.hkv();
  const bool i8 = qsa_kv_scaled(s.kv);
  return q.parallel_for<GqaBoundsK>(sycl::nd_range<1>(static_cast<size_t>(n_caches) * hkv * kD, kD), [=](sycl::nd_item<1> it) {
    const int g = static_cast<int>(it.get_group(0)), d = static_cast<int>(it.get_local_id(0));
    const int kvh = g % hkv, l = g / hkv;
    const size_t phys = static_cast<size_t>(b.pages[l][page]) * hkv + kvh;
    page_bounds(d, b.k[l], b.ks[l], i8, phys * kPage, kPage, b.kmin[l] + phys * kD, b.kmax[l] + phys * kD);
  });
}

sycl::event gqa_bounds_range(sycl::queue &q, const GqaShape &s, const GqaCache &c, int pos0, int m) {
  if (!c.kmin || !c.kmax) throw std::invalid_argument("gqa_bounds_range: the cache has no bounds");
  if (m < 1 || pos0 < 0) throw std::invalid_argument("gqa_bounds_range: rows / position");
  const int hkv = s.hkv(), p_lo = pos0 / kPage, p_hi = (pos0 + m - 1) / kPage, end = pos0 + m;
  const bool i8 = qsa_kv_scaled(s.kv);
  const uint8_t *kp = c.k;
  const float *ksp = c.ks;
  const int32_t *pages = c.pages;  // one sequence's table
  bf16 *kmin = c.kmin, *kmax = c.kmax;
  return q.parallel_for<GqaBoundsRangeK>(sycl::nd_range<1>(static_cast<size_t>(p_hi - p_lo + 1) * hkv * kD, kD), [=](sycl::nd_item<1> it) {
    const int g = static_cast<int>(it.get_group(0)), d = static_cast<int>(it.get_local_id(0));
    const int kvh = g % hkv, page = p_lo + g / hkv;
    const int n = std::min(kPage, end - page * kPage);
    const size_t phys = static_cast<size_t>(pages[page]) * hkv + kvh;
    page_bounds(d, kp, ksp, i8, phys * kPage, n, kmin + phys * kD, kmax + phys * kD);
  });
}

sycl::event gqa_select(sycl::queue &q, const GqaShape &s, int rows, const GqaCache &c, const int32_t *rec, int stride, int seg_rows, const bf16 *qb,
                       int budget, int pos_bound, const GqaScratch &sc, const int32_t *slot_heads) {
  if (!c.kmin || !c.kmax || !sc.sel_key || !sc.sel || !sc.sel_n) throw std::invalid_argument("gqa_select: bounds and selection scratch needed");
  check_budget(budget, "gqa_select");
  const int maxp = (pos_bound + kPage - 1) / kPage;
  if (maxp > kGqaSelPagesMax) throw std::invalid_argument("gqa_select: context");
  const ScoreJob job{.kmin = c.kmin, .kmax = c.kmax, .pages0 = c.pages, .slot_pages = c.slot_pages, .q = qb, .hkv = s.hkv(), .hq = s.hq(), .budget = budget,
                     .slot_heads = slot_heads, .max_pages = maxp, .keys = sc.sel_key, .key_stride = kGqaSelPagesMax, .sel = sc.sel, .sel_n = sc.sel_n};
  const SegUnits units{rec, stride, seg_rows, rows};
  score_pages_for(q, job, units, s.grp(), seg_size(rows, seg_rows));
  return select_pages(q, job, units);
}

size_t gqa_page_sets_bytes(const GqaShape &s, int rows_max, int block_rows, int max_tokens, GqaPageSets *layout) {
  GqaPageSets sets;
  sets.blocks = (rows_max + block_rows - 1) / block_rows;
  sets.key_stride = (max_tokens / kPage + kScorePages - 1) / kScorePages * kScorePages;
  const size_t units = static_cast<size_t>(sets.blocks) * s.hkv();
  if (layout) *layout = sets;
  return units * (static_cast<size_t>(sets.key_stride) + kGqaSelMax + 1) * sizeof(int32_t);
}

void gqa_page_sets_bind(const GqaShape &s, int rows_max, int block_rows, int max_tokens, void *base, GqaPageSets &sets) {
  gqa_page_sets_bytes(s, rows_max, block_rows, max_tokens, &sets);
  const size_t units = static_cast<size_t>(sets.blocks) * s.hkv();
  sets.keys = static_cast<uint32_t *>(base);
  sets.pages = reinterpret_cast<int32_t *>(sets.keys + units * sets.key_stride);
  sets.counts = sets.pages + units * kGqaSelMax;
}

sycl::event gqa_select_rows(sycl::queue &q, const GqaShape &s, int m, int pos0, int block_rows, const GqaCache &c, const bf16 *qb,
                            const GqaRowsSelect &select, const GqaPageSets &sets) {
  const int budget = select.pages;
  if (!c.kmin || !c.kmax || !sets.valid()) throw std::invalid_argument("gqa_select_rows: bounds and page-set scratch needed");
  if (c.slot_pages) throw std::invalid_argument("gqa_select_rows: one slot (page table without slot stride)");
  check_budget(budget, "gqa_select_rows");
  if (m < 1 || pos0 < 0 || block_rows < 1) throw std::invalid_argument("gqa_select_rows: rows / position / block rows");
  const BlockUnits units{pos0, block_rows, m, select.dense_last_block};
  const int maxp = (pos0 + m + kPage - 1) / kPage;
  if (units.count() > sets.blocks || maxp > sets.key_stride) throw std::invalid_argument("gqa_select_rows: the call's row blocks or context pass the page-set scratch");
  const ScoreJob job{.kmin = c.kmin, .kmax = c.kmax, .pages0 = c.pages, .slot_pages = 0, .q = qb, .hkv = s.hkv(), .hq = s.hq(), .budget = budget,
                     .slot_heads = nullptr, .max_pages = maxp, .keys = sets.keys, .key_stride = sets.key_stride, .sel = sets.pages, .sel_n = sets.counts};
  score_pages_for(q, job, units, s.grp(), block_rows);
  return select_pages(q, job, units);
}

}  // namespace xenithium::kernels
