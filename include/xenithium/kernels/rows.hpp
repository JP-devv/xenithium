// Xenithium · rows per decode-shaped step.
// The decode GEMMs hold every row of a step in registers (DPAS M = rows), so their row count
// is a compile-time constant. The row-dependent kernels (dense_gemm,
// the HC up-mix, the MoE route / gate_up / down / combine) are templates on the shape M and
// the public entry point picks it from the step's live rows (row_shape): a 1-4-row step runs
// the 4-row code, 5..8 rows the 8-row code. Row r's bits do not depend on
// the shape (every summation order comes from one row, or from (n, k) for the k splits).
// The per-row / per-segment kernels (GDN core, QSA stages, PLE, picks, logprobs) take the row
// count at run time; GDN's core stages at most kGdnRows rows per segment, so a sequence
// carries <= 4 rows of a step and 8 rows are segments (2 x 4, 4 x 2, 8 x 1).
// Everything that crosses kernels or steps (activations, mailboxes, GDN / PLE / QSA states)
// is laid out by live rows or by the capacity kMaxRows, never by the shape, so steps of
// either shape can follow each other in one sequence.
#pragma once

#include <cstdint>

namespace xenithium::kernels {

inline constexpr int kDecodeRows = 4;  // one token + 3 MTP drafts: the default step
// The largest kernel shape. The GEMM kernels (dense_gemm, the decode MoE) run 9-16 rows as one 16-row launch (two
// 8-row halves reading the same weight tiles) on 16 rows of x / y; the W4A16 decode GEMM runs 17-32 rows as one launch
// (four 8-row DPAS chains on each widened call), and qwen38's decode MoE, router, HC up-mix and CPU pool run 32 rows
// too (four 8-row quarters per item); the kernels that stop at 16 rows (the affine MoE: flint) check kStepRows16.
// 64: qwen38's decode MoE (eight 8-row blocks: a slot's packed rows in 32-row windows) and the dense GEMM as two
// 32-row launches; opt-in per model (a step above kStepRows32 rows runs only where a model asks for it); the kernels
// that stop at 32 rows check kStepRows32.
inline constexpr int kMaxShape = 64;
// The largest step the kernels' layouts carry (mailboxes, records, per-row scratch) is kMaxRows. An arch's StepModel
// says what it runs (max_rows): qwen35 32 (`XEN_STEP_ROWS`), qwen38 32 or 64 on request; the archs whose kernels stop
// at 8 rows (llama / flint: zero-point / affine W4, the flint MoE) keep kStepRows8. An arch that stops at 32 rows sizes
// and checks by kStepRows32.
inline constexpr int kStepRows8 = 8;
inline constexpr int kStepRows16 = 16;  // the 16-row shape (the affine MoE stops here)
inline constexpr int kStepRows32 = 32;  // the 32-row shape (every kernel but the 64-row ones stops here)
inline constexpr int kMaxRows = kMaxShape;

// The kernel shape that runs a step of `rows` live rows (1..kMaxShape).
constexpr int row_shape(int rows) { return rows <= 4 ? 4 : rows <= 8 ? 8 : rows <= 16 ? 16 : rows <= kStepRows32 ? kStepRows32 : kMaxShape; }

// Rows [lo, hi) of a step: the part of it one launch runs (a card's own rows: comm/wire.hpp, rows by owner).
struct RowRange {
  int lo = 0, hi = 0;
  constexpr int rows() const { return hi - lo; }
};


// ---- several sequences in one step ---------------------------------------------
// A step's rows may belong to several sequences: segment s = rows [s * seg_rows, +seg_rows)
// (every segment the same size, the last one may be short; seg_rows 0 = all rows, one
// segment). The stateful ops (GDN, PLE, QSA) read one control record per segment, `stride`
// device ints apart (kSegRec in a step):
// [0] position of the segment's first row
// [1] accepted rows of the segment's previous step (the GDN replay count)
// [2] the sequence's GDN parity (which of its two windows / pending logs is current)
// [3] the rope offset of the segment's rows: the first rope table row of the sequence's YaRN tier (qwen35 / llama
//     GQA with GqaShape:mrope, qwen38 QSA always), plus the M-RoPE delta after image input (qwen35); 0 = rope by position
// [4] state slot: which copy of the states / caches the sequence owns
// The slot is read on the device, so one recorded graph serves any slot assignment. A bare
// position word (stride 0, or any stride below kSegRec) is one segment on slot 0: the
// single-sequence callers keep working unchanged.
inline constexpr int kSegRec = 12;  // ints per record (the 16-row trees' second words included)
// a step's sequences (segments) at most: 64 (qwen38 on four cards, more than 16 slots: opt-in); the archs whose tables
// stop at 16 sequences (qwen35 and its probes) size and check by kSegs16 (a 32-row step is then 16 x 2, 8 x 4, 4 x 8
// or 2 x 16).
inline constexpr int kMaxSegs = 64;
inline constexpr int kSegs16 = 16;
inline constexpr int kSegPos = 0, kSegAccept = 1, kSegParity = 2, kSegRope = 3, kSegSlot = 4;
// DDTree: a segment's rows may be a draft TREE instead of a chain, a tree of up to 16 rows (4-bit parents), each word 64
// bits in two record ints (low, high).
// [5] / [8] (kSegTree / kSegTreeHi) = 0 for a chain, else kTreeFlag (bit 0) | the parent of row j (1..15) in bits
// [4 j, 4 j + 4): row 0 is the root (the last verified token), every other row's parent is an earlier row of the segment
// (rows in depth-first preorder: a row's parent is the row before it or one of that row's ancestors), row j sits at
// position [0] + depth(j) and reads only its ancestors.
// [6] / [9] (kSegPath / kSegPathHi) = the previous step's accepted rows (a root-to-node path of its tree): 0 = rows
// 0.. [1] - 1 (a chain), else kTreeFlag | the r-th accepted row in bits [4 r, 4 r + 4) for 1 <= r < [1] (tree_path_word;
// the 0-th is the root).
// [7] / [10] (kSegTreeAux / kSegTreeAux2) = tree_aux_word(parents): how the GDN core walks the tree with one state in
// registers (the "holder", the last row updated in place; before row 0 the state after the replay): bits [0, 16) = side
// rows (a leaf whose parent is the holder: computed from the holder's state without changing it), [16, 32) = rows whose
// state is saved (a later in-place row's parent once the holder moved on), [32, 48) = rows that first restore their
// parent's saved state, bit 48 = one row is saved (its state stays in shared local memory instead of
// GdnState:tree_S; the 8-row cores only). A chain (and a tree whose leaves come before their parent's next non-leaf
// child, every node with at most one non-leaf child) saves nothing.
inline constexpr int kSegTree = 5, kSegPath = 6, kSegTreeAux = 7, kSegTreeHi = 8, kSegPathHi = 9, kSegTreeAux2 = 10;
inline constexpr uint64_t kTreeFlag = 1;
inline constexpr int kTreeRows = 16;  // rows a tree segment holds at most (4-bit parents)
// a 64-bit record word from its low / high ints
constexpr uint64_t rec_word(const int32_t *w, int lo, int hi) {
  return static_cast<uint64_t>(static_cast<uint32_t>(w[lo])) | (static_cast<uint64_t>(static_cast<uint32_t>(w[hi])) << 32);
}
inline void rec_word_set(int32_t *w, int lo, int hi, uint64_t v) {
  w[lo] = static_cast<int32_t>(static_cast<uint32_t>(v)), w[hi] = static_cast<int32_t>(static_cast<uint32_t>(v >> 32));
}
constexpr int tree_parent(uint64_t tw, int j) { return j == 0 ? -1 : (tw & kTreeFlag) ? static_cast<int>((tw >> (4 * j)) & 15u) : j - 1; }
constexpr int tree_depth(uint64_t tw, int j) {
  int d = 0;
  while (j > 0) j = tree_parent(tw, j), ++d;
  return d;
}
// the aux word of a tree of n rows (par[0] = -1, par[j] < j); 0 for a chain
inline uint64_t tree_aux_word(const int *par, int n) {
  uint64_t side = 0, save = 0, rest = 0;
  bool kid[kTreeRows] = {};
  for (int j = 1; j < n; ++j) kid[par[j]] = true;
  int h = -1;
  for (int j = 0; j < n; ++j) {
    const int p = j == 0 ? -1 : par[j];
    if (!kid[j] && p == h && j > 0) {
      side |= 1ull << j;
      continue;
    }
    if (p != h) rest |= 1ull << j, save |= 1ull << p;
    h = j;
  }
  return side | (save << 16) | (rest << 32) | (__builtin_popcountll(save) == 1 ? 1ull << 48 : 0ull);
}
// the r-th accepted row of a path word (0 = a chain: row r)
constexpr int path_row(uint64_t pw, int r) { return (pw & kTreeFlag) ? (r == 0 ? 0 : static_cast<int>((pw >> (4 * r)) & 15u)) : r; }
// the path word of accepted rows rows[0.. n) (rows[0] = 0; 0 when they are 0.. n - 1)
inline uint64_t tree_path_word(const int *rows, int n) {
  uint64_t w = 0;
  bool chain = true;
  for (int r = 0; r < n; ++r) w |= static_cast<uint64_t>(rows[r]) << (4 * r), chain &= rows[r] == r;
  return chain ? 0ull : (w | kTreeFlag);
}
// the tree word ([5] / [8]) of n rows (par[0] = -1, par[j] < j)
inline uint64_t tree_word(const int *par, int n) {
  uint64_t w = kTreeFlag;
  for (int j = 1; j < n; ++j) w |= static_cast<uint64_t>(par[j]) << (4 * j);
  return w;
}
// A tree's rows from its nodes in any order with parents before children (jpar[i] = node i's parent, jpar[0] = -1,
// n <= kTreeRows nodes): depth-first preorder, each node's leaf children before its other children (join order within
// each): the order the GDN core walks with the fewest saves. node[r] = row r's node, par[r] = row r's parent row.
inline void tree_preorder(const int *jpar, int n, int *node, int *par) {
  bool kid[kTreeRows] = {};
  for (int i = 1; i < n; ++i) kid[jpar[i]] = true;
  int st[kTreeRows], sp = 0, r = 0, row_of[kTreeRows] = {};
  st[sp++] = 0;
  while (sp > 0) {
    const int x = st[--sp];
    row_of[x] = r;
    node[r] = x;
    par[r] = x == 0 ? -1 : row_of[jpar[x]];
    ++r;
    // pushed in reverse: popped leaves first, then the others, each in join order
    for (int i = n - 1; i >= 1; --i)
      if (jpar[i] == x && kid[i]) st[sp++] = i;
    for (int i = n - 1; i >= 1; --i)
      if (jpar[i] == x && !kid[i]) st[sp++] = i;
  }
}
// The path of row j from the root (its ancestors, then j) packed 4 bits per depth, and its depth
// (no array: walked up from j with its depth known, so a kernel keeps it in registers)
constexpr uint64_t tree_anc_word(uint64_t tw, int j, int *depth) {
  const int d = tree_depth(tw, j);
  uint64_t w = 0;
  for (int i = d, x = j; i >= 0; --i, x = tree_parent(tw, x)) w |= static_cast<uint64_t>(x) << (4 * i);
  *depth = d;
  return w;
}
// Segments of unequal rows. seg_rows = seg_het(B, k, n) (negative) marks a step of n segments whose first k have B rows
// each and the rest one row (the drafting sequences first, then the plain ones: the engine's layouts); seg_size is B
// (the kernels' per-segment shape), seg_count n. The layout is arithmetic, so the kernels read nothing more from the
// records. Row r's bits never depend on the layout: the kernels only find its segment another way.
// Two classes, seg_het2(B, k, n, B2): the first k segments B rows each, the rest B2 rows each (1 <= B2 <= B). k and n
// take 7 bits each (64 segments): B in bits 0-4, k 5-11, n 12-18, B2 - 1 19-23.
constexpr int seg_het2(int b, int k, int n, int b2) { return -(b | (k << 5) | (n << 12) | ((b2 - 1) << 19)); }
constexpr int seg_het(int b, int k, int n) { return seg_het2(b, k, n, 1); }
constexpr bool seg_is_het(int seg_rows) { return seg_rows < 0; }
constexpr int seg_het_b(int code) { return (-code) & 31; }
constexpr int seg_het_k(int code) { return ((-code) >> 5) & 127; }
constexpr int seg_het_n(int code) { return ((-code) >> 12) & 127; }
constexpr int seg_het_b2(int code) { return (((-code) >> 19) & 31) + 1; }
static_assert(seg_het_n(seg_het2(16, 63, 64, 15)) == 64 && seg_het_k(seg_het2(16, 63, 64, 15)) == 63 && seg_het_b2(seg_het2(16, 63, 64, 15)) == 15 &&
                  seg_het_b(seg_het2(16, 63, 64, 15)) == 16,
              "seg_het fields");
// rows of a seg_het step
constexpr int seg_het_rows(int code) { return seg_het_k(code) * seg_het_b(code) + (seg_het_n(code) - seg_het_k(code)) * seg_het_b2(code); }

// Segments of a step of `rows` rows split `seg_rows` each (0 = one segment; negative = seg_het).
constexpr int seg_count(int rows, int seg_rows) {
  return seg_rows < 0 ? seg_het_n(seg_rows) : seg_rows == 0 ? 1 : (rows + seg_rows - 1) / seg_rows;
}
constexpr int seg_size(int rows, int seg_rows) { return seg_rows < 0 ? seg_het_b(seg_rows) : seg_rows == 0 ? rows : seg_rows; }
// A seg_rows word a kernel entry point accepts for `rows` rows: 0, 1..rows, or seg_het with 1 <= k <= n <= kMaxSegs,
// B >= B2 >= 1 and k B + (n - k) B2 == rows (a heterogeneous step needs records: stride >= kSegRec).
constexpr bool seg_rows_ok(int rows, int seg_rows, int stride) {
  if (seg_rows >= 0) return seg_rows <= rows;
  const int b = seg_het_b(seg_rows), k = seg_het_k(seg_rows), n = seg_het_n(seg_rows), b2 = seg_het_b2(seg_rows);
  return stride >= kSegRec && b >= 1 && b2 >= 1 && b2 <= b && k >= 1 && k <= n && n <= kMaxSegs && k * b + (n - k) * b2 == rows;
}

// Segment s's first row and rows (any layout; rows = the step's rows).
constexpr void seg_span(int seg_rows, int rows, int s, int &row0, int &n) {
  if (seg_rows < 0) {
    const int b = seg_het_b(seg_rows), k = seg_het_k(seg_rows), b2 = seg_het_b2(seg_rows);
    row0 = s < k ? s * b : k * b + (s - k) * b2;
    n = s < k ? b : b2;
    return;
  }
  const int sz = seg_rows == 0 ? rows : seg_rows;
  row0 = s * sz;
  n = rows - row0 < sz ? rows - row0 : sz;
}

// Row r of a step on the device: its segment, its index inside the segment, its position and
// its slot. Plain integer code: usable in SIMT and ESIMD kernels alike.
struct SegRow {
  int seg, j, pos, slot;
};
inline SegRow seg_row(const int32_t *rec, int stride, int seg_rows, int r) {
  int s = 0, j = r;
  if (seg_rows < 0) {
    const int b = seg_het_b(seg_rows), kb = seg_het_k(seg_rows) * b, b2 = seg_het_b2(seg_rows);
    s = r < kb ? r / b : seg_het_k(seg_rows) + (r - kb) / b2;
    j = r < kb ? r - s * b : (r - kb) - (s - seg_het_k(seg_rows)) * b2;
  } else if (seg_rows > 0) {
    s = r / seg_rows;
    j = r - s * seg_rows;
  }
  const int32_t *b = rec + s * stride;
  return SegRow{s, j, b[kSegPos] + j, stride >= kSegRec ? b[kSegSlot] : 0};
}

// DDTree: the position a row's attention and rope see: a tree row's (record kSegTree) = the segment's first
// position + its depth; any other row's = its position (its KV slot stays at its position either way)
inline int seg_vpos(const int32_t *rec, int stride, const SegRow &sr) {
  if (stride >= kSegRec) {
    const int32_t *b = rec + sr.seg * stride;
    const uint64_t tw = rec_word(b, kSegTree, kSegTreeHi);
    if (tw & kTreeFlag) return b[kSegPos] + tree_depth(tw, sr.j);
  }
  return sr.pos;
}

}  // namespace xenithium:kernels
