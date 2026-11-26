// Xenithium · QSA constants, rotary table and the host reference (host-only: no SYCL). The
// semantics are in qsa.hpp. Built with precise floating point (xenithium_pack): the rotary table
// and the 8-bit KV codes must round like IEEE.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace xenithium::kernels {

inline constexpr int kQsaRows = 4;          // rows per step
inline constexpr int kQsaDim = 256;         // head_dim (kernel constant)
inline constexpr int kQsaRot = 64;          // rotated dims (NeoX: pairs d, d + 32)
inline constexpr int kQsaIdxDim = 128;      // indexer head_dim
inline constexpr int kQsaIdxHeads = 4;      // indexer query heads (one key head)
inline constexpr int kQsaCompress = 4;      // tokens per compressed key block
inline constexpr int kQsaTopBlocks = 512;   // indexer_budget / compress
inline constexpr int kQsaMaxSel = kQsaTopBlocks * kQsaCompress + kQsaCompress - 1;  // 2,051 tokens
inline constexpr int kQsaPage = 64;         // tokens per KV page
inline constexpr int kQsaPageBlocks = kQsaPage / kQsaCompress;  // 16 compressed keys per page
inline constexpr int kQsaRing = 16;         // raw index keys kept per sequence (by position % 16)
inline constexpr int kQsaChunk = 16384;     // most blocks per select work-group
inline constexpr int kQsaPfTiles = 8;  // attention tiles per prefill work-group (online softmax)
inline constexpr int kQsaPfChunk = 32768;  // the prefill's select chunk (one launch to 131,072 tokens; nch <= 8 lists)
inline constexpr int kQsaMinChunk = 4096;   // fewest blocks per chunk (one launch up to here)
inline constexpr int kQsaMaxChunks = 16;    // chunk work-groups per row (candidate lists of the merge)
inline constexpr int kQsaMaxBlocks = 262144;               // 1,048,576 tokens
inline constexpr int kQsaMaxTokens = kQsaMaxBlocks * kQsaCompress;
inline constexpr int kQsaTile = 64;         // attention tokens per work-group
inline constexpr int kQsaV4Group = 32;  // int8-v4: V dims per scale
inline constexpr int kQsaV4Scales = kQsaDim / kQsaV4Group;  // bf16 V scales per (token, kv head)
inline constexpr int kQsaTiles = (kQsaMaxSel + kQsaTile - 1) / kQsaTile;  // 33

// ---- rotary table (host) ------------------------------------------------------------------------
// bf16 [positions][64]: cos of the 32 frequencies | sin of them, times the attention factor.
// Static YaRN (factor > 1): inverse frequencies blend the original (pairs below `low`) and
// original / factor (above `high`) over a linear ramp; attention factor 1 + 0.1 ln(factor) unless
// given. Angles are fp32(p) * fp32(inv_freq) rounded to fp32.
struct QsaYarn {
  double factor = 1.0;      // 1 = no scaling
  int64_t orig = 262144;    // trained positions
  double beta_fast = 32, beta_slow = 1;
  double attn_factor = 0;   // 0 = 1 + 0.1 ln(factor)
};
struct QsaYarnBounds {
  int low = 0, high = 0;
};
QsaYarnBounds qsa_yarn_bounds(int rot_dim, double theta, int64_t orig, double beta_fast, double beta_slow);
// The static YaRN a load applies for a served context (XEN_CONTEXT, the longest request): factor 1 (the plain table) up
// to the trained positions, past them the smallest factor in steps of 0.01 with trained x factor >= context
// (333,000 of 262,144 -> 1.28; 524,288 -> 2; 1,048,576 -> 4). A smaller factor moves short contexts less.
QsaYarn qsa_yarn_for(int64_t context, int64_t trained);
std::vector<uint16_t> qsa_rope_table(int64_t positions, int rot_dim, double theta, const QsaYarn &y);
inline constexpr int kQsaRopeRow = kQsaRot;  // bf16 per position

// ---- host reference ----------------------------------------------------------------------------
// gemma RMSNorm then NeoX rope over the first 64 dims, on bf16 values; out bf16 values.
void qsa_ref_norm_rope(const float *x, const float *w, int dim, const uint16_t *rope_row, float eps, float *out);
// int8 (qsa.hpp): *scale = amax / 127, codes = rne(x (127 / amax)) clamped to +-127; all zero: 0, 0.
void qsa_ref_i8(const float *x, int n, int8_t *codes, float *scale);
// int8-v4's V row (qsa.hpp), 256 values: per group of kQsaV4Group dims a bf16 scale s = bf16(amax / 7) (bits in scales[g]),
// codes = rne(x (1 / s)) clamped to +-7 (both divisions IEEE); an all-zero group: s = 0, codes 0.
void qsa_ref_v4(const float *x, int8_t *codes, uint16_t *scales);
// Compressed key of a block from its 4 raw keys (bf16 values [4][128]), rope row of its first token.
void qsa_ref_compress(const float *raw4, const float *w, const uint16_t *rope_row, float eps, float *kc);
// Block score: sum_h relu(iq_h . kc) / sqrt(128); iq [4][128].
double qsa_ref_score(const float *iq, const float *kc);
// Top-k keys (ties: lowest index), ascending ids; all of them when n <= k.
std::vector<int> qsa_ref_select(const uint32_t *keys, int n, int k);
// Token list of a row at position p from its selected blocks (nblk = (p + 1) / 4).
std::vector<int> qsa_ref_tokens(const std::vector<int> &blocks, int p);
// softmax(q . K / 16) . V for one q head over ntok tokens; K / V values [ntok][256]; o [256].
void qsa_ref_attend(const float *q, int ntok, const float *K, const float *V, double *o);
// gated output of one head: bf16(bf16(o) bf16(sigmoid(gate))).
void qsa_ref_gate(const double *o, const float *gate, float *out);

}  // namespace xenithium::kernels
