// Xenithium · Gated DeltaNet prefill recurrence as a chunked bf16-DPAS scan.
// gdn_rows' step 5 (gdn_rows.hpp) over full chunks of 64 rows. Per value head, row form:
// S *= a_t; d = (v_t - S k_t) b_t; S += d k_t^T; o_t = bf16(S q_t)
// Chunk form from S0 over 64 rows, g = cumsum(ln a) inside the chunk (only e^(g_i - g_j) <= 1 formed):
// pre T = I + strict_lower(diag(b) (G o K K^T)), Ti = T^-1 (fp32 forward substitution);
// Uv = Ti diag(b) V (fp32), Wk = Ti diag(b e^g) K, Qg = diag(e^g) Q, Kt_i = e^(g_63 - g_i) k_i,
// M = lower_incl(G o Q K^T). The gram and the Ti products run on DPAS with hi / lo bf16
// operands (three passes, ~fp32); Wk, Qg, Kt and M are stored as bf16 DPAS B blocks.
// scan serial over chunks, S in fp32 registers:
// X | Y = [Wk; Qg] bf16(S0)^T; U = Uv - X; O = Y + M U; S = e^(g_63) S + U^T Kt
// with U and Kt as hi / lo bf16 pairs (two / three DPAS passes).
// Numerics: S0, Wk, Qg and M enter as bf16.
// Chunks sit at absolute positions that are multiples of 64, so a prompt cut into calls at multiples
// of 64 gives the same bits as one call. Rows before the first boundary and after the last full
// chunk stay on gdn_rows' row core (gdn_scan_split).
// Needs 256 GRF: built into its own shared library (xenithium_kernels_xl, `-options -doubleGRF`),
// because the grf_size kernel property does not reach ESIMD kernels under AOT.
#pragma once

#include <cstddef>
#include <cstdint>
#include <sycl/sycl.hpp>

#include "xenithium/kernels/linear_attention/gdn.hpp"

namespace xenithium::kernels {

inline constexpr int kGdnChunk = 64;

// A call's T = accept + m rows, row 0 at absolute position base = pos - accept: rows [0, lead) and
// [lead + 64 full, T) run on the row core, [lead, lead + 64 full) on the scan.
inline void gdn_scan_split(int base, int T, int &lead, int &full) {
  const int f = (kGdnChunk - base % kGdnChunk) % kGdnChunk;
  lead = f < T ? f : T;
  full = T > f ? (T - f) / kGdnChunk : 0;
}
// The most full chunks a call of m rows can hold (accept <= 4).
inline int gdn_scan_max_chunks(int m) { return (m + kGdnRows) / kGdnChunk; }

struct GdnScanScratch {
  gdn_bf16 *wk = nullptr, *qg = nullptr, *kt = nullptr, *mt = nullptr;  // per (chunk, value head) B blocks
  gdn_bf16 *kt_lo = nullptr;  // Kt's lo half (as kt)
  float *uv = nullptr;  // per (chunk, value head) Uv^T [128][64]
  float *gc = nullptr;  // per (chunk, value head) e^(g_63)
};
struct GdnScanSizes {
  size_t wk, mt, uv, gc;  // elements; qg, kt and kt_lo as wk
};
GdnScanSizes gdn_scan_sizes(const GdnShape &s, int rows_max);

// The gated RMSNorm folded into the scan's output (gdn_rows step 6 on the scan's rows): with y
// set, the scan stores y = bf16(o * rsqrt(mean(o^2) + eps) * w * sigmoid(bf16(z))) of each output row
// (o = the bf16 output row) instead of o. z row r at z + r * ldz (+ head * 128). The same expression as
// gdn_rows' norm, so the same bits; gdn_rows then norms only the row core's lead / trail rows.
struct GdnScanNorm {
  const float *z = nullptr;
  int ldz = 0;
  const gdn_bf16 *w = nullptr;  // [128]
  float eps = 0.f;
  gdn_bf16 *y = nullptr;  // [m][val_w]; nullptr = store o (no fold)
  bool swish = false;  // gate silu(z) = z sigmoid(z) (GdnShape:swish)
};

// Full chunks of a gdn_rows call (inputs as GdnRowsScratch: qn / kn [T][key_w], cv [T][conv_ch],
// gate [T][2 vh] decay | beta, lgate [T][vh] = ln decay). pos = the first new row's absolute
// position. Updates S in place and writes o rows t >= accept (o row t - accept). Two launches.
// part 0 = every chunk; part 1 = the chunks that end at or before the absolute position
// snap_pos (a multiple of 64), part 2 = the rest (the scan alone, its pre stays part 1's): S holds the state at snap_pos
// between a part 1 and a part 2 call, and the pair computes what part 0 does, bit for bit.
// pre_ev, if set, receives the precompute launch's event (part 2 has none); the returned event is the
// scan's (prefill stage traces time both). rec, if set (part 0 only): the call's first slice record on the device
// (rows.hpp), the kernels read pos from it (kSegPos) and take S at its slot (S + rec[kSegSlot] slot_S), so a recorded
// launch serves every position and slot.
sycl::event gdn_scan(sycl::queue &q, const GdnShape &s, int m, int pos, const int32_t *accept, const float *qn, const float *kn, const float *cv,
                     const float *gate, const float *lgate, float *S, gdn_bf16 *o, const GdnScanScratch &sc, const GdnScanNorm &nm = {}, int part = 0,
                     int snap_pos = -1, sycl::event *pre_ev = nullptr, const int32_t *rec = nullptr, size_t slot_S = 0);

}  // namespace xenithium:kernels
