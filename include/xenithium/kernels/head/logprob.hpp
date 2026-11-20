// Xenithium · log-softmax statistics of logits rows on the card.
// Logprobs are log-softmax values over the full vocabulary of the
// bf16-rounded logits. A row's logits are split over the cards (vocab halves), so each card
// reduces its own half to a small record and the host merges the records:
// max m_c, s_c = sum exp(x - m_c), count of x > xt (the target token's bf16 logit, passed in),
// and the card's `top` best (global id, value), ties to the lower id.
// Merged: lse = M + log(sum_c s_c exp(m_c - M)), logprob = xt - lse, rank = 1 + sum_c count_c,
// top = the best `top` of both lists. Only the records leave the card (64 floats per row), not
// the 4 x 124,160 floats of a card's logits.
// One work-group per row; three passes over the row (cache-resident after the first): max, the
// count above xt and a histogram of the high byte of the order-preserving bf16 key; then the sum
// of exponentials and a histogram of the low byte inside the threshold bin; then the collection of
// every key above the top-th largest key plus the lowest-index ties at it.
#pragma once

#include <cstdint>

#include <sycl/sycl.hpp>

#include "xenithium/kernels/rows.hpp"  // kMaxRows (kLpMaxRows)

namespace xenithium::kernels {

inline constexpr int kLpMaxRows = kMaxRows;  // rows of one head pass (a decode step's rows, rows.hpp kMaxRows)
inline constexpr int kLpMaxTop = 20;
inline constexpr int kLpRec = 64;  // floats per row record

// Record layout (floats): [0] max, [1] sum exp(x - max), [2] count x > xt, [3] entries k, then k
// pairs (global id as float, exact below 2^24; value) in no particular order.
struct LpArgs {
  float xt[kLpMaxRows] = {};  // per row; +inf = no target (count 0)
};

// logits fp32 [rows][ld] (n columns used), col_base: the global id of column 0. top 0..kLpMaxTop.
sycl::event logprob_stats(sycl::queue &q, const float *logits, int ld, int rows, int n, int col_base, int top, const LpArgs &a, float *rec);

// Prompt logprobs over many rows at once: the same records for any number of rows of bf16
// logits (the values logprob_stats rounds fp32 logits to, so the records are the same), the targets'
// bf16 logits from a device array xt [rows] (+inf = no target).
sycl::event logprob_stats_rows(sycl::queue &q, const sycl::ext::oneapi::bfloat16 *logits, int ld, int rows, int n, int col_base, int top, const float *xt,
                               float *rec);
// xt[r] = the logit of targets[r] (device int32 [rows], global ids) when this card holds that column
// (col_base <= t < col_base + n), else 0: the caller takes each row's value from the owning card.
sycl::event logprob_targets(sycl::queue &q, const sycl::ext::oneapi::bfloat16 *logits, int ld, int rows, int n, int col_base, const int32_t *targets, float *xt);

}  // namespace xenithium:kernels
