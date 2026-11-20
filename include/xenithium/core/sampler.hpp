// Host sampling for requests the device's greedy pick cannot serve (temperature > 0,
// penalties, logit_bias, logprobs, a token mask).
//
// The order of the logits processing: logit_bias, then the
// penalties (repetition over prompt + output: positive logits divided, negative ones
// multiplied; presence / frequency over the output: subtracted), then temperature, then
// min_p, top_k and top_p on what is left, then a draw from the renormalised distribution.
// Logprobs are the log-softmax of the raw logits (before any processing). A seed makes the
// output reproducible.
//
// Every draw is a function of (seed, output position) only: one uniform per position from a counter-based hash, one
// inverse-CDF walk over the kept set. A speculative row draws the same way and accepts the draft iff the draw equals it
// (probability p(draft), the most a deterministic draft can get, and the output follows p). So a seeded request writes
// the same text whatever the draft depth its step gave it (alone, or one of 8 streams).
//
// Coupled drafting: a request on the candidate path (temperature > 0, top_k in [1, kMaxCandidates]) draws
// by Gumbel-max instead, argmax over its kept set of log w + util::gumbel(key, position, token), and hands the same
// (key, position, 1 / T) to the drafter (draft_noise), which picks argmax of logit / T + the same noise: its draft is
// then the target's token whenever the two distributions put their noise-perturbed maxima on the same token (the
// acceptance rises from p(argmax q) toward sum min(p, q)). Still a function of (seed, position) only.
//
// Speed: with top_k in [1, kMaxCandidates] (the checkpoint default is 20) the row is
// read once. Only the tokens a modifier touches (logit_bias, the penalties' seen tokens) get
// processed values; every other token keeps its raw logit, so the top_k of the processed row is
// the top_k of the untouched raw logits merged with the touched tokens: exact, without copying or
// sorting the 248K-entry row. Other settings (top_k off, or above the cap) take the full path.
#pragma once

#include <cstdint>
#include <random>
#include <unordered_map>
#include <vector>

#include "xenithium/core/engine.hpp"

namespace xenithium::core {

class Sampler {
 public:
  static constexpr int kMaxCandidates = 256;  // top_k up to this reads the row once (candidates)

  // prompt: for the repetition penalty (prompt tokens count as seen).
  Sampler(const SamplingParams &p, const std::vector<int32_t> &prompt);

  // Picks the next token from one row of logits (`vocab` floats). Fills `lp` (when not
  // null) with the raw logprob of the picked token and the `top_n` best alternatives.
  // allowed (optional): a bitset over the vocabulary (bit t of word t / 64); only allowed tokens
  // can be picked. An empty allowed set is ignored (a broken constraint must not fail a request).
  int32_t sample(const float *logits, int64_t vocab, TokenLogprob *lp, int top_n, const uint64_t *allowed = nullptr);
  // The distribution sample() draws from: kept token ids and their unnormalised weights, in the
  // draw order (empty for greedy). For the gates.
  void distribution(const float *logits, int64_t vocab, std::vector<int32_t> &ids, std::vector<double> &w,
                    const uint64_t *allowed = nullptr);
  // Speculative verification of an argmax drafter's token `draft` against this row: the position's draw (the
  // token sample() would return here); the caller accepts the draft iff it is that token (probability p(draft); the
  // output follows p). Greedy requests: the pick. Returns the token.
  int32_t speculate(const float *logits, int64_t vocab, int32_t draft, const uint64_t *allowed = nullptr);
  // Records a token that was emitted (the device's pick or ours) for the penalties.
  void accept(int32_t token);
  // The drafter's noise for this request's next drafts: key, the output position the first draft predicts (the next
  // draw's), 1 / temperature; inv_t 0 = the drafter keeps its argmax (greedy, or a request off the coupled path).
  struct DraftNoise {
    uint64_t key = 0;
    uint64_t pos = 0;
    float inv_t = 0;
  };
  DraftNoise draft_noise() const;
  // Gumbel-max draws (the candidate path): the requests whose drafts are coupled.
  bool coupled() const;

  // Raw logprob of `token` and the top_n best (log-softmax of the unprocessed logits).
  static void raw_logprobs(const float *logits, int64_t vocab, int32_t token, int top_n, TokenLogprob &lp);
  // raw_logprobs in two halves: the row's part that does not depend on the token (max, log-sum-exp,
  // the top_n), which a speculative step computes for all its rows at once, then the token's rank and logprob. The same
  // values as raw_logprobs.
  struct LpStats {
    float max = 0;
    double lse = 0;
    std::vector<int32_t> top;
  };
  static void raw_logprob_stats(const float *logits, int64_t vocab, int top_n, LpStats &s);
  static void raw_logprobs_from(const float *logits, int64_t vocab, int32_t token, const LpStats &s, TokenLogprob &lp);
  // The n best ids of x (descending value, ties: the lower id first), skipping ids with skip[id] != 0
  // (skip may be null), appended to out in that order.
  static void top_ids(const float *x, int64_t vocab, size_t n, const uint8_t *skip, std::vector<int32_t> &out);
  // raw_logprobs from per-part statistics (each card reduces its vocabulary part on the
  // device): part c's max, sum of exp(x - max), count of values above the target, and its best
  // entries (id, value; any order, ties at the cut: the lower ids). xt: the target's value.
  struct LpPart {
    float max = 0;
    double sum = 0;
    int64_t above = 0;
    std::vector<std::pair<int32_t, float>> top;
  };
  static void merge_logprobs(const LpPart *parts, int n, float xt, int top_n, TokenLogprob &lp);

  // One row's kept set and its scratch. The rows of a speculative step can be prepared at once:
  // row j of a segment is only read when its drafts t[1..j] were all emitted, so it is prepared as if they had been
  // accepted (`extra`), and the draw then runs in order on this sampler's RNG - the same tokens as speculate() row by row.
  struct Row {
    int32_t pick = -1;                             // the greedy pick (T <= 0), else -1
    std::vector<std::pair<int32_t, float>> mods;   // touched tokens and their processed logits
    std::vector<uint8_t> flag;                     // [vocab]: 1 = in mods (cleared after each call)
    std::vector<float> work;
    std::vector<int32_t> idx;
    std::vector<double> w;                         // kept tokens' unnormalised weights (parallel to idx)
  };
  // Read-only on the sampler (several rows of one request may run on several threads); no token mask (a constrained
  // request's mask follows its grammar state: those rows stay on speculate()).
  void prepare_row(const float *logits, int64_t vocab, const int32_t *extra, int n_extra, Row &row) const;
  // speculate() on a prepared row, once the extras it assumed were accepted.
  int32_t speculate_row(Row &row, const float *logits, int64_t vocab, int32_t draft);

  // Candidate rows: the step model hands over each row's candidates instead of the
  // whole row - ids / values (rounded as the full row; any order, distinct ids) holding the row's k best under the
  // candidate path's order (value descending, lower id first) for every k <= T. cands_ok(T): this request's processing
  // reads nothing else (temperature > 0, top_k in [1, T], no logit_bias, no penalties); the caller adds: no token mask,
  // no logprobs. prepare_cands = prepare_row on such a row (the same kept set and weights, so the same draws). A kept set
  // left empty (every candidate -inf) needs the full row: speculate_row / sample_cands must not be called then
  // (row.idx.empty() && row.pick < 0); sample_cands returns -1 in that case without drawing.
  bool cands_ok(int T) const;
  void prepare_cands(const int32_t *ids, const float *vals, int n, Row &row) const;
  int32_t sample_cands(const int32_t *ids, const float *vals, int n);

 private:
  // Greedy pick (T <= 0) or the kept set + weights (T > 0) into row; returns the greedy pick or -1. extra: tokens taken
  // as accepted after the sampler's state (prepare_row).
  int32_t prepare(const float *logits, int64_t vocab, const uint64_t *allowed, const int32_t *extra, int n_extra, Row &row) const;
  int32_t prepare(const float *logits, int64_t vocab, const uint64_t *allowed) { return prepare(logits, vocab, allowed, nullptr, 0, cur_); }
  void modifiers(const float *logits, int64_t vocab, const uint64_t *allowed, const int32_t *extra, int n_extra, Row &row) const;  // fills mods + flag
  void full_path(const float *logits, int64_t vocab, const uint64_t *allowed, Row &row) const;
  int32_t draw(Row &row, const float *logits, int64_t vocab, int32_t draft, const uint64_t *allowed);  // speculate's draw
  // The candidate path's tail (shared with prepare_cands): c = (processed value, id) pairs of the candidates, the k
  // best kept in order, then temperature, min_p and top_p into row.idx / row.w.
  void keep_top(std::vector<std::pair<float, int32_t>> &c, size_t k, Row &row) const;
  // The inverse-CDF draw over row.idx / row.w with this position's uniform; -1 when nothing has weight.
  int32_t walk(const Row &row) const;
  // This output position's uniform in (0, 1): a hash of (key_, n_out_).
  double uniform() const;

  SamplingParams p_;
  uint64_t key_ = 0;    // the seed, or a random one
  uint64_t n_out_ = 0;  // tokens emitted so far (accept): the position of the next draw
  std::unordered_map<int32_t, int32_t> out_counts_;  // output tokens
  std::vector<int32_t> seen_;                        // distinct prompt + output tokens (repetition)
  std::unordered_map<int32_t, bool> seen_set_;
  Row cur_;                                          // the per-call scratch of the one-row calls
};

}  // namespace xenithium::core
