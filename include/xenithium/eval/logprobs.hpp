// Per-position log-prob records and the KL gate against a reference dump (host only).
//
// XLP1 file (little endian): char magic[4] "XLP1", int32 k, int32 n_chunks; per chunk int32 rows,
// then per row: int32 target, float target_lp, int32 ids[k], float lps[k]. ids/lps are the row's top
// k log-probs, descending (ties: lower id first); unused slots hold id -1, lp -inf. Row j of a chunk
// is the distribution after token j, i.e. it predicts token j + 1 (the last row predicts the token
// after the chunk; its target is the writer's own greedy pick). Written by `probe model --lp-out`
// and by `xenithium-kl dump` (a server's prompt log-probs: the reference dump); compared by
// `probe model --lp-ref` (exact on the test side) or `xenithium-kl compare` (dump vs dump).
#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace xenithium::eval {

struct LpRow {
  int32_t target = -1;
  float target_lp = 0;
  std::vector<int32_t> ids;  // k
  std::vector<float> lps;    // k, descending
};

struct LpSet {
  int k = 0;
  std::vector<std::vector<LpRow>> chunks;
};

bool write_lp(const std::string &path, const LpSet &s, std::string *err = nullptr);
bool read_lp(const std::string &path, LpSet &s, std::string *err = nullptr);

// Log-softmax of one row of logits, in double (as the ppl gate): the top k and the target's log-prob.
LpRow top_logprobs(const float *logits, int64_t vocab, int k, int32_t target);

// KL(ref || test) on the reference row's top k plus one tail bucket:
//   sum_i p_i (lr_i - lt_i) + P_tail ln(P_tail / Q_tail),  P_tail = 1 - sum p_i, Q_tail = 1 - sum q_i.
// Grouping never increases KL, so this is a lower bound on the full-vocab KL that loses only the
// reference row's truncation (exact when its tail carries no mass). test_lp(id) gives the
// test's log-prob of any id; `missing` counts ids it could only bound (dump vs dump).
struct KlRow {
  double kl = 0;
  bool top1_same = false;
  int missing = 0;
};
KlRow kl_coarse(const LpRow &ref, const std::function<double(int32_t, bool &)> &test_lp, int32_t test_top1);

// Accumulates KlRow per position bucket and overall.
struct KlStats {
  struct Bucket {
    int64_t rows = 0, top1 = 0, missing = 0;
    double kl = 0, kl_max = 0, nll_ref = 0, nll_test = 0;
    std::vector<double> kls;  // per row, for percentiles
  };
  std::vector<int64_t> edges;    // lower edges of buckets 1..: [0, e0), [e0, e1), ..., [e_last, inf)
  std::vector<Bucket> buckets;   // edges.size() + 1
  Bucket total;
  explicit KlStats(std::vector<int64_t> e);
  void add(int64_t pos, const KlRow &r, double ref_target_lp, double test_target_lp);
  // One table: positions, rows, ppl ref / test (each side's own targets), mean / p99 / max KL, top-1 same.
  std::string table() const;
  const Bucket &all() const { return total; }
};

double percentile(std::vector<double> v, double p);

}  // namespace xenithium::eval
