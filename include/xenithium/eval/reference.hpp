// The reference side of the KL gate (host only): token files, a server's prompt log-probs as
// XLP1 rows (the reference dump), and dump-vs-dump comparison. Used by tools/kl (xenithium-kl) and tests/test_eval.cpp.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "xenithium/eval/logprobs.hpp"
#include "xenithium/util/json.hpp"

namespace xenithium::eval {

// Token file of `probe model --ppl`: int32 n_chunks, then per chunk int32 length + ids. Throws.
std::vector<std::vector<int32_t>> read_tokens(const std::string &path);

// The request body for one chunk: prompt = the token ids, max_tokens 1, temperature 0,
// prompt_logprobs = logprobs = k, return_tokens_as_token_ids.
std::string completion_request(const std::string &model, const std::vector<int32_t> &ids, int k);

// One /v1/completions response (to completion_request's body) -> n rows for n prompt tokens. Row j < n - 1 from
// choices[0].prompt_logprobs[j + 1] (keyed by token id; the prompt token rides along with its own
// rank when outside the top k, so entries with rank > k are dropped from the top list); the last
// row from the generated token's top_logprobs ("token_id:N" keys), target = the server's greedy
// pick. Throws on a shape that does not line up.
std::vector<LpRow> rows_from_completion(const json::Value &resp, const std::vector<int32_t> &ids, int k);

// The served model name (first of /v1/models). Throws.
std::string served_model(const std::string &base_url);

// POST one chunk and parse it. Throws on HTTP or shape errors.
std::vector<LpRow> dump_reference_chunk(const std::string &base_url, const std::string &model, const std::vector<int32_t> &ids, int k,
                                        double timeout_s);

// Dump vs dump: KL(ref || test) per position, the test side known only on its own top k (ids
// outside it take its last log-prob, counted as bounded). Throws when chunks / rows / targets
// (all but each chunk's last row) differ.
struct DumpCompare {
  KlStats stats;
  int last_rows_differ = 0;  // chunks whose last-row greedy pick differs between the two
};
DumpCompare compare_dumps(const LpSet &ref, const LpSet &test, const std::vector<int64_t> &edges);

}  // namespace xenithium::eval
