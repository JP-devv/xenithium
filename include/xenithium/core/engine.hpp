// The core's interface to everything above it (server/, tools).
//
// Threading contract: the engine owns its step loop on its own thread. Callers only
// submit, cancel and read stats; all of those are thread-safe and never block on a
// step. Results come back through EventSink::on_events, called from the engine
// thread once per step with every request's tokens for that step, so the server pays
// one wake-up per step, not one per token.
//
// Every submitted id receives exactly one event with finish != Finish::none (its
// last); after it the id is never mentioned again. cancel() of an id still running
// produces that final event with Finish::cancelled; cancel() of an unknown or
// finished id is a no-op.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>


namespace xenithium::core {

// A token-level constraint on a request's output (structured output). The engine starts
// one per request from SamplingParams::constraint, asks it for the allowed tokens before every pick
// and feeds it every emitted token.
class TokenConstraint {
 public:
  virtual ~TokenConstraint() = default;
  // Bitset over the vocabulary (bit t of word t / 64) of the tokens allowed next; null = any token.
  virtual const uint64_t* allowed() = 0;
  virtual void accept(int32_t token) = 0;
};

// Immutable and shared by every request that uses it (the server caches compiled grammars).
class ConstraintSpec {
 public:
  virtual ~ConstraintSpec() = default;
  virtual std::unique_ptr<TokenConstraint> start() const = 0;
};

struct SamplingParams {
  float temperature = 1.0f;  // 0 = greedy
  float top_p = 1.0f;
  int32_t top_k = 0;  // 0 = off
  float min_p = 0.0f;
  bool has_seed = false;
  uint64_t seed = 0;
  int32_t max_tokens = 16;
  bool ignore_eos = false;
  float repetition_penalty = 1.0f;
  float presence_penalty = 0.0f;
  float frequency_penalty = 0.0f;
  std::vector<int32_t> stop_token_ids;  // in addition to the model's EOS ids
  int32_t logprobs = -1;         // >= 0: each generated token's logprob + this many top alternatives
  int32_t prompt_logprobs = -1;  // >= 0: the prompt's logprobs (echo) + this many top alternatives
  std::unordered_map<int32_t, float> logit_bias;  // added to the logits before sampling (OpenAI logit_bias)
  std::shared_ptr<const ConstraintSpec> constraint;  // structured output: only allowed tokens are picked
  // Prefix cache: a prompt position worth a snapshot of the sequence state (the server sets it to
  // where a chat request's generation prompt starts: the conversation's next turn shares everything before it). -1 = none.
  int32_t prefix_mark = -1;

  // Plain greedy decoding: the device's own picks, no logits leave the cards.
  bool greedy() const {
    return temperature <= 0.0f && repetition_penalty == 1.0f && presence_penalty == 0.0f &&
           frequency_penalty == 0.0f && logit_bias.empty() && !constraint;
  }
};


// Logprob of one token (natural log, over the full vocabulary after penalties and
// temperature) and the top alternatives, best first.
struct TokenLogprob {
  float logprob = 0.0f;
  int32_t rank = 0;  // 1 + the tokens with a larger logit
  std::vector<std::pair<int32_t, float>> top;
};

// `expired`: the request waited in the queue past the deadline (--queue-timeout) and was
// never admitted; it produced no output.
enum class Finish : uint8_t { none, stop, length, cancelled, error, expired };

const char* finish_name(Finish f);  // "stop", "length", ... ("" for none)

// Sampling defaults from a checkpoint's generation_config.json (temperature, top_p,
// top_k, min_p, repetition_penalty), applied to requests that do not set
// them. Missing file or keys leave `sp` unchanged; returns false only on a parse error.
bool load_generation_defaults(const std::string& model_dir, SamplingParams& sp, std::string& err);

// One request's output for one step. MTP depth 3 accepts at most 4 tokens per step;
// the inline array keeps the per-step path free of allocations.
struct Event {
  static constexpr int kMaxTokens = 16;  // a step's tokens for one request (a 16-row verify keeps up to 16)
  uint64_t id = 0;
  uint8_t n = 0;
  int32_t tokens[kMaxTokens] = {};
  Finish finish = Finish::none;
  // The last token is a stop token (an EOS or stop_token_ids): counted in the usage but not
  // part of the text.
  bool stop_token = false;
  std::string error;  // set with Finish::error
  // Only when the request asked for them (SamplingParams::logprobs / prompt_logprobs):
  std::vector<TokenLogprob> logprobs;         // one per token in `tokens`
  std::vector<TokenLogprob> prompt_logprobs;  // once, on the request's first event: one per prompt
                                              // token, [0] has NaN (nothing predicts it)
  // Once, on the request's first event: the prompt tokens its prefill continued from the
  // prefix cache instead of computing (0 on a cold start); -1 on every other event.
  int32_t cached_tokens = -1;
};

class EventSink {
 public:
  virtual ~EventSink() = default;
  // Engine thread. Must not block: copy and hand off.
  virtual void on_events(const Event* events, size_t n) = 0;
};

// Seconds histogram with fixed bounds (Prometheus le buckets), cheap to copy with Stats.
struct Histogram {
  static constexpr int kBounds = 15;
  static constexpr double bounds[kBounds] = {0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5,
                                             1,     2.5,   5,    10,    30,   60,  300};
  uint64_t bucket[kBounds + 1] = {};  // per bucket (not cumulative); the last one is +Inf
  double sum = 0;
  uint64_t count = 0;
  void observe(double seconds) {
    int i = 0;
    while (i < kBounds && seconds > bounds[i]) ++i;
    ++bucket[i];
    sum += seconds;
    ++count;
  }
};

// Counters are monotonic; the server exports them as Prometheus metrics.
struct Stats {
  uint64_t steps = 0;
  uint64_t spec_drafts = 0;           // one per request per speculative step
  uint64_t spec_accepted_tokens = 0;  // accepted draft tokens (bonus token excluded)
  uint64_t spec_draft_tokens = 0;     // proposed draft tokens (depth per speculative step)
  // Accepted drafts by position: [i] counts steps that accepted
  // draft i, so [i] / [i - 1] is position i's conditional acceptance ([0] / spec_drafts).
  static constexpr int kSpecPositions = 8;
  uint64_t spec_accepted_at[kSpecPositions] = {};
  // Lookup drafting (BatchConfig::lookup): speculative steps whose drafts came from the requests' own tokens, and the
  // tokens of those drafts accepted (included in the spec_* counters above).
  uint64_t lookup_steps = 0, lookup_accepted_tokens = 0;
  uint64_t prompt_tokens = 0;
  uint64_t generation_tokens = 0;
  uint64_t requests_finished = 0;
  uint32_t running = 0;
  uint32_t waiting = 0;
  uint32_t slots = 0;  // requests that run at once; 0 = unlimited (nothing ever waits)
  // Shared KV pool (0 = none: every slot owns its context): tokens in the pool, tokens held by the
  // running requests' pages, and requests preempted to free pages (their KV dropped; they wait and recompute it).
  int64_t kv_pool_tokens = 0, kv_used_tokens = 0;
  uint64_t preemptions = 0;
  uint64_t yarn_retiers = 0;  // of them, requests that grew past their YaRN tier (re-prefilled in the next)
  // Prefix cache: prompt tokens of the requests that could use it (queries) and the tokens they
  // continued from a cached state instead of prefilling (hits); requests continued by kind (in place from the slot's live
  // state, in place from a snapshot, a copy into another slot); snapshots held now.
  uint64_t prefix_queries = 0, prefix_hits = 0;
  uint64_t prefix_live = 0, prefix_restores = 0, prefix_forks = 0;
  uint64_t prefix_holds = 0;  // requests that waited for a running request's prefill of their common prefix
  uint32_t prefix_snapshots = 0;
  // KV lending (qwen38): parked sequences dropped for the lent experts, by the park timeout / under reservation pressure
  uint64_t lend_park_drops = 0, lend_pressure_drops = 0;
  // Request timing, from submit(): time in the queue, admission to the first token
  // (prefill), submit to the first token (TTFT), submit to the last event.
  Histogram queue_time, prefill_time, ttft, e2e;
  // Every submitted request's last event by reason, indexed by Finish (queue cancels and
  // queue-deadline expiries included, so the sum is all requests that ended).
  static constexpr int kFinishKinds = 6;
  uint64_t finished_by[kFinishKinds] = {};
};

struct ModelInfo {
  std::string name;  // served model name
  int32_t max_model_len = 0;
  SamplingParams default_sampling;  // generation_config.json defaults
  bool prompt_logprobs = true;      // the engine can return the prompt's logprobs (echo + logprobs)
};

class Engine {
 public:
  virtual ~Engine() = default;
  virtual void start(EventSink* sink) = 0;  // spawns the step thread
  virtual void stop() = 0;                  // joins it; outstanding ids are dropped
  // `id` is caller-chosen and unique among live requests.
  virtual void submit(uint64_t id, std::vector<int32_t> prompt, const SamplingParams& params) = 0;
  virtual void cancel(uint64_t id) = 0;
  virtual Stats stats() const = 0;
  virtual const ModelInfo& model() const = 0;
};

}  // namespace xenithium::core
