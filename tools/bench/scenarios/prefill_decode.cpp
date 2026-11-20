// Scenario "prefill-decode": one agent's long prompt is read while three other agents generate. Prompt processing
// and decoding share the same cards step by step, so each slows the other: the prompt's rate is compared with the
// same prompt read alone, the streams' rate while it is read with their own rate just before, and the longest pause
// a generating stream saw is reported. "late-arrival" asks how long a newcomer waits beside short chats; this asks
// how an engine divides its time between a reading agent and working agents that hold contexts of their own.
#include <algorithm>

#include "scenario.hpp"
#include "stats.hpp"

namespace xenithium::bench {
namespace {

constexpr ParamSpec kParams[] = {
    {"decoders", "3", "agents generating while the prompt is read"},
    {"depth", "64000", "the prompt that is read, in tokens"},
    {"decoder_depth", "32000", "the context each generating agent holds, in tokens (0 = none)"},
    {"after_s", "4", "seconds the agents generate before the prompt is sent: their rate alone is taken from this time"},
    {"stream_tokens", "8000", "tokens each agent generates (long enough to outlast the prompt's processing)"},
    {"reply_tokens", "16", "tokens generated after the prompt"},
    {"kind", "code", "corpus kind of the prompt and the contexts"},
};
constexpr std::string_view kQuestion = "\n\nIn one sentence, what is the text above?";
constexpr std::string_view kContextLead = "\n\nWith the text above in mind: ";
constexpr double kPercent = 100;

double median_of(const Row &row, std::string_view name) {
  for (const Metric &m : row.metrics)
    if (m.name == name) return summarize(m.samples).median;
  return 0;
}

class PrefillDecode final : public Scenario {
 public:
  std::string_view id() const override { return "prefill-decode"; }
  std::string_view title() const override { return "One agent reading a long prompt while others generate"; }
  std::string_view about() const override {
    return "A 64K-token prompt read alone, then read while three agents generate on 32K-token contexts of their own: the prompt's rate against alone, "
           "the agents' token rate against just before, and their longest pause.";
  }
  std::span<const ParamSpec> params() const override { return kParams; }

  ScenarioResult run(Session &session, const Params &params) const override {
    ScenarioResult result;
    const int reply_tokens = static_cast<int>(params.integer("reply_tokens")), stream_tokens = static_cast<int>(params.integer("stream_tokens"));
    const int64_t depth = params.integer("depth"), decoder_depth = params.integer("decoder_depth"), decoders = params.integer("decoders");
    const std::string kind = params.text("kind");
    std::string skipped;
    const int64_t deepest = std::max(depth + reply_tokens, decoder_depth + stream_tokens);
    if (depths_that_fit(session, std::span(&deepest, 1), 0, skipped).empty()) {
      result.rows.push_back(Row::note_only(skipped));
      return result;
    }
    // all of it is held at once: the prompt and every generating agent's context with what it generates
    const int64_t held_at_once = depth + reply_tokens + decoders * (decoder_depth + stream_tokens);
    if (const int64_t pool = session.kv_pool_tokens(); pool > 0 && held_at_once > pool) {
      result.rows.push_back(Row::note_only("skipped: the case holds " + with_thousands(held_at_once) + " tokens of context at once and the endpoint's KV pool is " +
                                           with_thousands(pool) + ": lower prefill-decode.depth, decoder_depth or decoders"));
      return result;
    }
    (void)session.chars_per_token(kind);  // calibrate before anything is timed
    size_t first_file = 0;  // every document reads another part of the corpus
    const auto new_prompt = [&] { return session.greedy(session.document_prompt(kind, depth, kQuestion, /*unique=*/true, first_file++).text, reply_tokens); };

    // the prompt read with nothing else running
    Row alone = Row::with_labels({{"prompt", "read alone"}});
    {
      const PowerSpan power(session);
      const Sample sample = session.client().chat(new_prompt());
      if (tally(result, sample)) {
        add_prefill_metrics(alone, sample);
        power.add_to(alone, sample.prompt_tokens);
      }
    }
    session.progress("prefill-decode: the prompt alone done");

    // the generating agents: each reads its context first (not timed), then generates on it from the cache
    const auto prompts = short_prompts();
    std::vector<ChatRequest> requests;
    for (int64_t i = 0; i < decoders; ++i) {
      // a task of another kind for each agent: the prompts come three per category in a row
      const std::string task(prompts[(static_cast<size_t>(i) * 4 + 3) % prompts.size()].text);
      std::string text = task;
      ChatRequest generate = session.sampled({}, stream_tokens);
      generate.thinking = Thinking::off;
      if (decoder_depth > 0) {
        text = session.document_prompt(kind, decoder_depth, std::string(kContextLead) + task, /*unique=*/true, first_file++).text;
        // the same request with a one-token reply: a prefix cache continues only from a prompt that is the same to
        // its last token, so the thinking mode (the prompt's tail) must not differ between this and the timed request
        ChatRequest read_context = generate;
        read_context.messages = {{"user", text}};
        read_context.max_tokens = 1;
        tally(result, session.client().chat(read_context));
      }
      generate.messages = {{"user", std::move(text)}};
      requests.push_back(std::move(generate));
    }
    requests.push_back(new_prompt());
    std::vector<double> start_after_s(requests.size(), 0.0);
    start_after_s.back() = params.real("after_s");
    const PowerSpan power(session);
    const std::vector<Sample> samples = session.run_together(requests, start_after_s);

    Row beside = Row::with_labels({{"prompt", "read beside " + std::to_string(decoders) + " generating"}});
    const Sample &reader = samples.back();
    bool usable = tally(result, reader);
    for (int64_t i = 0; i < decoders; ++i) usable &= tally(result, samples[static_cast<size_t>(i)]);
    if (usable) {
      add_prefill_metrics(beside, reader);
      const Clock::time_point from = reader.submitted, to = first_token_time(reader);
      int ended_early = 0;
      for (int64_t i = 0; i < decoders; ++i) {
        const Sample &stream = samples[static_cast<size_t>(i)];
        const StreamInWindow w = split_at(stream, from, to);
        ended_early += !w.outlasted;
        if (w.before_s > 0) beside.metric("decode before", "tok/s", Better::higher).samples.push_back(w.updates_before * stream.tokens_per_update() / w.before_s);
        beside.metric("decode meanwhile", "tok/s", Better::higher).samples.push_back(w.updates_inside * stream.tokens_per_update() / (reader.ttft_ms / 1e3));
        beside.metric("longest gap", "ms", Better::lower).samples.push_back(w.longest_gap_ms);
      }
      // what each side keeps of its speed alone; the two add up to about 100 % when the cards' time is only divided
      if (const double rate_alone = median_of(alone, metric::kPrefill); rate_alone > 0)
        beside.metric("prompt keeps", "%", Better::none).samples.push_back(kPercent * reader.prefill_tps() / rate_alone);
      if (const double before = median_of(beside, "decode before"); before > 0)
        beside.metric("decode keeps", "%", Better::none).samples.push_back(kPercent * median_of(beside, "decode meanwhile") / before);
      power.add_to(beside, 0);
      if (median_of(beside, "decode before") <= 0)
        beside.notes.push_back("the agents had not produced a token when the prompt was sent (their contexts were not served from a prefix cache?): raise after_s for this endpoint");
      if (ended_early > 0)
        beside.notes.push_back(std::to_string(ended_early) + " of the agents finished before the prompt's first token: raise stream_tokens for this endpoint");
    }
    session.progress("prefill-decode: the prompt beside " + std::to_string(decoders) + " generating agents done");
    result.rows.push_back(std::move(alone));
    result.rows.push_back(std::move(beside));
    result.charts = {
        {ChartKind::bars, "The prompt's rate: alone and beside the generating agents", "prompt", {std::string(metric::kPrefill)}, false},
        {ChartKind::bars, "The generating agents' rate: before and while the prompt is read", "prompt", {"decode before", "decode meanwhile"}, false},
        {ChartKind::range, "The generating agents' longest pause while the prompt is read", "prompt", {"longest gap"}, false},
        {ChartKind::bars, "Share of its own speed each side keeps", "prompt", {"prompt keeps", "decode keeps"}, false},
    };
    return result;
  }
};

}  // namespace

std::unique_ptr<Scenario> make_prefill_decode() { return std::make_unique<PrefillDecode>(); }

}  // namespace xenithium:bench
