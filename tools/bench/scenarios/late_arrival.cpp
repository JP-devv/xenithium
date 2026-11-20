// Scenario "late-arrival": a long prompt arrives while N streams are generating. The server has to read it between
// their decode steps: how it shares the time shows in the newcomer's TTFT and, for the streams already running, in
// the longest pause between two updates and in their token rate while the prompt is read.
#include <algorithm>

#include "scenario.hpp"

namespace xenithium::bench {
namespace {

constexpr ParamSpec kParams[] = {
    {"streams", "4,8", "requests already generating when the long prompt arrives"},
    {"depth", "32000", "the late prompt, in tokens"},
    {"after_s", "1.5", "seconds after the streams start at which it is sent"},
    {"stream_tokens", "2000", "tokens each stream generates (long enough to outlast the late prompt's processing)"},
    {"reply_tokens", "16", "tokens the late request generates"},
    {"kind", "code", "corpus kind of the late prompt"},
};
constexpr std::string_view kQuestion = "\n\nIn one sentence, what is the text above?";

class LateArrival final : public Scenario {
 public:
  std::string_view id() const override { return "late-arrival"; }
  std::string_view title() const override { return "A long prompt arriving beside running streams"; }
  std::string_view about() const override {
    return "A 32K-token prompt sent while 4 or 8 streams generate: its TTFT, and the running streams' longest pause and token rate meanwhile.";
  }
  std::span<const ParamSpec> params() const override { return kParams; }

  ScenarioResult run(Session &session, const Params &params) const override {
    ScenarioResult result;
    const int reply_tokens = static_cast<int>(params.integer("reply_tokens")), stream_tokens = static_cast<int>(params.integer("stream_tokens"));
    const int64_t depth = params.integer("depth");
    const std::string kind = params.text("kind");
    std::string skipped;
    if (depths_that_fit(session, std::span(&depth, 1), reply_tokens, skipped).empty()) {
      result.rows.push_back(Row::note_only(skipped));
      return result;
    }
    (void)session.chars_per_token(kind);  // calibrate before the timed round
    const auto prompts = short_prompts();
    for (const int64_t streams : params.integers("streams")) {
      std::vector<ChatRequest> requests;
      for (int64_t i = 0; i < streams; ++i) requests.push_back(session.sampled(std::string(prompts[static_cast<size_t>(i) % prompts.size()].text), stream_tokens));
      requests.push_back(session.greedy(session.document_prompt(kind, depth, kQuestion, /*unique=*/true).text, reply_tokens));
      std::vector<double> start_after_s(requests.size(), 0.0);
      start_after_s.back() = params.real("after_s");
      const std::vector<Sample> samples = session.run_together(requests, start_after_s);

      Row row = Row::with_labels({{"streams", std::to_string(streams)}});
      const Sample &late = samples.back();
      bool usable = tally(result, late);
      for (int64_t i = 0; i < streams; ++i) usable &= tally(result, samples[static_cast<size_t>(i)]);
      if (usable) {
        const Clock::time_point from = late.submitted, to = first_token_time(late);
        row.metric("late TTFT", "ms", Better::lower).samples.push_back(late.ttft_ms);
        row.metric(std::string(metric::kPromptTokens), "tokens", Better::none).samples.push_back(late.prompt_tokens);
        int ended_early = 0;
        for (int64_t i = 0; i < streams; ++i) {
          const Sample &stream = samples[static_cast<size_t>(i)];
          const StreamInWindow w = split_at(stream, from, to);
          ended_early += !w.outlasted;
          row.metric("longest gap", "ms", Better::lower).samples.push_back(w.longest_gap_ms);
          if (w.before_s > 0) row.metric("rate before", "tok/s", Better::higher).samples.push_back(w.updates_before * stream.tokens_per_update() / w.before_s);
          row.metric("rate meanwhile", "tok/s", Better::higher).samples.push_back(w.updates_inside * stream.tokens_per_update() / (late.ttft_ms / 1e3));
        }
        if (ended_early > 0)
          row.notes.push_back(std::to_string(ended_early) + " of the streams finished before the late prompt's first token: raise stream_tokens for this endpoint");
      }
      session.progress("late-arrival: beside " + std::to_string(streams) + " streams done");
      result.rows.push_back(std::move(row));
    }
    result.charts = {
        {ChartKind::bars, "The late prompt's time to first token", "streams", {"late TTFT"}, false},
        {ChartKind::range, "Running streams: longest pause while it is read", "streams", {"longest gap"}, false},
        {ChartKind::bars, "Running streams: token rate before and meanwhile", "streams", {"rate before", "rate meanwhile"}, false},
    };
    return result;
  }
};

}  // namespace

std::unique_ptr<Scenario> make_late_arrival() { return std::make_unique<LateArrival>(); }

}  // namespace xenithium:bench
