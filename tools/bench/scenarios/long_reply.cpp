// Scenario "long-reply": one long answer - a whole source file, a report - as a user asks for one and waits for it.
// The reply's own tokens become the context it attends to, so the decode step can slow down as the reply grows, and a
// server's scheduling or memory housekeeping shows as stalls in the stream. Reported per stretch of the reply: the
// decode step (the median time between streamed updates), the decode rate and the longest pause.
#include <algorithm>
#include <numeric>

#include "scenario.hpp"
#include "stats.hpp"

namespace xenithium::bench {
namespace {

constexpr ParamSpec kParams[] = {
    {"reply_tokens", "8192", "tokens of the reply (with --reply-length natural, the most it may take)"},
    {"stretches", "8", "stretches of equal length the reply is cut into"},
    {"passes", "1", "replies asked for (each with its own seed)"},
};
constexpr std::string_view kPrompt =
    "Write a complete, production-quality C++20 library for JSON in one file: a value type, a parser that reports the line and column of an error, a "
    "serializer with pretty printing, and unit tests for all of it. Explain every part in comments. Write all of it; do not stop early or leave parts out.";

class LongReply final : public Scenario {
 public:
  std::string_view id() const override { return "long-reply"; }
  std::string_view title() const override { return "One long reply, stretch by stretch (sampled)"; }
  std::string_view about() const override {
    return "A reply of 8K tokens (a whole source file) cut into eight stretches: the decode step, the decode rate and the longest pause in each, so a slowdown "
           "as the reply's own context grows, or stalls, show where they happen.";
  }
  std::span<const ParamSpec> params() const override { return kParams; }

  ScenarioResult run(Session &session, const Params &params) const override {
    ScenarioResult result;
    const int reply_tokens = static_cast<int>(params.integer("reply_tokens"));
    const size_t stretches = static_cast<size_t>(std::max<int64_t>(1, params.integer("stretches")));
    const double stretch_tokens = static_cast<double>(reply_tokens) / static_cast<double>(stretches);
    std::vector<Row> by_stretch(stretches);
    for (size_t s = 0; s < stretches; ++s) by_stretch[s].labels = {{"from token", with_thousands(static_cast<int64_t>(static_cast<double>(s) * stretch_tokens))}};
    Row whole = Row::with_labels({{"from token", "all"}});
    const PowerSpan power(session);
    for (int64_t pass = 0; pass < params.integer("passes"); ++pass) {
      ChatRequest request = session.sampled(std::string(kPrompt), reply_tokens);
      request.thinking = Thinking::off;  // the reply is the answer, not a reasoning pass before it
      const Sample sample = session.client().chat(request);
      if (!tally(result, sample)) continue;
      add_decode_metrics(whole, sample);
      whole.metric("tokens", "", Better::none).samples.push_back(sample.completion_tokens);
      add_stretches(sample, stretch_tokens, by_stretch);
      session.progress("long-reply: " + with_thousands(sample.completion_tokens) + " tokens in " + format_number(sample.wall_ms / 1e3) + " s");
    }
    power.add_to(whole, result.completion_tokens);
    result.rows.push_back(std::move(whole));
    for (Row &row : by_stretch)
      if (!row.metrics.empty()) result.rows.push_back(std::move(row));
    result.charts = {
        {ChartKind::line, "Decode rate along the reply", "from token", {std::string(metric::kDecode)}, false},
        {ChartKind::line, "Decode step and the longest pause along the reply", "from token", {std::string(metric::kUpdateGap), "longest gap"}, false},
    };
    return result;
  }

 private:
  // Every gap between two updates goes to the stretch its update falls in, by the tokens streamed before it (tokens
  // per update taken as the reply's mean: the stream does not say how many tokens each update carried).
  static void add_stretches(const Sample &sample, double stretch_tokens, std::vector<Row> &by_stretch) {
    const double per_update = sample.tokens_per_update();
    std::vector<std::vector<double>> gaps(by_stretch.size());
    for (size_t i = 1; i < sample.update_ms.size(); ++i) {
      const size_t s = std::min(by_stretch.size() - 1, static_cast<size_t>(static_cast<double>(i) * per_update / stretch_tokens));
      gaps[s].push_back(sample.update_ms[i] - sample.update_ms[i - 1]);
    }
    for (size_t s = 0; s < gaps.size(); ++s) {
      if (gaps[s].empty()) continue;  // past the end of a reply that ended early
      const double span_ms = std::accumulate(gaps[s].begin(), gaps[s].end(), 0.0);
      Row &row = by_stretch[s];
      row.metric(std::string(metric::kUpdateGap), "ms", Better::lower).samples.push_back(summarize(gaps[s]).median);
      row.metric("longest gap", "ms", Better::lower).samples.push_back(std::ranges::max(gaps[s]));
      if (span_ms > 0) row.metric(std::string(metric::kDecode), "tok/s", Better::higher).samples.push_back(static_cast<double>(gaps[s].size()) * per_update / (span_ms / 1e3));
    }
  }
};

}  // namespace

std::unique_ptr<Scenario> make_long_reply() { return std::make_unique<LongReply>(); }

}  // namespace xenithium:bench
