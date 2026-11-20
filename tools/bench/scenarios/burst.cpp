// Scenario "burst": several requests arrive together on ONE long document, each with its own question (parallel
// sub-agents, batch questions on a file). A server that sees their common prefix reads the document once; one that
// does not reads it once per request. Compared with a single request on a document of the same length.
#include <algorithm>

#include "scenario.hpp"

namespace xenithium::bench {
namespace {

constexpr ParamSpec kParams[] = {
    {"depth", "32000", "the shared document, in tokens"},
    {"requests", "4", "requests sent at once"},
    {"reply_tokens", "64", "tokens generated per request"},
    {"kind", "code", "corpus kind"},
};
constexpr std::string_view kQuestions[] = {
    "\n\nWhich file above is the largest, and what does it do?",
    "\n\nName three functions defined above and say what each is for.",
    "\n\nWhat would you test first in the code above?",
    "\n\nSummarise the text above in five bullet points.",
};

double all_done_s(const std::vector<Sample> &samples) {
  double last_ms = 0;
  for (const Sample &s : samples) last_ms = std::max(last_ms, s.wall_ms);
  return last_ms / 1e3;
}

class Burst final : public Scenario {
 public:
  std::string_view id() const override { return "burst"; }
  std::string_view title() const override { return "Burst on one shared document"; }
  std::string_view about() const override {
    return "N requests at once on the same 32K-token document with different questions: time until all are answered, against one request alone.";
  }
  std::span<const ParamSpec> params() const override { return kParams; }

  ScenarioResult run(Session &session, const Params &params) const override {
    ScenarioResult result;
    const int reply_tokens = static_cast<int>(params.integer("reply_tokens"));
    const std::string kind = params.text("kind");
    const int64_t depth = params.integer("depth"), count = params.integer("requests");
    std::string skipped;
    if (depths_that_fit(session, std::span(&depth, 1), reply_tokens, skipped).empty()) {
      result.rows.push_back(Row::note_only(skipped));
      return result;
    }
    // each round's document is new to the server: its own unique first line, shared by the round's requests
    auto round = [&](int64_t n, std::string label) {
      const std::string document = session.document_prompt(kind, depth, "", /*unique=*/true).text;
      std::vector<ChatRequest> requests;
      for (int64_t i = 0; i < n; ++i) requests.push_back(session.greedy(document + std::string(kQuestions[static_cast<size_t>(i) % std::size(kQuestions)]), reply_tokens));
      const std::vector<Sample> samples = session.run_together(requests);
      Row row = Row::with_labels({{"requests", std::move(label)}});
      bool all_ok = true;
      for (const Sample &sample : samples) {
        all_ok &= tally(result, sample);
        if (sample.ok) row.metric(std::string(metric::kTtft), "ms", Better::lower).samples.push_back(sample.ttft_ms);
      }
      if (all_ok) row.metric("all done", "s", Better::lower).samples.push_back(all_done_s(samples));
      result.rows.push_back(std::move(row));
    };
    round(1, "1 alone");
    round(count, std::to_string(count) + " at once");
    session.progress("burst: done");
    result.charts = {
        {ChartKind::bars, "Time until every request is answered", "requests", {"all done"}, false},
        {ChartKind::range, "Time to first token", "requests", {std::string(metric::kTtft)}, false},
    };
    return result;
  }
};

}  // namespace

std::unique_ptr<Scenario> make_burst() { return std::make_unique<Burst>(); }

}  // namespace xenithium:bench
