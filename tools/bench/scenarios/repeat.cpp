// Scenario "repeat": does the same request give the same reply? Greedy requests should; sampled requests with a seed
// should too, if the server's sampling is a function of the seed and the position. A long document sent twice shows
// whether a reply continued from the prefix cache equals the one computed from the prompt. Differences are not
// errors (batching, caches and kernels may round differently), but anyone comparing outputs between runs must know.
#include <algorithm>

#include "scenario.hpp"

namespace xenithium::bench {
namespace {

constexpr ParamSpec kParams[] = {
    {"repeats", "3", "times each request is sent"},
    {"reply_tokens", "200", "tokens generated per request"},
    {"cache_depth", "16000", "document length in tokens for the prefix-cache case (0 = leave it out)"},
    {"kind", "code", "corpus kind of that document"},
};
constexpr std::string_view kQuestion = "\n\nExplain what the text above contains, part by part.";

std::string reply_of(const Sample &s) { return s.reasoning + s.text; }

// Characters until the first difference; a.size when equal.
size_t common_prefix(const std::string &a, const std::string &b) {
  return static_cast<size_t>(std::ranges::mismatch(a, b).in1 - a.begin());
}

class Repeat final : public Scenario {
 public:
  std::string_view id() const override { return "repeat"; }
  std::string_view title() const override { return "The same request again (identical = 1)"; }
  std::string_view about() const override {
    return "One request sent three times, greedy and sampled with a seed, and a 16K-token document computed once and then continued from the prefix cache: are the replies identical?";
  }
  std::span<const ParamSpec> params() const override { return kParams; }

  ScenarioResult run(Session &session, const Params &params) const override {
    ScenarioResult result;
    const int reply_tokens = static_cast<int>(params.integer("reply_tokens"));
    const ShortPrompt &prompt = short_prompts()[3];  // a code prompt: long enough to diverge if anything does

    // `request` sent `repeats` times; every later reply against the first
    auto compare = [&](const ChatRequest &request, std::string label) {
      Row row = Row::with_labels({{"case", std::move(label)}});
      std::string first;
      for (int64_t i = 0; i < params.integer("repeats"); ++i) {
        const Sample sample = session.client().chat(request);
        if (!tally(result, sample)) continue;
        if (first.empty()) {
          first = reply_of(sample);
          continue;
        }
        const std::string again = reply_of(sample);
        row.flag("identical").samples.push_back(again == first ? 1 : 0);
        if (again != first)
          row.notes.push_back(row.labels.front().second + ": a repeat differs after " + std::to_string(common_prefix(first, again)) + " of " + std::to_string(first.size()) + " characters");
      }
      result.rows.push_back(std::move(row));
    };

    compare(session.greedy(std::string(prompt.text), reply_tokens), "greedy");
    compare(session.sampled(std::string(prompt.text), reply_tokens), "sampled, one seed");
    if (int64_t depth = params.integer("cache_depth"); depth > 0) {
      std::string skipped;
      if (!depths_that_fit(session, std::span(&depth, 1), reply_tokens, skipped).empty())
        compare(session.greedy(session.document_prompt(params.text("kind"), depth, kQuestion, /*unique=*/true).text, reply_tokens),
                "greedy, " + with_thousands(depth) + " tokens (then from the cache)");
    }
    session.progress("repeat: done");
    result.charts = {{ChartKind::dots, "Repeats identical to the first reply", "case", {"identical"}, false}};
    return result;
  }
};

}  // namespace

std::unique_ptr<Scenario> make_repeat() { return std::make_unique<Repeat>(); }

}  // namespace xenithium:bench
