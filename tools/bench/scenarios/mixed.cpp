// Scenario "mixed": one stream deep in a long document generates beside short streams. Engines that batch the
// decode step make the short streams pay for the long one's attention (or the other way round); the two classes are
// reported apart. The document is processed before the timed round, so the round is decode only.
#include "scenario.hpp"

namespace xenithium::bench {
namespace {

constexpr ParamSpec kParams[] = {
    {"depth", "64000", "the long stream's document, in tokens"},
    {"short_streams", "3", "short streams beside it"},
    {"reply_tokens", "400", "tokens every stream generates"},
    {"kind", "code", "corpus kind of the document"},
};
constexpr std::string_view kQuestion = "\n\nExplain what the text above contains, part by part, in detail.";

class Mixed final : public Scenario {
 public:
  std::string_view id() const override { return "mixed"; }
  std::string_view title() const override { return "A long-context stream beside short streams"; }
  std::string_view about() const override {
    return "One stream at 64K tokens of context generating with three short streams: decode tok/s of each class in the same steps.";
  }
  std::span<const ParamSpec> params() const override { return kParams; }

  ScenarioResult run(Session &session, const Params &params) const override {
    ScenarioResult result;
    const int reply_tokens = static_cast<int>(params.integer("reply_tokens"));
    const int64_t depth = params.integer("depth"), short_streams = params.integer("short_streams");
    std::string skipped;
    if (depths_that_fit(session, std::span(&depth, 1), reply_tokens, skipped).empty()) {
      result.rows.push_back(Row::note_only(skipped));
      return result;
    }
    const std::string document = session.document_prompt(params.text("kind"), depth, kQuestion, /*unique=*/true).text;
    if (!tally(result, session.client().chat(session.greedy(document, 1)))) return result;  // read once; the round then hits the cache

    const auto prompts = short_prompts();
    std::vector<ChatRequest> requests = {session.sampled(document, reply_tokens)};
    for (int64_t i = 0; i < short_streams; ++i) requests.push_back(session.sampled(std::string(prompts[static_cast<size_t>(i) % prompts.size()].text), reply_tokens));
    const std::vector<Sample> samples = session.run_together(requests);

    Row long_row = Row::with_labels({{"stream", "long (" + with_thousands(depth) + ")"}});
    Row short_row = Row::with_labels({{"stream", "short x " + std::to_string(short_streams)}});
    for (size_t i = 0; i < samples.size(); ++i)
      if (tally(result, samples[i])) add_decode_metrics(i == 0 ? long_row : short_row, samples[i]);
    result.rows.push_back(std::move(long_row));
    result.rows.push_back(std::move(short_row));
    session.progress("mixed: done");
    result.charts = {{ChartKind::bars, "Decode rate of each stream class in the same steps", "stream", {std::string(metric::kDecode)}, false}};
    return result;
  }
};

}  // namespace

std::unique_ptr<Scenario> make_mixed() { return std::make_unique<Mixed>(); }

}  // namespace xenithium:bench
