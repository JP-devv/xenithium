// Scenario "deep-decode": the decode speed with a long conversation behind it. A decode step attends to the whole
// context, so tok/s falls with depth; this is the number a user of a long document or a long agent session sees. The
// document is sent once (a new prompt: the cold TTFT), then again with the same text, which a prefix cache answers
// without reading it again (the hit's TTFT); every request decodes at the depth.
#include "scenario.hpp"

namespace xenithium::bench {
namespace {

constexpr ParamSpec kParams[] = {
    {"depths", "32000,109000,250000", "context lengths in tokens (those past the endpoint's context are skipped)"},
    {"kind", "code", "corpus kind of the document"},
    {"repeats", "2", "requests on the same document after the first"},
    {"reply_tokens", "400", "tokens generated per request"},
};
constexpr std::string_view kQuestion = "\n\nExplain what the text above contains, part by part, in detail.";

class DeepDecode final : public Scenario {
 public:
  std::string_view id() const override { return "deep-decode"; }
  std::string_view title() const override { return "Decode at depth (the document sent once, then prefix-cache hits)"; }
  std::string_view about() const override {
    return "Decode tok/s with 32K to 250K tokens of context, the cold TTFT of the document and the TTFT when the same document is sent again.";
  }
  std::span<const ParamSpec> params() const override { return kParams; }

  ScenarioResult run(Session &session, const Params &params) const override {
    ScenarioResult result;
    const int reply_tokens = static_cast<int>(params.integer("reply_tokens"));
    const std::string kind = params.text("kind");
    std::string skipped;
    const std::vector<int64_t> asked = params.integers("depths");
    for (const int64_t depth : depths_that_fit(session, asked, reply_tokens, skipped)) {
      const Session::DocumentPrompt prompt = session.document_prompt(kind, depth, kQuestion, /*unique=*/true);
      Row row = Row::with_labels({{"depth", with_thousands(depth)}});
      if (prompt.wrapped) row.notes.push_back("the " + kind + " corpus is shorter than this depth: its files repeat");
      for (int64_t i = 0; i <= params.integer("repeats"); ++i) {
        const Sample sample = session.chat_alone(session.sampled(prompt.text, reply_tokens));
        if (!tally(result, sample)) continue;
        const bool first = i == 0;
        row.metric(std::string(metric::kPromptTokens), "tokens", Better::none).samples.push_back(sample.prompt_tokens);
        row.metric(first ? "cold TTFT" : "hit TTFT", "ms", Better::lower).samples.push_back(sample.ttft_ms);
        if (sample.decode_tps() > 0) row.metric(std::string(metric::kDecode), "tok/s", Better::higher).samples.push_back(sample.decode_tps());
        row.metric(std::string(metric::kTokensPerUpdate), "", Better::higher).samples.push_back(sample.tokens_per_update());
        if (sample.update_gap_ms() > 0) row.metric(std::string(metric::kUpdateGap), "ms", Better::lower).samples.push_back(sample.update_gap_ms());
        if (!first && sample.cached_tokens >= 0) row.metric("cached", "tokens", Better::higher).samples.push_back(sample.cached_tokens);
      }
      session.progress("deep-decode: " + with_thousands(depth) + " tokens done");
      result.rows.push_back(std::move(row));
    }
    if (!skipped.empty()) result.rows.push_back(Row::note_only(skipped));
    result.charts = {
        {ChartKind::line, "Decode rate by context depth", "depth", {std::string(metric::kDecode)}, true},
        {ChartKind::bars, "Time to first token: new document against the repeat", "depth", {"cold TTFT", "hit TTFT"}, false},
    };
    return result;
  }
};

}  // namespace

std::unique_ptr<Scenario> make_deep_decode() { return std::make_unique<DeepDecode>(); }

}  // namespace xenithium:bench
