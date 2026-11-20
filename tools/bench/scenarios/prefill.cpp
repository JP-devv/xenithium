// Scenario "prefill": how fast the endpoint reads a prompt, by prompt length. Every prompt starts with a line no
// earlier request had, so a prefix cache cannot answer for it; the reply is a few tokens, so the time to the first
// token is the prompt's processing. Long prompts are where engines differ most: the attention grows with the depth.
#include "scenario.hpp"

namespace xenithium::bench {
namespace {

constexpr ParamSpec kParams[] = {
    {"depths", "2000,8000,32000,64000,128000,250000", "prompt lengths in tokens (those past the endpoint's context are skipped)"},
    {"kinds", "code", "corpus kinds to read (code, prose)"},
    {"runs", "2", "prompts per depth (each starts at another file of the corpus)"},
    {"reply_tokens", "16", "tokens generated after the prompt"},
};
constexpr std::string_view kQuestion = "\n\nIn one sentence, what is the text above?";

class Prefill final : public Scenario {
 public:
  std::string_view id() const override { return "prefill"; }
  std::string_view title() const override { return "Prompt processing by depth (prefix cache defeated)"; }
  std::string_view about() const override {
    return "New prompts of 2K to 250K tokens of real text: time to the first token and prompt tokens per second at each depth.";
  }
  std::span<const ParamSpec> params() const override { return kParams; }

  ScenarioResult run(Session &session, const Params &params) const override {
    ScenarioResult result;
    const int reply_tokens = static_cast<int>(params.integer("reply_tokens"));
    std::string skipped;
    const std::vector<int64_t> asked = params.integers("depths");
    const std::vector<int64_t> depths = depths_that_fit(session, asked, reply_tokens, skipped);
    for (const std::string &kind : params.list("kinds")) {
      (void)session.chars_per_token(kind);  // calibrate before the first row's power is measured
      for (const int64_t depth : depths) {
        Row row = Row::with_labels({{"kind", kind}, {"depth", with_thousands(depth)}});
        const PowerSpan power(session);
        int64_t prompt_tokens = 0;
        for (int64_t run = 0; run < params.integer("runs"); ++run) {
          const Session::DocumentPrompt prompt = session.document_prompt(kind, depth, kQuestion, /*unique=*/true, static_cast<size_t>(run));
          if (prompt.wrapped && row.notes.empty()) row.notes.push_back("the " + kind + " corpus is shorter than " + with_thousands(depth) + " tokens: its files repeat");
          const Sample sample = session.client().chat(session.greedy(prompt.text, reply_tokens));
          if (!tally(result, sample)) continue;
          add_prefill_metrics(row, sample);
          prompt_tokens += sample.prompt_tokens;
        }
        power.add_to(row, prompt_tokens);
        session.progress("prefill: " + kind + " at " + with_thousands(depth) + " tokens done");
        result.rows.push_back(std::move(row));
      }
    }
    if (!skipped.empty() && !result.rows.empty()) result.rows.back().notes.push_back(skipped);
    if (result.rows.empty() && !skipped.empty()) result.rows.push_back(Row::note_only(skipped));
    result.charts = {
        {ChartKind::line, "Prompt tokens per second by depth", "depth", {std::string(metric::kPrefill)}, true},
        {ChartKind::line, "Time to first token by depth", "depth", {std::string(metric::kTtft)}, true},
    };
    return result;
  }
};

}  // namespace

std::unique_ptr<Scenario> make_prefill() { return std::make_unique<Prefill>(); }

}  // namespace xenithium:bench
