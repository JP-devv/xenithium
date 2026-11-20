// Scenario "turns": an agent session. The client re-sends the whole conversation every turn (a document, then tool
// results and questions, each answered), so the prompt grows and all but its tail was seen before. With a prefix cache
// the TTFT of turn 2+ is the tail's processing; without one it is the whole conversation's, every turn.
#include "scenario.hpp"

namespace xenithium::bench {
namespace {

constexpr ParamSpec kParams[] = {
    {"start_tokens", "16000", "the first turn's document, in tokens"},
    {"turns", "6", "turns of the session"},
    {"turn_tokens", "800", "new text the user adds per later turn (a tool result), in tokens"},
    {"reply_tokens", "200", "tokens generated per turn"},
    {"kind", "code", "corpus kind"},
};
constexpr std::string_view kFirstQuestion = "\n\nRead the files above. Which of them would you look at first to understand the program, and why?";
constexpr std::string_view kLaterQuestion = "\n\nHere is one more file. How does it relate to what you have seen so far?";
constexpr std::string_view kNoAnswer = "(no answer)";  // a turn whose reply was all reasoning still needs an assistant message

class Turns final : public Scenario {
 public:
  std::string_view id() const override { return "turns"; }
  std::string_view title() const override { return "Agent session (the conversation re-sent every turn)"; }
  std::string_view about() const override {
    return "A growing conversation over a 16K-token document: TTFT and prompt tokens per turn - the prefix cache's case.";
  }
  std::span<const ParamSpec> params() const override { return kParams; }

  ScenarioResult run(Session &session, const Params &params) const override {
    ScenarioResult result;
    const int reply_tokens = static_cast<int>(params.integer("reply_tokens"));
    const std::string kind = params.text("kind");
    ChatRequest request = session.sampled(session.document_prompt(kind, params.integer("start_tokens"), kFirstQuestion, /*unique=*/true).text, reply_tokens);
    request.thinking = Thinking::off;  // the history holds answers, not reasoning
    Row later = Row::with_labels({{"turn", "2+"}});
    for (int64_t turn = 1; turn <= params.integer("turns"); ++turn) {
      const Sample sample = session.chat_alone(request);
      if (!tally(result, sample)) break;  // the session cannot go on without this turn's answer
      Row row = Row::with_labels({{"turn", std::to_string(turn)}});
      add_prefill_metrics(row, sample);
      if (sample.cached_tokens >= 0) row.metric("cached", "tokens", Better::higher).samples.push_back(sample.cached_tokens);
      if (turn > 1) later.metric(std::string(metric::kTtft), "ms", Better::lower).samples.push_back(sample.ttft_ms);
      result.rows.push_back(std::move(row));
      request.messages.push_back({"assistant", sample.text.empty() ? std::string(kNoAnswer) : sample.text});
      // the next turn's new text: the corpus from another file on, without a unique line (the history must repeat)
      request.messages.push_back({"user", session.document_prompt(kind, params.integer("turn_tokens"), kLaterQuestion, /*unique=*/false, static_cast<size_t>(turn) + 1).text});
    }
    if (!later.metrics.empty()) result.rows.push_back(std::move(later));
    session.progress("turns: " + std::to_string(result.rows.size()) + " rows");
    result.charts = {
        {ChartKind::bars, "Prompt tokens per turn: sent and served from the cache", "turn", {std::string(metric::kPromptTokens), "cached"}, false},
        {ChartKind::bars, "Time to first token per turn", "turn", {std::string(metric::kTtft)}, false},
    };
    return result;
  }
};

}  // namespace

std::unique_ptr<Scenario> make_turns() { return std::make_unique<Turns>(); }

}  // namespace xenithium:bench
