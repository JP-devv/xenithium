// Scenario "autocomplete": code completion in an editor. A developer types in a file; at a pause the editor asks for a
// completion of the code at the cursor - a few dozen tokens, greedy - and shows it. While the developer keeps typing,
// the request in flight is abandoned (its connection closed) and the next one sent with a slightly longer file, so
// nearly all of every prompt was seen a moment before: the server's prefix cache, and how fast it lets go of an
// abandoned request, decide the latency. What the developer feels is the whole completion's latency, not a token rate.
#include <algorithm>
#include <thread>

#include "scenario.hpp"

namespace xenithium::bench {
namespace {

constexpr ParamSpec kParams[] = {
    {"developers", "1,4", "developers typing at once, each in a file of their own"},
    {"completions", "8", "completions every developer waits for"},
    {"abandoned", "2", "requests overtaken by typing (abandoned) before each completion"},
    {"abandon_ms", "150", "how long an overtaken request runs before the next keystroke abandons it, in ms"},
    {"prefix_tokens", "2000", "the file up to the cursor when the developer starts, in tokens"},
    {"step_chars", "48", "characters typed between two requests"},
    {"reply_tokens", "48", "the most tokens a completion may take"},
};
constexpr std::string_view kInstructions =
    "You are a code completion engine. Continue the code exactly where it stops. Reply with the continuation only: no explanation, no code fence, no repetition "
    "of the code before the cursor.";

// A developer's typing: the file and where the cursor is.
struct Editor {
  std::string file;
  size_t cursor = 0;
  std::vector<Sample> completions, overtaken;
};

class Autocomplete final : public Scenario {
 public:
  std::string_view id() const override { return "autocomplete"; }
  std::string_view title() const override { return "Editor autocomplete (greedy, overtaken requests abandoned)"; }
  std::string_view about() const override {
    return "1 and 4 developers typing in 2,000-token files: every completion is preceded by requests the typing overtakes and abandons, each prompt a few "
           "characters longer than the last. The completion's whole latency and TTFT, and the prompt tokens the prefix cache served.";
  }
  std::span<const ParamSpec> params() const override { return kParams; }

  ScenarioResult run(Session &session, const Params &params) const override {
    ScenarioResult result;
    constexpr std::string_view kKind = "code";
    const double chars_per_token = session.chars_per_token(kKind);  // calibrated before anything is timed
    const int64_t completions = params.integer("completions"), abandoned = params.integer("abandoned"), step_chars = params.integer("step_chars");
    const size_t prefix_chars = static_cast<size_t>(static_cast<double>(params.integer("prefix_tokens")) * chars_per_token);
    const size_t typed_chars = static_cast<size_t>(completions * (abandoned + 1) * step_chars);
    size_t first_file = 0;
    for (const int64_t count : params.integers("developers")) {
      std::vector<Editor> editors(static_cast<size_t>(count));
      for (Editor &editor : editors) editor.file = session.corpus().document(kKind, prefix_chars + typed_chars, first_file++).text, editor.cursor = prefix_chars;
      const PowerSpan power(session);
      {
        std::vector<std::jthread> threads;
        for (Editor &editor : editors) threads.emplace_back([&session, &params, &editor, alone = count == 1] { type(session, params, editor, alone); });
      }  // joins every developer

      Row row = Row::with_labels({{"developers", std::to_string(count)}});
      int64_t tokens = 0;
      int closed_early = 0;
      for (const Editor &editor : editors) {
        for (const Sample &sample : editor.overtaken)
          if (tally(result, sample)) closed_early += sample.abandoned, tokens += sample.completion_tokens;
        for (const Sample &sample : editor.completions) {
          if (!tally(result, sample)) continue;
          tokens += sample.completion_tokens;
          row.metric("latency", "ms", Better::lower).samples.push_back(sample.wall_ms);
          row.metric(std::string(metric::kTtft), "ms", Better::lower).samples.push_back(sample.ttft_ms);
          row.metric(std::string(metric::kPromptTokens), "tokens", Better::none).samples.push_back(sample.prompt_tokens);
          if (sample.cached_tokens >= 0) row.metric("cached", "tokens", Better::higher).samples.push_back(sample.cached_tokens);
        }
      }
      row.metric("abandoned", "", Better::none).samples.push_back(closed_early);
      power.add_to(row, tokens);
      if (count == 1 && result.rows.empty())
        row.notes.push_back("cached: prompt tokens the prefix cache served, from the usage report, or with one developer from the growth of the server's counter over the request");
      session.progress("autocomplete: " + std::to_string(count) + " developer(s) done");
      result.rows.push_back(std::move(row));
    }
    result.charts = {
        {ChartKind::range, "A completion's whole latency by developers", "developers", {"latency"}, false},
        {ChartKind::range, "Time to first token of a completion by developers", "developers", {std::string(metric::kTtft)}, false},
    };
    return result;
  }

 private:
  // The editor's request for the file up to the cursor.
  static ChatRequest completion_request(const Session &session, const Params &params, const Editor &editor) {
    ChatRequest request = session.greedy({}, static_cast<int>(params.integer("reply_tokens")));
    request.messages = {{"system", std::string(kInstructions)}, {"user", editor.file.substr(0, editor.cursor)}};
    request.ignore_eos = false;  // a completion ends where the model ends it
    request.thinking = Thinking::off;
    return request;
  }

  // One developer: before each completion, requests the typing overtakes; then the completion, read to its end. Alone,
  // a completion is sent by itself so the prefix-cache counter tells how much of its prompt was cached.
  static void type(const Session &session, const Params &params, Editor &editor, bool alone) {
    const ReadLimits overtaken{.abandon_after_ms = params.real("abandon_ms")};
    const auto step = static_cast<size_t>(params.integer("step_chars"));
    for (int64_t c = 0; c < params.integer("completions"); ++c) {
      for (int64_t a = 0; a < params.integer("abandoned"); ++a) {
        editor.cursor = std::min(editor.cursor + step, editor.file.size());
        editor.overtaken.push_back(session.client().chat(completion_request(session, params, editor), overtaken));
      }
      editor.cursor = std::min(editor.cursor + step, editor.file.size());
      const ChatRequest request = completion_request(session, params, editor);
      editor.completions.push_back(alone ? session.chat_alone(request) : session.client().chat(request));
    }
  }
};

}  // namespace

std::unique_ptr<Scenario> make_autocomplete() { return std::make_unique<Autocomplete>(); }

}  // namespace xenithium:bench
