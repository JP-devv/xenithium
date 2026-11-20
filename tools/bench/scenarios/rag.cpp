// Scenario "rag": answers from retrieved passages, as a documentation assistant or a support bot gives them. Every
// request carries the same instructions (a system prompt a prefix cache can keep), then passages retrieved for the
// question - other ones every time - and the question; the answer is short. Nearly all the work is reading prompts,
// many users ask at once, and the time to the first token is what they wait for. One passage of each prompt holds the
// fact the question asks for, so the answers are checked too: a server that loses or garbles context under load
// answers wrong.
#include <algorithm>
#include <array>
#include <random>

#include "scenario.hpp"

namespace xenithium::bench {
namespace {

constexpr ParamSpec kParams[] = {
    {"concurrency", "1,4,8", "requests in flight at once (closed loop)"},
    {"requests", "16", "requests per level (at least two per stream are sent)"},
    {"passages", "8", "retrieved passages per prompt"},
    {"passage_tokens", "800", "tokens per passage"},
    {"reply_tokens", "160", "the most tokens an answer may take"},
    {"kind", "prose", "corpus kind the passages are cut from"},
};
constexpr int64_t kMinRequestsPerStream = 2;

constexpr std::string_view kInstructions =
    "You are the documentation assistant of an infrastructure team. Answer the user's question using only the numbered passages in the user's message. "
    "Quote the facts you rely on, name the passage numbers you used in square brackets, and say plainly when the passages do not contain the answer. "
    "Keep the answer to two or three sentences, and do not add information that is not in the passages.";

// The planted fact: a maintenance window of a named cluster. Drawn per request, so every question has its own answer.
constexpr std::array<std::string_view, 8> kClusters = {"aurora", "basalt", "cedar", "delta", "ember", "fjord", "granite", "harbor"};
constexpr std::array<std::string_view, 7> kWeekdays = {"Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"};
constexpr std::array<std::string_view, 6> kTimes = {"01:30", "02:15", "03:45", "04:00", "22:30", "23:10"};

struct Fact {
  std::string sentence, question;
  std::string_view weekday, time;
};

Fact draw_fact(std::mt19937_64 &rng) {
  const std::string_view cluster = kClusters[rng() % kClusters.size()], weekday = kWeekdays[rng() % kWeekdays.size()], time = kTimes[rng() % kTimes.size()];
  const std::string name = std::string(cluster) + "-" + std::to_string(rng() % 90 + 10);
  return {"Operations note: the maintenance window of the cluster " + name + " is every " + std::string(weekday) + " at " + std::string(time) + " UTC.",
          "When is the maintenance window of the cluster " + name + "?", weekday, time};
}

// `passage` with `sentence` put in after the line nearest its middle.
std::string with_sentence(std::string passage, const std::string &sentence) {
  const size_t line_end = passage.find('\n', passage.size() / 2);
  passage.insert(line_end == std::string::npos ? passage.size() : line_end + 1, sentence + "\n");
  return passage;
}

struct Question {
  ChatRequest request;
  Fact fact;
};

class Rag final : public Scenario {
 public:
  std::string_view id() const override { return "rag"; }
  std::string_view title() const override { return "Answers from retrieved passages (RAG) under load"; }
  std::string_view about() const override {
    return "1, 4 and 8 users at once asking questions over eight retrieved passages (about 6,400 tokens, other ones every time) under shared instructions: "
           "TTFT and request latency, the prompt tokens read per second by all requests together, and whether each answer holds the fact one passage planted.";
  }
  std::span<const ParamSpec> params() const override { return kParams; }

  ScenarioResult run(Session &session, const Params &params) const override {
    ScenarioResult result;
    const std::string kind = params.text("kind");
    const int64_t passages = params.integer("passages");
    const size_t passage_chars = static_cast<size_t>(static_cast<double>(params.integer("passage_tokens")) * session.chars_per_token(kind));  // calibrated before timing
    std::mt19937_64 rng(session.seed());
    size_t first_file = 0;  // every passage of the run starts at another file of the corpus
    for (const int64_t level : params.integers("concurrency")) {
      const int64_t count = std::max(params.integer("requests"), level * kMinRequestsPerStream);
      std::vector<Question> questions;
      std::vector<ChatRequest> requests;
      for (int64_t i = 0; i < count; ++i) questions.push_back(ask(session, params, passages, passage_chars, first_file, rng, kind)), requests.push_back(questions.back().request);
      const PowerSpan power(session);
      const Clock::time_point started = Clock::now();
      const std::vector<Sample> samples = session.run_closed_loop(requests, static_cast<int>(level));
      const double wall_s = seconds_since(started);

      Row row = Row::with_labels({{"users", std::to_string(level)}});
      int64_t prompt_tokens = 0, tokens = 0;
      for (size_t i = 0; i < samples.size(); ++i) {
        const Sample &sample = samples[i];
        if (!tally(result, sample)) continue;
        prompt_tokens += sample.prompt_tokens, tokens += sample.completion_tokens;
        row.metric(std::string(metric::kPromptTokens), "tokens", Better::none).samples.push_back(sample.prompt_tokens);
        row.metric(std::string(metric::kTtft), "ms", Better::lower).samples.push_back(sample.ttft_ms);
        row.metric("latency", "ms", Better::lower).samples.push_back(sample.wall_ms);
        if (sample.decode_tps() > 0) row.metric(std::string(metric::kDecode), "tok/s", Better::higher).samples.push_back(sample.decode_tps());
        const Fact &fact = questions[i].fact;
        const bool right = contains_text(sample.text, fact.weekday) && contains_text(sample.text, fact.time);
        row.flag("answer right").samples.push_back(right);
        if (!right && row.notes.size() < 2)
          row.notes.push_back(std::to_string(level) + " users: \"" + fact.question + "\" (" + std::string(fact.weekday) + " " + std::string(fact.time) + ") was answered: " + head_of(sample.text, 160));
      }
      if (prompt_tokens > 0) row.metric("read", "tok/s", Better::higher).samples.push_back(static_cast<double>(prompt_tokens) / wall_s);
      power.add_to(row, tokens);
      session.progress("rag: " + std::to_string(level) + " users done");
      result.rows.push_back(std::move(row));
    }
    result.charts = {
        {ChartKind::range, "Time to first token by users", "users", {std::string(metric::kTtft)}, false},
        {ChartKind::range, "Request latency by users", "users", {"latency"}, false},
        {ChartKind::line, "Prompt tokens read per second, all requests together", "users", {"read"}, true},
        {ChartKind::dots, "Answers that hold the planted fact", "users", {"answer right"}, false},
    };
    return result;
  }

 private:
  // One question over its own passages, the fact planted in one of them.
  static Question ask(Session &session, const Params &params, int64_t passages, size_t passage_chars, size_t &first_file, std::mt19937_64 &rng, const std::string &kind) {
    Fact fact = draw_fact(rng);
    const int64_t holder = static_cast<int64_t>(rng() % static_cast<uint64_t>(std::max<int64_t>(passages, 1)));
    std::string context = session.unique_line() + "Passages:\n\n";  // no two prompts share more than the instructions
    for (int64_t p = 0; p < passages; ++p) {
      std::string passage = session.corpus().document(kind, passage_chars, first_file++).text;
      if (p == holder) passage = with_sentence(std::move(passage), fact.sentence);
      context += "[" + std::to_string(p + 1) + "]\n" + passage + "\n";
    }
    ChatRequest request = session.sampled({}, static_cast<int>(params.integer("reply_tokens")));
    request.messages = {{"system", std::string(kInstructions)}, {"user", context + "Question: " + fact.question}};
    request.ignore_eos = false;  // the answer ends where the model ends it
    request.thinking = Thinking::off;
    return {std::move(request), std::move(fact)};
  }
};

}  // namespace

std::unique_ptr<Scenario> make_rag() { return std::make_unique<Rag>(); }

}  // namespace xenithium:bench
