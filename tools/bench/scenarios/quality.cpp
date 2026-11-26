// Scenario "quality": does the model still answer correctly? Questions with one checkable answer, asked greedily,
// counted per category - arithmetic, reasoning, reading code, extraction from a passage, following a format - plus
// facts planted across a long document that have to be found and combined. Speed numbers of a build or a lossy
// setting mean little if this score drops.
#include <algorithm>
#include <cstdio>
#include <ranges>

#include "quality.hpp"
#include "scenario.hpp"

namespace xenithium::bench {
namespace {

constexpr ParamSpec kParams[] = {
    {"reply_tokens", "1500", "the most tokens an answer may take; replies end on their own, so this only bounds a runaway one"},
    {"context_tokens", "8000", "length in tokens of the document the long-context questions are asked on (0 = leave them out)"},
    {"kind", "prose", "corpus kind of that document"},
};
constexpr std::string_view kContextCategory = "context";
constexpr size_t kAnswerShown = 120;  // characters of a wrong answer kept for the report

// The facts planted in the long document, each on a line of its own at a share of the document's length.
struct PlantedFact {
  std::string_view depot;
  int crates;
  double position;
};
constexpr PlantedFact kFacts[] = {{"Arden", 314, 0.15}, {"Brisk", 527, 0.40}, {"Corvin", 208, 0.65}, {"Dunmore", 96, 0.90}};

struct ContextQuestion {
  std::string_view id, question;
  AnswerKind kind;
  std::string_view answer;
};
constexpr ContextQuestion kContextQuestions[] = {
    {"context-lookup", "According to the inventory notes in the text above, how many crates does depot Arden hold?", AnswerKind::number, "314"},
    {"context-reverse", "According to the inventory notes in the text above, which depot holds 527 crates? Answer with the depot's name.", AnswerKind::words, "Brisk"},
    {"context-sum", "According to the inventory notes in the text above, how many crates do depot Arden and depot Corvin hold together?", AnswerKind::number, "522"},
    {"context-least", "According to the inventory notes in the text above, which of the four depots holds the fewest crates? Answer with the depot's name.", AnswerKind::words,
     "Dunmore"},
};

// `document` with every planted fact on a line of its own, at the first line end past its position.
std::string with_facts(std::string document) {
  // back to front, so an insertion does not move the positions still to come
  for (const PlantedFact &fact : std::views::reverse(kFacts)) {
    const size_t from = static_cast<size_t>(static_cast<double>(document.size()) * fact.position);
    const size_t line_end = document.find('\n', from);
    char note[96];
    std::snprintf(note, sizeof(note), "Inventory note: depot %.*s holds %d crates.\n", static_cast<int>(fact.depot.size()), fact.depot.data(), fact.crates);
    document.insert(line_end == std::string::npos ? document.size() : line_end + 1, note);
  }
  return document;
}

class Quality final : public Scenario {
 public:
  std::string_view id() const override { return "quality"; }
  std::string_view title() const override { return "Answer quality (greedy, one checkable answer per question)"; }
  std::string_view about() const override {
    return "Thirty questions with one right answer in five categories, and four on facts planted across an 8K-token document: the share answered correctly.";
  }
  std::span<const ParamSpec> params() const override { return kParams; }

  ScenarioResult run(Session &session, const Params &params) const override {
    Grader grader{session, static_cast<int>(params.integer("reply_tokens")), {}, {}};
    for (const QualityItem &item : quality_items()) {
      std::string prompt(item.passage);
      if (!prompt.empty()) prompt += "\n\n";
      grader.ask(item.id, item.category, item.question, prompt + std::string(item.question), item.kind, item.answer);
    }
    session.progress("quality: the short questions done");

    if (int64_t depth = params.integer("context_tokens"); depth > 0) {
      std::string skipped;
      if (depths_that_fit(session, std::span(&depth, 1), grader.reply_tokens, skipped).empty()) {
        grader.result.rows.push_back(Row::note_only(skipped));
      } else {
        // one document for the four questions: the first reads it, the rest may come from the prefix cache
        const std::string document = with_facts(session.document_prompt(params.text("kind"), depth, "", /*unique=*/true).text);
        for (const ContextQuestion &q : kContextQuestions) grader.ask(q.id, kContextCategory, q.question, document + "\n\n" + std::string(q.question), q.kind, q.answer);
        session.progress("quality: the long-context questions done");
      }
    }
    return grader.finish();
  }

 private:
  // Asks, grades and keeps the per-category rows in first-asked order.
  struct Grader {
    Session &session;
    int reply_tokens;
    ScenarioResult result;
    std::vector<Row> rows;

    // `question` as the report shows it; `prompt` as sent (with its passage or document)
    void ask(std::string_view id, std::string_view category, std::string_view question, const std::string &prompt, AnswerKind kind, std::string_view expected) {
      ChatRequest request = session.greedy(prompt + std::string(answer_instruction()), reply_tokens);
      request.ignore_eos = false;  // the reply ends where the model ends it
      // an answer needs no reasoning pass: asked without one unless the run turns thinking on, so the score does not
      // hang on how long a model thinks
      request.thinking = session.thinking() == Thinking::on ? Thinking::on : Thinking::off;
      const Sample sample = session.client().chat(request);
      if (!tally(result, sample)) return;
      const std::string given = answer_of(sample.text);
      const bool correct = is_correct(kind, expected, given);
      row_of(category).flag("correct").samples.push_back(correct ? 1 : 0);
      if (!correct)
        result.misses.push_back({std::string(id), std::string(category), std::string(question), std::string(expected), given.substr(0, kAnswerShown),
                                 given.empty() ? (sample.finish_reason == "length" ? "no ANSWER line: the reply hit reply_tokens" : "no ANSWER line") : std::string()});
    }

    Row &row_of(std::string_view category) {
      const auto it = std::ranges::find_if(rows, [&](const Row &row) { return row.labels.front().second == category; });
      if (it != rows.end()) return *it;
      rows.push_back(Row::with_labels({{"category", std::string(category)}}));
      return rows.back();
    }

    ScenarioResult finish() {
      double correct = 0, asked = 0;
      for (Row &row : rows) {
        for (const double sample : row.metrics.front().samples) correct += sample, asked += 1;
        result.rows.push_back(std::move(row));
      }
      if (asked > 0) {
        Row all = Row::with_labels({{"category", "all"}});
        all.metric(std::string(metric::kQualityScore), "%", Better::higher).samples.push_back(100.0 * correct / asked);
        result.rows.push_back(std::move(all));
      }
      result.charts = {{ChartKind::dots, "Questions answered correctly", "category", {"correct"}, false}};
      return result;
    }
  };
};

}  // namespace

std::unique_ptr<Scenario> make_quality() { return std::make_unique<Quality>(); }

}  // namespace xenithium:bench
