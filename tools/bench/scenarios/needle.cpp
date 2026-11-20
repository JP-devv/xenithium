// Scenario "needle": is a fact stated once in a long document still found? A sentence with a passphrase is put at a
// given depth of the document and asked for at the end (greedy, thinking off). It is the check to run with any lossy
// long-context setting (an attention that reads part of the context, a compressed KV): speed means nothing if this fails.
#include <cstdio>

#include "scenario.hpp"

namespace xenithium::bench {
namespace {

constexpr ParamSpec kParams[] = {
    {"depths", "32000,128000,250000", "document lengths in tokens (those past the endpoint's context are skipped)"},
    {"position_pct", "35", "where in the document the sentence stands, in percent of its length"},
    {"kind", "code", "corpus kind of the document"},
    {"reply_tokens", "48", "tokens allowed for the answer"},
};
constexpr std::string_view kQuestion = "\n\nWhat is the secret passphrase stated in the text above? Answer with the passphrase only.";
constexpr size_t kAnswerShown = 80;  // of a wrong answer, in the report

// A passphrase per depth, so no answer can come from an earlier request.
std::string passphrase_for(int64_t depth) {
  char word[32];
  std::snprintf(word, sizeof(word), "MAGENTA-%04lld", static_cast<long long>(7134 + depth % 2000));
  return word;
}

// `document` with the sentence on a line of its own at the first line end past `position` (a share of its length).
std::string with_sentence(std::string document, double position, const std::string &sentence) {
  const size_t from = static_cast<size_t>(static_cast<double>(document.size()) * position);
  const size_t line_end = document.find('\n', from);
  document.insert(line_end == std::string::npos ? document.size() : line_end + 1, sentence + "\n");
  return document;
}

class Needle final : public Scenario {
 public:
  std::string_view id() const override { return "needle"; }
  std::string_view title() const override { return "A fact stated once in a long document (found = 1)"; }
  std::string_view about() const override {
    return "A passphrase sentence at 35 % of a 32K to 250K-token document, asked for at the end: found or not. The check for lossy long-context settings.";
  }
  std::span<const ParamSpec> params() const override { return kParams; }

  ScenarioResult run(Session &session, const Params &params) const override {
    ScenarioResult result;
    const int reply_tokens = static_cast<int>(params.integer("reply_tokens"));
    const double position = static_cast<double>(params.integer("position_pct")) / 100.0;
    std::string skipped;
    const std::vector<int64_t> asked = params.integers("depths");
    for (const int64_t depth : depths_that_fit(session, asked, reply_tokens, skipped)) {
      const std::string passphrase = passphrase_for(depth);
      const std::string document = session.document_prompt(params.text("kind"), depth, "", /*unique=*/true).text;
      ChatRequest request = session.greedy(with_sentence(document, position, "The secret passphrase is " + passphrase + ".") + std::string(kQuestion), reply_tokens);
      request.ignore_eos = false;  // the answer is short: let the model stop
      request.thinking = Thinking::off;
      const Sample sample = session.client().chat(request);
      Row row = Row::with_labels({{"depth", with_thousands(depth)}});
      if (tally(result, sample)) {
        const bool found = (sample.text + sample.reasoning).find(passphrase) != std::string::npos;
        row.flag("found").samples.push_back(found ? 1 : 0);
        add_prefill_metrics(row, sample);
        if (!found) row.notes.push_back("at " + with_thousands(depth) + " the answer was: \"" + sample.text.substr(0, kAnswerShown) + "\"");
      }
      session.progress("needle: " + with_thousands(depth) + " tokens done");
      result.rows.push_back(std::move(row));
    }
    if (!skipped.empty()) result.rows.push_back(Row::note_only(skipped));
    result.charts = {{ChartKind::dots, "Passphrase found by document depth", "depth", {"found"}, false}};
    return result;
  }
};

}  // namespace

std::unique_ptr<Scenario> make_needle() { return std::make_unique<Needle>(); }

}  // namespace xenithium:bench
