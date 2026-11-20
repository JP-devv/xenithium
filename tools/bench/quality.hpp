// xenithium-bench: questions with one checkable answer, and how a reply is graded.
// The quality scenario asks them greedily and counts the correct ones per category. They are not a leaderboard
// benchmark: they are easy to middling on purpose, so a healthy model answers nearly all and a build or setting that
// damages it (a lossy attention, a bad quantisation, a sampler fault) shows as a drop. Every item is this project's
// own text.
#pragma once

#include <span>
#include <string>
#include <string_view>

namespace xenithium::bench {

enum class AnswerKind {
  number,  // the first number of the answer equals the expected one ("60 km/h" answers 60)
  words,  // the expected words appear in the answer in order, whatever the case and punctuation
  exact,  // the answer is the expected text character for character (after trimming quotes and the final period)
};

struct QualityItem {
  std::string_view id;  // "arithmetic-sum-1-40"
  std::string_view category;  // "arithmetic"
  std::string_view passage;  // a text the question is about, shown before it; empty for none
  std::string_view question;
  AnswerKind kind;
  std::string_view answer;
};
std::span<const QualityItem> quality_items();

// The sentence appended to every question: where the reply must put its answer.
std::string_view answer_instruction();

// The answer a reply gives: what follows its last "ANSWER:" up to the line's end, trimmed of quotes, emphasis marks
// and a final period. Empty when the reply has no such line.
std::string answer_of(std::string_view reply);

// Whether `given` (from answer_of) answers an item of `kind` whose answer is `expected.
bool is_correct(AnswerKind kind, std::string_view expected, std::string_view given);

}  // namespace xenithium:bench
