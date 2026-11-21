// xenithium-bench: the run's mode (--mode easy|medium|hard) - how easy the measured text is for a draft model.
//
// A drafting engine's decode rate depends on how well its draft guesses the text, so one build reads very differently
// on different prompts. The three modes are three fixed ways to measure it:
//   hard (the default): the published numbers. Long prompts are cut from this engine's own code and documentation,
//     the short prompts are open tasks written from scratch (PromptSet::standard), every sampled reply runs to its
//     max_tokens, and the decode headline is the agent mix's rate (a harmonic mean).
//   medium: the hard method over conventional template-made code and prose (the "medium" corpus).
//   easy: the method of the usual public benchmarks. Short tasks whose replies repeat or reshape what the prompt
//     gives (PromptSet::easy), replies that end where the model ends them, a fresh first line on every request (no
//     reply starts from the prefix cache), twenty samples per category, and a headline that is the weighted mean of the
//     categories' medians over a code-led mix. Long prompts are cut from the medium corpus: prefill speed depends on
//     the token count, not on the text.
// A mode only sets defaults: --corpus, --reply-length and --set still override them. The run file records the mode and
// its version beside the corpus and the reply length, so a report says what was overridden.
#pragma once

#include <span>
#include <string_view>

#include "corpus.hpp"
#include "session.hpp"

namespace xenithium::bench {

struct Mode {
  std::string_view id;  // "easy", "medium", "hard"
  int version;  // bumped when the mode's method or its own prompts change: two runs compare only at one version
  std::string_view about;  // one line: the usage text, `list`, the launcher's menu
  std::string_view corpus;  // the embedded set the long prompts are cut from
  PromptSet prompts;  // the short prompts of decode, concurrency and arrivals
  ReplyLength reply_length;
  bool fresh_prompts;  // a unique first line on every short prompt
  std::span<const std::string_view> defaults;  // scenario parameters ("decode.weights=easy") under the suite's and --set's
};

inline constexpr std::string_view kDefaultMode = "hard";

std::span<const Mode> modes();  // the default first, then the easier ones
const Mode *find_mode(std::string_view id);  // nullptr for an unknown id

}  // namespace xenithium::bench
