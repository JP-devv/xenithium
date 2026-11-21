// xenithium-bench: the text the prompts are made of.
//
// Long prompts need real text: random tokens give untypical attention, routing and draft acceptance. A corpus is a
// frozen set of files by kind ("code", "prose"). Two sets are embedded in the binary (tools/bench/corpus/<set> through
// xenithium-bench-embed): "hard", this engine's own sources and documentation, and "medium", conventional template-made
// code and prose that drafters guess well. --corpus DIR reads another directory of the <kind>/<file> layout. The set,
// its version and its hash go into every run file: two runs compare only on the same corpus.
#pragma once

#include <cstddef>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace xenithium::bench {

// Bumped by hand whenever a file of the set under tools/bench/corpus/<set> changes, or a standard short prompt
// (prompts.cpp) does (tests/test_bench.cpp pins each set's hash beside it). The easy prompts are versioned with the
// easy mode (mode.hpp).
inline constexpr int kCorpusVersion = 4;        // "hard"
inline constexpr int kMediumCorpusVersion = 1;  // "medium" (named "easy" before the easy mode got its own prompts)

// The embedded sets, the default first.
struct CorpusSet {
  std::string_view name;
  int version;
  std::string_view about;
};
std::span<const CorpusSet> corpus_sets();
const CorpusSet *find_corpus_set(std::string_view name);  // nullptr when no set has that name

namespace embedded {
struct File {
  const char *set;
  const char *kind;
  const char *name;
  const unsigned char *data;
  std::size_t size;
};
extern const File kFiles[];
extern const std::size_t kFileCount;
}  // namespace embedded

struct CorpusFile {
  std::string kind;
  std::string name;
  std::string text;
};

class Corpus {
 public:
  explicit Corpus(std::vector<CorpusFile> files);
  // One embedded set ("hard", "medium"); an unknown name gives an empty corpus.
  static Corpus from_embedded(std::string_view set = "hard");
  // DIR/<kind>/<file>, files in name order. nullopt + err when the directory holds none.
  static std::optional<Corpus> from_dir(const std::string &dir, std::string &err);

  // FNV-1a 64 over every file's kind, name and text, as 16 hex digits.
  std::string hash() const;
  std::vector<std::string> kinds() const;
  std::size_t chars(std::string_view kind) const;

  struct Document {
    std::string text;
    bool wrapped = false;  // the kind holds less text than asked: files repeat
  };
  // About `chars` characters of `kind`: whole files from the kind's file number `first_file` on, each under a line
  // naming it and inside a code fence, the last one cut at a line end. An unknown kind gives an empty document.
  Document document(std::string_view kind, std::size_t chars, std::size_t first_file = 0) const;

 private:
  std::vector<CorpusFile> files_;
};

// A turn said before the prompt's own: the prompt continues a conversation.
struct EarlierTurn {
  std::string_view role;  // "user" or "assistant"
  std::string_view text;
};

// A short prompt of the decode / concurrency scenarios: one user turn, after `earlier` turns when it continues a chat.
struct ShortPrompt {
  std::string_view id;        // "code-merge-lists"
  std::string_view category;  // "code"
  std::string_view text;
  int max_tokens;             // the reply length the category is measured at
  std::span<const EarlierTurn> earlier = {};
};

// The sets of short prompts. `standard`: open tasks the model writes from scratch, three per category (the hard and
// medium modes). `easy`: tasks whose reply mostly repeats or reshapes what the prompt gives - a diff or a whole file
// back, JSON from given data, a fix of given code, a chat that goes on - four per category, with the shorter reply
// lengths such tasks end at (the easy mode).
enum class PromptSet { standard, easy };
std::span<const ShortPrompt> short_prompts(PromptSet set = PromptSet::standard);

// The decode headline's weight per category: the share of a deployment's generated tokens that kind of output is.
struct CategoryWeight {
  std::string_view category;
  double weight;
};
// How a mix turns the categories' rates into one figure. `harmonic`: the rate at which that mix of tokens comes out
// (each category's time per token, weighted). `arithmetic`: the weighted mean of the categories' rates, the figure most
// public benchmarks quote; it reads higher, since a fast category lifts it more than a slow one pulls it down.
enum class MixAverage { harmonic, arithmetic };
// A named set of weights over every category (they sum to 1; a category may weigh 0).
struct WeightMix {
  std::string_view id;     // "agent": the decode scenario's `weights` parameter
  std::string_view about;  // where the shares come from (`list`, the docs)
  std::span<const CategoryWeight> weights;
  MixAverage average = MixAverage::harmonic;
};
std::span<const WeightMix> weight_mixes();          // the default first
std::span<const CategoryWeight> category_weights();  // the default mix: every category, in the usual order

}  // namespace xenithium::bench
