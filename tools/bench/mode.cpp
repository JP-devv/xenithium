#include "mode.hpp"

#include <array>

namespace xenithium::bench {
namespace {

// Four easy prompts per category, five passes: twenty samples per category, as many as the usual public benchmarks
// take. Three requests first, so the server's first-use costs stay out.
constexpr std::array<std::string_view, 3> kEasyDefaults = {"decode.weights=easy", "decode.passes=5", "decode.warmup=3"};

constexpr std::array kModes = {
    Mode{.id = "hard",
         .version = 1,
         .about = "the engine's own code and docs, open tasks, replies run to their length, the agent mix: the published numbers",
         .corpus = "hard",
         .prompts = PromptSet::standard,
         .reply_length = ReplyLength::fixed,
         .fresh_prompts = false,
         .defaults = {}},
    Mode{.id = "medium",
         .version = 1,
         .about = "the hard method over conventional template-made code and prose (CRUD layers, tests, guides)",
         .corpus = "medium",
         .prompts = PromptSet::standard,
         .reply_length = ReplyLength::fixed,
         .fresh_prompts = false,
         .defaults = {}},
    Mode{.id = "easy",
         .version = 1,
         .about = "tasks that repeat or reshape their input, replies end where the model ends them, a code-led mean: the public-benchmark style",
         .corpus = "medium",
         .prompts = PromptSet::easy,
         .reply_length = ReplyLength::natural,
         .fresh_prompts = true,
         .defaults = kEasyDefaults},
};

}  // namespace

std::span<const Mode> modes() { return kModes; }

const Mode *find_mode(std::string_view id) {
  for (const Mode &mode : kModes)
    if (mode.id == id) return &mode;
  return nullptr;
}

}  // namespace xenithium::bench
