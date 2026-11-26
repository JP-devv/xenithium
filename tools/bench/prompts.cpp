// xenithium-bench: the short single-turn prompts, by category.
// A category is a kind of output with its own token statistics (code repeats itself and drafts well, prose does not),
// so the decode rate is reported per category and combined by a weight mix (weight_mixes). The texts are this
// project's own. Changing one changes every later decode number: bump kCorpusVersion with it (corpus.hpp).
#include "corpus.hpp"

#include <array>

namespace xenithium::bench {
namespace {

constexpr int kChatTokens = 250, kCodeTokens = 600, kReasoningTokens = 600, kMathTokens = 600, kProseTokens = 800, kJsonTokens = 300, kEditTokens = 500,
              kSummaryTokens = 200;

constexpr std::array kPrompts = {
    ShortPrompt{"chat-followup", "chat",
                "I started running three times a week about a month ago and my knees ache a little after the longer runs. "
                "Is that normal for a beginner, and what should I change first?",
                kChatTokens},
    ShortPrompt{"chat-recommend", "chat",
                "I have a free weekend and about 200 euros. I like hiking, old towns and good coffee, and I live in Munich. "
                "Suggest two different plans and tell me which one you would pick.",
                kChatTokens},
    ShortPrompt{"chat-explain", "chat", "Why does the moon look bigger near the horizon? Explain it the way you would to a curious twelve-year-old.", kChatTokens},

    ShortPrompt{"code-merge-intervals", "code",
                "Write a Python function merge_intervals(intervals) that merges overlapping closed intervals given as (start, end) "
                "tuples in any order. Include a docstring, type hints and five unit tests with pytest.",
                kCodeTokens},
    ShortPrompt{"code-lru-cache", "code",
                "Implement a thread-safe LRU cache in C++20 as a class template LruCache<Key, Value> with get, put and a capacity "
                "given at construction. Use std::list and std::unordered_map, and explain the complexity of each operation in comments.",
                kCodeTokens},
    ShortPrompt{"code-rate-limiter", "code",
                "Write a token-bucket rate limiter in Go: a type Limiter with Allow() bool and Wait(ctx context.Context) error, "
                "safe for concurrent use, with a table-driven test.",
                kCodeTokens},

    ShortPrompt{"reason-schedule", "reasoning",
                "Five talks (A to E) must be scheduled in five consecutive one-hour slots. A must be before C. B cannot be first or "
                "last. D must be directly after E. C cannot be directly after A. List every valid order and show how you ruled the others out.",
                kReasoningTokens},
    ShortPrompt{"reason-estimate", "reasoning",
                "Estimate how many litres of water a city of 500,000 people uses for showers in one year. State every assumption, "
                "compute step by step, and give a low and a high bound.",
                kReasoningTokens},
    ShortPrompt{"reason-plan", "reasoning",
                "A team of four has six weeks to move a web service from one data centre to another with no more than ten minutes "
                "of downtime. Lay out a plan week by week, name the three biggest risks and how each is reduced.",
                kReasoningTokens},

    ShortPrompt{"math-derivative", "math",
                "Differentiate f(x) = x^3 * e^(2x) * sin(x) step by step with the product rule, simplify the result, then evaluate "
                "f'(x) at x = 0 and at x = pi / 2. Write every formula out.",
                kMathTokens},
    ShortPrompt{"math-dice", "math",
                "Three fair six-sided dice are rolled. Work out, showing every step and every count, the probability that their sum "
                "is 10, and then the expected value of the largest of the three dice.",
                kMathTokens},
    ShortPrompt{"math-elimination", "math",
                "Solve the system 2x + 3y - z = 5, x - y + 4z = 6, 3x + 2y + 2z = 13 by Gaussian elimination. Show the augmented "
                "matrix after every row operation, give x, y and z as fractions, and verify them in all three equations.",
                kMathTokens},

    ShortPrompt{"prose-story", "prose",
                "Write a short story of about 600 words about a night-shift ferry pilot who notices that one passenger boards every "
                "night but never gets off.",
                kProseTokens},
    ShortPrompt{"prose-essay", "prose", "Write an essay of about 600 words on why cities built before the car are pleasant to walk in, with concrete examples.",
                kProseTokens},
    ShortPrompt{"prose-letter", "prose",
                "Write a warm, specific letter of about 500 words from a retiring physics teacher to the class she taught for the last time.", kProseTokens},

    ShortPrompt{"json-extract", "json",
                "Extract the people, places and dates from this text as JSON with the keys \"people\", \"places\" and \"dates\" (arrays "
                "of strings), and output only the JSON.\n\nOn 14 March 1879 Albert Einstein was born in Ulm; the family moved to "
                "Munich a year later. In 1905, while working at the patent office in Bern, he published four papers. Mileva Maric, "
                "whom he had met in Zurich, was his first wife. He received the Nobel Prize in 1921 and moved to Princeton in 1933.",
                kJsonTokens},
    ShortPrompt{"json-catalog", "json",
                "Produce a JSON array of eight fictional products for a hardware store. Each object has \"sku\" (string), \"name\", "
                "\"category\", \"price_eur\" (number), \"in_stock\" (boolean) and \"tags\" (array of strings). Output only the JSON.",
                kJsonTokens},
    ShortPrompt{"json-config", "json",
                "Write a JSON configuration for a web service with three environments (dev, staging, prod). Each has a database "
                "(host, port, pool size), a cache (ttl seconds, max entries), logging (level, format) and feature flags. Output only the JSON.",
                kJsonTokens},

    ShortPrompt{"edit-refactor", "edit",
                "Refactor this function so that it has no nested conditionals and no duplicated work, keep its behaviour, and return "
                "the whole new function.\n\n```python\ndef price(order):\n    total = 0\n    for item in order['items']:\n        if item['qty'] > 0:\n"
                "            if item['type'] == 'book':\n                total = total + item['qty'] * item['price'] * 0.93\n            else:\n"
                "                if item['type'] == 'food':\n                    total = total + item['qty'] * item['price'] * 0.93\n                else:\n"
                "                    total = total + item['qty'] * item['price'] * 0.81\n    if order['country'] == 'DE':\n        if total > 100:\n"
                "            total = total - 5\n    return round(total, 2)\n```",
                kEditTokens},
    ShortPrompt{"edit-bugfix", "edit",
                "This function should return the index of the first element that is >= target in a sorted list, or len(xs) if there is "
                "none. It is wrong for some inputs. Fix it, return the whole function, then name the inputs that failed.\n\n```python\n"
                "def lower_bound(xs, target):\n    lo, hi = 0, len(xs) - 1\n    while lo < hi:\n        mid = (lo + hi) // 2\n"
                "        if xs[mid] < target:\n            lo = mid\n        else:\n            hi = mid - 1\n    return lo\n```",
                kEditTokens},
    ShortPrompt{"edit-add-feature", "edit",
                "Add an optional timeout in seconds to this function (None = wait forever) and raise TimeoutError when it expires. "
                "Return the whole new function.\n\n```python\nimport time\n\ndef wait_for(predicate, interval=0.1):\n    while True:\n"
                "        if predicate():\n            return True\n        time.sleep(interval)\n```",
                kEditTokens},

    ShortPrompt{"summary-incident", "summary",
                "Summarise this incident report in four sentences for a manager.\n\nAt 02:14 the primary database of the booking "
                "service stopped accepting writes after its disk filled up: a nightly export job had been changed the day before to "
                "write uncompressed files to the same volume. The on-call engineer was paged at 02:19, found the volume at 100 % at "
                "02:31 and deleted the export files at 02:38; writes resumed at 02:40. During the 26 minutes about 1,900 bookings "
                "failed and customers saw an error page. Reads were not affected. The export job has been moved to its own volume, an "
                "alert now fires at 80 % disk usage, and the change that removed the compression was not reviewed, which the team "
                "will address with a rule that job changes need a second pair of eyes.",
                kSummaryTokens},
    ShortPrompt{"summary-study", "summary",
                "Summarise this abstract in three sentences for a general reader.\n\nWe followed 4,200 adults aged 40 to 65 for nine "
                "years and recorded their daily step counts with wrist-worn devices for one week each year. Participants who "
                "averaged at least 7,000 steps a day had a 31 % lower rate of cardiovascular events than those under 4,000, after "
                "adjusting for age, sex, smoking, body mass index and income. The benefit levelled off above roughly 10,000 steps, "
                "and walking pace added no further effect once the step count was taken into account. Because step counts were "
                "measured for one week a year, and healthier people may simply walk more, the study shows an association and cannot "
                "prove that walking caused the difference.",
                kSummaryTokens},
    ShortPrompt{"summary-meeting", "summary",
                "Turn these meeting notes into a summary with the decisions and the open questions.\n\nNotes: release moved from "
                "the 12th to the 19th because the payment provider's sandbox was down for three days. Anna thinks the 19th is safe "
                "if the load test passes on Thursday; Ben is less sure because the new cache has only run in staging for a week. "
                "Agreed: the load test decides. If p99 stays under 300 ms we ship the 19th, otherwise the 26th. Nobody owns the "
                "customer email yet. Chloe will ask legal whether the new terms need a notice period. The mobile team needs the "
                "API freeze by Monday and has not heard back about the two new fields.",
                kSummaryTokens},
};

// The shares of a coding agent's generated tokens, from what an agent harness reading, editing, building and testing
// code generates, split by what was generated - the thinking
// (reasoning), the shell commands and scripts and source files written (code), the patches (edit), the other tool-call
// arguments and call envelopes (json), notes and docs written (prose), the narration beside tool calls (chat) and the
// report that ends a turn of work (summary). The weights below are these shares; maths is never asked for.
constexpr std::array kAgentMix = {
    CategoryWeight{"chat", 0.04}, CategoryWeight{"code", 0.19}, CategoryWeight{"reasoning", 0.63}, CategoryWeight{"math", 0.00},
    CategoryWeight{"prose", 0.03}, CategoryWeight{"json", 0.03}, CategoryWeight{"edit", 0.04}, CategoryWeight{"summary", 0.04},
};
// Every kind of output alike: the categories' own rates, no deployment in mind.
constexpr std::array kEvenMix = {
    CategoryWeight{"chat", 0.125}, CategoryWeight{"code", 0.125}, CategoryWeight{"reasoning", 0.125}, CategoryWeight{"math", 0.125},
    CategoryWeight{"prose", 0.125}, CategoryWeight{"json", 0.125}, CategoryWeight{"edit", 0.125}, CategoryWeight{"summary", 0.125},
};
constexpr std::array kMixes = {
    WeightMix{"agent", "a coding agent's generated tokens on Qwen3.8 27B and Flash-Next (measured: 63 % reasoning, 19 % code, 4 % each edit, summary, chat, 3 % each prose, json)", kAgentMix},
    WeightMix{"even", "every kind of output alike", kEvenMix},
};

}  // namespace

std::span<const ShortPrompt> short_prompts() { return kPrompts; }
std::span<const WeightMix> weight_mixes() { return kMixes; }
std::span<const CategoryWeight> category_weights() { return kAgentMix; }

}  // namespace xenithium:bench
