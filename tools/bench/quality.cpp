#include "quality.hpp"

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cmath>
#include <optional>
#include <vector>

namespace xenithium::bench {
namespace {

constexpr std::string_view kAnswerMark = "answer:";
// Asks for short working: a model that narrates every step of a loop runs past any reply budget before it answers.
constexpr std::string_view kAnswerInstruction =
    "\n\nShow at most a few lines of working, then give the final answer on the last line in the form `ANSWER: <answer>`, with nothing after it.";
// Two numbers this close are the same answer (an answer printed as 60.0 for 60).
constexpr double kNumberTolerance = 1e-6;

// The passage the extraction items ask about.
constexpr std::string_view kDepotReport =
    "Shift report, depot Halden. On Tuesday the night shift received 3 trucks. The first carried 120 crates of valves, the second 85 crates of gaskets, and the "
    "third 64 crates of bearings. Foreman Ilse Brandt signed for the valves at 22:40. Two crates of gaskets arrived damaged and were returned. The bearings went "
    "to bay 7; everything else went to bay 3. The next delivery is due on Friday at 06:15.";
constexpr std::string_view kNoPassage;

constexpr QualityItem kItems[] = {
    {"arithmetic-product-sum", "arithmetic", kNoPassage, "What is 17 x 23 + 19?", AnswerKind::number, "410"},
    {"arithmetic-square", "arithmetic", kNoPassage, "What is (144 / 12) squared, minus 44?", AnswerKind::number, "100"},
    {"arithmetic-percent", "arithmetic", kNoPassage, "What is 15 % of 240?", AnswerKind::number, "36"},
    {"arithmetic-speed", "arithmetic", kNoPassage, "A train covers 150 km in 2.5 hours at constant speed. What is its speed in km/h?", AnswerKind::number, "60"},
    {"arithmetic-sum-1-40", "arithmetic", kNoPassage, "What is the sum of the whole numbers from 1 to 40?", AnswerKind::number, "820"},
    {"arithmetic-remainder", "arithmetic", kNoPassage, "What is the remainder when 1000 is divided by 7?", AnswerKind::number, "6"},

    {"reasoning-order", "reasoning", kNoPassage, "Anna is taller than Ben. Ben is taller than Cara. Dan is shorter than Cara. Who is the shortest? Answer with the name.", AnswerKind::words,
     "Dan"},
    {"reasoning-weekday", "reasoning", kNoPassage, "If today is Wednesday, what day of the week is it 10 days from now?", AnswerKind::words, "Saturday"},
    {"reasoning-all-but", "reasoning", kNoPassage, "A farmer has 17 sheep. All but 9 run away. How many sheep does the farmer have left?", AnswerKind::number, "9"},
    {"reasoning-rate", "reasoning", kNoPassage, "A shop sells pens at 3 for 2 euros. How many pens can you buy with 12 euros?", AnswerKind::number, "18"},
    {"reasoning-syllogism", "reasoning", kNoPassage, "Every server in rack A runs Linux. Machine K7 does not run Linux. Is K7 in rack A? Answer yes or no.", AnswerKind::words, "no"},
    {"reasoning-bat-ball", "reasoning", kNoPassage,
     "A bat and a ball cost 1.10 euros together. The bat costs 1.00 euro more than the ball. How many cents does the ball cost? Answer with the number of cents.",
     AnswerKind::number, "5"},

    {"code-loop", "code", kNoPassage, "What does this Python program print?\n\ntotal = 0\nfor i in range(1, 8):\n    if i % 2 == 1:\n        total += i * i\nprint(total)", AnswerKind::number,
     "84"},
    {"code-slice", "code", kNoPassage, "What does this Python program print?\n\ns = \"benchmark\"\nprint(s[:5].upper() + str(len(s)))", AnswerKind::exact, "BENCH9"},
    {"code-dict", "code", kNoPassage,
     "What does this Python program print?\n\ncounts = {}\nfor word in \"the cat and the hat and the bat\".split():\n    counts[word] = counts.get(word, 0) + 1\n"
     "print(counts[\"the\"] * 10 + counts[\"and\"])",
     AnswerKind::number, "32"},
    {"code-recursion", "code", kNoPassage, "What does this Python program print?\n\ndef f(n):\n    if n < 2:\n        return n\n    return f(n - 1) + f(n - 2)\nprint(f(12))",
     AnswerKind::number, "144"},
    {"code-sort", "code", kNoPassage, "What does this Python program print?\n\nxs = [5, 3, 8, 1, 9, 2]\nxs.sort()\nprint(xs[1] * xs[-2] - len(xs))", AnswerKind::number, "10"},
    {"code-while", "code", kNoPassage,
     "What does this Python program print?\n\nn, steps = 27, 0\nwhile n != 1:\n    n = n // 2 if n % 2 == 0 else 3 * n + 1\n    steps += 1\n    if steps == 10:\n        break\n"
     "print(n)",
     AnswerKind::number, "214"},

    {"extraction-total", "extraction", kDepotReport, "How many crates did the three trucks carry in total?", AnswerKind::number, "269"},
    {"extraction-name", "extraction", kDepotReport, "Who signed for the valves? Answer with the full name.", AnswerKind::words, "Ilse Brandt"},
    {"extraction-kept", "extraction", kDepotReport, "How many crates of gaskets were kept?", AnswerKind::number, "83"},
    {"extraction-bay", "extraction", kDepotReport, "Which bay number received the bearings?", AnswerKind::number, "7"},
    {"extraction-day", "extraction", kDepotReport, "On which day of the week is the next delivery due?", AnswerKind::words, "Friday"},
    {"extraction-count", "extraction", kDepotReport, "How many of the trucks carried fewer than 100 crates?", AnswerKind::number, "2"},

    {"format-reverse", "format", kNoPassage, "Write the word \"engine\" backwards.", AnswerKind::words, "enigne"},
    {"format-sort", "format", kNoPassage, "Sort these words alphabetically and give them separated by single spaces: pear apple fig banana", AnswerKind::words, "apple banana fig pear"},
    {"format-uppercase", "format", kNoPassage, "Write the phrase \"tuned to the last byte\" in capital letters.", AnswerKind::exact, "TUNED TO THE LAST BYTE"},
    {"format-word-count", "format", kNoPassage, "How many words are in this sentence: \"The quick brown fox jumps over the lazy dog\"?", AnswerKind::number, "9"},
    {"format-letter", "format", kNoPassage, "What is the third letter of the word \"kernel\"? Answer with the letter.", AnswerKind::words, "r"},
    {"format-hex", "format", kNoPassage, "Write the decimal number 255 in hexadecimal, in lowercase and without a prefix.", AnswerKind::exact, "ff"},
};

char lower(char c) { return static_cast<char>(std::tolower(static_cast<unsigned char>(c))); }
bool is_word_char(char c) { return std::isalnum(static_cast<unsigned char>(c)) != 0; }

// The text's words in lowercase: runs of letters and digits.
std::vector<std::string> words_of(std::string_view text) {
  std::vector<std::string> words;
  std::string word;
  for (const char c : text) {
    if (is_word_char(c)) {
      word.push_back(lower(c));
    } else if (!word.empty()) {
      words.push_back(std::move(word));
      word.clear();
    }
  }
  if (!word.empty()) words.push_back(std::move(word));
  return words;
}

// The first number in the text ("about 1,234.5 km" -> 1234.5); thousands separators are read through.
std::optional<double> first_number(std::string_view text) {
  std::string digits;
  for (size_t i = 0; i < text.size(); ++i) {
    const char c = text[i];
    const bool is_digit = c >= '0' && c <= '9';
    const bool next_is_digit = i + 1 < text.size() && text[i + 1] >= '0' && text[i + 1] <= '9';
    if (is_digit) digits.push_back(c);
    else if (c == '-' && digits.empty() && next_is_digit) digits.push_back(c);
    else if (c == '.' && !digits.empty() && next_is_digit && digits.find('.') == std::string::npos) digits.push_back(c);
    else if (c == ',' && !digits.empty() && next_is_digit) continue;
    else if (!digits.empty()) break;
  }
  double value = 0;
  if (digits.empty() || std::from_chars(digits.data(), digits.data() + digits.size(), value).ec != std::errc()) return std::nullopt;
  return value;
}

// Without the quotes, emphasis marks and spaces models wrap an answer in, and without a final period.
std::string_view trimmed(std::string_view text) {
  constexpr std::string_view kWrapping = " \t\r\"'`*_";
  const size_t from = text.find_first_not_of(kWrapping);
  if (from == std::string_view::npos) return {};
  text = text.substr(from, text.find_last_not_of(kWrapping) - from + 1);
  if (!text.empty() && text.back() == '.') text.remove_suffix(1);
  const size_t last = text.find_last_not_of(kWrapping);
  return last == std::string_view::npos ? std::string_view() : text.substr(0, last + 1);
}

}  // namespace

std::span<const QualityItem> quality_items() { return kItems; }
std::string_view answer_instruction() { return kAnswerInstruction; }

std::string answer_of(std::string_view reply) {
  std::string lowered(reply);
  std::ranges::transform(lowered, lowered.begin(), lower);
  const size_t mark = lowered.rfind(kAnswerMark);
  if (mark == std::string::npos) return {};
  const size_t from = mark + kAnswerMark.size();
  const size_t line_end = std::min(reply.find('\n', from), reply.size());
  return std::string(trimmed(reply.substr(from, line_end - from)));
}

bool is_correct(AnswerKind kind, std::string_view expected, std::string_view given) {
  switch (kind) {
    case AnswerKind::number: {
      const auto want = first_number(expected), got = first_number(given);
      return want && got && std::abs(*want - *got) <= kNumberTolerance;
    }
    case AnswerKind::words: {
      const std::vector<std::string> want = words_of(expected), got = words_of(given);
      return !want.empty() && !std::ranges::search(got, want).empty();
    }
    case AnswerKind::exact: return trimmed(given) == expected;
  }
  return false;
}

}  // namespace xenithium:bench
