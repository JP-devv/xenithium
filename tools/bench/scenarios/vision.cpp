// Scenario "vision": image input. A picture of a 3 x 3 grid of coloured squares is sent with a question about one
// square, at several picture sizes: the time to the first token (decoding, the vision tower and the image's tokens in
// the prompt), the prompt tokens an image of that size costs, and whether the square's colour was named. Every
// picture is new (its colours are drawn per request), so no cache answers for it.
#include <algorithm>
#include <random>

#include "image.hpp"
#include "scenario.hpp"

namespace xenithium::bench {
namespace {

constexpr ParamSpec kParams[] = {
    {"sizes", "448,896,1344", "picture sides in pixels (the endpoint may scale a large one down)"},
    {"images", "1", "pictures per request (the question is about the last one)"},
    {"passes", "3", "requests per size"},
    {"reply_tokens", "16", "tokens allowed for the answer"},
};
constexpr int kColumns = 3;
constexpr std::string_view kCellNames[kColumns * kColumns] = {"top-left", "top-centre", "top-right", "middle-left", "centre", "middle-right", "bottom-left", "bottom-centre", "bottom-right"};
constexpr size_t kAnswerShown = 60;

// A picture with its question and the colour that answers it.
struct Puzzle {
  std::string data_url;
  std::string question;
  std::string_view answer;
};

Puzzle make_puzzle(int side, std::mt19937_64 &rng) {
  const auto colors = grid_colors();
  std::vector<Rgb> cells;
  std::vector<std::string_view> names;
  for (int i = 0; i < kColumns * kColumns; ++i) {
    const NamedColor &color = colors[rng() % colors.size()];
    cells.push_back(color.rgb);
    names.push_back(color.name);
  }
  const size_t asked = rng() % cells.size();
  return {png_data_url(grid_image(side, kColumns, cells)),
          "The picture shows a 3 x 3 grid of coloured squares. What colour is the " + std::string(kCellNames[asked]) + " square? Answer with the colour's name only.", names[asked]};
}

class Vision final : public Scenario {
 public:
  std::string_view id() const override { return "vision"; }
  std::string_view title() const override { return "Image input by picture size"; }
  std::string_view about() const override {
    return "A generated picture of coloured squares with a question about one of them, at 448 to 1,344 pixels: TTFT, the prompt tokens an image costs, and whether the colour was named.";
  }
  std::span<const ParamSpec> params() const override { return kParams; }

  ScenarioResult run(Session &session, const Params &params) const override {
    ScenarioResult result;
    std::mt19937_64 rng(session.seed());
    const int reply_tokens = static_cast<int>(params.integer("reply_tokens"));
    const int64_t images = std::max<int64_t>(1, params.integer("images"));
    for (const int64_t side : params.integers("sizes")) {
      Row row = Row::with_labels({{"picture", std::to_string(side) + " px" + (images > 1 ? " x " + std::to_string(images) : "")}});
      for (int64_t pass = 0; pass < params.integer("passes"); ++pass) {
        Message turn{"user", {}, {}};
        Puzzle puzzle;
        for (int64_t i = 0; i < images; ++i) {  // the last picture's question is the one asked
          puzzle = make_puzzle(static_cast<int>(side), rng);
          turn.image_urls.push_back(std::move(puzzle.data_url));
        }
        turn.content = images > 1 ? "Look at the last picture. " + puzzle.question : puzzle.question;
        ChatRequest request = session.greedy({}, reply_tokens);
        request.messages = {std::move(turn)};
        request.ignore_eos = false;  // the answer is one word
        request.thinking = Thinking::off;
        const Sample sample = session.client().chat(request);
        if (result.requests == 0 && refused_by_endpoint(sample)) {  // no image input there: skipped, not failed
          result.rows.push_back(Row::note_only("skipped: the endpoint does not take images (" + sample.error.substr(0, kAnswerShown * 2) + ")"));
          return result;
        }
        if (!tally(result, sample)) continue;
        const bool named = contains_text(sample.text, puzzle.answer);
        row.flag("answered").samples.push_back(named ? 1 : 0);
        add_prefill_metrics(row, sample);
        if (!named && row.notes.empty()) row.notes.push_back("at " + std::to_string(side) + " px the answer was \"" + sample.text.substr(0, kAnswerShown) + "\", the square is " + std::string(puzzle.answer));
      }
      session.progress("vision: " + std::to_string(side) + " px done");
      result.rows.push_back(std::move(row));
    }
    result.charts = {
        {ChartKind::bars, "Time to first token by picture size", "picture", {std::string(metric::kTtft)}, false},
        {ChartKind::bars, "Prompt tokens by picture size", "picture", {std::string(metric::kPromptTokens)}, false},
        {ChartKind::dots, "Colour named correctly", "picture", {"answered"}, false},
    };
    return result;
  }
};

}  // namespace

std::unique_ptr<Scenario> make_vision() { return std::make_unique<Vision>(); }

}  // namespace xenithium:bench
