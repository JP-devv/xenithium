// Scenario "concurrency": N users at once, each sending its next request when the last one is answered (a closed
// loop). The server's throughput is the tokens of all of them per second; what one user sees is the per-request rate
// and the TTFT, whose worst case shows requests waiting for a slot or for someone else's prefill.
#include <algorithm>

#include "scenario.hpp"

namespace xenithium::bench {
namespace {

constexpr ParamSpec kParams[] = {
    {"levels", "1,2,4,8,16", "requests in flight at once"},
    {"requests", "32", "requests per level (at least four per stream are sent)"},
    {"warmup", "2", "requests sent first and not counted"},
};
constexpr int64_t kMinRequestsPerStream = 4;

class Concurrency final : public Scenario {
 public:
  std::string_view id() const override { return "concurrency"; }
  std::string_view title() const override { return "Concurrent streams (closed loop, sampled)"; }
  std::string_view about() const override {
    return "N requests in flight at once over the short prompts: tok/s of all streams together, tok/s per request, and the TTFT's median and worst case.";
  }
  std::span<const ParamSpec> params() const override { return kParams; }

  ScenarioResult run(Session &session, const Params &params) const override {
    ScenarioResult result;
    const auto prompts = session.prompts();
    for (int64_t i = 0; i < params.integer("warmup"); ++i) (void)session.client().chat(session.sampled(prompts.front()));

    size_t next_prompt = 0;
    for (const int64_t level : params.integers("levels")) {
      const int64_t count = std::max(params.integer("requests"), level * kMinRequestsPerStream);
      std::vector<ChatRequest> requests;
      for (int64_t i = 0; i < count; ++i) {
        const ShortPrompt &prompt = prompts[next_prompt++ % prompts.size()];
        requests.push_back(session.sampled(prompt));
      }
      const PowerSpan power(session);
      const Clock::time_point started = Clock::now();
      const std::vector<Sample> samples = session.run_closed_loop(requests, static_cast<int>(level));
      const double wall_s = std::chrono::duration<double>(Clock::now() - started).count();

      Row row = Row::with_labels({{"streams", std::to_string(level)}});
      int64_t tokens = 0;
      for (const Sample &sample : samples) {
        if (!tally(result, sample)) continue;
        tokens += sample.completion_tokens;
        add_decode_metrics(row, sample);
      }
      if (tokens > 0) row.metric(std::string(metric::kTotal), "tok/s", Better::higher).samples.push_back(static_cast<double>(tokens) / wall_s);
      power.add_to(row, tokens);
      session.progress("concurrency: " + std::to_string(level) + " streams done");
      result.rows.push_back(std::move(row));
    }
    result.charts = {
        {ChartKind::line, "Throughput by streams", "streams", {std::string(metric::kTotal), std::string(metric::kDecode)}, true},
        {ChartKind::range, "Time to first token by streams", "streams", {std::string(metric::kTtft)}, false},
        {ChartKind::bars, "Tokens per joule by streams", "streams", {std::string(metric::kEfficiency)}, false},
    };
    return result;
  }
};

}  // namespace

std::unique_ptr<Scenario> make_concurrency() { return std::make_unique<Concurrency>(); }

}  // namespace xenithium:bench
