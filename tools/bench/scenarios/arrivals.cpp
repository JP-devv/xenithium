// Scenario "arrivals": an open-loop load. Requests arrive at a given rate with random gaps (a Poisson process),
// whether or not the earlier ones are answered - what a service with independent users sees. Unlike the closed loop
// of "concurrency", a server that falls behind is not spared: its queue grows, and the TTFT, the latency and the
// number of requests in flight show it. The rate at which they leave the ground is the load the endpoint sustains.
#include <algorithm>
#include <cmath>
#include <random>

#include "scenario.hpp"

namespace xenithium::bench {
namespace {

constexpr ParamSpec kParams[] = {
    {"rates", "0.5,1,2", "arrivals per second"},
    {"requests", "32", "requests per rate"},
    {"warmup", "2", "requests sent first and not counted"},
};
// Arrival times of a Poisson process, from 0 on.
std::vector<double> poisson_arrivals_s(size_t count, double per_second, std::mt19937_64 &rng) {
  std::vector<double> at_s(count, 0.0);
  for (size_t i = 1; i < count; ++i) at_s[i] = at_s[i - 1] + exponential_gap_s(per_second, rng);
  return at_s;
}

double end_s(const Sample &s, Clock::time_point origin) { return std::chrono::duration<double>(s.submitted - origin).count() + s.wall_ms / 1e3; }

// The most requests in flight at one moment.
int peak_in_flight(std::span<const Sample> samples, Clock::time_point origin) {
  std::vector<std::pair<double, int>> edges;  // (time, +1 at a submit, -1 at an end)
  for (const Sample &s : samples) {
    if (!s.ok) continue;
    edges.emplace_back(std::chrono::duration<double>(s.submitted - origin).count(), +1);
    edges.emplace_back(end_s(s, origin), -1);
  }
  std::ranges::sort(edges);
  int in_flight = 0, peak = 0;
  for (const auto &[at, step] : edges) peak = std::max(peak, in_flight += step);
  return peak;
}

class Arrivals final : public Scenario {
 public:
  std::string_view id() const override { return "arrivals"; }
  std::string_view title() const override { return "Open-loop load (Poisson arrivals, sampled)"; }
  std::string_view about() const override {
    return "Requests arriving at 0.5 to 2 per second with random gaps, answered or not: TTFT and latency per rate, the tokens per second served, and the most requests in flight.";
  }
  std::span<const ParamSpec> params() const override { return kParams; }

  ScenarioResult run(Session &session, const Params &params) const override {
    ScenarioResult result;
    const auto prompts = short_prompts();
    for (int64_t i = 0; i < params.integer("warmup"); ++i) (void)session.client().chat(session.sampled(std::string(prompts.front().text), prompts.front().max_tokens));

    std::mt19937_64 rng(session.seed());
    const size_t count = static_cast<size_t>(std::max<int64_t>(1, params.integer("requests")));
    size_t next_prompt = 0;
    for (const double rate : params.reals("rates")) {
      std::vector<ChatRequest> requests;
      for (size_t i = 0; i < count; ++i) {
        const ShortPrompt &prompt = prompts[next_prompt++ % prompts.size()];
        requests.push_back(session.sampled(std::string(prompt.text), prompt.max_tokens));
      }
      const std::vector<double> arrivals = poisson_arrivals_s(count, rate, rng);
      const PowerSpan power(session);
      const Clock::time_point origin = Clock::now();
      const std::vector<Sample> samples = session.run_together(requests, arrivals);

      char label[32];
      std::snprintf(label, sizeof(label), "%g", rate);
      Row row = Row::with_labels({{"arrivals/s", label}});
      int64_t tokens = 0;
      double last_end_s = 0;
      for (const Sample &sample : samples) {
        if (!tally(result, sample)) continue;
        tokens += sample.completion_tokens;
        last_end_s = std::max(last_end_s, end_s(sample, origin));
        row.metric(std::string(metric::kTtft), "ms", Better::lower).samples.push_back(sample.ttft_ms);
        row.metric("latency", "ms", Better::lower).samples.push_back(sample.wall_ms);
        if (sample.decode_tps() > 0) row.metric(std::string(metric::kDecode), "tok/s", Better::higher).samples.push_back(sample.decode_tps());
      }
      if (tokens > 0 && last_end_s > 0) {
        row.metric(std::string(metric::kTotal), "tok/s", Better::higher).samples.push_back(static_cast<double>(tokens) / last_end_s);
        row.metric("peak in flight", "", Better::none).samples.push_back(peak_in_flight(samples, origin));
      }
      power.add_to(row, tokens);
      session.progress(std::string("arrivals: ") + label + " per second done");
      result.rows.push_back(std::move(row));
    }
    result.charts = {
        {ChartKind::range, "Time to first token by arrival rate", "arrivals/s", {std::string(metric::kTtft)}, false},
        {ChartKind::range, "Request latency by arrival rate", "arrivals/s", {"latency"}, false},
        {ChartKind::line, "Tokens per second served by arrival rate", "arrivals/s", {std::string(metric::kTotal)}, true},
    };
    return result;
  }
};

}  // namespace

std::unique_ptr<Scenario> make_arrivals() { return std::make_unique<Arrivals>(); }

}  // namespace xenithium:bench
