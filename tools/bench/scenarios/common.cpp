#include "scenario.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>

namespace xenithium::bench {
namespace {

// Room kept between a prompt and the endpoint's context: the chat template, and the calibration's error (~2 %).
constexpr double kContextMarginShare = 0.03;
constexpr int64_t kContextMarginTokens = 256;

}  // namespace

bool tally(ScenarioResult &result, const Sample &sample) {
  ++result.requests;
  if (sample.ok) {
    result.completion_tokens += sample.completion_tokens;
    return true;
  }
  ++result.errors;
  if (result.first_error.empty()) result.first_error = sample.error;
  return false;
}

void add_decode_metrics(Row &row, const Sample &s) {
  if (s.decode_tps() > 0) row.metric(std::string(metric::kDecode), "tok/s", Better::higher).samples.push_back(s.decode_tps());
  row.metric(std::string(metric::kTtft), "ms", Better::lower).samples.push_back(s.ttft_ms);
  row.metric(std::string(metric::kTokensPerUpdate), "", Better::higher).samples.push_back(s.tokens_per_update());
  if (s.update_gap_ms() > 0) row.metric(std::string(metric::kUpdateGap), "ms", Better::lower).samples.push_back(s.update_gap_ms());
}

void add_prefill_metrics(Row &row, const Sample &s) {
  row.metric(std::string(metric::kPromptTokens), "tokens", Better::none).samples.push_back(s.prompt_tokens);
  row.metric(std::string(metric::kTtft), "ms", Better::lower).samples.push_back(s.ttft_ms);
  if (s.usage_reported && !s.cached_unknown) row.metric(std::string(metric::kPrefill), "tok/s", Better::higher).samples.push_back(s.prefill_tps());
  if (s.cached_unknown)
    row.notes.push_back("prefill tok/s left out where the cached prompt tokens are unknown: the server does not report them per request, and another client's "
                        "cache hits moved its server-wide counter during the request");
}

namespace {

Clock::time_point after_ms(Clock::time_point from, double ms) { return from + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double, std::milli>(ms)); }

}  // namespace

Clock::time_point first_token_time(const Sample &sample) { return after_ms(sample.submitted, sample.ttft_ms); }

StreamInWindow split_at(const Sample &stream, Clock::time_point from, Clock::time_point to) {
  StreamInWindow w;
  const auto at = [&](size_t i) { return after_ms(stream.submitted, stream.update_ms[i]); };
  for (size_t i = 0; i < stream.update_ms.size(); ++i) {
    const Clock::time_point t = at(i);
    if (t <= from) ++w.updates_before;
    else if (t <= to) ++w.updates_inside;
    if (i > 0 && t > from && at(i - 1) < to) w.longest_gap_ms = std::max(w.longest_gap_ms, std::chrono::duration<double, std::milli>(t - at(i - 1)).count());
  }
  if (!stream.update_ms.empty()) {
    w.before_s = std::chrono::duration<double>(from - at(0)).count();
    w.outlasted = at(stream.update_ms.size() - 1) >= to;
  }
  return w;
}

PowerSpan::PowerSpan(const Session &session) : meter_(session.power()), from_s_(meter_ ? meter_->mark() : 0) {}

void PowerSpan::add_to(Row &row, int64_t tokens) const {
  if (!meter_) return;
  const PowerUse use = meter_->use(from_s_, meter_->mark());
  if (use.joules <= 0) return;
  row.metric(std::string(metric::kPower), "W", Better::none).samples.push_back(use.watts());
  if (tokens > 0) row.metric(std::string(metric::kEfficiency), "tok/J", Better::higher).samples.push_back(static_cast<double>(tokens) / use.joules);
}

std::vector<int64_t> depths_that_fit(const Session &session, std::span<const int64_t> depths, int reply_tokens, std::string &skipped) {
  std::vector<int64_t> fit;
  const int64_t context = session.context_tokens();
  for (const int64_t depth : depths) {
    const int64_t needed = depth + reply_tokens + kContextMarginTokens + static_cast<int64_t>(static_cast<double>(depth) * kContextMarginShare);
    if (context <= 0 || needed <= context) fit.push_back(depth);
    else skipped += (skipped.empty() ? "" : ", ") + with_thousands(depth);
  }
  if (!skipped.empty()) skipped = "skipped (past the endpoint's context of " + with_thousands(context) + " tokens): " + skipped;
  return fit;
}

bool contains_text(std::string_view text, std::string_view needle) {
  const auto lower = [](char c) { return static_cast<char>(std::tolower(static_cast<unsigned char>(c))); };
  const auto found = std::ranges::search(text, needle, {}, lower, lower);
  return needle.empty() || !found.empty();
}

std::string head_of(std::string_view text, size_t chars) {
  std::string head(text.substr(0, chars));
  std::ranges::replace(head, '\n', ' ');
  return text.size() > chars ? head + "..." : head;
}

double seconds_since(Clock::time_point from) { return std::chrono::duration<double>(Clock::now() - from).count(); }

double exponential_gap_s(double per_second, std::mt19937_64 &rng) {
  constexpr double kTwoPow53 = 9007199254740992.0;
  const double uniform = static_cast<double>(rng() >> 11) / kTwoPow53;  // [0, 1)
  return -std::log1p(-uniform) / per_second;
}

MixRate mix_rate(std::span<const std::pair<std::string, double>> weights, const std::map<std::string, double, std::less<>> &tps, MixAverage average) {
  double covered = 0, seconds_per_token = 0, weighted_tps = 0;
  for (const auto &[kind, weight] : weights) {
    const auto it = tps.find(kind);
    if (weight <= 0 || it == tps.end() || it->second <= 0) continue;
    covered += weight;
    seconds_per_token += weight / it->second;
    weighted_tps += weight * it->second;
  }
  if (covered <= 0) return {};
  return {average == MixAverage::harmonic ? covered / seconds_per_token : weighted_tps / covered, covered};
}

}  // namespace xenithium:bench
