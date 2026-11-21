#include "session.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <thread>

namespace xenithium::bench {
namespace {

// The calibration text: long enough that the chat template's few tokens do not move the ratio (under 0.5 % at ~6K tokens).
constexpr size_t kCalibrationChars = 24'000;
// Used when the endpoint reports no usage: typical for source code and for English prose on BPE vocabularies.
constexpr double kTypicalCodeCharsPerToken = 3.3, kTypicalProseCharsPerToken = 4.2;

double typical_chars_per_token(std::string_view kind) { return kind == "code" ? kTypicalCodeCharsPerToken : kTypicalProseCharsPerToken; }

}  // namespace

Session::Session(const Client &client, const Corpus &corpus, SessionOptions options)
    : client_(client),
      corpus_(corpus),
      options_(std::move(options)),
      run_stamp_(static_cast<uint64_t>(std::chrono::system_clock::now().time_since_epoch().count())) {}

void Session::progress(std::string_view line) const {
  if (options_.progress) options_.progress(line);
}

ChatRequest Session::sampled(std::string user_text, int max_tokens) {
  ChatRequest r;
  r.messages = {{"user", std::move(user_text)}};
  r.max_tokens = max_tokens;
  r.sampling = options_.sampling;
  r.sampling.seed = static_cast<int64_t>(options_.seed + requests_seeded_++);
  r.ignore_eos = options_.reply_length == ReplyLength::fixed;
  r.thinking = options_.thinking;
  return r;
}

ChatRequest Session::sampled(const ShortPrompt &prompt) {
  ChatRequest r = sampled(std::string(prompt.text), prompt.max_tokens);
  std::vector<Message> earlier;
  for (const EarlierTurn &turn : prompt.earlier) earlier.push_back({std::string(turn.role), std::string(turn.text)});
  r.messages.insert(r.messages.begin(), earlier.begin(), earlier.end());
  if (options_.fresh_prompts) {  // before the conversation's first user turn: the whole request is new to a prefix cache
    const auto first_user = std::ranges::find(r.messages, std::string_view("user"), &Message::role);
    first_user->content.insert(0, unique_line());
  }
  return r;
}

ChatRequest Session::greedy(std::string user_text, int max_tokens) const {
  ChatRequest r;
  r.messages = {{"user", std::move(user_text)}};
  r.max_tokens = max_tokens;
  r.sampling = Sampling{.temperature = 0, .top_p = 1, .top_k = 0, .seed = std::nullopt};
  r.ignore_eos = true;
  r.thinking = options_.thinking;
  return r;
}

double Session::chars_per_token(std::string_view kind) {
  if (const auto it = chars_per_token_.find(kind); it != chars_per_token_.end()) return it->second;
  double ratio = typical_chars_per_token(kind);
  const std::string text = unique_line() + corpus_.document(kind, kCalibrationChars).text;
  if (!text.empty()) {
    const Sample s = client_.chat(greedy(text, 1));
    if (s.ok && s.usage_reported && s.prompt_tokens > 0) ratio = static_cast<double>(text.size()) / s.prompt_tokens;
  }
  chars_per_token_.emplace(std::string(kind), ratio);
  char line[120];
  std::snprintf(line, sizeof(line), "calibrated %.*s text: %.2f characters per token", static_cast<int>(kind.size()), kind.data(), ratio);
  progress(line);
  return ratio;
}

std::string Session::unique_line() {
  char line[64];
  std::snprintf(line, sizeof(line), "Session %016llx-%llu.\n\n", static_cast<unsigned long long>(run_stamp_), static_cast<unsigned long long>(unique_lines_++));
  return line;
}

Session::DocumentPrompt Session::document_prompt(std::string_view kind, int64_t tokens, std::string_view question, bool unique, size_t first_file) {
  const size_t chars = static_cast<size_t>(static_cast<double>(tokens) * chars_per_token(kind));
  const Corpus::Document doc = corpus_.document(kind, chars > question.size() ? chars - question.size() : 0, first_file);
  DocumentPrompt prompt{unique ? unique_line() : std::string(), doc.wrapped};
  prompt.text.append(doc.text).append(question);
  return prompt;
}

Sample Session::chat_alone(const ChatRequest &request) const {
  const auto before = counters();
  Sample sample = client_.chat(request);
  if (sample.ok && sample.cached_tokens < 0 && before)
    if (const auto after = counters()) sample.cached_tokens = static_cast<int>(after->prefix_hit_tokens - before->prefix_hit_tokens);
  return sample;
}

std::vector<Sample> Session::run_together(std::span<const ChatRequest> requests, std::span<const double> start_after_s) const {
  std::vector<Sample> samples(requests.size());
  std::vector<std::jthread> workers;
  workers.reserve(requests.size());
  for (size_t i = 0; i < requests.size(); ++i) {
    const double delay_s = i < start_after_s.size() ? start_after_s[i] : 0;
    workers.emplace_back([this, &requests, &samples, i, delay_s] {
      if (delay_s > 0) std::this_thread::sleep_for(std::chrono::duration<double>(delay_s));
      samples[i] = client_.chat(requests[i]);
    });
  }
  workers.clear();  // joins every request before the samples are read
  return samples;
}

std::vector<Sample> Session::run_closed_loop(std::span<const ChatRequest> requests, int workers) const {
  std::vector<Sample> samples(requests.size());
  std::atomic<size_t> next{0};
  std::vector<std::jthread> pool;
  for (int w = 0; w < workers; ++w)
    pool.emplace_back([this, &requests, &samples, &next] {
      for (size_t i = next.fetch_add(1); i < requests.size(); i = next.fetch_add(1)) samples[i] = client_.chat(requests[i]);
    });
  pool.clear();  // joins
  return samples;
}

std::optional<ServerCounters> Session::counters() const {
  if (!client_.capabilities().metrics) return std::nullopt;
  const auto text = client_.get("/metrics");
  return text ? std::optional(parse_counters(*text)) : std::nullopt;
}

}  // namespace xenithium:bench
