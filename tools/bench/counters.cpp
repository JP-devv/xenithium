#include "counters.hpp"

#include <charconv>
#include <cstdio>

namespace xenithium::bench {
namespace {

struct Series {
  std::string_view name;
  double ServerCounters::*field;
};
constexpr Series kSeries[] = {
    {"vllm:generation_tokens_total", &ServerCounters::generation_tokens},
    {"vllm:prompt_tokens_total", &ServerCounters::prompt_tokens},
    {"vllm:spec_decode_num_drafts_total", &ServerCounters::spec_steps},
    {"vllm:spec_decode_num_accepted_tokens_total", &ServerCounters::spec_accepted_tokens},
    {"vllm:prefix_cache_queries_total", &ServerCounters::prefix_query_tokens},
    {"vllm:prefix_cache_hits_total", &ServerCounters::prefix_hit_tokens},
    {"vllm:num_preemptions_total", &ServerCounters::preemptions},
};

std::string line_of(const char *format, double a, double b = 0) {
  char buf[160];
  std::snprintf(buf, sizeof(buf), format, a, b);
  return buf;
}

}  // namespace

ServerCounters parse_counters(std::string_view text) {
  ServerCounters c;
  for (size_t at = 0; at < text.size();) {
    const size_t eol = std::min(text.find('\n', at), text.size());
    const std::string_view line = text.substr(at, eol - at);
    at = eol + 1;
    if (line.empty() || line.front() == '#') continue;
    const size_t name_end = line.find_first_of("{ ");
    const size_t value_at = line.rfind(' ');
    if (name_end == std::string_view::npos || value_at == std::string_view::npos) continue;
    double value = 0;
    const std::string_view number = line.substr(value_at + 1);
    if (std::from_chars(number.data(), number.data() + number.size(), value).ec != std::errc{}) continue;
    for (const Series &s : kSeries)
      if (line.substr(0, name_end) == s.name) c.*(s.field) += value;
  }
  return c;
}

std::vector<std::string> describe_delta(const ServerCounters &before, const ServerCounters &after) {
  std::vector<std::string> lines;
  if (const double steps = after.spec_steps - before.spec_steps; steps > 0)
    lines.push_back(line_of("server: %.2f tokens per speculative step over %.0f steps", 1 + (after.spec_accepted_tokens - before.spec_accepted_tokens) / steps, steps));
  if (const double asked = after.prefix_query_tokens - before.prefix_query_tokens; asked > 0)
    lines.push_back(line_of("server: its prefix cache served %.0f of %.0f prompt tokens", after.prefix_hit_tokens - before.prefix_hit_tokens, asked));
  if (const double n = after.preemptions - before.preemptions; n > 0) lines.push_back(line_of("server: %.0f requests preempted", n));
  return lines;
}

}  // namespace xenithium:bench
