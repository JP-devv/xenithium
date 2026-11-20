// xenithium-bench: the server's own counters (a Prometheus /metrics next to the API, where the endpoint has one).
// Read before and after a scenario; the difference says what the server did for it: how many tokens a speculative
// step accepted, how much of the prompts came from its prefix cache, whether it preempted. The names are the ones
// vLLM-compatible servers export, so this works on any of them.
#pragma once

#include <string>
#include <string_view>
#include <vector>

namespace xenithium::bench {

struct ServerCounters {
  double generation_tokens = 0;
  double prompt_tokens = 0;
  double spec_steps = 0;  // speculative steps, one per request and step
  double spec_accepted_tokens = 0;  // draft tokens accepted
  double prefix_query_tokens = 0;  // prompt tokens looked up in the prefix cache
  double prefix_hit_tokens = 0;  // of those, continued from the cache
  double preemptions = 0;
};

// The counters of a /metrics body (every series of a name summed over its labels); absent ones stay 0.
ServerCounters parse_counters(std::string_view metrics_text);
// What happened between two readings, as report lines ("4.41 tokens per speculative step"); empty when nothing did.
std::vector<std::string> describe_delta(const ServerCounters &before, const ServerCounters &after);

}  // namespace xenithium:bench
