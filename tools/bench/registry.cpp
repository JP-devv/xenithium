// The scenarios and the suites: the one place that lists them. A new scenario is a file in scenarios/ with a
// make_<name> and a line in all_scenarios; a suite is a table of scenarios with the parameters it changes.
#include "scenario.hpp"

namespace xenithium::bench {

std::unique_ptr<Scenario> make_decode();
std::unique_ptr<Scenario> make_concurrency();
std::unique_ptr<Scenario> make_prefill();
std::unique_ptr<Scenario> make_deep_decode();
std::unique_ptr<Scenario> make_turns();
std::unique_ptr<Scenario> make_burst();
std::unique_ptr<Scenario> make_late_arrival();
std::unique_ptr<Scenario> make_mixed();
std::unique_ptr<Scenario> make_needle();
std::unique_ptr<Scenario> make_repeat();
std::unique_ptr<Scenario> make_structured();
std::unique_ptr<Scenario> make_arrivals();
std::unique_ptr<Scenario> make_vision();
std::unique_ptr<Scenario> make_quality();
std::unique_ptr<Scenario> make_long_agents();
std::unique_ptr<Scenario> make_prefill_decode();
std::unique_ptr<Scenario> make_chat_sessions();
std::unique_ptr<Scenario> make_rag();
std::unique_ptr<Scenario> make_autocomplete();
std::unique_ptr<Scenario> make_tool_calls();
std::unique_ptr<Scenario> make_long_reply();

std::vector<std::unique_ptr<Scenario>> all_scenarios() {
  std::vector<std::unique_ptr<Scenario>> all;
  all.push_back(make_decode());
  all.push_back(make_concurrency());
  all.push_back(make_prefill());
  all.push_back(make_deep_decode());
  all.push_back(make_turns());
  all.push_back(make_burst());
  all.push_back(make_late_arrival());
  all.push_back(make_mixed());
  all.push_back(make_needle());
  all.push_back(make_repeat());
  all.push_back(make_structured());
  all.push_back(make_arrivals());
  all.push_back(make_vision());
  all.push_back(make_quality());
  all.push_back(make_long_agents());
  all.push_back(make_prefill_decode());
  all.push_back(make_chat_sessions());
  all.push_back(make_rag());
  all.push_back(make_autocomplete());
  all.push_back(make_tool_calls());
  all.push_back(make_long_reply());
  return all;
}

const std::vector<Suite> &suites() {
  static const std::vector<Suite> kSuites = {
      {"quick", "A first look: every headline number once.", 3,
       {{"decode", {{"passes", "1"}}},
        {"concurrency", {{"levels", "1,4,8"}, {"requests", "16"}}},
        {"prefill", {{"depths", "2000,8000,32000"}, {"runs", "1"}}},
        {"deep-decode", {{"depths", "32000"}, {"repeats", "1"}}}}},
      {"standard", "The numbers to quote for a build: decode, concurrency, prefill to 128K, decode at depth, an agent session, answer quality.", 13,
       {{"decode", {}},
        {"concurrency", {}},
        {"prefill", {{"depths", "2000,8000,32000,64000,128000"}}},
        {"deep-decode", {{"depths", "32000,109000"}}},
        {"turns", {}},
        {"structured", {}},
        {"quality", {}}}},
      {"full", "Everything, to the endpoint's context.", 62,
       {{"decode", {{"passes", "5"}}},
        {"concurrency", {{"requests", "48"}}},
        {"prefill", {}},
        {"deep-decode", {}},
        {"turns", {}},
        {"burst", {}},
        {"late-arrival", {}},
        {"mixed", {}},
        {"needle", {}},
        {"repeat", {}},
        {"structured", {}},
        {"arrivals", {}},
        {"vision", {}},
        {"quality", {}},
        {"prefill-decode", {}},
        {"long-agents", {}},
        {"chat-sessions", {}},
        {"rag", {}},
        {"autocomplete", {}},
        {"tool-calls", {}},
        {"long-reply", {}}}},
      {"long", "Long contexts: prefill to 250K, decode at depth, the needle, a long agent session.", 13,
       {{"prefill", {{"depths", "32000,64000,128000,250000"}}},
        {"deep-decode", {}},
        {"needle", {}},
        {"turns", {{"start_tokens", "64000"}}}}},
      {"load", "Many requests at once: the concurrency sweep, arrivals at a rate, a late long prompt, a burst on one document, mixed depths.", 12,
       {{"concurrency", {{"requests", "48"}}}, {"arrivals", {}}, {"late-arrival", {}}, {"burst", {}}, {"mixed", {}}}},
      {"agents", "Coding agents on long contexts: 1, 2 and 4 agents each on a document sized from the KV pool, a prompt read beside generating agents, a session.", 25,
       {{"long-agents", {}}, {"prefill-decode", {}}, {"turns", {{"start_tokens", "64000"}}}}},
      {"check", "Is the output still right: answer quality, the needle to the context's end, reply repeatability, schema output, image input, tool calls.", 7,
       {{"quality", {}}, {"needle", {}}, {"repeat", {}}, {"structured", {}}, {"vision", {}}, {"tool-calls", {}}}},
      {"apps", "Real-world applications: chat users with pauses against a latency target, answers from retrieved passages under load, editor autocomplete "
               "with overtaken requests, an agent's tool calls, one long reply.",
       7,
       {{"chat-sessions", {}}, {"rag", {}}, {"autocomplete", {}}, {"tool-calls", {}}, {"long-reply", {}}}},
  };
  return kSuites;
}

}  // namespace xenithium:bench
