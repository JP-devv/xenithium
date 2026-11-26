// Xenithium · the Responses API (POST /v1/responses) on top of the chat path. Stateless: a request is
// translated into the chat request it means (instructions -> a system message, input items -> messages, flat function
// tools -> chat tools, text.format -> response_format, reasoning.effort, max_output_tokens), served by the chat
// machinery (template, images, parsers, structured output), and its output written back as Responses items: a
// reasoning item, a message with one output_text part, function_call items. Streaming emits the API's typed
// events (response.created... response.completed), one output item open at a time. Nothing is stored:
// previous_response_id, background and GET /v1/responses/{id} are refused.
#pragma once

#include <string>
#include <string_view>
#include <vector>

#include "xenithium/util/json.hpp"

namespace xenithium::server::responses {

// The Responses request -> the chat completions request it means. false + err / param (a 400).
bool to_chat(const json::Value &req, json::Value &chat, std::string &err, std::string &param);

struct Call {
  std::string call_id, name, arguments;
};

// The output items of a finished response (non-streaming): reasoning (when not empty), the message (when there is
// content), one function_call per call. lp_items: the output_text logprobs as a JSON array's items (no brackets), or
// empty for none; with_lp: write a logprobs member at all.
json::Value output_items(const std::string &reasoning, const std::string &content, const std::string &lp_items, bool with_lp,
                         const std::vector<Call> &calls);

// The fields of a response object besides its output and usage, echoed from the request.
struct Meta {
  std::string id, model;
  long long created = 0;
  json::Value request;  // the original Responses request (instructions, tools, tool_choice, text, reasoning...)
  double temperature = 1, top_p = 1;
  long long max_output_tokens = 0;
  int top_logprobs = 0;
};
// status: "in_progress", "completed" or "incomplete" (the token limit ended it). output / usage: null when absent.
std::string response_json(const Meta &m, const char *status, const json::Value *output, const json::Value *usage);
// cached: the input tokens the prefix cache served (input_tokens_details.cached_tokens)
json::Value usage(long long input, long long cached, long long output, long long reasoning);

// The streaming output: a state machine (one item open at a time; a change of kind or of tool call closes
// the open item). Each call appends SSE events ("event: <type>\ndata: {...}\n\n") to sse.
class Stream {
 public:
  void created(const Meta &m, std::string &sse);  // response.created + response.in_progress
  void reasoning(std::string_view text, std::string &sse);
  void content(std::string_view text, std::string_view lp_items, std::string &sse);
  void tool_start(std::string_view name, int index, std::string &sse);
  void tool_args(std::string_view args, std::string &sse);
  void close(std::string &sse);  // closes the open item, if any
  void completed(const Meta &m, const char *status, const json::Value &output, const json::Value &usage, std::string &sse);
  void error(int code, std::string_view message, std::string &sse);
  int sequence() const { return seq_; }

 private:
  enum class State { none, content, reasoning, tool } st_ = State::none;
  int seq_ = 0, output_index_ = 0, tool_index_ = -1;
  std::string item_id_, call_id_, name_, text_;
  bool tool_delta_ = false;
  void emit(const char *type, json::Value ev, std::string &sse);
  void open(State s, std::string &sse);
};

}  // namespace xenithium:server:responses
