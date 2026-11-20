// Scenario "tool-calls": an agent's function calling. The request carries a set of tools (read a file, search the code,
// run tests, the weather, a ticket) and a task that needs exactly one of them - or none. The reply must stream a call
// of the right function with arguments that parse and hold the task's values, or answer directly where no tool is
// needed. Then the loop goes on as an agent's does: the call's result goes back as a "tool" turn and the model answers
// from it. A model or a server whose tool handling breaks (a call printed as text, arguments cut short, a call where
// none was asked for, a result the template drops) fails here while its decode rate looks fine.
#include <algorithm>
#include <array>

#include "scenario.hpp"
#include "xenithium/util/json.hpp"

namespace xenithium::bench {
namespace {

namespace json = xenithium::json;

constexpr ParamSpec kParams[] = {
    {"tasks", "", "tasks to run, comma separated (read-file, search-code, run-tests, weather, ticket, no-tool); empty = all"},
    {"reply_tokens", "512", "the most tokens a reply may take"},
};

constexpr std::string_view kSystem =
    "You are a coding assistant with access to tools. Call a tool when the task needs one, with all of its required arguments; answer directly when it does not.";

constexpr std::string_view kTools = R"([
{"type":"function","function":{"name":"read_file","description":"Read a text file of the repository.",
  "parameters":{"type":"object","properties":{"path":{"type":"string","description":"The path from the repository's root"}},"required":["path"]}}},
{"type":"function","function":{"name":"search_code","description":"Search the repository's source code for a text.",
  "parameters":{"type":"object","properties":{"query":{"type":"string"},"max_results":{"type":"integer"}},"required":["query"]}}},
{"type":"function","function":{"name":"run_tests","description":"Run a test target and return its output.",
  "parameters":{"type":"object","properties":{"target":{"type":"string","description":"The test target's name"}},"required":["target"]}}},
{"type":"function","function":{"name":"get_weather","description":"The current weather in a city.",
  "parameters":{"type":"object","properties":{"city":{"type":"string"},"unit":{"type":"string","enum":["celsius","fahrenheit"]}},"required":["city","unit"]}}},
{"type":"function","function":{"name":"create_ticket","description":"File a ticket in the issue tracker.",
  "parameters":{"type":"object","properties":{"title":{"type":"string"},"priority":{"type":"string","enum":["low","medium","high"]},
    "labels":{"type":"array","items":{"type":"string"}}},"required":["title","priority"]}}}
])";

// One argument the call must carry: its value (a string, or an array's items) holds `holds, whatever the case.
struct ArgumentCheck {
  std::string_view key;
  std::string_view holds;
};
constexpr size_t kMaxChecks = 4;

struct ToolTask {
  std::string_view id;
  std::string_view prompt;
  std::string_view tool;  // the function the task needs; empty = none, answer directly
  std::array<ArgumentCheck, kMaxChecks> checks;  // an empty key ends them
  std::string_view result;  // what the tool returns: the "tool" turn of the second request
  std::string_view answer_holds;  // the final answer is right when it holds this
};

constexpr std::array kTasks = {
    ToolTask{"read-file", "Open the file src/server/http.cpp and tell me in two sentences what it does.", "read_file", {{{"path", "src/server/http.cpp"}}},
             "// http.cpp: the server's HTTP/1.1 layer. It parses requests, keeps connections alive and streams server-sent events to clients.\n"
             "#include \"http.hpp\"\n\nnamespace xenithium::server {\n// ... 640 lines ...\n}",
             "HTTP"},
    ToolTask{"search-code", "Where is the function parse_endpoint defined? Search the code for it.", "search_code", {{{"query", "parse_endpoint"}}},
             "tools/bench/client.cpp:200: std::optional<Endpoint> parse_endpoint(std::string_view url, std::string &err) {", "client.cpp"},
    ToolTask{"run-tests", "Run the tokenizer's tests and tell me whether they pass.", "run_tests", {{{"target", "tokenizer"}}},
             "test-tokenizer: 412 checks, 3 failed (byte fallback for U+1F600, a lone surrogate, an empty merge list)", "fail"},
    ToolTask{"weather", "What is the weather in Munich right now? I want the temperature in Celsius.", "get_weather", {{{"city", "Munich"}, {"unit", "celsius"}}},
             R"({"city":"Munich","unit":"celsius","temperature":14,"conditions":"light rain"})", "14"},
    ToolTask{"ticket", "File a high-priority ticket titled \"Login page times out\" with the labels auth and frontend.", "create_ticket",
             {{{"title", "Login page times out"}, {"priority", "high"}, {"labels", "auth"}, {"labels", "frontend"}}}, R"({"ticket":"OPS-4127","status":"open"})", "OPS-4127"},
    ToolTask{"no-tool", "What is 17 times 23? Answer with the number.", {}, {}, {}, "391"},
};

constexpr size_t kShownChars = 120;  // of a reply or an argument text quoted in a note

// An argument's value as text: a string as it is, an array's items one after the other, anything else as JSON.
std::string argument_text(const json::Value &value) {
  if (value.is_string()) return value.str;
  if (!value.is_array()) return json::dump(value);
  std::string items;
  for (const json::Value &item : value.arr) items += argument_text(item) + " ";
  return items;
}

// Whether the call's arguments parse as a JSON object that holds every checked value; `why` says what is off.
bool arguments_hold(const ToolCall &call, const ToolTask &task, std::string &why) {
  json::Value args;
  if (!json::parse(call.arguments, args) || !args.is_object()) return why = "the arguments are not a JSON object: " + head_of(call.arguments, kShownChars), false;
  for (const ArgumentCheck &check : task.checks) {
    if (check.key.empty()) break;
    const json::Value *value = args.get(check.key);
    if (!value) return why = "no argument \"" + std::string(check.key) + "\" in " + head_of(call.arguments, kShownChars), false;
    if (!contains_text(argument_text(*value), check.holds))
      return why = "\"" + std::string(check.key) + "\" is " + head_of(json::dump(*value), kShownChars) + ", not holding \"" + std::string(check.holds) + "\"", false;
  }
  return true;
}

// The task asked as an agent asks it: the system prompt, the tools, the user's task.
ChatRequest task_turn(const Session &session, const ToolTask &task, int reply_tokens) {
  ChatRequest request = session.greedy({}, reply_tokens);
  request.messages = {{"system", std::string(kSystem)}, {"user", std::string(task.prompt)}};
  request.ignore_eos = false;  // the reply ends with the call, or with the answer
  request.thinking = session.thinking() == Thinking::on ? Thinking::on : Thinking::off;
  request.tools = std::string(kTools);
  request.tool_choice = "auto";
  return request;
}

// The agent's next request: the conversation so far with the assistant's calls, and the tool's result for each.
ChatRequest result_turn(ChatRequest request, const Sample &called, const ToolTask &task) {
  Message assistant{"assistant", called.text};
  assistant.tool_calls = called.tool_calls;
  for (size_t i = 0; i < assistant.tool_calls.size(); ++i)
    if (assistant.tool_calls[i].id.empty()) assistant.tool_calls[i].id = "call_" + std::to_string(i);  // a server that streams no id
  request.messages.push_back(assistant);
  for (const ToolCall &call : assistant.tool_calls) {
    Message result{"tool", std::string(task.result)};
    result.tool_call_id = call.id;
    request.messages.push_back(std::move(result));
  }
  return request;
}

std::string call_text(const ToolCall &call) { return call.name + "(" + head_of(call.arguments, kShownChars) + ")"; }

class ToolCalls final : public Scenario {
 public:
  std::string_view id() const override { return "tool-calls"; }
  std::string_view title() const override { return "An agent's tool calls (function calling, greedy)"; }
  std::string_view about() const override {
    return "Tasks that need one of five tools (or none), the tools sent as an agent sends them: whether the right function was called with arguments that hold the "
           "task's values, the time until the call is complete, and whether the answer after the tool's result uses it.";
  }
  std::span<const ParamSpec> params() const override { return kParams; }

  ScenarioResult run(Session &session, const Params &params) const override {
    ScenarioResult result;
    const int reply_tokens = static_cast<int>(params.integer("reply_tokens"));
    const std::vector<std::string> only = params.list("tasks");
    int asked = 0, all_right = 0;
    for (const ToolTask &task : kTasks) {
      if (!only.empty() && std::ranges::find(only, task.id) == only.end()) continue;
      const ChatRequest request = task_turn(session, task, reply_tokens);
      const Sample called = session.client().chat(request);
      if (result.requests == 0 && refused_by_endpoint(called)) {  // no function calling there: skipped, not failed
        result.rows.push_back(Row::note_only("skipped: the endpoint refuses tools (" + head_of(called.error, kShownChars) + ")"));
        return result;
      }
      if (!tally(result, called)) continue;
      Row row = Row::with_labels({{"task", std::string(task.id)}});
      row.metric(std::string(metric::kTtft), "ms", Better::lower).samples.push_back(called.ttft_ms);
      row.metric("reply", "ms", Better::lower).samples.push_back(called.wall_ms);  // until the call (or the answer) is complete
      const bool right = asks_right(session, request, called, task, row, result);
      ++asked, all_right += right;
      session.progress("tool-calls: " + std::string(task.id) + (right ? " right" : " wrong"));
      result.rows.push_back(std::move(row));
    }
    if (asked > 0) {
      Row all = Row::with_labels({{"task", "all"}});
      all.metric("all right", "%", Better::higher).samples.push_back(100.0 * all_right / asked);
      result.rows.push_back(std::move(all));
    }
    result.charts = {
        {ChartKind::dots, "The right function called (none where none was needed)", "task", {"right call"}, false},
        {ChartKind::dots, "The answer right after the tool's result", "task", {"answer right"}, false},
        {ChartKind::bars, "Time until the call is complete, and the answer's TTFT after the result", "task", {"reply", "answer TTFT"}, false},
    };
    return result;
  }

 private:
  // Grades the first reply and, after a right call, sends the tool's result and grades the answer. True when every
  // check of the task passed.
  static bool asks_right(Session &session, const ChatRequest &request, const Sample &called, const ToolTask &task, Row &row, ScenarioResult &result) {
    const std::string id(task.id);
    if (task.tool.empty()) {  // a question no tool helps with
      const bool no_call = called.tool_calls.empty(), answered = no_call && contains_text(called.text, task.answer_holds);
      row.flag("right call").samples.push_back(no_call);
      row.flag("answer right").samples.push_back(answered);
      if (!no_call) row.notes.push_back(id + ": called " + call_text(called.tool_calls.front()) + " though no tool was needed");
      else if (!answered) row.notes.push_back(id + ": the answer did not hold \"" + std::string(task.answer_holds) + "\": " + head_of(called.text, kShownChars));
      return answered;
    }
    const bool right_call = !called.tool_calls.empty() && called.tool_calls.front().name == task.tool;
    row.flag("right call").samples.push_back(right_call);
    if (!right_call) {
      row.notes.push_back(id + (called.tool_calls.empty() ? ": no call, the reply began: " + head_of(called.text, kShownChars)
                                                          : ": called " + call_text(called.tool_calls.front()) + " instead of " + std::string(task.tool)));
      row.flag("answer right").samples.push_back(0);
      return false;
    }
    std::string why;
    const bool valid = arguments_hold(called.tool_calls.front(), task, why);
    row.flag("valid arguments").samples.push_back(valid);
    if (!valid) row.notes.push_back(id + ": " + why);
    const Sample answer = session.client().chat(result_turn(request, called, task));
    if (!tally(result, answer)) return false;
    row.metric("answer TTFT", "ms", Better::lower).samples.push_back(answer.ttft_ms);
    const bool used = contains_text(answer.text, task.answer_holds);
    row.flag("answer right").samples.push_back(used);
    if (!used) row.notes.push_back(id + ": the answer after the result did not hold \"" + std::string(task.answer_holds) + "\": " + head_of(answer.text, kShownChars));
    return valid && used;
  }
};

}  // namespace

std::unique_ptr<Scenario> make_tool_calls() { return std::make_unique<ToolCalls>(); }

}  // namespace xenithium:bench
