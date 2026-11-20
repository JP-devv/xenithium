// Xenithium · Nanbeige4.3's chat template (arch llama) rendered by hand.
#include "chat_internal.hpp"

namespace xenithium::text::chat::detail {

// ---- Nanbeige4.3's template (tokenizer_config.json chat_template, arch llama) ------------------------
// The same ChatML turns, <think> blocks and XML tool calls as above, with the differences its template spells out: a
// default system prompt, no reasoning-effort line, its own tools preamble, system / user / tool text untrimmed, every
// assistant turn with a think block (an empty one before the last query when preserve_thinking is false), reasoning
// split out of the content at </think> when the message has none, any other role rendered as a plain turn, and no
// error without a user query. tool_call_format 'json' (a template kwarg) is not offered: the XML default only.

constexpr std::string_view kNbDefaultSystem =
    "\xe4\xbd\xa0\xe6\x98\xaf\xe5\x8d\x97\xe5\x8c\x97\xe9\x98\x81\xef\xbc\x8c\xe4\xb8\x80\xe6\xac\xbe\xe7\x94\xb1"
    "BOSS\xe7\x9b\xb4\xe8\x81\x98\xe8\x87\xaa\xe4\xb8\xbb\xe7\xa0\x94\xe5\x8f\x91\xe5\xb9\xb6\xe8\xae\xad\xe7\xbb\x83"
    "\xe7\x9a\x84\xe4\xb8\x93\xe4\xb8\x9a\xe5\xa4\xa7\xe8\xaf\xad\xe8\xa8\x80\xe6\xa8\xa1\xe5\x9e\x8b\xe3\x80\x82";
constexpr std::string_view kNbToolSystem =
    "\xe4\xbd\xa0\xe6\x98\xaf\xe4\xb8\x80\xe4\xbd\x8d\xe5\xb7\xa5\xe5\x85\xb7\xe5\x87\xbd\xe6\x95\xb0\xe8\xb0\x83\xe7"
    "\x94\xa8\xe4\xb8\x93\xe5\xae\xb6\xef\xbc\x8c\xe4\xbd\xa0\xe4\xbc\x9a\xe5\xbe\x97\xe5\x88\xb0\xe4\xb8\x80\xe4\xb8"
    "\xaa\xe9\x97\xae\xe9\xa2\x98\xe5\x92\x8c\xe4\xb8\x80\xe7\xbb\x84\xe5\x8f\xaf\xe8\x83\xbd\xe7\x9a\x84\xe5\xb7\xa5"
    "\xe5\x85\xb7\xe5\x87\xbd\xe6\x95\xb0\xe3\x80\x82\xe6\xa0\xb9\xe6\x8d\xae\xe9\x97\xae\xe9\xa2\x98\xef\xbc\x8c\xe4"
    "\xbd\xa0\xe9\x9c\x80\xe8\xa6\x81\xe8\xbf\x9b\xe8\xa1\x8c\xe4\xb8\x80\xe4\xb8\xaa\xe6\x88\x96\xe5\xa4\x9a\xe4\xb8"
    "\xaa\xe5\x87\xbd\xe6\x95\xb0/\xe5\xb7\xa5\xe5\x85\xb7\xe8\xb0\x83\xe7\x94\xa8\xe4\xbb\xa5\xe5\xae\x9e\xe7\x8e\xb0"
    "\xe7\x9b\xae\xe7\x9a\x84\xef\xbc\x8c\xe8\xaf\xb7\xe5\xb0\xbd\xe9\x87\x8f\xe5\xb0\x9d\xe8\xaf\x95\xe6\x8e\xa2\xe7"
    "\xb4\xa2\xe9\x80\x9a\xe8\xbf\x87\xe5\xb7\xa5\xe5\x85\xb7\xe8\xa7\xa3\xe5\x86\xb3\xe9\x97\xae\xe9\xa2\x98\xe3\x80"
    "\x82\n\xe5\xa6\x82\xe6\x9e\x9c\xe6\xb2\xa1\xe6\x9c\x89\xe4\xb8\x80\xe4\xb8\xaa\xe5\x87\xbd\xe6\x95\xb0\xe5\x8f"
    "\xaf\xe4\xbb\xa5\xe4\xbd\xbf\xe7\x94\xa8\xef\xbc\x8c\xe8\xaf\xb7\xe7\x9b\xb4\xe6\x8e\xa5\xe4\xbd\xbf\xe7\x94\xa8"
    "\xe8\x87\xaa\xe7\x84\xb6\xe8\xaf\xad\xe8\xa8\x80\xe5\x9b\x9e\xe5\xa4\x8d\xe7\x94\xa8\xe6\x88\xb7\xe3\x80\x82\n\xe5"
    "\xa6\x82\xe6\x9e\x9c\xe7\xbb\x99\xe5\xae\x9a\xe7\x9a\x84\xe9\x97\xae\xe9\xa2\x98\xe7\xbc\xba\xe5\xb0\x91\xe5\x87"
    "\xbd\xe6\x95\xb0\xe6\x89\x80\xe9\x9c\x80\xe7\x9a\x84\xe5\x8f\x82\xe6\x95\xb0\xef\xbc\x8c\xe8\xaf\xb7\xe4\xbd\xbf"
    "\xe7\x94\xa8\xe8\x87\xaa\xe7\x84\xb6\xe8\xaf\xad\xe8\xa8\x80\xe8\xbf\x9b\xe8\xa1\x8c\xe6\x8f\x90\xe9\x97\xae\xef"
    "\xbc\x8c\xe5\x90\x91\xe7\x94\xa8\xe6\x88\xb7\xe8\xaf\xa2\xe9\x97\xae\xe5\xbf\x85\xe8\xa6\x81\xe4\xbf\xa1\xe6\x81"
    "\xaf\xe3\x80\x82\n\xe5\xa6\x82\xe6\x9e\x9c\xe8\xb0\x83\xe7\x94\xa8\xe7\xbb\x93\xe6\x9e\x9c\xe5\xb7\xb2\xe7\xbb\x8f"
    "\xe8\xb6\xb3\xe5\xa4\x9f\xe5\x9b\x9e\xe7\xad\x94\xe7\x94\xa8\xe6\x88\xb7\xe9\x97\xae\xe9\xa2\x98\xef\xbc\x8c\xe8"
    "\xaf\xb7\xe5\xaf\xb9\xe5\x8e\x86\xe5\x8f\xb2\xe7\xbb\x93\xe6\x9e\x9c\xe8\xbf\x9b\xe8\xa1\x8c\xe6\x80\xbb\xe7\xbb"
    "\x93\xef\xbc\x8c\xe4\xbd\xbf\xe7\x94\xa8\xe8\x87\xaa\xe7\x84\xb6\xe8\xaf\xad\xe8\xa8\x80\xe5\x9b\x9e\xe5\xa4\x8d"
    "\xe7\x94\xa8\xe6\x88\xb7\xe3\x80\x82";
constexpr std::string_view kNbToolsIntro =
    "# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function "
    "signatures within <tools></tools> XML tags:\n<tools>";
constexpr std::string_view kNbToolsOutro =
    "\n</tools>\n\nFor each function call, output the function name and arguments within the following XML format:\n"
    "<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n"
    "<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n"
    "</parameter>\n</function>\n</tool_call><|im_end|>\n";

// Python str.rstrip('\n') / lstrip('\n')
std::string_view strip_nl_right(std::string_view s) {
  while (!s.empty() && s.back() == '\n') s.remove_suffix(1);
  return s;
}
std::string_view strip_nl_left(std::string_view s) {
  while (!s.empty() && s.front() == '\n') s.remove_prefix(1);
  return s;
}

bool render_template_nanbeige(const std::vector<Message> &msgs, const std::vector<Value> *tools, const text::ChatOptions &opt,
                              std::string &out, std::string &err) {
  using text::Flag;
  out.clear();
  const bool first_system = msgs[0].role == "system";
  if (tools && !tools->empty()) {
    out += "<|im_start|>system\n";
    if (first_system) out.append(msgs[0].content()).append("\n\n");
    else out += kNbToolSystem;
    out += kNbToolsIntro;
    for (const auto &t : *tools) {
      out += '\n';
      json::write(t, out, json::Style::python);  // `tool | tojson`
    }
    out += kNbToolsOutro;
  } else if (first_system) {
    out.append("<|im_start|>system\n").append(msgs[0].content()).append("<|im_end|>\n");
  } else {
    out.append("<|im_start|>system\n").append(kNbDefaultSystem).append("<|im_end|>\n");
  }
  // The last real user query (a user turn that is a whole <tool_response> block is a tool result); none: the last index.
  const int n = static_cast<int>(msgs.size());
  int last_query = n - 1;
  for (int i = n - 1; i >= 0; --i) {
    if (msgs[i].role != "user") continue;
    const std::string c = msgs[i].content();
    if (!(starts_with(c, "<tool_response>") && ends_with(c, "</tool_response>"))) {
      last_query = i;
      break;
    }
  }
  const bool drop_old = opt.preserve_thinking == Flag::no;
  for (int i = 0; i < n; ++i) {
    const Message &m = msgs[i];
    const std::string full = m.content();
    if (m.role == "system") {
      if (i != 0) return fail(err, "System message must be at the beginning.");
    } else if (m.role == "assistant") {
      std::string_view content = full, reasoning;
      if (m.reasoning && m.reasoning->is_string()) {
        reasoning = m.reasoning->str;
      } else if (full.find("</think>") != std::string::npos) {
        std::string_view head = std::string_view(full).substr(0, full.find("</think>"));
        head = strip_nl_right(head);
        const size_t open = head.rfind("<think>");
        reasoning = strip_nl_left(open == std::string_view::npos ? head : head.substr(open + 7));
        content = strip_nl_right(strip_nl_left(std::string_view(full).substr(full.rfind("</think>") + 8)));
      }
      content = py_strip(content);
      reasoning = py_strip(reasoning);
      if (drop_old && i < last_query)
        out.append("<|im_start|>assistant\n<think>\n\n</think>\n\n").append(content);
      else
        out.append("<|im_start|>assistant\n<think>\n").append(reasoning).append("\n</think>\n\n").append(content);
      for (size_t k = 0; k < m.calls.size(); ++k) {
        const ToolCall &c = m.calls[k];
        if (k > 0) out += "\n<tool_call>\n<function=";
        else out += content.empty() ? "<tool_call>\n<function=" : "\n\n<tool_call>\n<function=";
        out.append(c.name).append(">\n");
        for (const auto &kv : c.args.obj) {
          out.append("<parameter=").append(kv.first).append(">\n");
          if (kv.second.is_string()) out += kv.second.str;
          else json::write(kv.second, out, json::Style::python);
          out += "\n</parameter>\n";
        }
        out += "</function>\n</tool_call>";
      }
      out += "<|im_end|>\n";
    } else if (m.role == "tool") {
      if (i > 0 && msgs[i - 1].role != "tool") out += "<|im_start|>user";
      out.append("\n<tool_response>\n").append(full).append("\n</tool_response>");
      if (i == n - 1 || msgs[i + 1].role != "tool") out += "<|im_end|>\n";
    } else if (!m.role.empty()) {
      out.append("<|im_start|>").append(m.role).append("\n").append(full).append("<|im_end|>\n");
    }
  }
  if (opt.add_generation_prompt) {
    out += "<|im_start|>assistant\n";
    out += opt.enable_thinking == Flag::no ? "<think>\n\n</think>\n\n" : "<think>\n";
  }
  return true;
}

}  // namespace xenithium:text:chat:detail
