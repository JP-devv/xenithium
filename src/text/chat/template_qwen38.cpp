// Xenithium · qwen38's chat_template.jinja rendered by hand (trim_blocks + lstrip_blocks): the qwen38 and
// qwen35 checkpoints ship it byte for byte.
#include "chat_internal.hpp"

namespace xenithium::text::chat::detail {

// ---- the template --------------------------------------------------------------------------------

constexpr std::string_view kToolsIntro = "# Tools\n\nYou have access to the following functions:\n\n<tools>";
constexpr std::string_view kToolsOutro =
    "\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO "
    "suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\n"
    "value_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second "
    "parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n"
    "<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner "
    "<function=...></function> block must be nested within <tool_call></tool_call> XML tags\n"
    "- Required parameters MUST be specified\n- You may provide optional reasoning for your "
    "function call in natural language BEFORE the function call, but NOT after\n- If there is "
    "no function call available, answer the question like normal with your current knowledge "
    "and do not tell the user about function calls\n</IMPORTANT>";
constexpr std::string_view kEffortXhigh =
    "Reasoning effort is set to xhigh. Please think carefully through the task, validate key "
    "assumptions, consider plausible alternatives, and prioritize correctness, consistency, and "
    "clarity in the final answer.";
constexpr std::string_view kEffortLow =
    "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to "
    "the conclusion without unnecessary elaboration.";

bool render_template_qwen38(const std::vector<Message> &msgs, const std::vector<Value> *tools,
                     const text::ChatOptions &opt, std::string &out, std::string &err) {
  using text::Flag;
  out.clear();
  // reasoning_effort|default('xhigh') replaces only an undefined value (a null one counts as undefined);
  // anything defined must be one of the three names, else the raise prints it with str.
  std::string_view instructions;
  if (opt.enable_thinking == Flag::unset || opt.enable_thinking == Flag::yes) {
    const Value *e = opt.reasoning_effort && !opt.reasoning_effort->is_null() ? opt.reasoning_effort : nullptr;
    if (!e) {
      instructions = kEffortXhigh;
    } else if (e->is_string() && (e->str == "xhigh" || e->str == "medium" || e->str == "low")) {
      instructions = e->str == "xhigh" ? kEffortXhigh : e->str == "low" ? kEffortLow : std::string_view();
    } else {
      std::string shown;
      py_str(*e, shown, true);
      return fail(err, "Unexpected reasoning effort " + shown +
                           ". Supported types are xhigh (default), medium, and low.");
    }
  }
  const bool first_system = msgs[0].role == "system";
  const std::string sys_full = first_system ? msgs[0].content() : std::string();
  const std::string_view sys = py_strip(sys_full);
  if (tools && !tools->empty()) {
    out += "<|im_start|>system\n";
    if (!instructions.empty()) out.append(instructions).append("\n\n");
    out += kToolsIntro;
    for (const auto &t : *tools) {
      out += '\n';
      json::write(t, out, json::Style::python);  // `tool | tojson`: json.dumps, keys unsorted
    }
    out += kToolsOutro;
    if (!sys.empty()) out.append("\n\n").append(sys);
    out += "<|im_end|>\n";
  } else if (!sys.empty()) {
    out += "<|im_start|>system\n";
    if (!instructions.empty()) out.append(instructions).append("\n\n");
    out.append(sys).append("<|im_end|>\n");
  } else if (!instructions.empty()) {
    out.append("<|im_start|>system\n").append(instructions).append("<|im_end|>\n");
  }
  // The last real user query, walking back: a user turn that is only a <tool_response> block is
  // a multi-step tool result, not a query. Reasoning after the query is always kept.
  const int n = static_cast<int>(msgs.size());
  int last_query = n - 1;
  bool multi_step = true;
  for (int i = n - 1; i >= 0 && multi_step; --i) {
    if (msgs[i].role != "user") continue;
    const std::string c = msgs[i].content();
    const std::string_view s = py_strip(c);
    if (!(starts_with(s, "<tool_response>") && ends_with(s, "</tool_response>"))) {
      multi_step = false;
      last_query = i;
    }
  }
  if (multi_step) return fail(err, "No user query found in messages.");
  const bool keep_thinking = opt.preserve_thinking == Flag::unset || opt.preserve_thinking == Flag::yes;
  for (int i = 0; i < n; ++i) {
    const Message &m = msgs[i];
    const std::string full = m.content();
    const std::string_view content = py_strip(full);
    if (m.role == "system") {
      if (i != 0) return fail(err, "System message must be at the beginning.");
    } else if (m.role == "user") {
      out.append("<|im_start|>user\n").append(content).append("<|im_end|>\n");
    } else if (m.role == "assistant") {
      std::string_view reasoning;
      if (m.reasoning && m.reasoning->is_string()) reasoning = py_strip(m.reasoning->str);
      if (keep_thinking || i > last_query)
        out.append("<|im_start|>assistant\n<think>\n").append(reasoning).append("\n</think>\n\n").append(content);
      else
        out.append("<|im_start|>assistant\n").append(content);
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
      // loop.previtem is a (truthy) message on every iteration but the first.
      if (i > 0 && msgs[i - 1].role != "tool") out += "<|im_start|>user";
      out.append("\n<tool_response>\n").append(content).append("\n</tool_response>");
      if (i == n - 1 || msgs[i + 1].role != "tool") out += "<|im_end|>\n";
    } else {
      return fail(err, "Unexpected message role.");
    }
  }
  if (opt.add_generation_prompt) {
    out += "<|im_start|>assistant\n";
    out += opt.enable_thinking == Flag::no ? "<think>\n\n</think>\n\n" : "<think>\n";
  }
  return true;
}

}  // namespace xenithium:text:chat:detail
