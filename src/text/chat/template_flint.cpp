// Xenithium · Recursive Flint's chat_template.jinja rendered by hand.
#include "chat_internal.hpp"

namespace xenithium::text::chat::detail {

// ---- Flint's chat_template.jinja ------------------------------------------------------------------------
// Role markers with no end-of-turn token (<|system|> <|developer|> <|user|> <|assistant|> <|observation|>), user /
// system / developer / tool text through flint_text (NUL and think tags removed, the template's own markers
// HTML-escaped), the tools preamble before the first non-system message (or at the end when there is none), assistant
// turns as <think>reasoning</think> + content + GLM-4.7-style calls, and a bare <|assistant|> as the generation prompt.
std::string flint_text(std::string_view t) {
  auto replace_all = [](std::string s, std::string_view from, std::string_view to) {
    std::string r;
    for (size_t at = 0;;) {
      const size_t k = s.find(from, at);
      if (k == std::string::npos) {
        r.append(s, at, std::string::npos);
        return r;
      }
      r.append(s, at, k - at).append(to);
      at = k + from.size();
    }
  };
  std::string s(t);
  s = replace_all(std::move(s), std::string_view("\0", 1), "");
  s = replace_all(std::move(s), "<think>", "");
  s = replace_all(std::move(s), "</think>", "");
  for (std::string_view tok : {"begin_of_text", "end_of_text", "system", "developer", "user", "assistant", "observation"})
    s = replace_all(std::move(s), "<|" + std::string(tok) + "|>", "&lt;|" + std::string(tok) + "|&gt;");
  return s;
}

bool render_template_flint(const std::vector<Message> &msgs, const std::vector<Value> *tools, const text::ChatOptions &opt, std::string &out,
                           std::string &err) {
  (void)err;
  out.clear();
  std::string preamble;
  if (tools && !tools->empty()) {
    preamble =
        "<|system|>\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function "
        "signatures within <tools></tools> XML tags:\n<tools>";
    for (const auto &t : *tools) {  // no separator: the loop's newlines are all trimmed (trim_blocks + `{%-`)
      const Value *fn = t.get("function");
      json::write(fn ? *fn : t, preamble, json::Style::python);  // `fn | tojson(ensure_ascii=False)`
    }
    preamble +=
        "</tools>\n\nFor each function call, output the function name and arguments within the following XML format:\n"
        "<tool_call>{function-name}<arg_key>{arg-key-1}</arg_key><arg_value>{arg-value-1}</arg_value>...</tool_call>\n";
  }
  bool pending = true;
  for (const Message &m : msgs) {
    if (pending && m.role != "system") {
      out += preamble;
      pending = false;
    }
    const std::string full = m.joined("\n");
    if (m.role == "system" || m.role == "developer" || m.role == "user") {
      out.append("<|").append(m.role).append("|>").append(flint_text(full));
    } else if (m.role == "assistant") {
      out += "<|assistant|>";
      if (m.reasoning && m.reasoning->is_string() && !m.reasoning->str.empty()) out.append("<think>").append(m.reasoning->str).append("</think>");
      out += full;
      for (const ToolCall &c : m.calls) {
        out.append("<tool_call>").append(c.name);
        for (const auto &kv : c.args.obj) {
          out.append("<arg_key>").append(kv.first).append("</arg_key><arg_value>");
          if (kv.second.is_string()) out += kv.second.str;
          else json::write(kv.second, out, json::Style::python);
          out += "</arg_value>";
        }
        out += "</tool_call>";
      }
    } else if (m.role == "tool") {
      out.append("<|observation|><tool_response>").append(flint_text(full)).append("</tool_response>");
    }
  }
  if (pending) out += preamble;
  if (opt.add_generation_prompt) out += "<|assistant|>";
  return true;
}

}  // namespace xenithium:text:chat:detail
