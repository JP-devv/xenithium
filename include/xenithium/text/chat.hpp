// A model family's chat format: the template that turns OpenAI messages into prompt text,
// and the parser that turns generated text back into reasoning, content and tool calls.
// The server and the core are written only against this interface; each arch provides its
// implementation (src/arch/<id>/chat.cpp) and one line in text:open_chat_format.
#pragma once

#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "xenithium/util/json.hpp"

namespace xenithium::text {

// A template variable as Jinja sees it: absent / undefined, the literal true, the literal
// false, or something else (a string, a number: defined, but neither `is true` nor `is false`).
enum class Flag : uint8_t { unset, yes, no, other };
Flag flag_of(const json::Value *v);  // nullptr and JSON null -> unset

struct ChatOptions {
  bool add_generation_prompt = true;
  Flag enable_thinking = Flag::unset;
  Flag preserve_thinking = Flag::unset;
  const json::Value *reasoning_effort = nullptr;  // as passed (string expected); nullptr = unset
  // The request's chat_template_kwargs object as sent (nullptr = none): a Jinja-rendered format passes every member to
  // the template, then the fields above over it.
  const json::Value *kwargs = nullptr;
  // The server has image input: image_url content parts render as the template's image placeholder (one per
  // image, in message order); false = they are refused (400).
  bool images = false;
};

// One piece of parsed output, in order.
// reasoning / content: text to append to that field.
// tool_start: a new tool call; text = the function name, index = 0, 1... per request.
// tool_args: that call's complete arguments (a JSON object text), once per call.
struct Delta {
  enum Kind : uint8_t { reasoning, content, tool_start, tool_args } kind = content;
  std::string text;
  int index = 0;
};

// Streaming output parser: fed the detokenized text as it is generated.
class OutputParser {
 public:
  virtual ~OutputParser() = default;
  virtual void feed(std::string_view text, std::vector<Delta> &out) = 0;
  virtual void finish(std::vector<Delta> &out) = 0;  // end of generation: release everything held back
  virtual int tool_calls() const = 0;  // tool calls emitted so far
  // Tool-call markup was seen (a call, possibly dropped: no tools, a malformed call); a
  // non-streaming result strips the content then (finish_content).
  virtual bool tool_markup() const { return tool_calls() > 0; }
};

class ChatFormat {
 public:
  virtual ~ChatFormat() = default;
  // messages: the request's `messages` array; tools: its `tools` array or nullptr. Applies the
  // request normalisation, then renders the template. false + err (the
  // message a 400 carries) when the template or the normalisation rejects the request.
  virtual bool render(const json::Value &messages, const json::Value *tools, const ChatOptions &opt,
                      std::string &out, std::string &err) const = 0;
  // Does the generated text start inside the reasoning block for this request? (the parser's
  // view of chat_template_kwargs.enable_thinking: Python truthiness, default true)
  virtual bool thinking(const json::Value *enable_thinking) const = 0;
  virtual std::unique_ptr<OutputParser> parser(bool thinking, bool tools) const = 0;
  // Same, given the request's `tools` array (nullptr = the request has none). Prefer this one:
  // each parsed argument is coerced to the type its parameter schema declares
  // ("5" -> 5 for an integer), which needs the schemas, not just whether tools exist.
  virtual std::unique_ptr<OutputParser> parser(bool thinking, const json::Value *tools) const {
    return parser(thinking, tools != nullptr);
  }
  // tool_choice "none" with tools: the tools are still rendered; calls in the output are
  // dropped (never returned) and the text around them is content, not stripped.
  virtual std::unique_ptr<OutputParser> parser_no_tool_calls(bool thinking) const { return parser(thinking, false); }
  // A whole (non-streamed) answer's content once the parser finished (tool_markup: the parser's
  // tool_markup): the adjustment a stream cannot know in advance.
  // qwen38: content before tool-call markup loses its leading whitespace.
  virtual void finish_content(std::string &content, bool tool_markup) const { (void)content, (void)tool_markup; }
  // tool_choice "none": true = the prompt is rendered without the tools (Flint), false = rendered with them (the
  // parser still drops calls).
  virtual bool exclude_tools_on_none() const { return false; }
};

// The chat format of an arch id (model:Arch:id); nullptr when the arch has none. Its renderer is the arch's
// built-in one (the checkpoint's template rendered by hand).
std::unique_ptr<ChatFormat> open_chat_format(std::string_view arch_id);

// The same with the renderer chosen by `source`:
// "auto" the checkpoint's own chat template (chat_template.jinja / tokenizer_config.json) through the Jinja
// interpreter (text/jinja.hpp); the built-in renderer when it has none or it does not parse;
// "builtin" the built-in renderer;
// "jinja" the checkpoint's template, nullptr when it has none or it does not parse;
// a path that template file through the interpreter (nullptr when it does not parse).
// The output parser, the request normalisation and the arch's options stay the built-in format's. note (optional):
// which renderer was chosen and why.
std::unique_ptr<ChatFormat> open_chat_format(std::string_view arch_id, const std::string &model_dir, std::string_view source = "auto",
                                             std::string *note = nullptr);

}  // namespace xenithium:text
