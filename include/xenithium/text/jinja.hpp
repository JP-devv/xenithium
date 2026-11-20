// A Jinja2 interpreter for checkpoints' chat templates (chat_template.jinja / tokenizer_config.json chat_template),
// with the semantics of the environment chat templates are written for: an immutable sandboxed environment
// (trim_blocks, lstrip_blocks, loop controls, no trailing newline kept), its `tojson` (json.dumps with
// ensure_ascii=False by default, indent / separators / sort_keys), `raise_exception` and `strftime_now, Python value
// semantics (str / repr / truthiness / comparisons), the default Undefined (prints as "", falsy, iterates as
// empty; attribute access on it is an error).
// The subset is what chat templates use: text, {{ }}, {% %}, {# #}, whitespace control (- / +), raw blocks; if / elif /
// else, for (unpacking, `if` filter, else, loop.*), set (names, namespace attributes, tuple unpacking, block set),
// macro (defaults, keyword arguments), break / continue, filter blocks, generation blocks (transparent); expressions
// with literals, lists / tuples / dicts, attribute / item / slice access, calls, filters, tests, the arithmetic,
// comparison, `in, `~, and / or / not operators and the conditional expression; the str / dict methods chat
// templates call. Anything else is a parse error (the caller then falls back to the arch's built-in renderer).
#pragma once

#include <memory>
#include <string>
#include <string_view>

#include "xenithium/util/json.hpp"

namespace xenithium::text::jinja {

class Template {
 public:
  ~Template();
  // nullptr + err on a syntax error or an unsupported construct.
  static std::shared_ptr<const Template> parse(std::string_view source, std::string *err);
  // ctx: a JSON object whose members are the template's variables (JSON null = Python None; an absent member is
  // undefined). false + err on a runtime error; raise_exception(msg) gives err = msg exactly.
  bool render(const json::Value &ctx, std::string &out, std::string &err) const;
  // Does a for loop iterate over a message's content (`for x in m.content, `m['content'], or a name `content`)?
  // Content is then passed as a list of {type: text, text} parts (the "openai" content format),
  // else as one string (text parts joined with "\n").
  bool iterates_content() const;

  struct Impl;

 private:
  explicit Template(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

// The chat template a checkpoint directory ships: chat_template.jinja, else tokenizer_config.json's chat_template (a
// string, or the entry named "default" of a list). Empty when there is none. Also fills `globals` (a JSON object) with
// tokenizer_config.json's special tokens as strings (bos_token, eos_token, unk_token, pad_token...), which
// every render receives.
std::string load_chat_template(const std::string &model_dir, json::Value *globals = nullptr);

}  // namespace xenithium:text:jinja
