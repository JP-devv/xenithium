// Xenithium · internal to src/text/chat/: the pieces the built-in chat formats share (request normalisation, the
// literal lexer of the output parsers, tool-call argument typing) and the entry points of each renderer and parser.
// Not installed: the public surface is xenithium/text/chat_formats.hpp.
#pragma once

#include <cstdint>
#include <memory>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "xenithium/text/chat.hpp"
#include "xenithium/util/json.hpp"

namespace xenithium::text::chat::detail {

using json::Type;
using json::Value;
inline constexpr size_t npos = std::string_view::npos;

// ---- the normalised conversation (request.cpp) -------------------------------------------------------

struct ToolCall {
  std::string name;
  Value args;  // always an object after normalisation
  std::string id;  // the request's call id (templates that pair tool results with their calls read it)
};

// An image content part's text: what the Qwen-VL templates print for it (the server expands the pad token into
// the image's tokens).
inline constexpr std::string_view kImagePart = "<|vision_start|><|image_pad|><|vision_end|>";
// A video content part's text (the server expands the pad token into the frames' timestamps and tokens).
inline constexpr std::string_view kVideoPart = "<|vision_start|><|video_pad|><|vision_end|>";

struct Message {
  std::string role;
  std::vector<std::string> parts;  // text parts in order; a string content is one part
  std::vector<uint8_t> image;  // per part: 1 = an image part (its text is kImagePart), 2 = a video part (kVideoPart);
                                   // empty = no media
  const Value *reasoning = nullptr;
  std::vector<ToolCall> calls;
  const Value *tool_call_id = nullptr;  // a tool message's tool_call_id, as sent

  // What the template's render_content prints: the parts with no separator.
  std::string content() const {
    std::string s;
    for (const auto &p : parts) s += p;
    return s;
  }
  std::string joined(std::string_view sep) const {
    std::string s;
    for (size_t k = 0; k < parts.size(); ++k) (s += k ? sep : "") += parts[k];
    return s;
  }
};

// ---- text helpers + request normalisation (request.cpp) ----------------------------------------------

size_t utf8_next(std::string_view s, size_t i, uint32_t &cp);
bool py_space(uint32_t c);
size_t skip_space(std::string_view s, size_t i);
size_t last_nonspace_end(std::string_view s);
bool all_space(std::string_view s);
std::string_view py_strip(std::string_view s);
bool starts_with(std::string_view s, std::string_view p);
bool ends_with(std::string_view s, std::string_view p);
std::string ascii_lower(std::string_view s);
void py_repr_str(std::string_view s, std::string &out);
void py_str(const Value &v, std::string &out, bool top);
bool lax_bool(const Value &v, bool &b);
bool fail(std::string &err, std::string msg);
bool content_parts(const Value &m, const std::string &role, const std::string &where, Message &out, bool images,
                   std::string &err);
void restore_nonfinite(Value &v);
bool py_json_loads(std::string_view s, Value &out);
bool tool_calls(const Value &m, const std::string &where0, Message &out, std::string &err);
// images: image_url parts become image parts (the server has image input); false = a 400.
bool normalise_messages(const Value &messages, std::vector<Message> &out, std::string &err, bool keep_developer = false, bool images = false);
bool normalise_tools(const Value &tools, std::vector<Value> &out, std::string &err);

// ---- built-in renderers (template_*.cpp) -------------------------------------------------------------

bool render_template_qwen38(const std::vector<Message> &msgs, const std::vector<Value> *tools,
                     const text::ChatOptions &opt, std::string &out, std::string &err);
bool render_template_nanbeige(const std::vector<Message> &msgs, const std::vector<Value> *tools, const text::ChatOptions &opt,
                              std::string &out, std::string &err);
bool render_template_flint(const std::vector<Message> &msgs, const std::vector<Value> *tools, const text::ChatOptions &opt, std::string &out,
                           std::string &err);

// ---- output parser ---------------------------------------------------------------------------------

// The literals of this grammar, plus the tokenizer's special tokens, which are dropped
// wherever they appear (this checkpoint's all_special_tokens; <|im_start|> is not one).
enum Lit : int8_t {
  kText = -1, kThinkStart, kThinkEnd, kToolStart, kToolEnd, kFuncPrefix, kFuncEnd,
  kParamStart, kParamEnd, kCloseAngle, kSpecial,
  kArgKeyStart, kArgKeyEnd, kArgValStart, kArgValEnd  // the Flint (GLM-4.7-style) tool-call tags
};
struct Literal {
  std::string_view text;
  Lit kind;
};
inline constexpr Literal kLiterals[] = {
    {"<think>", kThinkStart},     {"</think>", kThinkEnd},        {"<tool_call>", kToolStart},
    {"</tool_call>", kToolEnd},   {"<function=", kFuncPrefix},    {"</function>", kFuncEnd},
    {"<parameter=", kParamStart}, {"</parameter>", kParamEnd},    {">", kCloseAngle},
    {"<|im_end|>", kSpecial},     {"<|endoftext|>", kSpecial},    {"<|audio_start|>", kSpecial},
    {"<|audio_end|>", kSpecial},  {"<|audio_pad|>", kSpecial},    {"<|image_pad|>", kSpecial},
    {"<|video_pad|>", kSpecial},  {"<|vision_start|>", kSpecial}, {"<|vision_end|>", kSpecial},
};

// Flint's grammar: think / tool-call / argument tags; the tokenizer's special tokens never reach the parser (the
// detokenizer skips them), so none are listed.
inline constexpr Literal kFlintLiterals[] = {
    {"<think>", kThinkStart},      {"</think>", kThinkEnd},     {"<tool_call>", kToolStart},     {"</tool_call>", kToolEnd},
    {"<arg_key>", kArgKeyStart},   {"</arg_key>", kArgKeyEnd},  {"<arg_value>", kArgValStart},   {"</arg_value>", kArgValEnd},
};

// Leftmost literal tokenizer with a held tail: text that could still become a literal waits for
// the next chunk, so every split of the input yields the same token sequence. No literal is a
// prefix of another, so the longest match is simply the match.
class Lexer {
 public:
  Lexer() = default;
  template <size_t N>
  explicit Lexer(const Literal (&lits)[N]) : lits_(lits), n_(N) {}
  template <class F>
  void feed(std::string_view in, F &&emit) {
    buf_.append(in);
    run(false, emit);
  }
  template <class F>
  void finish(F &&emit) {
    run(true, emit);
  }

 private:
  template <class F>
  void run(bool final, F &&emit) {
    const std::string_view b = buf_;
    size_t from = 0, i = 0;
    while (i < b.size()) {
      if (b[i] != '<' && b[i] != '>') {
        ++i;
        continue;
      }
      const std::string_view rest = b.substr(i);
      const Literal *hit = nullptr;
      bool partial = false;
      for (size_t li = 0; li < n_; ++li) {
        const Literal &l = lits_[li];
        if (starts_with(rest, l.text)) {
          hit = &l;
          break;
        }
        partial |= rest.size() < l.text.size() && starts_with(l.text, rest);
      }
      if (hit) {
        if (i > from) emit(kText, b.substr(from, i - from));
        emit(hit->kind, hit->text);
        i += hit->text.size();
        from = i;
      } else if (partial && !final) {
        break;
      } else {
        ++i;
      }
    }
    if (i > from) emit(kText, b.substr(from, i - from));
    buf_.erase(0, i);
  }

  std::string buf_;
  const Literal *lits_ = kLiterals;
  size_t n_ = std::size(kLiterals);
};

// ---- tool-call arguments (parse_args.cpp) --------------------------------------------------------------

size_t ws_word(std::string_view s, size_t i, std::string_view word);
size_t open_tag(std::string_view s, size_t i);
size_t close_tag(std::string_view s, size_t i);
Value convert_args(std::string_view raw);
Value number_lexeme(std::string digits, double d);
bool py_int(std::string_view s, Value &out);
bool py_float(std::string_view s, double &out);
bool json_finite(const Value &v);
std::vector<std::string> schema_types(const Value *schema);
std::string norm_type(std::string_view t);
bool coerce(const std::string &value, const std::vector<std::string> &types, Value &out);
void fix_arg_types(Value &args, const Value *tools, std::string_view name);
bool find_json_name(std::string_view s, std::string &name);

// ---- output parsers (parser_*.cpp) -----------------------------------------------------------------------

// The qwen3 reasoning parser + the qwen3_xml tool parser (qwen38, qwen35, llama). raw_content: tool_choice "none".
std::unique_ptr<text::OutputParser> make_qwen3_parser(bool thinking, bool tools, const Value *schemas, bool raw_content = false);
// The GLM-4.7 reasoning + tool-call parsers (flint). suppress: tool_choice "none".
std::unique_ptr<text::OutputParser> make_glm47_parser(bool thinking, bool tools, const Value *schemas, bool suppress = false);

}  // namespace xenithium:text:chat:detail
