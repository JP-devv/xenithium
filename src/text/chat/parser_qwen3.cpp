// Xenithium · the qwen3 reasoning + qwen3_xml tool-call output parser (qwen38, qwen35, llama).
#include "chat_internal.hpp"

namespace xenithium::text::chat::detail {

namespace {


// The non-streaming result, produced incrementally. Two passes:
// pass 1 (reasoning parser) splits reasoning from content; pass 2 (tool parser) runs the qwen3
// tool-call state machine over that content. Content before the first call is released as soon
// as it cannot change; see text for the one place a stream differs from the non-streaming result.
class Qwen3Parser final : public text::OutputParser {
 public:
  Qwen3Parser(bool thinking, bool tools, const Value *schemas, bool raw_content = false)
      : thinking_(thinking), tools_(tools), raw_(raw_content), schemas_(schemas), s1_(thinking ? kReasoning : kAfter) {}

  void feed(std::string_view text, std::vector<text::Delta> &out) override {
    out_ = &out;
    if (thinking_) lex1_.feed(text, [&](Lit k, std::string_view t) { pass1(k, t); });
    else pass2_text(text);  // thinking off: the reasoning pass returns the text untouched
  }

  void finish(std::vector<text::Delta> &out) override {
    out_ = &out;
    if (thinking_) lex1_.finish([&](Lit k, std::string_view t) { pass1(k, t); });
    lex2_.finish([&](Lit k, std::string_view t) { pass2(k, t); });
    if (s2_ != kContent) end_call();  // a call cut off by the end of generation still counts
    if (!tool_seen_) {
      emit_content(held_);
    } else if (!called_) {
      // Text after a call survives only if no call announced a name; whitespace-only content
      // next to a call is dropped.
      const std::string rest = held_ + (named_ ? std::string() : after_);
      if (emitted_ || !all_space(rest)) emit_content(rest);
    }  // else content.strip: what is still held is trailing whitespace
    held_.clear();
    after_.clear();
  }

  int tool_calls() const override { return calls_; }
  bool tool_markup() const override { return called_; }  // a complete call was parsed (returned or not)

 private:
  enum S1 : uint8_t { kReasoning, kAfter };
  enum S2 : uint8_t { kContent, kPreamble, kName, kArgs, kBetween };

  // Pass 1: reasoning runs to </think>, or to <tool_call> (an implicit end; the tag stays in
  // the content). <think> inside reasoning and every later </think> are dropped, as are special
  // tokens; tool markup is plain text here.
  void pass1(Lit k, std::string_view t) {
    if (k == kSpecial) return;
    if (s1_ == kReasoning) {
      if (k == kThinkStart) return;
      if (k == kThinkEnd || k == kToolStart) {
        s1_ = kAfter;
        if (k == kToolStart) pass2_text(t);
        return;
      }
      if (!out_->empty() && out_->back().kind == text::Delta::reasoning) out_->back().text += t;
      else out_->push_back({text::Delta::reasoning, std::string(t), 0});
      return;
    }
    if (k != kThinkEnd) pass2_text(t);
  }

  void pass2_text(std::string_view t) {
    lex2_.feed(t, [&](Lit k, std::string_view x) { pass2(k, x); });
  }

  // Pass 2: the qwen3 tool-call state machine. Text in the preamble (between
  // <tool_call> and <function=) and between </function> and </tool_call> is dropped; inside
  // the arguments everything but </function> is argument text.
  void pass2(Lit k, std::string_view t) {
    if (k == kSpecial) return;
    switch (s2_) {
      case kContent:
        if (k == kToolStart) return begin_call(kPreamble);
        if (k == kFuncPrefix) return begin_call(kName);  // <function= without <tool_call>
        if (k == kThinkEnd) return;
        return text(t);
      case kPreamble:
        if (k == kToolEnd) end_call(), s2_ = kContent;
        else if (k == kFuncPrefix) s2_ = kName;
        return;
      case kName:
        if (k == kCloseAngle) s2_ = kArgs;
        else if (k == kFuncEnd) end_call(), s2_ = kBetween;
        else name_ += t;
        return;
      case kArgs:
        if (k == kFuncEnd) end_call(), s2_ = kBetween;
        else args_ += t;
        return;
      case kBetween:
        if (k == kToolEnd) s2_ = kContent;
        else if (k == kToolStart) begin_call(kPreamble);
        else if (k == kFuncPrefix) begin_call(kName);
        return;
    }
  }

  void begin_call(S2 next) {
    tool_seen_ = open_ = true;
    name_.clear();
    args_.clear();
    s2_ = next;
  }

  // A call is complete: it is a tool call when its name is non-empty after strip; it is emitted
  // whole, since nothing about it changes later.
  void end_call() {
    if (!open_) return;
    if (raw_) {  // tool_choice "none": the call's markup is dropped, nothing is returned
      open_ = false;
      return;
    }
    open_ = false;
    if (name_.empty() && args_.empty()) return;
    // With no name after <function=, one is taken from a `"name": "..."` in the body
    // (its arguments still come from <parameter=> tags only).
    std::string raw_name = name_;
    if (raw_name.empty()) find_json_name(args_, raw_name);
    if (raw_name.empty()) return;
    named_ = true;  // from here on, text after a call is never content
    const std::string name(py_strip(raw_name));
    if (name.empty()) return;
    Value args = all_space(args_) ? Value::object() : convert_args(args_);
    fix_arg_types(args, schemas_, name);
    std::string args_text = json::dump(args, json::Style::python);
    called_ = true;
    if (!tools_) return;  // a request without tools: the call is parsed (it shapes the content) but not returned
    out_->push_back({text::Delta::tool_start, name, calls_});
    out_->push_back({text::Delta::tool_args, std::move(args_text), calls_});
    ++calls_;
  }

  // Content text of pass 2. The content is stripped when a call follows, which an
  // append-only stream cannot know in advance: trailing whitespace is held until more text comes
  // (exact), and leading whitespace until the first real text, after which it is released (the
  // one deviation: with a later call a non-streaming result would have stripped it). Text after the first
  // call is held until the end and kept only if no call announced a name.
  void text(std::string_view t) {
    if (raw_) return emit_content(t);  // tool_choice "none": all text around calls is content, unstripped
    if (tool_seen_) {
      if (!named_) after_ += t;
      return;
    }
    held_ += t;
    const size_t end = last_nonspace_end(held_);
    if (end == 0) return;
    emit_content(std::string_view(held_).substr(0, end));
    held_.erase(0, end);
  }

  void emit_content(std::string_view t) {
    if (t.empty()) return;
    emitted_ = true;
    if (!out_->empty() && out_->back().kind == text::Delta::content) out_->back().text += t;
    else out_->push_back({text::Delta::content, std::string(t), 0});
  }

  const bool thinking_, tools_, raw_;
  const Value *schemas_;
  std::vector<text::Delta> *out_ = nullptr;
  Lexer lex1_, lex2_;
  S1 s1_;
  S2 s2_ = kContent;
  std::string name_, args_;  // the open call
  bool open_ = false, tool_seen_ = false, named_ = false, called_ = false, emitted_ = false;
  std::string held_, after_;
  int calls_ = 0;
};

}  // namespace

std::unique_ptr<text::OutputParser> make_qwen3_parser(bool thinking, bool tools, const Value *schemas, bool raw_content) {
  return std::make_unique<Qwen3Parser>(thinking, tools, schemas, raw_content);
}

}  // namespace xenithium:text:chat:detail
