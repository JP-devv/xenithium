// Xenithium · the GLM-4.7-style reasoning + tool-call output parser (flint).
#include "chat_internal.hpp"

namespace xenithium::text::chat::detail {

namespace {


// Flint's output parser (the GLM-4.7 reasoning and tool-call formats). Thinking off, a <think> block is still
// recognised and dropped. Thinking on, the output starts in
// reasoning: <think> there is dropped, </think> ends it, <tool_call> ends it and opens a call. In
// content <think> re-enters reasoning and </think> is dropped. A call is <tool_call>NAME[<arg_key>K</arg_key>
// <arg_value>V</arg_value>...]</tool_call> (the name runs to the first <arg_key>; the end of generation closes an open
// call); arguments are the complete key / value pairs (key stripped, value as is, a repeated key keeps its first
// position and last value), typed by the tool's schema; a call is returned when its stripped name is non-empty and
// (with tools) one of the request's tools. Reasoning loses trailing whitespace. Content is the text before the first
// call (plus the text after it when no call was accepted), stripped when a call was accepted. tool_choice "none"
// (suppress): calls are dropped whole, the rest is content as is.
class Glm47Parser final : public text::OutputParser {
 public:
  Glm47Parser(bool thinking, bool tools, const Value *schemas, bool suppress = false)
      : thinking_(thinking), tools_(tools), suppress_(suppress), schemas_(schemas), lex_(kFlintLiterals), reasoning_(thinking) {}

  void feed(std::string_view text, std::vector<text::Delta> &out) override {
    out_ = &out;
    lex_.feed(text, [&](Lit k, std::string_view t) { step(k, t); });
  }

  void finish(std::vector<text::Delta> &out) override {
    out_ = &out;
    lex_.finish([&](Lit k, std::string_view t) { step(k, t); });
    if (in_call_) end_call();
    rws_.clear();  // reasoning loses its trailing whitespace
    if (suppress_) {
      emit_content(held_);
    } else if (calls_ == 0 && !called_) {
      // no accepted call: the text after a call event is content too; whitespace-only content next to a call is
      // dropped; with no call at all the content is as generated
      std::string rest = held_ + after_;
      if (!(tool_seen_ && !emitted_ && all_space(rest))) emit_content(rest);
    }  // else content.strip: what is still held is trailing whitespace, the text after the first call is dropped
    held_.clear();
    after_.clear();
  }

  int tool_calls() const override { return calls_; }
  bool tool_markup() const override { return called_; }

 private:
  void step(Lit k, std::string_view t) {
    if (in_call_) {
      if (k == kToolEnd) return end_call();
      if (!in_args_ && k == kArgKeyStart) in_args_ = true;
      (in_args_ ? args_ : name_) += t;
      return;
    }
    if (reasoning_) {
      if (k == kThinkStart) return;
      if (k == kThinkEnd) {
        reasoning_ = false;
        return;
      }
      if (k == kToolStart) {
        reasoning_ = false;
        return begin_call();
      }
      if (thinking_) add_reasoning(t);
      return;
    }
    if (k == kThinkStart) {  // thinking off: the block is still recognised, and dropped
      reasoning_ = true;
      return;
    }
    if (k == kThinkEnd) return;
    if (k == kToolStart) return begin_call();
    text(t);
  }

  void add_reasoning(std::string_view t) {
    rws_ += t;
    const size_t end = last_nonspace_end(rws_);
    if (end == 0) return;
    if (!out_->empty() && out_->back().kind == text::Delta::reasoning) out_->back().text.append(rws_, 0, end);
    else out_->push_back({text::Delta::reasoning, rws_.substr(0, end), 0});
    rws_.erase(0, end);
  }

  void begin_call() {
    rws_.clear();
    tool_seen_ = in_call_ = true;
    in_args_ = false;
    name_.clear();
    args_.clear();
  }

  void end_call() {
    in_call_ = false;
    if (suppress_) return;
    if (name_.empty() && args_.empty()) return;
    std::string name(py_strip(name_));
    Value args = Value::object();
    if (name.empty()) {
      if (all_space(args_)) return;
      // no name: the body as a JSON object {"name"..., "arguments" | "parameters" | the rest}
      Value body;
      if (!py_json_loads(py_strip(args_), body) || !body.is_object()) return;
      const Value *nm = body.get("name");
      if (!nm || !nm->is_string() || nm->str.empty()) return;
      name = nm->str;
      const Value *a = body.get("arguments");
      if (!a) a = body.get("parameters");
      if (a && a->is_object()) {
        args = *a;
      } else if (a && a->is_string()) {
        if (!py_json_loads(a->str, args) || !args.is_object()) args = Value::object();
      } else {
        for (const auto &kv : body.obj)
          if (kv.first != "name") args.set(kv.first, kv.second);
      }
    } else {
      args = flint_args(args_);
    }
    if (tools_ && !tool_known(name)) return;  // not one of the request's tools
    fix_arg_types(args, schemas_, name);
    called_ = true;
    if (!tools_) return;  // a request without tools: parsed (it shapes the content) but not returned
    out_->push_back({text::Delta::tool_start, name, calls_});
    out_->push_back({text::Delta::tool_args, json::dump(args, json::Style::python), calls_});
    ++calls_;
  }

  bool tool_known(const std::string &name) const {
    if (!schemas_ || !schemas_->is_array()) return true;
    for (const auto &t : schemas_->arr) {
      const Value *f = t.get_object("function");
      const std::string *n = f ? f->get_string("name") : nullptr;
      if (n && *n == name) return true;
    }
    return false;
  }

  // <arg_key>K</arg_key>\s*<arg_value>V</arg_value> pairs, leftmost, non-greedy.
  static Value flint_args(std::string_view raw) {
    Value params = Value::object();
    constexpr std::string_view ks = "<arg_key>", ke = "</arg_key>", vs = "<arg_value>", ve = "</arg_value>";
    size_t pos = 0;
    while ((pos = raw.find(ks, pos)) != std::string_view::npos) {
      const size_t k0 = pos + ks.size(), k1 = raw.find(ke, k0);
      if (k1 == std::string_view::npos) break;
      size_t v = skip_space(raw, k1 + ke.size());
      if (raw.substr(v, vs.size()) != vs) {
        pos = pos + 1;  // this <arg_key> starts no pair; the next one may
        continue;
      }
      const size_t v0 = v + vs.size(), v1 = raw.find(ve, v0);
      if (v1 == std::string_view::npos) {
        pos = pos + 1;
        continue;
      }
      params.set(std::string(py_strip(raw.substr(k0, k1 - k0))), Value::string(std::string(raw.substr(v0, v1 - v0))));
      pos = v1 + ve.size();
    }
    return params;
  }

  void text(std::string_view t) {
    if (suppress_) {
      held_ += t;
      return;
    }
    if (tool_seen_) {
      after_ += t;
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

  const bool thinking_, tools_, suppress_;
  const Value *schemas_;
  std::vector<text::Delta> *out_ = nullptr;
  Lexer lex_;
  bool reasoning_;
  bool in_call_ = false, in_args_ = false, tool_seen_ = false, called_ = false, emitted_ = false;
  std::string name_, args_, rws_, held_, after_;
  int calls_ = 0;
};

}  // namespace

std::unique_ptr<text::OutputParser> make_glm47_parser(bool thinking, bool tools, const Value *schemas, bool suppress) {
  return std::make_unique<Glm47Parser>(thinking, tools, schemas, suppress);
}

}  // namespace xenithium:text:chat:detail
