// Xenithium · the built-in chat formats behind text:ChatFormat (xenithium/text/chat_formats.hpp):
// render: the request normalisation (request.cpp) followed by the arch's chat template rendered by hand
// (template_*.cpp), or the checkpoint's own template through the Jinja interpreter (JinjaChat).
// parser: the reasoning + tool-call parsers of a non-streaming response (parser_*.cpp).
// tests/data/chat_golden*.json hold the cases of every rule; comments say why a rule exists, the goldens say that it
// holds.
#include "xenithium/text/chat_formats.hpp"

#include "chat_internal.hpp"

namespace xenithium::text {

using chat::detail::Message;
using chat::detail::Type;
using chat::detail::Value;
using namespace chat::detail;

namespace {

class Glm47Chat final : public text::ChatFormat {
 public:
  bool render(const Value &messages, const Value *tools, const text::ChatOptions &opt, std::string &out, std::string &err) const override {
    std::vector<Message> msgs;
    if (!normalise_messages(messages, msgs, err, true, opt.images)) return false;
    std::vector<Value> tl;
    const bool have_tools = tools && !tools->is_null();
    if (have_tools && !normalise_tools(*tools, tl, err)) return false;
    return render_template_flint(msgs, have_tools ? &tl : nullptr, opt, out, err);
  }
  // The parser's view: on unless chat_template_kwargs sets enable_thinking (a JSON null counts as unset).
  bool thinking(const Value *v) const override {
    if (!v || v->is_null()) return true;
    switch (v->type) {
      case Type::boolean: return v->b;
      case Type::number: return v->num != 0;
      case Type::string: return !v->str.empty();
      case Type::array: return !v->arr.empty();
      case Type::object: return !v->obj.empty();
      default: return true;
    }
  }
  std::unique_ptr<text::OutputParser> parser(bool thinking, bool tools) const override { return make_glm47_parser(thinking, tools, nullptr); }
  std::unique_ptr<text::OutputParser> parser_no_tool_calls(bool thinking) const override {
    return make_glm47_parser(thinking, false, nullptr, true);
  }
  std::unique_ptr<text::OutputParser> parser(bool thinking, const Value *tools) const override {
    const bool have = tools && !tools->is_null();
    return make_glm47_parser(thinking, have, have ? tools : nullptr);
  }
  bool exclude_tools_on_none() const override { return true; }
  void finish_content(std::string &content, bool tool_markup) const override {
    if (tool_markup) content.erase(0, skip_space(content, 0));
  }
};

class Qwen3Chat final : public text::ChatFormat {
 public:
  explicit Qwen3Chat(ChatTemplate t) : tpl_(t) {}
  bool render(const Value &messages, const Value *tools, const text::ChatOptions &opt, std::string &out,
              std::string &err) const override {
    std::vector<Message> msgs;
    if (!normalise_messages(messages, msgs, err, false, opt.images)) return false;
    std::vector<Value> tl;
    const bool have_tools = tools && !tools->is_null();
    if (have_tools && !normalise_tools(*tools, tl, err)) return false;
    if (tpl_ == ChatTemplate::nanbeige43) return render_template_nanbeige(msgs, have_tools ? &tl : nullptr, opt, out, err);
    return render_template_qwen38(msgs, have_tools ? &tl : nullptr, opt, out, err);
  }

  // chat_template_kwargs.get("enable_thinking", True) under Python truthiness: an explicit null
  // is off here although the template treats it as undefined (on).
  bool thinking(const Value *v) const override {
    if (!v) return true;
    switch (v->type) {
      case Type::null: return false;
      case Type::boolean: return v->b;
      case Type::number: return v->num != 0;
      case Type::string: return !v->str.empty();
      case Type::array: return !v->arr.empty();
      case Type::object: return !v->obj.empty();
    }
    return true;
  }

  std::unique_ptr<text::OutputParser> parser(bool thinking, bool tools) const override {
    return make_qwen3_parser(thinking, tools, nullptr);
  }

  void finish_content(std::string &content, bool tool_markup) const override {
    if (tool_markup) content.erase(0, skip_space(content, 0));
  }
  std::unique_ptr<text::OutputParser> parser_no_tool_calls(bool thinking) const override {
    return make_qwen3_parser(thinking, false, nullptr, true);
  }

  std::unique_ptr<text::OutputParser> parser(bool thinking, const Value *tools) const override {
    const bool have = tools && !tools->is_null();
    return make_qwen3_parser(thinking, have, have ? tools : nullptr);
  }

 private:
  ChatTemplate tpl_;
};

// The checkpoint's template through the Jinja interpreter, over the normalised conversation.
class JinjaChat final : public text::ChatFormat {
 public:
  JinjaChat(std::unique_ptr<text::ChatFormat> base, std::shared_ptr<const text::jinja::Template> tpl, Value globals, bool keep_developer)
      : base_(std::move(base)), tpl_(std::move(tpl)), globals_(std::move(globals)), keep_developer_(keep_developer) {}

  bool render(const Value &messages, const Value *tools, const text::ChatOptions &opt, std::string &out, std::string &err) const override {
    std::vector<Message> msgs;
    if (!normalise_messages(messages, msgs, err, keep_developer_, opt.images)) return false;
    std::vector<Value> tl;
    const bool have_tools = tools && !tools->is_null();
    if (have_tools && !normalise_tools(*tools, tl, err)) return false;
    const bool parts = tpl_->iterates_content();
    Value conv = Value::array();
    for (const auto &m : msgs) {
      Value o = Value::object();
      o.set("role", Value::string(m.role));
      if (parts) {
        Value c = Value::array();
        for (size_t k = 0; k < m.parts.size(); ++k) {
          Value part = Value::object();
          if (k < m.image.size() && m.image[k]) {  // the template prints its own placeholder for an image / video item
            part.set("type", Value::string(m.image[k] == 2 ? "video" : "image"));
          } else {
            part.set("type", Value::string("text"));
            part.set("text", Value::string(m.parts[k]));
          }
          c.push(std::move(part));
        }
        o.set("content", std::move(c));
      } else {
        o.set("content", Value::string(m.joined("\n")));
      }
      if (m.reasoning) o.set("reasoning_content", *m.reasoning);
      if (!m.calls.empty()) {
        Value calls = Value::array();
        for (const auto &c : m.calls) {
          Value fn = Value::object();
          fn.set("name", Value::string(c.name));
          fn.set("arguments", c.args);
          Value call = Value::object();
          call.set("id", Value::string(c.id));
          call.set("type", Value::string("function"));
          call.set("function", std::move(fn));
          calls.push(std::move(call));
        }
        o.set("tool_calls", std::move(calls));
      }
      if (m.tool_call_id) o.set("tool_call_id", *m.tool_call_id);
      conv.push(std::move(o));
    }
    Value ctx = globals_;
    if (opt.kwargs && opt.kwargs->is_object())  // null kwargs are dropped (the template sees them undefined)
      for (const auto &kv : opt.kwargs->obj)
        if (!kv.second.is_null()) ctx.set(kv.first, kv.second);
    ctx.set("messages", std::move(conv));
    if (have_tools) {
      Value t = Value::array();
      for (auto &x : tl) t.push(std::move(x));
      ctx.set("tools", std::move(t));
    } else {
      ctx.set("tools", Value());
    }
    ctx.set("add_generation_prompt", Value::boolean(opt.add_generation_prompt));
    auto flag = [&](const char *k, text::Flag f) {
      if (f == text::Flag::yes || f == text::Flag::no) ctx.set(k, Value::boolean(f == text::Flag::yes));
      else if (f == text::Flag::unset) ctx.erase(k);
    };
    flag("enable_thinking", opt.enable_thinking);
    flag("preserve_thinking", opt.preserve_thinking);
    if (opt.reasoning_effort) ctx.set("reasoning_effort", *opt.reasoning_effort);
    return tpl_->render(ctx, out, err);
  }
  bool thinking(const Value *v) const override { return base_->thinking(v); }
  std::unique_ptr<text::OutputParser> parser(bool thinking, bool tools) const override { return base_->parser(thinking, tools); }
  std::unique_ptr<text::OutputParser> parser(bool thinking, const Value *tools) const override { return base_->parser(thinking, tools); }
  std::unique_ptr<text::OutputParser> parser_no_tool_calls(bool thinking) const override { return base_->parser_no_tool_calls(thinking); }
  void finish_content(std::string &content, bool tool_markup) const override { base_->finish_content(content, tool_markup); }
  bool exclude_tools_on_none() const override { return base_->exclude_tools_on_none(); }

 private:
  std::unique_ptr<text::ChatFormat> base_;
  std::shared_ptr<const text::jinja::Template> tpl_;
  Value globals_;
  bool keep_developer_;
};

}  // namespace

std::unique_ptr<ChatFormat> make_jinja_chat_format(std::unique_ptr<ChatFormat> base, std::shared_ptr<const jinja::Template> tpl, json::Value globals,
                                                   bool keep_developer) {
  return std::make_unique<JinjaChat>(std::move(base), std::move(tpl), std::move(globals), keep_developer);
}

std::unique_ptr<ChatFormat> make_chat_format(ChatTemplate t) {
  if (t == ChatTemplate::flint) return std::make_unique<Glm47Chat>();
  return std::make_unique<Qwen3Chat>(t);
}

}  // namespace xenithium:text
