// Xenithium · the Responses API over the chat path: the input-item -> message rules, the output
// items, the streaming state machine.
#include "xenithium/server/responses.hpp"

#include <random>

namespace xenithium::server::responses {

namespace {

using json::Value;

std::string uuid_hex() {
  thread_local std::mt19937_64 rng{std::random_device{}()};
  static const char *hex = "0123456789abcdef";
  std::string s(32, '0');
  uint64_t bits = 0;
  for (size_t i = 0; i < s.size(); ++i) {
    if (i % 16 == 0) bits = rng();
    s[i] = hex[bits & 15];
    bits >>= 4;
  }
  return s;
}

bool fail(std::string &err, std::string &param, std::string m, std::string p) {
  err = std::move(m);
  param = std::move(p);
  return false;
}

bool truthy(const Value *v) { return v && v->is_bool() && v->b; }

// The text of an assistant item's content (the first part's text).
const std::string *first_text(const Value &content) {
  if (content.is_string()) return &content.str;
  if (content.is_array() && !content.arr.empty() && content.arr[0].is_object()) return content.arr[0].get_string("text");
  return nullptr;
}

}  // namespace

bool to_chat(const Value &r, Value &chat, std::string &err, std::string &param) {
  chat = Value::object();
  if (const Value *v = r.get("previous_response_id"); v && !v->is_null())
    return fail(err, param, "previous_response_id is not supported: this server stores no responses; send the whole conversation in input",
                "previous_response_id");
  if (truthy(r.get("background"))) return fail(err, param, "background responses are not supported: this server stores no responses", "background");
  if (const Value *v = r.get("prompt"); v && !v->is_null()) return fail(err, param, "prompt templates are not supported", "prompt");
  if (const Value *v = r.get("previous_input_messages"); v && !v->is_null())
    return fail(err, param, "previous_input_messages is not supported", "previous_input_messages");
  if (const Value *v = r.get("conversation"); v && !v->is_null()) return fail(err, param, "conversations are not supported: this server stores none", "conversation");

  Value messages = Value::array();
  if (const Value *ins = r.get("instructions"); ins && !ins->is_null()) {
    if (!ins->is_string()) return fail(err, param, "instructions must be a string", "instructions");
    if (!ins->str.empty()) {
      Value m = Value::object();
      m.set("role", Value::string("system"));
      m.set("content", *ins);
      messages.push(std::move(m));
    }
  }
  const Value *input = r.get("input");
  if (!input || input->is_null()) return fail(err, param, "input is required", "input");
  auto prev_assistant = [&]() -> Value * {
    if (messages.arr.empty()) return nullptr;
    Value &last = messages.arr.back();
    const std::string *role = last.get_string("role");
    return role && *role == "assistant" ? &last : nullptr;
  };
  if (input->is_string()) {
    Value m = Value::object();
    m.set("role", Value::string("user"));
    m.set("content", *input);
    messages.push(std::move(m));
  } else if (input->is_array()) {
    for (size_t i = 0; i < input->arr.size(); ++i) {
      const Value &item = input->arr[i];
      const std::string where = "input[" + std::to_string(i) + "]";
      if (!item.is_object()) return fail(err, param, where + ": an input item must be an object", "input");
      const std::string *type = item.get_string("type");
      const std::string *role = item.get_string("role");
      if (role && (!type || *type == "message")) {
        if (*role == "assistant") {
          const Value *c = item.get("content");
          const std::string *text = c ? first_text(*c) : nullptr;
          if (!text) return fail(err, param, where + ": an assistant message needs text content", "input");
          if (Value *pa = prev_assistant(); pa && (!pa->get("content") || pa->get("content")->is_null())) {
            pa->set("content", Value::string(*text));
            continue;
          }
          Value m = Value::object();
          m.set("role", Value::string("assistant"));
          m.set("content", Value::string(*text));
          messages.push(std::move(m));
          continue;
        }
        Value m = Value::object();
        m.set("role", Value::string(*role));
        if (const Value *c = item.get("content")) {
          if (c->is_array()) {  // the Responses parts as the chat parts they mean
            Value parts = Value::array();
            for (const Value &p : c->arr) {
              const std::string *pt = p.is_object() ? p.get_string("type") : nullptr;
              if (pt && (*pt == "input_text" || *pt == "output_text")) {
                Value t = Value::object();
                t.set("type", Value::string("text"));
                if (const Value *tx = p.get("text")) t.set("text", *tx);
                parts.push(std::move(t));
              } else if (pt && *pt == "input_file") {
                return fail(err, param, where + ": input_file parts are not supported", "input");
              } else {
                parts.push(p);
              }
            }
            m.set("content", std::move(parts));
          } else {
            m.set("content", *c);
          }
        }
        messages.push(std::move(m));
        continue;
      }
      if (!type) return fail(err, param, where + ": an input item needs a type or a role", "input");
      if (*type == "function_call") {
        const std::string *call_id = item.get_string("call_id"), *name = item.get_string("name"), *args = item.get_string("arguments");
        if (!call_id || !name || !args) return fail(err, param, where + ": a function_call needs call_id, name and arguments (strings)", "input");
        std::string fname = *name;
        if (const std::string *ns = item.get_string("namespace"); ns && !ns->empty()) fname = *ns + "." + *name;
        Value fn = Value::object();
        fn.set("name", Value::string(fname));
        fn.set("arguments", Value::string(*args));
        Value tc = Value::object();
        tc.set("id", Value::string(*call_id));
        tc.set("type", Value::string("function"));
        tc.set("function", std::move(fn));
        if (Value *pa = prev_assistant()) {
          Value *calls = pa->get("tool_calls");
          if (!calls || !calls->is_array()) calls = &pa->set("tool_calls", Value::array());
          calls->push(std::move(tc));
          continue;
        }
        Value m = Value::object();
        m.set("role", Value::string("assistant"));
        Value calls = Value::array();
        calls.push(std::move(tc));
        m.set("tool_calls", std::move(calls));
        messages.push(std::move(m));
      } else if (*type == "function_call_output") {
        const std::string *call_id = item.get_string("call_id");
        const Value *out = item.get("output");
        if (!call_id || !out) return fail(err, param, where + ": a function_call_output needs call_id and output", "input");
        Value m = Value::object();
        m.set("role", Value::string("tool"));
        m.set("content", *out);
        m.set("tool_call_id", Value::string(*call_id));
        messages.push(std::move(m));
      } else if (*type == "reasoning") {
        if (const Value *enc = item.get("encrypted_content"); enc && !enc->is_null())
          return fail(err, param, "Encrypted content is not supported.", "input");
        std::string text;
        const Value *content = item.get_array("content");
        const Value *summary = item.get_array("summary");
        if (content && !content->arr.empty() && content->arr[0].get_string("text")) text = *content->arr[0].get_string("text");
        else if (summary && !summary->arr.empty() && summary->arr[0].get_string("text")) text = *summary->arr[0].get_string("text");
        if (Value *pa = prev_assistant(); pa && !pa->get("reasoning")) {
          pa->set("reasoning", Value::string(text));
          continue;
        }
        Value m = Value::object();
        m.set("role", Value::string("assistant"));
        m.set("reasoning", Value::string(text));
        messages.push(std::move(m));
      } else {
        return fail(err, param, "Unsupported input item type: " + *type, "input");
      }
    }
  } else {
    return fail(err, param, "input must be a string or a list of input items", "input");
  }
  chat.set("messages", std::move(messages));

  if (const Value *tools = r.get("tools"); tools && !tools->is_null()) {
    if (!tools->is_array()) return fail(err, param, "tools must be a list", "tools");
    Value out = Value::array();
    for (const Value &t : tools->arr) {
      const std::string *type = t.is_object() ? t.get_string("type") : nullptr;
      if (!type || *type != "function")
        return fail(err, param, "only function tools are supported (tool type '" + (type ? *type : std::string("?")) + "')", "tools");
      const std::string *name = t.get_string("name");
      if (!name) return fail(err, param, "a function tool needs a name", "tools");
      Value fn = Value::object();
      for (const auto &kv : t.obj)
        if (kv.first != "type") fn.set(kv.first, kv.second);
      Value ct = Value::object();
      ct.set("type", Value::string("function"));
      ct.set("function", std::move(fn));
      out.push(std::move(ct));
    }
    if (!out.arr.empty()) chat.set("tools", std::move(out));
  }
  if (const Value *tc = r.get("tool_choice"); tc && !tc->is_null()) {
    if (tc->is_string()) {
      chat.set("tool_choice", *tc);
    } else if (tc->is_object()) {
      const std::string *type = tc->get_string("type");
      const std::string *name = tc->get_string("name");
      if (!type || *type != "function" || !name) return fail(err, param, "tool_choice: only \"auto\", \"none\", \"required\" or {\"type\": \"function\", \"name\": ...}", "tool_choice");
      Value fn = Value::object();
      fn.set("name", Value::string(*name));
      Value o = Value::object();
      o.set("type", Value::string("function"));
      o.set("function", std::move(fn));
      chat.set("tool_choice", std::move(o));
    } else {
      return fail(err, param, "tool_choice must be a string or an object", "tool_choice");
    }
  }
  if (const Value *text = r.get("text"); text && !text->is_null()) {
    const Value *fmt = text->is_object() ? text->get("format") : nullptr;
    if (fmt && !fmt->is_null()) {
      const std::string *type = fmt->is_object() ? fmt->get_string("type") : nullptr;
      if (!type) return fail(err, param, "text.format needs a type", "text");
      if (*type == "json_object") {
        Value rf = Value::object();
        rf.set("type", Value::string("json_object"));
        chat.set("response_format", std::move(rf));
      } else if (*type == "json_schema") {
        const Value *schema = fmt->get("schema");
        if (schema && !schema->is_null()) {  // a json_schema format without a schema is unconstrained
          Value js = Value::object();
          js.set("name", Value::string(fmt->get_string("name") ? *fmt->get_string("name") : "response"));
          js.set("schema", *schema);
          if (const Value *st = fmt->get("strict")) js.set("strict", *st);
          Value rf = Value::object();
          rf.set("type", Value::string("json_schema"));
          rf.set("json_schema", std::move(js));
          chat.set("response_format", std::move(rf));
        }
      } else if (*type != "text") {
        return fail(err, param, "text.format type must be text, json_object or json_schema", "text");
      }
    }
  }
  if (const Value *rs = r.get("reasoning"); rs && rs->is_object())
    if (const Value *eff = rs->get("effort"); eff && !eff->is_null()) chat.set("reasoning_effort", *eff);
  if (const Value *v = r.get("max_output_tokens"); v && !v->is_null()) chat.set("max_tokens", *v);
  bool lp = false;
  if (const Value *inc = r.get("include"); inc && inc->is_array())
    for (const Value &x : inc->arr) lp |= x.is_string() && x.str == "message.output_text.logprobs";
  if (lp) {
    chat.set("logprobs", Value::boolean(true));
    const Value *tl = r.get("top_logprobs");
    chat.set("top_logprobs", tl && !tl->is_null() ? *tl : Value::integer(0));
  }
  for (const char *k : {"model", "stream", "temperature", "top_p", "top_k", "min_p", "presence_penalty", "frequency_penalty",
                        "repetition_penalty", "seed", "stop", "ignore_eos", "logit_bias", "parallel_tool_calls",
                        "chat_template_kwargs", "skip_special_tokens", "include_stop_str_in_output", "min_tokens"})
    if (const Value *v = r.get(k)) chat.set(k, *v);
  return true;
}

json::Value output_items(const std::string &reasoning, const std::string &content, const std::string &lp_items, bool with_lp,
                         const std::vector<Call> &calls) {
  Value out = Value::array();
  if (!reasoning.empty()) {
    Value part = Value::object();
    part.set("text", Value::string(reasoning));
    part.set("type", Value::string("reasoning_text"));
    Value parts = Value::array();
    parts.push(std::move(part));
    Value it = Value::object();
    it.set("id", Value::string("rs_" + uuid_hex()));
    it.set("summary", Value::array());
    it.set("type", Value::string("reasoning"));
    it.set("content", std::move(parts));
    it.set("encrypted_content", Value());
    it.set("status", Value());
    out.push(std::move(it));
  }
  if (!content.empty()) {
    Value part = Value::object();
    part.set("annotations", Value::array());
    part.set("text", Value::string(content));
    part.set("type", Value::string("output_text"));
    if (with_lp) {
      Value lps;
      json::parse("[" + lp_items + "]", lps);
      part.set("logprobs", std::move(lps));
    } else {
      part.set("logprobs", Value());
    }
    Value parts = Value::array();
    parts.push(std::move(part));
    Value it = Value::object();
    it.set("id", Value::string("msg_" + uuid_hex()));
    it.set("content", std::move(parts));
    it.set("role", Value::string("assistant"));
    it.set("status", Value::string("completed"));
    it.set("type", Value::string("message"));
    out.push(std::move(it));
  }
  for (const Call &c : calls) {
    Value it = Value::object();
    it.set("arguments", Value::string(c.arguments));
    it.set("call_id", Value::string(c.call_id));
    it.set("name", Value::string(c.name));
    it.set("type", Value::string("function_call"));
    it.set("id", Value::string("fc_" + uuid_hex()));
    it.set("status", Value::string("completed"));
    out.push(std::move(it));
  }
  return out;
}

json::Value usage(long long input, long long output, long long reasoning) {
  Value in_d = Value::object();
  in_d.set("cached_tokens", Value::integer(0));
  Value out_d = Value::object();
  out_d.set("reasoning_tokens", Value::integer(reasoning));
  Value u = Value::object();
  u.set("input_tokens", Value::integer(input));
  u.set("input_tokens_details", std::move(in_d));
  u.set("output_tokens", Value::integer(output));
  u.set("output_tokens_details", std::move(out_d));
  u.set("total_tokens", Value::integer(input + output));
  return u;
}

std::string response_json(const Meta &m, const char *status, const json::Value *output, const json::Value *usage_v) {
  const Value &r = m.request;
  auto echo = [&](const char *k) -> Value {
    const Value *v = r.get(k);
    return v ? *v : Value();
  };
  Value o = Value::object();
  o.set("id", Value::string(m.id));
  o.set("created_at", Value::integer(m.created));
  if (std::string(status) == "incomplete") {
    Value d = Value::object();
    d.set("reason", Value::string("max_output_tokens"));
    o.set("incomplete_details", std::move(d));
  } else {
    o.set("incomplete_details", Value());
  }
  o.set("instructions", echo("instructions"));
  o.set("metadata", echo("metadata"));
  o.set("model", Value::string(m.model));
  o.set("object", Value::string("response"));
  o.set("output", output ? *output : Value::array());
  const Value *ptc = r.get("parallel_tool_calls");
  o.set("parallel_tool_calls", Value::boolean(!(ptc && ptc->is_bool()) || ptc->b));
  o.set("temperature", Value::real(m.temperature));
  const Value *tc = r.get("tool_choice");
  o.set("tool_choice", tc && !tc->is_null() ? *tc : Value::string("auto"));
  const Value *tools = r.get("tools");
  o.set("tools", tools && tools->is_array() ? *tools : Value::array());
  o.set("top_p", Value::real(m.top_p));
  o.set("background", Value::boolean(false));
  o.set("max_output_tokens", Value::integer(m.max_output_tokens));
  o.set("max_tool_calls", echo("max_tool_calls"));
  o.set("previous_response_id", Value());
  o.set("prompt", Value());
  o.set("reasoning", echo("reasoning"));
  o.set("service_tier", r.get_string("service_tier") ? *r.get("service_tier") : Value::string("auto"));
  o.set("status", Value::string(status));
  o.set("text", echo("text"));
  o.set("top_logprobs", Value::integer(m.top_logprobs));
  o.set("truncation", Value::string("disabled"));
  o.set("usage", usage_v ? *usage_v : Value());
  o.set("user", echo("user"));
  return json::dump(o);
}

// ---- streaming

void Stream::emit(const char *type, json::Value ev, std::string &sse) {
  ev.set("sequence_number", Value::integer(seq_++));
  ev.set("type", Value::string(type));
  sse += "event: ";
  sse += type;
  sse += "\ndata: ";
  json::write(ev, sse);
  sse += "\n\n";
}

void Stream::created(const Meta &m, std::string &sse) {
  for (const char *type : {"response.created", "response.in_progress"}) {
    Value ev = Value::object();
    Value resp;
    json::parse(response_json(m, "in_progress", nullptr, nullptr), resp);
    ev.set("response", std::move(resp));
    emit(type, std::move(ev), sse);
  }
}

void Stream::open(State s, std::string &sse) {
  st_ = s;
  item_id_ = uuid_hex();
  text_.clear();
  Value ev = Value::object();
  ev.set("output_index", Value::integer(output_index_));
  Value item = Value::object();
  item.set("id", Value::string(item_id_));
  if (s == State::content) {
    item.set("content", Value::array());
    item.set("role", Value::string("assistant"));
    item.set("status", Value::string("in_progress"));
    item.set("type", Value::string("message"));
    ev.set("item", std::move(item));
    emit("response.output_item.added", std::move(ev), sse);
    Value pe = Value::object();
    pe.set("content_index", Value::integer(0));
    pe.set("item_id", Value::string(item_id_));
    pe.set("output_index", Value::integer(output_index_));
    Value part = Value::object();
    part.set("annotations", Value::array());
    part.set("text", Value::string(""));
    part.set("type", Value::string("output_text"));
    part.set("logprobs", Value::array());
    pe.set("part", std::move(part));
    emit("response.content_part.added", std::move(pe), sse);
  } else if (s == State::reasoning) {
    item.set("summary", Value::array());
    item.set("type", Value::string("reasoning"));
    item.set("status", Value::string("in_progress"));
    ev.set("item", std::move(item));
    emit("response.output_item.added", std::move(ev), sse);
    Value pe = Value::object();
    pe.set("content_index", Value::integer(0));
    pe.set("item_id", Value::string(item_id_));
    pe.set("output_index", Value::integer(output_index_));
    Value part = Value::object();
    part.set("text", Value::string(""));
    part.set("type", Value::string("reasoning_text"));
    pe.set("part", std::move(part));
    emit("response.reasoning_part.added", std::move(pe), sse);
  } else if (s == State::tool) {
    item.set("arguments", Value::string(""));
    item.set("call_id", Value::string(call_id_));
    item.set("name", Value::string(name_));
    item.set("type", Value::string("function_call"));
    item.set("status", Value::string("in_progress"));
    ev.set("item", std::move(item));
    emit("response.output_item.added", std::move(ev), sse);
  }
}

void Stream::close(std::string &sse) {
  if (st_ == State::none) return;
  if (st_ == State::content) {
    Value part = Value::object();
    part.set("annotations", Value::array());
    part.set("text", Value::string(text_));
    part.set("type", Value::string("output_text"));
    part.set("logprobs", Value());
    Value d = Value::object();
    d.set("content_index", Value::integer(0));
    d.set("item_id", Value::string(item_id_));
    d.set("logprobs", Value::array());
    d.set("output_index", Value::integer(output_index_));
    d.set("text", Value::string(text_));
    emit("response.output_text.done", std::move(d), sse);
    Value pd = Value::object();
    pd.set("content_index", Value::integer(0));
    pd.set("item_id", Value::string(item_id_));
    pd.set("output_index", Value::integer(output_index_));
    pd.set("part", part);
    emit("response.content_part.done", std::move(pd), sse);
    Value item = Value::object();
    item.set("id", Value::string(item_id_));
    Value parts = Value::array();
    if (!text_.empty()) parts.push(std::move(part));
    item.set("content", std::move(parts));
    item.set("role", Value::string("assistant"));
    item.set("status", Value::string("completed"));
    item.set("type", Value::string("message"));
    Value od = Value::object();
    od.set("item", std::move(item));
    od.set("output_index", Value::integer(output_index_));
    emit("response.output_item.done", std::move(od), sse);
  } else if (st_ == State::reasoning) {
    Value part = Value::object();
    part.set("text", Value::string(text_));
    part.set("type", Value::string("reasoning_text"));
    Value d = Value::object();
    d.set("content_index", Value::integer(0));
    d.set("item_id", Value::string(item_id_));
    d.set("output_index", Value::integer(output_index_));
    d.set("text", Value::string(text_));
    emit("response.reasoning_text.done", std::move(d), sse);
    Value pd = Value::object();
    pd.set("content_index", Value::integer(0));
    pd.set("item_id", Value::string(item_id_));
    pd.set("output_index", Value::integer(output_index_));
    pd.set("part", part);
    emit("response.reasoning_part.done", std::move(pd), sse);
    Value item = Value::object();
    item.set("id", Value::string(item_id_));
    item.set("summary", Value::array());
    item.set("type", Value::string("reasoning"));
    Value parts = Value::array();
    parts.push(std::move(part));
    item.set("content", std::move(parts));
    item.set("status", Value::string("completed"));
    Value od = Value::object();
    od.set("item", std::move(item));
    od.set("output_index", Value::integer(output_index_));
    emit("response.output_item.done", std::move(od), sse);
  } else {
    if (tool_delta_) {
      Value d = Value::object();
      d.set("arguments", Value::string(text_));
      d.set("item_id", Value::string(item_id_));
      d.set("name", Value::string(name_));
      d.set("output_index", Value::integer(output_index_));
      emit("response.function_call_arguments.done", std::move(d), sse);
    }
    Value item = Value::object();
    item.set("arguments", Value::string(text_));
    item.set("call_id", Value::string(call_id_));
    item.set("name", Value::string(name_));
    item.set("type", Value::string("function_call"));
    item.set("id", Value::string(item_id_));
    item.set("status", Value::string("completed"));
    Value od = Value::object();
    od.set("item", std::move(item));
    od.set("output_index", Value::integer(output_index_));
    emit("response.output_item.done", std::move(od), sse);
  }
  ++output_index_;
  st_ = State::none;
}

void Stream::reasoning(std::string_view text, std::string &sse) {
  if (st_ != State::reasoning) {
    close(sse);
    open(State::reasoning, sse);
  }
  text_ += text;
  Value d = Value::object();
  d.set("content_index", Value::integer(0));
  d.set("delta", Value::string(text));
  d.set("item_id", Value::string(item_id_));
  d.set("output_index", Value::integer(output_index_));
  emit("response.reasoning_text.delta", std::move(d), sse);
}

void Stream::content(std::string_view text, std::string_view lp_items, std::string &sse) {
  if (text.empty()) return;  // no message is opened for an empty content delta
  if (st_ != State::content) {
    close(sse);
    open(State::content, sse);
  }
  text_ += text;
  Value d = Value::object();
  d.set("content_index", Value::integer(0));
  d.set("delta", Value::string(text));
  d.set("item_id", Value::string(item_id_));
  Value lps = Value::array();
  if (!lp_items.empty()) json::parse("[" + std::string(lp_items) + "]", lps);
  d.set("logprobs", std::move(lps));
  d.set("output_index", Value::integer(output_index_));
  emit("response.output_text.delta", std::move(d), sse);
}

void Stream::tool_start(std::string_view name, int index, std::string &sse) {
  close(sse);
  call_id_ = "call_" + uuid_hex();
  name_ = std::string(name);
  tool_index_ = index;
  tool_delta_ = false;
  open(State::tool, sse);
}

void Stream::tool_args(std::string_view args, std::string &sse) {
  if (st_ != State::tool || args.empty()) return;
  text_ += args;
  tool_delta_ = true;
  Value d = Value::object();
  d.set("delta", Value::string(args));
  d.set("item_id", Value::string(item_id_));
  d.set("output_index", Value::integer(output_index_));
  emit("response.function_call_arguments.delta", std::move(d), sse);
}

void Stream::completed(const Meta &m, const char *status, const json::Value &output, const json::Value &usage_v, std::string &sse) {
  close(sse);
  Value ev = Value::object();
  Value resp;
  json::parse(response_json(m, status, &output, &usage_v), resp);
  ev.set("response", std::move(resp));
  emit("response.completed", std::move(ev), sse);  // also when the status is incomplete
}

void Stream::error(int code, std::string_view message, std::string &sse) {
  Value ev = Value::object();
  ev.set("code", Value::string(std::to_string(code)));
  ev.set("message", Value::string(message));
  ev.set("param", Value());
  emit("error", std::move(ev), sse);
}

}  // namespace xenithium:server:responses
