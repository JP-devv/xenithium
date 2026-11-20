// Xenithium · the chat request's normalisation ('openai' content format): messages, tool calls, tools;
// plus the Python-semantics text helpers the renderers and parsers share. The cases:
// tests/data/chat_golden*.json.
#include "chat_internal.hpp"

namespace xenithium::text::chat::detail {

// ---- text helpers ------------------------------------------------------------------------------

// Decodes one UTF-8 code point at s[i]; returns its length, or 0 for a sequence truncated by the
// end of `s` (callers treat that as "not known yet"). Invalid lead bytes decode as themselves.
size_t utf8_next(std::string_view s, size_t i, uint32_t &cp) {
  const auto c = static_cast<unsigned char>(s[i]);
  const size_t n = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 1;
  if (i + n > s.size()) return 0;
  if (n == 1) {
    cp = c;
    return 1;
  }
  cp = c & (0x7f >> n);
  for (size_t k = 1; k < n; ++k) cp = (cp << 6) | (static_cast<unsigned char>(s[i + k]) & 0x3f);
  return n;
}

// Python's str.isspace: the template's |trim and the parser's.strip both use it, so a
// U+3000 or U+00A0 around content is stripped too.
bool py_space(uint32_t c) {
  return (c >= 0x09 && c <= 0x0d) || (c >= 0x1c && c <= 0x20) || c == 0x85 || c == 0xa0 ||
         c == 0x1680 || (c >= 0x2000 && c <= 0x200a) || c == 0x2028 || c == 0x2029 ||
         c == 0x202f || c == 0x205f || c == 0x3000;
}

size_t skip_space(std::string_view s, size_t i) {
  while (i < s.size()) {
    uint32_t cp;
    const size_t n = utf8_next(s, i, cp);
    if (n == 0 || !py_space(cp)) break;
    i += n;
  }
  return i;
}

// End of the last complete code point that is not whitespace (0 if none). A truncated sequence
// at the end counts as neither, so it is never released before it is complete.
size_t last_nonspace_end(std::string_view s) {
  size_t i = 0, end = 0;
  while (i < s.size()) {
    uint32_t cp;
    const size_t n = utf8_next(s, i, cp);
    if (n == 0) break;
    i += n;
    if (!py_space(cp)) end = i;
  }
  return end;
}

bool all_space(std::string_view s) { return skip_space(s, 0) == s.size(); }

std::string_view py_strip(std::string_view s) {
  s.remove_prefix(skip_space(s, 0));
  return s.substr(0, last_nonspace_end(s));
}

bool starts_with(std::string_view s, std::string_view p) { return s.substr(0, p.size()) == p; }
bool ends_with(std::string_view s, std::string_view p) {
  return s.size() >= p.size() && s.substr(s.size() - p.size()) == p;
}

std::string ascii_lower(std::string_view s) {
  std::string r(s);
  for (auto &c : r) c = static_cast<char>(c >= 'A' && c <= 'Z' ? c + 32 : c);
  return r;
}

// Python repr of a string (only reached when an error message prints a container).
void py_repr_str(std::string_view s, std::string &out) {
  const bool sq = s.find('\'') != npos, dq = s.find('"') != npos;
  const char q = sq && !dq ? '"' : '\'';
  out += q;
  for (char ch : s) {
    const auto c = static_cast<unsigned char>(ch);
    if (ch == q || ch == '\\') {
      out += '\\';
      out += ch;
    } else if (ch == '\n') {
      out += "\\n";
    } else if (ch == '\r') {
      out += "\\r";
    } else if (ch == '\t') {
      out += "\\t";
    } else if (c < 0x20 || c == 0x7f) {
      char b[8];
      std::snprintf(b, sizeof(b), "\\x%02x", c);
      out += b;
    } else {
      out += ch;
    }
  }
  out += q;
}

// Python str of a JSON value as json.loads produced it: what the template's `~` prints.
void py_str(const Value &v, std::string &out, bool top) {
  switch (v.type) {
    case Type::null: out += "None"; return;
    case Type::boolean: out += v.b ? "True" : "False"; return;
    case Type::number: out += v.is_int() ? v.str : json::python_float_repr(v.num); return;
    case Type::string:
      if (top) out += v.str;
      else py_repr_str(v.str, out);
      return;
    case Type::array:
      out += '[';
      for (size_t i = 0; i < v.arr.size(); ++i) {
        if (i) out += ", ";
        py_str(v.arr[i], out, false);
      }
      out += ']';
      return;
    case Type::object:
      out += '{';
      for (size_t i = 0; i < v.obj.size(); ++i) {
        if (i) out += ", ";
        py_repr_str(v.obj[i].first, out);
        out += ": ";
        py_str(v.obj[i].second, out, false);
      }
      out += '}';
      return;
  }
}

// pydantic's lax bool (tool `strict` / `defer_loading`): "true", 1, "yes"... are accepted.
bool lax_bool(const Value &v, bool &b) {
  if (v.is_bool()) {
    b = v.b;
    return true;
  }
  if (v.is_number()) {
    if (v.num != 1 && v.num != 0) return false;
    b = v.num == 1;
    return true;
  }
  if (!v.is_string()) return false;
  const std::string s = ascii_lower(v.str);
  for (const char *t : {"1", "on", "t", "true", "y", "yes"})
    if (s == t) return b = true, true;
  for (const char *f : {"0", "off", "f", "false", "n", "no"})
    if (s == f) return b = false, true;
  return false;
}

bool fail(std::string &err, std::string msg) {
  err = std::move(msg);
  return false;
}

// ---- request normalisation -----------------------------------------------------------------------


// Content -> text parts. A string is one text part and a list's parts stay in
// order (refusal / thinking parts become text). Media is a 400, except image_url parts when the server has image input
// (images): they become image parts (kImagePart; the server decodes the URLs in the same order).
bool content_parts(const Value &m, const std::string &role, const std::string &where, Message &out, bool images,
                   std::string &err) {
  auto add_image = [&](const Value &p, const std::string &pw) {
    const Value *iu = p.get("image_url");
    const std::string *url = iu && iu->is_string() ? &iu->str : iu && iu->is_object() ? iu->get_string("url") : nullptr;
    if (!url) return fail(err, pw + ".image_url: Field required (an object with a url)");
    if (role == "system") return fail(err, "System message cannot contain images.");
    out.image.resize(out.parts.size(), 0);
    out.parts.emplace_back(kImagePart);
    out.image.push_back(1);
    return true;
  };
  auto add_video = [&](const Value &p, const std::string &pw) {  // {"type": "video_url", "video_url": {"url"...}}
    const Value *vu = p.get("video_url");
    const std::string *url = vu && vu->is_string() ? &vu->str : vu && vu->is_object() ? vu->get_string("url") : nullptr;
    if (!url) return fail(err, pw + ".video_url: Field required (an object with a url)");
    if (role == "system") return fail(err, "System message cannot contain videos.");
    out.image.resize(out.parts.size(), 0);
    out.parts.emplace_back(kVideoPart);
    out.image.push_back(2);
    return true;
  };
  const Value *c = m.get("content");
  if (!c) return true;  // absent: an empty list
  if (c->is_null()) {
    // Only the assistant message type admits null; for every other role it is a 400.
    if (role == "assistant") return true;
    return fail(err, where + ".content: Input should be a valid string or list of content parts");
  }
  if (c->is_string()) {
    out.parts.push_back(c->str);
    return true;
  }
  if (!c->is_array()) return fail(err, where + ".content: Input should be a valid string or list");
  for (size_t k = 0; k < c->arr.size(); ++k) {
    const Value &p = c->arr[k];
    const std::string pw = where + ".content[" + std::to_string(k) + "]";
    if (p.is_string()) {
      out.parts.push_back(p.str);
      continue;
    }
    if (!p.is_object()) return fail(err, pw + ": Input should be a valid content part");
    const std::string *type = p.get_string("type");
    if (!type) {
      if (images && p.get("image_url")) {  // the part's type is inferred from its key
        if (!add_image(p, pw)) return false;
        continue;
      }
      if (images && p.get("video_url")) {
        if (!add_video(p, pw)) return false;
        continue;
      }
      for (const char *media : {"image_url", "image_pil", "image_embeds", "audio_embeds",
                                "video_embeds", "prompt_embeds", "audio_url", "input_audio", "video_url"})
        if (p.get(media)) return fail(err, pw + ": " + media + " content is not supported by this server (text only)");
      return fail(err, pw + ": Missing 'type' field in multimodal part.");
    }
    const char *field = *type == "text"       ? "text"
                        : *type == "refusal"  ? "refusal"
                        : *type == "thinking" ? "thinking"
                                              : nullptr;
    if (field) {
      const Value *t = p.get(field);
      if (!t || t->is_null()) {
        if (*type == "thinking") return fail(err, pw + ": thinking part without text");
        continue;  // a text / refusal part whose payload is missing is skipped
      }
      if (!t->is_string()) return fail(err, pw + "." + field + ": Input should be a valid string");
      out.parts.push_back(t->str);
      continue;
    }
    if (images && (*type == "image_url" || *type == "input_image")) {  // input_image: the image_url field is the URL string
      if (!add_image(p, pw)) return false;
      continue;
    }
    if (images && *type == "video_url") {
      if (!add_video(p, pw)) return false;
      continue;
    }
    for (const char *media : {"image_url", "input_image", "input_audio", "audio_url", "video_url", "file",
                              "image_embeds", "audio_embeds", "video_embeds", "image_pil", "prompt_embeds"})
      if (*type == media)
        return fail(err, pw + ": '" + *type + "' content is not supported by this server (text only)");
    return fail(err, pw + ": Unsupported chat content part type: '" + *type + "'");
  }
  return true;
}

// json.loads: RFC JSON plus the bare NaN / Infinity / -Infinity Python also accepts (they are
// written back as NaN / Infinity by tojson). They are swapped for sentinel strings, parsed
// strictly, and turned into numbers whose lexeme is the token.
constexpr std::string_view kNonFinite = "\x01xen-nonfinite:";

void restore_nonfinite(Value &v) {
  if (v.is_string() && starts_with(v.str, kNonFinite)) {
    const std::string tok = v.str.substr(kNonFinite.size());
    v = Value();
    v.type = Type::number;
    v.num = tok == "NaN" ? std::nan("") : tok[0] == '-' ? -HUGE_VAL : HUGE_VAL;
    v.str = tok;
    return;
  }
  for (auto &e : v.arr) restore_nonfinite(e);
  for (auto &kv : v.obj) restore_nonfinite(kv.second);
}

bool py_json_loads(std::string_view s, Value &out) {
  if (json::parse(s, out)) return true;
  std::string t;
  bool in_str = false, esc = false, swapped = false;
  for (size_t i = 0; i < s.size(); ++i) {
    const char c = s[i];
    if (in_str) {
      t += c;
      if (esc) esc = false;
      else if (c == '\\') esc = true;
      else if (c == '"') in_str = false;
      continue;
    }
    if (c == '"') in_str = true;
    bool hit = false;
    for (std::string_view tok : {"-Infinity", "Infinity", "NaN"}) {
      if (s.substr(i, tok.size()) == tok) {
        t.append("\"\\u0001xen-nonfinite:").append(tok).append("\"");
        i += tok.size() - 1;
        hit = swapped = true;
        break;
      }
    }
    if (!hit) t += c;
  }
  if (!swapped || !json::parse(t, out)) return false;
  restore_nonfinite(out);
  return true;
}

// tool_calls of an assistant message: validated as the request model does, then the arguments
// JSON string parsed; malformed or non-object arguments become {} (one
// bad turn in history must not make the conversation unrecoverable).
bool tool_calls(const Value &m, const std::string &where0, Message &out, std::string &err) {
  const Value *tc = m.get("tool_calls");
  if (!tc || tc->is_null()) return true;
  const std::string where = where0 + ".tool_calls";
  if (!tc->is_array()) return fail(err, where + ": Input should be a valid list");
  for (size_t k = 0; k < tc->arr.size(); ++k) {
    const Value &c = tc->arr[k];
    const std::string w = where + "[" + std::to_string(k) + "]";
    if (!c.is_object()) return fail(err, w + ": Input should be an object");
    if (!c.get_string("id")) return fail(err, w + ".id: Field required (string)");
    const std::string *type = c.get_string("type");
    if (!type) return fail(err, w + ".type: Field required");
    if (*type != "function") {
      if (*type == "custom" && c.get_object("custom"))
        return fail(err, "chat completions only support assistant tool_calls of type 'function'.");
      return fail(err, w + ".type: Input should be 'function'");
    }
    const Value *fn = c.get_object("function");
    if (!fn) return fail(err, w + ".function: Field required");
    const std::string *name = fn->get_string("name");
    if (!name) return fail(err, w + ".function.name: Input should be a valid string");
    const std::string *args = fn->get_string("arguments");
    if (!args) return fail(err, w + ".function.arguments: Input should be a valid string");
    ToolCall call{*name, Value::object(), *c.get_string("id")};
    Value parsed;
    if (!args->empty() && py_json_loads(*args, parsed) && parsed.is_object()) call.args = std::move(parsed);
    out.calls.push_back(std::move(call));
  }
  return true;
}

// keep_developer: the template has a 'developer' role of its own (Flint), so it is kept.
bool normalise_messages(const Value &messages, std::vector<Message> &out, std::string &err, bool keep_developer, bool images) {
  if (!messages.is_array()) return fail(err, "messages: Input should be a valid list");
  if (messages.arr.empty())
    return fail(err, "Cannot apply chat template to an empty conversation. Provide at least one message.");
  bool developer = false;
  for (size_t i = 0; i < messages.arr.size(); ++i) {
    const Value &m = messages.arr[i];
    const std::string where = "messages[" + std::to_string(i) + "]";
    if (!m.is_object()) return fail(err, where + ": Input should be an object");
    const std::string *role = m.get_string("role");
    if (!role) return fail(err, where + ".role: Field required (string)");
    Message msg;
    msg.role = *role;
    if (!content_parts(m, *role, where, msg, images, err)) return false;
    if (!msg.image.empty()) msg.image.resize(msg.parts.size(), 0);
    if (*role == "assistant") {
      if (!tool_calls(m, where, msg, err)) return false;
      // `reasoning_content` is the deprecated spelling of `reasoning` (`reasoning` wins when both
      // are set); the template then reads it back as `reasoning_content, of any JSON type.
      const Value *r = m.get("reasoning");
      if (!r || r->is_null()) r = m.get("reasoning_content");
      if (r && !r->is_null()) msg.reasoning = r;
    } else if (*role == "tool") {
      // Tool results sent as text parts are joined with '\n' into one string.
      msg.parts.assign(1, msg.joined("\n"));
      if (const Value *id = m.get("tool_call_id"); id && !id->is_null()) msg.tool_call_id = id;
    }
    developer |= *role == "developer";
    out.push_back(std::move(msg));
  }
  if (!developer || keep_developer) return true;
  // The template has no 'developer' role, so it is renamed to system; if that leaves a system
  // message anywhere but first (or more than one), they all merge into one at position 0: each
  // one's parts joined with '\n', the non-empty results joined with "\n\n".
  bool merge = false;
  int systems = 0;
  for (size_t i = 0; i < out.size(); ++i) {
    if (out[i].role == "developer") out[i].role = "system";
    if (out[i].role == "system") merge |= i > 0 || systems++ > 0;
  }
  if (!merge) return true;
  std::string joined;
  std::vector<Message> rest;
  for (auto &m : out) {
    if (m.role != "system") {
      rest.push_back(std::move(m));
      continue;
    }
    const std::string c = m.joined("\n");
    if (!c.empty()) (joined += joined.empty() ? "" : "\n\n") += c;
  }
  Message sys;
  sys.role = "system";
  sys.parts.assign(1, joined);
  out.clear();
  out.push_back(std::move(sys));
  for (auto &m : rest) out.push_back(std::move(m));
  return true;
}

// Tools as the template sees them: exactly
// {type, function{name, description, parameters[, strict][, defer_loading]}[, defer_loading]}
// survive, description/parameters null when absent, other keys dropped.
bool normalise_tools(const Value &tools, std::vector<Value> &out, std::string &err) {
  if (!tools.is_array()) return fail(err, "tools: Input should be a valid list");
  if (tools.arr.empty())
    return fail(err, "`tools` must not be an empty array. Either provide at least one tool or omit the field entirely.");
  for (size_t i = 0; i < tools.arr.size(); ++i) {
    const Value &t = tools.arr[i];
    const std::string w = "tools[" + std::to_string(i) + "]";
    if (!t.is_object()) return fail(err, w + ": Input should be an object");
    if (const Value *ty = t.get("type"); ty && !(ty->is_string() && ty->str == "function"))
      return fail(err, w + ".type: Input should be 'function'");
    const Value *fn = t.get_object("function");
    if (!fn) return fail(err, w + ".function: Field required (object)");
    const std::string *name = fn->get_string("name");
    if (!name) return fail(err, w + ".function.name: Field required (string)");
    Value f = Value::object();
    f.set("name", Value::string(*name));
    const Value *d = fn->get("description");
    if (d && !d->is_null() && !d->is_string())
      return fail(err, w + ".function.description: Input should be a valid string");
    f.set("description", d ? *d : Value());
    const Value *p = fn->get("parameters");
    if (p && !p->is_null() && !p->is_object())
      return fail(err, w + ".function.parameters: Input should be a valid dictionary");
    f.set("parameters", p ? *p : Value());
    bool b;
    for (const char *k : {"strict", "defer_loading"}) {
      const Value *v = fn->get(k);
      if (!v || v->is_null()) continue;
      if (!lax_bool(*v, b)) return fail(err, w + ".function." + k + ": Input should be a valid boolean");
      f.set(k, Value::boolean(b));
    }
    // A tool-level defer_loading is copied into the function when that one is unset.
    const Value *top = t.get("defer_loading");
    bool top_set = false, top_b = false;
    if (top && !top->is_null()) {
      if (!lax_bool(*top, top_b)) return fail(err, w + ".defer_loading: Input should be a valid boolean");
      top_set = true;
      if (!f.get("defer_loading")) f.set("defer_loading", Value::boolean(top_b));
    }
    Value tool = Value::object();
    tool.set("type", Value::string("function"));
    tool.set("function", std::move(f));
    if (top_set) tool.set("defer_loading", Value::boolean(top_b));
    out.push_back(std::move(tool));
  }
  return true;
}

}  // namespace xenithium:text:chat:detail
