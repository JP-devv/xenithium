// Xenithium · tool-call arguments: qwen3_xml's converter and the typing of argument values by the tool's schema.
#include "chat_internal.hpp"

namespace xenithium::text::chat::detail {

// `\s*word` at i -> its end (with trailing \s*), or npos.
size_t ws_word(std::string_view s, size_t i, std::string_view word) {
  i = skip_space(s, i);
  if (s.substr(i, word.size()) != word) return npos;
  return skip_space(s, i + word.size());
}

// `<\s*parameter\s*=` at i -> its end, or npos.
size_t open_tag(std::string_view s, size_t i) {
  const size_t j = ws_word(s, i + 1, "parameter");
  return j == npos || j >= s.size() || s[j] != '=' ? npos : j + 1;
}

// `<\s*/\s*parameter\s*>` at i -> its end, or npos.
size_t close_tag(std::string_view s, size_t i) {
  size_t j = ws_word(s, i + 1, "/");
  if (j != npos) j = ws_word(s, j, "parameter");
  return j == npos || j >= s.size() || s[j] != '>' ? npos : j + 1;
}

// qwen3_xml's argument converter: <parameter=NAME>VALUE</parameter> pairs, a value also ending
// at the next <parameter=; one wrapping newline trimmed on each side; values stay strings; a
// parameter with no terminator is dropped; a repeated name keeps its first position, last value.
Value convert_args(std::string_view raw) {
  Value params = Value::object();
  for (size_t pos = raw.find('<'); pos != npos; pos = raw.find('<', pos)) {
    size_t name_from = open_tag(raw, pos);
    const size_t gt = name_from == npos ? npos : raw.find('>', name_from = skip_space(raw, name_from));
    size_t vend = npos, next = npos;
    if (gt != npos) {
      for (size_t j = raw.find('<', gt + 1); j != npos; j = raw.find('<', j + 1)) {
        if (const size_t e = close_tag(raw, j); e != npos) {
          vend = j, next = e;
          break;
        }
        if (open_tag(raw, j) != npos) {
          vend = j, next = j;
          break;
        }
      }
    }
    if (vend == npos) {
      ++pos;
      continue;
    }
    std::string_view v = raw.substr(gt + 1, vend - gt - 1);
    if (starts_with(v, "\n")) v.remove_prefix(1);
    if (ends_with(v, "\n")) v.remove_suffix(1);
    params.set(raw.substr(name_from, gt - name_from), Value::string(v));
    pos = next;
  }
  return params;
}

// --- argument types from the tool schema, Python semantics ---

Value number_lexeme(std::string digits, double d) {
  Value v;
  v.type = Type::number;
  v.str = std::move(digits);
  v.num = d;
  return v;
}

// Python int: surrounding whitespace, a sign, ASCII digits with single '_' between digits.
bool py_int(std::string_view s, Value &out) {
  s = py_strip(s);
  bool neg = false;
  if (!s.empty() && (s[0] == '+' || s[0] == '-')) neg = s[0] == '-', s.remove_prefix(1);
  if (s.empty() || s.front() == '_' || s.back() == '_') return false;
  std::string digits;
  for (size_t i = 0; i < s.size(); ++i) {
    if (s[i] == '_') {
      if (s[i - 1] == '_') return false;
      continue;
    }
    if (s[i] < '0' || s[i] > '9') return false;
    digits += s[i];
  }
  const size_t nz = digits.find_first_not_of('0');
  digits = nz == std::string::npos ? "0" : digits.substr(nz);
  if (neg && digits != "0") digits.insert(0, "-");
  out = number_lexeme(digits, std::strtod(digits.c_str(), nullptr));
  return true;
}

// Python float, finite results only (inf / nan / bad syntax -> false).
bool py_float(std::string_view s, double &out) {
  s = py_strip(s);
  std::string clean;
  size_t i = 0;
  auto digits = [&]() {
    bool any = false;
    while (i < s.size()) {
      if (s[i] >= '0' && s[i] <= '9') {
        clean += s[i++];
        any = true;
      } else if (s[i] == '_' && any && i + 1 < s.size() && s[i + 1] >= '0' && s[i + 1] <= '9') {
        ++i;
      } else {
        break;
      }
    }
    return any;
  };
  if (i < s.size() && (s[i] == '+' || s[i] == '-')) clean += s[i++];
  bool mant = digits();
  if (i < s.size() && s[i] == '.') {
    clean += s[i++];
    mant |= digits();
  }
  if (!mant) return false;
  if (i < s.size() && (s[i] == 'e' || s[i] == 'E')) {
    clean += s[i++];
    if (i < s.size() && (s[i] == '+' || s[i] == '-')) clean += s[i++];
    if (!digits()) return false;
  }
  if (i != s.size()) return false;
  out = std::strtod(clean.c_str(), nullptr);
  return std::isfinite(out);
}

bool json_finite(const Value &v) {
  if (v.is_number()) return v.is_int() || std::isfinite(v.num);
  for (const auto &e : v.arr)
    if (!json_finite(e)) return false;
  for (const auto &kv : v.obj)
    if (!json_finite(kv.second)) return false;
  return true;
}

// The types a schema node names: `type, the types of `enum` values, anyOf/oneOf/allOf; a schema
// node that names nothing counts as "string".
std::vector<std::string> schema_types(const Value *schema) {
  std::vector<std::string> types;
  auto add = [&](std::string_view t) {
    for (const auto &x : types)
      if (x == t) return;
    types.emplace_back(t);
  };
  if (!schema || !schema->is_object()) return {"string"};
  if (const Value *t = schema->get("type")) {
    if (t->is_string()) add(t->str);
    for (const auto &e : t->arr)
      if (e.is_string()) add(e.str);
  }
  if (const Value *en = schema->get_array("enum"))
    for (const auto &e : en->arr)
      add(e.is_null()     ? "null"
          : e.is_bool()   ? "boolean"
          : e.is_number() ? (e.is_int() ? "integer" : "number")
          : e.is_string() ? "string"
          : e.is_array()  ? "array"
                          : "object");
  for (const char *k : {"anyOf", "oneOf", "allOf"})
    if (const Value *ch = schema->get_array(k))
      for (const auto &c : ch->arr)
        for (const auto &t : schema_types(&c)) add(t);
  if (types.empty()) types.emplace_back("string");
  return types;
}

std::string norm_type(std::string_view t) {
  static constexpr std::pair<std::string_view, std::string_view> kAlias[] = {
      {"str", "string"},     {"text", "string"},     {"varchar", "string"},  {"char", "string"},
      {"enum", "string"},    {"int", "integer"},     {"int32", "integer"},   {"int64", "integer"},
      {"uint", "integer"},   {"uint32", "integer"},  {"uint64", "integer"},  {"long", "integer"},
      {"short", "integer"},  {"unsigned", "integer"}, {"float", "number"},   {"float32", "number"},
      {"float64", "number"}, {"double", "number"},   {"bool", "boolean"},    {"dict", "object"},
      {"arr", "array"},      {"list", "array"},      {"sequence", "array"}};
  const std::string s = ascii_lower(py_strip(t));
  for (const auto &[a, b] : kAlias)
    if (s == a) return std::string(b);
  return s;
}

// Coercion to the schema's types: candidates in the order null > integer > number > boolean > object >
// array > string, then a json.loads fallback. True + `out` when the string is replaced.
bool coerce(const std::string &value, const std::vector<std::string> &types, Value &out) {
  std::vector<std::string> norm;
  for (const auto &t : types) norm.push_back(norm_type(t));
  auto has = [&](std::string_view t) {
    for (const auto &x : norm)
      if (x == t) return true;
    return false;
  };
  auto loads = [&](Value &v) { return json::parse(value, v) && json_finite(v); };
  for (std::string_view cand : {"null", "integer", "number", "boolean", "object", "array", "string"}) {
    if (!has(cand)) continue;
    if (cand == "string") return false;
    if (cand == "null") {
      if (ascii_lower(value) == "null") return out = Value(), true;
    } else if (cand == "integer") {
      if (py_int(value, out)) return true;
    } else if (cand == "number") {
      double d;
      if (!py_float(value, d)) continue;
      if (d != std::trunc(d)) return out = Value::real(d), true;
      // An integral float becomes a Python int: its exact decimal digits, -0 as 0.
      char buf[400];
      const auto r = std::to_chars(buf, buf + sizeof(buf), d, std::chars_format::fixed, 0);
      std::string digits(buf, r.ptr);
      if (digits == "-0") digits = "0";
      return out = number_lexeme(digits, d == 0 ? 0.0 : d), true;
    } else if (cand == "boolean") {
      const std::string l = ascii_lower(py_strip(value));
      if (l == "true" || l == "1") return out = Value::boolean(true), true;
      if (l == "false" || l == "0") return out = Value::boolean(false), true;
    } else if (Value v; loads(v)) {  // object / array: any finite JSON value is taken as is
      return out = std::move(v), true;
    }
  }
  if (Value v; loads(v)) return out = std::move(v), true;
  return false;
}

// The `properties` of the first request tool named `name`; nullptr when absent or empty.
const Value *tool_properties(const Value *tools, std::string_view name) {
  if (!tools || !tools->is_array()) return nullptr;
  for (const auto &t : tools->arr) {
    const Value *fn = t.get_object("function");
    const std::string *n = fn ? fn->get_string("name") : nullptr;
    if (!n || *n != name) continue;
    const Value *p = fn->get_object("parameters");
    const Value *props = p ? p->get_object("properties") : nullptr;
    return props && !props->obj.empty() ? props : nullptr;
  }
  return nullptr;
}

void fix_arg_types(Value &args, const Value *tools, std::string_view name) {
  const Value *props = tool_properties(tools, name);
  if (!props) return;
  for (auto &kv : args.obj) {
    const Value *prop = props->get(kv.first);
    if (!prop || !prop->is_object() || !kv.second.is_string()) continue;
    Value typed;
    if (coerce(kv.second.str, schema_types(prop), typed)) kv.second = std::move(typed);
  }
}

// `"name"\s*:\s*"([^"]*)"`: the first match's group, or false.
bool find_json_name(std::string_view s, std::string &name) {
  for (size_t p = s.find("\"name\""); p != npos; p = s.find("\"name\"", p + 1)) {
    const size_t j = ws_word(s, p + 6, ":");
    if (j == npos || j >= s.size() || s[j] != '"') continue;
    const size_t q = s.find('"', j + 1);
    if (q == npos) continue;
    name.assign(s.substr(j + 1, q - j - 1));
    return true;
  }
  return false;
}

}  // namespace xenithium:text:chat:detail
