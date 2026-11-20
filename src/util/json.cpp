#include "xenithium/util/json.hpp"

#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string_view>
#include <unordered_map>

namespace xenithium::json {

// ---------------------------------------------------------------- Value

Value Value::boolean(bool v) {
  Value r;
  r.type = Type::boolean;
  r.b = v;
  return r;
}

Value Value::integer(int64_t v) {
  Value r;
  r.type = Type::number;
  r.num = static_cast<double>(v);
  r.str = std::to_string(v);
  return r;
}

Value Value::real(double v) {
  Value r;
  r.type = Type::number;
  r.num = v;
  r.str = python_float_repr(v);
  return r;
}

Value Value::string(std::string_view s) {
  Value r;
  r.type = Type::string;
  r.str.assign(s);
  return r;
}

Value Value::array() {
  Value r;
  r.type = Type::array;
  return r;
}

Value Value::object() {
  Value r;
  r.type = Type::object;
  return r;
}

bool Value::is_int() const {
  if (type != Type::number) return false;
  for (char c : str)
    if (c == '.' || c == 'e' || c == 'E') return false;
  return true;
}

const Value* Value::get(std::string_view key) const {
  if (type != Type::object) return nullptr;
  for (const auto& kv : obj)
    if (kv.first == key) return &kv.second;
  return nullptr;
}

Value* Value::get(std::string_view key) {
  return const_cast<Value*>(static_cast<const Value*>(this)->get(key));
}

Value& Value::set(std::string_view key, Value v) {
  if (type != Type::object) {
    *this = object();
  }
  for (auto& kv : obj)
    if (kv.first == key) {
      kv.second = std::move(v);
      return kv.second;
    }
  obj.push_back(Member{std::string(key), std::move(v)});
  return obj.back().second;
}

Value& Value::push(Value v) {
  if (type != Type::array) *this = array();
  arr.push_back(std::move(v));
  return arr.back();
}

bool Value::erase(std::string_view key) {
  if (type != Type::object) return false;
  for (size_t i = 0; i < obj.size(); ++i)
    if (obj[i].first == key) {
      obj.erase(obj.begin() + static_cast<std::ptrdiff_t>(i));
      return true;
    }
  return false;
}

const std::string* Value::get_string(std::string_view key) const {
  const Value* v = get(key);
  return v && v->is_string() ? &v->str : nullptr;
}

const Value* Value::get_array(std::string_view key) const {
  const Value* v = get(key);
  return v && v->is_array() ? v : nullptr;
}

const Value* Value::get_object(std::string_view key) const {
  const Value* v = get(key);
  return v && v->is_object() ? v : nullptr;
}

// ---------------------------------------------------------------- parser

namespace {

void append_utf8(std::string& out, uint32_t cp) {
  if (cp < 0x80) {
    out.push_back(static_cast<char>(cp));
  } else if (cp < 0x800) {
    out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else if (cp < 0x10000) {
    out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else {
    out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  }
}

// Length of the valid UTF-8 sequence at p (0 if invalid). Rejects overlongs,
// surrogates and code points above U+10FFFF, as Python's strict decoder does.
size_t utf8_seq(const unsigned char* p, const unsigned char* end) {
  const unsigned c = p[0];
  if (c < 0x80) return 1;
  size_t n;
  uint32_t cp;
  if (c >= 0xC2 && c <= 0xDF) {
    n = 2;
    cp = c & 0x1F;
  } else if (c >= 0xE0 && c <= 0xEF) {
    n = 3;
    cp = c & 0x0F;
  } else if (c >= 0xF0 && c <= 0xF4) {
    n = 4;
    cp = c & 0x07;
  } else {
    return 0;
  }
  if (static_cast<size_t>(end - p) < n) return 0;
  for (size_t i = 1; i < n; ++i) {
    if ((p[i] & 0xC0) != 0x80) return 0;
    cp = (cp << 6) | (p[i] & 0x3F);
  }
  if (n == 3 && (cp < 0x800 || (cp >= 0xD800 && cp <= 0xDFFF))) return 0;
  if (n == 4 && (cp < 0x10000 || cp > 0x10FFFF)) return 0;
  return n;
}

struct Parser {
  const char* p;
  const char* end;
  int max_depth;
  std::string err;

  bool fail(const char* what) {
    if (err.empty()) err = what;
    return false;
  }

  void ws() {
    while (p < end && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) ++p;
  }

  bool literal(const char* word, size_t n) {
    if (static_cast<size_t>(end - p) < n || std::memcmp(p, word, n) != 0) return fail("invalid literal");
    p += n;
    return true;
  }

  static int hex(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
  }

  bool hex4(uint32_t& v) {
    if (end - p < 4) return fail("short \\u escape");
    v = 0;
    for (int i = 0; i < 4; ++i) {
      const int h = hex(p[i]);
      if (h < 0) return fail("bad \\u escape");
      v = (v << 4) | static_cast<uint32_t>(h);
    }
    p += 4;
    return true;
  }

  bool string(std::string& out) {
    ++p;  // opening quote
    out.clear();
    const char* run = p;
    while (true) {
      if (p >= end) return fail("unterminated string");
      const unsigned char c = static_cast<unsigned char>(*p);
      if (c == '"') {
        out.append(run, p);
        ++p;
        return true;
      }
      if (c < 0x20) return fail("control character in string");
      if (c == '\\') {
        out.append(run, p);
        ++p;
        if (p >= end) return fail("unterminated escape");
        const char e = *p++;
        switch (e) {
          case '"': out.push_back('"'); break;
          case '\\': out.push_back('\\'); break;
          case '/': out.push_back('/'); break;
          case 'b': out.push_back('\b'); break;
          case 'f': out.push_back('\f'); break;
          case 'n': out.push_back('\n'); break;
          case 'r': out.push_back('\r'); break;
          case 't': out.push_back('\t'); break;
          case 'u': {
            uint32_t cp;
            if (!hex4(cp)) return false;
            if (cp >= 0xD800 && cp <= 0xDBFF) {
              uint32_t lo = 0;
              if (end - p >= 6 && p[0] == '\\' && p[1] == 'u') {
                const char* save = p;
                p += 2;
                if (!hex4(lo)) return false;
                if (lo >= 0xDC00 && lo <= 0xDFFF) {
                  cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                } else {
                  p = save;  // not a pair: the high half alone is replaced
                  cp = 0xFFFD;
                }
              } else {
                cp = 0xFFFD;
              }
            } else if (cp >= 0xDC00 && cp <= 0xDFFF) {
              cp = 0xFFFD;
            }
            append_utf8(out, cp);
            break;
          }
          default: return fail("invalid escape");
        }
        run = p;
        continue;
      }
      if (c < 0x80) {
        ++p;
        continue;
      }
      const size_t n = utf8_seq(reinterpret_cast<const unsigned char*>(p), reinterpret_cast<const unsigned char*>(end));
      if (n == 0) return fail("invalid UTF-8");
      p += n;
    }
  }

  bool number(Value& v) {
    const char* start = p;
    if (p < end && *p == '-') ++p;
    if (p >= end) return fail("bad number");
    if (*p == '0') {
      ++p;
    } else if (*p >= '1' && *p <= '9') {
      while (p < end && *p >= '0' && *p <= '9') ++p;
    } else {
      return fail("bad number");
    }
    if (p < end && *p == '.') {
      ++p;
      if (p >= end || *p < '0' || *p > '9') return fail("bad fraction");
      while (p < end && *p >= '0' && *p <= '9') ++p;
    }
    if (p < end && (*p == 'e' || *p == 'E')) {
      ++p;
      if (p < end && (*p == '+' || *p == '-')) ++p;
      if (p >= end || *p < '0' || *p > '9') return fail("bad exponent");
      while (p < end && *p >= '0' && *p <= '9') ++p;
    }
    v.type = Type::number;
    v.str.assign(start, p);
    // from_chars handles out-of-range values by returning an error; map those
    // to inf / 0 as strtod (and Python) do.
    auto r = std::from_chars(start, p, v.num);
    if (r.ec == std::errc::result_out_of_range) v.num = std::strtod(v.str.c_str(), nullptr);
    return true;
  }

  static void dedup(Value& o) {
    auto& kv = o.obj;
    if (kv.size() < 2) return;
    bool dup = false;
    if (kv.size() <= 16) {
      for (size_t i = 1; i < kv.size() && !dup; ++i)
        for (size_t j = 0; j < i; ++j)
          if (kv[i].first == kv[j].first) {
            dup = true;
            break;
          }
    } else {
      std::unordered_map<std::string_view, size_t> seen;
      seen.reserve(kv.size());
      for (size_t i = 0; i < kv.size(); ++i)
        if (!seen.emplace(kv[i].first, i).second) {
          dup = true;
          break;
        }
    }
    if (!dup) return;
    // Python dict semantics: first position, last value.
    std::vector<Member> outv;
    std::unordered_map<std::string, size_t> pos;
    for (auto& e : kv) {
      auto it = pos.find(e.first);
      if (it == pos.end()) {
        pos.emplace(e.first, outv.size());
        outv.push_back(std::move(e));
      } else {
        outv[it->second].second = std::move(e.second);
      }
    }
    kv = std::move(outv);
  }

  bool value(Value& v, int depth) {
    if (depth > max_depth) return fail("nesting too deep");
    ws();
    if (p >= end) return fail("unexpected end");
    switch (*p) {
      case 'n': v = Value{}; return literal("null", 4);
      case 't': v = Value::boolean(true); return literal("true", 4);
      case 'f': v = Value::boolean(false); return literal("false", 5);
      case '"': v.type = Type::string; return string(v.str);
      case '[': {
        ++p;
        v = Value::array();
        ws();
        if (p < end && *p == ']') {
          ++p;
          return true;
        }
        while (true) {
          v.arr.emplace_back();
          if (!value(v.arr.back(), depth + 1)) return false;
          ws();
          if (p >= end) return fail("unterminated array");
          if (*p == ',') {
            ++p;
            continue;
          }
          if (*p == ']') {
            ++p;
            return true;
          }
          return fail("expected , or ]");
        }
      }
      case '{': {
        ++p;
        v = Value::object();
        ws();
        if (p < end && *p == '}') {
          ++p;
          return true;
        }
        while (true) {
          ws();
          if (p >= end || *p != '"') return fail("expected object key");
          v.obj.emplace_back();
          if (!string(v.obj.back().first)) return false;
          ws();
          if (p >= end || *p != ':') return fail("expected :");
          ++p;
          if (!value(v.obj.back().second, depth + 1)) return false;
          ws();
          if (p >= end) return fail("unterminated object");
          if (*p == ',') {
            ++p;
            continue;
          }
          if (*p == '}') {
            ++p;
            dedup(v);
            return true;
          }
          return fail("expected , or }");
        }
      }
      default:
        if (*p == '-' || (*p >= '0' && *p <= '9')) return number(v);
        return fail("unexpected character");
    }
  }
};

}  // namespace

bool parse(std::string_view text, Value& out, std::string* err, int max_depth) {
  Parser ps{text.data(), text.data() + text.size(), max_depth, {}};
  out = Value{};
  bool ok = ps.value(out, 0);
  if (ok) {
    ps.ws();
    if (ps.p != ps.end) ok = ps.fail("trailing characters");
  }
  if (!ok && err) {
    *err = ps.err + " at offset " + std::to_string(ps.p - text.data());
  }
  return ok;
}

// ---------------------------------------------------------------- writer

std::string python_float_repr(double d) {
  if (std::isnan(d)) return "nan";
  if (std::isinf(d)) return d < 0 ? "-inf" : "inf";
  char buf[64];
  auto r = std::to_chars(buf, buf + sizeof(buf), d, std::chars_format::scientific);
  std::string_view s(buf, static_cast<size_t>(r.ptr - buf));
  std::string out;
  if (!s.empty() && s[0] == '-') {
    out.push_back('-');
    s.remove_prefix(1);
  }
  const size_t epos = s.find('e');
  std::string digits;
  for (char c : s.substr(0, epos))
    if (c != '.') digits.push_back(c);
  int exp = 0;
  std::from_chars(s.data() + epos + 1 + (s[epos + 1] == '+' ? 1 : 0), s.data() + s.size(), exp);
  if (exp >= -4 && exp < 16) {
    if (exp >= 0) {
      const size_t ip = static_cast<size_t>(exp) + 1;
      if (digits.size() <= ip) {
        out += digits;
        out.append(ip - digits.size(), '0');
        out += ".0";
      } else {
        out.append(digits, 0, ip);
        out.push_back('.');
        out.append(digits, ip, std::string::npos);
      }
    } else {
      out += "0.";
      out.append(static_cast<size_t>(-exp - 1), '0');
      out += digits;
    }
  } else {
    out.push_back(digits[0]);
    if (digits.size() > 1) {
      out.push_back('.');
      out.append(digits, 1, std::string::npos);
    }
    char eb[16];
    std::snprintf(eb, sizeof(eb), "e%c%02d", exp < 0 ? '-' : '+', exp < 0 ? -exp : exp);
    out += eb;
  }
  return out;
}

void write_string(std::string_view s, std::string& out) {
  static const char* kHex = "0123456789abcdef";
  out.push_back('"');
  const char* run = s.data();
  const char* end = s.data() + s.size();
  for (const char* p = run; p < end; ++p) {
    const unsigned char c = static_cast<unsigned char>(*p);
    if (c >= 0x20 && c != '"' && c != '\\') continue;
    out.append(run, p);
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      case '\b': out += "\\b"; break;
      case '\f': out += "\\f"; break;
      default: {
        const char esc[6] = {'\\', 'u', '0', '0', kHex[c >> 4], kHex[c & 15]};
        out.append(esc, 6);
      }
    }
    run = p + 1;
  }
  out.append(run, end);
  out.push_back('"');
}

namespace {

void write_number(const Value& v, std::string& out, Style style) {
  if (style == Style::python && !v.is_int()) {
    if (std::isnan(v.num)) {
      out += "NaN";
    } else if (std::isinf(v.num)) {
      out += v.num < 0 ? "-Infinity" : "Infinity";
    } else {
      out += python_float_repr(v.num);
    }
    return;
  }
  if (!v.str.empty()) {
    out += v.str;
  } else {
    out += python_float_repr(v.num);
  }
}

void write_impl(const Value& v, std::string& out, Style style) {
  const bool py = style == Style::python;
  switch (v.type) {
    case Type::null: out += "null"; return;
    case Type::boolean: out += v.b ? "true" : "false"; return;
    case Type::number: write_number(v, out, style); return;
    case Type::string: write_string(v.str, out); return;
    case Type::array: {
      out.push_back('[');
      for (size_t i = 0; i < v.arr.size(); ++i) {
        if (i) out += py ? ", " : ",";
        write_impl(v.arr[i], out, style);
      }
      out.push_back(']');
      return;
    }
    case Type::object: {
      out.push_back('{');
      for (size_t i = 0; i < v.obj.size(); ++i) {
        if (i) out += py ? ", " : ",";
        write_string(v.obj[i].first, out);
        out += py ? ": " : ":";
        write_impl(v.obj[i].second, out, style);
      }
      out.push_back('}');
      return;
    }
  }
}

}  // namespace

void write(const Value& v, std::string& out, Style style) { write_impl(v, out, style); }

std::string dump(const Value& v, Style style) {
  std::string s;
  write_impl(v, s, style);
  return s;
}

}  // namespace xenithium:json
