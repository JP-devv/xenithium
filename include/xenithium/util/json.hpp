// JSON DOM, parser and writer for the server and the chat template.
// Objects keep insertion order and Python's duplicate-key rule (the last value wins,
// at the first key's position), because the chat template's `tojson` must reproduce
// json.dumps of what the client sent byte for byte. Numbers keep their source
// lexeme, so integers round-trip exactly at any size.
#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace xenithium::json {

enum class Type : uint8_t { null, boolean, number, string, array, object };

struct Member;

struct Value {
  Type type = Type::null;
  bool b = false;
  double num = 0.0;
  std::string str;  // string payload, or a number's lexeme
  std::vector<Value> arr;
  std::vector<Member> obj;  // insertion order

  // Out of line: Member (which holds a Value) must be complete first.
  Value();
  ~Value();
  Value(const Value&);
  Value(Value&&) noexcept;
  Value& operator=(const Value&);
  Value& operator=(Value&&) noexcept;

  static Value boolean(bool v);
  static Value integer(int64_t v);
  static Value real(double v);
  static Value string(std::string_view s);
  static Value array();
  static Value object();

  bool is_null() const { return type == Type::null; }
  bool is_bool() const { return type == Type::boolean; }
  bool is_number() const { return type == Type::number; }
  bool is_string() const { return type == Type::string; }
  bool is_array() const { return type == Type::array; }
  bool is_object() const { return type == Type::object; }
  // A number written without a fraction or exponent (Python parses it as int).
  bool is_int() const;

  const Value* get(std::string_view key) const;  // nullptr when absent
  Value* get(std::string_view key);
  Value& set(std::string_view key, Value v);  // insert or replace in place
  Value& push(Value v);  // arrays
  bool erase(std::string_view key);

  // Convenience lookups: nullptr when absent or of another type.
  const std::string* get_string(std::string_view key) const;
  const Value* get_array(std::string_view key) const;
  const Value* get_object(std::string_view key) const;
};

// An object member; pair-like so loops read kv.first / kv.second.
struct Member {
  std::string first;
  Value second;
};

inline Value::Value() = default;
inline Value::~Value() = default;
inline Value::Value(const Value&) = default;
inline Value::Value(Value&&) noexcept = default;
inline Value& Value::operator=(const Value&) = default;
inline Value& Value::operator=(Value&&) noexcept = default;

// Strict RFC 8259 (as Python's json.loads): rejects invalid UTF-8, raw control
// characters in strings, trailing commas and trailing garbage. A lone surrogate
// escape becomes U+FFFD. Nesting is limited to `max_depth.
bool parse(std::string_view text, Value& out, std::string* err = nullptr, int max_depth = 512);

enum class Style : uint8_t {
  compact,  // separators "," and ":", numbers as their lexeme
  python,  // json.dumps(ensure_ascii=False) defaults: ", " and ": ", floats as repr
};

void write(const Value& v, std::string& out, Style style = Style::compact);
std::string dump(const Value& v, Style style = Style::compact);
// Appends `s` as a quoted JSON string (json.dumps escaping, ensure_ascii=False).
void write_string(std::string_view s, std::string& out);
// Python's repr of a float ("1.0", "1e-05", "1e+16", "inf").
std::string python_float_repr(double d);

}  // namespace xenithium:json
