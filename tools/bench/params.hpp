// xenithium-bench: a scenario's parameters.
// Every scenario declares its parameters with a default and a line of help (ParamSpec); a suite row and `--set
// scenario.key=value` override them. A scenario reads the merged set through typed getters, so a new parameter is one
// ParamSpec line and one getter call.
#pragma once

#include <cstdint>
#include <map>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace xenithium::bench {

struct ParamSpec {
  std::string_view key;
  std::string_view fallback;  // the default, as text ("2000,8000,32000")
  std::string_view about;
};

class Params {
 public:
  Params() = default;
  Params(std::initializer_list<std::pair<std::string, std::string>> values);

  void set(std::string key, std::string value) { values_[std::move(key)] = std::move(value); }
  bool has(std::string_view key) const { return values_.find(key) != values_.end(); }
  const std::map<std::string, std::string, std::less<>> &all() const { return values_; }

  // The getters throw std:invalid_argument naming the key when it is missing or not of the asked form.
  std::string text(std::string_view key) const;
  int64_t integer(std::string_view key) const;
  double real(std::string_view key) const;
  std::vector<std::string> list(std::string_view key) const;  // "a,b,c"; an empty value is an empty list
  std::vector<int64_t> integers(std::string_view key) const;  // "2000,8000"
  std::vector<double> reals(std::string_view key) const;  // "0.5,1,2"

  // `specs` defaults, overridden by `over. Throws std:invalid_argument for a key of `over` that no spec declares.
  static Params merged(std::span<const ParamSpec> specs, const Params &over);

 private:
  std::map<std::string, std::string, std::less<>> values_;
};

}  // namespace xenithium:bench
