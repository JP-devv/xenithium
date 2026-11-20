#include "params.hpp"

#include <charconv>
#include <stdexcept>

namespace xenithium::bench {
namespace {

std::invalid_argument bad(std::string_view key, std::string_view why) { return std::invalid_argument("parameter " + std::string(key) + ": " + std::string(why)); }

template <typename T>
T number(std::string_view key, std::string_view text) {
  T value{};
  const auto [end, ec] = std::from_chars(text.data(), text.data() + text.size(), value);
  if (ec != std::errc{} || end != text.data() + text.size()) throw bad(key, "\"" + std::string(text) + "\" is not a number");
  return value;
}

}  // namespace

Params::Params(std::initializer_list<std::pair<std::string, std::string>> values) {
  for (const auto &[key, value] : values) values_[key] = value;
}

std::string Params::text(std::string_view key) const {
  const auto it = values_.find(key);
  if (it == values_.end()) throw bad(key, "not declared by the scenario");
  return it->second;
}

int64_t Params::integer(std::string_view key) const { return number<int64_t>(key, text(key)); }
double Params::real(std::string_view key) const { return number<double>(key, text(key)); }

std::vector<std::string> Params::list(std::string_view key) const {
  const std::string all = text(key);
  std::vector<std::string> items;
  for (size_t at = 0; at < all.size();) {
    const size_t comma = std::min(all.find(',', at), all.size());
    if (comma > at) items.push_back(all.substr(at, comma - at));
    at = comma + 1;
  }
  return items;
}

std::vector<int64_t> Params::integers(std::string_view key) const {
  std::vector<int64_t> values;
  for (const std::string &item : list(key)) values.push_back(number<int64_t>(key, item));
  return values;
}

std::vector<double> Params::reals(std::string_view key) const {
  std::vector<double> values;
  for (const std::string &item : list(key)) values.push_back(number<double>(key, item));
  return values;
}

Params Params::merged(std::span<const ParamSpec> specs, const Params &over) {
  Params merged;
  for (const ParamSpec &spec : specs) merged.set(std::string(spec.key), std::string(spec.fallback));
  for (const auto &[key, value] : over.all()) {
    if (!merged.has(key)) throw bad(key, "the scenario has no such parameter");
    merged.set(key, value);
  }
  return merged;
}

}  // namespace xenithium:bench
