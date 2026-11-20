#include "format.hpp"

#include <cmath>
#include <cstdio>

namespace xenithium::bench {
namespace {

constexpr double kThousand = 1e3, kMillion = 1e6;

// "%.1f" without a trailing ".0": 32.0 -> "32", 1.5 -> "1.5".
std::string one_decimal(double value) {
  char buf[32];
  std::snprintf(buf, sizeof(buf), "%.1f", value);
  std::string text = buf;
  if (text.ends_with(".0")) text.resize(text.size() - 2);
  return text;
}

}  // namespace

std::string with_thousands(int64_t value) {
  std::string digits = std::to_string(value);
  for (int at = static_cast<int>(digits.size()) - 3; at > (value < 0 ? 1 : 0); at -= 3) digits.insert(static_cast<size_t>(at), ",");
  return digits;
}

std::string format_number(double value) {
  if (!std::isfinite(value)) return "-";
  const double magnitude = std::abs(value);
  if (magnitude >= kThousand || value == std::floor(value)) return with_thousands(static_cast<int64_t>(std::llround(value)));  // a count stays a count
  char buf[32];
  std::snprintf(buf, sizeof(buf), magnitude >= 100 ? "%.0f" : magnitude >= 10 ? "%.1f" : "%.2f", value);
  return buf;
}

std::string short_number(double value) {
  const double magnitude = std::abs(value);
  if (magnitude >= kMillion) return one_decimal(value / kMillion) + "M";
  if (magnitude >= kThousand) return one_decimal(value / kThousand) + "K";
  return magnitude >= 10 || value == std::floor(value) ? one_decimal(value) : format_number(value);
}

}  // namespace xenithium:bench
