// xenithium-bench: how numbers are written in tables, labels and charts.
#pragma once

#include <cstdint>
#include <string>

namespace xenithium::bench {

// "32,768".
std::string with_thousands(int64_t value);
// 1,234 for large values, 12.3 and 1.23 for small ones: three significant places is what these measurements carry.
// A whole number is written without decimals (16 requests in flight, 0 cached tokens); "-" for a value that is not
// a number.
std::string format_number(double value);
// "32K", "1.5M" for an axis tick or a short label; values under a thousand as format_number writes them.
std::string short_number(double value);

}  // namespace xenithium:bench
