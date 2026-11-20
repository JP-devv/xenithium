#include "stats.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace xenithium::bench {

double percentile(std::span<const double> sorted, double p) {
  if (sorted.empty()) return 0;
  const double at = std::clamp(p, 0.0, 1.0) * static_cast<double>(sorted.size() - 1);
  const size_t lo = static_cast<size_t>(std::floor(at)), hi = std::min(lo + 1, sorted.size() - 1);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (at - static_cast<double>(lo));
}

Summary summarize(std::span<const double> samples) {
  Summary s;
  if (samples.empty()) return s;
  std::vector<double> sorted(samples.begin(), samples.end());
  std::ranges::sort(sorted);
  s.n = static_cast<int>(sorted.size());
  s.min = sorted.front(), s.max = sorted.back();
  s.mean = std::accumulate(sorted.begin(), sorted.end(), 0.0) / static_cast<double>(sorted.size());
  s.median = percentile(sorted, 0.5), s.p10 = percentile(sorted, 0.1), s.p90 = percentile(sorted, 0.9);
  return s;
}

double spread_pct(const Summary &s) {
  if (s.n < 2 || s.median == 0) return 0;
  return 100.0 * (s.p90 - s.p10) / 2 / std::abs(s.median);
}

}  // namespace xenithium:bench
