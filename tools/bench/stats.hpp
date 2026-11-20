// xenithium-bench: the summary of a metric's samples.
#pragma once

#include <span>

namespace xenithium::bench {

struct Summary {
  int n = 0;
  double median = 0, mean = 0, min = 0, max = 0;
  double p10 = 0, p90 = 0;  // linear interpolation between order statistics
};

// The value at fraction p (0..1) of ascending `sorted`; 0 for an empty span.
double percentile(std::span<const double> sorted, double p);
Summary summarize(std::span<const double> samples);
// The half-width of the samples' middle 80 % relative to the median, in percent: how far a single sample of this
// metric strays. `compare` calls a change real only when it is larger than both runs' spread.
double spread_pct(const Summary &s);

}  // namespace xenithium:bench
