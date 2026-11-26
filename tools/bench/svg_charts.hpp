// xenithium-bench: the charts of the HTML report, as inline SVG text.
// Plain data in, one <svg> element out. Nothing here knows a scenario or a run: the report (html_report.cpp) turns
// rows and metrics into these inputs. The SVG carries no colours or fonts of its own - it names CSS classes (s0..s2
// for the series, t / v / g for labels, values and grid lines) that the page's style sheet defines, so one chart
// follows the page's light and dark themes and prints.
// The charts are interactive through the page's script (html_assets.cpp), and only through data: an element the
// reader can point at is a group of class "hov" with a data-tip (a Tip as JSON) and a transparent "hit" area; a
// series' marks carry class k<i> and a legend entry of class "lg" with data-k=<i> shows or hides them; the timeline's
// power trace carries its points in data-trace. Every tipped group also holds a <title> with the same text on one
// line, so a page without its script still answers a pointer.
#pragma once

#include <string>
#include <string_view>
#include <vector>

#include "stats.hpp"

namespace xenithium::bench::svg {

// The most series a chart draws apart (the style sheet has this many series colours).
inline constexpr size_t kSeriesColors = 3;

// What a chart element tells the reader who points at it: a heading and labelled values, each value with the colour
// of its series where it has one.
struct TipLine {
  std::string label, value;
  int series = -1;  // the swatch's series colour; -1 = none
};
struct Tip {
  std::string title;
  std::vector<TipLine> lines;
};

// The text around a chart's numbers: the unit of its values ("tok/s") and the name of what its rows or its x axis
// are ("streams", "depth").
struct Axes {
  std::string unit;
  std::string x_name;
};

// One value per category; a NaN leaves the category's bar out.
struct BarSeries {
  std::string name;
  std::vector<double> values;
};
// Horizontal bars, one group per category, each bar labelled with its value.
std::string bar_chart(const std::vector<std::string> &categories, const std::vector<BarSeries> &series, const Axes &axes);

struct LinePoint {
  double x = 0, y = 0;
  std::string x_text;  // the tick under the point ("32K")
};
struct LineSeries {
  std::string name;
  std::vector<LinePoint> points;  // ascending x
};
// Lines over a numeric x axis starting at y = 0; log_x spaces the points by the logarithm of x.
std::string line_chart(const std::vector<LineSeries> &series, bool log_x, const Axes &axes);

struct RangeRow {
  std::string label;
  Summary summary;
};
// Per row where a metric's samples lie: a hair from min to max, a band from p10 to p90, a tick at the median.
std::string range_chart(const std::vector<RangeRow> &rows, const Axes &axes);

struct DotRow {
  std::string label;
  std::vector<bool> yes;
  std::vector<std::string> wrong;  // the checks behind its misses (named in the tooltip)
};
// One dot per yes / no sample, filled when yes, with the count beside it.
std::string dot_chart(const std::vector<DotRow> &rows);

// The run over time: a strip of what ran when, and under it (when `watts` is not empty) the power the cards drew.
struct TimelineSpan {
  std::string label;
  double from_s = 0, to_s = 0;
  std::vector<TipLine> details = {};  // shown when the reader points at the span, after its start and duration
  std::string href = {};  // where a click on the span goes ("#decode"); empty = nowhere
};
struct TimelinePower {
  std::vector<double> at_s;
  std::vector<std::vector<double>> watts;  // [card][point]
  std::vector<double> mhz;  // [point]; empty or all zero = no clock strip
  std::vector<std::string> card_names;  // [card], for the trace's tooltip
  double cap_watts = 0;  // of all cards together; 0 = no line
  double idle_watts = 0;
};
std::string timeline(const std::vector<TimelineSpan> &spans, double total_s, const TimelinePower &power);

// A request as the tool times it: the wait for the first token, then updates of several tokens at an even gap.
struct RequestAnatomy {
  double ttft_ms = 0;
  double update_gap_ms = 0;
  double tokens_per_update = 0;
  double decode_tps = 0;
};
std::string request_anatomy(const RequestAnatomy &request);

// Text safe inside an element or an attribute value.
std::string escaped(std::string_view text);
// "1 min 2 s", "45 s": a span of the run for a label or a tooltip.
std::string duration_text(double seconds);

}  // namespace xenithium:bench:svg
