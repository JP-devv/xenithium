#include "svg_charts.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>

#include "format.hpp"
#include "xenithium/util/json.hpp"

namespace xenithium::bench::svg {
namespace {

namespace json = xenithium::json;

// ---- measures shared by the charts (SVG user units; the page scales a chart to its column) ----

constexpr double kChartWidth = 560;  // a chart in a two-column grid
constexpr double kWideWidth = 1120;  // a chart across the page
constexpr double kCharWidth = 6.3;  // of the 11 px label font, on average
constexpr double kTextDrop = 4;  // from a text's middle to its baseline
constexpr double kRowPad = 9;  // between two rows or bar groups
constexpr double kBarHeight = 12;
constexpr double kBarGap = 3;  // between a group's bars
constexpr double kLegendHeight = 22;
constexpr double kMinLabelColumn = 44, kMaxLabelColumn = 190;
constexpr size_t kMaxLabelChars = 28;
constexpr int kAxisIntervals = 4;
constexpr double kSecondsPerMinute = 60, kMinutesPerHour = 60;

// A coordinate with at most one decimal: a tenth of a unit is under a tenth of a pixel on any screen.
std::string num(double value) {
  char buf[32];
  std::snprintf(buf, sizeof(buf), "%.1f", value);
  std::string text = buf;
  if (text.ends_with(".0")) text.resize(text.size() - 2);
  return text == "-0" ? "0" : text;
}

std::string shortened(std::string_view text, size_t max_chars) {
  if (text.size() <= max_chars) return std::string(text);
  return std::string(text.substr(0, max_chars - 1)) + "\xE2\x80\xA6";  // an ellipsis
}

// "177 tok/s", "3.40" (a bare number), "41.5 ms".
std::string with_unit(double value, std::string_view unit) { return format_number(value) + (unit.empty() ? "" : " " + std::string(unit)); }

// A tip as the JSON the page's script reads: {"t": title, "r": [[label, value, series]...]}.
std::string tip_json(const Tip &tip) {
  std::string out = "{\"t\":";
  json::write_string(tip.title, out);
  out += ",\"r\":[";
  for (size_t i = 0; i < tip.lines.size(); ++i) {
    out += i ? ",[" : "[";
    json::write_string(tip.lines[i].label, out);
    out += ",";
    json::write_string(tip.lines[i].value, out);
    out += "," + std::to_string(tip.lines[i].series) + "]";
  }
  return out + "]}";
}

// The same tip on one line, for a <title>: "code: decode 177 tok/s; TTFT 43 ms".
std::string tip_line(const Tip &tip) {
  std::string out = tip.title;
  for (size_t i = 0; i < tip.lines.size(); ++i) out += (i ? "; " : ": ") + tip.lines[i].label + " " + tip.lines[i].value;
  return out;
}

// The elements of one chart.
class Canvas {
 public:
  Canvas(double width, double height, std::string_view label, std::string_view extra_class = {}) {
    body_ = "<svg class=\"ch";
    if (!extra_class.empty()) body_ += " " + std::string(extra_class);
    body_ += "\" viewBox=\"0 0 " + num(width) + " " + num(height) + "\" role=\"img\" aria-label=\"" + escaped(label) + "\">";
  }
  void rect(double x, double y, double w, double h, std::string_view css) {
    body_ += "<rect class=\"" + std::string(css) + "\" x=\"" + num(x) + "\" y=\"" + num(y) + "\" width=\"" + num(std::max(w, 0.0)) + "\" height=\"" + num(h) + "\"/>";
  }
  void line(double x1, double y1, double x2, double y2, std::string_view css) {
    body_ += "<line class=\"" + std::string(css) + "\" x1=\"" + num(x1) + "\" y1=\"" + num(y1) + "\" x2=\"" + num(x2) + "\" y2=\"" + num(y2) + "\"/>";
  }
  void circle(double cx, double cy, double r, std::string_view css) {
    body_ += "<circle class=\"" + std::string(css) + "\" cx=\"" + num(cx) + "\" cy=\"" + num(cy) + "\" r=\"" + num(r) + "\"/>";
  }
  // `points` as "x,y x,y...": an open line (polyline) or a closed area (polygon).
  void polyline(const std::string &points, std::string_view css) { body_ += "<polyline class=\"" + std::string(css) + "\" points=\"" + points + "\"/>"; }
  void polygon(const std::string &points, std::string_view css) { body_ += "<polygon class=\"" + std::string(css) + "\" points=\"" + points + "\"/>"; }
  enum class Anchor { start, middle, end };
  void text(double x, double y, std::string_view css, Anchor anchor, std::string_view content) {
    body_ += "<text class=\"" + std::string(css) + "\" x=\"" + num(x) + "\" y=\"" + num(y) + "\"";
    if (anchor != Anchor::start) body_ += anchor == Anchor::middle ? " text-anchor=\"middle\"" : " text-anchor=\"end\"";
    body_ += ">" + escaped(content) + "</text>";
  }
  // A group the reader can point at (and reach with the keyboard): what is drawn until end_group, over a transparent
  // hit area, with the tip the page's script shows. `attributes` are added to the group as they are (` data-href=..`).
  void begin_group(std::string_view css, const Tip &tip, double hit_x, double hit_y, double hit_w, double hit_h, std::string_view attributes = {}) {
    body_ += "<g class=\"hov";
    if (!css.empty()) body_ += " " + std::string(css);
    body_ += "\" tabindex=\"0\" data-tip=\"" + escaped(tip_json(tip)) + "\"" + std::string(attributes) + "><title>" + escaped(tip_line(tip)) + "</title>";
    rect(hit_x, hit_y, hit_w, hit_h, "hit");
  }
  void end_group() { body_ += "</g>"; }
  // Raw markup the chart composed itself (a group with a data attribute of its own).
  void raw(std::string_view markup) { body_ += markup; }
  std::string finish() { return std::move(body_) + "</svg>"; }

 private:
  std::string body_;
};
using Anchor = Canvas::Anchor;

std::string series_class(std::string_view prefix, size_t index) { return std::string(prefix) + std::to_string(index % kSeriesColors); }
// The classes of a series' marks: its colour and the key the legend toggles.
std::string mark_class(std::string_view prefix, size_t index) { return series_class(prefix, index) + " k" + std::to_string(index); }

// The width of the label column that fits the longest label.
double label_column(const std::vector<std::string> &labels) {
  size_t longest = 0;
  for (const std::string &label : labels) longest = std::max(longest, std::min(label.size(), kMaxLabelChars));
  return std::clamp(static_cast<double>(longest) * kCharWidth + 10, kMinLabelColumn, kMaxLabelColumn);
}

// The smallest "round" number not under `value` that divides into kAxisIntervals round steps (1, 2, 2.5 or 5 x 10^k).
double axis_top(double value) {
  if (!(value > 0)) return 1;
  const double raw_step = value / kAxisIntervals;
  const double magnitude = std::pow(10.0, std::floor(std::log10(raw_step)));
  for (const double m : {1.0, 2.0, 2.5, 5.0, 10.0})
    if (m * magnitude >= raw_step) return m * magnitude * kAxisIntervals;
  return value;
}

// A row of "■ name" entries from `x` on, each a button that shows or hides its series; nothing when there is one
// series (its name is the chart's title).
void legend(Canvas &c, double x, double y, const std::vector<std::string> &names) {
  if (names.size() < 2) return;
  for (size_t i = 0; i < names.size(); ++i) {
    const double w = 15 + static_cast<double>(names[i].size()) * kCharWidth;
    c.raw("<g class=\"lg\" data-k=\"" + std::to_string(i) + "\" tabindex=\"0\" role=\"button\" aria-label=\"show or hide " + escaped(names[i]) + "\">");
    c.rect(x - 2, y - 12, w + 4, 16, "hit");
    c.rect(x, y - 9, 10, 10, series_class("s", i));
    c.text(x + 15, y, "t", Anchor::start, names[i]);
    c.raw("</g>");
    x += w + 16;
  }
}

std::string point(double x, double y) { return num(x) + "," + num(y); }

// An x value as the tooltip writes it: "32,000", "0.5".
std::string x_value_text(double x) { return x == std::floor(x) && std::abs(x) < 1e15 ? with_thousands(static_cast<int64_t>(x)) : format_number(x); }

std::string titled(std::string_view name, std::string_view value) { return name.empty() ? std::string(value) : std::string(name) + " " + std::string(value); }

}  // namespace

std::string escaped(std::string_view text) {
  std::string out;
  out.reserve(text.size());
  for (const char c : text) {
    switch (c) {
      case '&': out += "&amp;"; break;
      case '<': out += "&lt;"; break;
      case '>': out += "&gt;"; break;
      case '"': out += "&quot;"; break;
      default: out.push_back(c);
    }
  }
  return out;
}

std::string duration_text(double seconds) {
  const long total = std::lround(seconds);
  const long minutes = total / static_cast<long>(kSecondsPerMinute), hours = minutes / static_cast<long>(kMinutesPerHour);
  char buf[48];
  if (hours > 0) std::snprintf(buf, sizeof(buf), "%ld h %ld min", hours, minutes % static_cast<long>(kMinutesPerHour));
  else if (minutes > 0) std::snprintf(buf, sizeof(buf), "%ld min %ld s", minutes, total % static_cast<long>(kSecondsPerMinute));
  else std::snprintf(buf, sizeof(buf), "%ld s", total);
  return buf;
}

std::string bar_chart(const std::vector<std::string> &all_categories, const std::vector<BarSeries> &all_series, const Axes &axes) {
  constexpr double kValueColumn = 70;  // room after the longest bar for its value
  // a category no series has a value for (a summary row of another metric) is left out
  std::vector<std::string> categories;
  std::vector<BarSeries> series;
  for (const BarSeries &s : all_series) series.push_back({s.name, {}});
  for (size_t row = 0; row < all_categories.size(); ++row) {
    const auto has_value = [row](const BarSeries &s) { return row < s.values.size() && std::isfinite(s.values[row]); };
    if (std::ranges::none_of(all_series, has_value)) continue;
    categories.push_back(all_categories[row]);
    for (size_t s = 0; s < all_series.size(); ++s) series[s].values.push_back(has_value(all_series[s]) ? all_series[s].values[row] : std::nan(""));
  }
  const double label_w = label_column(categories), top = series.size() > 1 ? kLegendHeight : 2;
  const double group_h = static_cast<double>(series.size()) * (kBarHeight + kBarGap) - kBarGap + kRowPad;
  const double bar_x = label_w, bar_room = kChartWidth - label_w - kValueColumn;
  double largest = 0;
  for (const BarSeries &s : series)
    for (const double v : s.values)
      if (std::isfinite(v)) largest = std::max(largest, v);

  Canvas c(kChartWidth, top + static_cast<double>(categories.size()) * group_h, "bars");
  std::vector<std::string> names;
  for (const BarSeries &s : series) names.push_back(s.name);
  legend(c, bar_x, 12, names);
  for (size_t row = 0; row < categories.size(); ++row) {
    const double y = top + static_cast<double>(row) * group_h;
    Tip tip{titled(axes.x_name, categories[row]), {}};
    for (size_t s = 0; s < series.size(); ++s)
      if (std::isfinite(series[s].values[row])) tip.lines.push_back({series[s].name, with_unit(series[s].values[row], axes.unit), static_cast<int>(s)});
    c.begin_group("bar", tip, 0, y - kRowPad / 2, kChartWidth, group_h);
    c.text(label_w - 8, y + (group_h - kRowPad) / 2 + kTextDrop, "t", Anchor::end, shortened(categories[row], kMaxLabelChars));
    for (size_t s = 0; s < series.size(); ++s) {
      const double v = series[s].values[row];
      if (!std::isfinite(v)) continue;
      const double bar_y = y + static_cast<double>(s) * (kBarHeight + kBarGap);
      const double w = largest > 0 ? std::max(1.0, v / largest * bar_room) : 1.0;
      c.rect(bar_x, bar_y, w, kBarHeight, mark_class("s", s));
      c.text(bar_x + w + 6, bar_y + kBarHeight / 2 + kTextDrop, "v k" + std::to_string(s), Anchor::start, format_number(v));
    }
    c.end_group();
  }
  return c.finish();
}

std::string line_chart(const std::vector<LineSeries> &series, bool log_x, const Axes &axes) {
  constexpr double kHeight = 262, kLeft = 50, kRight = 22, kBottom = 42, kInset = 16;
  constexpr double kMinTickSpacing = 38;  // between two x labels
  constexpr size_t kMaxLabelledPoints = 8, kMaxLabelledSeries = 2;
  constexpr double kLabelAbove = -8, kLabelBelow = 16;  // a value's baseline against its point
  constexpr double kPointRadius = 3.2;
  const double top = (series.size() > 1 ? kLegendHeight : 0) + 22;
  const double plot_bottom = kHeight - kBottom;

  double x_min = 0, x_max = 0, y_max = 0;
  bool any = false;
  const auto scaled = [log_x](double x) { return log_x ? std::log(std::max(x, 1e-9)) : x; };
  std::vector<LinePoint> xs;  // every x of any series, ascending, once
  for (const LineSeries &s : series)
    for (const LinePoint &p : s.points) {
      const double sx = scaled(p.x);
      x_min = any ? std::min(x_min, sx) : sx, x_max = any ? std::max(x_max, sx) : sx, y_max = std::max(y_max, p.y);
      any = true;
      if (std::ranges::none_of(xs, [&](const LinePoint &have) { return have.x == p.x; })) xs.push_back(p);
    }
  std::ranges::sort(xs, {}, &LinePoint::x);
  const double y_top = axis_top(y_max);
  const auto x_of = [&](double x) {
    return x_max > x_min ? kLeft + kInset + (scaled(x) - x_min) / (x_max - x_min) * (kChartWidth - kLeft - kRight - 2 * kInset) : (kLeft + kChartWidth - kRight) / 2;
  };
  const auto y_of = [&](double y) { return plot_bottom - y / y_top * (plot_bottom - top); };

  Canvas c(kChartWidth, kHeight, "line");
  std::vector<std::string> names;
  for (const LineSeries &s : series) names.push_back(s.name);
  legend(c, kLeft, 12, names);
  if (!axes.unit.empty()) c.text(kLeft - 7, top - 12, "t", Anchor::end, axes.unit);
  for (int tick = 0; tick <= kAxisIntervals; ++tick) {
    const double value = y_top * tick / kAxisIntervals, y = y_of(value);
    c.line(kLeft, y, kChartWidth - kRight, y, tick == 0 ? "ax" : "g");
    c.text(kLeft - 7, y + kTextDrop, "t", Anchor::end, short_number(value));
  }
  double last_tick_x = -kMinTickSpacing;
  for (const LinePoint &p : xs) {
    const double x = x_of(p.x);
    if (x - last_tick_x < kMinTickSpacing) continue;
    c.text(x, plot_bottom + 17, "t", Anchor::middle, p.x_text);
    last_tick_x = x;
  }
  if (!axes.x_name.empty()) c.text(kChartWidth - kRight, plot_bottom + 34, "t", Anchor::end, axes.x_name + (log_x ? " (log scale)" : "") + " \xE2\x86\x92");
  for (size_t s = 0; s < series.size(); ++s) {
    std::string points;
    for (const LinePoint &p : series[s].points) points += (points.empty() ? "" : " ") + point(x_of(p.x), y_of(p.y));
    if (series[s].points.size() > 1) c.polyline(points, mark_class("l", s));
  }
  // a column per x the reader can point at: a guide line, every series' value there, the points drawn larger
  for (size_t i = 0; i < xs.size(); ++i) {
    const double x = x_of(xs[i].x);
    const double from = i == 0 ? kLeft : (x_of(xs[i - 1].x) + x) / 2, to = i + 1 == xs.size() ? kChartWidth - kRight : (x + x_of(xs[i + 1].x)) / 2;
    Tip tip{titled(axes.x_name, x_value_text(xs[i].x)), {}};
    for (size_t s = 0; s < series.size(); ++s)
      for (const LinePoint &p : series[s].points)
        if (p.x == xs[i].x) tip.lines.push_back({series[s].name, with_unit(p.y, axes.unit), static_cast<int>(s)});
    c.begin_group("col", tip, from, top - 6, to - from, plot_bottom - top + 6);
    c.line(x, top - 6, x, plot_bottom, "guide");
    for (size_t s = 0; s < series.size(); ++s)
      for (const LinePoint &p : series[s].points)
        if (p.x == xs[i].x) c.circle(x, y_of(p.y), kPointRadius, mark_class("s", s) + " pt");
    c.end_group();
  }
  // of two series, the lower value at an x is labelled below its point and the higher above, so the two labels part
  // where the lines meet or cross (equal values: the second series below)
  const auto lower_at = [&](size_t s, const LinePoint &p) {
    if (series.size() != kMaxLabelledSeries) return false;
    const std::vector<LinePoint> &others = series[1 - s].points;
    const auto other = std::ranges::find_if(others, [&](const LinePoint &q) { return q.x == p.x; });
    return other != others.end() && (p.y < other->y || (p.y == other->y && s > 0));
  };
  for (size_t s = 0; s < series.size(); ++s) {
    if (series.size() > kMaxLabelledSeries || series[s].points.size() > kMaxLabelledPoints) continue;
    for (const LinePoint &p : series[s].points) {
      const bool below = lower_at(s, p) && y_of(p.y) + kLabelBelow < plot_bottom - 2;  // unless below runs into the axis
      c.text(x_of(p.x), y_of(p.y) + (below ? kLabelBelow : kLabelAbove), "v k" + std::to_string(s), Anchor::middle, format_number(p.y));
    }
  }
  return c.finish();
}

std::string range_chart(const std::vector<RangeRow> &rows, const Axes &axes) {
  constexpr double kRowHeight = 26, kValueColumn = 118, kBandHeight = 10, kMedianHeight = 16, kTop = 20;
  std::vector<std::string> labels;
  double largest = 0;
  for (const RangeRow &row : rows) labels.push_back(row.label), largest = std::max(largest, row.summary.max);
  const double label_w = label_column(labels), track_x = label_w, track_w = kChartWidth - label_w - kValueColumn;
  const auto x_of = [&](double v) { return track_x + (largest > 0 ? v / largest * track_w : 0); };

  Canvas c(kChartWidth, kTop + static_cast<double>(rows.size()) * kRowHeight, "range");
  c.text(track_x, 11, "t", Anchor::start, "hair: min to max \xC2\xB7 band: p10 to p90 \xC2\xB7 tick: median");
  c.text(kChartWidth, 11, "t", Anchor::end, "median / max");
  for (size_t i = 0; i < rows.size(); ++i) {
    const Summary &s = rows[i].summary;
    const double row_top = kTop + static_cast<double>(i) * kRowHeight, mid = row_top + kRowHeight / 2;
    const Tip tip{titled(axes.x_name, rows[i].label),
                  {{"median", with_unit(s.median, axes.unit), 0},
                   {"p10 to p90", format_number(s.p10) + " to " + with_unit(s.p90, axes.unit)},
                   {"min to max", format_number(s.min) + " to " + with_unit(s.max, axes.unit)},
                   {"mean", with_unit(s.mean, axes.unit)},
                   {"samples", std::to_string(s.n)}}};
    c.begin_group("rng", tip, 0, row_top, kChartWidth, kRowHeight);
    c.text(label_w - 8, mid + kTextDrop, "t", Anchor::end, shortened(rows[i].label, kMaxLabelChars));
    c.line(track_x, mid, track_x + track_w, mid, "g");
    if (s.n > 0) {
      c.line(x_of(s.min), mid, x_of(s.max), mid, "hair");
      c.rect(x_of(s.p10), mid - kBandHeight / 2, std::max(2.0, x_of(s.p90) - x_of(s.p10)), kBandHeight, "band");
      c.line(x_of(s.median), mid - kMedianHeight / 2, x_of(s.median), mid + kMedianHeight / 2, "med");
      c.text(kChartWidth, mid + kTextDrop, "v", Anchor::end, format_number(s.median) + " / " + format_number(s.max));
    }
    c.end_group();
  }
  return c.finish();
}

std::string dot_chart(const std::vector<DotRow> &rows) {
  constexpr double kRowHeight = 24, kCountColumn = 52, kMaxPitch = 16, kMaxRadius = 5;
  std::vector<std::string> labels;
  size_t most = 1;
  for (const DotRow &row : rows) labels.push_back(row.label), most = std::max(most, row.yes.size());
  const double label_w = label_column(labels);
  const double pitch = std::min(kMaxPitch, (kChartWidth - label_w - kCountColumn) / static_cast<double>(most));
  const double radius = std::min(kMaxRadius, pitch / 2 - 1);

  Canvas c(kChartWidth, static_cast<double>(rows.size()) * kRowHeight + 4, "dots");
  for (size_t i = 0; i < rows.size(); ++i) {
    const double row_top = static_cast<double>(i) * kRowHeight + 2, mid = row_top + kRowHeight / 2;
    const auto yes = static_cast<size_t>(std::ranges::count(rows[i].yes, true));
    std::string pattern;  // the samples in order, ✓ / ✗
    for (const bool y : rows[i].yes) pattern += y ? "\xE2\x9C\x93" : "\xE2\x9C\x97";
    Tip tip{rows[i].label, {{"yes", std::to_string(yes) + " of " + std::to_string(rows[i].yes.size())}, {"in order", pattern}}};
    if (!rows[i].wrong.empty()) {
      std::string names;
      for (const std::string &name : rows[i].wrong) names += (names.empty() ? "" : ", ") + name;
      tip.lines.push_back({"wrong", names});
    }
    c.begin_group("dot", tip, 0, row_top, kChartWidth, kRowHeight);
    c.text(label_w - 8, mid + kTextDrop, "t", Anchor::end, shortened(rows[i].label, kMaxLabelChars));
    for (size_t d = 0; d < rows[i].yes.size(); ++d) c.circle(label_w + pitch / 2 + static_cast<double>(d) * pitch, mid, radius, rows[i].yes[d] ? "yes" : "no");
    c.text(label_w + static_cast<double>(most) * pitch + 10, mid + kTextDrop, "v", Anchor::start, std::to_string(yes) + "/" + std::to_string(rows[i].yes.size()));
    c.end_group();
  }
  return c.finish();
}

namespace {

// The power trace as the page's script reads it from data-trace: the plot's left edge and width, the run's length,
// the card names, the points ([seconds, MHz, watts of card 0...]) and the spans ([label, from, to]).
std::string trace_json(const std::vector<TimelineSpan> &spans, double total_s, const TimelinePower &power, double left, double width) {
  std::string out = "{\"x0\":" + num(left) + ",\"w\":" + num(width) + ",\"t\":" + num(total_s) + ",\"c\":[";
  for (size_t card = 0; card < power.watts.size(); ++card) {
    if (card) out += ",";
    json::write_string(card < power.card_names.size() ? power.card_names[card] : "card " + std::to_string(card), out);
  }
  out += "],\"p\":[";
  for (size_t p = 0; p < power.at_s.size(); ++p) {
    out += (p ? ",[" : "[") + num(power.at_s[p]) + "," + num(p < power.mhz.size() ? power.mhz[p] : 0);
    for (const auto &card : power.watts) out += "," + num(p < card.size() ? card[p] : 0);
    out += "]";
  }
  out += "],\"s\":[";
  for (size_t i = 0; i < spans.size(); ++i) {
    out += i ? ",[" : "[";
    json::write_string(spans[i].label, out);
    out += "," + num(spans[i].from_s) + "," + num(spans[i].to_s) + "]";
  }
  return out + "]}";
}

}  // namespace

std::string timeline(const std::vector<TimelineSpan> &spans, double total_s, const TimelinePower &power) {
  constexpr double kLeft = 50, kRight = 6, kCalloutRow = 14, kCalloutRows = 3, kStripHeight = 30, kAxisHeight = 22, kPowerHeight = 150, kClockHeight = 44, kSectionGap = 16;
  constexpr int kTimeTicks = 8;
  const bool has_power = !power.at_s.empty() && !power.watts.empty();
  const bool has_clock = has_power && std::ranges::any_of(power.mhz, [](double mhz) { return mhz > 0; });
  const double plot_w = kWideWidth - kLeft - kRight;
  const auto x_of = [&](double s) { return kLeft + (total_s > 0 ? std::clamp(s / total_s, 0.0, 1.0) * plot_w : 0); };
  const double strip_top = kCalloutRow * kCalloutRows + 4, strip_bottom = strip_top + kStripHeight;
  const double power_top = strip_bottom + kAxisHeight + kSectionGap, power_bottom = power_top + kPowerHeight;
  const double clock_top = power_bottom + kSectionGap, clock_bottom = clock_top + kClockHeight;
  const double height = !has_power ? strip_bottom + kAxisHeight : has_clock ? clock_bottom + 4 : power_bottom + 4;

  Canvas c(kWideWidth, height, "the run over time", "wide");
  // what ran when: a span wide enough carries its name, a narrow one is named above it, on the row that has room
  double callout_free_from[static_cast<int>(kCalloutRows)] = {};
  for (size_t i = 0; i < spans.size(); ++i) {
    const double x = x_of(spans[i].from_s), w = std::max(1.0, x_of(spans[i].to_s) - x - 1.5), label_w = static_cast<double>(spans[i].label.size()) * kCharWidth;
    Tip tip{spans[i].label, {{"started at", duration_text(spans[i].from_s)}, {"took", duration_text(spans[i].to_s - spans[i].from_s)}}};
    tip.lines.insert(tip.lines.end(), spans[i].details.begin(), spans[i].details.end());
    c.begin_group("span", tip, x, strip_top, w, kStripHeight, spans[i].href.empty() ? std::string() : " data-href=\"" + escaped(spans[i].href) + "\"");
    c.rect(x, strip_top, w, kStripHeight, i % 2 ? "span1" : "span0");
    if (label_w + 10 < w) c.text(x + 6, strip_top + kStripHeight / 2 + kTextDrop, "spant", Anchor::start, spans[i].label);
    c.end_group();
    if (label_w + 10 < w) continue;
    const double label_x = std::min(x, kWideWidth - kRight - label_w);
    for (int row = static_cast<int>(kCalloutRows) - 1; row >= 0; --row) {  // the row next to the strip first
      if (label_x < callout_free_from[row]) continue;
      const double baseline = kCalloutRow * (row + 1) - 3;
      c.text(label_x, baseline, "t", Anchor::start, spans[i].label);
      c.line(x + w / 2, baseline + 3, x + w / 2, strip_top, "ax");
      callout_free_from[row] = label_x + label_w + 8;
      break;
    }
  }
  // minute ticks on a round step
  const double minutes = total_s / kSecondsPerMinute, step_min = std::max(1.0, axis_top(minutes) / kTimeTicks);
  for (double m = 0; m <= minutes; m += step_min) {
    const double x = x_of(m * kSecondsPerMinute);
    c.line(x, strip_bottom, x, strip_bottom + 5, "ax");
    c.text(x, strip_bottom + 17, "t", m == 0 ? Anchor::start : Anchor::middle, num(m) + " min");
  }
  if (!has_power) return c.finish();

  // the cards' power, stacked; the limit and the idle draw as dashed lines (the page's legend names them)
  const size_t points = power.at_s.size();
  double largest = power.cap_watts;
  for (size_t p = 0; p < points; ++p) {
    double sum = 0;
    for (const auto &card : power.watts) sum += p < card.size() ? card[p] : 0;
    largest = std::max(largest, sum);
  }
  const double watts_top = axis_top(largest);
  const auto y_of = [&](double watts) { return power_bottom - watts / watts_top * kPowerHeight; };
  for (int tick = 0; tick <= kAxisIntervals; ++tick) {
    const double watts = watts_top * tick / kAxisIntervals;
    c.line(kLeft, y_of(watts), kWideWidth - kRight, y_of(watts), tick == 0 ? "ax" : "g");
    c.text(kLeft - 7, y_of(watts) + kTextDrop, "t", Anchor::end, short_number(watts) + (tick == kAxisIntervals ? " W" : ""));
  }
  std::vector<double> below(points, 0.0);
  for (size_t card = 0; card < power.watts.size(); ++card) {
    std::string upper, lower;
    for (size_t p = 0; p < points; ++p) {
      const double w = p < power.watts[card].size() ? power.watts[card][p] : 0;
      upper += point(x_of(power.at_s[p]), y_of(below[p] + w)) + " ";
      lower = point(x_of(power.at_s[p]), y_of(below[p])) + " " + lower;
      below[p] += w;
    }
    c.polygon(upper + lower, series_class("area", card));
  }
  for (const double watts : {power.cap_watts, power.idle_watts})
    if (watts > 0) c.line(kLeft, y_of(watts), kWideWidth - kRight, y_of(watts), "cap");
  const double trace_bottom = has_clock ? clock_bottom : power_bottom;
  if (has_clock) {
    // the GPU clock on its own scale, from the lowest reading to the highest
    const auto [low_it, high_it] = std::ranges::minmax_element(power.mhz);
    const double low = *low_it, high = std::max(*high_it, low + 1);
    std::string clock;
    for (size_t p = 0; p < points && p < power.mhz.size(); ++p)
      clock += point(x_of(power.at_s[p]), clock_bottom - (power.mhz[p] - low) / (high - low) * kClockHeight) + " ";
    c.line(kLeft, clock_bottom, kWideWidth - kRight, clock_bottom, "ax");
    c.text(kLeft - 7, clock_top + kTextDrop + 2, "t", Anchor::end, short_number(high));
    c.text(kLeft - 7, clock_bottom, "t", Anchor::end, short_number(low) + " MHz");
    c.polyline(clock, "l2");
  }
  // the reader's pointer over the trace: a guide line and the readings at that moment (drawn by the page's script)
  c.raw("<g class=\"trace\" data-trace=\"" + escaped(trace_json(spans, total_s, power, kLeft, plot_w)) + "\">");
  c.rect(kLeft, power_top, plot_w, trace_bottom - power_top, "hit");
  c.line(kLeft, power_top, kLeft, trace_bottom, "tguide");
  c.raw("</g>");
  return c.finish();
}

std::string request_anatomy(const RequestAnatomy &r) {
  constexpr double kHeight = 132, kLeft = 8, kWaitShare = 0.24, kAxisY = 78, kTokenSize = 9, kTokenGap = 3;
  constexpr int kUpdatesShown = 14;
  constexpr double kSeveralTokens = 1.2;
  const double wait_end = kLeft + (kWideWidth - 2 * kLeft) * kWaitShare;
  const double pitch = (kWideWidth - kLeft - wait_end - 70) / kUpdatesShown;
  const int tokens = std::max(1, static_cast<int>(std::lround(r.tokens_per_update)));

  Canvas c(kWideWidth, kHeight, "a request as the tool times it", "wide");
  c.line(kLeft, kAxisY, kWideWidth - kLeft, kAxisY, "ax");
  // the wait for the first token
  c.begin_group("", {"Time to first token", {{"median", format_number(r.ttft_ms) + " ms", 0}, {"holds", "the connection, the queue, the prompt's processing"}}}, kLeft, kAxisY - 30,
                wait_end - kLeft, 60);
  c.rect(kLeft, kAxisY - 7, wait_end - kLeft, 14, "wait");
  c.end_group();
  c.text(kLeft, kAxisY - 16, "t", Anchor::start, "request sent");
  c.text(kLeft, kAxisY + 26, "v", Anchor::start, "time to first token  " + format_number(r.ttft_ms) + " ms");
  c.text(kLeft, kAxisY + 42, "t", Anchor::start, "connect \xC2\xB7 queue \xC2\xB7 prompt processing");
  // the updates: a tick each, with the tokens it carries stacked above
  const double stack_top = kAxisY - 14 - static_cast<double>(tokens) * (kTokenSize + kTokenGap);
  c.begin_group("",
                {"Streamed updates",
                 {{"update gap", format_number(r.update_gap_ms) + " ms", 0}, {"tokens per update", format_number(r.tokens_per_update)}, {"decode", format_number(r.decode_tps) + " tok/s"}}},
                wait_end - pitch / 2, std::min(stack_top, kAxisY - 30) - 4, pitch * kUpdatesShown, kAxisY + 12 - std::min(stack_top, kAxisY - 30));
  for (int u = 0; u < kUpdatesShown; ++u) {
    const double x = wait_end + static_cast<double>(u) * pitch;
    c.line(x, kAxisY - 9, x, kAxisY + 9, "med");
    for (int t = 0; t < tokens; ++t) c.rect(x - kTokenSize / 2, kAxisY - 14 - static_cast<double>(t + 1) * (kTokenSize + kTokenGap), kTokenSize, kTokenSize, "s0");
  }
  c.end_group();
  c.text(wait_end, kAxisY + 26, "v", Anchor::start, "an update every " + format_number(r.update_gap_ms) + " ms, " + format_number(r.tokens_per_update) + " tokens each");
  // more than one token per update is a speculative step (or a server that batches its stream)
  c.text(wait_end, kAxisY + 42, "t", Anchor::start, r.tokens_per_update > kSeveralTokens ? "several tokens per update: drafted tokens verified in one decode step" : "one token per decode step");
  c.text(kWideWidth - kLeft, kAxisY + 26, "v", Anchor::end, "= " + format_number(r.decode_tps) + " tok/s");
  c.text(kWideWidth - kLeft, kAxisY + 42, "t", Anchor::end, "decode rate after the first update");
  return c.finish();
}

}  // namespace xenithium:bench:svg
