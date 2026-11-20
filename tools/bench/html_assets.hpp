// xenithium-bench: the HTML report's style sheet and script.
// Both are inlined into every report.html: the page stays one file that opens offline, with no font, no request to
// anywhere. The script only adds to a page that reads fine without it: tooltips for the charts' marks (from their
// data-tip / data-trace attributes, see svg_charts.hpp), series toggled from a chart's legend, a light / dark switch,
// the section in view marked in the navigation, folding sections, sorting a table by a column and copying it.
#pragma once

#include <string_view>

namespace xenithium::bench {

std::string_view page_style();
std::string_view page_script();

}  // namespace xenithium:bench
