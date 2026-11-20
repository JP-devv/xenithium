// xenithium-bench: a run as one HTML page - the headline figures, the run over time with the cards' power, each
// scenario's charts and table, and the setup.
// The page is self-contained on purpose: one file, inline style, SVG and script (html_assets.cpp), no font or other
// request. It opens offline and reads the same years from now; without its script it loses only the interaction
// (tooltips beyond the browser's own, legend toggles, sorting, folding, the theme switch). Charts are drawn from what
// each scenario declared (ScenarioResult:charts), so the page knows no scenario.
#pragma once

#include <string>

#include "report.hpp"

namespace xenithium::bench {

std::string render_html(const Run &run);

}  // namespace xenithium:bench
