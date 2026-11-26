// Xenithium · gate placement: the live expert exchange's policy (model/placement.hpp). Written for this engine. Host
// only: a layer's hot count never changes, the strongest cold expert replaces the weakest hot one only past the
// hysteresis, the per-layer budget and the round's cap hold (largest gaps first), no_enter keeps an expert out, undo
// restores the flags, the decay runs once a round, the prior seeds the scores per layer.
#include <cmath>
#include <cstdio>
#include <vector>

#include "xenithium/model/placement.hpp"

namespace {

int g_checks = 0, g_failures = 0;
#define CHECK(cond, ...)                                \
  do {                                                  \
    ++g_checks;                                         \
    if (!(cond)) {                                      \
      ++g_failures;                                     \
      std::printf("[FAIL] %s:%d: ", __FILE__, __LINE__); \
      std::printf(__VA_ARGS__);                         \
      std::printf("\n");                                \
    }                                                   \
  } while (0)

using xenithium::model::Placement;
using xenithium::model::PlacementOptions;

int hot_count(const Placement &p, int l, int E) {
  int n = 0;
  for (int e = 0; e < E; ++e) n += p.hot(l, e) ? 1 : 0;
  return n;
}

}  // namespace

int main() {
  const int L = 3, E = 8;
  // experts 0-3 hot in every layer
  std::vector<uint8_t> hot(static_cast<size_t>(L) * E, 0);
  for (int l = 0; l < L; ++l)
    for (int e = 0; e < 4; ++e) hot[static_cast<size_t>(l) * E + e] = 1;
  {  // no reads: nothing moves; reads past the hysteresis: the strongest cold replaces the weakest hot
    PlacementOptions o;
    o.budget = 1, o.max_swaps = 8, o.decay = 0.5;
    Placement p(L, E, hot, nullptr, o);
    CHECK(p.round().empty(), "no reads, no exchange");
    std::vector<double> d(static_cast<size_t>(L) * E, 0.0);
    d[0 * E + 6] = 10;  // layer 0: cold 6 read 10 times
    d[0 * E + 0] = 5, d[0 * E + 1] = 5, d[0 * E + 2] = 5, d[0 * E + 3] = 3;  // its weakest hot: 3
    d[1 * E + 7] = 2;   // layer 1: cold 7 read twice, hot all 0: 2 > 0 x 1.25 + 1
    d[2 * E + 5] = 1;   // layer 2: 1 is not > 0 x 1.25 + 1
    p.add_counts(d);
    const auto sw = p.round();
    CHECK(sw.size() == 2, "two exchanges, got %zu", sw.size());
    if (sw.size() == 2) {
      CHECK(sw[0].layer == 0 && sw[0].hot == 3 && sw[0].cold == 6, "the largest gap first (layer 0: 3 out, 6 in)");
      CHECK(sw[1].layer == 1 && sw[1].hot == 0 && sw[1].cold == 7, "layer 1: the lowest id of the weakest ties out, 7 in");
    }
    for (int l = 0; l < L; ++l) CHECK(hot_count(p, l, E) == 4, "layer %d keeps 4 hot", l);
    CHECK(p.hot(0, 6) && !p.hot(0, 3) && p.hot(1, 7) && !p.hot(1, 0), "the flags show the exchanges");
    CHECK(std::fabs(p.score(0, 6) - 5.0) < 1e-12, "decay once a round: 10 x 0.5 = %g", p.score(0, 6));
    p.undo(sw[1]);
    CHECK(p.hot(1, 0) && !p.hot(1, 7) && p.swaps() == 1, "undo restores the flags and the count");
  }
  {  // the round's cap keeps the largest gaps; the budget per layer; no_enter
    PlacementOptions o;
    o.budget = 2, o.max_swaps = 3, o.decay = 1.0;
    Placement p(L, E, hot, nullptr, o);
    std::vector<double> d(static_cast<size_t>(L) * E, 0.0);
    for (int l = 0; l < L; ++l) d[static_cast<size_t>(l) * E + 4] = 10 + l, d[static_cast<size_t>(l) * E + 5] = 20 + l;
    p.add_counts(d);
    std::vector<uint8_t> ne(static_cast<size_t>(L) * E, 0);
    ne[2 * E + 5] = 1;  // layer 2's best cold may not enter
    const auto sw = p.round(&ne);
    CHECK(sw.size() == 3, "capped at 3, got %zu", sw.size());
    if (sw.size() == 3) {
      CHECK(sw[0].layer == 1 && sw[0].cold == 5, "gap 21 first");
      CHECK(sw[1].layer == 0 && sw[1].cold == 5, "gap 20 next");
      CHECK(sw[2].layer == 2 && sw[2].cold == 4, "gap 12 (layer 2's 5 kept out)");
    }
    CHECK(!p.hot(2, 5), "no_enter held");
    for (int l = 0; l < L; ++l) CHECK(hot_count(p, l, E) == 4, "layer %d keeps 4 hot", l);
  }
  {  // the prior: per layer, scaled to prior_reads
    std::vector<double> prior(static_cast<size_t>(L) * E, 0.0);
    prior[0 * E + 1] = 300, prior[0 * E + 2] = 100;
    PlacementOptions o;
    Placement p(L, E, hot, &prior, o, 16);
    CHECK(std::fabs(p.score(0, 1) - 12.0) < 1e-12 && std::fabs(p.score(0, 2) - 4.0) < 1e-12, "prior 3:1 of 16 reads");
    CHECK(p.score(1, 0) == 0.0, "a layer without counts starts at 0");
  }
  std::printf("%s: %d check(s), %d failure(s)\n", g_failures ? "FAIL" : "PASS", g_checks, g_failures);
  return g_failures ? 1 : 0;
}
