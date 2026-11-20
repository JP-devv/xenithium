// Xenithium · gate route-record: the hot-expert ranking recorded while serving (model/route_record.hpp).
// Written for this engine. Host only: no file before the first save, one read per expert per step, the fold and its
// decay per active period, the file written atomically and read back by the next record, idle periods neither decay nor
// write, a file of another shape neither read nor overwritten.
#include <unistd.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>

#include "xenithium/model/route_counts.hpp"
#include "xenithium/model/route_record.hpp"

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

bool near(double a, double b) { return std::fabs(a - b) <= 1e-9 * std::max(1.0, std::fabs(b)); }

}  // namespace

int main() {
  namespace fs = std::filesystem;
  using xenithium::model::RouteRecord;
  const char *tmp = std::getenv("TMPDIR");
  const fs::path dir = fs::path(tmp ? tmp : "/tmp") / ("xen-route-record-" + std::to_string(getpid()));
  fs::remove_all(dir);
  const std::string file = (dir / "experts" / "hot-rank.xrc").string();  // the folder does not exist yet either
  const int L = 3, E = 8;
  const double d = std::pow(0.5, 1.0 / 60);

  {  // a first run: no file; counts, then the first save creates it (and its folder)
    RouteRecord r(L, E, file, 0.05);
    CHECK(r.ok() && !r.loaded() && !fs::exists(file), "no file: not loaded, nothing written yet (%s)", r.status().c_str());
    const int32_t a[] = {1, 2, 2, 1, 7, -1, 9};  // two rows naming 1 and 2: one read each; out-of-range ids ignored
    r.add_step(0, a, 7);
    const int32_t b[] = {2, 3};
    r.add_step(0, b, 2);  // the next step: 2 counts again
    r.add_step(2, b, 2);
    CHECK(near(r.pending(), 7), "one read per expert per step: pending %g (want 7)", r.pending());
    CHECK(!r.due(), "not due before the period");
    std::this_thread::sleep_for(std::chrono::milliseconds(60));
    CHECK(r.due(), "due after the period");
    CHECK(r.save() && fs::exists(file) && !fs::exists(file + ".tmp"), "the first save writes the file (no temporary left): %s", r.status().c_str());
    const auto c = xenithium::model::read_route_counts(file, L, E);
    CHECK(near(c[1], 1) && near(c[2], 2) && near(c[3], 1) && near(c[7], 1) && near(c[2 * E + 2], 1) && near(c[2 * E + 3], 1) && near(c[E + 1], 0),
          "the file holds the period's reads (layer 0: e1 1, e2 2, e3 1, e7 1; layer 2: e2, e3 1)");
    const auto t0 = fs::last_write_time(file);
    CHECK(r.save() && fs::last_write_time(file) == t0 && r.periods() == 1, "an idle period: no decay, no write");
    // a period with reads: the old scores decay once
    const int32_t e[] = {5};
    r.add_step(1, e, 1);
    CHECK(r.save(), "second save");
    const auto c2 = xenithium::model::read_route_counts(file, L, E);
    CHECK(near(c2[2], 2 * d) && near(c2[E + 5], 1) && r.periods() == 2, "fold: score x 0.5^(1/60) + reads (e2 %.6f, want %.6f)", c2[2], 2 * d);
  }
  {  // the next run reads it back and continues
    RouteRecord r(L, E, file, 60);
    CHECK(r.ok() && r.loaded() && near(r.scores()[2], 2 * d) && near(r.scores()[E + 5], 1), "the next run starts from the saved scores");
    std::vector<double> delta(static_cast<size_t>(L) * E, 0.0);
    delta[4] = 10;  // a device histogram's counts
    r.add_counts(delta);
    CHECK(r.save(), "save after add_counts");
    const auto c = xenithium::model::read_route_counts(file, L, E);
    CHECK(near(c[4], 10) && near(c[2], 2 * d * d), "add_counts folds like add_step (e4 %g, e2 %.6f)", c[4], c[2]);
  }
  {  // another model's shape: not read, never overwritten
    const auto before = fs::file_size(file);
    RouteRecord r(L + 1, E, file, 0);
    CHECK(!r.ok() && !r.loaded(), "a file of another shape is refused (%s)", r.status().c_str());
    const int32_t a[] = {1};
    r.add_step(0, a, 1);
    CHECK(!r.save() && fs::file_size(file) == before, "and never overwritten");
  }
  {  // no path: kept in memory
    RouteRecord r(L, E, "", 0);
    const int32_t a[] = {0};
    r.add_step(0, a, 1);
    CHECK(r.save() && near(r.scores()[0], 1), "an empty path keeps the scores in memory");
  }
  fs::remove_all(dir);
  std::printf("%s: %d checks, %d failures\n", g_failures ? "FAIL" : "PASS", g_checks, g_failures);
  return g_failures ? 1 : 0;
}
