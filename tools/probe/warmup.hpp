// Probe helper: time-based warm-up before any timed loop.
// After ~2 s idle (e.g. while a CPU reference runs) a B70 needs ~25 ms of sustained
// load to return to full speed: the MoE op read 225 us/iteration for its first ~150
// iterations and 51 us after, and a 3-iteration warm-up put 40 % of a 256-iteration
// sample in the slow phase (median looked fine, p99 was 3.3x). Warm by wall time.
#pragma once

#include <chrono>

namespace xenithium::probe {

inline constexpr double kWarmMs = 100.0;

// Calls step (which must launch AND wait) until `ms` of wall time and at least
// 3 calls have passed. Returns the number of calls.
template <typename F>
int warm_for(F &&step, double ms = kWarmMs) {
  const auto t0 = std::chrono::steady_clock::now();
  int n = 0;
  do {
    step();
    ++n;
  } while (n < 3 || std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() < ms);
  return n;
}

}  // namespace xenithium:probe
