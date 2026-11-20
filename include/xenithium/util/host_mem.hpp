// Xenithium · host RAM: what /proc/meminfo says is there, and the share a load may pin.
// The MoE archs keep their cold experts in pinned host memory,
// and the prefix cache adds its snapshots on first use. Pinned pages cannot be swapped: a plan past the host's RAM
// does not fail an allocation, it gets the server (and whatever else the OOM killer picks) killed. Loads check their
// pinned plan against MemAvailable first and refuse with the numbers instead.
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>

#include <unistd.h>

namespace xenithium::util {

struct HostMem {
  uint64_t total = 0;  // MemTotal, bytes
  uint64_t available = 0;  // MemAvailable, bytes
  bool ok = false;  // both lines read
};

inline HostMem host_mem() {
  HostMem m;
  std::FILE *f = std::fopen("/proc/meminfo", "r");
  if (!f) return m;
  char line[256];
  bool t = false, a = false;
  while (std::fgets(line, sizeof line, f)) {
    unsigned long long kb = 0;
    if (std::sscanf(line, "MemTotal: %llu kB", &kb) == 1) m.total = kb * 1024ull, t = true;
    else if (std::sscanf(line, "MemAvailable: %llu kB", &kb) == 1) m.available = kb * 1024ull, a = true;
  }
  std::fclose(f);
  m.ok = t && a;
  return m;
}

// Resident set of this process (/proc/self/statm), bytes; 0 when unreadable. Host buffers the GPU driver pins (the
// prefix-cache snapshots, cold experts) are mapped through the driver and do not show in it: MemAvailable does.
inline uint64_t self_rss() {
  std::FILE *f = std::fopen("/proc/self/statm", "r");
  if (!f) return 0;
  unsigned long long size = 0, res = 0;
  const int n = std::fscanf(f, "%llu %llu", &size, &res);
  std::fclose(f);
  return n == 2 ? res * static_cast<unsigned long long>(sysconf(_SC_PAGESIZE)) : 0;
}

// Low host RAM, the point where a serving process warns: MemAvailable under 8 % of MemTotal or under 3 GiB, whichever is
// larger. Past it the kernel starts swapping the process (decode steps go from ms to seconds) and then its OOM killer
// ends it with SIGKILL - exit status 137, no message of its own.
inline bool host_mem_low(const HostMem &m) {
  if (!m.ok || m.total == 0) return false;
  const double floor_b = std::max(0.08 * static_cast<double>(m.total), 3.0 * 1073741824.0);
  return static_cast<double>(m.available) < floor_b;
}

// Host RAM a load leaves to the OS and the other processes on the host (GiB).
inline constexpr double kHostKeepGib = 2.0;
// Host RAM a serving process takes beyond its pinned plan (its own heap, the SYCL / Level Zero runtime and driver
// allocations, the warm-up), GiB.
inline constexpr double kLoadHostOverheadGib = 2.5;

// Snapshots of `one` bytes a prefix cache of `want_gib` may hold, clamped to the host RAM available NOW (call it after
// the load's pinned allocations) minus kHostKeepGib. *clamped_gib = the GiB actually granted (== want when not cut).
inline int snap_capacity(double want_gib, double one, double *clamped_gib = nullptr) {
  if (want_gib <= 0) {
    if (clamped_gib) *clamped_gib = 0;
    return 0;
  }
  if (one <= 0) {  // snapshots without host bytes (KV pages only): no host RAM to cut
    if (clamped_gib) *clamped_gib = want_gib;
    return 1000000;
  }
  double gib = want_gib;
  const HostMem m = host_mem();
  if (m.ok) gib = std::min(gib, std::max(0.0, static_cast<double>(m.available) / 1073741824.0 - kHostKeepGib));
  if (clamped_gib) *clamped_gib = gib;
  return static_cast<int>(std::min(1e6, gib * 1073741824.0 / one));
}

}  // namespace xenithium:util
