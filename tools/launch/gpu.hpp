// GPU power and temperature, read-only, from the kernel's hwmon interface of the Intel xe driver (no
// root, no xpu-smi): /sys/class/drm/cardN/device/hwmon/hwmonM/{energy*_input,power1_cap,temp*,fan1_input}.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace xenithium::launch {

struct GpuInfo {
  int card = -1;  // N of /sys/class/drm/cardN
  std::string bdf;  // PCI address, 0000:03:00.0
  double cap_w = -1;  // power1_cap, the card's current power limit (W), -1 = unreadable
  int fan_rpm = -1;
  double temp_pkg = -1, temp_vram = -1;  // degrees C
  bool have_energy = false;
  uint64_t energy_uj = 0;  // the board's energy counter, microjoules
};

// The xe cards under `drm_root, in card order; empty when there are none.
std::vector<GpuInfo> read_gpus(const std::string &drm_root = "/sys/class/drm");

// Average power between two energy readings (W); -1 when the counter did not advance or wrapped.
double power_between(uint64_t energy0_uj, uint64_t energy1_uj, double seconds);

}  // namespace xenithium:launch
