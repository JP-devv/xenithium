// Xenithium · which card a device is, and the card it runs as (XEN_EMU_CARD emulation).
// Every Fabric names its cards at start ("card 0..." on stderr, once per device and process). The engine reads the
// effective card from here: kernels size their waves by xe_cores (the emulated card's cores when emulating), and
// l0:vram_state / Device:budget_vram report the emulated card's memory, so every budget, the auto KV pool and the
// hot-expert fit size themselves for it.
// Emulation runs a smaller Battlemage card on a bigger one (a B60 or B65 on a B70; a B60 on a B65): the same Xe2 cores,
// ISA and AOT binary (bmg = IP 20.1.0, the G21 die; the G31 cards run it), fewer cores and, for the B60, less memory and
// bandwidth. What it covers: the identity, the VRAM (exact for every budget), the launch shapes that depend on the
// core count. Not emulated: the missing cores and the bandwidth (the B70's driver exposes one compute engine, and a
// kernel holding cores from another queue, process or the render engine is time-sliced with the engine's instead of
// sharing the card), the clock (B60 / B65 2,400 MHz vs 2,800; a cap needs root), the L2 size and the PCIe link.
#pragma once

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cstdint>
#include <string>

#include "xenithium/rt/l0.hpp"
#include "xenithium/util/cards.hpp"

namespace xenithium::rt {

inline constexpr int kXvePerCore = 8;  // Xe2: 8 vector engines per Xe core

// The card `info` is, with XEN_EMU_CARD applied (util:card_view; throws on a bad or too large XEN_EMU_CARD).
util::CardView card_view(const l0::DeviceInfo &info);

// Bytes of device memory the emulation hides on this device (0 without emulation; never throws).
uint64_t emulated_vram_cut(const l0::DeviceInfo &info);

// Prints card_view(info).describe to stderr the first time a process opens that device.
void log_card(size_t index, const l0::DeviceInfo &info);

// The Xe cores a kernel sizes its waves for: the device's, or the emulated card's when that is smaller. Header-inline:
// the 256-GRF kernels live in a shared library and read it too (the emulation travels in the environment).
inline int xe_cores(const sycl::device &d) {
  int hw = 0;
  try {
    if (d.has(sycl::aspect::ext_intel_gpu_eu_count) && d.has(sycl::aspect::ext_intel_gpu_eu_count_per_subslice)) {
      const int eu = static_cast<int>(d.get_info<sycl::ext::intel::info::device::gpu_eu_count>());
      const int per = static_cast<int>(d.get_info<sycl::ext::intel::info::device::gpu_eu_count_per_subslice>());
      if (eu > 0 && per > 0) hw = eu / per;
    }
    if (hw <= 0) hw = static_cast<int>(d.get_info<sycl::info::device::max_compute_units>()) / kXvePerCore;
  } catch (...) {
  }
  if (hw <= 0) hw = 32;  // the B70
  const util::CardSpec *e = util::emulated_card();
  return e && e->xe_cores < hw ? e->xe_cores : hw;
}

}  // namespace xenithium:rt
