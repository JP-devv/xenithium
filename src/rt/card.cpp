// Xenithium · which card a device is, and the card it runs as (rt/card.hpp; the table and the logic: util/cards.hpp).
#include "xenithium/rt/card.hpp"

#include <cstdio>
#include <mutex>
#include <set>

namespace xenithium::rt {

util::CardView card_view(const l0::DeviceInfo &info) {
  return util::card_view(info.device_id, info.name, static_cast<int>(info.eu_total()) / kXvePerCore, info.vram_bytes);
}

uint64_t emulated_vram_cut(const l0::DeviceInfo &info) {
  if (util::emu_card_off(util::emu_card_value())) return 0;
  try {
    return card_view(info).vram_cut;
  } catch (...) {
    return 0;  // the Fabric refuses the setting; a size query does not
  }
}

void log_card(size_t index, const l0::DeviceInfo &info) {
  static std::mutex mu;
  static std::set<std::string> seen;
  const util::CardView v = card_view(info);  // throws on a bad XEN_EMU_CARD before anything is allocated
  std::lock_guard<std::mutex> lk(mu);
  if (!seen.insert(info.pci_bdf() + "/" + info.name).second) return;
  std::fprintf(stderr, "card %zu: %s; %s at %s\n", index, v.describe().c_str(), info.name.c_str(), info.pci_bdf().c_str());
}

}  // namespace xenithium:rt
