// Xenithium · the block drafters' `architectures` names (config.json): the server's check, the drafter's loader and
// the launcher's model scan read this one list.
#pragma once

#include <string>
#include <string_view>

namespace xenithium::arch {

// DFlash 2 (`DFlash2DraftModel`).
inline constexpr std::string_view kDFlash2Arch = "DFlash2DraftModel";
// DSpine (`DFlashDraftModel` + the DSpine code bridge, kernels/drafter/dspine.hpp). Its public b16 checkpoint was
// renamed `DSpineDraftModel` later with nothing else changed: both names load as DSpine.
inline constexpr std::string_view kDSpineArchs[] = {"DFlashDraftModel", "DSpineDraftModel"};

inline bool is_dspine_arch(std::string_view a) {
  for (const auto n : kDSpineArchs)
    if (a == n) return true;
  return false;
}
inline bool is_block_drafter_arch(std::string_view a) { return a == kDFlash2Arch || is_dspine_arch(a); }

// "DFlash2DraftModel, DFlashDraftModel or DSpineDraftModel", for messages
inline std::string block_drafter_arch_names() {
  std::string s(kDFlash2Arch);
  constexpr size_t n = std::size(kDSpineArchs);
  for (size_t i = 0; i < n; ++i) s += (i + 1 < n ? ", " : " or ") + std::string(kDSpineArchs[i]);
  return s;
}

}  // namespace xenithium::arch
