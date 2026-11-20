// Xenithium · qwen38 hyper-connection weights per card, as the mixer kernels take them.
//
// Every decoder layer has two mixers (before attention / GDN and before the MLP); the model
// has one more, the final mixer before lm_head. Each layer mixer's down matrix carries its 4
// injection rows after the 320 low-rank rows (the dense plan appends block_inject_weight:
// 324 rows, padded to 336), so one GEMM gives both. The final mixer has no injection.
// All replicated: every card runs every mixer on its own copy (no cross-card traffic).
#pragma once

#include <vector>

#include "xenithium/arch/qwen38/arch.hpp"
#include "xenithium/kernels/residual/hc.hpp"
#include "xenithium/load/dense_store.hpp"

namespace xenithium::arch::qwen38 {

struct HcSet {
  kernels::HcShape shape;
  std::vector<std::vector<kernels::HcWeights>> attn, mlp;  // [card][layer]
  std::vector<kernels::HcWeights> final_mixer;             // [card]
  int down_ld = 0;  // columns of a layer mixer's down output (lowrank + injection, padded)
  int ple_layer = -1;

  static HcSet from(const load::DenseStore &dense, const Config &cfg, int cards);
};

}  // namespace xenithium::arch::qwen38
