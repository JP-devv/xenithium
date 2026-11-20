// Xenithium · qwen35 gated full-attention weights per (card, layer) for kernels:gqa_decode.
//
// Dense plan (heads split): q_proj by whole heads (query | gate per head, 12 heads per card) with k_proj and v_proj's
// kv-head slices appended as ONE W4 matrix [q|gate x 12 | k x 2 | v x 2] x hidden, o_proj split along k (W4), q / k
// norms replicated (raw bf16, the (1 + w) RMSNorm).
#pragma once

#include <vector>

#include "xenithium/arch/qwen35/arch.hpp"
#include "xenithium/kernels/attention/gqa.hpp"
#include "xenithium/load/dense_store.hpp"

namespace xenithium::arch::qwen35 {

struct AttnSet {
  kernels::GqaShape shape;                          // card = 0; use shape_of(c)
  std::vector<int> layers;                          // the full-attention layers, ascending
  std::vector<std::vector<kernels::GqaWeights>> w;  // [card][layer] (default weights on other layers)
  kernels::GqaShape shape_of(int card) const {
    kernels::GqaShape s = shape;
    s.card = card;
    return s;
  }
  // Arch flint: per layer its window (0 = the whole context) and rotary dims (0 = no positional encoding); empty for
  // the other archs (every layer = `shape`).
  std::vector<int> window, rot;
  kernels::GqaShape shape_of(int card, int layer) const {
    kernels::GqaShape s = shape_of(card);
    if (!window.empty()) s.window = window[static_cast<size_t>(layer)], s.rot = rot[static_cast<size_t>(layer)];
    return s;
  }
  // Arch flint: the KV ring of a sliding layer (window + a step's rows: GqaCache::ring_pages), 0 = paged like the rest
  int ring_pages(int layer, int rows) const {
    if (window.empty() || window[static_cast<size_t>(layer)] <= 0) return 0;
    return (window[static_cast<size_t>(layer)] + rows - 2) / 64 + 2;
  }
  // Only the layers present in the store are filled (`layers` lists them); the others stay default.
  static AttnSet from(const load::DenseStore &dense, const Config &cfg, int cards, kernels::QsaKv kv, const std::vector<int> &only = {});
};

}  // namespace xenithium::arch::qwen35
