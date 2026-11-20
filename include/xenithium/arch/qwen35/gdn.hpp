// Xenithium · qwen35 Gated DeltaNet weights per (card, layer) for kernels:gdn_decode.
//
// Dense plan (heads split): in_proj_qkv by q | k | v sections with in_proj_z appended as ONE W4 matrix, in_proj_a /
// in_proj_b raw bf16 rows (read by the core kernel), conv1d by the same sections (raw bf16), A_log and dt_bias by value
// head (raw), norm replicated, out_proj split along k (W4). The same kernel as qwen38's with the W4 GEMMs and the swish
// output gate (GdnShape::swish: y = norm(o) x silu(z)).
#pragma once

#include <vector>

#include "xenithium/arch/qwen35/arch.hpp"
#include "xenithium/kernels/linear_attention/gdn.hpp"
#include "xenithium/load/dense_store.hpp"

namespace xenithium::arch::qwen35 {

struct GdnSet {
  kernels::GdnShape shape;                          // card = 0; use shape_of(c)
  std::vector<int> layers;                          // the linear-attention layers, ascending
  std::vector<std::vector<kernels::GdnWeights>> w;  // [card][layer] (default weights on other layers)
  kernels::GdnShape shape_of(int card) const {
    kernels::GdnShape s = shape;
    s.card = card;
    return s;
  }
  // Only the layers present in the store are filled (`layers` lists them); the others stay default.
  static GdnSet from(const load::DenseStore &dense, const Config &cfg, int cards, const std::vector<int> &only = {});
};

}  // namespace xenithium::arch::qwen35
