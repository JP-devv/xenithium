// Xenithium · qwen38 Gated DeltaNet weights per (card, layer) for kernels::gdn_decode.
//
// Dense plan (heads split): in_proj_qkv by q | k | v sections with in_proj_z appended (int8),
// in_proj_a and in_proj_b raw bf16 rows (read by the core kernel), conv1d by the same sections (raw bf16), A_log and
// dt_bias by value head (raw), norm replicated, out_proj split along k (int8).
#pragma once

#include <vector>

#include "xenithium/arch/qwen38/arch.hpp"
#include "xenithium/kernels/linear_attention/gdn.hpp"
#include "xenithium/load/dense_store.hpp"

namespace xenithium::arch::qwen38 {

struct GdnSet {
  kernels::GdnShape shape;                         // card = 0; use shape_of(c)
  std::vector<int> layers;                         // the linear-attention layers, ascending
  std::vector<std::vector<kernels::GdnWeights>> w; // [card][layer] (default weights on other layers)
  kernels::GdnShape shape_of(int card) const {
    kernels::GdnShape s = shape;
    s.card = card;
    return s;
  }
  static GdnSet from(const load::DenseStore &dense, const Config &cfg, int cards);
};

}  // namespace xenithium::arch::qwen38
