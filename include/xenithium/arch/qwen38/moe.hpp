// Xenithium · qwen38 MoE weights per (card, layer) for kernels::moe_decode.
//
// Router: mlp.gate (512 rows) with mlp.shared_expert_gate appended (bf16, 528 rows padded).
// Shared expert: gate_proj with up_proj appended (int8, 1280 rows) and down_proj (int8). All
// replicated (dense plan). The expert blob tables are set separately: they come from wherever
// the experts live (ExpertStore, or a probe's own single-layer upload).
#pragma once

#include <vector>

#include "xenithium/arch/qwen38/arch.hpp"
#include "xenithium/arch/qwen38/experts.hpp"
#include "xenithium/kernels/moe/moe.hpp"
#include "xenithium/load/dense_store.hpp"

namespace xenithium::arch::qwen38 {

kernels::MoeBlobLayout moe_blob_layout(const ExpertLayout &lay);

struct MoeSet {
  kernels::MoeShape shape;                            // card = 0; use shape_of(c)
  std::vector<std::vector<kernels::MoeWeights>> w;    // [card][layer], blobs null
  kernels::MoeShape shape_of(int card) const {
    kernels::MoeShape s = shape;
    s.card = card;
    return s;
  }
  static MoeSet from(const load::DenseStore &dense, const Config &cfg, int cards, bool tp = false);
};

}  // namespace xenithium::arch::qwen38
