// Xenithium · qwen38 QSA (full attention) weights per (card, layer) for kernels::qsa_decode.
//
// Dense plan (heads split): q_proj (q | gate per head) with k_proj and v_proj appended (int8, one
// GEMM: [12 heads x 512 | k 256 | v 256] per card; TP4 with the KV split by tokens: [6 heads x 512 | the partner's 6 q
// heads x 256 | k 256 | v 256]), o_proj split along k (int8), the indexer
// projection replicated (bf16), q/k norms and the indexer norms replicated (raw bf16).
#pragma once

#include <vector>

#include "xenithium/arch/qwen38/arch.hpp"
#include "xenithium/kernels/attention/qsa.hpp"
#include "xenithium/load/dense_store.hpp"

namespace xenithium::arch::qwen38 {

struct QsaSet {
  kernels::QsaShape shape;                          // card = 0; use shape_of(c)
  std::vector<int> layers;                          // the full-attention layers, ascending
  std::vector<std::vector<kernels::QsaWeights>> w;  // [card][layer] (default weights on other layers)
  kernels::QsaShape shape_of(int card) const {
    kernels::QsaShape s = shape;
    s.card = card;
    return s;
  }
  // split: how a kv head's cards store its KV (kernels/attention/qsa.hpp; tokens = the partner's q rows in q_proj)
  static QsaSet from(const load::DenseStore &dense, const Config &cfg, int cards, kernels::QsaKv kv,
                     kernels::QsaKvSplit split = kernels::QsaKvSplit::heads);
};

}  // namespace xenithium::arch::qwen38
