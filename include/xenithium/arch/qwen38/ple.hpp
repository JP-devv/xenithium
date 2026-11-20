// Xenithium · qwen38 PLE block weights per card for kernels::ple_decode.
//
// Dense plan: key_proj with value_proj appended (int8, [S*H + H][E], one GEMM), the three norms
// and conv1d raw bf16; all replicated (every card computes the same residual). The n-gram table
// is host-side (ple_table.hpp).
#pragma once

#include <vector>

#include "xenithium/arch/qwen38/arch.hpp"
#include "xenithium/kernels/embed/ple.hpp"
#include "xenithium/load/dense_store.hpp"

namespace xenithium::arch::qwen38 {

struct PleSet {
  kernels::PleShape shape;
  int layer = -1;                        // the decoder layer whose attention mixer the block precedes
  std::vector<kernels::PleWeights> w;    // [card]
  static PleSet from(const load::DenseStore &dense, const Config &cfg, int cards);
};

}  // namespace xenithium::arch::qwen38
