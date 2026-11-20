// Xenithium · arch flint: the MoE layers on the cards for the runtime (arch::qwen35::Model with Config::flint).
//
// Every MoE layer's 256 routed experts and its shared expert live on every card as kernels::moe_aff blobs (TP over the
// cards by intermediate column: each card holds its 1/C of every expert: 864 KiB per expert per card at TP2, the
// whole 47-layer set 10.2 GiB; TP1 20.3 GiB), packed at load from the checkpoint's stacked `mlp.switch_mlp.*` and
// `mlp.shared_expert.*` tensors. The router (`mlp.gate.weight`, bf16) comes from the dense store. A layer's step:
// kernels::moe_aff_decode on every card (router, route, gate_up, down, combine + the two-card exchange), out bf16
// [rows][hidden] on every card (the same bits on both).
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "xenithium/arch/qwen35/arch.hpp"
#include "xenithium/kernels/moe/moe_aff.hpp"
#include "xenithium/kernels/comm/wire.hpp"
#include "xenithium/load/dense_store.hpp"
#include "xenithium/model/checkpoint.hpp"
#include "xenithium/rt/fabric.hpp"

namespace xenithium::arch::flint {

class MoeSet {
 public:
  using bf16 = sycl::ext::oneapi::bfloat16;

  // Loads every Config::moe layer (threads: the host packing threads). Throws on a shape the kernels do not take.
  MoeSet(Fabric &fab, const model::Checkpoint &ck, const load::DenseStore &dense, const qwen35::Config &cfg, int threads);
  MoeSet(const MoeSet &) = delete;
  MoeSet &operator=(const MoeSet &) = delete;
  ~MoeSet();

  bool has(int layer) const { return layer >= 0 && layer < static_cast<int>(layers_.size()) && layers_[static_cast<size_t>(layer)].on; }
  // One layer on every card: x[c] bf16 [row_shape(rows)][hidden] (finite); out(c) bf16 [rows][hidden] afterwards.
  // Submits and returns; parity alternates per call (both cards' calls in the same order).
  void run(int layer, int rows, const std::vector<const bf16 *> &x);
  bf16 *out(int c) const { return cards_[static_cast<size_t>(c)].out; }
  // Prefill: the many-row block's scratch for up to rows_max rows on every card (once), then m rows of one layer on
  // every card (kernels::moe_aff_rows): x[c] bf16 [m][hidden]; wire[c] the chunk-sized two-card wire (null with one
  // card; split: the fp32 partial stays in rows_part(c) for the caller's split sum, out unused); out[c] bf16 [m][hidden].
  void reserve_rows(int rows_max);
  void run_rows(int layer, int m, const std::vector<const bf16 *> &x, const std::vector<const kernels::PairRowsWire *> &wire, int parity,
                const std::vector<bf16 *> &out);
  const float *rows_part(int c) const { return cards_[static_cast<size_t>(c)].rows.part; }
  size_t rows_bytes(int c) const { return cards_[static_cast<size_t>(c)].rows_bytes; }  // the many-row scratch on card c
  const kernels::MoeAffEvents &events(int c) const { return ev_[static_cast<size_t>(c)]; }
  void set_wire_bf16(bool on);
  int parity() const { return parity_; }  // the wire's parity (step graphs: recorded state)
  void set_parity(int p) { parity_ = p & 1; }
  size_t bytes(int c) const { return cards_[static_cast<size_t>(c)].bytes; }  // expert blobs on card c
  uint32_t wire_error(int c);
  double load_seconds() const { return load_s_; }

 private:
  struct Layer {
    bool on = false;
    std::vector<kernels::MoeAffWeights> w;  // [card]
  };
  struct Card {
    kernels::MoeAffScratch sc;
    kernels::PairWire wire;
    bf16 *out = nullptr;
    size_t bytes = 0;
    kernels::MoeAffRowsScratch rows;  // reserve_rows
    size_t rows_bytes = 0;
    std::vector<void *> mem;
  };
  Fabric &fab_;
  kernels::MoeAffShape shape_;
  std::vector<Layer> layers_;
  std::vector<Card> cards_;
  std::vector<kernels::MoeAffEvents> ev_;
  int parity_ = 0;
  double load_s_ = 0;
};

}  // namespace xenithium::arch::flint
