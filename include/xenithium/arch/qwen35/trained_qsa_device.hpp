#pragma once
#include <memory>
#include "xenithium/arch/qwen35/trained_qsa.hpp"
#include "xenithium/kernels/attention/gqa.hpp"
#include "xenithium/rt/fabric.hpp"

namespace xenithium::arch::qwen35::trained_qsa {
inline constexpr int kMaxContext = 262144;
// Per-rank workspace is shared in compute-queue order. Each covered layer owns
// its pooled history, open-block tail and transient rows until accept().
class Device {
 public:
  Device(Fabric &fab, std::span<const Layer> weights, int slots, int capacity, int prefill_rows, int step_rows, bool xmx_prefill = false);
  static size_t bytes_per_rank(size_t layers, int hidden, int slots, int capacity, int prefill_rows, int step_rows, bool xmx_prefill = false);
  ~Device();
  kernels::GqaTrainedPrefill prefill(int card, const kernels::gqa_bf16 *x, int rows, int slot, int pos, int bank_layer = 0, int fa_rows = 0);
  void stage(int card, const kernels::gqa_bf16 *x, int rows, int kv_heads, kernels::GqaStep &step, int bank_layer = 0, int model_layer = -1,
             uint32_t track=~uint32_t(0),bool sparse=true);
  void accept(int slot, int pos, int row0, int keep, bool tree_batch=false, uint64_t path=0);
 private:
  struct Impl;
  std::unique_ptr<Impl> p_;
};
}
