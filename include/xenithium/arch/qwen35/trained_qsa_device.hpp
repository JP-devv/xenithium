#pragma once
#include <memory>
#include <utility>
#include "xenithium/arch/qwen35/trained_qsa.hpp"
#include "xenithium/kernels/attention/gqa.hpp"
#include "xenithium/rt/fabric.hpp"

namespace xenithium::arch::qwen35::trained_qsa {
inline constexpr int kMaxContext = 262144;
// Per-rank workspace is shared in compute-queue order. Each covered layer owns
// its pooled history, open-block tail and transient rows until accept().
class Device {
 public:
  Device(Fabric &fab, const Layer &weights, int slots, int capacity, int prefill_rows, int step_rows);
  Device(Fabric &fab, std::span<const Layer> weights, int slots, int capacity, int prefill_rows, int step_rows, bool xmx_prefill = false);
  static size_t bytes_per_rank(size_t layers, int hidden, int slots, int capacity, int prefill_rows, int step_rows, bool xmx_prefill = false);
  ~Device();
  kernels::GqaTrainedPrefill prefill(int card, const kernels::gqa_bf16 *x, int rows, int slot, int pos, int bank_layer = 0, int fa_rows = 0, bool capture = false);
  void replay_prefill_topk(int card,const kernels::gqa_bf16 *,int rows,int slot,int pos,int bank,int layer,int fa_rows,
                           const kernels::GqaShape &,const kernels::GqaCache &,const int32_t *,const kernels::GqaRowsScratch &,int ldy);
  void stage(int card, const kernels::gqa_bf16 *x, int rows, int kv_heads, kernels::GqaStep &step, int bank_layer = 0, int model_layer = -1,
             uint32_t track=~uint32_t(0),bool sparse=true,bool tree_probe=false,
             std::pair<sycl::event,sycl::event> *selector_span=nullptr,uint32_t dense_rows=0);
  void accept(int slot, int pos, int row0, int keep, bool tree_batch=false, uint64_t path=0);
  // Submission only on the compute queues. The prefix proof is owned by Model.
  size_t copy_aligned_prefix(int dst, int src, int pos);
  void audit_tree(int card,int bank,int rows,const kernels::GqaShape &,const kernels::GqaWeights &,
                  const kernels::GqaCache &,kernels::GqaStep,const kernels::GqaScratch &);
  void audit_tree_after_kv(int slot,int pos,int keep,uint64_t path);
  // Separate opt-in diagnostic, after both ranks' live attention submissions.
  void audit_fp16(int card,int bank_layer,int rows,const kernels::GqaShape &,const kernels::GqaWeights &,
                  const kernels::GqaCache &,kernels::GqaStep,const kernels::GqaScratch &);
  struct Selection { int eligible, pages; std::vector<int32_t> ids; };
  bool selection_ready(int card) const;
  std::vector<int32_t> read_logical_pages(int card, int row, int pages);
  void check_selector_reference(int card, int row, const Selection &picked);
  // Diagnostic readback only, after a stage that actually scored this row.
  Selection read_selection(int card, int row);
 private:
  kernels::GqaTrainedPrefill prefill_plan(int card,int rows,int slot,int pos,int bank,int fa_rows,int chunk,bool capture,bool frozen,bool audit,bool skip_subtiles=false);
  struct Impl;
  std::unique_ptr<Impl> p_;
};
}
