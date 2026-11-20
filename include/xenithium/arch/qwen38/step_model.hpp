// core:StepModel over the qwen38 Model: the serving engine's view of the model.
// Decode steps are one row per sequence (step_batch with 1-row segments); prefill runs on the
// slot's many-row path in ModelOptions::prefill_chunk pieces.
#pragma once

#include <algorithm>
#include <vector>

#include "xenithium/arch/qwen38/model.hpp"
#include "xenithium/core/batch_engine.hpp"

namespace xenithium::arch::qwen38 {

class StepModelAdapter final : public core::StepModel {
 public:
  explicit StepModelAdapter(Model &m) : m_(m) {}
  int slots() const override { return m_.slots(); }
  int64_t capacity() const override { return m_.capacity(); }
  int64_t vocab() const override { return m_.config().vocab; }
  int max_rows() const override { return step_rows_; }
  // With 32-row steps (XEN_STEP_ROWS 32) a step of fewer than kRows32From sequences keeps the 16-row budget: its drafts
  // per sequence and layouts are the 16-row step's, so it runs the 16-row shape. With 64-row steps (XEN_STEP_ROWS 64) a
  // step of fewer than kRows64From sequences keeps the 32-row budget and its rule.
  int step_rows_at(int sequences) const override {
    if (step_rows_ <= kernels::kStepRows16) return step_rows_;
    if (sequences < kRows32From) return kernels::kStepRows16;
    return step_rows_ > kernels::kStepRows32 && sequences < kRows64From ? kernels::kStepRows32 : step_rows_;
  }
  // Sequences from which a step takes the 32-row budget (step_rows_at).
  static constexpr int kRows32From = 6;
  // Sequences from which a step takes the 64-row budget (step_rows_at).
  static constexpr int kRows64From = 13;
  // Rows a decode step may carry: 64 (the 64-row shape, where the model was loaded with ModelOptions::step_rows_cap
  // 64), 32 (the 32-row shape, `XEN_STEP_ROWS`), 16 (the 16-row shape), 8 or 4. Sets the drafters' depth per request
  // count below.
  void set_step_rows(int rows) {
    step_rows_ = rows <= kernels::kDecodeRows ? kernels::kDecodeRows
                 : rows <= kernels::kStepRows8 ? kernels::kStepRows8
                 : rows <= kernels::kStepRows16 ? kernels::kStepRows16
                 : rows <= kernels::kStepRows32 ? kernels::kStepRows32
                                                : std::max(kernels::kStepRows32, m_.step_rows_cap());
  }
  int prefill_chunk() const override { return m_.prefill_chunk(); }
  void begin(int slot) override;
  void prefill(int slot, const int32_t *tokens, int n, int32_t next) override;
  int prefill_layers() const override { return m_.config().layers; }
  // Four cards: every prompt call while others decode is a whole prefill_chunk call run in parts of whole layers
  // (serve.cpp prefill_part_ms 90), long prompts too, instead of the stall budget's calls.
  bool prefill_long_parts() const override { return m_.cards() == 4; }
  void prefill_begin(int slot, const int32_t *tokens, int n, int32_t next) override;
  bool prefill_advance(int layers) override { return m_.prefill_advance(layers); }
  void prefill_abort() override { m_.prefill_abort(); }
  // Prompt logprobs: the head runs on every 4-row slice of the prefill.
  bool prompt_logprobs() const override { return true; }
  void prefill_rows(int slot, const int32_t *tokens, int n, int32_t next, const RowsFn &on_rows) override;
  // Logprobs on the cards: Model:logprob_stats per row, merged on the host; only the
  // per-card records are copied. set_device_logprobs(false) = the host path (full logits rows).
  bool device_logprobs() const override { return dev_lp_; }
  void set_device_logprobs(bool on) { dev_lp_ = on; }
  void row_logprobs(int rows, const int32_t *targets, int top_n, core::TokenLogprob *out) override;
  void prefill_logprobs(int32_t token, int top_n, core::TokenLogprob &out) override;
  void prefill_each(int slot, const int32_t *tokens, int n, int32_t next, const EachFn &on_rows) override;
  // Every prompt row's logprobs from Model::prefill_logprobs (one many-row head per chunk); set_rows_logprobs(false) =
  // prefill_each + row_logprobs per 4-row slice, for A/B gates.
  bool prefill_row_logprobs(int slot, const int32_t *tokens, int n, int32_t next, const int32_t *targets, int top_n, core::TokenLogprob *out) override;
  void set_rows_logprobs(bool on) { rows_lp_ = on; }
  // Pass 1 KV-only on steps after which nothing drafts (default on); false = the whole pass, for A/B gates.
  void set_kv_only_pass1(bool on) { kv_only_ = on; }
  // MTP (several slots). A step holds step_rows rows in all (a sequence at most 4), so the depth follows the sequences
  // in it: 1 sequence MTP-3, 2 MTP-1 each; then with 8 rows 3-4 MTP-1 each (6-8 rows), 5+ plain rows; with 4 rows 3-4
  // plain rows (the drafter's pass 1 still runs on plain rows, KV-only: every slot's drafter KV stays complete for its
  // later drafts). 2 sequences draft MTP-1 each, although MTP-3 each would fit 8 rows (set_two_deep).
  static constexpr int kDepth = 3;
  // With both drafters loaded the step's sequence count picks the drafter (Model::mtp_drafts); each drafter keeps its
  // own rule. The user's cap (depth_) is the DFlash one then (its block - 1); the MTP layer's is kDepth.
  int draft_depth(int sequences) const override {
    if (!m_.has_drafter() && !m_.has_dflash()) return 0;
    if (!m_.mtp_drafts(std::max(1, sequences))) {  // the DFlash drafter: up to its block - 1, a segment's rows <= seg_rows_max()
      const int cap = std::min(depth_, seg_rows_max() - 1);
      if (sequences <= 1) return cap;
      return std::max(0, std::min(cap, step_rows_at(sequences) / sequences - 1));
    }
    const int cap = m_.has_dflash() ? std::min(depth_, kDepth) : depth_;
    if (sequences <= 1) return cap;
    if (sequences == 2 && !two_deep_) return 1;
    return std::max(0, std::min(cap, step_rows_at(sequences) / sequences - 1));
  }
  // The user's depth cap (--mtp-depth / XEN_MTP_DEPTH): MTP 1..kDepth (the default); DFlash 1..its block - 1.
  void set_depth(int d) { depth_ = std::clamp(d, 1, m_.has_dflash() ? m_.dflash()->max_depth() : kDepth); }
  // Rows one sequence's verify may carry (the model's segment limit).
  int seg_rows_max() const { return m_.seg_rows_max(); }
  int depth() const { return depth_; }
  int depth_ = kDepth;
  // A speculative step's relative cost for the adaptive depth (BatchConfig::adaptive_depth): the verify of n x rows
  // rows, the drafter's pass 1 over them and the chain passes (rows - 2 of one row per sequence).
  bool step_recorded() const override { return m_.times().record > 0; }
  double spec_cost(int n, int rows) const override {
    if ((!m_.has_drafter() && !m_.has_dflash()) || n < 1 || rows < 1) return 0;
    const int all = n * rows, a8 = std::min(all, kernels::kStepRows8);
    // past 8 rows the 16-row shape (spec_cost_rows' fit: + kShape16Ms, then kRow16Ms per row); past 16 the 32-row shape
    // (shape32_ms)
    const double verify = 8.9 + 0.9 * a8 + 0.4 * n + (all > 4 ? 0.67 : 0.0) + (all > a8 ? kShape16Ms + kRow16Ms * (std::min(all, 16) - a8) : 0.0) +
                          shape32_ms(all) + shape64_ms(all);
    if (!m_.mtp_drafts(n))  // one pass (any depth), or the deferred context; + the MTP layer's KV pass
      return verify + (rows > 1 ? kDfPassMs : kDfCtxMs) + (m_.has_drafter() ? pass1_ms(all) : 0.0);
    const double chain = rows > 2 ? (rows - 2) * (kChainMs + kChainSeqMs * (n - 1)) : 0.0;
    return verify + pass1_ms(all) + chain + (m_.has_dflash() ? kDfCtxMs : 0.0);
  }
  static constexpr double kPass1Ms = 0.3, kPass1RowMs = 0.08, kChainMs = 0.6, kChainSeqMs = 0.1;
  static double pass1_ms(int rows) { return kPass1Ms + kPass1RowMs * rows; }  // the MTP layer's pass 1 over a step's rows
  // Unequal segments (Model::verify_batch's B-rows-then-one-row layouts): with the DFlash drafter a depth per segment
  // after it; with the MTP layer its pass 1 runs over the verify's layout (a seg_het word) and its layouts draft one
  // token (layout_depth): the chained passes are not in the accept time the engine takes. spec_cost_rows: a fit of the
  // verify + the drafter's pass; the engine scales the times it takes by it.
  bool unequal_segments() const override { return m_.has_dflash() || m_.has_drafter(); }
  int layout_depth(int n) const override { return m_.mtp_drafts(std::max(1, n)) ? 1 : draft_depth(1); }
  // Past 8 rows (the 16-row shape) the verify is the 8-row fit at 8 rows + kShape16Ms + kRow16Ms per further row.
  double spec_cost_rows(int n, int rows, bool drafting) const override {
    if ((!m_.has_dflash() && !m_.has_drafter()) || n < 1 || rows < n) return 0;
    const int r8 = std::min(rows, kernels::kStepRows8);
    const double verify = 8.23 + 1.466 * r8 + 0.383 * n + (rows > 4 ? 0.141 : 0.0) + (rows > r8 ? kShape16Ms + kRow16Ms * (std::min(rows, 16) - r8) : 0.0) +
                          shape32_ms(rows) + shape64_ms(rows);
    // the drafter of the step (the MTP layer's pass 1 runs drafting or not), + the other one's upkeep
    if (m_.mtp_drafts(n)) return verify + pass1_ms(rows) + (m_.has_dflash() ? kDfCtxMs : 0.0);
    return verify + (drafting ? kDfPassMs : kDfCtxMs) + (m_.has_drafter() ? pass1_ms(rows) : 0.0);
  }
  static constexpr double kShape16Ms = 2.79, kRow16Ms = 0.934;
  // Past 16 rows the 32-row shape: + kShape32Ms, then kRow32Ms per row past 16.
  static constexpr double kShape32Ms = 4.34, kRow32Ms = 0.514;
  static double shape32_ms(int rows) { return rows > 16 ? kShape32Ms + kRow32Ms * (rows - 16) : 0.0; }
  // Past 32 rows the 64-row shape (XEN_STEP_ROWS 64): over the 32-row line above, + kShape64Ms and kRow64Ms per row
  // past 32.
  static constexpr double kShape64Ms = 5.1, kRow64Ms = 0.11;
  static double shape64_ms(int rows) { return rows > 32 ? kShape64Ms + kRow64Ms * (rows - 32) : 0.0; }
  static constexpr double kDfPassMs = 1.6, kDfCtxMs = 0.05;  // a DFlash pass / a deferred context pass
  // For A/B gates (probe serve --two-deep 1): 2 sequences at MTP-3 each in the 8-row shape.
  void set_two_deep(bool on) { two_deep_ = on; }
  std::vector<int32_t> first_drafts(int slot, int32_t first, int depth, int sequences) override;
  // DFlash draft trees (ModelOptions::dflash_tree; the QSA / PLE / GDN tree rows verify them at any context: the
  // selection runs per row)
  std::vector<int> draft_parents(int slot) const override { return m_.draft_parents(slot); }
  void draft_sampling(int slot, uint64_t key, uint64_t pos, float inv_t) override { m_.set_draft_noise(slot, key, pos, inv_t); }
  void verify(const std::vector<SpecSeg> &segs) override;
  std::vector<std::vector<int32_t>> accept(const std::vector<SpecAccept> &a) override;
  int32_t prefill_pick() override;
  void prefill_logits(std::vector<float> &row) override;
  void step(const std::vector<int> &slots, const int32_t *tokens) override;
  void picks(int rows, int32_t *out) override;
  void logits(int rows, std::vector<float> &out) override { m_.logits(rows, out); }
  int cand_tiles() const override { return HeadCands::max_tiles(); }
  int logits_cands(int rows, int T, std::vector<int32_t> &ids, std::vector<float> &vals) override { return m_.logits_cands(rows, T, ids, vals); }
  // The shared KV pool (ModelOptions::kv_pool_tokens); 0 with the fixed per-slot layout.
  int64_t kv_pool_tokens() const override { return m_.kv_shared() ? m_.kv_pool_tokens() : 0; }
  int64_t kv_free_tokens() const override { return m_.kv_free_tokens(); }
  bool kv_reserve(int slot, int64_t tokens) override { return m_.kv_reserve(slot, tokens); }
  void kv_release(int slot) override { m_.kv_release(slot); }
  int idle_work(int budget) override {  // the snapshot ids' host buffers first, then KV lending's refills
    const int p = m_.snap_prepin(1);
    return p ? p : m_.lend_idle(budget);
  }
  bool kv_blocks_lend(int slot) const override { return m_.kv_blocks_lend(slot); }
  bool kv_reserve_lends(int slot, int64_t tokens) const override { return m_.kv_reserve_lends(slot, tokens); }
  int64_t kv_held_tokens(int slot) const override { return m_.kv_held_tokens(slot); }
  // The prefix cache (ModelOptions::snap_gib; Model's snapshots, live resume, forks).
  int snap_capacity() const override { return m_.snap_capacity(); }
  int snap_save(int slot) override { return m_.snap_save(slot); }
  void snap_restore(int slot, int id) override { m_.snap_restore(slot, id); }
  void snap_fork(int dst, int src, int id) override { m_.snap_fork(dst, src, id); }
  void snap_free(int id) override { m_.snap_free(id); }
  bool snap_live(int slot) const override { return m_.live_ok(slot); }
  void resume(int slot) override { m_.use_slot(slot); }
  void kv_trim(int slot, int64_t tokens) override { m_.kv_trim(slot, tokens); }
  int kv_page_tokens() const override { return model::KvPages::kPageTokens; }
  // Prefill calls that start and end at multiples of 64 with 64+ rows compute every row as a cold prefill does (gated
  // by probe prefix part 0); a call under 64 rows does not.
  int snap_align() const override { return 64; }

 private:
  Model &m_;
  int last_ = 0;  // row of the last prefilled token in the prefill's last slice
  bool dev_lp_ = true;
  bool rows_lp_ = true;
  bool kv_only_ = true;
  int step_rows_ = kernels::kStepRows16;  // 32 by XEN_STEP_ROWS (set_step_rows)
  bool two_deep_ = false;
  // merge the two cards' records of row r (rec = [card][n][kLpRec]) into out
  void merge_row(const std::vector<float> &rec, int n, int r, float xt, int top_n, core::TokenLogprob &out) const;
  std::vector<float> lp_rec_, lp_xt_;
  std::vector<float> rows_;  // prefill_rows' logits of one slice
  std::vector<int> vr_;  // rows of each segment of the last verify (unequal ones with MTP too)
  int vsegs_ = 0;            // its segments
  // accept's two drafters: the one that drafts after this verify; the other one follows the step
  std::vector<std::vector<int32_t>> accept_dflash(const std::vector<SpecAccept> &a);
  std::vector<std::vector<int32_t>> accept_mtp(const std::vector<SpecAccept> &a);
  // pass 1's next tokens: each verified row's following token (rows past n[k] repeat the segment's last kept one)
  int pass1_next(const std::vector<SpecAccept> &a, int32_t *nx) const;
};

}  // namespace xenithium::arch::qwen38
