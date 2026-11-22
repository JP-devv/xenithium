// core:StepModel over the qwen35 Model (several sequences): the serving engine's view of
// the dense Qwen3.8-27B. ModelOptions::slots sequences, each with its own GDN states, KV and drafter KV; a decode step
// carries every decoding sequence (Model::verify_batch, up to Model::step_rows()). Plain decode = 1-row segments; with
// the checkpoint's MTP drafter (ModelOptions::mtp) every step verifies each sequence's fed token + its drafts, the depth
// by the sequences in the step (draft_depth).
// Prefill calls run the many-row path (Model::prefill) in ModelOptions::prefill_chunk pieces; prompt logprobs read the
// head's rows per 4-row slice. Decode rows' logprobs are computed on the cards (device_logprobs(), Model::
// logprob_stats; set_device_logprobs(false) = the host path of full logits rows).
#pragma once

#include <algorithm>
#include <vector>

#include "xenithium/arch/qwen35/model.hpp"
#include "xenithium/core/batch_engine.hpp"

namespace xenithium::arch::qwen35 {

class StepModelAdapter final : public core::StepModel {
 public:
  explicit StepModelAdapter(Model &m) : m_(m) {
    if (m.dflash()) set_depths(m.step_rows() > kernels::kStepRows16 ? kDFlashDepths32 : kDFlashDepths);
  }
  int slots() const override { return m_.slots(); }
  int64_t capacity() const override { return m_.capacity(); }
  int64_t vocab() const override { return m_.config().vocab; }
  // rows one decode step carries in all (the engine sizes speculative segments by it)
  int max_rows() const override { return m_.step_rows(); }  // llama / flint: 8
  int prefill_chunk() const override { return m_.prefill_chunk(); }
  void begin(int slot) override;
  void note_prompt(int slot,int64_t total,int64_t restored) override { m_.note_prompt(slot,total,restored); }
  void prefill(int slot, const int32_t *tokens, int n, int32_t next) override;
  // a call in parts of whole layers: other slots decode between the parts
  int prefill_layers() const override { return m_.config().layers; }
  void prefill_begin(int slot, const int32_t *tokens, int n, int32_t next) override;
  bool prefill_advance(int layers) override { return m_.prefill_advance(layers); }
  void prefill_abort() override { m_.prefill_abort(); }
  // one call for several slots (Model::prefill_begin(PrefillSeg *, count))
  int prefill_merge_max() const override { return m_.prefill_merge_max(); }
  int prefill_merge_rows() const override { return m_.prefill_merge_rows(); }
  int prefill_merge_pad() const override { return Model::kPfMergeAlign; }
  bool prefill_mergeable(int slot, const int32_t *tokens, int n) const override { return m_.prefill_mergeable(slot, tokens, n); }
  void prefill_begin_many(const std::vector<PrefillSeg> &segs) override;
  void prefill_result(int slot) override;
  // every prompt call whole and in parts while others decode (small calls would pay the weights pass each)
  bool prefill_long_parts() const override { return true; }
  bool prompt_logprobs() const override { return true; }
  void prefill_rows(int slot, const int32_t *tokens, int n, int32_t next, const RowsFn &on_rows) override;
  static constexpr int kDepth = 3;  // MTP-3: a 4-row verify (Model::kRows)
  // DFlash's default depths (the most drafts per sequence: the adaptive depth picks below them by spec_cost and the
  // acceptance it sees): one or two sequences the whole block (7 drafts, an 8-row verify each), three 4 (3 x 5 rows),
  // four 3 (4 x 4), five to eight 1 (2 rows each, 10-16 rows), nine and more plain rows. With draft trees each segment
  // drafts its whole tree.
  static constexpr int kDepthSeqs = kernels::kSegs16;  // the tables' sequence counts (1..16)
  static_assert(core::Event::kMaxTokens >= kernels::kTreeRows, "a 16-row verify of one sequence keeps up to 16 tokens in one event");
  static constexpr int kDFlashDepths[kDepthSeqs] = {7, 7, 4, 3, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0};
  // With 32-row steps (XEN_STEP_ROWS 32) 1-4 sequences keep the 16-row table, 5-8 draft up to 3 (8 x 4 rows; MTP-3 from
  // XEN_DRAFTER_MTP_FROM 5), 9 / 10 up to 2 (9 x 3 / 10 x 3 rows; the MTP layer's pass 1 + one chained pass) and 11-16
  // up to 1 (16 x 2).
  static constexpr int kDFlashDepths32[kDepthSeqs] = {7, 7, 4, 3, 3, 3, 3, 3, 2, 2, 1, 1, 1, 1, 1, 1};
  // Drafts per sequence in a step of n sequences: depth_[n - 1] (each segment <= Model::seg_rows_max() rows, the step <=
  // max_rows()). Defaults: depth_'s own table (MTP) / kDFlashDepths (DFlash); set_depths replaces them.
  int draft_depth(int sequences) const override {
    if (!m_.has_drafter() || sequences < 1) return 0;
    const int d = depth_[static_cast<size_t>(std::min(sequences, kDepthSeqs) - 1)];
    return std::max(0, std::min({d, m_.draft_depth_max(sequences), m_.seg_rows_max() - 1, max_rows() / sequences - 1}));
  }
  // A speculative step's relative cost for the adaptive depth (BatchConfig::adaptive_depth; the engine scales its
  // measured step times by it, so only the ratios between depths at one sequence count matter). The verify of n segments
  // x rows: kVerifyMs + kRowMs per row up to 8, + kRow16Ms per row past 8 with kShape16Ms for the 16-row shape, + kSegMs
  // per segment. The drafter: DFlash one pass at any depth; MTP pass 1 over the rows + one chained pass per further
  // draft; a 1-row step only the deferred context / KV rows.
  double spec_cost(int n, int rows) const override {
    if (!m_.has_drafter() || n < 1 || rows < 1) return 0;
    if (m_.config().flint && n > 1) return flint_cost(n, rows);
    const int all = n * rows, a8 = std::min(all, kernels::kStepRows8);
    const double verify = kVerifyMs + kRowMs * a8 + kSegMs * n + (all > a8 ? kShape16Ms + kRow16Ms * (std::min(all, 16) - a8) : 0.0) +
                          (all > 16 ? kShape32Ms + kRow32Ms * (all - 16) : 0.0);
    if (rows == 1) return verify + kCtxMs;
    if (!m_.mtp_drafts(n)) return verify + kDfPassMs + (all > 16 ? kDfPass32Ms : 0.0);
    return verify + kMtpPass1Ms + (rows - 2) * kMtpChainMs;
  }
  // Unequal segments (qwen35 only: llama / flint keep 8-row steps) with either drafter: verify_batch takes the engine's
  // B-row-then-one-row layouts, draft_batch a depth per segment. The engine's layout cost per step is the verify of n
  // segments and `rows` rows in all (spec_cost's terms) + a drafting accept (DFlash one pass at any depth; MTP pass 1) or
  // the context-only one. MTP's chained passes (+kMtpChainMs each) are not in the engine's measured accept time per
  // layout, so its layouts draft one token (layout_depth 1).
  bool unequal_segments() const override { return m_.has_drafter() && !m_.config().flint && !m_.config().llama(); }
  int layout_depth(int n) const override { return m_.mtp_drafts(std::max(1, n)) ? 1 : draft_depth(1); }
  // Two classes (verify_batch takes B rows then B2 rows each: kernels::seg_het2). Where the table's one depth for all
  // leaves rows of a 32-row step free, some requests may draft up to the depth below (0 = one depth for all); the engine
  // picks how many and how deep by expected tokens per ms (BatchEngine::pick_deep). One table per drafter of the step:
  // the block drafter's pass costs the same at any depth, the MTP layer's a chained pass per further draft (once for the
  // whole step).
  static constexpr int kDeepDf32[kDepthSeqs] = {0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  static constexpr int kDeepMtp32[kDepthSeqs] = {0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 2, 2, 2, 2, 2, 0};
  bool unequal_classes() const override { return unequal_segments(); }
  int deep_depth(int n) const override {
    if (n < 1 || n > kDepthSeqs || max_rows() != kernels::kStepRows32) return 0;
    const int d = draft_depth(n);
    const int t = std::min({(m_.mtp_drafts(n) ? kDeepMtp32 : kDeepDf32)[n - 1], m_.draft_depth_max(n), m_.seg_rows_max() - 1});
    return d >= 1 && t > d ? t : 0;
  }
  double draft_pass_ms(int n, int depth) const override {
    if (!m_.has_drafter() || depth < 1) return 0;
    return m_.mtp_drafts(std::max(1, n)) ? kMtpPass1Ms + (depth - 1) * kMtpChainMs : kDfPassMs;
  }
  double spec_cost_rows(int n, int rows, bool drafting) const override {
    if (!m_.has_drafter() || n < 1 || rows < n) return 0;
    const int a8 = std::min(rows, kernels::kStepRows8);
    const double verify = kVerifyMs + kRowMs * a8 + kSegMs * n + (rows > a8 ? kShape16Ms + kRow16Ms * (std::min(rows, 16) - a8) : 0.0) +
                          (rows > 16 ? kShape32Ms + kRow32Ms * (rows - 16) : 0.0);
    if (!drafting) return verify + kCtxMs;
    return verify + (m_.mtp_drafts(n) ? kMtpPass1Ms : kDfPassMs + (rows > 16 ? kDfPass32Ms : 0.0));
  }
  static constexpr double kVerifyMs = 13.77, kRowMs = 0.409, kRow16Ms = 0.431, kShape16Ms = 1.146, kSegMs = 0.412;
  // past 16 rows the 32-row shape (kShape32Ms) + kRow32Ms per further row; the DFlash pass over a 17-32-row block:
  // + kDfPass32Ms
  static constexpr double kShape32Ms = 1.2, kRow32Ms = 0.34, kDfPass32Ms = 1.45;
  static constexpr double kCtxMs = 0.3, kDfPassMs = 4.2, kMtpPass1Ms = 2.15, kMtpChainMs = 1.5;
  // Arch flint's step costs, a different curve: its MoE rows reach more experts, so a row's cost depends on the row
  // count. kFlRowsMs: the verify of `all` rows in one segment, + kFlSegMs per further segment; the drafter: one DFlash
  // pass (kFlDfPassMs), a depth-0 step only its deferred context rows (kFlCtxMs: one 64-row context pass per 64 steps).
  // Steps of 2+ sequences only: one sequence's verify carries fewer rows than asked (the confidence cut, single-sequence
  // passes), and its step times stay priced by spec_cost's own terms.
  static constexpr double kFlRowsMs[kernels::kStepRows8] = {3.210, 3.875, 4.583, 4.970, 6.399, 6.761, 7.258, 7.623};
  static constexpr double kFlSegMs = 0.15, kFlDfPassMs = 1.45, kFlCtxMs = 0.03;
  static double flint_cost(int n, int rows) {
    const int all = std::min(n * rows, kernels::kStepRows8);
    const double verify = kFlRowsMs[all - 1] + kFlSegMs * (n - 1);
    return verify + (rows == 1 ? kFlCtxMs : kFlDfPassMs);
  }
  // the DFlash drafter writes no target KV (its own context ring): the verify's 1 + d rows only; with
  // the MTP layer beside it, MTP's rule (its passes write KV ahead)
  int64_t kv_ahead(int depth) const override { return m_.dflash() && !m_.drafter() ? 1 + depth : core::StepModel::kv_ahead(depth); }
  void set_depths(const std::vector<int> &d) {
    for (size_t i = 0; i < d.size() && i < static_cast<size_t>(kDepthSeqs); ++i) depth_[i] = d[i];
  }
  void set_depths(const int (&d)[kDepthSeqs]) {
    for (size_t i = 0; i < static_cast<size_t>(kDepthSeqs); ++i) depth_[i] = d[i];
  }
  std::vector<int32_t> first_drafts(int slot, int32_t first, int depth, int sequences) override;
  // coupled drafts for sampled requests (MTP and DFlash)
  void draft_sampling(int slot, uint64_t key, uint64_t pos, float inv_t) override { m_.set_draft_noise(slot, key, pos, inv_t); }
  void verify(const std::vector<SpecSeg> &segs) override;
  std::vector<int> draft_parents(int slot) const override { return m_.draft_parents(slot); }  // DDTree
  bool verifies_trees(std::span<const int> slots, std::span<const int> rows) const override;
  // A lone sequence drafts DSpine's whole 16-row block (a 16-row tree, the model's ModelOptions::seg_rows 16) after a
  // verify that accepted wide_from or more drafts (ServeOptions::drafter_wide)
  int wide_depth() const override { return wide_from_ > 0 && m_.dflash() && m_.seg_rows_max() > kernels::kStepRows8 ? m_.seg_rows_max() - 1 : 0; }
  int wide_from() const override { return wide_from_; }
  void set_wide_from(int n) { wide_from_ = n; }
  std::vector<std::vector<int32_t>> accept(const std::vector<SpecAccept> &a) override;
  int32_t prefill_pick() override;
  void prefill_logits(std::vector<float> &row) override;
  void step(const std::vector<int> &slots, const int32_t *tokens) override;
  void picks(int rows, int32_t *out) override;
  void logits(int rows, std::vector<float> &out) override { m_.logits(rows, out); }
  bool device_logprobs() const override { return dev_lp_; }
  void set_device_logprobs(bool on) { dev_lp_ = on; }
  void row_logprobs(int rows, const int32_t *targets, int top_n, core::TokenLogprob *out) override;
  void prefill_logprobs(int32_t token, int top_n, core::TokenLogprob &out) override;
  void prefill_each(int slot, const int32_t *tokens, int n, int32_t next, const EachFn &on_rows) override;
  int cand_tiles() const override { return HeadCands::max_tiles(); }  // 
  int logits_cands(int rows, int T, std::vector<int32_t> &ids, std::vector<float> &vals) override { return m_.logits_cands(rows, T, ids, vals); }
  // the shared KV pool (ModelOptions:kv_pool_tokens); 0 with the fixed per-slot layout
  int64_t kv_pool_tokens() const override { return m_.kv_shared() ? m_.kv_pool_tokens() : 0; }
  int64_t kv_free_tokens() const override { return m_.kv_free_tokens(); }
  bool kv_reserve(int slot, int64_t tokens) override { return m_.kv_reserve(slot, tokens); }
  void kv_release(int slot) override { m_.kv_release(slot); }
  // the prefix cache (ModelOptions:snap_gib; Model's snapshot store, live resume, forks)
  int snap_capacity() const override { return m_.snap_capacity(); }
  int idle_work(int budget) override {  // one snapshot id per call, then KV lending's refills
    const int p = m_.snap_prepin(1);
    return p ? p : m_.lend_idle(std::min(budget, 16));
  }
  int snap_save(int slot) override { return m_.snap_save(slot); }
  void snap_restore(int slot, int id) override { m_.snap_restore(slot, id); }
  void snap_fork(int dst, int src, int id) override { m_.snap_fork(dst, src, id); }
  void snap_free(int id) override { m_.snap_free(id); }
  bool snap_live(int slot) const override { return m_.live_ok(slot); }
  void resume(int slot) override { m_.use_slot(slot); }
  void kv_trim(int slot, int64_t tokens) override { m_.kv_trim(slot, tokens); }
  int64_t kv_held_tokens(int slot) const override { return m_.kv_held_tokens(slot); }
  int kv_page_tokens() const override { return Model::k_page_tokens; }
  // Prefill calls at multiples of 64 with 64+ rows compute every row as a cold prefill does; a cut elsewhere or a call
  // under 64 rows does not. That holds with the fa body's context splits off (XEN_ATTN_TAIL 0); with them a row's bits
  // depend on its call's size.
  int snap_align() const override { return 64; }
  int rope_tiers() const override { return m_.rope_tiers(); }  // ModelOptions::yarn_tiers
  int64_t rope_tier_limit(int t) const override { return m_.rope_tier_limit(t); }
  void set_rope_tier(int slot, int t) override {
    m_.use_slot(slot);
    m_.set_rope_tier(t);
  }
  // the chat cut's snapshot from inside the call (Model:snap_arm / snap_take)
  bool snap_inside() const override { return m_.snap_inside(); }
  void snap_arm(int slot, int64_t pos) override { m_.snap_arm(slot, pos); }
  int snap_take() override { return m_.snap_take(); }

 private:
  Model &m_;
  int last_ = 0;   // head row of the last prefilled token (its call's last 4-row slice)
  int vrows_ = 0;  // rows of the last verify's first segment (vr_ = each segment's)
  int vsegs_ = 0;  // its segments
  std::vector<int> vr_;
  // MTP's default depths: 1-4 sequences MTP-3, 5-8 MTP-1, nine and more plain. With DFlash loaded kDFlashDepths /
  // kDFlashDepths32 replace this table.
  int depth_[kDepthSeqs] = {3, 3, 3, 3, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0};
  int wide_from_ = 0;
  std::vector<float> rows_;
  bool dev_lp_ = true;
  std::vector<float> lp_rec_, lp_xt_;
};

}  // namespace xenithium::arch::qwen35
