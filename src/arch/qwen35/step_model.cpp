#include "xenithium/arch/qwen35/step_model.hpp"

#include "xenithium/core/sampler.hpp"
#include "xenithium/kernels/head/logprob.hpp"

#include <algorithm>
#include <stdexcept>

namespace xenithium::arch::qwen35 {

void StepModelAdapter::begin(int slot) {
  m_.use_slot(slot);
  m_.reset();
}

void StepModelAdapter::prefill(int slot, const int32_t *tokens, int n, int32_t next) {
  m_.use_slot(slot);
  m_.prefill(tokens, n, {}, next);
  last_ = (n - 1) % Model::kRows;  // chunks are multiples of 4 rows: the head's last slice holds the call's last rows
}

void StepModelAdapter::prefill_begin(int slot, const int32_t *tokens, int n, int32_t next) {
  m_.use_slot(slot);
  m_.prefill_begin(tokens, n, next);
  last_ = (n - 1) % Model::kRows;
}

void StepModelAdapter::prefill_begin_many(const std::vector<PrefillSeg> &segs) {
  std::vector<Model::PrefillSeg> s(segs.size());
  for (size_t i = 0; i < segs.size(); ++i) s[i] = Model::PrefillSeg{segs[i].slot, segs[i].tokens, segs[i].n, segs[i].next};
  m_.prefill_begin(s.data(), static_cast<int>(s.size()));
  last_ = 0;
}

void StepModelAdapter::prefill_result(int slot) {
  m_.use_slot(slot);
  last_ = m_.prefill_last_row();  // the member's row of the call's one head
}

void StepModelAdapter::prefill_rows(int slot, const int32_t *tokens, int n, int32_t next, const RowsFn &on_rows) {
  m_.use_slot(slot);
  const int64_t p0 = m_.position();
  m_.prefill(
      tokens, n,
      [&](int64_t pos, int rows) {
        m_.logits(rows, rows_);
        on_rows(static_cast<int>(pos - p0), rows, rows_.data());
      },
      next);
  last_ = (n - 1) % Model::kRows;
}

// Logprobs on the cards. Each card's record of its vocabulary part, merged here.
void StepModelAdapter::row_logprobs(int rows, const int32_t *targets, int top_n, core::TokenLogprob *out) {
  if (rows < 1 || rows > kernels::kLpMaxRows) throw std::invalid_argument("qwen35 adapter: logprobs for 1..16 rows (one head pass)");
  m_.logprob_stats(rows, targets, std::clamp(top_n, 0, kernels::kLpMaxTop), lp_rec_, lp_xt_);
  const int C = m_.cards();
  for (int r = 0; r < rows; ++r) {
    if (targets[r] < 0) continue;
    core::Sampler::LpPart parts[2];
    for (int c = 0; c < C; ++c) {
      const float *x = lp_rec_.data() + (static_cast<size_t>(c) * rows + r) * kernels::kLpRec;
      auto &p = parts[c];
      p.max = x[0];
      p.sum = x[1];
      p.above = static_cast<int64_t>(x[2]);
      const int k = static_cast<int>(x[3]);
      for (int i = 0; i < k; ++i) p.top.push_back({static_cast<int32_t>(x[4 + 2 * i]), x[5 + 2 * i]});
    }
    core::Sampler::merge_logprobs(parts, C, lp_xt_[static_cast<size_t>(r)], top_n, out[r]);
  }
}

void StepModelAdapter::prefill_logprobs(int32_t token, int top_n, core::TokenLogprob &out) {
  int32_t t[kernels::kLpMaxRows];
  std::fill(std::begin(t), std::end(t), -1);
  t[last_] = token;
  core::TokenLogprob lp[kernels::kLpMaxRows];
  row_logprobs(last_ + 1, t, top_n, lp);
  out = std::move(lp[last_]);
}

void StepModelAdapter::prefill_each(int slot, const int32_t *tokens, int n, int32_t next, const EachFn &on_rows) {
  m_.use_slot(slot);
  const int64_t p0 = m_.position();
  m_.prefill(tokens, n, [&](int64_t pos, int rows) { on_rows(static_cast<int>(pos - p0), rows); }, next);
  last_ = (n - 1) % Model::kRows;
}

int32_t StepModelAdapter::prefill_pick() { return m_.picks(0, last_ + 1)[static_cast<size_t>(last_)]; }

void StepModelAdapter::prefill_logits(std::vector<float> &row) {
  std::vector<float> all;
  m_.logits(last_ + 1, all);
  const auto v = static_cast<size_t>(m_.config().vocab);
  row.assign(all.begin() + static_cast<std::ptrdiff_t>(static_cast<size_t>(last_) * v),
             all.begin() + static_cast<std::ptrdiff_t>(static_cast<size_t>(last_ + 1) * v));
}

std::vector<int32_t> StepModelAdapter::first_drafts(int slot, int32_t first, int depth, int sequences) {
  m_.use_slot(slot);
  // MTP: the prompt's last row reruns on the drafter with the emitted token (its drafter KV) even when no draft is wanted;
  // DFlash: the prefill put the whole prompt in the drafter's context, depth 0 runs nothing; both:
  // the drafter of a step of `sequences`
  const bool mtp = m_.mtp_drafts(std::max(1, sequences));
  auto d = m_.draft_after_prefill(first, mtp ? std::clamp(depth, 1, m_.draft_depth_max()) : std::clamp(depth, 0, m_.draft_depth_max()), std::max(1, sequences));
  d.resize(std::min(d.size(), static_cast<size_t>(std::max(depth, 0))));
  return d;
}

// A tree step attends densely: with page selection, only while every sequence of the step ends under the budget.
bool StepModelAdapter::verifies_trees(std::span<const int> slots, std::span<const int> rows) const {
  int64_t pos_bound = 0;
  for (size_t i = 0; i < slots.size(); ++i) pos_bound = std::max<int64_t>(pos_bound, m_.position(slots[i]) + rows[i]);
  return !m_.selects_pages(pos_bound);
}

void StepModelAdapter::verify(const std::vector<SpecSeg> &segs) {
  // DDTree: segments whose drafts are a tree (SpecSeg::parent; any number of them)
  std::vector<Model::SeqRows> v;
  for (const auto &g : segs) v.push_back({g.slot, g.tokens, g.rows, g.parent});
  m_.verify_batch(v);
  vrows_ = segs.empty() ? 0 : segs[0].rows;
  vsegs_ = static_cast<int>(segs.size());
  vr_.clear();
  for (const auto &g : segs) vr_.push_back(g.rows);  // unequal segments
}

std::vector<std::vector<int32_t>> StepModelAdapter::accept(const std::vector<SpecAccept> &a) {
  if (static_cast<int>(a.size()) != vsegs_) throw std::invalid_argument("qwen35 adapter: one accept per verified segment");
  std::vector<int> keep, depth;
  int32_t nx[kernels::kStepRows32];
  for (int k = 0, r0 = 0; k < vsegs_; r0 += vr_[static_cast<size_t>(k)], ++k) {
    const SpecAccept &x = a[static_cast<size_t>(k)];
    keep.push_back(x.n);
    // pass 1 over every verified row (rows past n write drafter KV that later passes rewrite); each segment's own rows
    // and depth (unequal segments)
    for (int r = 0; r < vr_[static_cast<size_t>(k)]; ++r) nx[r0 + r] = x.next[std::min(r, x.n - 1)];
    depth.push_back(std::max(0, std::min(x.depth, m_.draft_depth_max())));
  }
  // the walked rows of a tree (per segment)
  std::vector<const int *> paths;
  bool any = false;
  for (const auto &x : a) paths.push_back(x.path), any |= x.path != nullptr;
  if (any) m_.accept_batch(keep, paths);
  else m_.accept_batch(keep);
  // depth 0: pass 1 writes the drafter KV only (every slot's drafter KV stays complete for its later drafts)
  auto d = m_.draft_batch(nx, depth);
  for (int k = 0; k < vsegs_; ++k) d[static_cast<size_t>(k)].resize(std::min(d[static_cast<size_t>(k)].size(), static_cast<size_t>(std::max(a[static_cast<size_t>(k)].depth, 0))));
  return d;
}

void StepModelAdapter::step(const std::vector<int> &slots, const int32_t *tokens) {
  std::vector<Model::SeqRows> v;
  for (size_t i = 0; i < slots.size(); ++i) v.push_back({slots[i], tokens + i, 1});
  m_.verify_batch(v);
  m_.accept_batch(std::vector<int>(slots.size(), 1));  // a plain step keeps its rows (the GDN commits them at the next step)
}

void StepModelAdapter::picks(int rows, int32_t *out) {
  const auto t = m_.picks(0, rows);
  std::copy(t.begin(), t.begin() + rows, out);
}

}  // namespace xenithium::arch::qwen35
