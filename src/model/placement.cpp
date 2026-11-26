// Xenithium · hot-expert placement that follows the traffic: see model/placement.hpp.
#include "xenithium/model/placement.hpp"

#include <algorithm>
#include <stdexcept>

namespace xenithium::model {

Placement::Placement(int layers, int experts, std::vector<uint8_t> hot, const std::vector<double> *prior, const PlacementOptions &o,
                     double prior_reads)
    : o_(o), L_(layers), E_(experts), hot_(std::move(hot)) {
  if (layers < 1 || experts < 1 || hot_.size() != static_cast<size_t>(layers) * static_cast<size_t>(experts) || o.budget < 0 || o.max_swaps < 0)
    throw std::invalid_argument("placement: shape / options");
  score_.assign(hot_.size(), 0.0);
  if (prior && prior->size() == score_.size())
    for (int l = 0; l < L_; ++l) {
      double tot = 0;
      for (int e = 0; e < E_; ++e) tot += (*prior)[idx(l, e)];
      if (tot > 0)
        for (int e = 0; e < E_; ++e) score_[idx(l, e)] = (*prior)[idx(l, e)] / tot * prior_reads;
    }
  seen_.assign(static_cast<size_t>(E_), 0u);
}

void Placement::add(int layer, const int32_t *ids, int n) {
  if (layer < 0 || layer >= L_) return;
  if (++stamp_ == 0) std::fill(seen_.begin(), seen_.end(), 0u), stamp_ = 1;
  for (int i = 0; i < n; ++i) {
    const int e = ids[i];
    if (e < 0 || e >= E_ || seen_[static_cast<size_t>(e)] == stamp_) continue;
    seen_[static_cast<size_t>(e)] = stamp_;
    score_[idx(layer, e)] += 1.0;
  }
}

void Placement::add_counts(const std::vector<double> &delta) {
  if (delta.size() != score_.size()) throw std::invalid_argument("placement: counts of another shape");
  for (size_t i = 0; i < delta.size(); ++i) score_[i] += delta[i];
}

std::vector<Placement::Swap> Placement::round(const std::vector<uint8_t> *no_enter) {
  std::vector<Swap> cand;
  std::vector<int> hot_e, cold_e;
  if (no_enter && no_enter->size() != hot_.size()) throw std::invalid_argument("placement: no_enter of another shape");
  for (int l = 0; l < L_ && o_.budget > 0 && o_.max_swaps > 0; ++l) {
    const double *s = score_.data() + idx(l, 0);
    const uint8_t *h = hot_.data() + idx(l, 0);
    const uint8_t *ne = no_enter ? no_enter->data() + idx(l, 0) : nullptr;
    hot_e.clear(), cold_e.clear();
    for (int e = 0; e < E_; ++e)
      if (h[e]) hot_e.push_back(e);
      else if (!ne || !ne[e]) cold_e.push_back(e);
    const size_t k = std::min<size_t>(static_cast<size_t>(o_.budget), std::min(hot_e.size(), cold_e.size()));
    if (k == 0) continue;
    // the k weakest hot and the k strongest cold (ties: lower id first)
    std::partial_sort(hot_e.begin(), hot_e.begin() + static_cast<std::ptrdiff_t>(k), hot_e.end(),
                      [&](int a, int b) { return s[a] != s[b] ? s[a] < s[b] : a < b; });
    std::partial_sort(cold_e.begin(), cold_e.begin() + static_cast<std::ptrdiff_t>(k), cold_e.end(),
                      [&](int a, int b) { return s[a] != s[b] ? s[a] > s[b] : a < b; });
    for (size_t i = 0; i < k; ++i) {
      const int hc = hot_e[i], cc = cold_e[i];
      if (s[cc] <= s[hc] * o_.hysteresis + o_.margin) break;
      cand.push_back({l, hc, cc, s[cc] - s[hc]});
    }
  }
  // the largest gaps first (ties: lower layer, then the pair's order in it)
  std::stable_sort(cand.begin(), cand.end(), [](const Swap &a, const Swap &b) { return a.gain > b.gain; });
  if (cand.size() > static_cast<size_t>(o_.max_swaps)) cand.resize(static_cast<size_t>(o_.max_swaps));
  for (const Swap &w : cand) {
    hot_[idx(w.layer, w.hot)] = 0;
    hot_[idx(w.layer, w.cold)] = 1;
  }
  swaps_ += static_cast<int64_t>(cand.size());
  for (double &x : score_) x *= o_.decay;
  ++rounds_;
  return cand;
}

void Placement::undo(const Swap &w) {
  hot_[idx(w.layer, w.hot)] = 1;
  hot_[idx(w.layer, w.cold)] = 0;
  --swaps_;
}

}  // namespace xenithium::model
