#include "xenithium/core/sampler.hpp"
#include "xenithium/util/gumbel.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace xenithium::core {

namespace {

constexpr float kNegInf = -std::numeric_limits<float>::infinity();

inline bool bit(const uint64_t *m, int64_t t) { return (m[t >> 6] >> (t & 63)) & 1u; }

inline uint64_t mix64(uint64_t z) {  // splitmix64's finaliser
  z += 0x9e3779b97f4a7c15ull;
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
  return z ^ (z >> 31);
}

// Appends the n best ids among those with keep(id) (descending value, ties: the lower id first).
// One pass; a sorted buffer of n entries takes the rare value above its current worst. Once the
// buffer is full, blocks of kSkip values with none above the threshold are skipped with one vectorised compare (the
// threshold only grows, so every skipped value would have failed the per-value test: the same decisions).
template <class Keep>
void select_top(const float *x, int64_t vocab, size_t n, Keep keep, std::vector<int32_t> &out) {
  if (n == 0) return;
  constexpr int64_t kSkip = 64;
  std::vector<int32_t> buf;
  buf.reserve(n + 1);
  float thr = kNegInf;  // buf's worst value once full
  for (int64_t i = 0; i < vocab;) {
    if (buf.size() == n)
      while (i + kSkip <= vocab) {
        int any = 0;
        for (int64_t j = 0; j < kSkip; ++j) any |= x[i + j] > thr;
        if (any) break;
        i += kSkip;
      }
    for (const int64_t e = std::min(vocab, i + kSkip); i < e; ++i) {
      const float v = x[i];
      if (buf.size() == n && !(v > thr)) continue;  // equal: a later id loses the tie
      if (!keep(i)) continue;
      // insert after every entry >= v (equal earlier ids stay ahead)
      size_t at = buf.size();
      while (at > 0 && x[buf[at - 1]] < v) --at;
      buf.insert(buf.begin() + static_cast<std::ptrdiff_t>(at), static_cast<int32_t>(i));
      if (buf.size() > n) buf.pop_back();
      if (buf.size() == n) thr = x[buf.back()];
    }
  }
  out.insert(out.end(), buf.begin(), buf.end());
}

}  // namespace

Sampler::Sampler(const SamplingParams &p, const std::vector<int32_t> &prompt) : p_(p) {
  if (p.has_seed) key_ = static_cast<uint64_t>(p.seed);
  else key_ = std::random_device{}() ^ (static_cast<uint64_t>(std::random_device{}()) << 32);
  if (p_.repetition_penalty != 1.0f)
    for (int32_t t : prompt)
      if (seen_set_.emplace(t, true).second) seen_.push_back(t);
}

void Sampler::accept(int32_t token) {
  ++n_out_;
  ++out_counts_[token];
  if (p_.repetition_penalty != 1.0f && seen_set_.emplace(token, true).second) seen_.push_back(token);
}

void Sampler::top_ids(const float *x, int64_t vocab, size_t n, const uint8_t *skip, std::vector<int32_t> &out) {
  n = std::min<size_t>(n, static_cast<size_t>(vocab));
  if (skip) select_top(x, vocab, n, [&](int64_t i) { return skip[i] == 0; }, out);
  else select_top(x, vocab, n, [](int64_t) { return true; }, out);
}

void Sampler::raw_logprobs(const float *x, int64_t vocab, int32_t token, int top_n, TokenLogprob &lp) {
  LpStats s;
  raw_logprob_stats(x, vocab, top_n, s);
  raw_logprobs_from(x, vocab, token, s, lp);
}

void Sampler::raw_logprobs_from(const float *x, int64_t vocab, int32_t token, const LpStats &s, TokenLogprob &lp) {
  const float xt = x[token];
  int64_t above = 0;
  for (int64_t i = 0; i < vocab; ++i) above += x[i] > xt;
  lp.rank = static_cast<int32_t>(above + 1);
  lp.logprob = static_cast<float>(static_cast<double>(x[token]) - s.lse);
  lp.top.clear();
  for (int32_t i : s.top) lp.top.push_back({i, static_cast<float>(static_cast<double>(x[i]) - s.lse)});
}

void Sampler::raw_logprob_stats(const float *x, int64_t vocab, int top_n, LpStats &st) {
  float m = kNegInf;
  for (int64_t i = 0; i < vocab; ++i) m = std::max(m, x[i]);
  st.max = m;
  // float exp per element in 16 independent lanes (vectorises without reassociation), summed in double
  double z = 0;
  constexpr int64_t kBlock = 4096;
  for (int64_t b = 0; b < vocab; b += kBlock) {
    const int64_t e = std::min(vocab, b + kBlock);
    float s[16] = {};
    int64_t i = b;
    for (; i + 16 <= e; i += 16)
      for (int j = 0; j < 16; ++j) s[j] += std::exp(x[i + j] - m);
    for (; i < e; ++i) s[0] += std::exp(x[i] - m);
    for (float v : s) z += v;
  }
  st.lse = m + std::log(z);
  st.top.clear();
  if (top_n > 0) top_ids(x, vocab, static_cast<size_t>(top_n), nullptr, st.top);
}

void Sampler::merge_logprobs(const LpPart *parts, int n, float xt, int top_n, TokenLogprob &lp) {
  float m = kNegInf;
  int64_t above = 0;
  for (int c = 0; c < n; ++c) {
    m = std::max(m, parts[c].max);
    above += parts[c].above;
  }
  double z = 0;
  for (int c = 0; c < n; ++c) z += parts[c].sum * std::exp(static_cast<double>(parts[c].max) - m);
  const double lse = m + std::log(z);
  lp.rank = static_cast<int32_t>(above + 1);
  lp.logprob = static_cast<float>(static_cast<double>(xt) - lse);
  std::vector<std::pair<int32_t, float>> all;
  for (int c = 0; c < n; ++c) all.insert(all.end(), parts[c].top.begin(), parts[c].top.end());
  std::sort(all.begin(), all.end(), [](const auto &a, const auto &b) { return a.second != b.second ? a.second > b.second : a.first < b.first; });
  lp.top.clear();
  for (size_t i = 0; i < all.size() && static_cast<int>(i) < top_n; ++i)
    lp.top.push_back({all[i].first, static_cast<float>(static_cast<double>(all[i].second) - lse)});
}

void Sampler::modifiers(const float *x, int64_t vocab, const uint64_t *allowed, const int32_t *extra, int n_extra, Row &row) const {
  auto &mods = row.mods;
  auto &flag = row.flag;
  mods.clear();
  if (flag.size() != static_cast<size_t>(vocab)) flag.assign(static_cast<size_t>(vocab), 0);
  auto touch = [&](int32_t t) {
    if (t < 0 || t >= vocab || flag[static_cast<size_t>(t)]) return;
    flag[static_cast<size_t>(t)] = 1;
    mods.push_back({t, 0.0f});
  };
  const bool counted = p_.frequency_penalty != 0.0f || p_.presence_penalty != 0.0f;
  for (const auto &[t, b] : p_.logit_bias) touch(t);
  if (p_.repetition_penalty != 1.0f)
    for (int32_t t : seen_) touch(t);
  if (counted)
    for (const auto &[t, c] : out_counts_) touch(t);
  // tokens taken as accepted (prepare_row) count as output tokens: seen, and in the counts
  if (p_.repetition_penalty != 1.0f || counted)
    for (int i = 0; i < n_extra; ++i) touch(extra[i]);
  auto extra_count = [&](int32_t t) {
    int32_t c = 0;
    for (int i = 0; i < n_extra; ++i) c += extra[i] == t;
    return c;
  };
  // the processed value, in the full pass's order (the same float operations)
  const float r = p_.repetition_penalty;
  for (auto &[t, v] : mods) {
    v = x[t];
    if (const auto b = p_.logit_bias.find(t); b != p_.logit_bias.end()) v += b->second;
    if (r != 1.0f && (seen_set_.count(t) || (n_extra > 0 && extra_count(t) > 0))) v = v > 0 ? v / r : v * r;
    if (counted) {
      int32_t c = n_extra > 0 ? extra_count(t) : 0;
      if (const auto it = out_counts_.find(t); it != out_counts_.end()) c += it->second;
      if (c > 0) v -= p_.frequency_penalty * static_cast<float>(c) + p_.presence_penalty;
    }
    if (allowed && !bit(allowed, t)) v = kNegInf;
  }
}

// The full-vocabulary path: a processed copy of the row, then min_p / top_k / top_p over it.
void Sampler::full_path(const float *logits, int64_t vocab, const uint64_t *allowed, Row &row) const {
  auto &work = row.work;
  auto &idx = row.idx;
  work.assign(logits, logits + vocab);
  float *x = work.data();
  for (const auto &[t, v] : row.mods) x[t] = v;
  if (allowed)
    for (int64_t i = 0; i < vocab; ++i)
      if (!bit(allowed, i)) x[i] = kNegInf;
  const float inv_t = 1.0f / p_.temperature;
  float m = kNegInf;
  for (int64_t i = 0; i < vocab; ++i) {
    x[i] *= inv_t;
    m = std::max(m, x[i]);
  }
  const float floor_x = p_.min_p > 0.0f ? m + std::log(p_.min_p) : kNegInf;
  idx.clear();
  for (int64_t i = 0; i < vocab; ++i)
    if (x[i] >= floor_x && x[i] > kNegInf) idx.push_back(static_cast<int32_t>(i));
  auto better = [&](int32_t a, int32_t b) { return x[a] > x[b] || (x[a] == x[b] && a < b); };
  if (p_.top_k > 0 && static_cast<size_t>(p_.top_k) < idx.size()) {
    std::nth_element(idx.begin(), idx.begin() + p_.top_k, idx.end(), better);
    idx.resize(static_cast<size_t>(p_.top_k));
    std::sort(idx.begin(), idx.end(), better);
  }
  if (p_.top_p < 1.0f) {
    double z = 0;
    for (int32_t i : idx) z += std::exp(static_cast<double>(x[i]) - m);
    const double need = static_cast<double>(p_.top_p) * z;
    size_t head = std::min<size_t>(idx.size(), 1024);
    while (true) {
      std::partial_sort(idx.begin(), idx.begin() + static_cast<std::ptrdiff_t>(head), idx.end(), better);
      double before = 0;
      size_t keep = 0;
      while (keep < head && (keep == 0 || before < need)) before += std::exp(static_cast<double>(x[idx[keep++]]) - m);
      if (keep < head || head == idx.size()) {
        idx.resize(keep);
        break;
      }
      head = std::min(idx.size(), head * 8);
    }
  }
  row.w.resize(idx.size());
  for (size_t i = 0; i < idx.size(); ++i) row.w[i] = std::exp(static_cast<double>(x[idx[i]]) - m);
}

int32_t Sampler::prepare(const float *x, int64_t vocab, const uint64_t *allowed, const int32_t *extra, int n_extra, Row &row) const {
  if (allowed) {  // an empty allowed set is ignored
    bool any = false;
    for (int64_t w = 0; w < (vocab + 63) / 64 && !any; ++w) any = allowed[w] != 0;
    if (!any) allowed = nullptr;
  }
  modifiers(x, vocab, allowed, extra, n_extra, row);
  const uint8_t *flag = row.flag.data();
  auto untouched = [&](int64_t i) { return flag[i] == 0 && (!allowed || bit(allowed, i)); };
  int32_t pick = -1;
  row.idx.clear();
  row.w.clear();
  if (p_.temperature <= 0.0f) {
    // greedy over the processed logits; ties -> the lowest id
    std::vector<int32_t> one;
    select_top(x, vocab, 1, untouched, one);
    float best = kNegInf;
    if (!one.empty()) pick = one[0], best = x[one[0]];
    for (const auto &[t, v] : row.mods)
      if (pick < 0 || v > best || (v == best && t < pick)) pick = t, best = v;
    if (pick < 0) pick = 0;
  } else if (p_.top_k > 0 && p_.top_k <= kMaxCandidates) {
    // candidates: the top_k untouched raw logits + every touched token, processed
    const auto k = static_cast<size_t>(p_.top_k);
    std::vector<int32_t> cand;
    select_top(x, vocab, k, untouched, cand);
    std::vector<std::pair<float, int32_t>> c;  // (processed value, id)
    c.reserve(cand.size() + row.mods.size());
    for (int32_t t : cand) c.push_back({x[t], t});
    for (const auto &[t, v] : row.mods)
      if (v > kNegInf) c.push_back({v, t});
    keep_top(c, k, row);
  } else {
    full_path(x, vocab, allowed, row);
  }
  for (const auto &[t, v] : row.mods) row.flag[static_cast<size_t>(t)] = 0;
  row.pick = pick;
  return pick;
}

void Sampler::keep_top(std::vector<std::pair<float, int32_t>> &c, size_t k, Row &row) const {
  // the k best in order (ids are distinct, so the order is total: the same prefix a full sort gives; a
  // repetition penalty over a long prompt puts thousands of touched tokens here)
  const auto better = [](const auto &a, const auto &b) { return a.first > b.first || (a.first == b.first && a.second < b.second); };
  if (c.size() > k) {
    std::partial_sort(c.begin(), c.begin() + static_cast<std::ptrdiff_t>(k), c.end(), better);
    c.resize(k);
  } else {
    std::sort(c.begin(), c.end(), better);
  }
  const float inv_t = 1.0f / p_.temperature;
  for (auto &e : c) e.first *= inv_t;
  const float m = c.empty() ? 0.0f : c[0].first;  // the processed row's max is the first candidate
  const float floor_x = p_.min_p > 0.0f ? m + std::log(p_.min_p) : kNegInf;
  // c is sorted: the kept tokens are a prefix
  for (const auto &e : c)
    if (e.first >= floor_x && e.first > kNegInf) {
      row.idx.push_back(e.second);
      row.w.push_back(std::exp(static_cast<double>(e.first) - m));
    }
  if (p_.top_p < 1.0f && !row.idx.empty()) {
    const double z = std::accumulate(row.w.begin(), row.w.end(), 0.0);
    const double need = static_cast<double>(p_.top_p) * z;
    double before = 0;
    size_t keep = 0;
    while (keep < row.idx.size() && (keep == 0 || before < need)) before += row.w[keep++];
    row.idx.resize(keep);
    row.w.resize(keep);
  }
}

bool Sampler::cands_ok(int T) const {
  return p_.temperature > 0.0f && p_.top_k > 0 && p_.top_k <= T && p_.top_k <= kMaxCandidates && p_.logit_bias.empty() &&
         p_.repetition_penalty == 1.0f && p_.frequency_penalty == 0.0f && p_.presence_penalty == 0.0f;
}

void Sampler::prepare_cands(const int32_t *ids, const float *vals, int n, Row &row) const {
  // == prepare()'s candidate branch with no touched token: its select_top keeps the k best raw values (value
  // descending, lower id first), which the candidates hold (cands_ok); keep_top then sorts the same pairs the same way.
  row.mods.clear();
  row.idx.clear();
  row.w.clear();
  row.pick = -1;
  std::vector<std::pair<float, int32_t>> c;
  c.reserve(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) c.push_back({vals[i], ids[i]});
  keep_top(c, static_cast<size_t>(p_.top_k), row);
}

double Sampler::uniform() const {
  const uint64_t h = mix64(key_ ^ mix64(n_out_ ^ 0x5851f42d4c957f2dull));
  return (static_cast<double>(h >> 11) + 0.5) * 0x1p-53;  // (0, 1), 53 bits
}

bool Sampler::coupled() const { return p_.temperature > 0.0f && p_.top_k > 0 && p_.top_k <= kMaxCandidates; }

Sampler::DraftNoise Sampler::draft_noise() const {
  DraftNoise d;
  d.key = key_;
  d.pos = n_out_;
  d.inv_t = coupled() ? 1.0f / p_.temperature : 0.0f;
  return d;
}

int32_t Sampler::walk(const Row &row) const {
  if (coupled()) {  // Gumbel-max over the kept set (row.w = exp(x / T - m): log w is the scaled logit)
    int32_t out = -1;
    double best = -std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < row.idx.size(); ++i) {
      if (!(row.w[i] > 0)) continue;
      const double s = std::log(row.w[i]) + static_cast<double>(util::gumbel(key_, n_out_, row.idx[i]));
      if (s > best) best = s, out = row.idx[i];  // ties: the earlier (better) candidate
    }
    return out;
  }
  double total = 0;
  for (double v : row.w) total += v;
  if (!(total > 0)) return -1;
  double r = uniform() * total;
  int32_t out = -1;
  for (size_t i = 0; i < row.idx.size(); ++i) {
    if (row.w[i] <= 0) continue;
    out = row.idx[i];  // the last kept token takes the rounding remainder
    if (r < row.w[i]) break;
    r -= row.w[i];
  }
  return out;
}

int32_t Sampler::sample_cands(const int32_t *ids, const float *vals, int n) {
  prepare_cands(ids, vals, n, cur_);
  if (cur_.idx.empty()) return -1;  // the full row decides (sample())
  return walk(cur_);
}

int32_t Sampler::draw(Row &row, const float *logits, int64_t vocab, int32_t draft, const uint64_t *allowed) {
  (void)draft;  // the draw does not depend on it: the caller accepts the draft iff it is this token
  if (row.pick >= 0) return row.pick;  // greedy
  if (row.idx.empty()) return sample(logits, vocab, nullptr, 0, allowed);
  const int32_t t = walk(row);
  return t >= 0 ? t : row.idx.front();
}

int32_t Sampler::speculate(const float *logits, int64_t vocab, int32_t draft, const uint64_t *allowed) {
  prepare(logits, vocab, allowed);
  return draw(cur_, logits, vocab, draft, allowed);
}

void Sampler::prepare_row(const float *logits, int64_t vocab, const int32_t *extra, int n_extra, Row &row) const {
  prepare(logits, vocab, nullptr, extra, n_extra, row);
}

int32_t Sampler::speculate_row(Row &row, const float *logits, int64_t vocab, int32_t draft) { return draw(row, logits, vocab, draft, nullptr); }

void Sampler::distribution(const float *logits, int64_t vocab, std::vector<int32_t> &ids, std::vector<double> &w, const uint64_t *allowed) {
  prepare(logits, vocab, allowed);
  ids = cur_.idx;
  w = cur_.w;
}

int32_t Sampler::sample(const float *logits, int64_t vocab, TokenLogprob *lp, int top_n, const uint64_t *allowed) {
  int32_t pick = prepare(logits, vocab, allowed);
  if (pick < 0) {
    if (cur_.idx.empty()) {  // nothing finite is left (every allowed logit -inf): the first allowed / raw argmax
      std::vector<int32_t> one;
      top_ids(logits, vocab, 1, nullptr, one);
      pick = one.empty() ? 0 : one[0];
    } else {
      pick = walk(cur_);
      if (pick < 0) pick = cur_.idx.front();
    }
  }
  if (lp) raw_logprobs(logits, vocab, pick, top_n, *lp);
  return pick;
}

}  // namespace xenithium::core
