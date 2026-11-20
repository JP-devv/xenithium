#include "xenithium/eval/logprobs.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <numeric>

namespace xenithium::eval {

namespace {
bool fail(std::string *err, const std::string &m) {
  if (err) *err = m;
  return false;
}
}  // namespace

bool write_lp(const std::string &path, const LpSet &s, std::string *err) {
  std::ofstream f(path, std::ios::binary);
  if (!f) return fail(err, "cannot write " + path);
  const int32_t k = s.k, n = static_cast<int32_t>(s.chunks.size());
  f.write("XLP1", 4);
  f.write(reinterpret_cast<const char *>(&k), 4);
  f.write(reinterpret_cast<const char *>(&n), 4);
  std::vector<int32_t> ids(static_cast<size_t>(k));
  std::vector<float> lps(static_cast<size_t>(k));
  for (const auto &c : s.chunks) {
    const int32_t rows = static_cast<int32_t>(c.size());
    f.write(reinterpret_cast<const char *>(&rows), 4);
    for (const auto &r : c) {
      std::fill(ids.begin(), ids.end(), -1);
      std::fill(lps.begin(), lps.end(), -std::numeric_limits<float>::infinity());
      const size_t m = std::min(r.ids.size(), ids.size());
      std::copy_n(r.ids.begin(), m, ids.begin());
      std::copy_n(r.lps.begin(), m, lps.begin());
      f.write(reinterpret_cast<const char *>(&r.target), 4);
      f.write(reinterpret_cast<const char *>(&r.target_lp), 4);
      f.write(reinterpret_cast<const char *>(ids.data()), 4LL * k);
      f.write(reinterpret_cast<const char *>(lps.data()), 4LL * k);
    }
  }
  return f.good() ? true : fail(err, "write error on " + path);
}

bool read_lp(const std::string &path, LpSet &s, std::string *err) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return fail(err, "cannot open " + path);
  char magic[4];
  int32_t k = 0, n = 0;
  if (!f.read(magic, 4) || std::memcmp(magic, "XLP1", 4) != 0) return fail(err, path + ": not an XLP1 file");
  if (!f.read(reinterpret_cast<char *>(&k), 4) || !f.read(reinterpret_cast<char *>(&n), 4) || k <= 0 || k > 4096 || n < 0 || n > 1000000)
    return fail(err, path + ": bad header");
  s.k = k;
  s.chunks.assign(static_cast<size_t>(n), {});
  for (auto &c : s.chunks) {
    int32_t rows = 0;
    if (!f.read(reinterpret_cast<char *>(&rows), 4) || rows < 0 || rows > (1 << 24)) return fail(err, path + ": bad chunk header");
    c.resize(static_cast<size_t>(rows));
    for (auto &r : c) {
      r.ids.resize(static_cast<size_t>(k));
      r.lps.resize(static_cast<size_t>(k));
      if (!f.read(reinterpret_cast<char *>(&r.target), 4) || !f.read(reinterpret_cast<char *>(&r.target_lp), 4) ||
          !f.read(reinterpret_cast<char *>(r.ids.data()), 4LL * k) || !f.read(reinterpret_cast<char *>(r.lps.data()), 4LL * k))
        return fail(err, path + ": truncated");
      int m = k;
      while (m > 0 && r.ids[static_cast<size_t>(m - 1)] < 0) --m;  // drop the padding
      r.ids.resize(static_cast<size_t>(m));
      r.lps.resize(static_cast<size_t>(m));
    }
  }
  if (f.peek() != std::char_traits<char>::eof()) return fail(err, path + ": trailing bytes");
  return true;
}

LpRow top_logprobs(const float *l, int64_t V, int k, int32_t target) {
  int64_t best = 0;
  for (int64_t i = 1; i < V; ++i)
    if (l[i] > l[best]) best = i;
  const double mx = l[best];
  double se = 0;
  for (int64_t i = 0; i < V; ++i) se += std::exp(static_cast<double>(l[i]) - mx);
  const double lse = mx + std::log(se);
  k = static_cast<int>(std::min<int64_t>(k, V));
  std::vector<int32_t> idx(static_cast<size_t>(V));
  std::iota(idx.begin(), idx.end(), 0);
  auto before = [&](int32_t a, int32_t b) { return l[a] > l[b] || (l[a] == l[b] && a < b); };
  std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), before);
  LpRow r;
  r.target = target;
  r.target_lp = static_cast<float>(static_cast<double>(l[target]) - lse);
  r.ids.assign(idx.begin(), idx.begin() + k);
  for (int32_t id : r.ids) r.lps.push_back(static_cast<float>(static_cast<double>(l[id]) - lse));
  return r;
}

KlRow kl_coarse(const LpRow &ref, const std::function<double(int32_t, bool &)> &test_lp, int32_t test_top1) {
  KlRow o;
  double P = 0, Q = 0, kl = 0;
  for (size_t i = 0; i < ref.ids.size(); ++i) {
    bool exact = true;
    const double lr = ref.lps[i], lt = test_lp(ref.ids[i], exact);
    o.missing += !exact;
    const double p = std::exp(lr), q = std::exp(lt);
    P += p;
    Q += q;
    kl += p * (lr - lt);
  }
  // The reference dump's log-probs are fp32: their sum over a full row misses 1 by ~1e-7, so a tail
  // below 1e-6 is rounding, not mass (and 1 - Q there is ~0: the term would explode).
  const double pt = 1.0 - P, qt = std::max(1.0 - Q, 1e-30);
  if (pt > 1e-6) kl += pt * std::log(pt / qt);
  o.kl = std::max(kl, 0.0);  // rounding of exp/log can leave -1e-17 on identical rows
  o.top1_same = !ref.ids.empty() && ref.ids[0] == test_top1;
  return o;
}

KlStats::KlStats(std::vector<int64_t> e) : edges(std::move(e)), buckets(edges.size() + 1) {}

void KlStats::add(int64_t pos, const KlRow &r, double ref_target_lp, double test_target_lp) {
  size_t b = 0;
  while (b < edges.size() && pos >= edges[b]) ++b;
  for (Bucket *x : {&buckets[b], &total}) {
    ++x->rows;
    x->top1 += r.top1_same;
    x->missing += r.missing;
    x->kl += r.kl;
    x->kl_max = std::max(x->kl_max, r.kl);
    x->nll_ref -= ref_target_lp;
    x->nll_test -= test_target_lp;
    x->kls.push_back(r.kl);
  }
}

double percentile(std::vector<double> v, double p) {
  if (v.empty()) return 0;
  std::sort(v.begin(), v.end());
  return v[std::min(v.size() - 1, static_cast<size_t>(p * static_cast<double>(v.size())))];
}

std::string KlStats::table() const {
  std::string out;
  char line[256];
  std::snprintf(line, sizeof line, "  %-17s %7s %9s %9s %10s %10s %10s %8s %7s\n", "positions", "rows", "ppl ref", "ppl test", "KL mean", "KL p99",
                "KL max", "top-1 =", "bound");
  out += line;
  auto row = [&](const std::string &name, const Bucket &x) {
    if (x.rows == 0) return;
    const double n = static_cast<double>(x.rows);
    std::snprintf(line, sizeof line, "  %-17s %7lld %9.4f %9.4f %10.6f %10.6f %10.6f %7.2f%% %7lld\n", name.c_str(), static_cast<long long>(x.rows),
                  std::exp(x.nll_ref / n), std::exp(x.nll_test / n), x.kl / n, percentile(x.kls, 0.99), x.kl_max, 100.0 * static_cast<double>(x.top1) / n,
                  static_cast<long long>(x.missing));
    out += line;
  };
  for (size_t b = 0; b < buckets.size(); ++b) {
    const std::string lo = std::to_string(b == 0 ? 0 : edges[b - 1]);
    row(b < edges.size() ? lo + "-" + std::to_string(edges[b]) : lo + "+", buckets[b]);
  }
  row("all", total);
  return out;
}

}  // namespace xenithium::eval
