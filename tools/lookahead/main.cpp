// xenithium-lookahead: how well can a prefill call's routed cold experts be
// named before the layer runs? Scores routing predictors offline on a capture from
// `xenithium-probe model --pf-route-trace F` (XLA1: per prefill call and layer the attention mixer's
// output xa, the MLP mixer's output xm = the router's input, and the MoE's routing ids).
// A predictor applies layer t's router (bf16 [experts][hidden] from the checkpoint, fp32 sums on the
// host) to some earlier rows and takes each row's top K; the call's predicted set is the union over
// its rows. The routed set is the union of the device's top-k ids at t. Predictors:
// self router t on xm_t (the host router against the device's ids: a self-check)
// next router t on xm_{t-1} (one MoE + one attention block of lead time)
// next2 router t on xm_{t-2} (two layers of lead)
// attn router t on xa_t (only layer t's attention block of lead)
// static the first N cold experts of layer t by the hot ranking (the prefix), N = K x 2
// (no input: the baseline a predictor has to beat on bytes)
// Cold sets come from the hot ranking (XRC1, read as the engine places under expert TP: one stable
// ranking over (layer, expert), the top `hot` per card in VRAM), for each --hot count.
// Per (predictor, K, hot): over all calls, routed cold experts (A), predicted cold (P), predicted and
// routed (both); recall = both / A, precision = both / P; per 512-row call the bytes a prefetch would
// copy (P x the blob bytes per card) and the routed cold bytes it would miss. Also the recall over
// all routed experts (hot or cold), which does not depend on the placement.
// usage: xenithium-lookahead --model DIR --capture F --hot-rank F.xrc [--hot 23251,22585]
// [--k 10,12,16,20,24,32] [--threads T] [--labels a,b..]
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

#include "xenithium/arch/qwen38/arch.hpp"
#include "xenithium/model/checkpoint.hpp"

namespace {

namespace xm = xenithium::model;

float bf16f(uint16_t b) {
  const uint32_t u = static_cast<uint32_t>(b) << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

std::vector<int> int_list(const std::string &s) {
  std::vector<int> v;
  std::stringstream ss(s);
  std::string t;
  while (std::getline(ss, t, ','))
    if (!t.empty()) v.push_back(std::atoi(t.c_str()));
  return v;
}

std::vector<std::string> str_list(const std::string &s) {
  std::vector<std::string> v;
  std::stringstream ss(s);
  std::string t;
  while (std::getline(ss, t, ','))
    if (!t.empty()) v.push_back(t);
  return v;
}

void pread_all(int fd, void *dst, size_t n, uint64_t off) {
  auto *p = static_cast<uint8_t *>(dst);
  while (n > 0) {
    const ssize_t r = ::pread(fd, p, n, static_cast<off_t>(off));
    if (r <= 0) throw std::runtime_error("lookahead: short read of the checkpoint");
    p += r;
    n -= static_cast<size_t>(r);
    off += static_cast<uint64_t>(r);
  }
}

bool read_n(std::FILE *f, void *dst, size_t bytes) { return std::fread(dst, 1, bytes, f) == bytes; }

// Each row's top `K` expert ids by logit (descending; ties to the lower id), router W [E][H] fp32,
// rows x [m][H] bf16. 16 partial sums per dot: a fixed order (deterministic) that vectorises.
void top_rows(const std::vector<float> &W, int E, int H, const uint16_t *x, int m, int K, int threads, std::vector<int32_t> &out) {
  out.assign(static_cast<size_t>(m) * K, -1);
  auto work = [&](int r0, int r1) {
    std::vector<float> xf(static_cast<size_t>(H)), lg(static_cast<size_t>(E));
    std::vector<int32_t> idx(static_cast<size_t>(E));
    for (int r = r0; r < r1; ++r) {
      for (int h = 0; h < H; ++h) xf[static_cast<size_t>(h)] = bf16f(x[static_cast<size_t>(r) * H + h]);
      for (int e = 0; e < E; ++e) {
        const float *w = W.data() + static_cast<size_t>(e) * H;
        float acc[16] = {};
        for (int h = 0; h + 16 <= H; h += 16)
          for (int j = 0; j < 16; ++j) acc[j] += w[h + j] * xf[static_cast<size_t>(h + j)];
        float s = 0;
        for (int j = 0; j < 16; ++j) s += acc[j];
        // as the device's topk keys: the logit rounded to bf16 (nearest even), ties to the lower id
        uint32_t u;
        std::memcpy(&u, &s, 4);
        u = (u + 0x7fffu + ((u >> 16) & 1u)) & 0xffff0000u;
        std::memcpy(&s, &u, 4);
        lg[static_cast<size_t>(e)] = s;
      }
      for (int e = 0; e < E; ++e) idx[static_cast<size_t>(e)] = e;
      std::partial_sort(idx.begin(), idx.begin() + K, idx.end(), [&](int32_t a, int32_t b) {
        return lg[static_cast<size_t>(a)] != lg[static_cast<size_t>(b)] ? lg[static_cast<size_t>(a)] > lg[static_cast<size_t>(b)] : a < b;
      });
      std::copy(idx.begin(), idx.begin() + K, out.begin() + static_cast<ptrdiff_t>(r) * K);
    }
  };
  std::vector<std::thread> th;
  const int T = std::max(1, std::min(threads, m));
  for (int t = 0; t < T; ++t) th.emplace_back(work, m * t / T, m * (t + 1) / T);
  for (auto &t : th) t.join();
}

struct Tally {
  double routed_cold = 0, pred_cold = 0, both_cold = 0;  // expert counts summed over (call, layer)
  double routed_all = 0, pred_all = 0, both_all = 0;
  double cold_rows = 0, cold_rows_hit = 0;  // routes (row assignments) to routed cold experts, and to the hit ones
  double layers = 0;  // (call, layer) pairs scored
};

}  // namespace

int main(int argc, char **argv) {
  std::string model, capture, hot_rank, hot_arg = "23251,22585", k_arg = "10,12,16,20,24,32", labels_arg;
  int threads = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string a = argv[i], v = argv[i + 1];
    if (a == "--model") model = v;
    else if (a == "--capture") capture = v;
    else if (a == "--hot-rank") hot_rank = v;
    else if (a == "--hot") hot_arg = v;
    else if (a == "--k") k_arg = v;
    else if (a == "--threads") threads = std::atoi(v.c_str());
    else if (a == "--labels") labels_arg = v;
    else {
      std::fprintf(stderr, "lookahead: unknown option %s\n", a.c_str());
      return 2;
    }
  }
  if (model.empty() || capture.empty() || hot_rank.empty()) {
    std::fprintf(stderr, "usage: xenithium-lookahead --model DIR --capture F --hot-rank F.xrc [--hot N,N] [--k K,K] [--threads T] [--labels a,b]\n");
    return 2;
  }
  const std::vector<int> hots = int_list(hot_arg), ks = int_list(k_arg);
  const std::vector<std::string> labels = str_list(labels_arg);
  const int Kmax = *std::max_element(ks.begin(), ks.end());

  std::FILE *cf = std::fopen(capture.c_str(), "rb");
  if (!cf) {
    std::fprintf(stderr, "lookahead: cannot open %s\n", capture.c_str());
    return 2;
  }
  uint32_t magic = 0;
  int32_t hdr[4];
  uint64_t stride = 0;
  if (!read_n(cf, &magic, 4) || magic != 0x31414C58u || !read_n(cf, hdr, 16) || !read_n(cf, &stride, 8)) {
    std::fprintf(stderr, "lookahead: %s is not an XLA1 capture\n", capture.c_str());
    return 2;
  }
  const int L = hdr[0], E = hdr[1], TK = hdr[2], H = hdr[3];
  if (Kmax > E || TK > E) throw std::runtime_error("lookahead: K above the expert count");

  // ---- routers from the checkpoint ----
  const auto ck = xm::Checkpoint::open(model);
  std::vector<std::vector<float>> W(static_cast<size_t>(L));
  {
    std::map<uint32_t, int> fds;
    for (int l = 0; l < L; ++l) {
      const auto &t = ck.at(std::string(xenithium::arch::qwen38::kPrefix) + "layers." + std::to_string(l) + ".mlp.gate.weight");
      if (t.dtype != xm::DType::bf16 || t.dim(0) != E || t.dim(1) != H)
        throw std::runtime_error("lookahead: router " + t.name + " is not bf16 [experts][hidden]");
      if (!fds.count(t.shard)) {
        const int fd = ::open(ck.shards()[t.shard].path.c_str(), O_RDONLY);
        if (fd < 0) throw std::runtime_error("lookahead: cannot open " + ck.shards()[t.shard].path);
        fds[t.shard] = fd;
      }
      std::vector<uint16_t> raw(static_cast<size_t>(E) * H);
      pread_all(fds[t.shard], raw.data(), raw.size() * 2, t.offset);
      W[static_cast<size_t>(l)].resize(raw.size());
      for (size_t i = 0; i < raw.size(); ++i) W[static_cast<size_t>(l)][i] = bf16f(raw[i]);
    }
    for (const auto &p : fds) ::close(p.second);
  }

  // ---- cold sets per hot count (expert_store.cpp's TP placement) and each layer's cold rank order ----
  std::vector<double> rank;
  {
    std::FILE *rf = std::fopen(hot_rank.c_str(), "rb");
    uint32_t mg = 0;
    int32_t dims[2];
    if (!rf || !read_n(rf, &mg, 4) || mg != 0x31435258u || !read_n(rf, dims, 8) || dims[0] != L || dims[1] != E) {
      std::fprintf(stderr, "lookahead: %s is not an XRC1 file for %d x %d\n", hot_rank.c_str(), L, E);
      return 2;
    }
    rank.resize(static_cast<size_t>(L) * E);
    if (!read_n(rf, rank.data(), rank.size() * 8)) throw std::runtime_error("lookahead: short XRC1 file");
    std::fclose(rf);
  }
  std::vector<std::pair<int, int>> order;
  for (int l = 0; l < L; ++l)
    for (int e = 0; e < E; ++e) order.emplace_back(l, e);
  std::stable_sort(order.begin(), order.end(), [&](const auto &a, const auto &b) {
    return rank[static_cast<size_t>(a.first) * E + a.second] > rank[static_cast<size_t>(b.first) * E + b.second];
  });
  const size_t NH = hots.size();
  std::vector<std::vector<char>> cold(NH, std::vector<char>(static_cast<size_t>(L) * E, 1));
  std::vector<std::vector<std::vector<int>>> cold_order(NH, std::vector<std::vector<int>>(static_cast<size_t>(L)));  // most routed first
  for (size_t hi = 0; hi < NH; ++hi) {
    for (int64_t j = 0; j < hots[hi] && j < static_cast<int64_t>(order.size()); ++j)
      cold[hi][static_cast<size_t>(order[static_cast<size_t>(j)].first) * E + order[static_cast<size_t>(j)].second] = 0;
    for (const auto &p : order)
      if (cold[hi][static_cast<size_t>(p.first) * E + p.second]) cold_order[hi][static_cast<size_t>(p.first)].push_back(p.second);
    int64_t nc = 0;
    for (char c : cold[hi]) nc += c;
    std::printf("hot %d per card: %lld cold experts (%.1f per layer)\n", hots[hi], static_cast<long long>(nc), static_cast<double>(nc) / L);
  }

  // ---- score every call ----
  static const char *pred_names[5] = {"self", "next", "next2", "attn", "static"};
  constexpr int NP = 5;
  // tallies[file][pred][k][hot]; file index NF = all files
  std::vector<std::vector<std::vector<std::vector<Tally>>>> tally;
  auto tal = [&](int file, int p, size_t ki, size_t hi) -> Tally & {
    while (static_cast<int>(tally.size()) <= file)
      tally.emplace_back(NP, std::vector<std::vector<Tally>>(ks.size(), std::vector<Tally>(NH)));
    return tally[static_cast<size_t>(file)][static_cast<size_t>(p)][ki][hi];
  };
  int64_t calls = 0, first_calls = 0, self_rows = 0, self_rows_equal = 0;
  double rows_all = 0;
  std::vector<int> calls_per_file;
  for (;;) {
    int32_t rec[4];
    if (!read_n(cf, rec, 16)) break;
    const int file = rec[0], m = rec[3];
    const size_t xs = static_cast<size_t>(m) * H;
    std::vector<std::vector<uint16_t>> xa(static_cast<size_t>(L), std::vector<uint16_t>(xs)), xmv(static_cast<size_t>(L), std::vector<uint16_t>(xs));
    std::vector<std::vector<int32_t>> ids(static_cast<size_t>(L), std::vector<int32_t>(static_cast<size_t>(m) * TK));
    for (int l = 0; l < L; ++l)
      if (!read_n(cf, xa[static_cast<size_t>(l)].data(), xs * 2) || !read_n(cf, xmv[static_cast<size_t>(l)].data(), xs * 2) ||
          !read_n(cf, ids[static_cast<size_t>(l)].data(), ids[static_cast<size_t>(l)].size() * 4))
        throw std::runtime_error("lookahead: truncated capture");
    ++calls;
    rows_all += m;
    first_calls += rec[2] == 0;
    if (static_cast<int>(calls_per_file.size()) <= file) calls_per_file.resize(static_cast<size_t>(file) + 1, 0);
    ++calls_per_file[static_cast<size_t>(file)];
    std::vector<int32_t> top;
    std::vector<int> rows_of(static_cast<size_t>(E));
    std::vector<char> routed(static_cast<size_t>(E)), pred(static_cast<size_t>(E));
    for (int t = 0; t < L; ++t) {
      // the routed set at t and its rows per expert
      std::fill(rows_of.begin(), rows_of.end(), 0);
      for (int32_t e : ids[static_cast<size_t>(t)]) {
        if (e < 0 || e >= E) throw std::runtime_error("lookahead: routed id out of range");
        ++rows_of[static_cast<size_t>(e)];
      }
      for (int e = 0; e < E; ++e) routed[static_cast<size_t>(e)] = rows_of[static_cast<size_t>(e)] > 0;
      for (int p = 0; p < NP; ++p) {
        const uint16_t *x = nullptr;
        if (p == 0) x = xmv[static_cast<size_t>(t)].data();
        if (p == 1 && t >= 1) x = xmv[static_cast<size_t>(t - 1)].data();
        if (p == 2 && t >= 2) x = xmv[static_cast<size_t>(t - 2)].data();
        if (p == 3) x = xa[static_cast<size_t>(t)].data();
        if (p != 4 && !x) continue;
        if (p != 4) top_rows(W[static_cast<size_t>(t)], E, H, x, m, Kmax, threads, top);
        if (p == 0) {  // self-check: the host's top-k set per row == the device's
          for (int r = 0; r < m; ++r) {
            std::vector<int32_t> a(top.begin() + static_cast<ptrdiff_t>(r) * Kmax, top.begin() + static_cast<ptrdiff_t>(r) * Kmax + TK);
            std::vector<int32_t> b(ids[static_cast<size_t>(t)].begin() + static_cast<ptrdiff_t>(r) * TK, ids[static_cast<size_t>(t)].begin() + static_cast<ptrdiff_t>(r + 1) * TK);
            std::sort(a.begin(), a.end());
            std::sort(b.begin(), b.end());
            ++self_rows;
            self_rows_equal += a == b;
          }
        }
        for (size_t ki = 0; ki < ks.size(); ++ki) {
          const int K = ks[ki];
          for (size_t hi = 0; hi < NH; ++hi) {
            std::fill(pred.begin(), pred.end(), 0);
            if (p == 4) {
              const auto &co = cold_order[hi][static_cast<size_t>(t)];
              for (size_t j = 0; j < co.size() && static_cast<int>(j) < 2 * K; ++j) pred[static_cast<size_t>(co[j])] = 1;
            } else {
              for (int r = 0; r < m; ++r)
                for (int j = 0; j < K; ++j) pred[static_cast<size_t>(top[static_cast<size_t>(r) * Kmax + j])] = 1;
            }
            for (int f : {file, -1, rec[2] == 0 ? -2 : -3}) {
              if (f == -3) continue;
              Tally &T = tal(f == -1 ? 1000 : f == -2 ? 1001 : f, p, ki, hi);
              T.layers += 1;
              for (int e = 0; e < E; ++e) {
                const bool c = cold[hi][static_cast<size_t>(t) * E + e], rt = routed[static_cast<size_t>(e)], pr = pred[static_cast<size_t>(e)];
                T.routed_all += rt;
                T.pred_all += pr;
                T.both_all += rt && pr;
                if (!c) continue;
                T.routed_cold += rt;
                T.pred_cold += pr;
                T.both_cold += rt && pr;
                T.cold_rows += rows_of[static_cast<size_t>(e)];
                if (pr) T.cold_rows_hit += rows_of[static_cast<size_t>(e)];
              }
            }
          }
        }
      }
    }
    std::fprintf(stderr, "lookahead: call %lld (file %d, %d rows at %d) scored\n", static_cast<long long>(calls), file, m, rec[2]);
  }
  std::fclose(cf);
  if (calls == 0) {
    std::fprintf(stderr, "lookahead: no calls in the capture\n");
    return 1;
  }
  std::printf("capture: %lld prefill calls, %.0f rows (%.1f per call), %d layers x %d experts, top %d; blob %.3f MB per card\n", static_cast<long long>(calls), rows_all,
              rows_all / static_cast<double>(calls), L, E, TK, static_cast<double>(stride) / 1e6);
  std::printf("self-check: host router top-%d set == device ids in %lld / %lld rows (%.3f %%)\n", TK, static_cast<long long>(self_rows_equal), static_cast<long long>(self_rows),
              100.0 * static_cast<double>(self_rows_equal) / static_cast<double>(std::max<int64_t>(1, self_rows)));

  auto table = [&](int fi, const std::string &title, double ncalls) {
    std::printf("\n== %s (%.0f calls)\n", title.c_str(), ncalls);
    for (size_t hi = 0; hi < NH; ++hi) {
      std::printf("\nhot %d per card\n", hots[hi]);
      std::printf("| predictor | K | routed cold per layer | predicted cold per layer | cold recall (%%) | cold precision (%%) | cold routes covered (%%) | "
                  "copied per call per card (GB) | routed cold per call per card (GB) | missed per call per card (GB) | all-expert recall (%%) | predicted per layer |\n");
      std::printf("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n");
      for (int p = 0; p < NP; ++p)
        for (size_t ki = 0; ki < ks.size(); ++ki) {
          const Tally &T = tally[static_cast<size_t>(fi)][static_cast<size_t>(p)][ki][hi];
          if (T.layers == 0) continue;
          if (p == 0 && ks[ki] != TK) continue;  // the self-check only at the model's top-k
          const double gb = static_cast<double>(stride) / 1e9;
          std::printf("| %s | %s%d | %.2f | %.2f | %.1f | %.1f | %.1f | %.3f | %.3f | %.3f | %.1f | %.1f |\n", pred_names[p], p == 4 ? "2x" : "", ks[ki], T.routed_cold / T.layers,
                      T.pred_cold / T.layers, 100.0 * T.both_cold / std::max(1.0, T.routed_cold), 100.0 * T.both_cold / std::max(1.0, T.pred_cold),
                      100.0 * T.cold_rows_hit / std::max(1.0, T.cold_rows), T.pred_cold * gb / ncalls, T.routed_cold * gb / ncalls,
                      (T.routed_cold - T.both_cold) * gb / ncalls, 100.0 * T.both_all / std::max(1.0, T.routed_all), T.pred_all / T.layers);
        }
    }
  };
  for (size_t f = 0; f < calls_per_file.size(); ++f)
    if (calls_per_file[f] > 0)
      table(static_cast<int>(f), "file " + std::to_string(f) + (f < labels.size() ? " (" + labels[f] + ")" : ""), calls_per_file[f]);
  table(1000, "all files", static_cast<double>(calls));
  if (first_calls > 0) table(1001, "first call of each sequence (rows from position 0: a 512-token prompt's one call)", static_cast<double>(first_calls));
  std::printf("\nnote: 'next' scores layers 1..%d, 'next2' 2..%d, the others 0..%d; layer counts differ per predictor (per-layer columns are means over the scored layers).\n", L - 1, L - 1, L - 1);
  return 0;
}
