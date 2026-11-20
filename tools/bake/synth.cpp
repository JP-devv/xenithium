// xenithium-bake synth: AutoRound on a real layer's bf16 experts with synthetic inputs.
// xenithium-bake synth --src BF16_DIR [--layer 0] [--experts 16] [--topk 4] [--samples 24]
// [--seqlen 512] [--eval 8] [--iters 200] [--gpus N] [--no-aq 1] [--seed 7]
// x ~ N(0, 1) with log-normal per-channel scales (outlier channels, as after an RMS norm), random
// top-k routing with softmax weights, target T = the bf16 experts' exact routed sum on the
// unquantised x (what the BF16 model computes). Prints the eval loss of RTN and of the tuned
// parameters on held-out samples: the gate is tuned < RTN. (GPU vs the CPU reference: the test.)
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <random>
#include <stdexcept>
#include <thread>
#include <unistd.h>

#include "bake.hpp"
#include "xenithium/model/checkpoint.hpp"
#include "xenithium/quant/autoround.hpp"

namespace xenithium::bake {

namespace q = xenithium::quant;
namespace m = xenithium::model;

void read_tensor_range(const m::Checkpoint &ck, const m::TensorInfo &t, uint64_t off, uint64_t bytes, void *dst) {
  if (off + bytes > t.bytes) throw std::runtime_error(t.name + ": range out of the tensor");
  const std::string &path = ck.shards()[t.shard].path;
  const int fd = ::open(path.c_str(), O_RDONLY);
  if (fd < 0) throw std::runtime_error("cannot open " + path);
  uint64_t done = 0;
  while (done < bytes) {
    const ssize_t n = ::pread(fd, static_cast<char *>(dst) + done, bytes - done, static_cast<off_t>(t.offset + off + done));
    if (n <= 0) {
      ::close(fd);
      throw std::runtime_error("short read in " + path);
    }
    done += static_cast<uint64_t>(n);
  }
  ::close(fd);
}

std::vector<std::unique_ptr<q::ArWorker>> make_workers(const Args &a) {
  if (a.has("cpu")) throw std::runtime_error("--cpu was removed: tuning runs on the GPUs only (the host cores feed them)");
  std::vector<std::unique_ptr<q::ArWorker>> w;
  const int have = q::ar_gpu_count();
  const int gpus = static_cast<int>(a.num("gpus", have));
  if (gpus > have) throw std::runtime_error("--gpus " + std::to_string(gpus) + " but " + std::to_string(have) + " GPU(s) found");
  if (gpus < 1) throw std::runtime_error("no GPU to tune on (--gpus " + std::to_string(gpus) + ", " + std::to_string(have) + " found)");
  for (int i = 0; i < gpus; ++i) w.push_back(q::make_gpu_worker(i));
  return w;
}

int cmd_synth(const Args &a) {
  const std::string src = a.str("src", home() + "/models/Qwen3.8-Flash-Next");
  const int layer = static_cast<int>(a.num("layer", 0));
  q::ArShape s;
  s.experts = static_cast<int>(a.num("experts", 16));
  s.topk = static_cast<int>(a.num("topk", 4));
  const int samples = static_cast<int>(a.num("samples", 24)), seqlen = static_cast<int>(a.num("seqlen", 512));
  const int eval = static_cast<int>(a.num("eval", 8));
  q::ArConfig cfg;
  cfg.iters = static_cast<int>(a.num("iters", 200));
  cfg.act_quant = !a.has("no-aq");
  cfg.seed = static_cast<uint64_t>(a.num("seed", 7));
  const int H = s.hidden, I = s.inter;

  const auto ck = m::Checkpoint::open(src);
  const std::string P = "model.language_model.layers." + std::to_string(layer) + ".mlp.experts.";
  const auto &tgu = ck.at(P + "gate_up_proj"), &tdn = ck.at(P + "down_proj");
  if (tgu.dim(0) < s.experts || tgu.dim(1) != 2 * I || tgu.dim(2) != H || tdn.dim(1) != H || tdn.dim(2) != I)
    throw std::runtime_error("unexpected expert tensor shapes");
  std::vector<uint16_t> wgu(static_cast<size_t>(s.experts) * 2 * I * H), wdn(static_cast<size_t>(s.experts) * H * I);
  read_tensor_range(ck, tgu, 0, wgu.size() * 2, wgu.data());
  read_tensor_range(ck, tdn, 0, wdn.size() * 2, wdn.data());
  const q::ExpertSource es{wgu.data(), wdn.data()};

  // data
  const int64_t R = static_cast<int64_t>(samples) * seqlen;
  std::mt19937_64 rng(cfg.seed);
  std::normal_distribution<float> nd(0.f, 1.f);
  std::vector<float> chs(static_cast<size_t>(H));
  for (auto &c : chs) c = std::exp(0.6f * nd(rng));
  std::vector<uint16_t> x(static_cast<size_t>(R) * H), t(static_cast<size_t>(R) * H);
  std::vector<int32_t> ids(static_cast<size_t>(R) * s.topk);
  std::vector<float> w(ids.size());
  for (int64_t r = 0; r < R; ++r) {
    for (int c = 0; c < H; ++c) x[static_cast<size_t>(r) * H + c] = q::ar_bf16_bits(nd(rng) * chs[static_cast<size_t>(c)]);
    std::vector<std::pair<float, int>> lg(static_cast<size_t>(s.experts));
    for (int e = 0; e < s.experts; ++e) lg[static_cast<size_t>(e)] = {2.f * nd(rng), e};
    std::partial_sort(lg.begin(), lg.begin() + s.topk, lg.end(), [](auto &p, auto &q2) { return p.first > q2.first; });
    float mx = lg[0].first, sum = 0.f;
    for (int k = 0; k < s.topk; ++k) sum += std::exp(lg[static_cast<size_t>(k)].first - mx);
    for (int k = 0; k < s.topk; ++k) {
      ids[static_cast<size_t>(r) * s.topk + k] = lg[static_cast<size_t>(k)].second;
      w[static_cast<size_t>(r) * s.topk + k] = std::exp(lg[static_cast<size_t>(k)].first - mx) / sum;
    }
  }
  {  // T: the bf16 experts on the unquantised x (h rounded to bf16 as the BF16 model does)
    const int th = static_cast<int>(std::thread::hardware_concurrency());
    std::vector<std::thread> pool;
    for (int ti = 0; ti < th; ++ti)
      pool.emplace_back([&, ti] {
        std::vector<float> xr(static_cast<size_t>(H)), h(static_cast<size_t>(I)), acc(static_cast<size_t>(H)), wrow(static_cast<size_t>(H));
        for (int64_t r = ti; r < R; r += th) {
          for (int c = 0; c < H; ++c) xr[static_cast<size_t>(c)] = q::ar_bf16(x[static_cast<size_t>(r) * H + c]);
          std::fill(acc.begin(), acc.end(), 0.f);
          for (int k = 0; k < s.topk; ++k) {
            const size_t e = static_cast<size_t>(ids[static_cast<size_t>(r) * s.topk + k]);
            for (int j = 0; j < I; ++j) {
              double g = 0, u = 0;
              const uint16_t *gr = wgu.data() + (e * 2 * I + j) * H, *ur = wgu.data() + (e * 2 * I + I + j) * H;
              for (int c = 0; c < H; ++c) g += xr[static_cast<size_t>(c)] * q::ar_bf16(gr[c]), u += xr[static_cast<size_t>(c)] * q::ar_bf16(ur[c]);
              const float gf = q::ar_round_bf16(static_cast<float>(g)), uf = q::ar_round_bf16(static_cast<float>(u));
              h[static_cast<size_t>(j)] = q::ar_round_bf16(gf / (1.f + std::exp(-gf)) * uf);
            }
            const float wk = w[static_cast<size_t>(r) * s.topk + k];
            for (int n = 0; n < H; ++n) {
              double y = 0;
              const uint16_t *dr = wdn.data() + (e * H + n) * I;
              for (int j = 0; j < I; ++j) y += h[static_cast<size_t>(j)] * q::ar_bf16(dr[j]);
              acc[static_cast<size_t>(n)] += wk * static_cast<float>(y);
            }
          }
          for (int n = 0; n < H; ++n) t[static_cast<size_t>(r) * H + n] = q::ar_bf16_bits(acc[static_cast<size_t>(n)]);
        }
      });
    for (auto &p : pool) p.join();
  }
  q::LayerData d;
  d.samples = samples, d.seqlen = seqlen, d.x = x.data(), d.ids = ids.data(), d.w = w.data(), d.t = t.data();

  auto workers = make_workers(a);
  std::vector<q::ExpertQuant> out;
  const auto st = q::tune_layer(s, cfg, d, es, workers, eval, out);
  for (size_t i = 0; i < st.workers.size(); ++i)
    std::printf("worker %-10s experts %4d routes %9lld\n", st.workers[i].c_str(), st.experts_per_worker[i],
                static_cast<long long>(st.assignments_per_worker[i]));
  std::printf("layer %d, %d experts, top-%d, %d x %d rows (%d eval), iters %d, aq %d\n", layer, s.experts, s.topk, samples, seqlen, eval,
              cfg.iters, cfg.act_quant ? 1 : 0);
  std::printf("eval loss  RTN %.6e  tuned %.6e  (%.2f%%)  best step %d  %.1f s\n", st.rtn_loss, st.tuned_loss,
              100.0 * (st.tuned_loss / st.rtn_loss - 1.0), st.best_iter, st.seconds);
  // code agreement with plain RTN (how much the rounding moved)
  int64_t diff = 0, tot = 0;
  for (int e = 0; e < s.experts; ++e) {
    q::ExpertQuant r;
    q::rtn_expert(s, es, e, r);
    for (size_t i = 0; i < r.gu_q.size(); ++i) diff += r.gu_q[i] != out[static_cast<size_t>(e)].gu_q[i];
    for (size_t i = 0; i < r.dn_q.size(); ++i) diff += r.dn_q[i] != out[static_cast<size_t>(e)].dn_q[i];
    tot += static_cast<int64_t>(r.gu_q.size() + r.dn_q.size());
  }
  std::printf("codes changed vs RTN: %.3f%%\n", 100.0 * static_cast<double>(diff) / static_cast<double>(tot));
  return 0;
}

}  // namespace xenithium:bake
