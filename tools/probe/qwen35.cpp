// xenithium-probe, the qwen35 (Qwen3.8-27B) subcommands: w4load, head35, mlp35.
// w4load: the qwen35 trunk's W4 weight path on both cards.
// Loads one layer's pack-quantized Linears through DenseStore (Store:w4: the compressed-tensors
// words sliced per card, packed into the W4 blocks + fp16 [k/128][n] scales), then for every matrix
// on every card runs w4a16_gemm_m4 (4 rows: 2 of small integers, 2 of real bf16 values) and compares
// with w4a16_ref on a host slice built independently from the raw tensors (slice_rows / col0 applied
// to the checkpoint words directly). Correctness line first, then the timed table per matrix.
// xenithium-probe w4load --model DIR [--layers 0,3] [--iters I] [--threads T]
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "xenithium/arch/qwen35/arch.hpp"
#include "xenithium/arch/qwen35/head.hpp"
#include "xenithium/arch/qwen35/mlp.hpp"
#include "xenithium/kernels/pack.hpp"
#include "xenithium/kernels/residual/rmsnorm.hpp"
#include "xenithium/kernels/gemm/w4a16_gemm.hpp"
#include "xenithium/load/dense_store.hpp"
#include "xenithium/model/arch.hpp"
#include "xenithium/model/bulk_read.hpp"
#include "xenithium/model/checkpoint.hpp"
#include "xenithium/model/config.hpp"
#include "xenithium/rt/fabric.hpp"
#include "qwen35_host.hpp"
#include "warmup.hpp"

namespace {
namespace m = xenithium::model;
namespace k = xenithium::kernels;
using bf16 = sycl::ext::oneapi::bfloat16;

double ev_ms(const sycl::event &e) {
  const auto s = e.get_profiling_info<sycl::info::event_profiling::command_start>();
  const auto t = e.get_profiling_info<sycl::info::event_profiling::command_end>();
  return static_cast<double>(t - s) / 1e6;
}

float f16_to_float_impl(uint16_t h) {
  const uint32_t sign = (h >> 15) & 1u, exp = (h >> 10) & 31u, man = h & 1023u;
  float v;
  if (exp == 0) v = std::ldexp(static_cast<float>(man), -24);
  else if (exp == 31) v = man ? NAN : INFINITY;
  else v = std::ldexp(static_cast<float>(man | 1024u), static_cast<int>(exp) - 25);
  return sign ? -v : v;
}

}  // namespace

// A card's slice of one pack-quantized spec rebuilt on the host from the raw tensors, independently of
// DenseStore (slice_rows / col0 on the words and the fp16 scales), in the GPTQ orientation w4a16_ref takes:
// qweight [k/8][n] words, scales fp32 [k/128][n]. srcs: the spec's (packed, scale) pairs, appends after.
HostW4 host_w4_slice(const m::Checkpoint &ck, const m::DenseSpec &sp, const m::BulkBuffer &buf, const std::vector<const m::TensorInfo *> &srcs, int c,
                     int C, int n, int kk) {
  const bool gptq = sp.w4_layout == m::DenseSpec::W4Layout::gptq;
  const auto &tp = *srcs[0];
  auto n_of = [&](const m::TensorInfo &t) { return gptq ? t.dim(1) : t.dim(0); };
  const int64_t k_all = (gptq ? tp.dim(0) : tp.dim(1)) * 8;
  std::vector<std::pair<int64_t, int64_t>> rows;
  std::vector<size_t> row_src;
  int64_t col0 = 0;
  if (sp.split == m::DenseSpec::Split::rows) {
    for (const auto &r : xenithium::load::slice_rows(n_of(tp), sp.sections, c, C)) rows.push_back(r), row_src.push_back(0);
    for (size_t a = 0; a < sp.append.size(); ++a)
      for (const auto &r : xenithium::load::slice_rows(n_of(*srcs[2 + 2 * a]), {}, c, C)) rows.push_back(r), row_src.push_back(a + 1);
  } else if (sp.split == m::DenseSpec::Split::cols) {
    rows.push_back({0, n_of(tp)}), row_src.push_back(0);
    col0 = k_all / C * c;
  } else {
    rows.push_back({0, n_of(tp)}), row_src.push_back(0);
  }
  HostW4 h;
  h.scales.assign(static_cast<size_t>(kk / 128) * n, 0.f);
  h.gptq.assign(static_cast<size_t>(kk / 8) * n, 0x88888888u);
  if (gptq) {
    // qweight [k/8][n_src]: the slice's columns of word rows [col0/8, (col0 + kk)/8); scales [k/128][n_src] likewise.
    for (size_t i = 0; i < rows.size(); ++i) {
      const auto &tq = *srcs[2 * row_src[i]];
      const int64_t n_src = tq.dim(1);
      const uint32_t *pw = buf.as<uint32_t>(tq);
      const uint16_t *ps = buf.as<uint16_t>(*srcs[2 * row_src[i] + 1]);
      const int64_t len = rows[i].second - rows[i].first;
      for (int64_t wr = 0; wr < kk / 8; ++wr)
        std::memcpy(&h.gptq[static_cast<size_t>(wr) * n + h.rows], pw + static_cast<size_t>(col0 / 8 + wr) * n_src + rows[i].first, static_cast<size_t>(len) * 4);
      for (int g = 0; g < kk / 128; ++g)
        for (int64_t j = 0; j < len; ++j)
          h.scales[static_cast<size_t>(g) * n + h.rows + j] = f16_to_float(ps[static_cast<size_t>(col0 / 128 + g) * n_src + rows[i].first + j]);
      h.rows += len;
    }
    (void)ck;
    return h;
  }
  std::vector<uint32_t> ctw(static_cast<size_t>(n) * (kk / 8), 0x88888888u);
  for (size_t i = 0; i < rows.size(); ++i) {
    const uint32_t *pw = buf.as<uint32_t>(*srcs[2 * row_src[i]]);
    const uint16_t *ps = buf.as<uint16_t>(*srcs[2 * row_src[i] + 1]);
    for (int64_t r = rows[i].first; r < rows[i].second; ++r, ++h.rows) {
      std::memcpy(&ctw[static_cast<size_t>(h.rows) * (kk / 8)], pw + static_cast<size_t>(r) * (k_all / 8) + col0 / 8, static_cast<size_t>(kk / 8) * 4);
      for (int g = 0; g < kk / 128; ++g) h.scales[static_cast<size_t>(g) * n + h.rows] = f16_to_float(ps[static_cast<size_t>(r) * (k_all / 128) + col0 / 128 + g]);
    }
  }
  k::ct_to_gptq_qweight(ctw.data(), n, kk, h.gptq.data());
  (void)ck;
  return h;
}
// The (packed, scale) tensor pairs of a w4 spec in either layout, appends after, as bulk_read takes them.
std::vector<const m::TensorInfo *> w4_sources(const m::Checkpoint &ck, const m::DenseSpec &sp) {
  std::vector<const m::TensorInfo *> srcs = {&ck.at(sp.name), &ck.at(xenithium::arch::qwen35::scale_name_of(sp.name))};
  for (const auto &an : sp.append) {
    srcs.push_back(&ck.at(an));
    srcs.push_back(&ck.at(xenithium::arch::qwen35::scale_name_of(an)));
  }
  return srcs;
}

std::vector<float> deq_w4(const HostW4 &h, int n, int kk) {
  std::vector<float> d(static_cast<size_t>(n) * kk);
  for (int col = 0; col < n; ++col)
    for (int k_ = 0; k_ < kk; ++k_) {
      const int q = static_cast<int>((h.gptq[static_cast<size_t>(k_ / 8) * n + col] >> (4 * (k_ % 8))) & 15u);
      d[static_cast<size_t>(col) * kk + k_] = static_cast<float>(q - 8) * h.scales[static_cast<size_t>(k_ / 128) * n + col];
    }
  return d;
}

float f16_to_float(uint16_t h) { return f16_to_float_impl(h); }

// : the decode W4 GEMMs on every layer's real weights, card 0 and card 1 at once.
// Per role: alone (one copy, cache-warm when it fits the 24 MiB L2), rotated over its distinct copies (a DRAM number),
// and inside the step's GEMM order with nothing between the launches (the chain). 1 row and 4 rows.
void w4_chain_times(xenithium::Fabric &fab, const xenithium::load::DenseStore &dense, const std::vector<m::DenseSpec> &plan, int iters) {
  const char *roles[] = {"gdn in_proj_qkv (+z)", "gdn out_proj", "attn q|k|v", "attn o_proj", "mlp gate|up", "mlp down"};
  auto role_of = [](const std::string &n) {
    if (n.find("in_proj_qkv") != std::string::npos) return 0;
    if (n.find("linear_attn.out_proj") != std::string::npos) return 1;
    if (n.find("q_proj") != std::string::npos) return 2;
    if (n.find("o_proj") != std::string::npos) return 3;
    if (n.find("gate_proj") != std::string::npos) return 4;
    if (n.find("down_proj") != std::string::npos) return 5;
    return -1;
  };
  using W = xenithium::load::W4Weights;
  const int C = 2;
  // per card: [layer] -> the layer's matrices in step order (mixer in, mixer out, gate|up, down)
  std::vector<std::vector<std::vector<std::pair<int, W>>>> step(C, std::vector<std::vector<std::pair<int, W>>>(64));
  for (const auto &sp : plan) {
    const int r = role_of(sp.name);
    if (r < 0) throw std::runtime_error("w4load: unknown role " + sp.name);
    for (int c = 0; c < C; ++c) step[c][sp.layer].push_back({r, dense.at(c, sp.name).w4()});
  }
  for (int c = 0; c < C; ++c)
    for (auto &l : step[c]) std::sort(l.begin(), l.end(), [](const auto &a, const auto &b) { return a.first < b.first; });
  int kmax = 0, nmax = 0;
  for (auto &l : step[0])
    for (auto &rw : l) kmax = std::max(kmax, rw.second.k), nmax = std::max(nmax, rw.second.n);
  std::vector<bf16 *> dx(C);
  std::vector<float *> dy(C);
  std::mt19937 rng(11);
  std::normal_distribution<float> nd(0.f, 1.f);
  std::vector<bf16> xb(8ull * kmax);
  for (auto &v : xb) v = bf16(nd(rng));
  for (int c = 0; c < C; ++c) {
    dx[c] = static_cast<bf16 *>(fab.alloc(c, xb.size() * 2));
    dy[c] = static_cast<float *>(fab.alloc(c, 8ull * nmax * 4));
    fab.compute(c).memcpy(dx[c], xb.data(), xb.size() * 2).wait();
  }
  auto launch = [&](int c, int rows, const W &w) { return k::w4a16_gemm_rows(fab.compute(c), rows, dx[c], w.w, w.scales, dy[c], w.n, w.k); };
  auto med = [](std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v.empty() ? 0.0 : v[v.size() / 2];
  };
  xenithium::probe::warm_for([&] {
    for (int c = 0; c < C; ++c)
      for (auto &rw : step[c][0]) launch(c, 1, rw.second);
    for (int c = 0; c < C; ++c) fab.compute(c).wait();
  });
  const int reps = std::max(3, iters / 10);
  for (int rows : {1, 4}) {
    std::printf("\nw4 chain, %d row(s), card 0 (card 1 runs the chain at the same time), us per launch (median), floor = bytes / 604 GB/s:\n", rows);
    std::printf("| role | n | k | copies | floor | alone | rotated | rotated x floor | chain | chain x floor |\n| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n");
    std::vector<std::vector<double>> chain(6);
    double wall = 1e30;
    for (int rep = 0; rep < reps + 1; ++rep) {
      std::vector<std::pair<int, sycl::event>> ev;
      const auto t0 = std::chrono::steady_clock::now();
      for (int l = 0; l < 64; ++l)
        for (int c = 0; c < C; ++c)
          for (auto &rw : step[c][l]) {
            auto e = launch(c, rows, rw.second);
            if (c == 0) ev.push_back({rw.first, e});
          }
      for (int c = 0; c < C; ++c) fab.compute(c).wait();
      const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
      if (rep == 0) continue;
      wall = std::min(wall, ms);
      for (auto &re : ev) chain[re.first].push_back(ev_ms(re.second) * 1e3);
    }
    double fl_sum = 0, ch_sum = 0;
    for (int r = 0; r < 6; ++r) {
      std::vector<W> cp;
      for (auto &l : step[0])
        for (auto &rw : l)
          if (rw.first == r) cp.push_back(rw.second);
      if (cp.empty()) continue;
      std::vector<double> alone, rot;
      for (int i = 0; i < iters; ++i) {
        auto e = launch(0, rows, cp[0]);
        e.wait();
        alone.push_back(ev_ms(e) * 1e3);
      }
      for (int rep = 0; rep < reps; ++rep) {
        std::vector<sycl::event> ev;
        for (auto &w : cp) ev.push_back(launch(0, rows, w));
        fab.compute(0).wait();
        for (auto &e : ev) rot.push_back(ev_ms(e) * 1e3);
      }
      const double bytes = static_cast<double>(cp[0].n) * cp[0].k / 2 + static_cast<double>(cp[0].k / 128) * cp[0].n * 2;
      const double fl = bytes / 604e3;
      const double ch = med(chain[r]);
      fl_sum += fl * static_cast<double>(cp.size());
      ch_sum += ch * static_cast<double>(cp.size());
      std::printf("| %s | %d | %d | %zu | %.1f | %.1f | %.1f | %.3f | %.1f | %.3f |\n", roles[r], cp[0].n, cp[0].k, cp.size(), fl, med(alone), med(rot), med(rot) / fl, ch,
                  ch / fl);
    }
    std::printf("w4 chain %d row(s): floor %.3f ms, sum of chain medians %.3f ms (%.3f x), best wall (both cards) %.3f ms\n", rows, fl_sum / 1e3, ch_sum / 1e3,
                ch_sum / fl_sum, wall);
  }
  for (int c = 0; c < C; ++c) fab.release(dx[c]), fab.release(dy[c]);
}

int cmd_w4load(const std::string &model, const std::string &layers_arg, int iters, int threads) {
  const auto mc = m::ModelConfig::load(model);
  const auto arch = m::open_arch(mc);
  if (arch->id() != "qwen35") {
    std::fprintf(stderr, "w4load: %s is arch %s, not qwen35\n", model.c_str(), std::string(arch->id()).c_str());
    return 2;
  }
  const m::Checkpoint ck = m::Checkpoint::open(model);
  std::vector<int> layers;
  // "all": every layer loads; layers 0 and 3 are checked and timed alone, then
  // every matrix role is timed rotated over its 16-64 distinct copies (cache-free) and as the step's GEMM chain.
  const bool all = layers_arg == "all";
  if (all)
    for (int l = 0; l < 64; ++l) layers.push_back(l);
  else {
    std::stringstream ss(layers_arg);
    std::string t;
    while (std::getline(ss, t, ','))
      if (!t.empty()) layers.push_back(std::stoi(t));
  }
  std::vector<m::DenseSpec> plan;
  for (const auto &sp : arch->dense_plan(ck, m::DeployParams{}))
    if (sp.store == m::DenseSpec::Store::w4 && std::find(layers.begin(), layers.end(), sp.layer) != layers.end()) plan.push_back(sp);
  if (plan.empty()) {
    std::fprintf(stderr, "w4load: no w4 matrices in layers %s\n", layers_arg.c_str());
    return 2;
  }
  const auto loader = xenithium::l0::Loader::open();
  xenithium::Fabric fab(loader, {0, 1});
  const int C = 2;
  xenithium::load::DenseStore dense(fab);
  dense.load(ck, plan, threads);
  const auto &st = dense.stats();
  double gib = 0;
  for (const auto &kv : st.card_bytes[0]) gib += static_cast<double>(kv.second);
  std::printf("w4load: %zu w4 matrices of layers %s loaded in %.2f s (read wait %.2f, pack %.2f, h2d wait %.2f; %.3f GiB per card)\n",
              plan.size(), layers_arg.c_str(), st.total_s, st.read_wait_s, st.pack_s, st.h2d_wait_s, gib / 1073741824.0);

  // Inputs: 4 rows x k; rows 0-1 small integers (group sums exact in fp32), rows 2-3 ~N(0, 1) bf16.
  std::mt19937 rng(7);
  std::normal_distribution<float> nd(0.f, 1.f);
  int fails = 0;
  double worst_rel = 0;
  std::printf("%-44s %4s %6s %6s | %10s %10s | %8s %7s %7s\n", "matrix (card slice)", "card", "n", "k", "max|err|", "rel", "us", "GB/s", "x floor");
  for (const auto &sp : plan) {
    if (all && sp.layer != 0 && sp.layer != 3) continue;
    const auto srcs = w4_sources(ck, sp);
    const auto buf = m::bulk_read(ck, srcs);
    for (int c = 0; c < C; ++c) {
      const auto &dt = dense.at(c, sp.name);
      const auto w = dt.w4();
      const int n = w.n, kk = w.k;
      const HostW4 hs = host_w4_slice(ck, sp, buf, srcs, c, C, n, kk);
      if (hs.rows != dt.rows) {
        std::printf("  ! %s card %d: host slice has %lld rows, the store %lld\n", sp.name.c_str(), c, static_cast<long long>(hs.rows), static_cast<long long>(dt.rows));
        ++fails;
        continue;
      }
      const auto &gptq = hs.gptq;
      const auto &scf = hs.scales;
      std::vector<float> xf(4ull * kk);
      std::vector<bf16> xb(xf.size());
      for (size_t i = 0; i < xf.size(); ++i) {
        const float v = i < 2ull * kk ? static_cast<float>(static_cast<int>(rng() % 9) - 4) : nd(rng);
        xb[i] = bf16(v);
        xf[i] = static_cast<float>(xb[i]);
      }
      std::vector<double> want(4ull * n);
      k::w4a16_ref(xf.data(), gptq.data(), scf.data(), 4, n, kk, want.data());

      auto &q = fab.compute(c);
      auto *dx = static_cast<bf16 *>(fab.alloc(c, xb.size() * 2));
      auto *dy = static_cast<float *>(fab.alloc(c, 4ull * n * 4));
      q.memcpy(dx, xb.data(), xb.size() * 2).wait();
      auto launch = [&] { return k::w4a16_gemm_m4(q, dx, w.w, w.scales, dy, n, kk); };
      launch().wait_and_throw();
      std::vector<float> got(4ull * n);
      q.memcpy(got.data(), dy, got.size() * 4).wait();
      double max_err = 0, max_abs = 0;
      for (size_t i = 0; i < got.size(); ++i) {
        max_err = std::max(max_err, std::fabs(static_cast<double>(got[i]) - want[i]));
        max_abs = std::max(max_abs, std::fabs(want[i]));
      }
      const double rel = max_abs > 0 ? max_err / max_abs : max_err;
      worst_rel = std::max(worst_rel, rel);
      // Run-to-run bit identity.
      launch().wait_and_throw();
      std::vector<float> again(got.size());
      q.memcpy(again.data(), dy, again.size() * 4).wait();
      const bool same = std::memcmp(got.data(), again.data(), got.size() * 4) == 0;
      if (rel > 1e-5 || !same) ++fails;

      xenithium::probe::warm_for([&] { launch().wait(); });
      std::vector<double> ms;
      for (int i = 0; i < iters; ++i) {
        auto e = launch();
        e.wait_and_throw();
        ms.push_back(ev_ms(e));
      }
      std::sort(ms.begin(), ms.end());
      const double med = ms[ms.size() / 2];
      const double bytes = static_cast<double>(n) * kk / 2 + static_cast<double>(kk / 128) * n * 2;
      std::string nm = sp.name.substr(std::strlen(xenithium::arch::qwen35::kPrefix));
      nm = nm.substr(0, nm.rfind('.'));
      if (!sp.append.empty()) nm += " (+" + std::to_string(sp.append.size()) + ")";
      std::printf("%-44s %4d %6d %6d | %10.3e %10.2e | %8.1f %7.1f %7.2f%s\n", nm.c_str(), c, n, kk, max_err, rel, med * 1e3, bytes / (med * 1e6),
                  med / (bytes / 604e6), same ? "" : "  NOT run-to-run identical");
      fab.release(dx);
      fab.release(dy);
    }
  }
  if (all) w4_chain_times(fab, dense, plan, iters);
  std::printf("w4load: %s - %zu matrices x %d cards vs w4a16_ref on independent host slices, worst rel %.2e (gate 1e-5), %d failure(s)\n",
              fails ? "FAIL" : "PASS", all ? plan.size() * 2 / 64 : plan.size(), C, worst_rel, fails);
  return fails ? 1 : 0;
}

// ---- probe head35: the step's ends on real weights, both cards --------------------------------
// embed (pinned host table) -> final (1 + w) norm -> lm_head (int8 W8A16, vocab-parallel) -> greedy pick across the
// cards. Checks: embed exact; norm vs rms_norm_ref (2 ulps of max(|ref|, rms)); lm_head logits vs the double reference
// on the same int8 rows (sampled columns); pick == host argmax over both halves; cards identical; then per-stage times.
// xenithium-probe head35 --model DIR [--iters I] [--threads T]
int cmd_head35(const std::string &model, int iters, int threads) {
  namespace q35 = xenithium::arch::qwen35;
  const auto mc = m::ModelConfig::load(model);
  const auto arch = m::open_arch(mc);
  if (arch->id() != "qwen35") {
    std::fprintf(stderr, "head35: %s is arch %s, not qwen35\n", model.c_str(), std::string(arch->id()).c_str());
    return 2;
  }
  const q35::Config cfg = q35::Config::parse(mc);
  const m::Checkpoint ck = m::Checkpoint::open(model);
  const std::string P = q35::kPrefix, EMB = P + "embed_tokens.weight", NW = P + "norm.weight", LM = "lm_head.weight";
  std::vector<m::DenseSpec> plan;
  for (const auto &sp : arch->dense_plan(ck, m::DeployParams{}))
    if (sp.name == EMB || sp.name == NW || sp.name == LM) plan.push_back(sp);
  const auto loader = xenithium::l0::Loader::open();
  xenithium::Fabric fab(loader, {0, 1});
  xenithium::load::DenseStore dense(fab);
  dense.load(ck, plan, threads);
  std::printf("head35: %zu tensors loaded in %.2f s (embedding %.2f GiB pinned host)\n", plan.size(), dense.stats().total_s,
              dense.stats().host_bytes / 1073741824.0);
  q35::Head head(fab, dense, cfg);
  constexpr int rows = 4;
  const int H = head.hidden(), V = head.vocab_per_card();
  const int32_t ids[rows] = {11, 1000, 50000, 248043};
  head.set_ids(ids, rows);
  head.embed(rows);
  head.final_norm(rows);
  head.lm_pick(rows);
  head.wait();
  int rc = 0;
  auto check = [&](bool ok, const std::string &what) {
    std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
    rc |= !ok;
  };
  auto read_bf = [&](int c, const bf16 *p, size_t n) {
    std::vector<uint16_t> v(n);
    fab.compute(c).memcpy(v.data(), p, n * 2).wait();
    return v;
  };
  std::vector<uint16_t> res_d[2], xn_d[2];
  std::vector<float> lg[2];
  std::vector<int32_t> tok[2];
  for (int c = 0; c < 2; ++c) {
    res_d[c] = read_bf(c, head.residual(c), static_cast<size_t>(rows) * H);
    xn_d[c] = read_bf(c, head.card(c).xn, static_cast<size_t>(rows) * H);
    lg[c].resize(static_cast<size_t>(rows) * V);
    fab.compute(c).memcpy(lg[c].data(), head.card(c).logits, lg[c].size() * 4).wait();
    tok[c] = head.tokens(c, rows);
  }
  // ---- host references from the raw checkpoint ----
  const auto *table = reinterpret_cast<const uint16_t *>(dense.at(0, EMB).ptr);
  std::vector<uint16_t> res_ref(static_cast<size_t>(rows) * H);
  for (int r = 0; r < rows; ++r) std::memcpy(&res_ref[static_cast<size_t>(r) * H], table + static_cast<size_t>(ids[r]) * H, H * 2);
  check(res_d[0] == res_ref && res_d[1] == res_ref, "embedding gather (pinned host table): residual == table rows on both cards");
  const auto &tn = ck.at(NW), &tl = ck.at(LM);
  const auto raw = m::bulk_read(ck, {&tn, &tl});
  std::vector<uint16_t> xn_ref(res_ref.size()), tmp = res_ref;
  k::rms_norm_ref(rows, H, tmp.data(), nullptr, nullptr, raw.as<uint16_t>(tn), static_cast<float>(cfg.rms_eps), xn_ref.data());
  {
    auto lin = [](uint16_t v) { return (v & 0x8000) ? -static_cast<int>(v & 0x7fff) : static_cast<int>(v); };
    int mx = 0;
    size_t off = 0, beyond = 0;
    double worst = 0;
    for (int r = 0; r < rows; ++r) {
      double ss = 0;
      for (int i = 0; i < H; ++i) ss += std::pow(k::bf16_value(xn_ref[static_cast<size_t>(r) * H + i]), 2);
      const double rms = std::sqrt(ss / H);
      for (int i = 0; i < H; ++i) {
        const size_t at = static_cast<size_t>(r) * H + i;
        const int u = std::abs(lin(xn_d[0][at]) - lin(xn_ref[at]));
        mx = std::max(mx, u);
        off += u != 0;
        const double y = k::bf16_value(xn_ref[at]), sc = std::fabs(k::bf16_value(xn_d[0][at]) - y) / std::max(std::fabs(y), rms);
        worst = std::max(worst, sc);
        beyond += sc > 1.0 / 128;
      }
    }
    char line[256];
    std::snprintf(line, sizeof line, "final norm vs rms_norm_ref: max %d ulp, worst |diff| / max(|ref|, rms) %.2e (2 ulps = 7.8e-03), %zu / %zu differ", mx,
                  worst, off, xn_ref.size());
    check(beyond == 0, line);
    check(xn_d[0] == xn_d[1], "final norm: both cards bit-identical (replicated)");
  }
  // Pick vs the device's own logits (exact), and both cards agree.
  bool pick_ok = true;
  for (int r = 0; r < rows; ++r) {
    std::vector<float> full(lg[0].begin() + static_cast<size_t>(r) * V, lg[0].begin() + static_cast<size_t>(r + 1) * V);
    full.insert(full.end(), lg[1].begin() + static_cast<size_t>(r) * V, lg[1].begin() + static_cast<size_t>(r + 1) * V);
    pick_ok &= tok[0][r] == k::pick_ref(full.data(), 2 * V, 0) && tok[1][r] == tok[0][r];
  }
  check(pick_ok, "greedy pick == host argmax over both cards' logits; both cards hold the same tokens");
  check(head.wire_error(0) == 0 && head.wire_error(1) == 0, "pick wire: no mailbox timeout");
  // lm_head logits vs the double reference on the same int8 rows (sampled columns per card), input = the device's own norm output.
  const auto *lmw = raw.as<uint16_t>(tl);
  std::vector<float> x(static_cast<size_t>(rows) * H);
  for (size_t i = 0; i < x.size(); ++i) x[i] = k::bf16_value(xn_d[0][i]);
  double worst = 0;
  std::mt19937 rng(9);
  for (int c = 0; c < 2; ++c)
    for (int sidx = 0; sidx < 1024; ++sidx) {
      const int col = static_cast<int>(rng() % V);
      const uint16_t *wr = lmw + (static_cast<size_t>(c) * V + col) * H;
      std::vector<int8_t> code(H);
      float sc;
      k::quantize_i8_rows(wr, 1, H, code.data(), &sc, 1);
      double yr[rows], mag = 0;
      k::dense_ref_i8(x.data(), code.data(), &sc, rows, 1, H, yr);
      for (int r = 0; r < rows; ++r) mag = std::max(mag, std::fabs(yr[r]));
      for (int r = 0; r < rows; ++r) worst = std::max(worst, std::fabs(lg[c][static_cast<size_t>(r) * V + col] - yr[r]) / std::max(mag, 1e-3));
    }
  {
    char line[200];
    std::snprintf(line, sizeof line, "lm_head logits vs double reference on the same int8 rows (2 x 1,024 sampled columns): worst rel err %.2e", worst);
    check(worst <= 1e-5, line);
  }
  // Quantisation effect end to end: the whole lm_head in bf16 (host, threads) on the host-reference norm vs the device tokens.
  {
    std::vector<double> lbf(static_cast<size_t>(rows) * 2 * V);
    std::vector<float> xb(static_cast<size_t>(rows) * H);
    for (size_t i = 0; i < xb.size(); ++i) xb[i] = k::bf16_value(xn_ref[i]);
    std::vector<std::thread> th;
    const int T = std::max(1, threads);
    for (int t = 0; t < T; ++t)
      th.emplace_back([&, t] {
        for (int64_t col = t; col < 2LL * V; col += T) {
          const uint16_t *wr = lmw + static_cast<size_t>(col) * H;
          double acc[rows] = {0, 0, 0, 0};
          for (int i = 0; i < H; ++i) {
            const double w = k::bf16_value(wr[i]);
            for (int r = 0; r < rows; ++r) acc[r] += w * xb[static_cast<size_t>(r) * H + i];
          }
          for (int r = 0; r < rows; ++r) lbf[static_cast<size_t>(r) * 2 * V + col] = acc[r];
        }
      });
    for (auto &t : th) t.join();
    int agree = 0;
    for (int r = 0; r < rows; ++r) {
      std::vector<float> row(lbf.begin() + static_cast<size_t>(r) * 2 * V, lbf.begin() + static_cast<size_t>(r + 1) * 2 * V);
      const int64_t tb = k::pick_ref(row.data(), 2 * V, 0);
      agree += tb == tok[0][r];
      std::printf("       row %d id %6d -> device token %6d, all-bf16 reference token %6lld (bf16 logit %.3f vs %.3f at the device pick)\n", r, ids[r],
                  tok[0][r], static_cast<long long>(tb), static_cast<float>(bf16(row[static_cast<size_t>(tb)])),
                  static_cast<float>(bf16(row[static_cast<size_t>(tok[0][r])])));
    }
    std::printf("       int8 path vs all-bf16 reference: %d / %d rows pick the same token\n", agree, rows);
  }

  // ---- timing: the whole chain per step (both cards), and each stage's kernel time on card 0 ----
  std::vector<double> wall, st[5];
  auto step = [&] {
    head.embed(rows);
    head.final_norm(rows);
    head.lm_pick(rows);
    head.wait();
  };
  xenithium::probe::warm_for(step);
  for (int i = 0; i < iters; ++i) {
    const auto t0 = std::chrono::steady_clock::now();
    step();
    wall.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    const auto &e = head.events(0);
    const sycl::event *evs[5] = {&e.embed, &e.norm, &e.lm, &e.chunks, &e.pick};
    for (int j = 0; j < 5; ++j) st[j].push_back(ev_ms(*evs[j]) * 1e3);
  }
  char lmname[64];
  std::snprintf(lmname, sizeof lmname, "lm_head %dx%d i8", V, H);
  const char *names[5] = {"embed gather (host table)", "final norm (1 + w)", lmname, "pick chunk keys", "pick merge + wire"};
  const double bytes[5] = {0, 0, static_cast<double>(V) * H + 4.0 * V, 4.0 * V * 4, 0};
  std::printf("\nper-stage kernel time, card 0, %d iterations, both cards running the chain\n%-28s %10s %10s %10s\n", iters, "stage", "median us", "p90 us",
              "x floor");
  double sum = 0;
  for (int j = 0; j < 5; ++j) {
    std::sort(st[j].begin(), st[j].end());
    const double med = st[j][st[j].size() / 2], p90 = st[j][st[j].size() * 9 / 10];
    sum += med;
    if (bytes[j] > 0) std::printf("%-28s %10.2f %10.2f %10.2f\n", names[j], med, p90, med / (bytes[j] / 604e3));
    else std::printf("%-28s %10.2f %10.2f %10s\n", names[j], med, p90, "-");
  }
  std::sort(wall.begin(), wall.end());
  std::printf("%-28s %10.2f\n%-28s %10.2f %10.2f   (host wall, submit + wait of both cards)\n", "sum of stages", sum, "whole chain",
              wall[wall.size() / 2] * 1e3, wall[wall.size() * 9 / 10] * 1e3);
  check(head.wire_error(0) == 0 && head.wire_error(1) == 0, "pick wire after timing: no mailbox timeout");
  std::printf("head35: %s\n", rc ? "FAIL" : "PASS");
  return rc;
}

// ---- probe mlp35: the dense SwiGLU MLP on real weights, both cards -----------------------------
// Per layer: gate | up (w4a16_gemm_rows) -> swiglu -> down -> pair_reduce through the mailbox, vs dense_mlp_ref on
// host slices built independently from the raw tensors (both cards' partials summed in double, one rounding to bf16).
// Gates: out within 2 bf16 ulps of max(|ref|, row rms) with <= 5 % of elements off; both cards identical; run-to-run
// identity. Then per-stage times on card 0 (5-8 rows: a GEMM stage is two M = 4 launches and its event times the second only).
// xenithium-probe mlp35 --model DIR [--layers 0,3] [--rows N] [--iters I] [--threads T]
int cmd_mlp35(const std::string &model, const std::string &layers_arg, int rows, int iters, int threads) {
  namespace q35 = xenithium::arch::qwen35;
  const auto mc = m::ModelConfig::load(model);
  const auto arch = m::open_arch(mc);
  if (arch->id() != "qwen35") {
    std::fprintf(stderr, "mlp35: %s is arch %s, not qwen35\n", model.c_str(), std::string(arch->id()).c_str());
    return 2;
  }
  const q35::Config cfg = q35::Config::parse(mc);
  const m::Checkpoint ck = m::Checkpoint::open(model);
  std::vector<int> layers;
  {
    std::stringstream ss(layers_arg);
    std::string t;
    while (std::getline(ss, t, ','))
      if (!t.empty()) layers.push_back(std::stoi(t));
  }
  if (rows < 1 || rows > q35::Mlp::kRows) {
    std::fprintf(stderr, "mlp35: --rows 1..%d\n", q35::Mlp::kRows);
    return 2;
  }
  std::vector<m::DenseSpec> plan;
  for (const auto &sp : arch->dense_plan(ck, m::DeployParams{}))
    if (sp.store == m::DenseSpec::Store::w4 && sp.comp == m::Component::dense_mlp && std::find(layers.begin(), layers.end(), sp.layer) != layers.end())
      plan.push_back(sp);
  const auto loader = xenithium::l0::Loader::open();
  xenithium::Fabric fab(loader, {0, 1});
  const int C = 2;
  xenithium::load::DenseStore dense(fab);
  dense.load(ck, plan, threads);
  std::printf("mlp35: %zu w4 matrices of layers %s loaded in %.2f s\n", plan.size(), layers_arg.c_str(), dense.stats().total_s);
  q35::Mlp mlp(fab, cfg);
  const int H = mlp.hidden(), ipc = mlp.ipc();
  int rc = 0;
  auto check = [&](bool ok, const std::string &what) {
    std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
    rc |= !ok;
  };
  // Inputs: rows x H, ~N(0, 1) bf16 (a residual stream after a norm is O(1)); the same x on both cards.
  std::mt19937 rng(11);
  std::normal_distribution<float> nd(0.f, 1.f);
  std::vector<bf16> xb(static_cast<size_t>(q35::Mlp::kRows) * H);
  std::vector<float> xf(xb.size());
  for (size_t i = 0; i < xb.size(); ++i) {
    xb[i] = bf16(nd(rng));
    xf[i] = static_cast<float>(xb[i]);
  }
  std::vector<const bf16 *> x(C);
  for (int c = 0; c < C; ++c) {
    auto *d = static_cast<bf16 *>(fab.alloc(c, xb.size() * 2));
    fab.compute(c).memcpy(d, xb.data(), xb.size() * 2).wait();
    x[c] = d;
  }
  const double bytes_gu = static_cast<double>(2 * ipc) * H / 2 + static_cast<double>(H / 128) * 2 * ipc * 2;
  const double bytes_dn = static_cast<double>(H) * ipc / 2 + static_cast<double>(ipc / 128) * H * 2;
  std::printf("per card: gate | up %d x %d (%.1f MB), down %d x %d (%.1f MB); floor per layer %.1f us at 604 GB/s\n", 2 * ipc, H, bytes_gu / 1e6, H, ipc,
              bytes_dn / 1e6, (bytes_gu + bytes_dn) / 604e3);
  std::printf("%-6s %4s | %8s %10s %10s | %8s %8s %8s %8s %8s %8s\n", "layer", "rows", "max ulp", "worst", "off", "gu us", "silu us", "down us", "reduce us",
              "sum us", "x floor");
  for (const int L : layers) {
    std::vector<q35::MlpWeights> w(C);
    for (int c = 0; c < C; ++c) w[c] = mlp.weights(dense, c, L);
    // Host reference: both cards' partials from independent host slices, summed in double.
    std::vector<double> part_sum(static_cast<size_t>(rows) * H, 0.0);
    for (int c = 0; c < C; ++c) {
      const m::DenseSpec *sgu = nullptr, *sdn = nullptr;
      for (const auto &sp : plan)
        if (sp.layer == L) {
          if (sp.name.find("gate_proj") != std::string::npos) sgu = &sp;
          if (sp.name.find("down_proj") != std::string::npos) sdn = &sp;
        }
      const auto srcs_gu = w4_sources(ck, *sgu), srcs_dn = w4_sources(ck, *sdn);
      const auto buf_gu = m::bulk_read(ck, srcs_gu), buf_dn = m::bulk_read(ck, srcs_dn);
      const HostW4 hgu = host_w4_slice(ck, *sgu, buf_gu, srcs_gu, c, C, 2 * ipc, H);
      const HostW4 hdn = host_w4_slice(ck, *sdn, buf_dn, srcs_dn, c, C, H, ipc);
      if (hgu.rows != 2 * ipc || hdn.rows != H) {
        check(false, "host slice rows");
        continue;
      }
      std::vector<double> part(part_sum.size());
      k::dense_mlp_ref(rows, H, ipc, xf.data(), hgu.gptq.data(), hgu.scales.data(), hdn.gptq.data(), hdn.scales.data(), part.data());
      for (size_t i = 0; i < part.size(); ++i) part_sum[i] += part[i];
    }
    std::vector<uint16_t> ref(part_sum.size());
    for (size_t i = 0; i < ref.size(); ++i) ref[i] = k::bf16_bits(static_cast<float>(part_sum[i]));
    // Device, twice (run-to-run identity), both cards.
    std::vector<uint16_t> got[2], again[2];
    for (int rep = 0; rep < 2; ++rep) {
      mlp.run(rows, x, w);
      mlp.wait();
      for (int c = 0; c < C; ++c) {
        auto &v = rep ? again[c] : got[c];
        v.resize(ref.size());
        fab.compute(c).memcpy(v.data(), mlp.card(c).out, v.size() * 2).wait();
      }
    }
    auto lin = [](uint16_t v) { return (v & 0x8000) ? -static_cast<int>(v & 0x7fff) : static_cast<int>(v); };
    int mx = 0;
    size_t off = 0, beyond = 0;
    double worst = 0;
    for (int r = 0; r < rows; ++r) {
      double ss = 0;
      for (int i = 0; i < H; ++i) ss += std::pow(k::bf16_value(ref[static_cast<size_t>(r) * H + i]), 2);
      const double rms = std::sqrt(ss / H);
      for (int i = 0; i < H; ++i) {
        const size_t at = static_cast<size_t>(r) * H + i;
        const int u = std::abs(lin(got[0][at]) - lin(ref[at]));
        mx = std::max(mx, u);
        off += u != 0;
        const double y = k::bf16_value(ref[at]), sc = std::fabs(k::bf16_value(got[0][at]) - y) / std::max(std::fabs(y), rms);
        worst = std::max(worst, sc);
        beyond += sc > 1.0 / 128;
      }
    }
    const bool same_cards = got[0] == got[1], same_run = got[0] == again[0] && got[1] == again[1];
    // Two chained bf16 roundings (g / u, then h) before the down sum: a one-ulp flip upstream moves a few % of the
    // outputs by one ulp (layer 0: 1.9 %), so the count gate is 5 % here (the mixer's 1 % is a single rounding).
    const bool ok = beyond == 0 && off * 20 <= ref.size() && same_cards && same_run && mlp.wire_error(0) == 0 && mlp.wire_error(1) == 0;
    rc |= !ok;
    // Timing: card 0's stages, both cards running.
    std::vector<double> st[4];
    xenithium::probe::warm_for([&] {
      mlp.run(rows, x, w);
      mlp.wait();
    });
    for (int i = 0; i < iters; ++i) {
      mlp.run(rows, x, w);
      mlp.wait();
      const auto &e = mlp.events(0);
      const sycl::event *evs[4] = {&e.gu, &e.silu, &e.down, &e.reduce};
      for (int j = 0; j < 4; ++j) st[j].push_back(ev_ms(*evs[j]) * 1e3);
    }
    double med[4], sum = 0;
    for (int j = 0; j < 4; ++j) {
      std::sort(st[j].begin(), st[j].end());
      med[j] = st[j][st[j].size() / 2];
      sum += med[j];
    }
    std::printf("%-6d %4d | %8d %10.2e %5zu/%zu | %8.1f %8.1f %8.1f %8.1f %8.1f %8.2f  [%s]%s%s%s\n", L, rows, mx, worst, off, ref.size(), med[0], med[1], med[2], med[3],
                sum, sum / ((bytes_gu + bytes_dn) / 604e3), ok ? "PASS" : "FAIL", same_cards ? "" : " cards differ", same_run ? "" : " not run-to-run identical",
                beyond ? " beyond 2 ulps" : "");
  }
  check(mlp.wire_error(0) == 0 && mlp.wire_error(1) == 0, "pair wire: no mailbox timeout");
  for (int c = 0; c < C; ++c) fab.release(const_cast<bf16 *>(x[c]));
  if (rows > 4) std::printf("(rows > 4: each GEMM stage is two M = 4 launches; its time above is the second launch's only)\n");
  std::printf("mlp35: %s - %zu layer(s) x %d rows vs dense_mlp_ref on independent host slices (2 ulps of max(|ref|, rms), <= 5 %% off)\n",
              rc ? "FAIL" : "PASS", layers.size(), rows);
  return rc;
}
