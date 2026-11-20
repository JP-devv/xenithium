// xenithium-probe, the llama (Nanbeige4.3-33B) subcommands.
// w4load (arch llama): one or more layers' int4 g32 + zero-point Linears through DenseStore (W4Layout:ct_zp: the codes
// as W4 blocks, bf16 scales -> fp16 [k/32][n], zero points -> nibbles [k/64][n], registered for the W4 GEMMs). Per card
// slice: w4a16_gemm_rows at 4 rows vs w4zp_ref on a slice rebuilt on the host from the raw tensors (independently of the
// store), run-to-run identity, and the time of the matrix alone; with several layers loaded, each role is also timed
// rotated over its copies (the matrices past the L2: the honest GB/s).
// xenithium-probe w4load --model DIR [--layers 0,3 | all] [--iters I] [--threads T] [--tp 1|2]
// w4pf (arch llama): the many-row (prefill) W4 GEMMs on the zero-point matrices (the dequantised mode, w4a16_rows.cpp)
// per shape 12-15 and the default, card 0's TP slice of one layer: gate|up (w4a16_gemm_pf_swiglu), down, q|k|v and o
// (w4a16_gemm_pf), at --rows m (default 512,2048); ms (median of --iters), TFLOPS (2 m n k).
// xenithium-probe w4pf --model DIR [--layers 0] [--rows 512,2048] [--iters I] [--tp 1|2]
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "warmup.hpp"
#include "xenithium/arch/llama/arch.hpp"
#include "xenithium/kernels/pack.hpp"
#include "xenithium/kernels/gemm/w4a16_gemm.hpp"
#include "xenithium/load/dense_store.hpp"
#include "xenithium/model/arch.hpp"
#include "xenithium/model/bulk_read.hpp"
#include "xenithium/rt/fabric.hpp"

namespace m = xenithium::model;
namespace k = xenithium::kernels;
using bf16 = sycl::ext::oneapi::bfloat16;

namespace {

double ev_ms(const sycl::event &e) {
  const auto a = e.get_profiling_info<sycl::info::event_profiling::command_start>();
  const auto b = e.get_profiling_info<sycl::info::event_profiling::command_end>();
  return static_cast<double>(b - a) * 1e-6;
}

float bf16f(uint16_t x) {
  const uint32_t u = static_cast<uint32_t>(x) << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

std::string base_of(const std::string &packed) { return packed.substr(0, packed.size() - std::strlen(".weight_packed")); }

// A card's slice of a ct_zp spec from the raw tensors: ct words [n][k/8], scales [n][k/32] fp32, zero points [n][k/32].
struct HostZp {
  std::vector<uint32_t> words;
  std::vector<float> scales;
  std::vector<uint8_t> zeros;
  int64_t rows = 0;
};

HostZp host_slice(const m::Checkpoint &ck, const m::DenseSpec &sp, const m::BulkBuffer &buf, int c, int C, int n_pad, int kk) {
  std::vector<std::string> mods = {base_of(sp.name)};
  for (const auto &a : sp.append) mods.push_back(base_of(a));
  const int64_t G = k::kW4ZpGroup;
  HostZp h;
  h.words.assign(static_cast<size_t>(n_pad) * (kk / 8), 0x88888888u);
  h.scales.assign(static_cast<size_t>(n_pad) * (kk / G), 0.f);
  h.zeros.assign(static_cast<size_t>(n_pad) * (kk / G), 8);
  for (const auto &mod : mods) {
    const auto &tw = ck.at(mod + ".weight_packed"), &ts = ck.at(mod + ".weight_scale"), &tz = ck.at(mod + ".weight_zero_point");
    const int64_t n = tw.dim(0), kall = tw.dim(1) * 8;
    int64_t r0 = 0, r1 = n, col0 = 0;
    if (sp.split == m::DenseSpec::Split::rows) r0 = n / C * c, r1 = n / C * (c + 1);
    if (sp.split == m::DenseSpec::Split::cols) col0 = kall / C * c;
    const auto *pw = reinterpret_cast<const uint32_t *>(buf.data(tw));
    const auto *ps = reinterpret_cast<const uint16_t *>(buf.data(ts));
    const auto *pz = reinterpret_cast<const uint32_t *>(buf.data(tz));
    for (int64_t r = r0; r < r1; ++r, ++h.rows) {
      for (int64_t j = 0; j < kk / 8; ++j) h.words[static_cast<size_t>(h.rows * (kk / 8) + j)] = pw[r * (kall / 8) + col0 / 8 + j];
      for (int64_t g = 0; g < kk / G; ++g) {
        const int64_t sg = col0 / G + g;
        h.scales[static_cast<size_t>(h.rows * (kk / G) + g)] = ts.dtype == m::DType::bf16 ? bf16f(ps[r * (kall / G) + sg]) : 0.f;
        h.zeros[static_cast<size_t>(h.rows * (kk / G) + g)] = static_cast<uint8_t>((pz[(r / 8) * (kall / G) + sg] >> (4 * (r % 8))) & 15u);
      }
    }
  }
  return h;
}

}  // namespace

int cmd_w4load_llama(const std::string &model, const std::string &layers_arg, int iters, int threads, int tp) {
  const auto mc = m::ModelConfig::load(model);
  const auto arch = m::open_arch(mc);
  const auto cfg = xenithium::arch::llama::parse_config(mc);
  const m::Checkpoint ck = m::Checkpoint::open(model);
  std::vector<int> layers;
  const bool all = layers_arg == "all";
  if (all)
    for (int l = 0; l < cfg.layers; ++l) layers.push_back(l);
  else {
    std::stringstream ss(layers_arg);
    std::string t;
    while (std::getline(ss, t, ','))
      if (!t.empty()) layers.push_back(std::stoi(t));
  }
  const int C = tp == 1 ? 1 : 2;
  m::DeployParams dp;
  dp.cards = C;
  std::vector<m::DenseSpec> plan;
  for (const auto &sp : arch->dense_plan(ck, dp))
    if (sp.store == m::DenseSpec::Store::w4 && std::find(layers.begin(), layers.end(), sp.layer) != layers.end()) plan.push_back(sp);
  if (plan.empty()) {
    std::fprintf(stderr, "w4load: no w4 matrices in layers %s\n", layers_arg.c_str());
    return 2;
  }
  const auto loader = xenithium::l0::Loader::open();
  std::vector<size_t> devs;
  for (int c = 0; c < C; ++c) devs.push_back(static_cast<size_t>(c));
  xenithium::Fabric fab(loader, devs);
  xenithium::load::DenseStore dense(fab);
  dense.load(ck, plan, threads);
  const auto &st = dense.stats();
  double gib = 0;
  for (const auto &kv : st.card_bytes[0]) gib += static_cast<double>(kv.second);
  std::printf("w4load (llama): %zu w4 matrices of layers %s loaded in %.2f s (read wait %.2f, pack %.2f, h2d wait %.2f; %.3f GiB per card, TP%d)\n", plan.size(),
              layers_arg.c_str(), st.total_s, st.read_wait_s, st.pack_s, st.h2d_wait_s, gib / 1073741824.0, C);

  std::mt19937 rng(7);
  std::normal_distribution<float> nd(0.f, 1.f);
  int fails = 0;
  double worst_rel = 0;
  // bytes the GEMM reads per matrix: codes + fp16 scales (k/32) + zero nibbles (k/64 bytes)
  auto bytes_of = [](int n, int kk) { return static_cast<double>(n) * kk / 2 + static_cast<double>(kk / 32) * n * 2 + static_cast<double>(kk / 64) * n; };
  std::printf("%-40s %4s %6s %6s | %10s %10s | %8s %7s %7s\n", "matrix (card slice)", "card", "n", "k", "max|err|", "rel", "us", "GB/s", "x floor");
  for (const auto &sp : plan) {
    if (all && sp.layer != 0 && sp.layer != cfg.layers / 2) continue;
    std::vector<const m::TensorInfo *> srcs;
    for (const auto &mod : [&] {
           std::vector<std::string> v = {base_of(sp.name)};
           for (const auto &a : sp.append) v.push_back(base_of(a));
           return v;
         }())
      for (const char *suf : {".weight_packed", ".weight_scale", ".weight_zero_point"}) srcs.push_back(&ck.at(mod + suf));
    const auto buf = m::bulk_read(ck, srcs);
    for (int c = 0; c < C; ++c) {
      const auto w = dense.at(c, sp.name).w4();
      const int n = w.n, kk = w.k;
      const HostZp hs = host_slice(ck, sp, buf, c, C, n, kk);
      if (hs.rows != dense.at(c, sp.name).rows || !w.zeros || k::w4_zero_points(w.w) != w.zeros) {
        std::printf("  ! %s card %d: host slice rows %lld vs store %lld, or zero points not registered\n", sp.name.c_str(), c, static_cast<long long>(hs.rows),
                    static_cast<long long>(dense.at(c, sp.name).rows));
        ++fails;
        continue;
      }
      std::vector<float> xf(4ull * kk);
      std::vector<bf16> xb(xf.size());
      for (size_t i = 0; i < xf.size(); ++i) {
        const float v = i < 2ull * kk ? static_cast<float>(static_cast<int>(rng() % 9) - 4) : nd(rng);
        xb[i] = bf16(v);
        xf[i] = static_cast<float>(xb[i]);
      }
      std::vector<double> want(4ull * n);
      k::w4zp_ref(xf.data(), hs.words.data(), hs.scales.data(), hs.zeros.data(), 4, n, kk, want.data());
      auto &q = fab.compute(c);
      auto *dx = static_cast<bf16 *>(fab.alloc(c, xb.size() * 2));
      auto *dy = static_cast<float *>(fab.alloc(c, 4ull * n * 4));
      q.memcpy(dx, xb.data(), xb.size() * 2).wait();
      auto launch = [&] { return k::w4a16_gemm_rows(q, 4, dx, w.w, w.scales, dy, n, kk); };
      launch().wait_and_throw();
      std::vector<float> got(4ull * n), again(4ull * n);
      q.memcpy(got.data(), dy, got.size() * 4).wait();
      double max_err = 0, max_abs = 0;
      for (size_t i = 0; i < got.size(); ++i) {
        max_err = std::max(max_err, std::fabs(static_cast<double>(got[i]) - want[i]));
        max_abs = std::max(max_abs, std::fabs(want[i]));
      }
      const double rel = max_abs > 0 ? max_err / max_abs : max_err;
      worst_rel = std::max(worst_rel, rel);
      launch().wait_and_throw();
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
      const double med = ms[ms.size() / 2], bytes = bytes_of(n, kk);
      std::string nm = base_of(sp.name).substr(std::strlen(xenithium::arch::llama::kPrefix));
      if (!sp.append.empty()) nm += " (+" + std::to_string(sp.append.size()) + ")";
      std::printf("%-40s %4d %6d %6d | %10.3e %10.2e | %8.1f %7.1f %7.2f%s\n", nm.c_str(), c, n, kk, max_err, rel, med * 1e3, bytes / (med * 1e6),
                  med / (bytes / 604e6), same ? "" : "  NOT run-to-run identical");
      fab.release(dx);
      fab.release(dy);
    }
  }
  // Rotated: per role, the loaded layers' copies in turn (card 0), 1 and 4 rows: past the L2.
  if (layers.size() >= 4) {
    std::map<std::string, std::vector<xenithium::load::W4Weights>> roles;
    for (const auto &sp : plan) {
      std::string role = base_of(sp.name);
      role = role.substr(role.find(".", role.find("layers.") + 7) + 1);
      roles[role].push_back(dense.at(0, sp.name).w4());
    }
    auto &q = fab.compute(0);
    int kmax = 0, nmax = 0;
    for (auto &kv : roles) kmax = std::max(kmax, kv.second[0].k), nmax = std::max(nmax, kv.second[0].n);
    auto *dx = static_cast<bf16 *>(fab.alloc(0, 8ull * kmax * 2));
    auto *dy = static_cast<float *>(fab.alloc(0, 8ull * nmax * 4));
    q.fill(dx, bf16(0.01f), 8ull * kmax).wait();
    std::printf("rotated (card 0, %zu copies per role): role, rows, us per GEMM, GB/s, x floor\n", layers.size());
    for (auto &kv : roles) {
      const auto &ws = kv.second;
      for (const int rows : {1, 4}) {
        auto run = [&] {
          sycl::event first, last;
          for (size_t i = 0; i < ws.size(); ++i) {
            last = k::w4a16_gemm_rows(q, rows, dx, ws[i].w, ws[i].scales, dy, ws[i].n, ws[i].k);
            if (i == 0) first = last;
          }
          last.wait_and_throw();
          return std::make_pair(first, last);
        };
        run();
        std::vector<double> per;
        for (int it = 0; it < std::max(3, iters / 20); ++it) {
          const auto [a, b] = run();
          const auto t0 = a.get_profiling_info<sycl::info::event_profiling::command_start>();
          const auto t1 = b.get_profiling_info<sycl::info::event_profiling::command_end>();
          per.push_back(static_cast<double>(t1 - t0) * 1e-6 / ws.size());
        }
        std::sort(per.begin(), per.end());
        const double med = per[per.size() / 2], bytes = bytes_of(ws[0].n, ws[0].k);
        std::printf("  %-22s %d %8.1f %7.1f %7.2f\n", kv.first.c_str(), rows, med * 1e3, bytes / (med * 1e6), med / (bytes / 604e6));
      }
    }
    fab.release(dx);
    fab.release(dy);
  }
  std::printf("w4load (llama): %s - matrices x %d card(s) vs w4zp_ref on independent host slices, worst rel %.2e (gate 1e-5), %d failure(s)\n",
              fails ? "FAIL" : "PASS", C, worst_rel, fails);
  return fails ? 1 : 0;
}

int cmd_w4pf_llama(const std::string &model, const std::string &layers_arg, const std::string &rows_arg, int iters, int threads, int tp) {
  const auto mc = m::ModelConfig::load(model);
  const auto arch = m::open_arch(mc);
  const m::Checkpoint ck = m::Checkpoint::open(model);
  const int layer = layers_arg.empty() || layers_arg == "0,24,47" ? 0 : std::stoi(layers_arg);
  std::vector<int> rows;
  {
    std::stringstream ss(rows_arg == "1,4" || rows_arg.empty() ? std::string("512,2048") : rows_arg);
    std::string t;
    while (std::getline(ss, t, ','))
      if (!t.empty()) rows.push_back(std::stoi(t));
  }
  const int C = tp == 1 ? 1 : 2;
  m::DeployParams dp;
  dp.cards = C;
  std::vector<m::DenseSpec> plan;
  for (const auto &sp : arch->dense_plan(ck, dp))
    if (sp.store == m::DenseSpec::Store::w4 && sp.layer == layer) plan.push_back(sp);
  const auto loader = xenithium::l0::Loader::open();
  std::vector<size_t> devs;
  for (int c = 0; c < C; ++c) devs.push_back(static_cast<size_t>(c));
  xenithium::Fabric fab(loader, devs);
  xenithium::load::DenseStore dense(fab);
  dense.load(ck, plan, threads);
  auto &q = fab.compute(0);
  int kmax = 0, nmax = 0;
  for (const auto &sp : plan) kmax = std::max(kmax, dense.at(0, sp.name).w4().k), nmax = std::max(nmax, dense.at(0, sp.name).w4().n);
  const int mmax = *std::max_element(rows.begin(), rows.end());
  auto *a = static_cast<bf16 *>(fab.alloc(0, static_cast<size_t>(mmax) * kmax * 2));
  auto *cbuf = static_cast<float *>(fab.alloc(0, static_cast<size_t>(mmax) * nmax * 4));
  auto *h = static_cast<bf16 *>(fab.alloc(0, static_cast<size_t>(mmax) * nmax * 2));
  {
    std::vector<bf16> ah(static_cast<size_t>(mmax) * kmax);
    std::mt19937 rng(5);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (auto &v : ah) v = bf16(nd(rng));
    q.memcpy(a, ah.data(), ah.size() * 2).wait();
  }
  std::printf("w4pf (llama): layer %d, card 0 of TP%d, zero-point matrices (dequantised W4A16 GEMM); ms median of %d, TFLOPS = 2 m n k\n", layer, C, iters);
  std::printf("%-28s %6s %6s %6s | %-9s %9s %8s\n", "matrix", "m", "n", "k", "shape", "ms", "TFLOPS");
  for (const auto &sp : plan) {
    const auto w = dense.at(0, sp.name).w4();
    const bool glu = sp.name.find("gate_proj") != std::string::npos;
    std::string nm = base_of(sp.name).substr(std::strlen(xenithium::arch::llama::kPrefix));
    for (int mm : rows)
      for (const int shape : {-1, 12, 13, 14, 15}) {
        auto run = [&] {
          return glu ? k::w4a16_gemm_pf_swiglu(q, mm, a, w.k, w.w, w.scales, h, w.n / 2, w.n / 2, w.k, shape)
                     : k::w4a16_gemm_pf(q, mm, a, w.k, w.w, w.scales, cbuf, w.n, w.n, w.k, shape);
        };
        run().wait_and_throw();
        std::vector<double> ms;
        for (int i = 0; i < iters; ++i) {
          auto e = run();
          e.wait_and_throw();
          ms.push_back(ev_ms(e));
        }
        std::sort(ms.begin(), ms.end());
        const double md = ms[ms.size() / 2];
        std::printf("%-28s %6d %6d %6d | %-9s %9.3f %8.1f\n", (nm + (glu ? " (+up, swiglu)" : "")).c_str(), mm, w.n, w.k, shape < 0 ? "auto" : std::to_string(shape).c_str(), md,
                    2.0 * mm * w.n * w.k / (md * 1e9));
      }
  }
  fab.release(a);
  fab.release(cbuf);
  fab.release(h);
  return 0;
}
