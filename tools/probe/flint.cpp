// xenithium-probe, the flint (Recursive Flint INT4) subcommands.
// w4load (arch flint): one or more layers' MLX affine int4 g64 Linears through DenseStore (W4Layout:mlx: the U8 code rows
// as W4 blocks, bf16 scales and biases [k/64][n], registered for the W4 GEMMs). Per card slice: w4a16_gemm_rows at 4 rows
// vs w4aff_ref on a slice rebuilt on the host from the raw tensors (independently of the store; rows 0-1 integer
// activations, rows 2-3 normal), run-to-run identity, and the time of the matrix alone; with 4+ layers loaded, each role is
// also timed rotated over its copies (past the L2: the honest GB/s). The lm_head is included with --layers head.
// xenithium-probe w4load --model DIR [--layers 0,2,3 | all] [--iters I] [--threads T] [--tp 1|2]
// gqaflint (arch flint the attention decode of one card's TP2 share (16 q / 2 kv heads x 128, hidden 2048,
// synthetic W4 projections) per layer kind - sliding (window 513, rope, a 10-page KV ring, sinks) and full (no rope,
// the whole context, sinks) - at contexts 8K / 32K / 131K / 262K, 1 / 4 / 8 rows, int8 and bf16 KV: the median device
// time of each launch (qkv, prep, attn, merge, o) and of the attention stage (attn + merge) against its byte floor (the
// K / V the rows' contexts span, + int8 scales, / 604 GB/s).
// xenithium-probe gqaflint [--iters I] [--context N]
// moeflint: one MoE layer of the checkpoint on both cards (TP2: each card's half of every expert's and the
// shared expert's intermediate columns, the router replicated). Correctness: rows 1 / 4 / 8 of N(0, 1) bf16 inputs, the
// cards' partials summed on the host vs the reference over the dequantised checkpoint tensors teacher-forced at the
// device's route (and the route vs the double reference). Timing on card 0: every launch's median over steps whose
// inputs rotate through 64 sets (other experts each step: no L2 reuse across steps), against the floor = the bytes of
// the step's unique experts' blobs + the shared blob + the router / 604 GB/s.
// xenithium-probe moeflint --model DIR [--layers L] [--iters I]
#include <algorithm>
#include <chrono>
#include <thread>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "warmup.hpp"
#include "xenithium/arch/flint/arch.hpp"
#include "xenithium/kernels/embed/embed_aff.hpp"
#include "xenithium/kernels/attention/gqa.hpp"
#include "xenithium/kernels/gemm/mlp.hpp"
#include "xenithium/kernels/moe/moe_aff.hpp"
#include "xenithium/kernels/attention/qsa_ref.hpp"
#include "xenithium/kernels/pack.hpp"
#include "xenithium/kernels/gemm/w4a16_gemm.hpp"
#include "xenithium/load/dense_store.hpp"
#include "xenithium/model/arch.hpp"
#include "xenithium/model/checkpoint.hpp"
#include "xenithium/model/config.hpp"
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

std::string base_of(const std::string &packed) { return packed.substr(0, packed.size() - std::strlen(".weight")); }

// A card's slice of an mlx spec from the raw tensors: ct words [n][k/8] (the U8 rows), scales and biases [n][k/64] fp32.
struct HostAff {
  std::vector<uint32_t> words;
  std::vector<float> scales, biases;
  int64_t rows = 0;
};

HostAff host_slice(const m::Checkpoint &ck, const m::DenseSpec &sp, const m::BulkBuffer &buf, int c, int C, int n_pad, int kk) {
  std::vector<std::string> mods = {base_of(sp.name)};
  for (const auto &a : sp.append) mods.push_back(base_of(a));
  const int64_t G = k::kW4AffGroup;
  HostAff h;
  h.words.assign(static_cast<size_t>(n_pad) * (kk / 8), 0);
  h.scales.assign(static_cast<size_t>(n_pad) * (kk / G), 0.f);
  h.biases.assign(static_cast<size_t>(n_pad) * (kk / G), 0.f);
  for (const auto &mod : mods) {
    const auto &tw = ck.at(mod + ".weight"), &ts = ck.at(mod + ".scales"), &tb = ck.at(mod + ".biases");
    const int64_t n = tw.dim(0), kall = tw.dim(1) * 2;
    int64_t r0 = 0, r1 = n, col0 = 0;
    if (sp.split == m::DenseSpec::Split::rows) r0 = n / C * c, r1 = n / C * (c + 1);
    if (sp.split == m::DenseSpec::Split::cols) col0 = kall / C * c;
    const auto *pw = reinterpret_cast<const uint8_t *>(buf.data(tw));
    const auto *ps = reinterpret_cast<const uint16_t *>(buf.data(ts));
    const auto *pb = reinterpret_cast<const uint16_t *>(buf.data(tb));
    for (int64_t r = r0; r < r1; ++r, ++h.rows) {
      std::memcpy(&h.words[static_cast<size_t>(h.rows * (kk / 8))], pw + r * (kall / 2) + col0 / 2, static_cast<size_t>(kk / 2));
      for (int64_t g = 0; g < kk / G; ++g) {
        const int64_t sg = col0 / G + g;
        h.scales[static_cast<size_t>(h.rows * (kk / G) + g)] = bf16f(ps[r * (kall / G) + sg]);
        h.biases[static_cast<size_t>(h.rows * (kk / G) + g)] = bf16f(pb[r * (kall / G) + sg]);
      }
    }
  }
  return h;
}

}  // namespace

int cmd_w4load_flint(const std::string &model, const std::string &layers_arg, int iters, int threads, int tp) {
  const auto mc = m::ModelConfig::load(model);
  const auto arch = m::open_arch(mc);
  const auto cfg = xenithium::arch::flint::Config::parse(mc);
  const m::Checkpoint ck = m::Checkpoint::open(model);
  std::vector<int> layers;
  const bool all = layers_arg == "all";
  if (all)
    for (int l = 0; l < cfg.layers; ++l) layers.push_back(l);
  else {
    std::stringstream ss(layers_arg);
    std::string t;
    while (std::getline(ss, t, ','))
      if (!t.empty()) layers.push_back(t == "head" ? -1 : std::stoi(t));
  }
  const int C = tp == 1 ? 1 : 2;
  m::DeployParams dp;
  dp.cards = C;
  std::vector<m::DenseSpec> plan;
  for (const auto &sp : arch->dense_plan(ck, dp))
    if ((sp.store == m::DenseSpec::Store::w4 && std::find(layers.begin(), layers.end(), sp.layer) != layers.end()) || sp.comp == m::Component::embedding)
      plan.push_back(sp);
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
  int fails_embed = 0;
  const auto &st = dense.stats();
  double gib = 0;
  for (const auto &kv : st.card_bytes[0]) gib += static_cast<double>(kv.second);
  std::printf("w4load (flint): %zu w4 matrices of layers %s loaded in %.2f s (read wait %.2f, pack %.2f, h2d wait %.2f; %.3f GiB per card, TP%d)\n", plan.size(),
              layers_arg.c_str(), st.total_s, st.read_wait_s, st.pack_s, st.h2d_wait_s, gib / 1073741824.0, C);

  // The 8-bit embedding: embed_aff8 on 256 random ids (+ the first / last id and one outside the vocabulary) on every card
  // == embed_aff8_ref on the raw tensors, bit for bit.
  {
    const std::string E = "model.embed_tokens";
    const auto ebuf = m::bulk_read(ck, {&ck.at(E + ".weight"), &ck.at(E + ".scales"), &ck.at(E + ".biases")});
    const auto *tw = ebuf.data(ck.at(E + ".weight"));
    const auto *ts = reinterpret_cast<const uint16_t *>(ebuf.data(ck.at(E + ".scales")));
    const auto *tb = reinterpret_cast<const uint16_t *>(ebuf.data(ck.at(E + ".biases")));
    std::mt19937 erng(11);
    std::vector<int32_t> ids(256);
    for (auto &v : ids) v = static_cast<int32_t>(erng() % cfg.vocab);
    ids[0] = 0, ids[1] = static_cast<int32_t>(cfg.vocab - 1), ids[2] = static_cast<int32_t>(cfg.vocab);
    const int H = cfg.hidden, R = static_cast<int>(ids.size());
    std::vector<uint16_t> want(static_cast<size_t>(R) * H);
    for (int r = 0; r < R; ++r) k::embed_aff8_ref(ids[r], tw, ts, tb, cfg.vocab, H, &want[static_cast<size_t>(r) * H]);
    bool same = true;
    double us = 0;
    for (int c = 0; c < C; ++c) {
      auto &q = fab.compute(c);
      auto *did = static_cast<int32_t *>(fab.alloc(c, ids.size() * 4));
      auto *dout = static_cast<bf16 *>(fab.alloc(c, want.size() * 2));
      q.memcpy(did, ids.data(), ids.size() * 4).wait();
      const auto &dw = dense.at(c, E + ".weight"), &ds = dense.at(c, E + ".scales"), &db = dense.at(c, E + ".biases");
      auto launch = [&](int rows) {
        return k::embed_aff8(q, rows, did, dw.ptr, reinterpret_cast<const uint16_t *>(ds.ptr), reinterpret_cast<const uint16_t *>(db.ptr), cfg.vocab, H, dout, H);
      };
      launch(R).wait_and_throw();
      std::vector<uint16_t> got(want.size());
      q.memcpy(got.data(), dout, got.size() * 2).wait();
      same &= got == want;
      std::vector<double> ms;
      for (int i = 0; i < 20; ++i) {
        auto e = launch(4);
        e.wait();
        ms.push_back(ev_ms(e));
      }
      std::sort(ms.begin(), ms.end());
      us = ms[ms.size() / 2] * 1e3;
      fab.release(did);
      fab.release(dout);
    }
    if (!same) ++fails_embed;
    std::printf("embedding (int8 g64, %d rows incl. id 0, vocab - 1 and one outside): embed_aff8 %s embed_aff8_ref on %d card(s); 4 rows %.1f us\n", R,
                same ? "==" : "!=", C, us);
  }

  std::mt19937 rng(7);
  std::normal_distribution<float> nd(0.f, 1.f);
  int fails = fails_embed;
  double worst_rel = 0;
  // bytes the GEMM reads per matrix: codes + bf16 scales and biases (k/64 each)
  auto bytes_of = [](int n, int kk) { return static_cast<double>(n) * kk / 2 + static_cast<double>(kk / 64) * n * 4; };
  std::printf("%-40s %4s %6s %6s | %10s %10s | %8s %7s %7s\n", "matrix (card slice)", "card", "n", "k", "max|err|", "rel", "us", "GB/s", "x floor");
  for (const auto &sp : plan) {
    if (sp.store != m::DenseSpec::Store::w4) continue;
    if (all && sp.layer != 0 && sp.layer != 2 && sp.layer != cfg.layers / 2) continue;
    std::vector<const m::TensorInfo *> srcs;
    for (const auto &mod : [&] {
           std::vector<std::string> v = {base_of(sp.name)};
           for (const auto &a : sp.append) v.push_back(base_of(a));
           return v;
         }())
      for (const char *suf : {".weight", ".scales", ".biases"}) srcs.push_back(&ck.at(mod + suf));
    const auto buf = m::bulk_read(ck, srcs);
    for (int c = 0; c < C; ++c) {
      const auto w = dense.at(c, sp.name).w4();
      const int n = w.n, kk = w.k;
      const HostAff hs = host_slice(ck, sp, buf, c, C, n, kk);
      if (hs.rows != dense.at(c, sp.name).rows || !w.biases || k::w4_affine(w.w) != w.biases) {
        std::printf("  ! %s card %d: host slice rows %lld vs store %lld, or biases not registered\n", sp.name.c_str(), c, static_cast<long long>(hs.rows),
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
      k::w4aff_ref(xf.data(), hs.words.data(), hs.scales.data(), hs.biases.data(), 4, n, kk, want.data());
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
      std::string nm = (sp.name.rfind("model.", 0) == 0 ? base_of(sp.name).substr(6) : base_of(sp.name));
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
      if (sp.store != m::DenseSpec::Store::w4 || sp.layer < 0) continue;
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
    std::printf("rotated (card 0, every loaded copy per role in turn): role, rows, us per GEMM, GB/s, x floor\n");
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
        // bound of the affine terms: the same matrices read as symmetric g128 (registration lifted; wrong values, same codes)
        std::vector<const uint16_t *> saved;
        for (const auto &w : ws) saved.push_back(k::w4_affine(w.w)), k::w4_clear_affine(w.w);
        run();
        std::vector<double> sym;
        for (int it = 0; it < std::max(3, iters / 20); ++it) {
          const auto [a, b] = run();
          sym.push_back(static_cast<double>(b.get_profiling_info<sycl::info::event_profiling::command_end>() -
                                            a.get_profiling_info<sycl::info::event_profiling::command_start>()) * 1e-6 / ws.size());
        }
        for (size_t i = 0; i < ws.size(); ++i) k::w4_set_affine(ws[i].w, saved[i]);
        std::sort(sym.begin(), sym.end());
        std::printf("  %-28s %d %8.1f %7.1f %7.2f   (as sym g128: %.1f us)\n", kv.first.c_str(), rows, med * 1e3, bytes / (med * 1e6), med / (bytes / 604e6),
                    sym[sym.size() / 2] * 1e3);
      }
    }
    fab.release(dx);
    fab.release(dy);
  }
  std::printf("w4load (flint): %s - matrices x %d card(s) vs w4aff_ref on independent host slices, worst rel %.2e (gate 1e-5), %d failure(s)\n",
              fails ? "FAIL" : "PASS", C, worst_rel, fails);
  return fails ? 1 : 0;
}


int cmd_gqa_flint(int iters, int context) {
  const auto loader = xenithium::l0::Loader::open();
  xenithium::Fabric fab(loader, std::vector<size_t>{0});
  auto &q = fab.compute(0);
  const int H = 2048, D = 128;
  std::vector<int> ctxs = context > 0 ? std::vector<int>{context} : std::vector<int>{8192, 32768, 131072, 262144};
  const int cap_max = *std::max_element(ctxs.begin(), ctxs.end()) + 64;
  std::mt19937 rng(5);
  auto upload = [&](const void *src, size_t bytes) {
    void *d = fab.alloc(0, std::max<size_t>(bytes, 64));
    q.memcpy(d, src, bytes).wait();
    return d;
  };
  auto w4 = [&](int n, int kk) {
    std::vector<uint8_t> b(static_cast<size_t>(n) * kk / 2);
    for (auto &x : b) x = static_cast<uint8_t>(rng());
    std::vector<sycl::half> sc(static_cast<size_t>(kk / 128) * n, sycl::half(0.01f));
    k::W4Mat m;
    m.w = static_cast<uint8_t *>(upload(b.data(), b.size()));
    m.scales = static_cast<sycl::half *>(upload(sc.data(), sc.size() * 2));
    m.n = n, m.k = kk;
    return m;
  };
  std::vector<uint16_t> ones(D, 0x3f80), sinks(16, 0);
  const auto rope_h = k::qsa_rope_table(cap_max, 128, 1e4, k::QsaYarn{});
  auto *rope = static_cast<bf16 *>(upload(rope_h.data(), rope_h.size() * 2));
  {  // clocks up: ~0.5 s of back-to-back work before the first measurement
    void *buf = fab.alloc(0, 256 << 20);
    for (int i = 0; i < 200; ++i) q.memset(buf, i, 256 << 20);
    q.wait();
    fab.release(buf);
  }
  std::printf("gqaflint: one card's TP2 share (16 q / 2 kv heads x 128, hidden 2048), median of %d steps, us; floor = K / V bytes / 604 GB/s\n", iters);
  std::printf("%-7s %-4s %7s %4s | %6s %6s %7s %6s %6s | %8s %8s %6s\n", "layer", "kv", "context", "rows", "qkv", "prep", "attn", "merge", "o", "att+mrg",
              "floor", "x");
  for (const k::QsaKv kv : {k::QsaKv::i8, k::QsaKv::bf16}) {
    for (const bool sliding : {true, false}) {
      k::GqaShape sh{H, 16, 2, 1, 0, 1e-6f, kv};
      sh.head_dim = D, sh.gate = false, sh.qk_norm = true, sh.norm_plus1 = false, sh.rot = sliding ? 128 : 0, sh.window = sliding ? 513 : 0;
      sh.split_merge = std::getenv("XEN_GQA_WALK_MERGE") == nullptr;
      k::GqaWeights w;
      w.qkv = w4(sh.qkv_n(), H);
      w.o = w4(H, sh.o_k());
      w.q_norm = static_cast<bf16 *>(upload(ones.data(), D * 2));
      w.k_norm = static_cast<bf16 *>(upload(ones.data(), D * 2));
      w.sinks = static_cast<bf16 *>(upload(sinks.data(), sinks.size() * 2));
      for (const int ctx : ctxs) {
        const int cap = ctx + 64, pages = sliding ? 10 : cap / 64;
        k::GqaCache c;
        c.max_tokens = cap;
        c.slot_pages = pages;
        c.ring_pages = sliding ? pages : 0;
        std::vector<int32_t> pt(pages);
        for (int i = 0; i < pages; ++i) pt[i] = i;
        c.pages = static_cast<int32_t *>(upload(pt.data(), pages * 4));
        const size_t kvb = static_cast<size_t>(pages) * sh.hkv() * 64 * D * sh.kv_elem();
        c.k = static_cast<uint8_t *>(fab.alloc(0, kvb));
        c.v = static_cast<uint8_t *>(fab.alloc(0, kvb));
        q.memset(c.k, 0x11, kvb);
        q.memset(c.v, 0x11, kvb);
        if (kv == k::QsaKv::i8) {
          const size_t ns = static_cast<size_t>(pages) * sh.hkv() * 64 * 4;
          c.ks = static_cast<float *>(fab.alloc(0, ns));
          c.vs = static_cast<float *>(fab.alloc(0, ns));
          q.memset(c.ks, 0, ns);
          q.memset(c.vs, 0, ns);
        }
        k::GqaScratch sc;
        sc.y = static_cast<float *>(fab.alloc(0, 8ull * sh.qkv_n() * 4));
        sc.q = static_cast<bf16 *>(fab.alloc(0, 8ull * 16 * D * 2));
        sc.part = static_cast<float *>(fab.alloc(0, k::gqa_part_floats(sh, 8) * 4));
        sc.attn = static_cast<bf16 *>(fab.alloc(0, 8ull * 16 * D * 2));
        sc.partial = static_cast<float *>(fab.alloc(0, 8ull * H * 4));
        auto *x = static_cast<bf16 *>(fab.alloc(0, 8ull * H * 2));
        auto *out = static_cast<bf16 *>(fab.alloc(0, 8ull * H * 2));
        q.memset(x, 0, 8ull * H * 2);
        auto *pos = static_cast<int32_t *>(fab.alloc(0, 64));
        q.wait();
        for (const int rows : {1, 4, 8}) {
          const int p = ctx - rows;
          q.memcpy(pos, &p, 4).wait();
          k::GqaStep st;
          st.pos = pos;
          st.pos_bound = ctx;
          // the group sweep: attention + merge per (rows per group, register file); 0 x 0 = the default
          std::string sweep;
          double best = 1e30;
          std::string best_s;
          for (const int ar : {1, 2, 4, 8})
            for (const int grf : {128, 256}) {
              if (ar > rows && ar > 1) continue;
              k::GqaStep s2 = st;
              s2.attn_rows = ar, s2.attn_grf = grf;
              std::vector<double> tt;
              for (int it = 0; it < iters / 2 + 3; ++it) {
                const auto e = k::gqa_decode(q, sh, rows, x, w, c, sliding ? rope : nullptr, s2, sc, nullptr, 0, out);
                q.wait_and_throw();
                if (it >= 3) tt.push_back(ev_ms(e.attn) + ev_ms(e.merge));
              }
              std::sort(tt.begin(), tt.end());
              const double us = tt[tt.size() / 2] * 1e3;
              char b[48];
              std::snprintf(b, sizeof b, " %dx%d %.1f", ar, grf, us);
              sweep += b;
              if (us < best) best = us, best_s = b;
            }
          std::vector<double> t[6];
          for (int it = 0; it < iters + 3; ++it) {
            const auto e = k::gqa_decode(q, sh, rows, x, w, c, sliding ? rope : nullptr, st, sc, nullptr, 0, out);
            q.wait_and_throw();
            if (it < 3) continue;
            t[0].push_back(ev_ms(e.qkv)), t[1].push_back(ev_ms(e.prep)), t[2].push_back(ev_ms(e.attn)), t[3].push_back(ev_ms(e.merge));
            t[4].push_back(ev_ms(e.o)), t[5].push_back(ev_ms(e.attn) + ev_ms(e.merge));
          }
          double med[6];
          for (int i = 0; i < 6; ++i) {
            std::sort(t[i].begin(), t[i].end());
            med[i] = t[i][t[i].size() / 2] * 1e3;
          }
          // the tokens the rows' contexts span (sliding: the last window + the rows)
          const double toks = sliding ? std::min(ctx, 513 + rows - 1) : ctx;
          const double bytes = toks * sh.hkv() * D * sh.kv_elem() * 2 + (kv == k::QsaKv::i8 ? toks * sh.hkv() * 8 : 0);
          const double floor_us = bytes / 604e3;
          std::printf("%-7s %-4s %7d %4d | %6.1f %6.1f %7.1f %6.1f %6.1f | %8.1f %8.1f %6.2f | best%s |%s\n", sliding ? "sliding" : "full",
                      kv == k::QsaKv::i8 ? "int8" : "bf16", ctx, rows, med[0], med[1], med[2], med[3], med[4], med[5], floor_us, med[5] / floor_us, best_s.c_str(),
                      sweep.c_str());
        }
        for (void *ptr : {static_cast<void *>(const_cast<int32_t *>(c.pages)), static_cast<void *>(c.k), static_cast<void *>(c.v), static_cast<void *>(c.ks),
                          static_cast<void *>(c.vs), static_cast<void *>(sc.y), static_cast<void *>(sc.q), static_cast<void *>(sc.part),
                          static_cast<void *>(sc.attn), static_cast<void *>(sc.partial), static_cast<void *>(x), static_cast<void *>(out), static_cast<void *>(pos)})
          if (ptr) fab.release(ptr);
        if (sliding) break;  // a sliding layer's work does not depend on the context
      }
    }
  }
  return 0;
}

namespace {
float c_first_sum(float a, float b) { return a + b; }
}  // namespace

int cmd_moe_flint(const std::string &model, int layer, int iters) {
  const int gu_ks = std::getenv("XEN_MOE_GU_KS") ? std::atoi(std::getenv("XEN_MOE_GU_KS")) : 0;
  const int dn_ks = std::getenv("XEN_MOE_DN_KS") ? std::atoi(std::getenv("XEN_MOE_DN_KS")) : 0;
  const auto mc = m::ModelConfig::load(model);
  const auto cfg = xenithium::arch::flint::Config::parse(mc);
  if (layer < 0 || layer >= cfg.layers || cfg.dense[layer]) {
    std::fprintf(stderr, "moeflint: layer %d is not an MoE layer\n", layer);
    return 2;
  }
  const auto ck = m::Checkpoint::open(model);
  const int H = cfg.hidden, I = cfg.moe_inter, SI = cfg.shared_inter, E = cfg.experts, K = cfg.topk, C = 2;
  const std::string P = "model.layers." + std::to_string(layer) + ".mlp.";
  std::vector<const m::TensorInfo *> ts;
  for (const char *mat : {"switch_mlp.gate_proj", "switch_mlp.up_proj", "switch_mlp.down_proj", "shared_expert.gate_proj", "shared_expert.up_proj",
                          "shared_expert.down_proj"})
    for (const char *suf : {".weight", ".scales", ".biases"}) ts.push_back(&ck.at(P + mat + suf));
  ts.push_back(&ck.at(P + "gate.weight"));
  const auto t0 = std::chrono::steady_clock::now();
  const auto buf = m::bulk_read(ck, ts);
  auto u8 = [&](const std::string &n) { return buf.data(ck.at(P + n)); };
  auto u16 = [&](const std::string &n) { return reinterpret_cast<const uint16_t *>(buf.data(ck.at(P + n))); };
  auto src_of = [&](int e) {
    const size_t gw = static_cast<size_t>(I) * H / 2, gs = static_cast<size_t>(I) * H / 64, dw = static_cast<size_t>(H) * I / 2, ds = static_cast<size_t>(H) * I / 64;
    k::MoeAffSrc s2;
    s2.gate_w = u8("switch_mlp.gate_proj.weight") + e * gw, s2.up_w = u8("switch_mlp.up_proj.weight") + e * gw, s2.down_w = u8("switch_mlp.down_proj.weight") + e * dw;
    s2.gate_s = u16("switch_mlp.gate_proj.scales") + e * gs, s2.gate_b = u16("switch_mlp.gate_proj.biases") + e * gs;
    s2.up_s = u16("switch_mlp.up_proj.scales") + e * gs, s2.up_b = u16("switch_mlp.up_proj.biases") + e * gs;
    s2.down_s = u16("switch_mlp.down_proj.scales") + e * ds, s2.down_b = u16("switch_mlp.down_proj.biases") + e * ds;
    s2.inter = I;
    return s2;
  };
  k::MoeAffSrc ssrc{u8("shared_expert.gate_proj.weight"), u8("shared_expert.up_proj.weight"), u8("shared_expert.down_proj.weight"),
                    u16("shared_expert.gate_proj.scales"), u16("shared_expert.gate_proj.biases"), u16("shared_expert.up_proj.scales"),
                    u16("shared_expert.up_proj.biases"), u16("shared_expert.down_proj.scales"), u16("shared_expert.down_proj.biases"), SI};
  const uint16_t *router = u16("gate.weight");
  const auto loader = xenithium::l0::Loader::open();
  xenithium::Fabric fab(loader, std::vector<size_t>{0, 1});
  struct Card {
    k::MoeAffShape s;
    k::MoeAffWeights w;
    k::MoeAffScratch sc;
    bf16 *x = nullptr, *out = nullptr;
    std::vector<void *> mem;
  };
  std::vector<Card> cards(C);
  std::vector<uint16_t> rp(k::dense_packed_elems(E, H));
  k::pack_dense_bf16(router, E, H, rp.data());
  for (int c = 0; c < C; ++c) {
    Card &cd = cards[c];
    auto &q = fab.compute(c);
    auto alloc = [&](size_t b) {
      void *ptr = fab.alloc(c, std::max<size_t>(b, 4096));
      cd.mem.push_back(ptr);
      return ptr;
    };
    cd.s = k::MoeAffShape{H, I, E, K, SI, C, c};
    cd.w.blob = k::moe_aff_layout(H, cd.s.inter_c());
    cd.w.shared_lay = k::moe_aff_layout(H, cd.s.shared_c());
    auto *pool = static_cast<uint8_t *>(alloc(cd.w.blob.bytes * E));
    std::vector<uint8_t> blob(cd.w.blob.bytes);
    std::vector<const uint8_t *> table(E);
    for (int e = 0; e < E; ++e) {
      k::moe_aff_pack(H, C, c, src_of(e), cd.w.blob, blob.data());
      q.memcpy(pool + e * cd.w.blob.bytes, blob.data(), blob.size()).wait();
      table[e] = pool + e * cd.w.blob.bytes;
    }
    std::vector<uint8_t> sb(cd.w.shared_lay.bytes);
    k::moe_aff_pack(H, C, c, ssrc, cd.w.shared_lay, sb.data());
    auto *dsb = static_cast<uint8_t *>(alloc(sb.size()));
    q.memcpy(dsb, sb.data(), sb.size()).wait();
    cd.w.shared = dsb;
    auto *dt = static_cast<const uint8_t **>(alloc(E * sizeof(void *)));
    q.memcpy(dt, table.data(), E * sizeof(void *)).wait();
    cd.w.blobs = dt;
    auto *dr = static_cast<uint16_t *>(alloc(rp.size() * 2));
    q.memcpy(dr, rp.data(), rp.size() * 2).wait();
    cd.w.router.w = dr, cd.w.router.kind = k::DenseKind::bf16, cd.w.router.n = E, cd.w.router.k = H;
    auto *tab = static_cast<float *>(alloc(k::kSiluTabSize * 4));
    k::swiglu_table(q, tab).wait_and_throw();
    cd.w.silu_tab = tab;
    cd.sc.logits = static_cast<float *>(alloc(8ull * E * 4));
    cd.sc.route = static_cast<k::MoeRoute *>(alloc(sizeof(k::MoeRoute)));
    cd.sc.h = static_cast<bf16 *>(alloc(k::moe_aff_h_elems(cd.s) * 2));
    cd.sc.y = static_cast<float *>(alloc(k::moe_aff_y_elems(cd.s) * 4));
    cd.sc.partial = static_cast<float *>(alloc(8ull * H * 4));
    cd.x = static_cast<bf16 *>(alloc(64ull * 8 * H * 2));
    cd.out = static_cast<bf16 *>(alloc(8ull * H * 2));
  }
  const double load_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  std::printf("moeflint: layer %d, %d experts x inter %d (TP2: %d per card) + shared %d; expert blob %.1f KiB per card, loaded in %.1f s\n", layer, E, I,
              I / C, SI, cards[0].w.blob.bytes / 1024.0, load_s);
  // inputs: 64 sets of 8 rows of N(0, 1) bf16
  std::mt19937 rng(17);
  std::normal_distribution<float> nd(0.f, 1.f);
  std::vector<uint16_t> xs(64ull * 8 * H);
  for (auto &v : xs) {
    float f = nd(rng);
    uint32_t u;
    std::memcpy(&u, &f, 4);
    u += 0x7fffu + ((u >> 16) & 1u);
    v = static_cast<uint16_t>(u >> 16);
  }
  for (int c = 0; c < C; ++c) fab.compute(c).memcpy(cards[c].x, xs.data(), xs.size() * 2).wait();
  auto bf = [](uint16_t b) {
    const uint32_t u = static_cast<uint32_t>(b) << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
  };
  // correctness
  std::vector<float> routerf(static_cast<size_t>(E) * H);
  for (size_t i = 0; i < routerf.size(); ++i) routerf[i] = bf(router[i]);
  const k::MoeAffRefExpert rsh = k::moe_aff_dequant(H, ssrc);
  int fails = 0;
  for (const int rows : {1, 4, 8}) {
    std::vector<float> part[2];
    k::MoeRoute rt;
    for (int c = 0; c < C; ++c) {
      auto &q = fab.compute(c);
      k::moe_aff_decode(q, cards[c].s, rows, cards[c].x, cards[c].w, cards[c].sc, nullptr, 0, cards[c].out);
      q.wait_and_throw();
      part[c].resize(static_cast<size_t>(rows) * H);
      q.memcpy(part[c].data(), cards[c].sc.partial, part[c].size() * 4).wait();
      if (c == 0) q.memcpy(&rt, cards[0].sc.route, sizeof rt).wait();
    }
    double worst = 0;
    size_t route_bad = 0;
    for (int r = 0; r < rows; ++r) {
      std::vector<float> xf(H);
      for (int i = 0; i < H; ++i) xf[i] = bf(xs[static_cast<size_t>(r) * H + i]);
      int32_t ids[k::kMoeMaxTopk];
      float wts[k::kMoeMaxTopk];
      std::vector<double> lg(E);
      k::moe_aff_route_ref(xf.data(), routerf.data(), H, E, K, ids, wts, lg.data());
      std::vector<int> a(rt.ids[r], rt.ids[r] + K), b(ids, ids + K);
      std::sort(a.begin(), a.end()), std::sort(b.begin(), b.end());
      if (a != b) {
        double kth = 1e9;
        for (int t = 0; t < K; ++t) kth = std::min(kth, 1.0 / (1.0 + std::exp(-lg[ids[t]])));
        for (int e : a)
          if (std::find(b.begin(), b.end(), e) == b.end() && std::fabs(1.0 / (1.0 + std::exp(-lg[e])) - kth) > 1e-6) ++route_bad;
      }
      std::vector<k::MoeAffRefExpert> ex;
      ex.reserve(K);
      std::vector<const k::MoeAffRefExpert *> pe;
      for (int t = 0; t < K; ++t) ex.push_back(k::moe_aff_dequant(H, src_of(rt.ids[r][t])));
      for (auto &e : ex) pe.push_back(&e);
      std::vector<double> ref(H);
      k::moe_aff_ref(xf.data(), H, K, rt.ids[r], rt.topw[r], pe.data(), rsh, ref.data());
      double ss = 0;
      for (double v : ref) ss += v * v;
      const double rms = std::sqrt(ss / H);
      for (int i = 0; i < H; ++i) {
        const size_t j = static_cast<size_t>(r) * H + i;
        const double e = std::fabs(static_cast<double>(part[0][j]) + part[1][j] - ref[i]) / std::max(std::fabs(ref[i]), rms);
        worst = std::isfinite(e) ? std::max(worst, e) : HUGE_VAL;
      }
    }
    const bool ok = route_bad == 0 && worst < 2.0 / 128;
    fails += !ok;
    std::printf("  %d row(s): route %zu bad, out (two cards' partials, h computed) worst %.2e of max(|ref|, rms) vs the reference [%s]\n", rows, route_bad, worst,
                ok ? "PASS" : "FAIL");
  }
  // the wired two-card step: both cards' out equal, == bf16 of the wire-less partials' sum (fp32 wire: bit-equal), and
  // timing of the full step on card 0 (both cards stepping)
  k::PairWire wire[2];
  for (int c = 0; c < C; ++c) {
    wire[c].inbox = static_cast<uint32_t *>(fab.alloc_mailbox(c, k::pair_wire_words(H) * 4));
    wire[c].error = static_cast<uint32_t *>(fab.alloc(c, 4096));
    cards[c].mem.push_back(wire[c].inbox), cards[c].mem.push_back(wire[c].error);
  }
  for (int c = 0; c < C; ++c) {
    wire[c].peer_inbox = wire[1 - c].inbox;
    k::pair_wire_init(fab.compute(c), H, wire[c].inbox).wait();
    fab.compute(c).memset(wire[c].error, 0, 4).wait();
  }
  int parity = 0;
  auto step2 = [&](int rows, int set, bool b16, k::MoeAffEvents *ev0) {
    for (int c = 0; c < C; ++c) wire[c].bf16 = b16;
    // one submitting thread per card (the server's shape): a single thread's card-1 launches trail card 0's by the
    // submit time of five launches, which card 0's exchange would wait out
    std::thread th([&] {
      k::moe_aff_decode(fab.compute(1), cards[1].s, rows, cards[1].x + static_cast<size_t>(set) * 8 * H, cards[1].w, cards[1].sc, &wire[1], parity, cards[1].out,
                        gu_ks, dn_ks);
    });
    const auto e = k::moe_aff_decode(fab.compute(0), cards[0].s, rows, cards[0].x + static_cast<size_t>(set) * 8 * H, cards[0].w, cards[0].sc, &wire[0], parity,
                                     cards[0].out, gu_ks, dn_ks);
    th.join();
    if (ev0) *ev0 = e;
    for (int c = 0; c < C; ++c) fab.compute(c).wait_and_throw();
    parity ^= 1;
  };
  for (const bool b16 : {false, true})
    for (const int rows : {1, 4, 8}) {
      step2(rows, 0, b16, nullptr);
      std::vector<uint16_t> o[2];
      for (int c = 0; c < C; ++c) {
        o[c].resize(static_cast<size_t>(rows) * H);
        fab.compute(c).memcpy(o[c].data(), cards[c].out, o[c].size() * 2).wait();
      }
      std::vector<float> p[2];
      for (int c = 0; c < C; ++c) {
        k::moe_aff_decode(fab.compute(c), cards[c].s, rows, cards[c].x, cards[c].w, cards[c].sc, nullptr, 0, nullptr, gu_ks, dn_ks);
        fab.compute(c).wait_and_throw();
        p[c].resize(o[0].size());
        fab.compute(c).memcpy(p[c].data(), cards[c].sc.partial, p[c].size() * 4).wait();
      }
      size_t diff = 0, nonf = 0;
      for (size_t i = 0; i < o[0].size(); ++i) {
        float want;
        if (b16) {
          const float a = static_cast<float>(bf16(p[0][i])), b = static_cast<float>(bf16(p[1][i]));
          want = static_cast<float>(bf16(c_first_sum(a, b)));
        } else {
          want = static_cast<float>(bf16(p[0][i] + p[1][i]));
        }
        diff += o[0][i] != o[1][i] || bf(o[0][i]) != want;
        nonf += !std::isfinite(bf(o[0][i]));
      }
      uint32_t err[2] = {0, 0};
      for (int c = 0; c < C; ++c) fab.compute(c).memcpy(&err[c], wire[c].error, 4).wait();
      const bool ok = diff == 0 && nonf == 0 && !err[0] && !err[1];
      fails += !ok;
      std::printf("  wired %s %d row(s): both cards' out equal and == the partials' sum: %zu differ, %zu non-finite, wire errors %u / %u [%s]\n",
                  b16 ? "bf16" : "fp32", rows, diff, nonf, err[0], err[1], ok ? "PASS" : "FAIL");
    }
  std::printf("wired step timing, card 0 (us, median of %d; bf16 wire): router start -> combine end\n", iters);
  for (const int rows : {1, 2, 4, 8}) {
    std::vector<double> tt, ts[5], gap;
    for (int it = 0; it < iters + 20; ++it) {
      k::MoeAffEvents e;
      step2(rows, it % 64, true, &e);
      if (it < 20) continue;
      auto st = [](const sycl::event &v) { return v.get_profiling_info<sycl::info::event_profiling::command_start>(); };
      auto en = [](const sycl::event &v) { return v.get_profiling_info<sycl::info::event_profiling::command_end>(); };
      tt.push_back(static_cast<double>(en(e.combine) - st(e.router)) * 1e-3);
      const sycl::event *seq[5] = {&e.router, &e.route, &e.gate_up, &e.down, &e.combine};
      double busy = 0;
      for (int i = 0; i < 5; ++i) {
        const double d = static_cast<double>(en(*seq[i]) - st(*seq[i])) * 1e-3;
        ts[i].push_back(d);
        busy += d;
      }
      gap.push_back(tt.back() - busy);
    }
    auto med = [](std::vector<double> &v) {
      std::sort(v.begin(), v.end());
      return v[v.size() / 2];
    };
    std::printf("  %d row(s): step %.1f = router %.1f route %.1f gate_up %.1f down %.1f combine + exchange %.1f, gaps %.1f\n", rows, med(tt), med(ts[0]), med(ts[1]),
                med(ts[2]), med(ts[3]), med(ts[4]), med(gap));
  }
  // timing (card 0)
  auto &q = fab.compute(0);
  Card &cd = cards[0];
  std::printf("timing, card 0 alone (us, median of %d steps over 64 rotating input sets; combine = the card's partial, no exchange)\n", iters);
  std::printf("%4s %7s | %6s %6s %7s %6s %7s | %7s %7s %6s\n", "rows", "experts", "router", "route", "gate_up", "down", "combine", "total", "floor", "x");
  for (const int rows : {1, 2, 4, 8}) {
    std::vector<double> t[6], uniq;
    if (rows == 1)  // clocks up: 2000 8-row steps before the first block
      for (int it = 0; it < 2000; ++it) k::moe_aff_decode(q, cd.s, 8, cd.x + static_cast<size_t>(it % 64) * 8 * H, cd.w, cd.sc, nullptr, 0, nullptr, gu_ks, dn_ks);
    const int warm = 20;
    for (int it = 0; it < iters + warm; ++it) {
      const bf16 *x = cd.x + static_cast<size_t>(it % 64) * 8 * H;
      const auto e = k::moe_aff_decode(q, cd.s, rows, x, cd.w, cd.sc, nullptr, 0, nullptr, gu_ks, dn_ks);
      q.wait_and_throw();
      if (it < warm) continue;
      int32_t cnt = 0;
      q.memcpy(&cnt, &cd.sc.route->count, 4).wait();
      uniq.push_back(cnt);
      t[0].push_back(ev_ms(e.router)), t[1].push_back(ev_ms(e.route)), t[2].push_back(ev_ms(e.gate_up)), t[3].push_back(ev_ms(e.down));
      t[4].push_back(ev_ms(e.combine));
      t[5].push_back(static_cast<double>(e.combine.get_profiling_info<sycl::info::event_profiling::command_end>() -
                                         e.router.get_profiling_info<sycl::info::event_profiling::command_start>()) *
                     1e-6);
    }
    double med[6];
    for (int i = 0; i < 6; ++i) {
      std::sort(t[i].begin(), t[i].end());
      med[i] = t[i][t[i].size() / 2] * 1e3;
    }
    double u = 0;
    for (double v : uniq) u += v;
    u /= static_cast<double>(uniq.size());
    const double bytes = u * cd.w.blob.bytes + cd.w.shared_lay.bytes + static_cast<double>(E) * H * 2;
    const double fl = bytes / 604e3;
    std::printf("%4d %7.1f | %6.1f %6.1f %7.1f %6.1f %7.1f | %7.1f %7.1f %6.2f\n", rows, u, med[0], med[1], med[2], med[3], med[4], med[5], fl, med[5] / fl);
  }
  for (int c = 0; c < C; ++c)
    for (void *ptr : cards[c].mem) fab.release(ptr);
  return fails ? 1 : 0;
}
