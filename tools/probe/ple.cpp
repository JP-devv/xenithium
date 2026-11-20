// xenithium-probe ple: the PLE block on real weights and the real n-gram table, both cards.
// xenithium-probe ple --model DIR [--iters I] [--threads T]
// Correctness, on the reference sequence
// with eos inside, the reference layer's residual input and output
// weights): ids -> rows read from the drive -> dequantised -> the block on each card, steps of
// 1-4 rows. Checked: the dequantised rows == the reference's; the gate / conv teacher-forced at the
// device's kv against the host reference (int8 weights as the device's GEMM gives them); both
// cards bit-identical; the residual against the reference layer (bf16 weights: what int8 costs).
// Then a real-data run: 64 tokens, residual = the token's embedding in every stream (the model's
// input), the same teacher-forced checks.
// Timing: the block per step on the cards (chained, no host wait) against its read floor, and the
// table read per step from the drive, cold (page cache dropped) and warm, by reader threads.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include "xenithium/arch/qwen38/arch.hpp"
#include "xenithium/arch/qwen38/ple.hpp"
#include "xenithium/arch/qwen38/ple_table.hpp"
#include "xenithium/kernels/pack.hpp"
#include "xenithium/kernels/embed/ple.hpp"
#include "xenithium/kernels/embed/ple_ref.hpp"
#include "xenithium/load/dense_store.hpp"
#include "xenithium/model/arch.hpp"
#include "xenithium/model/checkpoint.hpp"
#include "xenithium/model/config.hpp"
#include "xenithium/rt/fabric.hpp"

namespace {
namespace m = xenithium::model;
namespace k = xenithium::kernels;
namespace q38 = xenithium::arch::qwen38;
using bf16 = sycl::ext::oneapi::bfloat16;

double ev_us(const sycl::event &e) {
  const auto a = e.get_profiling_info<sycl::info::event_profiling::command_start>();
  const auto b = e.get_profiling_info<sycl::info::event_profiling::command_end>();
  return static_cast<double>(b - a) * 1e-3;
}
double median(std::vector<double> v) {
  if (v.empty()) return 0;
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}
double pct(std::vector<double> v, double p) {
  if (v.empty()) return 0;
  std::sort(v.begin(), v.end());
  return v[std::min(v.size() - 1, static_cast<size_t>(p * v.size()))];
}
float bfr(float v) { return k::bf16_value(k::bf16_bits(v)); }

struct BoundCmp {
  double scaled = 0;
  size_t off = 0, n = 0;
  void add_row(const float *dev, const float *ref, int len) {
    double ss = 0;
    for (int i = 0; i < len; ++i) ss += static_cast<double>(ref[i]) * ref[i];
    const double rms = std::sqrt(ss / len);
    for (int i = 0; i < len; ++i) {
      ++n;
      off += dev[i] != ref[i];
      scaled = std::max(scaled, std::fabs(dev[i] - ref[i]) / std::max({static_cast<double>(std::fabs(ref[i])), rms, 1e-30}));
    }
  }
  std::string text() const {
    char b[128];
    std::snprintf(b, sizeof b, "worst %.2e of max(|ref|, rms), %zu / %zu differ (%.3f %%)", scaled, off, n, 100.0 * off / std::max<size_t>(1, n));
    return b;
  }
  bool ok(double lim, double pct_off) const { return scaled <= lim && off <= pct_off / 100.0 * n; }
};

std::vector<float> vals(const uint16_t *b, size_t n) {
  std::vector<float> v(n);
  for (size_t i = 0; i < n; ++i) v[i] = k::bf16_value(b[i]);
  return v;
}

struct Dev {
  sycl::queue *q = nullptr;
  k::PleState st;
  k::PleScratch sc;
  int32_t *pos = nullptr;
  bf16 *res = nullptr;
  std::vector<void *> mem;
};
Dev make_dev(xenithium::Fabric &fab, int c, const k::PleShape &s) {
  Dev d;
  d.q = &fab.compute(c);
  auto alloc = [&](size_t bytes) {
    void *p = fab.alloc(c, bytes);
    d.q->memset(p, 0, bytes).wait();
    d.mem.push_back(p);
    return p;
  };
  d.st.ring = static_cast<bf16 *>(alloc(k::ple_ring_elems(s) * 2));
  d.sc.emb = static_cast<bf16 *>(alloc(static_cast<size_t>(k::kDenseRows) * s.emb * 2));
  d.sc.kv = static_cast<float *>(alloc(static_cast<size_t>(k::kDenseRows) * s.kv_n() * 4));
  d.sc.gated = static_cast<bf16 *>(alloc(static_cast<size_t>(k::kDenseRows) * s.width() * 2));
  d.pos = static_cast<int32_t *>(alloc(64));
  d.res = static_cast<bf16 *>(alloc(static_cast<size_t>(k::kDenseRows) * s.width() * 2));
  return d;
}

struct Checks {
  BoundCmp gated, x, out, ref_out;
  bool cards_equal = true, emb_equal = true;
};

// Runs tokens [0, T) in steps of `plan` rows on both cards, residual inputs res_in [T][C];
// teacher-forced checks; ref_out (optional) = the reference layer's residual output.
void run_sequence(xenithium::Fabric &fab, Dev d[2], const q38::PleSet &ps, q38::PleTable &tab, const q38::Config &cfg, const std::vector<int64_t> &toks,
                  const std::vector<uint16_t> &res_in, const std::vector<uint16_t> *ref_out, const std::vector<uint16_t> *ref_emb, Checks &ck,
                  uint16_t *pinned) {
  const auto &s = ps.shape;
  const int C = s.width(), H = s.hidden, E = s.emb, KV = s.kv_n(), T = static_cast<int>(toks.size());
  const int plan[] = {1, 2, 3, 4, 4, 1, 3, 2, 4, 4};
  for (int c = 0; c < 2; ++c) d[c].q->memset(d[c].st.ring, 0x7f, k::ple_ring_elems(s) * 2).wait();
  // host copies of the norm / conv weights (bf16 values) for the reference
  auto host_bf = [&](const bf16 *p, size_t n) {
    std::vector<uint16_t> b(n);
    d[0].q->memcpy(b.data(), p, n * 2).wait();
    return vals(b.data(), n);
  };
  const auto nk = host_bf(ps.w[0].nk, C), nq = host_bf(ps.w[0].nq, C), ncw = host_bf(ps.w[0].ncw, C), cw = host_bf(ps.w[0].conv, static_cast<size_t>(C) * 4);
  k::PleContext ctx;
  ctx.reset(tab.layout());
  std::vector<std::vector<float>> xh;
  int p = 0;
  for (int si = 0; p < T; ++si) {
    const int R = std::min(plan[si % 10], T - p);
    std::vector<int64_t> ids(static_cast<size_t>(R) * tab.heads());
    tab.ids(&toks[p], R, ctx, ids.data());
    std::vector<uint8_t> codes(ids.size() * tab.head_dim());
    tab.read(ids.data(), R, codes.data());
    tab.dequant(codes.data(), R, pinned);
    if (ref_emb) ck.emb_equal &= std::equal(pinned, pinned + static_cast<size_t>(R) * E, ref_emb->begin() + static_cast<size_t>(p) * E);
    std::vector<uint16_t> resv(static_cast<size_t>(k::kDenseRows) * C, 0), out[2];
    std::copy(res_in.begin() + static_cast<size_t>(p) * C, res_in.begin() + static_cast<size_t>(p + R) * C, resv.begin());
    std::vector<float> kvd;
    std::vector<uint16_t> gd, ring;
    for (int c = 0; c < 2; ++c) {
      d[c].q->memcpy(d[c].pos, &p, 4).wait();
      d[c].q->memcpy(d[c].res, resv.data(), resv.size() * 2).wait();
      k::ple_decode(*d[c].q, s, R, reinterpret_cast<const bf16 *>(pinned), ps.w[c], d[c].st, d[c].pos, d[c].sc, d[c].res).conv.wait();
      out[c].resize(resv.size());
      d[c].q->memcpy(out[c].data(), d[c].res, out[c].size() * 2).wait();
    }
    ck.cards_equal &= out[0] == out[1];
    kvd.resize(static_cast<size_t>(k::kDenseRows) * KV);
    gd.resize(static_cast<size_t>(k::kDenseRows) * C);
    ring.resize(k::ple_ring_elems(s));
    d[0].q->memcpy(kvd.data(), d[0].sc.kv, kvd.size() * 4).wait();
    d[0].q->memcpy(gd.data(), d[0].sc.gated, gd.size() * 2).wait();
    d[0].q->memcpy(ring.data(), d[0].st.ring, ring.size() * 2).wait();
    for (int r = 0; r < R; ++r) {
      const int t = p + r;
      std::vector<float> key(C), val(H), g(C), x(C);
      for (int i = 0; i < C; ++i) key[i] = bfr(kvd[static_cast<size_t>(r) * KV + i]);
      for (int i = 0; i < H; ++i) val[i] = bfr(kvd[static_cast<size_t>(r) * KV + C + i]);
      const auto rv = vals(resv.data() + static_cast<size_t>(r) * C, C);
      k::ple_ref_gate(H, s.streams, key.data(), val.data(), rv.data(), nk.data(), nq.data(), ncw.data(), s.eps, g.data(), x.data());
      const auto gdev = vals(gd.data() + static_cast<size_t>(r) * C, C), xdev = vals(ring.data() + static_cast<size_t>(t % k::kPleRing) * C, C);
      ck.gated.add_row(gdev.data(), g.data(), C);
      ck.x.add_row(xdev.data(), x.data(), C);
      if (static_cast<int>(xh.size()) <= t) xh.resize(t + 1);
      xh[t] = xdev;
      const float *taps[k::kPleTaps];
      for (int j = 0; j < k::kPleTaps; ++j) {
        const int q = t - k::kPleDil * (k::kPleTaps - 1 - j);
        taps[j] = q >= 0 ? xh[q].data() : nullptr;
      }
      std::vector<float> o = rv;
      k::ple_ref_conv(C, taps, cw.data(), gdev.data(), o.data());
      const auto odev = vals(out[0].data() + static_cast<size_t>(r) * C, C);
      ck.out.add_row(odev.data(), o.data(), C);
      if (ref_out) {
        const auto ro = vals(ref_out->data() + static_cast<size_t>(t) * C, C);
        ck.ref_out.add_row(odev.data(), ro.data(), C);
      }
    }
    p += R;
  }
  (void)cfg;
}

int g_fail = 0;
void report(bool ok, const std::string &what) {
  std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
  if (!ok) ++g_fail;
}

}  // namespace

int cmd_ple_run(const std::string &model, int iters, int threads) {
  const auto mc = m::ModelConfig::load(model);
  const auto arch = m::open_arch(mc);
  const q38::Config cfg = q38::Config::parse(mc);
  const m::Checkpoint ck = m::Checkpoint::open(model);
  const std::string P = q38::kPrefix;
  std::vector<m::DenseSpec> plan;
  for (const auto &sp : arch->dense_plan(ck, m::DeployParams{}))
    if (sp.comp == m::Component::ple || sp.name == P + "embed_tokens.weight") plan.push_back(sp);
  const auto loader = xenithium::l0::Loader::open();
  xenithium::Fabric fab(loader, {0, 1});
  xenithium::load::DenseStore dense(fab);
  dense.load(ck, plan, threads);
  const auto &cb = dense.stats().card_bytes[0];
  std::printf("ple: %zu dense tensors loaded in %.2f s (PLE block %.1f MiB per card, replicated)\n", plan.size(), dense.stats().total_s,
              (cb.count(m::Component::ple) ? cb.at(m::Component::ple) : 0) / 1048576.0);
  const auto t0 = std::chrono::steady_clock::now();
  q38::PleTable tab(ck, cfg, 32);
  std::printf("ple: table %lld rows x %d FP8 in %zu shards; layout == checkpoint buffers; opened in %.1f ms\n",
              static_cast<long long>(tab.layout().padded_rows), tab.head_dim(), static_cast<size_t>(cfg.ple_split_parts),
              std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
  const q38::PleSet ps = q38::PleSet::from(dense, cfg, 2);
  const auto &s = ps.shape;
  const int C = s.width(), H = s.hidden, E = s.emb;
  Dev d[2] = {make_dev(fab, 0, s), make_dev(fab, 1, s)};
  auto *pinned = static_cast<uint16_t *>(fab.alloc_host(static_cast<size_t>(k::kDenseRows) * E * 2));

  // ---- the reference sequence -------------------------------------------------------------------
  {
    const char *rp = std::getenv("XENITHIUM_PLE_REF");
    const std::string path = rp ? rp : ".spike/r08/ref/ple_ref.bin";
    std::ifstream f(path, std::ios::binary);
    if (!f) {
      std::printf("[SKIP] the reference sequence: no dump at %s\n", path.c_str());
    } else {
      int32_t hdr[8];
      f.read(reinterpret_cast<char *>(hdr), sizeof hdr);
      const int T = hdr[0], NH = hdr[1];
      std::vector<int64_t> toks(T);
      f.read(reinterpret_cast<char *>(toks.data()), T * 8);
      f.seekg(static_cast<std::streamoff>(T) * NH * 8 + static_cast<std::streamoff>(T) * E, std::ios::cur);
      auto rd = [&](size_t n) {
        std::vector<uint16_t> b(n);
        f.read(reinterpret_cast<char *>(b.data()), static_cast<std::streamsize>(n * 2));
        return b;
      };
      const auto emb = rd(static_cast<size_t>(T) * E), hidden = rd(static_cast<size_t>(T) * C);
      rd(static_cast<size_t>(T) * C);
      rd(static_cast<size_t>(T) * H);
      rd(static_cast<size_t>(T) * C);
      const auto out = rd(static_cast<size_t>(T) * C);
      Checks c;
      run_sequence(fab, d, ps, tab, cfg, toks, hidden, &out, &emb, c, pinned);
      report(c.emb_equal, "reference sequence (40 tokens, eos at 7, 8, 20): dequantised rows from the drive == the reference's");
      report(c.gated.ok(1.0 / 64, 0.5), "gated vs the host reference from the device's kv: " + c.gated.text());
      report(c.x.ok(1.0 / 64, 0.5), "conv input vs the reference: " + c.x.text());
      report(c.out.ok(1.0 / 64, 0.5), "residual vs conv + add from the device's ring and gated: " + c.out.text());
      report(c.cards_equal, "both cards bit-identical");
      std::printf("[INFO] residual vs the reference layer (bf16 weights; ours int8 kv): %s\n", c.ref_out.text().c_str());
    }
  }
  // ---- real data: embeddings as the residual ------------------------------------------------------
  {
    const auto *embt = reinterpret_cast<const uint16_t *>(dense.at(0, P + "embed_tokens.weight").ptr);
    std::mt19937_64 rng(808);
    const int T = 64;
    std::vector<int64_t> toks(T);
    for (auto &t : toks) t = static_cast<int64_t>(rng() % cfg.vocab);
    toks[30] = cfg.eos[0];
    std::vector<uint16_t> res(static_cast<size_t>(T) * C);
    for (int t = 0; t < T; ++t)
      for (int st = 0; st < s.streams; ++st)
        std::memcpy(&res[static_cast<size_t>(t) * C + static_cast<size_t>(st) * H], embt + static_cast<size_t>(toks[t]) * H, static_cast<size_t>(H) * 2);
    Checks c;
    run_sequence(fab, d, ps, tab, cfg, toks, res, nullptr, nullptr, c, pinned);
    report(c.gated.ok(1.0 / 64, 0.5), "real data (64 tokens, residual = embedding x 4): gated vs the reference: " + c.gated.text());
    report(c.x.ok(1.0 / 64, 0.5), "real data: conv input vs the reference: " + c.x.text());
    report(c.out.ok(1.0 / 64, 0.5), "real data: residual vs conv + add: " + c.out.text());
    report(c.cards_equal, "real data: both cards bit-identical");
  }

  // ---- timing: the block on the cards -------------------------------------------------------------
  {
    const double kv_mb = ps.w[0].kv.bytes() / 1e6;
    const double floor_mb = kv_mb + (3.0 * C * 2 + C * 4.0 * 2) / 1e6;  // + norms, conv weights
    const double floor_us = floor_mb * 1e6 / 605.9e9 * 1e6;
    // Cold L2: the kv weights rotate over 4 distinct copies (4 x 33 MB > the 24 MiB L2), as a step
    // whose other layers evict them; one copy alone ran below the read floor (L2 hits).
    constexpr int kCopies = 4;
    std::vector<k::PleWeights> wc[2];
    std::vector<void *> extra;
    for (int c = 0; c < 2; ++c) {
      for (int i = 0; i < kCopies; ++i) {
        k::PleWeights w = ps.w[c];
        if (i) {
          const size_t wb = ps.w[c].kv.bytes() - static_cast<size_t>(ps.w[c].kv.n) * 4;
          void *pw = fab.alloc(c, wb);
          void *psc = fab.alloc(c, static_cast<size_t>(ps.w[c].kv.n) * 4);
          d[c].q->memcpy(pw, ps.w[c].kv.w, wb).wait();
          d[c].q->memcpy(psc, ps.w[c].kv.scale, static_cast<size_t>(ps.w[c].kv.n) * 4).wait();
          w.kv.w = pw;
          w.kv.scale = static_cast<const float *>(psc);
          extra.push_back(pw);
          extra.push_back(psc);
        }
        wc[c].push_back(w);
      }
    }
    std::printf("\nblock per step, chained (%d iterations after 100 ms warm-up, kv weights rotated over %d copies), both cards; floor %.2f MB = %.1f us at 605.9 GB/s\n",
                iters, kCopies, floor_mb, floor_us);
    std::printf("%-5s %8s %8s %8s %8s %10s %8s\n", "rows", "upload", "kv", "gate", "conv", "span (us)", "x floor");
    for (int R : {1, 4}) {
      std::vector<double> up, kv, gt, cv, span;
      int32_t pos0 = 100;
      for (int c = 0; c < 2; ++c) d[c].q->memcpy(d[c].pos, &pos0, 4).wait();
      int it_n = 0;
      auto once = [&](bool rec) {
        k::PleEvents e[2];
        const int cp = it_n++ % kCopies;
        for (int c = 0; c < 2; ++c) e[c] = k::ple_decode(*d[c].q, s, R, reinterpret_cast<const bf16 *>(pinned), wc[c][cp], d[c].st, d[c].pos, d[c].sc, d[c].res);
        if (rec) {
          up.push_back(ev_us(e[0].up));
          kv.push_back(ev_us(e[0].kv));
          gt.push_back(ev_us(e[0].gate));
          cv.push_back(ev_us(e[0].conv));
          const auto a = e[0].up.get_profiling_info<sycl::info::event_profiling::command_start>();
          const auto b = e[0].conv.get_profiling_info<sycl::info::event_profiling::command_end>();
          span.push_back(static_cast<double>(b - a) * 1e-3);
        }
        return e[1].conv;
      };
      const auto w0 = std::chrono::steady_clock::now();
      while (std::chrono::duration<double>(std::chrono::steady_clock::now() - w0).count() < 0.1) once(false).wait();
      for (int i = 0; i < iters; ++i) {
        once(true);
        if (i % 16 == 15) d[0].q->wait(), d[1].q->wait();
      }
      d[0].q->wait();
      d[1].q->wait();
      std::printf("%-5d %8.1f %8.1f %8.1f %8.1f %10.1f %8.2f\n", R, median(up), median(kv), median(gt), median(cv), median(span), median(span) / floor_us);
    }
    for (auto *p : extra) fab.release(p);
  }

  // ---- timing: the table read per step ----------------------------------------------------------
  {
    std::printf("\ntable read per step (host, pread through the page cache), %d steps each\n", iters);
    std::printf("%-8s %-7s %-6s %10s %10s %10s\n", "threads", "tokens", "cache", "median us", "p90 us", "max us");
    std::mt19937_64 rng(99);
    for (int th : {16, 32, 64}) {
      q38::PleTable t2(ck, cfg, th);
      for (int R : {1, 4}) {
        for (int cold = 1; cold >= 0; --cold) {
          std::vector<double> us;
          std::vector<int64_t> warm_ids;
          for (int i = 0; i < iters; ++i) {
            std::vector<int64_t> toks(R);
            for (auto &x : toks) x = static_cast<int64_t>(rng() % cfg.vocab);
            std::vector<int64_t> ids(static_cast<size_t>(R) * t2.heads());
            k::PleContext cx;
            cx.reset(t2.layout());
            if (cold || warm_ids.empty()) {
              t2.ids(toks.data(), R, cx, ids.data());
              if (!cold) warm_ids = ids;
            } else {
              ids = warm_ids;
            }
            std::vector<uint8_t> codes(ids.size() * t2.head_dim());
            if (cold) t2.reader().drop_cache();
            const auto a = std::chrono::steady_clock::now();
            t2.read(ids.data(), R, codes.data());
            us.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - a).count());
          }
          std::printf("%-8d %-7d %-6s %10.1f %10.1f %10.1f\n", th, R, cold ? "cold" : "warm", median(us), pct(us, 0.9), pct(us, 1.0));
        }
      }
    }
  }
  fab.release(pinned);
  for (auto &x : d)
    for (auto *p : x.mem) fab.release(p);
  std::printf("%s\n", g_fail ? "ple: FAILED" : "ple: all checks passed");
  return g_fail ? 1 : 0;
}
