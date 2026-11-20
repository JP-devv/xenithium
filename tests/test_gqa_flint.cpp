// Gate for the GQA decode at arch flint's shape (kernels/attention/gqa.hpp: head_dim 128, 8 q heads per kv head, no output gate,
// plain q / k RMS norms (w x norm(x)), learned sinks). Two layer kinds:
// sliding: NeoX rope over all 128 dims (theta 1e4), window 513 (the row's token and the 512 before), KV in a ring of
// 10 pages per slot (16 for the many-row calls);
// full: no rope, the whole context, linear pages.
// One card with the model's 32 q / 4 kv heads, hidden 512, synthetic W4 projections; the KV pool holds random history.
// Checks, teacher-forced at the device's own projections:
// q vs bf16(rope(bf16(norm(bf16(y)) w))) from the device's fp32 projections: <= 1 bf16 ulp, <= 1 % off
// K / V the rows the step wrote vs the same from y (<= 1 code / ulp), at their ring slots
// attn vs (sum_t p_t v_t) / (sum_t p_t + exp(sink - m)) over the row's window as stored, in double
// and bit identities: attention groups of 1 / 2 / 4 / 8 rows at 128 / 256 GRF == the default; R rows at P == R one-row
// steps; run-to-run; the many-row batch body (gqa_rows attention 0) and the fa body (attention 3: window, ring, sinks;
// every rows-per-group the same bits; P hi only) at chunk edges vs the same reference.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "xenithium/kernels/attention/gqa.hpp"
#include "xenithium/kernels/pack.hpp"
#include "xenithium/kernels/attention/qsa_ref.hpp"
#include "xenithium/kernels/rows.hpp"
#include "xenithium/rt/device.hpp"

namespace {
namespace k = xenithium::kernels;
using bf16 = sycl::ext::oneapi::bfloat16;
using MK = xenithium::l0::MemKind;

int g_failures = 0;
void expect(bool ok, const std::string &what) {
  std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
  if (!ok) ++g_failures;
}
float bfr(float v) { return k::bf16_value(k::bf16_bits(v)); }
float ulp_bf16(float v) {
  const float a = std::fabs(v);
  if (a < 1e-30f) return 1e-30f;
  return std::ldexp(1.f, std::ilogb(a) - 7);
}

struct Ctx {
  xenithium::Device &dev;
  sycl::queue &q;
  std::vector<void *> mem;
  template <class T>
  T *alloc(size_t n) {
    T *p = dev.alloc<T>(std::max<size_t>(n, 4096 / sizeof(T)), MK::Device);
    mem.push_back(p);
    return p;
  }
  void clear() {
    for (void *p : mem) dev.release(p);
    mem.clear();
  }
};

k::W4Mat make_w4(Ctx &cx, int n, int kk, float sd, std::mt19937 &rng) {
  std::vector<uint32_t> qw(static_cast<size_t>(kk / 8) * n);
  for (auto &v : qw) v = static_cast<uint32_t>(rng());
  std::vector<sycl::half> sc(static_cast<size_t>(kk / 128) * n);
  std::uniform_real_distribution<float> us(0.5f, 1.5f);
  for (auto &v : sc) v = sycl::half(us(rng) * sd / 4.6f);
  std::vector<uint8_t> blocks(static_cast<size_t>(n) * kk / 2);
  k::pack_w4_gptq(qw.data(), n, kk, blocks.data());
  auto *wb = cx.alloc<uint8_t>(blocks.size());
  auto *sb = cx.alloc<sycl::half>(sc.size());
  cx.q.memcpy(wb, blocks.data(), blocks.size());
  cx.q.memcpy(sb, sc.data(), sc.size() * 2).wait();
  k::W4Mat m;
  m.w = wb, m.scales = sb, m.n = n, m.k = kk;
  return m;
}

struct Rig {
  k::GqaShape s;
  k::GqaWeights w;
  k::GqaCache c;
  k::GqaScratch sc;
  int32_t *pos = nullptr;
  bf16 *rope = nullptr, *x = nullptr, *out = nullptr;
  std::vector<uint16_t> rope_h, qn_h, kn_h, sink_h;
  int cap = 0, phys = 0;  // positions, physical pages
};

// sliding: window 513 + rope + a ring of `ring` pages; else full (no rope, linear pages over cap)
Rig make_rig(Ctx &cx, k::QsaKv kv, int cap, bool sliding, int ring, uint32_t seed, bool sinks = true) {
  Rig r;
  r.s = k::GqaShape{512, 32, 4, 1, 0, 1e-6f, kv};
  r.s.head_dim = 128, r.s.gate = false, r.s.qk_norm = true, r.s.norm_plus1 = false;
  r.s.rot = sliding ? 128 : 0;
  r.s.window = sliding ? 513 : 0;
  r.s.split_merge = std::getenv("XEN_T_WALK_MERGE") == nullptr;
  r.cap = cap;
  std::mt19937 rng(seed);
  // x / 8 below keeps q.k / sqrt(128) of order +-8 after the norms
  r.w.qkv = make_w4(cx, r.s.qkv_n(), r.s.hidden, 1.f, rng);
  r.w.o = make_w4(cx, r.s.hidden, r.s.o_k(), 1.f / std::sqrt(static_cast<float>(r.s.o_k())), rng);
  const int D = 128, hkv = r.s.hkv(), hq = r.s.hq();
  std::uniform_real_distribution<float> un(0.4f, 1.6f), usk(-4.f, 6.f);
  for (int i = 0; i < D; ++i) r.qn_h.push_back(k::bf16_bits(un(rng))), r.kn_h.push_back(k::bf16_bits(un(rng)));
  for (int h = 0; h < hq; ++h) r.sink_h.push_back(k::bf16_bits(usk(rng)));
  auto up16 = [&](const std::vector<uint16_t> &v) {
    auto *p = cx.alloc<bf16>(v.size());
    cx.q.memcpy(p, v.data(), v.size() * 2).wait();
    return p;
  };
  r.w.q_norm = up16(r.qn_h), r.w.k_norm = up16(r.kn_h);
  if (sinks) r.w.sinks = up16(r.sink_h);
  r.phys = sliding ? ring : cap / 64;
  r.c.max_tokens = cap;
  r.c.ring_pages = sliding ? ring : 0;
  r.c.slot_pages = r.phys;
  std::vector<int32_t> pt(r.phys);
  for (int i = 0; i < r.phys; ++i) pt[i] = r.phys - 1 - i;  // a permuted page table (reversed: a bijection at any size)
  auto *ptd = cx.alloc<int32_t>(r.phys);
  cx.q.memcpy(ptd, pt.data(), r.phys * 4).wait();
  r.c.pages = ptd;
  const size_t elems = static_cast<size_t>(r.phys) * hkv * 64 * D * r.s.kv_elem();
  r.c.k = cx.alloc<uint8_t>(elems);
  r.c.v = cx.alloc<uint8_t>(elems);
  std::vector<uint8_t> hk(elems), hv(elems);
  std::normal_distribution<float> nd(0.f, 1.f);
  if (kv == k::QsaKv::i8) {
    for (size_t i = 0; i < elems; ++i) hk[i] = static_cast<uint8_t>(static_cast<int8_t>(rng() % 255 - 127)), hv[i] = static_cast<uint8_t>(static_cast<int8_t>(rng() % 255 - 127));
    const size_t ns = static_cast<size_t>(r.phys) * hkv * 64;
    r.c.ks = cx.alloc<float>(ns);
    r.c.vs = cx.alloc<float>(ns);
    std::vector<float> ks(ns), vs(ns);
    for (auto &v : ks) v = std::fabs(nd(rng)) * 3.f / 127.f;
    for (auto &v : vs) v = std::fabs(nd(rng)) * 2.f / 127.f;
    cx.q.memcpy(r.c.ks, ks.data(), ns * 4);
    cx.q.memcpy(r.c.vs, vs.data(), ns * 4).wait();
  } else {
    auto *pk = reinterpret_cast<uint16_t *>(hk.data()), *pv = reinterpret_cast<uint16_t *>(hv.data());
    for (size_t i = 0; i < elems / 2; ++i) pk[i] = k::bf16_bits(nd(rng) * 3.f), pv[i] = k::bf16_bits(nd(rng) * 2.f);
  }
  cx.q.memcpy(r.c.k, hk.data(), elems);
  cx.q.memcpy(r.c.v, hv.data(), elems).wait();
  if (sliding) {
    r.rope_h = k::qsa_rope_table(cap, 128, 1e4, k::QsaYarn{});
    r.rope = up16(r.rope_h);
  }
  r.sc.y = cx.alloc<float>(8ull * r.s.qkv_n());
  r.sc.q = cx.alloc<bf16>(8ull * hq * D);
  r.sc.part = cx.alloc<float>(k::gqa_part_floats(r.s, 8));
  r.sc.attn = cx.alloc<bf16>(8ull * hq * D);
  r.sc.partial = cx.alloc<float>(8ull * r.s.hidden);
  r.pos = cx.alloc<int32_t>(1);
  r.x = cx.alloc<bf16>(8ull * r.s.hidden);
  r.out = cx.alloc<bf16>(8ull * r.s.hidden);
  std::vector<bf16> xh(8ull * r.s.hidden);
  for (auto &v : xh) v = bf16(nd(rng));
  cx.q.memcpy(r.x, xh.data(), xh.size() * 2).wait();
  return r;
}

std::vector<uint16_t> step(Ctx &cx, Rig &r, int rows, int p, int attn_rows = 0, int grf = 0, const bf16 *x = nullptr) {
  cx.q.memcpy(r.pos, &p, 4).wait();
  k::GqaStep st;
  st.pos = r.pos;
  st.pos_bound = std::min(r.cap, p + rows + 64);
  st.attn_rows = attn_rows;
  st.attn_grf = grf;
  k::gqa_decode(cx.q, r.s, rows, x ? x : r.x, r.w, r.c, r.rope, st, r.sc, nullptr, 0, r.out);
  cx.q.wait_and_throw();
  std::vector<uint16_t> a(static_cast<size_t>(rows) * r.s.o_k());
  cx.q.memcpy(a.data(), r.sc.attn, a.size() * 2).wait();
  return a;
}

void check(Ctx &cx, Rig &r, int rows, int p, const std::vector<uint16_t> &attn, const std::string &label, const float *dy = nullptr,
           const bf16 *dq = nullptr, double tol = 2.0 / 128) {
  const int D = 128, hq = r.s.hq(), hkv = r.s.hkv(), G = r.s.grp(), ldy = r.s.qkv_n(), qn = r.s.q_n(), win = r.s.window;
  std::vector<float> y(static_cast<size_t>(rows) * ldy);
  std::vector<uint16_t> qd(static_cast<size_t>(rows) * hq * D);
  cx.q.memcpy(y.data(), dy ? dy : r.sc.y, y.size() * 4);
  cx.q.memcpy(qd.data(), dq ? dq : r.sc.q, qd.size() * 2).wait();
  const bool i8 = r.s.kv == k::QsaKv::i8;
  const size_t elems = static_cast<size_t>(r.phys) * hkv * 64 * D * r.s.kv_elem();
  std::vector<uint8_t> hk(elems), hv(elems);
  std::vector<float> ks, vs;
  std::vector<int32_t> pt(r.phys);
  cx.q.memcpy(pt.data(), r.c.pages, r.phys * 4);
  cx.q.memcpy(hk.data(), r.c.k, elems);
  cx.q.memcpy(hv.data(), r.c.v, elems).wait();
  if (i8) {
    const size_t ns = static_cast<size_t>(r.phys) * hkv * 64;
    ks.resize(ns), vs.resize(ns);
    cx.q.memcpy(ks.data(), r.c.ks, ns * 4);
    cx.q.memcpy(vs.data(), r.c.vs, ns * 4).wait();
  }
  auto kval = [&](const std::vector<uint8_t> &pool, const std::vector<float> &sc, int tok, int h, int d) -> float {
    const int lp = r.c.ring_pages ? (tok / 64) % r.c.ring_pages : tok / 64;
    const size_t slot = (static_cast<size_t>(pt[lp]) * hkv + h) * 64 + tok % 64;
    if (i8) return static_cast<float>(static_cast<int8_t>(pool[slot * D + d])) * sc[slot];
    uint16_t b;
    std::memcpy(&b, &pool[(slot * D + d) * 2], 2);
    return k::bf16_value(b);
  };
  // bf16(rope(bf16(norm(bf16 v) w))), norm in fp32 as the device (the same expression up to fp order of the sum)
  auto norm_rope = [&](const float *v, const std::vector<uint16_t> &nw, int pos, float *o) {
    float n[128];
    double ss = 0;
    for (int d = 0; d < D; ++d) ss += static_cast<double>(bfr(v[d])) * bfr(v[d]);
    const float inv = 1.f / std::sqrt(static_cast<float>(ss / D) + r.s.eps);
    for (int d = 0; d < D; ++d) n[d] = bfr(bfr(v[d]) * inv * k::bf16_value(nw[d]));
    for (int d = 0; d < D; ++d) {
      if (!r.s.rot) {
        o[d] = n[d];
        continue;
      }
      const uint16_t *row = &r.rope_h[static_cast<size_t>(pos) * 128];
      const int i = d < 64 ? d : d - 64;
      const float cs = k::bf16_value(row[i]), sn = k::bf16_value(row[64 + i]);
      o[d] = bfr(d < 64 ? n[d] * cs - n[d + 64] * sn : n[d] * cs + n[d - 64] * sn);
    }
  };
  size_t q_off = 0, q_bad = 0, kv_bad = 0;
  double worst = 0;
  std::vector<float> qh(D), kh(D);
  for (int rr = 0; rr < rows; ++rr) {
    const int pp = p + rr;
    const float *yr = &y[static_cast<size_t>(rr) * ldy];
    for (int h = 0; h < hq; ++h) {
      norm_rope(yr + h * D, r.qn_h, pp, qh.data());
      for (int d = 0; d < D; ++d) {
        const float dv = k::bf16_value(qd[(static_cast<size_t>(rr) * hq + h) * D + d]);
        if (dv != qh[d]) ++q_off;
        if (std::fabs(dv - qh[d]) > ulp_bf16(qh[d])) ++q_bad;
      }
    }
    for (int h = 0; h < hkv; ++h) {
      norm_rope(yr + qn + h * D, r.kn_h, pp, kh.data());
      float amax_k = 0, amax_v = 0;
      for (int d = 0; d < D; ++d) amax_k = std::max(amax_k, std::fabs(kh[d])), amax_v = std::max(amax_v, std::fabs(bfr(yr[qn + (hkv + h) * D + d])));
      for (int d = 0; d < D; ++d) {
        const float kd = kval(hk, ks, pp, h, d), vd = kval(hv, vs, pp, h, d), vr = bfr(yr[qn + (hkv + h) * D + d]);
        const float tk = i8 ? amax_k / 127.f * 1.01f : ulp_bf16(kh[d]) * 2.01f, tv = i8 ? amax_v / 127.f * 1.01f : ulp_bf16(vr);
        if (std::fabs(kd - kh[d]) > tk || std::fabs(vd - vr) > tv) ++kv_bad;
      }
    }
    const int lo = win > 0 && pp + 1 > win ? pp + 1 - win : 0;
    for (int h = 0; h < hq; ++h) {
      const int kvh = h / G;
      std::vector<double> sc(pp + 1), o(D, 0.0);
      double mx = -1e300;
      for (int t = lo; t <= pp; ++t) {
        double dot = 0;
        for (int d = 0; d < D; ++d) dot += static_cast<double>(k::bf16_value(qd[(static_cast<size_t>(rr) * hq + h) * D + d])) * kval(hk, ks, t, kvh, d);
        sc[t] = dot / std::sqrt(128.0);
        mx = std::max(mx, sc[t]);
      }
      const bool sk = r.w.sinks != nullptr;
      const double sink = k::bf16_value(r.sink_h[h]);
      if (sk) mx = std::max(mx, sink);
      double l = sk ? std::exp(sink - mx) : 0.0;
      for (int t = lo; t <= pp; ++t) l += (sc[t] = std::exp(sc[t] - mx));
      for (int t = lo; t <= pp; ++t)
        for (int d = 0; d < D; ++d) o[d] += sc[t] / l * kval(hv, vs, t, kvh, d);
      double ss = 0;
      for (int d = 0; d < D; ++d) ss += o[d] * o[d];
      const double rms = std::sqrt(ss / D);
      for (int d = 0; d < D; ++d) {
        const float dv = k::bf16_value(attn[(static_cast<size_t>(rr) * hq + h) * D + d]);
        const double e = std::fabs(dv - o[d]) / std::max({std::fabs(o[d]), rms, 1e-30});
        worst = std::isfinite(e) ? std::max(worst, e) : HUGE_VAL;  // a non-finite output fails (std:max would drop a NaN)
      }
    }
  }
  const size_t nq = static_cast<size_t>(rows) * hq * D;
  char buf[320];
  std::snprintf(buf, sizeof buf, "%s: q %zu / %zu off by 1 ulp (%zu worse), K / V rows %zu bad, attn worst %.2e of max(|ref|, rms)", label.c_str(), q_off, nq,
                q_bad, kv_bad, worst);
  // q / k: the device's rsqrt against the host's 1 / sqrt moves a rare element by one more ulp (<= 0.1 %)
  expect(q_bad * 1000 <= nq && q_off * 100 <= nq && kv_bad == 0 && worst <= tol, buf);
}

size_t g_prep_q = 0, g_prep_n = 0, g_prep_steps = 0, g_prep_steps_diff = 0;

void run(Ctx &cx, k::QsaKv kv, bool sliding) {
  const std::string kn = std::string(kv == k::QsaKv::i8 ? "int8" : "bf16") + (sliding ? " sliding" : " full");
  const int cap = 4352;
  for (const int p : {0, 5, 63, 200, 505, 512, 513, 520, 640, 1000, 4093}) {
    for (const int rows : {1, 4, 8}) {
      if (p + rows > cap) continue;
      if (!sliding && p != 0 && p != 63 && p != 520 && p != 4093) continue;
      Rig r = make_rig(cx, kv, cap, sliding, 10, 1000 + p);
      const auto a = step(cx, r, rows, p);
      const std::string lab = kn + " " + std::to_string(rows) + " row(s) at " + std::to_string(p);
      check(cx, r, rows, p, a, lab);
      {  // the decode steps' ESIMD prep vs the SIMT prep (q, and the attention over the K / V each wrote)
        const int hqd = rows * r.s.hq() * 128;
        std::vector<uint16_t> qe(hqd), qs(hqd);
        cx.q.memcpy(qe.data(), r.sc.q, hqd * 2).wait();
        k::gqa_set_prep(0);
        const auto b = step(cx, r, rows, p);
        cx.q.memcpy(qs.data(), r.sc.q, hqd * 2).wait();
        k::gqa_set_prep(1);
        size_t nq = 0, na = 0;
        for (int i = 0; i < hqd; ++i) nq += qe[i] != qs[i];
        for (size_t i = 0; i < a.size(); ++i) na += a[i] != b[i];
        g_prep_q += nq, g_prep_n += hqd, g_prep_steps += 1, g_prep_steps_diff += nq > 0 || na > 0;
        expect(nq * 1000 <= static_cast<size_t>(hqd), lab + ": ESIMD prep vs SIMT prep: q " + std::to_string(nq) + " of " + std::to_string(hqd) + " differ (rsqrt rounding), attn " + std::to_string(na));
        step(cx, r, rows, p);
      }
      bool same = true;
      std::string odd;
      for (const int ar : {1, 2, 4, 8})
        for (const int grf : {128, 256}) {
          if (grf == 128 && ar > 1) continue;
          const auto b = step(cx, r, rows, p, ar, grf);
          if (b != a) {
            size_t nd = 0;
            for (size_t i = 0; i < a.size(); ++i) nd += a[i] != b[i];
            odd += " " + std::to_string(ar) + "x" + std::to_string(grf) + ":" + std::to_string(nd);
          }
          same &= b == a;
        }
      if (std::getenv("XEN_T_GROUPS"))  // diagnostic: each group size against the reference
        for (const int grf : {128, 256})
          for (const int ar : {1, 2, 4, 8}) {
            const auto b = step(cx, r, rows, p, ar, grf);
            size_t nan = 0;
            for (auto u : b) nan += (u & 0x7f80) == 0x7f80;
            check(cx, r, rows, p, b, lab + " group " + std::to_string(ar) + "x" + std::to_string(grf) + " (" + std::to_string(nan) + " non-finite)");
          }
      same &= step(cx, r, rows, p) == a;
      expect(same, lab + ": attention groups 1/2/4/8 rows, 128/256 GRF, rerun bit-identical" + (odd.empty() ? "" : " (differ:" + odd + ")"));
      if (rows > 1) {
        Rig r1 = make_rig(cx, kv, cap, sliding, 10, 1000 + p);
        bool seq = true;
        for (int i = 0; i < rows; ++i) {
          const auto a1 = step(cx, r1, 1, p + i, 0, 0, r.x + static_cast<size_t>(i) * r.s.hidden);
          seq &= std::equal(a1.begin(), a1.end(), a.begin() + static_cast<size_t>(i) * r.s.o_k());
        }
        expect(seq, lab + " == " + std::to_string(rows) + " one-row steps bit for bit");
      }
      cx.clear();
    }
  }
  // no sinks: the plain softmax (the sink's term is the only difference)
  Rig r = make_rig(cx, kv, cap, sliding, 10, 77, false);
  check(cx, r, 4, 700, step(cx, r, 4, 700), kn + " without sinks, 4 rows at 700");
  cx.clear();
  // a ring too small for the rows is refused
  if (sliding) {
    Rig rs = make_rig(cx, kv, cap, sliding, 9, 78);
    bool refused = false;
    try {
      step(cx, rs, 8, 700);
    } catch (const std::invalid_argument &) {
      refused = true;
    }
    expect(refused, kn + ": a 9-page ring is refused for window 513 at 8 rows");
    cx.clear();
  }
}

void run_rows(Ctx &cx, k::QsaKv kv, bool sliding, int m, int pos0) {
  const int cap = 4352;
  Rig r = make_rig(cx, kv, cap, sliding, (513 + m - 2) / 64 + 2, 7000 + pos0);
  const int H = r.s.hidden, hq = r.s.hq(), D = 128;
  std::mt19937 rng(99 + m);
  std::normal_distribution<float> nd(0.f, 1.f);
  std::vector<bf16> xh(static_cast<size_t>(m) * H);
  for (auto &v : xh) v = bf16(nd(rng));
  auto *x = cx.alloc<bf16>(xh.size());
  cx.q.memcpy(x, xh.data(), xh.size() * 2).wait();
  std::vector<int32_t> p4((m + 3) / 4);
  for (size_t i = 0; i < p4.size(); ++i) p4[i] = pos0 + 4 * static_cast<int>(i);
  auto *dp4 = cx.alloc<int32_t>(p4.size());
  cx.q.memcpy(dp4, p4.data(), p4.size() * 4).wait();
  k::GqaRowsScratch sc;
  sc.y = cx.alloc<float>(static_cast<size_t>(m) * r.s.qkv_n());
  sc.q = cx.alloc<bf16>(static_cast<size_t>(m) * hq * D);
  sc.part = cx.alloc<float>(k::gqa_part_floats(r.s, k::kGqaRowsBatch));
  sc.attn = cx.alloc<bf16>(static_cast<size_t>(m) * r.s.o_k());
  sc.partial = cx.alloc<float>(static_cast<size_t>(m) * H);
  auto *out = cx.alloc<bf16>(static_cast<size_t>(m) * H);
  r.c.slot_pages = 0;
  k::gqa_rows(cx.q, r.s, m, x, r.w, r.c, r.rope, dp4, pos0, sc, nullptr, 0, out, k::kGqaAttnBatch, 0, true, 0);
  cx.q.wait_and_throw();
  std::vector<uint16_t> a(static_cast<size_t>(m) * r.s.o_k());
  cx.q.memcpy(a.data(), sc.attn, a.size() * 2).wait();
  char lab[160];
  std::snprintf(lab, sizeof lab, "%s %s gqa_rows (batch body): %d rows at %d", kv == k::QsaKv::i8 ? "int8" : "bf16", sliding ? "sliding" : "full", m, pos0);
  check(cx, r, m, pos0, a, lab, sc.y, sc.q);
  // the fa body (window, ring, sinks at grp 8) at 32 / 16 / 24 rows per group and P hi + lo (the same bits for
  // every group size) and P hi only, each against the reference
  std::vector<uint16_t> fa32;
  for (const int qb : {32, 16, 24}) {
    k::gqa_rows(cx.q, r.s, m, x, r.w, r.c, r.rope, dp4, pos0, sc, nullptr, 0, out, k::kGqaAttnFa, qb, true, 0);
    cx.q.wait_and_throw();
    std::vector<uint16_t> f(static_cast<size_t>(m) * r.s.o_k());
    cx.q.memcpy(f.data(), sc.attn, f.size() * 2).wait();
    if (qb == 32) {
      fa32 = f;
      std::snprintf(lab, sizeof lab, "%s %s gqa_rows (fa body, 32 rows per group): %d rows at %d", kv == k::QsaKv::i8 ? "int8" : "bf16", sliding ? "sliding" : "full", m,
                    pos0);
      check(cx, r, m, pos0, f, lab, sc.y, sc.q);
    } else {
      size_t nd = 0;
      for (size_t i = 0; i < f.size(); ++i) nd += f[i] != fa32[i];
      std::snprintf(lab, sizeof lab, "%s %s fa body %d rows per group == 32 bit for bit (%d rows at %d; %zu differ)", kv == k::QsaKv::i8 ? "int8" : "bf16",
                    sliding ? "sliding" : "full", qb, m, pos0, nd);
      expect(nd == 0, lab);
    }
  }
  k::gqa_rows(cx.q, r.s, m, x, r.w, r.c, r.rope, dp4, pos0, sc, nullptr, 0, out, k::kGqaAttnFa, 0, false, 0);
  cx.q.wait_and_throw();
  std::vector<uint16_t> fh(static_cast<size_t>(m) * r.s.o_k());
  cx.q.memcpy(fh.data(), sc.attn, fh.size() * 2).wait();
  std::snprintf(lab, sizeof lab, "%s %s gqa_rows (fa body, P hi only): %d rows at %d", kv == k::QsaKv::i8 ? "int8" : "bf16", sliding ? "sliding" : "full", m, pos0);
  check(cx, r, m, pos0, fh, lab, sc.y, sc.q);
  if (kv == k::QsaKv::i8) {  // int8 q.k (hi + lo q codes, P hi only), 32 and 16 rows per group alike
    std::vector<uint16_t> f8[2];
    for (int i = 0; i < 2; ++i) {
      k::gqa_rows(cx.q, r.s, m, x, r.w, r.c, r.rope, dp4, pos0, sc, nullptr, 0, out, k::kGqaAttnFa, i ? 16 : 32, false, 2);
      cx.q.wait_and_throw();
      f8[i].resize(static_cast<size_t>(m) * r.s.o_k());
      cx.q.memcpy(f8[i].data(), sc.attn, f8[i].size() * 2).wait();
    }
    std::snprintf(lab, sizeof lab, "int8 %s gqa_rows (fa body, int8 q.k hi + lo, P hi only): %d rows at %d", sliding ? "sliding" : "full", m, pos0);
    check(cx, r, m, pos0, f8[0], lab, sc.y, sc.q);
    std::snprintf(lab, sizeof lab, "int8 %s fa body int8 q.k: 16 rows per group == 32 bit for bit (%d rows at %d)", sliding ? "sliding" : "full", m, pos0);
    expect(f8[0] == f8[1], lab);
  }
  cx.clear();
}

}  // namespace

int main() {
  try {
    const auto loader = xenithium::l0::Loader::open();
    if (loader.size() == 0) {
      std::printf("no device\n");
      return 1;
    }
    xenithium::Device dev = xenithium::Device::open(loader, 0);
    Ctx cx{dev, dev.compute(), {}};
    for (const k::QsaKv kv : {k::QsaKv::i8, k::QsaKv::bf16})
      for (const bool sl : {true, false}) run(cx, kv, sl);
    for (const k::QsaKv kv : {k::QsaKv::i8, k::QsaKv::bf16})
      for (const bool sl : {true, false})
        for (const auto &[m, pos0] : std::vector<std::pair<int, int>>{{64, 0}, {300, 0}, {300, 1000}, {37, 4000}, {700, 300}}) run_rows(cx, kv, sl, m, pos0);
  } catch (const std::exception &e) {
    std::printf("[FAIL] exception: %s\n", e.what());
    return 1;
  }
  std::printf("ESIMD prep vs SIMT prep over the decode sweep: %zu of %zu steps differ, %zu of %zu q elements\n", g_prep_steps_diff, g_prep_steps, g_prep_q, g_prep_n);
  std::printf("%s: %d failure(s)\n", g_failures ? "FAILED" : "OK", g_failures);
  return g_failures ? 1 : 0;
}
