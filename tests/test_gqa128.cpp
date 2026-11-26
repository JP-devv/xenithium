// Gate for the GQA decode at arch llama's shape (kernels/attention/gqa.hpp: head_dim 128, 7 q heads per kv head, no output gate,
// no q / k norms, the rope over all 128 dims): one card with the model's 56 q / 8 kv heads, hidden 512, synthetic W4
// projections. The KV pool is filled with random history first (int8 codes + scales, or bf16), then steps of 1-8 rows
// run at positions around page and split edges. Checks, teacher-forced at the device's own projections:
// q vs bf16(rope(bf16(y))) from the device's fp32 projections y: <= 1 bf16 ulp, <= 1 % of elements off
// K / V the rows the step wrote (codes dequantised) vs the same from y (<= 1 code / ulp)
// attn vs softmax(q K / sqrt(128)) V in double over every token 0..p as stored (2 bf16 ulps of max(|ref|, rms))
// and bit identities: attention groups of 1 / 2 / 4 / 8 rows at 128 and 256 GRF == the default; R rows at P == R
// one-row steps at P.. P + R - 1; run-to-run.
#include <algorithm>
#include <cmath>
#include <cstdio>
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
};

// a symmetric g128 W4 matrix [n][k] with random codes and fp16 scales (the attention is the subject here)
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
  std::vector<uint16_t> rope_h;
  int cap = 0;
};

Rig make_rig(Ctx &cx, k::QsaKv kv, int cap, uint32_t seed) {
  Rig r;
  r.s = k::GqaShape{512, 56, 8, 1, 0, 1e-5f, kv};
  r.s.head_dim = 128, r.s.gate = false, r.s.qk_norm = false, r.s.rot = 128;
  r.cap = cap;
  std::mt19937 rng(seed);
  r.w.qkv = make_w4(cx, r.s.qkv_n(), r.s.hidden, 1.f, rng);  // projections ~N(0, 1)-ish: wide score range
  r.w.o = make_w4(cx, r.s.hidden, r.s.o_k(), 1.f / std::sqrt(static_cast<float>(r.s.o_k())), rng);
  const int pages = cap / 64, D = 128, hkv = r.s.hkv();
  r.c.max_tokens = cap;
  r.c.slot_pages = pages;
  std::vector<int32_t> pt(pages);
  for (int i = 0; i < pages; ++i) pt[i] = i;
  auto *ptd = cx.alloc<int32_t>(pages);
  cx.q.memcpy(ptd, pt.data(), pages * 4).wait();
  r.c.pages = ptd;
  const size_t elems = static_cast<size_t>(pages) * hkv * 64 * D * r.s.kv_elem();
  r.c.k = cx.alloc<uint8_t>(elems);
  r.c.v = cx.alloc<uint8_t>(elems);
  // random history: int8 codes in [-127, 127] with per-(token, head) scales, or bf16 ~N(0, 1)
  std::vector<uint8_t> hk(elems), hv(elems);
  std::normal_distribution<float> nd(0.f, 1.f);
  if (kv == k::QsaKv::i8) {
    for (size_t i = 0; i < elems; ++i) hk[i] = static_cast<uint8_t>(static_cast<int8_t>(rng() % 255 - 127)), hv[i] = static_cast<uint8_t>(static_cast<int8_t>(rng() % 255 - 127));
    const size_t ns = static_cast<size_t>(pages) * hkv * 64;
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
  r.rope_h = k::qsa_rope_table(cap, 128, 1.2e8, k::QsaYarn{});
  r.rope = cx.alloc<bf16>(r.rope_h.size());
  cx.q.memcpy(r.rope, r.rope_h.data(), r.rope_h.size() * 2).wait();
  const int hq = r.s.hq();
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

// One step: rows at position p; returns the attention output (bf16 bits [rows][hq * D]).
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

// teacher-forced checks of the step just run (rows at p)
// (dy, dq: the projections and q of a many-row call; tol: the attention bound in max(|ref|, rms))
void check(Ctx &cx, Rig &r, int rows, int p, const std::vector<uint16_t> &attn, const std::string &label, const float *dy = nullptr,
           const bf16 *dq = nullptr, double tol = 2.0 / 128) {
  const int D = 128, hq = r.s.hq(), hkv = r.s.hkv(), G = r.s.grp(), ldy = r.s.qkv_n(), qn = r.s.q_n();
  std::vector<float> y(static_cast<size_t>(rows) * ldy);
  std::vector<uint16_t> qd(static_cast<size_t>(rows) * hq * D);
  cx.q.memcpy(y.data(), dy ? dy : r.sc.y, y.size() * 4);
  cx.q.memcpy(qd.data(), dq ? dq : r.sc.q, qd.size() * 2).wait();
  const bool i8 = r.s.kv == k::QsaKv::i8;
  const size_t elems = static_cast<size_t>(r.cap / 64) * hkv * 64 * D * r.s.kv_elem();
  std::vector<uint8_t> hk(elems), hv(elems);
  std::vector<float> ks, vs;
  cx.q.memcpy(hk.data(), r.c.k, elems);
  cx.q.memcpy(hv.data(), r.c.v, elems).wait();
  if (i8) {
    const size_t ns = static_cast<size_t>(r.cap / 64) * hkv * 64;
    ks.resize(ns), vs.resize(ns);
    cx.q.memcpy(ks.data(), r.c.ks, ns * 4);
    cx.q.memcpy(vs.data(), r.c.vs, ns * 4).wait();
  }
  auto kval = [&](const std::vector<uint8_t> &pool, const std::vector<float> &sc, int tok, int h, int d) -> float {
    const size_t slot = (static_cast<size_t>(tok / 64) * hkv + h) * 64 + tok % 64;  // identity page table
    if (i8) return static_cast<float>(static_cast<int8_t>(pool[slot * D + d])) * sc[slot];
    uint16_t b;
    std::memcpy(&b, &pool[(slot * D + d) * 2], 2);
    return k::bf16_value(b);
  };
  auto rope = [&](const float *v, int pos, float *o) {  // bf16(rope(bf16(v)))
    const uint16_t *row = &r.rope_h[static_cast<size_t>(pos) * 128];
    for (int d = 0; d < D; ++d) {
      const int i = d < 64 ? d : d - 64;
      const float cs = k::bf16_value(row[i]), sn = k::bf16_value(row[64 + i]);
      const float n = bfr(v[d]), oth = bfr(d < 64 ? v[d + 64] : v[d - 64]);
      o[d] = bfr(d < 64 ? n * cs - oth * sn : n * cs + oth * sn);
    }
  };
  size_t q_off = 0, q_bad = 0, kv_bad = 0, n_attn = 0, off_attn = 0;
  double worst = 0;
  std::vector<float> qh(D), kh(D);
  for (int rr = 0; rr < rows; ++rr) {
    const int pp = p + rr;
    const float *yr = &y[static_cast<size_t>(rr) * ldy];
    for (int h = 0; h < hq; ++h) {
      rope(yr + h * D, pp, qh.data());
      for (int d = 0; d < D; ++d) {
        const float dv = k::bf16_value(qd[(static_cast<size_t>(rr) * hq + h) * D + d]);
        if (dv != qh[d]) ++q_off;
        if (std::fabs(dv - qh[d]) > ulp_bf16(qh[d])) ++q_bad;
      }
    }
    for (int h = 0; h < hkv; ++h) {  // the K / V rows this step wrote
      rope(yr + qn + h * D, pp, kh.data());
      float amax_k = 0, amax_v = 0;
      for (int d = 0; d < D; ++d) amax_k = std::max(amax_k, std::fabs(kh[d])), amax_v = std::max(amax_v, std::fabs(bfr(yr[qn + (hkv + h) * D + d])));
      for (int d = 0; d < D; ++d) {
        const float kd = kval(hk, ks, pp, h, d), vd = kval(hv, vs, pp, h, d), vr = bfr(yr[qn + (hkv + h) * D + d]);
        const float tk = i8 ? amax_k / 127.f * 1.01f : ulp_bf16(kh[d]), tv = i8 ? amax_v / 127.f * 1.01f : ulp_bf16(vr);
        if (std::fabs(kd - kh[d]) > tk || std::fabs(vd - vr) > tv) ++kv_bad;
      }
    }
    // attention per q head over tokens 0..pp as stored, q = the device's
    for (int h = 0; h < hq; ++h) {
      const int kvh = h / G;
      std::vector<double> sc(pp + 1), o(D, 0.0);
      double mx = -1e300;
      for (int t = 0; t <= pp; ++t) {
        double dot = 0;
        for (int d = 0; d < D; ++d) dot += static_cast<double>(k::bf16_value(qd[(static_cast<size_t>(rr) * hq + h) * D + d])) * kval(hk, ks, t, kvh, d);
        sc[t] = dot / std::sqrt(128.0);
        mx = std::max(mx, sc[t]);
      }
      double l = 0;
      for (int t = 0; t <= pp; ++t) l += (sc[t] = std::exp(sc[t] - mx));
      for (int t = 0; t <= pp; ++t)
        for (int d = 0; d < D; ++d) o[d] += sc[t] / l * kval(hv, vs, t, kvh, d);
      double ss = 0;
      for (int d = 0; d < D; ++d) ss += o[d] * o[d];
      const double rms = std::sqrt(ss / D);
      for (int d = 0; d < D; ++d) {
        const float dv = k::bf16_value(attn[(static_cast<size_t>(rr) * hq + h) * D + d]);
        ++n_attn;
        off_attn += dv != bfr(static_cast<float>(o[d]));
        const double e = std::fabs(dv - o[d]) / std::max({std::fabs(o[d]), rms, 1e-30});
        worst = std::isfinite(e) ? std::max(worst, e) : HUGE_VAL;  // a non-finite output fails (std:max would drop a NaN)
      }
    }
  }
  const size_t nq = static_cast<size_t>(rows) * hq * D;
  char buf[320];
  std::snprintf(buf, sizeof buf, "%s: q %zu / %zu off by 1 ulp (%zu worse), K / V rows %zu bad, attn worst %.2e of max(|ref|, rms) (%.1f %% not the rounded ref)",
                label.c_str(), q_off, nq, q_bad, kv_bad, worst, 100.0 * off_attn / std::max<size_t>(1, n_attn));
  expect(q_bad == 0 && q_off * 100 <= nq && kv_bad == 0 && worst <= tol, buf);
}

// Many rows (prefill, gqa_rows): m rows at pos0 through attention body `attn` (0: the decode group kernel per 64-row
// batch, 3: the fa body; qk / p_lo its numerics switches), checked as the decode steps are (the same reference).
void run_rows(Ctx &cx, k::QsaKv kv, int m, int pos0, int attn, int qk, bool p_lo, double tol, std::vector<uint16_t> *out_attn = nullptr) {
  const int cap = 4352;
  Rig r = make_rig(cx, kv, cap, 7000 + pos0);
  r.c.slot_pages = 0;  // one sequence's view (the model's kv_view): the fa body takes no slot stride
  const int H = r.s.hidden, hq = r.s.hq(), D = 128;
  std::mt19937 rng(99 + m);
  std::normal_distribution<float> nd(0.f, 1.f);
  std::vector<bf16> xh(static_cast<size_t>(m) * H);
  // x / 8: q.k / sqrt(128) of order +-8 (x ~ N(0, 1) gives +-60: a one-hot softmax, where any q.k rounding flips the
  // pick and the int8 q.k switches cannot be judged)
  for (auto &v : xh) v = bf16(nd(rng) * 0.125f);
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
  k::gqa_rows(cx.q, r.s, m, x, r.w, r.c, r.rope, dp4, pos0, sc, nullptr, 0, out, attn, 0, p_lo, qk);
  cx.q.wait_and_throw();
  std::vector<uint16_t> a(static_cast<size_t>(m) * r.s.o_k());
  cx.q.memcpy(a.data(), sc.attn, a.size() * 2).wait();
  char lab[160];
  std::snprintf(lab, sizeof lab, "%s gqa_rows attention %d (qk %d, P %s): %d rows at %d", kv == k::QsaKv::i8 ? "int8" : "bf16", attn, qk, p_lo ? "hi + lo" : "hi only", m,
                pos0);
  check(cx, r, m, pos0, a, lab, sc.y, sc.q, tol);
  if (out_attn) *out_attn = a;
  for (void *p : cx.mem) cx.dev.release(p);
  cx.mem.clear();
}

void run(Ctx &cx, k::QsaKv kv) {
  const char *kn = kv == k::QsaKv::i8 ? "int8" : "bf16";
  const int cap = 4352;  // 68 pages: positions past 4,096 (tiles per split 1 -> 2)
  for (const int p : {0, 5, 63, 200, 4093}) {
    for (const int rows : {1, 4, 8}) {
      if (p + rows > cap) continue;
      Rig r = make_rig(cx, kv, cap, 1000 + p);
      const auto a = step(cx, r, rows, p);
      check(cx, r, rows, p, a, std::string(kn) + " " + std::to_string(rows) + " row(s) at " + std::to_string(p));
      // groups x GRF bit identity (the K / V rows are rewritten with the same bits each time)
      bool same = true;
      for (const int ar : {1, 2, 4, 8})
        for (const int grf : {128, 256}) {
          if (grf == 128 && ar > 1) continue;  // 128-GRF instances: single rows only
          same &= step(cx, r, rows, p, ar, grf) == a;
        }
      same &= step(cx, r, rows, p) == a;  // run-to-run
      expect(same, std::string(kn) + " " + std::to_string(rows) + " row(s) at " + std::to_string(p) + ": attention groups 1/2/4/8 rows, 128/256 GRF, rerun bit-identical");
      if (rows > 1) {  // R rows at p == R one-row steps at p.. p + R - 1 (each row's own x)
        Rig r1 = make_rig(cx, kv, cap, 1000 + p);
        bool seq = true;
        for (int i = 0; i < rows; ++i) {
          const auto a1 = step(cx, r1, 1, p + i, 0, 0, r.x + static_cast<size_t>(i) * r.s.hidden);
          seq &= std::equal(a1.begin(), a1.end(), a.begin() + static_cast<size_t>(i) * r.s.o_k());
        }
        expect(seq, std::string(kn) + " " + std::to_string(rows) + " rows at " + std::to_string(p) + " == " + std::to_string(rows) + " one-row steps bit for bit");
      }
      for (void *m : cx.mem) cx.dev.release(m);
      cx.mem.clear();
    }
  }
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
    run(cx, k::QsaKv::i8);
    run(cx, k::QsaKv::bf16);
    for (const k::QsaKv kv : {k::QsaKv::i8, k::QsaKv::bf16})
      for (const auto &[m, pos0] : std::vector<std::pair<int, int>>{{64, 0}, {300, 0}, {300, 1000}, {37, 4000}}) {
        std::vector<uint16_t> a0, a3;
        run_rows(cx, kv, m, pos0, k::kGqaAttnBatch, 0, true, 2.0 / 128, &a0);
        run_rows(cx, kv, m, pos0, k::kGqaAttnFa, 0, true, 2.0 / 128, &a3);
        run_rows(cx, kv, m, pos0, k::kGqaAttnFa, 0, false, 4.0 / 128);  // P hi only (a numerics switch)
        if (kv == k::QsaKv::i8)  // int8 q.k hi + lo, P hi only (qk 2; qk 1, one code, is not offered for llama)
          run_rows(cx, kv, m, pos0, k::kGqaAttnFa, 2, false, 4.0 / 128);
      }
  } catch (const std::exception &e) {
    std::printf("[FAIL] exception: %s\n", e.what());
    return 1;
  }
  std::printf("%s: %d failure(s)\n", g_failures ? "FAILED" : "OK", g_failures);
  return g_failures ? 1 : 0;
}
