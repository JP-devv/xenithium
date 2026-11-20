// The 16- and 32-row steps: the kernels' 16-row and 32-row shapes keep every row's bits.
// dense_gemm (int8 and bf16 weights, k split by shape, with and without a k % 128 tail) at 9-16 rows (one 16-row launch:
// two 8-row halves per work-group) and at 1-8 rows: every live row == the same row run alone (rows = 1).
// The W4A16 decode GEMM (w4a16_gemm_rows) and its swiglu form at 2-16 rows, on
// shapes and k splits that take each 9-16-row form (C16 at k split 1 and above, the halves H16 where they fit one wave;
// 9-12 rows the 4-row second chain): every live row == the row alone, y rows past the
// launch's shape (9-16 rows: past the live ones) and h rows past the live ones untouched.
// The same up to 32 rows (dense_gemm's 32-row shape of four 8-row quarters, its
// two launches at k split 32; the W4A16 C32 form and its swiglu form; a k split 16 case).
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include <sycl/sycl.hpp>

#include "xenithium/kernels/gemm/dense.hpp"
#include "xenithium/kernels/gemm/mlp.hpp"
#include "xenithium/kernels/gemm/w4a16_gemm.hpp"

namespace k = xenithium::kernels;
using bf16 = sycl::ext::oneapi::bfloat16;

static int g_checks = 0, g_fail = 0;
static void expect(bool ok, const std::string &what) {
  ++g_checks;
  if (!ok) ++g_fail;
  std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
}

int main() {
  constexpr int kR = k::kStepRows32;  // 32
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
  std::mt19937 g(5);
  std::normal_distribution<float> nd(0.f, 1.f);
  struct Shape {
    int n, k;
  };
  for (const auto kind : {k::DenseKind::i8, k::DenseKind::bf16})
    for (const Shape sh : {Shape{2560, 2560}, Shape{512, 6144}, Shape{1024, 2576}, Shape{256, 8192}}) {
      k::DenseWeights w;
      w.kind = kind, w.n = sh.n, w.k = sh.k;
      const size_t nk = static_cast<size_t>(sh.n) * sh.k;
      void *wb = sycl::malloc_device(kind == k::DenseKind::i8 ? nk : nk * 2, q);
      float *sc = sycl::malloc_device<float>(sh.n, q);
      if (kind == k::DenseKind::i8) {
        std::vector<int8_t> h(nk);
        for (auto &v : h) v = static_cast<int8_t>(static_cast<int>(g() % 255) - 127);
        q.memcpy(wb, h.data(), nk).wait();
      } else {
        std::vector<bf16> h(nk);
        for (auto &v : h) v = bf16(nd(g) * 0.05f);
        q.memcpy(wb, h.data(), nk * 2).wait();
      }
      std::vector<float> hs(sh.n);
      for (auto &v : hs) v = 0.001f + 0.01f * std::abs(nd(g));
      q.memcpy(sc, hs.data(), sh.n * 4).wait();
      w.w = wb, w.scale = kind == k::DenseKind::i8 ? sc : nullptr;
      bf16 *x = sycl::malloc_device<bf16>(static_cast<size_t>(kR) * sh.k, q);
      float *y = sycl::malloc_device<float>(static_cast<size_t>(kR) * sh.n, q), *y1 = sycl::malloc_device<float>(static_cast<size_t>(kR) * sh.n, q);
      std::vector<bf16> hx(static_cast<size_t>(kR) * sh.k);
      for (auto &v : hx) v = bf16(nd(g));
      q.memcpy(x, hx.data(), hx.size() * 2).wait();
      // each row alone
      for (int r = 0; r < kR; ++r) k::dense_gemm(q, 1, x + static_cast<size_t>(r) * sh.k, w, y1 + static_cast<size_t>(r) * sh.n);
      q.wait();
      std::vector<uint32_t> ref(static_cast<size_t>(kR) * sh.n), got(ref.size());
      q.memcpy(ref.data(), y1, ref.size() * 4).wait();
      for (int rows = 2; rows <= kR; ++rows) {
        q.fill(y, 0.f, static_cast<size_t>(kR) * sh.n).wait();
        k::dense_gemm(q, rows, x, w, y).wait();
        q.memcpy(got.data(), y, got.size() * 4).wait();
        size_t bad = 0;
        for (size_t e = 0; e < static_cast<size_t>(rows) * sh.n; ++e) bad += got[e] != ref[e];
        expect(bad == 0, std::string("dense_gemm ") + (kind == k::DenseKind::i8 ? "int8" : "bf16") + " n " + std::to_string(sh.n) + " k " + std::to_string(sh.k) +
                             " (k split " + std::to_string(k::dense_auto_ksplit(sh.n, sh.k)) + "): " + std::to_string(rows) + " rows (shape " +
                             std::to_string(k::row_shape(rows)) + ") == each row alone" + (bad ? " (" + std::to_string(bad) + " differ)" : ""));
      }
      sycl::free(wb, q), sycl::free(sc, q), sycl::free(x, q), sycl::free(y, q), sycl::free(y1, q);
    }
  // ---- W4A16 (symmetric g128) ----
  float *silu = sycl::malloc_device<float>(65536, q);
  k::swiglu_table(q, silu).wait();
  struct W4Case {
    int n, k, ks;  // ks 0 = the auto split
  };
  for (const W4Case wc : {W4Case{17408, 1280, 0}, W4Case{8704, 2560, 1}, W4Case{8704, 2560, 2}, W4Case{5120, 3072, 0}, W4Case{8192, 2560, 0}, W4Case{1024, 1280, 0}, W4Case{2048, 5120, 16}}) {
    const size_t wb = static_cast<size_t>(wc.n) * wc.k / 2, nsc = static_cast<size_t>(wc.k / 128) * wc.n;
    uint8_t *w = sycl::aligned_alloc_device<uint8_t>(4096, wb, q);
    sycl::half *sc = sycl::aligned_alloc_device<sycl::half>(4096, nsc, q);
    {
      std::vector<uint8_t> hw(wb);
      for (auto &v : hw) v = static_cast<uint8_t>(g());
      std::vector<sycl::half> hs(nsc);
      for (auto &v : hs) v = sycl::half((nd(g) >= 0 ? 1.f : -1.f) * (0.002f + 0.01f * std::abs(nd(g))));  // signed scales (AutoRound)
      q.memcpy(w, hw.data(), wb).wait();
      q.memcpy(sc, hs.data(), nsc * 2).wait();
    }
    bf16 *x = sycl::aligned_alloc_device<bf16>(4096, static_cast<size_t>(kR) * wc.k, q);
    std::vector<bf16> hx(static_cast<size_t>(kR) * wc.k);
    for (auto &v : hx) v = bf16(nd(g));
    q.memcpy(x, hx.data(), hx.size() * 2).wait();
    const size_t ny = static_cast<size_t>(kR) * wc.n;
    float *y = sycl::malloc_device<float>(ny, q), *y1 = sycl::malloc_device<float>(ny, q);
    const int nh = wc.n / 2;
    bf16 *h = sycl::malloc_device<bf16>(static_cast<size_t>(kR) * nh, q), *h1 = sycl::malloc_device<bf16>(static_cast<size_t>(kR) * nh, q);
    for (int r = 0; r < kR; ++r) {
      k::w4a16_gemm_rows(q, 1, x + static_cast<size_t>(r) * wc.k, w, sc, y1 + static_cast<size_t>(r) * wc.n, wc.n, wc.k, wc.ks);
      k::w4a16_gemm_rows_swiglu(q, 1, x + static_cast<size_t>(r) * wc.k, w, sc, silu, h1 + static_cast<size_t>(r) * nh, nh, wc.k, wc.ks);
    }
    q.wait();
    std::vector<uint32_t> ref(ny), got(ny);
    std::vector<uint16_t> href(static_cast<size_t>(kR) * nh), hgot(href.size());
    q.memcpy(ref.data(), y1, ny * 4).wait();
    q.memcpy(href.data(), h1, href.size() * 2).wait();
    const int ks = wc.ks > 0 ? wc.ks : k::w4a16_auto_ksplit(wc.n, wc.k);
    size_t bad_y = 0, bad_h = 0, dirty = 0;
    std::string fails;
    for (int rows = 2; rows <= kR; ++rows) {
      q.fill(y, -7.f, ny).wait();
      q.fill(h, bf16(-7.f), static_cast<size_t>(kR) * nh).wait();
      k::w4a16_gemm_rows(q, rows, x, w, sc, y, wc.n, wc.k, wc.ks);
      k::w4a16_gemm_rows_swiglu(q, rows, x, w, sc, silu, h, nh, wc.k, wc.ks).wait();
      q.memcpy(got.data(), y, ny * 4).wait();
      q.memcpy(hgot.data(), h, hgot.size() * 2).wait();
      size_t by = 0, bh = 0;
      for (size_t e = 0; e < static_cast<size_t>(rows) * wc.n; ++e) by += got[e] != ref[e];
      for (size_t e = 0; e < static_cast<size_t>(rows) * nh; ++e) bh += hgot[e] != href[e];
      const float m7 = -7.f;
      uint32_t m7b;
      std::memcpy(&m7b, &m7, 4);
      // the 1-8-row launches write their shape's rows of y (1, 2, 4 or 8); 9-32 rows only the live ones; h always the live ones
      const int yrows = rows <= 2 ? rows : rows <= 4 ? 4 : rows <= 8 ? 8 : rows;
      for (size_t e = static_cast<size_t>(yrows) * wc.n; e < ny; ++e) dirty += got[e] != m7b;
      for (size_t e = static_cast<size_t>(rows) * nh; e < hgot.size(); ++e) dirty += hgot[e] != 0xc0e0;  // bf16(-7)
      if (by || bh) fails += " " + std::to_string(rows) + ":" + std::to_string(by) + "/" + std::to_string(bh);
      bad_y += by, bad_h += bh;
    }
    expect(bad_y == 0 && bad_h == 0 && dirty == 0, "w4a16_gemm_rows / _swiglu n " + std::to_string(wc.n) + " k " + std::to_string(wc.k) + " (k split " +
                                                       std::to_string(ks) + "): 2-32 rows == each row alone, rows past the shape (9-32: the live rows) untouched" +
                                                       (fails.empty() ? "" : " (rows:y/h differ" + fails + ")") + (dirty ? " (" + std::to_string(dirty) + " dirty)" : ""));
    {
      // ---- W4A8 decode: the same matrix registered, int8 activations per (row, 128-k group) ----
      k::w4a8_rows_register(w);
      k::w4a8_rows_reserve(q, wc.k);
      k::w4a8_rows_set(true);
      for (int r = 0; r < kR; ++r) {
        k::w4a16_gemm_rows(q, 1, x + static_cast<size_t>(r) * wc.k, w, sc, y1 + static_cast<size_t>(r) * wc.n, wc.n, wc.k, wc.ks);
        k::w4a16_gemm_rows_swiglu(q, 1, x + static_cast<size_t>(r) * wc.k, w, sc, silu, h1 + static_cast<size_t>(r) * nh, nh, wc.k, wc.ks);
      }
      q.wait();
      std::vector<uint32_t> r8(ny);
      std::vector<uint16_t> hr8(href.size());
      q.memcpy(r8.data(), y1, ny * 4).wait();
      q.memcpy(hr8.data(), h1, hr8.size() * 2).wait();
      double e2 = 0, s2 = 0;  // against the bf16 activations' result: the int8 rounding only
      for (size_t e = 0; e < ny; ++e) {
        float a, b;
        std::memcpy(&a, &r8[e], 4), std::memcpy(&b, &ref[e], 4);
        e2 += (static_cast<double>(a) - b) * (static_cast<double>(a) - b), s2 += static_cast<double>(b) * b;
      }
      const double rel = std::sqrt(e2 / std::max(s2, 1e-30));
      size_t by8 = 0, bh8 = 0, dirty8 = 0;
      std::string fails8;
      for (int rows = 2; rows <= kR; ++rows) {
        q.fill(y, -7.f, ny).wait();
        q.fill(h, bf16(-7.f), static_cast<size_t>(kR) * nh).wait();
        k::w4a16_gemm_rows(q, rows, x, w, sc, y, wc.n, wc.k, wc.ks);
        k::w4a16_gemm_rows_swiglu(q, rows, x, w, sc, silu, h, nh, wc.k, wc.ks).wait();
        q.memcpy(got.data(), y, ny * 4).wait();
        q.memcpy(hgot.data(), h, hgot.size() * 2).wait();
        size_t by = 0, bh = 0;
        for (size_t e = 0; e < static_cast<size_t>(rows) * wc.n; ++e) by += got[e] != r8[e];
        for (size_t e = 0; e < static_cast<size_t>(rows) * nh; ++e) bh += hgot[e] != hr8[e];
        const float m7 = -7.f;
        uint32_t m7b;
        std::memcpy(&m7b, &m7, 4);
        for (size_t e = static_cast<size_t>(rows) * wc.n; e < ny; ++e) dirty8 += got[e] != m7b;  // every instance stores the live rows only
        for (size_t e = static_cast<size_t>(rows) * nh; e < hgot.size(); ++e) dirty8 += hgot[e] != 0xc0e0;
        if (by || bh) fails8 += " " + std::to_string(rows) + ":" + std::to_string(by) + "/" + std::to_string(bh);
        by8 += by, bh8 += bh;
      }
      k::w4a8_rows_set(false);
      k::w4a8_rows_unregister(w);
      char rb[48];
      std::snprintf(rb, sizeof rb, "%.2e", rel);
      expect(by8 == 0 && bh8 == 0 && dirty8 == 0 && rel > 1e-4 && rel < 2e-2,
             "W4A8 decode rows n " + std::to_string(wc.n) + " k " + std::to_string(wc.k) + " (k split " + std::to_string(ks) +
                 "): 2-32 rows == each row alone (gemm and swiglu), rows past the live ones untouched, relative rms distance to W4A16 " + rb +
                 (fails8.empty() ? "" : " (rows:y/h differ" + fails8 + ")") + (dirty8 ? " (" + std::to_string(dirty8) + " dirty)" : ""));
      // off again: the bf16 activations' bits
      k::w4a16_gemm_rows(q, 8, x, w, sc, y, wc.n, wc.k, wc.ks).wait();
      q.memcpy(got.data(), y, ny * 4).wait();
      size_t back = 0;
      for (size_t e = 0; e < static_cast<size_t>(8) * wc.n; ++e) back += got[e] != ref[e];
      expect(back == 0, "W4A8 decode off / unregistered: the W4A16 bits again");
    }
    sycl::free(w, q), sycl::free(sc, q), sycl::free(x, q), sycl::free(y, q), sycl::free(y1, q), sycl::free(h, q), sycl::free(h1, q);
  }
  sycl::free(silu, q);
  std::printf("rows16 (to 32 rows): %d checks, %d failed\n", g_checks, g_fail);
  return g_fail ? 1 : 0;
}
