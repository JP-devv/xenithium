// Gate for the kn-layout decode GEMM (kernels/gemm/w4kn_gemm.hpp, branch onednn-refactor):
// w4kn_repack(nib) == the host permutation (byte p * 16 + l of nib block (g, t) -> [t * 16 + l][g * 64 + p] ^ 0x88);
// w4kn_gemm_rows == w4a16_gemm_rows bit for bit at the same k split (same B operand bytes, same summation order),
// rows 1..8, two shapes, k splits 1 / 4 / auto (auto: max |d| <= 1e-6 of max |ref|, the split may differ).
// Pf_dense's oneDNN route for int8 matrices (the engine's bits on the check call and after; split-K; a shape that
// stays on the engine's body) and the W4A8 oneDNN GLU's registered h codes (a down on them == its own quantiser).
#include <sycl/sycl.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "xenithium/kernels/gemm/w4a16_gemm.hpp"
#include "xenithium/kernels/gemm/w4kn_gemm.hpp"
#include "xenithium/kernels/gemm/onednn_pf.hpp"
#include "xenithium/kernels/gemm/pf_dense.hpp"
#include "xenithium/kernels/pack.hpp"

namespace k = xenithium::kernels;
using bf16 = sycl::ext::oneapi::bfloat16;

int g_failures = 0;
void expect(bool ok, const char *what) {
  std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", what);
  if (!ok) ++g_failures;
}

void run(sycl::queue &q, int n, int kk) {
  std::mt19937 rng(static_cast<uint32_t>(n * 7 + kk));
  const size_t wb = static_cast<size_t>(n) * kk / 2;
  std::vector<uint8_t> nib(wb), want(wb), got(wb);
  for (auto &b : nib) b = static_cast<uint8_t>(rng());
  const int tiles = n / 16;
  for (int g = 0; g < kk / 128; ++g)
    for (int t = 0; t < tiles; ++t)
      for (int p = 0; p < 64; ++p)
        for (int l = 0; l < 16; ++l) want[(static_cast<size_t>(t) * 16 + l) * (kk / 2) + g * 64 + p] = nib[(static_cast<size_t>(g) * tiles + t) * 1024 + p * 16 + l] ^ 0x88;
  auto *dn = sycl::malloc_device<uint8_t>(wb, q), *dk = sycl::malloc_device<uint8_t>(wb, q);
  q.memcpy(dn, nib.data(), wb).wait();
  k::w4kn_repack(q, dn, n, kk, dk).wait();
  q.memcpy(got.data(), dk, wb).wait();
  char what[200];
  std::snprintf(what, sizeof what, "n %d k %d: w4kn_repack == the host permutation", n, kk);
  expect(got == want, what);
  std::vector<sycl::half> hs(static_cast<size_t>(kk / 128) * n);
  for (auto &s : hs) s = static_cast<sycl::half>(0.001f * (1 + rng() % 32));
  std::vector<bf16> ha(static_cast<size_t>(8) * kk);
  for (auto &x : ha) x = static_cast<bf16>((static_cast<int>(rng() % 2001) - 1000) * 1e-3f);
  auto *S = sycl::malloc_device<sycl::half>(hs.size(), q);
  auto *A = sycl::malloc_device<bf16>(ha.size(), q);
  auto *C0 = sycl::malloc_device<float>(static_cast<size_t>(8) * n, q), *C1 = sycl::malloc_device<float>(static_cast<size_t>(8) * n, q);
  q.memcpy(S, hs.data(), hs.size() * 2).wait();
  q.memcpy(A, ha.data(), ha.size() * 2).wait();
  for (int rows = 1; rows <= 8; ++rows) {
    bool bits = true;
    double worst = 0;
    for (int ks : {1, 4, 0}) {
      q.memset(C0, 0, 8ull * n * 4), q.memset(C1, 0xff, 8ull * n * 4).wait();
      k::w4a16_gemm_rows(q, rows, A, dn, S, C0, n, kk, ks).wait();
      k::w4kn_gemm_rows(q, rows, A, dk, S, C1, n, kk, ks).wait();
      std::vector<float> r0(static_cast<size_t>(rows) * n), r1(r0.size());
      q.memcpy(r0.data(), C0, r0.size() * 4), q.memcpy(r1.data(), C1, r1.size() * 4).wait();
      double mref = 0, md = 0;
      for (size_t i = 0; i < r0.size(); ++i) mref = std::max(mref, static_cast<double>(std::fabs(r0[i]))), md = std::max(md, static_cast<double>(std::fabs(r0[i] - r1[i])));
      if (ks > 0) bits &= r0 == r1;
      worst = std::max(worst, md / std::max(mref, 1e-30));
    }
    std::snprintf(what, sizeof what, "n %d k %d rows %d: w4kn_gemm_rows == w4a16_gemm_rows bit for bit at k split 1 / 4, auto split within %.1e", n, kk, rows, worst);
    expect(bits && worst <= 1e-6, what);
  }
  sycl::free(dn, q), sycl::free(dk, q), sycl::free(S, q), sycl::free(A, q), sycl::free(C0, q), sycl::free(C1, q);
}

// The W4A8 prefill GEMM on oneDNN (onednn_pf, XEN_PREFILL_GEMM=onednn) vs the ESIMD body on the same codes: plain and
// GLU, m = 1024 / 1500 rows (the oneDNN path: calls of >= 1024 rows). Worst |d| relative to max |ref| (bf16 / fp32 output
// rounding order differs; the integer group sums are the same).
void run_pf(sycl::queue &q, int m, int n, int kk) {
  std::mt19937 rng(static_cast<uint32_t>(m + n));
  const size_t wb = static_cast<size_t>(n) * kk / 2;
  std::vector<uint8_t> nib(wb);
  for (auto &b : nib) b = static_cast<uint8_t>(rng());
  std::vector<sycl::half> hs(static_cast<size_t>(kk / 128) * n);
  for (auto &x : hs) x = static_cast<sycl::half>(0.0005f * (1 + rng() % 32));
  std::vector<bf16> ha(static_cast<size_t>(m) * kk);
  for (auto &x : ha) x = static_cast<bf16>((static_cast<int>(rng() % 2001) - 1000) * 1e-3f);
  auto *W = sycl::malloc_device<uint8_t>(wb, q);
  auto *S = sycl::malloc_device<sycl::half>(hs.size(), q);
  auto *A = sycl::malloc_device<bf16>(ha.size(), q);
  auto *C = sycl::malloc_device<float>(static_cast<size_t>(m) * n, q);
  auto *H = sycl::malloc_device<bf16>(static_cast<size_t>(m) * n / 2, q);
  q.memcpy(W, nib.data(), wb), q.memcpy(S, hs.data(), hs.size() * 2), q.memcpy(A, ha.data(), ha.size() * 2).wait();
  k::w4a16_pf_set_a8(true);
  k::w4a16_pf_a8_reserve(q, m, kk);
  std::vector<float> c0(static_cast<size_t>(m) * n), c1(c0.size());
  std::vector<bf16> h0(static_cast<size_t>(m) * n / 2), h1(h0.size());
  for (int on = 0; on < 2; ++on) {
    k::w4a16_pf_set_onednn(on ? 6144 : 0);
    k::w4a16_gemm_pf(q, m, A, kk, W, S, C, n, n, kk).wait();
    q.memcpy(on ? c1.data() : c0.data(), C, c0.size() * 4).wait();
    k::w4a16_gemm_pf_swiglu(q, m, A, kk, W, S, H, n / 2, n / 2, kk).wait();
    q.memcpy(on ? h1.data() : h0.data(), H, h0.size() * 2).wait();
  }
  {  // The oneDNN GLU registered h's codes (glu_q8); a W4A8 down on h reads them = its own quantiser's bits
    const int nd = 1024, kd = n / 2;
    std::vector<uint8_t> dnib(static_cast<size_t>(nd) * kd / 2);
    for (auto &b : dnib) b = static_cast<uint8_t>(rng());
    std::vector<sycl::half> ds(static_cast<size_t>(kd / 128) * nd, static_cast<sycl::half>(0.001f));
    auto *Wd = sycl::malloc_device<uint8_t>(dnib.size(), q);
    auto *Sd = sycl::malloc_device<sycl::half>(ds.size(), q);
    auto *Cd = sycl::malloc_device<float>(static_cast<size_t>(m) * nd, q);
    q.memcpy(Wd, dnib.data(), dnib.size()), q.memcpy(Sd, ds.data(), ds.size() * 2).wait();
    k::w4a16_pf_a8_reserve(q, m, std::max(kk, kd));
    std::vector<float> d0(static_cast<size_t>(m) * nd), d1(d0.size());
    k::w4a16_pf_set_onednn(6144);
    k::w4a16_gemm_pf_swiglu(q, m, A, kk, W, S, H, n / 2, n / 2, kk);  // oneDNN GLU: h + its codes
    k::w4a16_gemm_pf(q, m, H, kd, Wd, Sd, Cd, nd, nd, kd).wait();  // reads the registered codes
    q.memcpy(d1.data(), Cd, d1.size() * 4).wait();
    k::w4a16_pf_a8_input_clear(q);
    k::w4a16_gemm_pf(q, m, H, kd, Wd, Sd, Cd, nd, nd, kd).wait();  // its own quantiser on the same h
    q.memcpy(d0.data(), Cd, d0.size() * 4).wait();
    char w2[160];
    std::snprintf(w2, sizeof w2, "W4A8 down m %d k %d on the GLU's registered codes == its own quantiser (bit for bit)", m, kd);
    expect(std::memcmp(d0.data(), d1.data(), d0.size() * 4) == 0, w2);
    sycl::free(Wd, q), sycl::free(Sd, q), sycl::free(Cd, q);
  }
  k::w4a16_pf_set_onednn(0);
  k::w4a16_pf_set_a8(false);
  double mc = 0, dc = 0, mh = 0, dh = 0;
  for (size_t i = 0; i < c0.size(); ++i) mc = std::max(mc, static_cast<double>(std::fabs(c0[i]))), dc = std::max(dc, static_cast<double>(std::fabs(c0[i] - c1[i])));
  for (size_t i = 0; i < h0.size(); ++i) {
    const double a = static_cast<float>(h0[i]), b = static_cast<float>(h1[i]);
    mh = std::max(mh, std::fabs(a)), dh = std::max(dh, std::fabs(a - b));
  }
  {  // host swiglu of the plain output (gate = columns [0, n/2), up = [n/2, n)) vs each GLU
    double e0 = 0, e1 = 0, mr = 0;
    for (int r = 0; r < m; ++r)
      for (int j = 0; j < n / 2; ++j) {
        const float g = static_cast<float>(static_cast<bf16>(c0[static_cast<size_t>(r) * n + j])), u = static_cast<float>(static_cast<bf16>(c0[static_cast<size_t>(r) * n + n / 2 + j]));
        const double ref = g / (1.0 + std::exp(-g)) * u;
        mr = std::max(mr, std::fabs(ref));
        e0 = std::max(e0, std::fabs(ref - static_cast<float>(h0[static_cast<size_t>(r) * n / 2 + j])));
        e1 = std::max(e1, std::fabs(ref - static_cast<float>(h1[static_cast<size_t>(r) * n / 2 + j])));
      }
    std::printf("  host swiglu of the plain output vs ESIMD GLU %.2e, vs oneDNN GLU %.2e (of max |ref| %.3e)\n", e0 / mr, e1 / mr, mr);
  }
  char what[220];
  std::snprintf(what, sizeof what, "W4A8 prefill m %d n %d k %d: oneDNN vs ESIMD, plain worst %.2e, GLU worst %.2e of max |ref|", m, n, kk, dc / mc, dh / std::max(mh, 1e-30));
  expect(dc / mc < 1e-5 && dh / std::max(mh, 1e-30) < 2e-3, what);
  sycl::free(W, q), sycl::free(S, q), sycl::free(A, q), sycl::free(C, q), sycl::free(H, q);
}

// Pf_dense's oneDNN route (int8 matrices, W8A16): the first call of a shape checks oneDNN against the engine's
// body, the later ones run oneDNN only where the bits matched; every call must give the engine's bits. ks > 1 = split-K.
void run_dense(sycl::queue &q, int m, int n, int kk, int ks) {
  std::mt19937 rng(static_cast<uint32_t>(m * 7 + n + kk));
  std::vector<int8_t> codes(static_cast<size_t>(n) * kk), packed(k::dense_packed_elems(n, kk));
  for (auto &c : codes) c = static_cast<int8_t>(static_cast<int>(rng() % 255) - 127);
  k::pack_dense_i8(codes.data(), n, kk, packed.data());
  std::vector<float> sc(static_cast<size_t>(n));
  for (auto &x : sc) x = 0.0005f * (1 + rng() % 64);
  std::vector<bf16> ha(static_cast<size_t>(m) * kk);
  for (auto &x : ha) x = static_cast<bf16>((static_cast<int>(rng() % 2001) - 1000) * 1e-3f);
  auto *W = sycl::malloc_device<int8_t>(packed.size(), q);
  auto *S = sycl::malloc_device<float>(sc.size(), q);
  auto *A = sycl::malloc_device<bf16>(ha.size(), q);
  const size_t out = static_cast<size_t>(ks) * m * n;
  auto *C = sycl::malloc_device<float>(out, q);
  q.memcpy(W, packed.data(), packed.size()), q.memcpy(S, sc.data(), sc.size() * 4), q.memcpy(A, ha.data(), ha.size() * 2).wait();
  k::DenseWeights w;
  w.kind = k::DenseKind::i8, w.w = W, w.scale = S, w.n = n, w.k = kk;
  auto call = [&](std::vector<float> &h) {
    if (ks > 1) k::pf_dense_splitk(q, m, w, A, kk, C, n, ks);
    else k::pf_dense(q, m, w, A, kk, k::PfOut{C, nullptr, n});
    h.resize(out);
    q.memcpy(h.data(), C, out * 4).wait();
  };
  std::vector<float> ref, first, second;
  k::pf_dense_set_onednn(0);
  call(ref);
  k::pf_dense_set_onednn(256);
  call(first);  // the check (the engine's result)
  call(second);  // oneDNN where it matched
  k::pf_dense_set_onednn(0);
  char what[160];
  std::snprintf(what, sizeof what, "pf_dense oneDNN route m %d n %d k %d ks %d: the engine's bits on the check call and after", m, n, kk, ks);
  expect(std::memcmp(ref.data(), first.data(), out * 4) == 0 && std::memcmp(ref.data(), second.data(), out * 4) == 0, what);
  sycl::free(W, q), sycl::free(S, q), sycl::free(A, q), sycl::free(C, q);
}

int main() {
  sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order{}};
  run_dense(q, 512, 2560, 3072, 1);  // qwen38 GDN out half k (oneDNN, same bits)
  run_dense(q, 512, 336, 10240, 16);  // the HC down's 16 k slices (batched)
  run_dense(q, 512, 12800, 2560, 1);  // the PLE kv: oneDNN sums k in another order here -> the engine's body stays
  run(q, 256, 1024);
  run(q, 5120, 3072);
  run_pf(q, 1024, 6144, 1024);
  run_pf(q, 1500, 6144, 1024);
  std::printf("%s: %d failure(s)\n", g_failures ? "FAILED" : "OK", g_failures);
  return g_failures ? 1 : 0;
}
