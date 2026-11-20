// xenithium-bake head-calib: the int2 draft head calibrated on the drafter's hidden states
// (host only; the inputs from head-capture).
// xenithium-bake head-calib --src BF16_DIR --capture FILE --mtp DRAFTER_DIR --out NEW_DIR
// [--eval-rows 2048] [--damp 0.01] [--threads N] [--write gptq|act|rtn|none]
// GPTQ on the int2 format (codes q in 0..3, value (q - 1.5) x an fp32 scale per (row, 128-group)): one
// Hessian H = X^T X / n of the head's inputs (every vocabulary row sees the same inputs), damped, its
// inverse's upper Cholesky factor U; per vocabulary row, the columns in order: a group's scale is chosen
// when the group starts (MSE weighted by diag H, 0.5-1.7 x rms, on the error-updated weights), each
// column is rounded to the nearest level and its error, scaled by 1 / U[i][i], is pushed into the later
// columns along U's row (inside the group at once, past it per group).
// Reported on the held-out rows (the capture's eval rows): top-1 agreement of each head with the BF16
// head (lm_head.weight of the source) and the relative logit error, for: int8 per row (the target's head
// format: the int8-head drafter), RTN int2 (`xenithium-bake mtp`'s), GPTQ int2.
// NEW_DIR = DRAFTER_DIR with draft-head.safetensors replaced (the rest hard-linked).
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <map>
#include <sstream>
#include <stdexcept>
#include <thread>

#include "bake.hpp"
#include "xenithium/kernels/pack.hpp"
#include "xenithium/quant/export.hpp"

namespace xenithium::bake {

namespace q = xenithium::quant;
namespace m = xenithium::model;
namespace k = xenithium::kernels;
namespace fs = std::filesystem;

namespace {

using clk = std::chrono::steady_clock;
double secs(clk::time_point t) { return std::chrono::duration<double>(clk::now() - t).count(); }

void pfor(int n, int threads, const std::function<void(int)> &fn) {
  std::atomic<int> next{0};
  std::vector<std::thread> pool;
  for (int t = 0; t < std::max(1, std::min(threads, n)); ++t)
    pool.emplace_back([&] {
      for (int i; (i = next.fetch_add(1)) < n;) fn(i);
    });
  for (auto &t : pool) t.join();
}

// a. b over n (a multiple of 16) with 16 partial sums: vectorises without reassociation
inline float dot16(const float *a, const float *b, int n) {
  float acc[16] = {};
  for (int i = 0; i < n; i += 16)
    for (int u = 0; u < 16; ++u) acc[u] += a[i + u] * b[i + u];
  float s = 0;
  for (int u = 0; u < 16; ++u) s += acc[u];
  return s;
}

float bf(uint16_t b) { return k::bf16_value(b); }

// Argmax over the vocabulary of W. x for each eval row (W float [V][H]); ties: the lowest id. Also
// accumulates sum (W x - R x)^2 and sum (R x)^2 against ref logits when ref_logits != null.
void argmax_rows(const std::vector<float> &W, int V, int H, const std::vector<float> &X, int E, int threads, std::vector<int32_t> &best) {
  const int B = 64;
  const int nb = (V + B - 1) / B;
  std::vector<float> bv(static_cast<size_t>(nb) * E);
  std::vector<int32_t> bi(static_cast<size_t>(nb) * E);
  pfor(nb, threads, [&](int b) {
    const int v0 = b * B, v1 = std::min(V, v0 + B);
    for (int e = 0; e < E; ++e) {
      float mv = -INFINITY;
      int32_t mi = v0;
      const float *x = X.data() + static_cast<size_t>(e) * H;
      for (int v = v0; v < v1; ++v) {
        const float y = dot16(W.data() + static_cast<size_t>(v) * H, x, H);
        if (y > mv) mv = y, mi = v;
      }
      bv[static_cast<size_t>(b) * E + e] = mv;
      bi[static_cast<size_t>(b) * E + e] = mi;
    }
  });
  best.assign(static_cast<size_t>(E), 0);
  for (int e = 0; e < E; ++e) {
    float mv = -INFINITY;
    int32_t mi = 0;
    for (int b = 0; b < nb; ++b)
      if (bv[static_cast<size_t>(b) * E + e] > mv) mv = bv[static_cast<size_t>(b) * E + e], mi = bi[static_cast<size_t>(b) * E + e];
    best[static_cast<size_t>(e)] = mi;
  }
}

// The rank of token ref[e] under Wq. x (0 = Wq's own top-1) per eval row: does a top-K
// shortlist from the int2 head contain the BF16 head's pick (a rescoring of K rows then recovers it).
void ranks_of(const std::vector<float> &Wq, int V, int H, const std::vector<float> &X, int E, const std::vector<int32_t> &ref, int threads,
              std::vector<int> &rank) {
  std::vector<float> yr(static_cast<size_t>(E));
  for (int e = 0; e < E; ++e) yr[static_cast<size_t>(e)] = dot16(Wq.data() + static_cast<size_t>(ref[static_cast<size_t>(e)]) * H, X.data() + static_cast<size_t>(e) * H, H);
  const int B = 64, nb = (V + B - 1) / B;
  std::vector<int> cnt(static_cast<size_t>(nb) * E, 0);
  pfor(nb, threads, [&](int b) {
    for (int e = 0; e < E; ++e) {
      int c = 0;
      for (int v = b * B; v < std::min(V, (b + 1) * B); ++v)
        c += dot16(Wq.data() + static_cast<size_t>(v) * H, X.data() + static_cast<size_t>(e) * H, H) > yr[static_cast<size_t>(e)];
      cnt[static_cast<size_t>(b) * E + e] = c;
    }
  });
  rank.assign(static_cast<size_t>(E), 0);
  for (int b = 0; b < nb; ++b)
    for (int e = 0; e < E; ++e) rank[static_cast<size_t>(e)] += cnt[static_cast<size_t>(b) * E + e];
}

// sum over eval rows and vocab of (Wq x - W x)^2 / sum (W x)^2, on a vocabulary subsample (every 16th row)
double logit_err(const std::vector<float> &Wq, const std::vector<float> &W, int V, int H, const std::vector<float> &X, int E, int threads) {
  const int stride = 16, nv = (V + stride - 1) / stride;
  std::vector<double> num(static_cast<size_t>(nv)), den(static_cast<size_t>(nv));
  pfor(nv, threads, [&](int i) {
    const int v = i * stride;
    double a = 0, b = 0;
    for (int e = 0; e < E; ++e) {
      const float *x = X.data() + static_cast<size_t>(e) * H;
      const double r = dot16(W.data() + static_cast<size_t>(v) * H, x, H), y = dot16(Wq.data() + static_cast<size_t>(v) * H, x, H);
      a += (y - r) * (y - r);
      b += r * r;
    }
    num[static_cast<size_t>(i)] = a, den[static_cast<size_t>(i)] = b;
  });
  double a = 0, b = 0;
  for (int i = 0; i < nv; ++i) a += num[static_cast<size_t>(i)], b += den[static_cast<size_t>(i)];
  return std::sqrt(a / b);
}

// Cholesky A = L L^T in place (lower), double, n x n row-major. Throws if not positive definite.
void cholesky(std::vector<double> &A, int n, int threads) {
  for (int j = 0; j < n; ++j) {
    double d = A[static_cast<size_t>(j) * n + j];
    for (int p = 0; p < j; ++p) d -= A[static_cast<size_t>(j) * n + p] * A[static_cast<size_t>(j) * n + p];
    if (d <= 0) throw std::runtime_error("head-calib: Hessian not positive definite (raise --damp)");
    const double ljj = std::sqrt(d);
    A[static_cast<size_t>(j) * n + j] = ljj;
    const double *rj = A.data() + static_cast<size_t>(j) * n;
    pfor((n - j - 1 + 63) / 64, threads, [&](int blk) {
      for (int i = j + 1 + blk * 64; i < std::min(n, j + 1 + (blk + 1) * 64); ++i) {
        double *ri = A.data() + static_cast<size_t>(i) * n;
        double s = ri[j];
        for (int p = 0; p < j; ++p) s -= ri[p] * rj[p];
        ri[j] = s / ljj;
      }
    });
  }
  for (int i = 0; i < n; ++i)
    for (int j = i + 1; j < n; ++j) A[static_cast<size_t>(i) * n + j] = 0;
}

// Inverse of L L^T from L (lower): Linv by forward substitution, then Linv^T Linv.
std::vector<double> chol_inverse(const std::vector<double> &L, int n, int threads) {
  std::vector<double> Li(static_cast<size_t>(n) * n, 0.0);  // Li lower
  pfor(n, threads, [&](int c) {  // column c of Linv: solve L y = e_c
    std::vector<double> y(static_cast<size_t>(n), 0.0);
    y[static_cast<size_t>(c)] = 1.0 / L[static_cast<size_t>(c) * n + c];
    for (int i = c + 1; i < n; ++i) {
      double s = 0;
      for (int p = c; p < i; ++p) s += L[static_cast<size_t>(i) * n + p] * y[static_cast<size_t>(p)];
      y[static_cast<size_t>(i)] = -s / L[static_cast<size_t>(i) * n + i];
    }
    for (int i = c; i < n; ++i) Li[static_cast<size_t>(i) * n + c] = y[static_cast<size_t>(i)];
  });
  std::vector<double> R(static_cast<size_t>(n) * n, 0.0);
  pfor(n, threads, [&](int i) {  // R[i][j] = sum_p Li[p][i] Li[p][j], p >= max(i, j)
    for (int j = i; j < n; ++j) {
      double s = 0;
      for (int p = j; p < n; ++p) s += Li[static_cast<size_t>(p) * n + i] * Li[static_cast<size_t>(p) * n + j];
      R[static_cast<size_t>(i) * n + j] = s;
      R[static_cast<size_t>(j) * n + i] = s;
    }
  });
  return R;
}

// Weighted-MSE scale of a 128-group (levels (q - 1.5) s), d = the per-column weights.
float group_scale(const float *v, const float *d, int n) {
  double ss = 0;
  for (int i = 0; i < n; ++i) ss += static_cast<double>(v[i]) * v[i];
  const double rms = std::sqrt(ss / n);
  if (rms <= 0) return 0.f;
  float best_s = 0.f;
  double best_e = -1;
  for (int t = 0; t <= 48; ++t) {  // 0.5.. 1.7 x rms, step 0.025
    const float sc = static_cast<float>(rms * (0.5 + 0.025 * t));
    double e = 0;
    for (int i = 0; i < n; ++i) {
      const float qv = std::clamp(std::floor(v[i] / sc) + 2.f, 0.f, 3.f);
      const double dl = static_cast<double>(v[i]) - (qv - 1.5) * sc;
      e += d[i] * dl * dl;
    }
    if (best_e < 0 || e < best_e) best_e = e, best_s = sc;
  }
  return best_s;
}

}  // namespace

int cmd_head_calib(const Args &a) {
  const std::string src = a.str("src", home() + "/models/Qwen3.8-Flash-Next-Uncensored");
  const std::string cap = a.str("capture"), mtp = a.str("mtp"), out = a.str("out"), write = a.str("write", "gptq");
  if (cap.empty() || (write != "none" && (mtp.empty() || out.empty()))) throw std::runtime_error("--capture, --mtp and --out are required");
  const int threads = static_cast<int>(a.num("threads", std::thread::hardware_concurrency()));
  const double damp = std::strtod(a.str("damp", "0.01").c_str(), nullptr);
  const auto t0 = clk::now();

  // ---- the capture ----
  std::ifstream f(cap, std::ios::binary);
  uint32_t magic = 0;
  int32_t H = 0;
  int64_t ntr = 0, nev = 0;
  f.read(reinterpret_cast<char *>(&magic), 4).read(reinterpret_cast<char *>(&H), 4).read(reinterpret_cast<char *>(&ntr), 8).read(reinterpret_cast<char *>(&nev), 8);
  if (!f || magic != 0x31494858u || H % 128) throw std::runtime_error("head-calib: not an XHI1 capture: " + cap);
  const int E = static_cast<int>(std::min<int64_t>(nev, a.num("eval-rows", 2048)));
  std::vector<uint16_t> xb(static_cast<size_t>(ntr + nev) * H);
  f.read(reinterpret_cast<char *>(xb.data()), static_cast<std::streamsize>(xb.size() * 2));
  if (!f) throw std::runtime_error("head-calib: short capture");
  std::vector<float> Xev(static_cast<size_t>(E) * H);
  for (size_t i = 0; i < Xev.size(); ++i) Xev[i] = bf(xb[static_cast<size_t>(ntr) * H + i]);
  // --eval-train N: N training rows (an even stride: every source) appended to the eval rows;
  // the held-out samples alone are 1-2 texts
  const int Et = static_cast<int>(a.num("eval-train", 0));
  for (int j = 0; j < Et; ++j) {
    const int64_t r = static_cast<int64_t>(j) * ntr / Et;
    for (int c = 0; c < H; ++c) Xev.push_back(bf(xb[static_cast<size_t>(r) * H + c]));
  }
  const int Eall = E + Et;

  // ---- the BF16 head ----
  const auto ck = m::Checkpoint::open(src);
  const auto &tl = ck.at("lm_head.weight");
  if (tl.dtype != m::DType::bf16 || tl.shape.size() != 2 || tl.dim(1) != H) throw std::runtime_error("head-calib: lm_head.weight is not BF16 [vocab, hidden]");
  const int V = static_cast<int>(tl.dim(0)), G = H / 128;
  std::vector<uint16_t> wb(static_cast<size_t>(V) * H);
  read_tensor_range(ck, tl, 0, tl.bytes, wb.data());
  std::vector<float> W(wb.size());
  for (size_t i = 0; i < W.size(); ++i) W[i] = bf(wb[i]);
  std::printf("capture %lld train / %lld eval rows (eval on %d), head %d x %d (%.0f s)\n", static_cast<long long>(ntr), static_cast<long long>(nev), E, V, H, secs(t0));

  // ---- the Hessian of the head's inputs ----
  auto t1 = clk::now();
  std::vector<double> Hs(static_cast<size_t>(H) * H, 0.0);
  {
    const int R = 1024;
    std::vector<float> XT(static_cast<size_t>(H) * R);
    for (int64_t r0 = 0; r0 < ntr; r0 += R) {
      const int rn = static_cast<int>(std::min<int64_t>(R, ntr - r0));
      std::fill(XT.begin(), XT.end(), 0.f);
      pfor(H / 64, threads, [&](int cb) {
        for (int r = 0; r < rn; ++r)
          for (int c = cb * 64; c < cb * 64 + 64; ++c) XT[static_cast<size_t>(c) * R + r] = bf(xb[static_cast<size_t>(r0 + r) * H + c]);
      });
      pfor(H, threads, [&](int i) {
        const float *xi = XT.data() + static_cast<size_t>(i) * R;
        double *hi = Hs.data() + static_cast<size_t>(i) * H;
        for (int j = i; j < H; ++j) hi[j] += dot16(xi, XT.data() + static_cast<size_t>(j) * R, R);
      });
    }
    double dmean = 0;
    for (int i = 0; i < H; ++i) {
      for (int j = 0; j < i; ++j) Hs[static_cast<size_t>(i) * H + j] = Hs[static_cast<size_t>(j) * H + i];
      dmean += Hs[static_cast<size_t>(i) * H + i];
    }
    dmean /= H;
    for (int i = 0; i < H; ++i) Hs[static_cast<size_t>(i) * H + i] += damp * dmean;
  }
  std::vector<float> dg(static_cast<size_t>(H));
  for (int i = 0; i < H; ++i) dg[static_cast<size_t>(i)] = static_cast<float>(Hs[static_cast<size_t>(i) * H + i]);
  std::printf("Hessian: %lld rows, damp %.3g x mean diag (%.0f s)\n", static_cast<long long>(ntr), damp, secs(t1));
  t1 = clk::now();
  // U: upper Cholesky factor of H^-1 (H^-1 = U^T U), of H with its columns in the order perm
  auto make_U = [&](const std::vector<int> &perm) {
    std::vector<double> Hp(static_cast<size_t>(H) * H);
    for (int i = 0; i < H; ++i)
      for (int j = 0; j < H; ++j) Hp[static_cast<size_t>(i) * H + j] = Hs[static_cast<size_t>(perm[static_cast<size_t>(i)]) * H + perm[static_cast<size_t>(j)]];
    cholesky(Hp, H, threads);
    std::vector<double> Hi = chol_inverse(Hp, H, threads);
    cholesky(Hi, H, threads);  // Hi = Lh (lower); U = Lh^T
    std::vector<float> Uo(static_cast<size_t>(H) * H);
    for (int i = 0; i < H; ++i)
      for (int j = 0; j < H; ++j) Uo[static_cast<size_t>(i) * H + j] = j >= i ? static_cast<float>(Hi[static_cast<size_t>(j) * H + i]) : 0.f;
    return Uo;
  };
  std::vector<int> ident(static_cast<size_t>(H)), act(static_cast<size_t>(H));
  for (int i = 0; i < H; ++i) ident[static_cast<size_t>(i)] = act[static_cast<size_t>(i)] = i;
  std::stable_sort(act.begin(), act.end(), [&](int x, int y) { return dg[static_cast<size_t>(x)] > dg[static_cast<size_t>(y)]; });
  const std::vector<float> U = make_U(ident);
  const std::vector<float> Ua = make_U(act);
  std::printf("H^-1 Cholesky (%.0f s)\n", secs(t1));

  // ---- the heads ----
  std::vector<int32_t> ref, got;
  t1 = clk::now();
  argmax_rows(W, V, H, Xev, Eall, threads, ref);
  std::printf("BF16 head picks on %d held-out + %d training rows (%.0f s)\n", E, Et, secs(t1));
  auto report = [&](const char *name, const std::vector<float> &Wq) {
    const auto tr = clk::now();
    argmax_rows(Wq, V, H, Xev, Eall, threads, got);
    int same = 0, same_t = 0;
    for (int e = 0; e < E; ++e) same += got[static_cast<size_t>(e)] == ref[static_cast<size_t>(e)];
    for (int e = E; e < Eall; ++e) same_t += got[static_cast<size_t>(e)] == ref[static_cast<size_t>(e)];
    const double le = logit_err(Wq, W, V, H, Xev, std::min(E, 256), threads);
    std::printf("  %-22s top-1 == BF16: held-out %6.2f %%, training rows %6.2f %%   logit rel. error %.4f   (%.0f s)\n", name, 100.0 * same / std::max(E, 1),
                100.0 * same_t / std::max(Et, 1), le, secs(tr));
    if (a.num("recall", 0)) {
      std::vector<int> rk;
      ranks_of(Wq, V, H, Xev, Eall, ref, threads, rk);
      std::printf("    BF16 pick inside its top-K (all %d rows):", Eall);
      for (int K : {1, 2, 4, 8, 16, 32, 64}) {
        int in = 0;
        for (int e = 0; e < Eall; ++e) in += rk[static_cast<size_t>(e)] < K;
        std::printf("  K=%d %.2f %%", K, 100.0 * in / Eall);
      }
      std::printf("\n");
    }
    std::fflush(stdout);
  };
  {  // int8 per row (the target head's format)
    std::vector<float> Wq(W.size());
    pfor(V, threads, [&](int v) {
      const float *w = W.data() + static_cast<size_t>(v) * H;
      float amax = 0;
      for (int i = 0; i < H; ++i) amax = std::max(amax, std::fabs(w[i]));
      const float s = amax > 0 ? amax / 127.f : 1.f;
      for (int i = 0; i < H; ++i) Wq[static_cast<size_t>(v) * H + i] = std::nearbyint(w[i] / s) * s;
    });
    report("int8 per row", Wq);
  }
  std::vector<uint8_t> codes(static_cast<size_t>(V) * H);
  std::vector<float> scale(static_cast<size_t>(G) * V);
  std::vector<float> Wq(W.size());
  auto deq = [&] {
    pfor(V, threads, [&](int v) {
      for (int c = 0; c < H; ++c)
        Wq[static_cast<size_t>(v) * H + c] = (codes[static_cast<size_t>(v) * H + c] - 1.5f) * scale[static_cast<size_t>(c / 128) * V + v];
    });
  };
  // RTN int2 (xenithium-bake mtp's)
  pfor((V + 1023) / 1024, threads, [&](int b) { k::quantize_i2_groups(wb.data(), V, H, codes.data(), scale.data(), b * 1024, std::min(V, (b + 1) * 1024)); });
  deq();
  report("int2 RTN (round 64)", Wq);
  std::vector<uint8_t> rtn_codes = codes;
  std::vector<float> rtn_scale = scale;
  // GPTQ int2
  t1 = clk::now();
  pfor(V, threads, [&](int v) {
    std::vector<float> w(W.begin() + static_cast<std::ptrdiff_t>(v) * H, W.begin() + static_cast<std::ptrdiff_t>(v + 1) * H);
    float err[128];
    for (int g = 0; g < G; ++g) {
      const int b0 = g * 128, b1 = b0 + 128;
      const float s = group_scale(w.data() + b0, dg.data() + b0, 128);
      scale[static_cast<size_t>(g) * V + v] = s;
      for (int i = b0; i < b1; ++i) {
        const float qv = s > 0.f ? std::clamp(std::floor(w[static_cast<size_t>(i)] / s) + 2.f, 0.f, 3.f) : 1.f;
        codes[static_cast<size_t>(v) * H + i] = static_cast<uint8_t>(qv);
        const float e = (w[static_cast<size_t>(i)] - (qv - 1.5f) * s) / U[static_cast<size_t>(i) * H + i];
        err[i - b0] = e;
        const float *ui = U.data() + static_cast<size_t>(i) * H;
        for (int j = i + 1; j < b1; ++j) w[static_cast<size_t>(j)] -= e * ui[j];
      }
      for (int i = b0; i < b1; ++i) {  // the group's error into the later groups
        const float e = err[i - b0];
        const float *ui = U.data() + static_cast<size_t>(i) * H;
        for (int j = b1; j < H; ++j) w[static_cast<size_t>(j)] -= e * ui[j];
      }
    }
  });
  std::printf("GPTQ int2: %d rows (%.0f s)\n", V, secs(t1));
  deq();
  report("int2 GPTQ", Wq);
  std::vector<uint8_t> g_codes = codes;
  std::vector<float> g_scale = scale;
  // GPTQ with act order (columns by descending diag H) and static group scales (weighted MSE on the
  // original row): the high-energy columns first, their error absorbed by the rest
  t1 = clk::now();
  pfor(V, threads, [&](int v) {
    const float *w0 = W.data() + static_cast<size_t>(v) * H;
    for (int g = 0; g < G; ++g) scale[static_cast<size_t>(g) * V + v] = group_scale(w0 + g * 128, dg.data() + g * 128, 128);
    std::vector<float> w(static_cast<size_t>(H));
    for (int i = 0; i < H; ++i) w[static_cast<size_t>(i)] = w0[act[static_cast<size_t>(i)]];
    for (int i = 0; i < H; ++i) {
      const int col = act[static_cast<size_t>(i)];
      const float s = scale[static_cast<size_t>(col / 128) * V + v];
      const float qv = s > 0.f ? std::clamp(std::floor(w[static_cast<size_t>(i)] / s) + 2.f, 0.f, 3.f) : 1.f;
      codes[static_cast<size_t>(v) * H + col] = static_cast<uint8_t>(qv);
      const float e = (w[static_cast<size_t>(i)] - (qv - 1.5f) * s) / Ua[static_cast<size_t>(i) * H + i];
      const float *ui = Ua.data() + static_cast<size_t>(i) * H;
      for (int j = i + 1; j < H; ++j) w[static_cast<size_t>(j)] -= e * ui[j];
    }
  });
  std::printf("GPTQ int2 act order: %d rows (%.0f s)\n", V, secs(t1));
  deq();
  report("int2 GPTQ act order", Wq);
  if (write == "gptq") codes = g_codes, scale = g_scale;
  if (write == "none") return 0;
  if (write == "rtn") codes = rtn_codes, scale = rtn_scale;
  if (write != "gptq" && write != "act" && write != "rtn") throw std::runtime_error("--write gptq|act|rtn|none");

  // ---- write NEW_DIR ----
  if (fs::exists(out) && fs::equivalent(mtp, out)) throw std::runtime_error("--out must not be --mtp");
  fs::create_directories(out);
  for (const auto &e : fs::directory_iterator(mtp)) {
    const std::string name = e.path().filename().string();
    if (name == "draft-head.safetensors" || !e.is_regular_file()) continue;
    std::error_code ec;
    fs::remove(out + "/" + name);
    fs::create_hard_link(e.path(), out + "/" + name, ec);
    if (ec) fs::copy_file(e.path(), out + "/" + name, fs::copy_options::overwrite_existing);
  }
  std::vector<uint8_t> packed(static_cast<size_t>(V) * H / 4, 0);
  for (size_t i = 0; i < codes.size(); ++i) packed[i / 4] |= static_cast<uint8_t>(codes[i] << (2 * (i % 4)));
  q::write_safetensors(out + "/draft-head.safetensors",
                       {{"mtp.draft_head.codes", "U8", {V, H / 4}, packed.data(), packed.size()},
                        {"mtp.draft_head.scales", "F32", {G, V}, scale.data(), scale.size() * 4}},
                       {{"format", "pt"}, {"xenithium_scheme", std::string("draft-head-int2-g128-f32scale-") + write}, {"calibration", cap}});
  std::printf("draft head (%s) -> %s/draft-head.safetensors (%.0f s)\n", write.c_str(), out.c_str(), secs(t0));
  return 0;
}


// ============================================================================================================
// xenithium-bake head4-calib: the dense 27B's lm_head as int4 g128 symmetric (the trunk's
// W4 format: code q - 8 in [-8, 7] times a signed fp16 scale per (vocabulary row, 128-k group)), GPTQ on the head's
// generation-time inputs.
// xenithium-bake head4-calib --capture FILE --src BF16_DIR --base SERVED_DIR --out NEW_DIR
// [--eval-rows 1024] [--damp 0.01] [--threads N] [--write gptq|act|rtn|mse|none]
// FILE = an XHI1 capture of the final-normed hidden states (`probe model35 --head-x-out`): training rows (the
// Hessian) then held-out rows (the report). The BF16 head is `lm_head.weight` of --src (checked byte-equal to
// --base's). Heads reported on the held-out rows (every n-th, --eval-rows of them) against the BF16 head: top-1
// agreement, full-vocabulary KL(BF16 || quantised) in nats, relative logit error; for int8 per row (today's served
// head), RTN int4 (the group's largest |w| on code -8, the AutoRound sign convention), RTN int4 with a
// weighted-MSE clip, GPTQ int4 (scale per group at the group's start on the error-updated weights, weighted-MSE
// clip; error pushed along the upper Cholesky factor of H^-1) and GPTQ with act order (static scales).
// NEW_DIR = SERVED_DIR hard-linked + lm_head-w4.safetensors (lm_head.qweight I32 [k/8][n], lm_head.scales F16
// [k/128][n], lm_head.qzeros I32 [k/128][n/8] = 0x77777777: the GPTQ orientation the served trunk uses) and an index
// copy listing them; lm_head.weight stays (the engine picks the head format by a switch).
namespace {

// one fp16 rounding (the stored scale)
float to_half(float x) { return static_cast<float>(static_cast<_Float16>(x)); }
uint16_t half_bits(float x) {
  const _Float16 h = static_cast<_Float16>(x);
  uint16_t b;
  std::memcpy(&b, &h, 2);
  return b;
}

// The signed scale of a 128-group by weighted MSE: s = -extreme / 8 x alpha (alpha 1 = the plain AutoRound / RTN
// scale: the extreme lands on code -8), alpha in [0.70, 1.0]; d = per-column weights (diag H; null = 1). The scale
// is rounded to fp16 before the codes are taken (the stored value).
float group_scale4(const float *v, const float *d, int n, bool search) {
  int im = 0;
  for (int i = 1; i < n; ++i)
    if (std::fabs(v[i]) > std::fabs(v[im])) im = i;
  if (v[im] == 0.f) return 0.f;
  const float s0 = -v[im] / 8.f;
  float best_s = to_half(s0);
  if (!search) return best_s;
  double best_e = -1;
  for (int t = 0; t <= 30; ++t) {
    const float sc = to_half(s0 * (1.f - 0.01f * t));
    if (sc == 0.f) continue;
    double e = 0;
    for (int i = 0; i < n; ++i) {
      const float qv = std::clamp(std::nearbyint(v[i] / sc), -8.f, 7.f);
      const double dl = static_cast<double>(v[i]) - qv * sc;
      e += (d ? d[i] : 1.f) * dl * dl;
    }
    if (best_e < 0 || e < best_e) best_e = e, best_s = sc;
  }
  return best_s;
}

struct HeadStats {
  double top1 = 0, kl = 0, kl_max = 0;
};

// Held-out comparison of Wq against W (both float [V][H]) on rows X [E][H], full vocabulary, one pass:
// per (64-row vocabulary block, eval row) partials of logsumexp for both heads and of sum p_ref (zr - zq).
HeadStats compare_heads(const std::vector<float> &W, const std::vector<float> &Wq, int V, int H, const std::vector<float> &X, int E, int threads) {
  const int B = 64, nb = (V + B - 1) / B;
  struct P {
    float mr, sr, a, mq, sq, br, bq;
    int ir, iq;
  };
  std::vector<P> part(static_cast<size_t>(nb) * E);
  pfor(nb, threads, [&](int b) {
    const int v0 = b * B, v1 = std::min(V, v0 + B);
    float zr[64], zq[64];
    for (int e = 0; e < E; ++e) {
      const float *x = X.data() + static_cast<size_t>(e) * H;
      P p{-INFINITY, 0, 0, -INFINITY, 0, -INFINITY, -INFINITY, v0, v0};
      for (int v = v0; v < v1; ++v) {
        zr[v - v0] = dot16(W.data() + static_cast<size_t>(v) * H, x, H);
        zq[v - v0] = dot16(Wq.data() + static_cast<size_t>(v) * H, x, H);
        p.mr = std::max(p.mr, zr[v - v0]);
        p.mq = std::max(p.mq, zq[v - v0]);
        if (zr[v - v0] > p.br) p.br = zr[v - v0], p.ir = v;
        if (zq[v - v0] > p.bq) p.bq = zq[v - v0], p.iq = v;
      }
      for (int v = v0; v < v1; ++v) {
        const float er = std::exp(zr[v - v0] - p.mr);
        p.sr += er;
        p.a += er * (zr[v - v0] - zq[v - v0]);
        p.sq += std::exp(zq[v - v0] - p.mq);
      }
      part[static_cast<size_t>(b) * E + e] = p;
    }
  });
  HeadStats st;
  for (int e = 0; e < E; ++e) {
    double mr = -INFINITY, mq = -INFINITY, br = -INFINITY, bq = -INFINITY;
    int ir = 0, iq = 0;
    for (int b = 0; b < nb; ++b) {
      const P &p = part[static_cast<size_t>(b) * E + e];
      mr = std::max(mr, static_cast<double>(p.mr)), mq = std::max(mq, static_cast<double>(p.mq));
      if (p.br > br) br = p.br, ir = p.ir;
      if (p.bq > bq) bq = p.bq, iq = p.iq;
    }
    double sr = 0, sq = 0, a = 0;
    for (int b = 0; b < nb; ++b) {
      const P &p = part[static_cast<size_t>(b) * E + e];
      const double fr = std::exp(p.mr - mr);
      sr += p.sr * fr, a += p.a * fr, sq += p.sq * std::exp(p.mq - mq);
    }
    const double lser = mr + std::log(sr), lseq = mq + std::log(sq);
    const double kl = a / sr - lser + lseq;
    st.kl += kl, st.kl_max = std::max(st.kl_max, kl);
    st.top1 += ir == iq;
  }
  st.kl /= E, st.top1 = 100.0 * st.top1 / E;
  return st;
}

}  // namespace

int cmd_head4_calib(const Args &a) {
  const std::string cap = a.str("capture"), src = a.str("src", home() + "/models/swift15-27b-full"), base = a.str("base", home() + "/models/swift15-w4a16-autoround");
  const std::string out = a.str("out"), write_arg = a.str("write", "gptq");
  if (cap.empty() || (write_arg != "none" && out.empty())) throw std::runtime_error("--capture and --out are required");
  std::vector<std::string> writes;  // --write a,b: one NEW_DIR per variant (OUT-a, OUT-b; OUT alone for one variant)
  {
    std::stringstream ss(write_arg);
    for (std::string w; std::getline(ss, w, ',');)
      if (!w.empty() && w != "none") {
        if (w != "gptq" && w != "act" && w != "rtn" && w != "mse") throw std::runtime_error("--write gptq|act|rtn|mse|none (comma list)");
        writes.push_back(w);
      }
  }
  auto wants = [&](const char *w) { return std::find(writes.begin(), writes.end(), w) != writes.end(); };
  const int threads = static_cast<int>(a.num("threads", std::thread::hardware_concurrency()));
  const double damp = std::strtod(a.str("damp", "0.01").c_str(), nullptr);
  const auto t0 = clk::now();

  // ---- the capture ----
  std::ifstream f(cap, std::ios::binary);
  uint32_t magic = 0;
  int32_t H = 0;
  int64_t ntr = 0, nev = 0;
  f.read(reinterpret_cast<char *>(&magic), 4).read(reinterpret_cast<char *>(&H), 4).read(reinterpret_cast<char *>(&ntr), 8).read(reinterpret_cast<char *>(&nev), 8);
  if (!f || magic != 0x31494858u || H % 128 || ntr < 1 || nev < 1) throw std::runtime_error("head4-calib: not an XHI1 capture with training and held-out rows: " + cap);
  std::vector<uint16_t> xb(static_cast<size_t>(ntr + nev) * H);
  f.read(reinterpret_cast<char *>(xb.data()), static_cast<std::streamsize>(xb.size() * 2));
  if (!f) throw std::runtime_error("head4-calib: short capture");
  const int E = static_cast<int>(std::min<int64_t>(nev, a.num("eval-rows", 1024)));
  std::vector<float> Xev(static_cast<size_t>(E) * H);
  for (int e = 0; e < E; ++e) {  // every n-th held-out row: all the held-out prompts, not the first few
    const int64_t r = ntr + static_cast<int64_t>(e) * nev / E;
    for (int c = 0; c < H; ++c) Xev[static_cast<size_t>(e) * H + c] = bf(xb[static_cast<size_t>(r) * H + c]);
  }

  // ---- the BF16 head (the full checkpoint's, == the served directory's) ----
  const auto ck = m::Checkpoint::open(src);
  const auto &tl = ck.at("lm_head.weight");
  if (tl.dtype != m::DType::bf16 || tl.shape.size() != 2 || tl.dim(1) != H) throw std::runtime_error("head4-calib: lm_head.weight is not BF16 [vocab, hidden]");
  const int V = static_cast<int>(tl.dim(0)), G = H / 128;
  std::vector<uint16_t> wb(static_cast<size_t>(V) * H);
  read_tensor_range(ck, tl, 0, tl.bytes, wb.data());
  {
    const auto cb = m::Checkpoint::open(base);
    const auto &tb = cb.at("lm_head.weight");
    if (tb.bytes != tl.bytes) throw std::runtime_error("head4-calib: --base lm_head.weight has another size");
    std::vector<uint16_t> wb2(wb.size());
    read_tensor_range(cb, tb, 0, tb.bytes, wb2.data());
    const bool same = std::memcmp(wb.data(), wb2.data(), wb.size() * 2) == 0;
    std::printf("lm_head.weight of %s %s that of %s\n", src.c_str(), same ? "is byte-identical to" : "DIFFERS from", base.c_str());
  }
  std::vector<float> W(wb.size());
  for (size_t i = 0; i < W.size(); ++i) W[i] = bf(wb[i]);
  std::printf("capture %lld training / %lld held-out rows (report on %d), head %d x %d (%.0f s)\n", static_cast<long long>(ntr), static_cast<long long>(nev), E, V, H, secs(t0));

  // ---- the Hessian of the head's inputs (training rows) ----
  auto t1 = clk::now();
  std::vector<double> Hs(static_cast<size_t>(H) * H, 0.0);
  {
    const int R = 1024;
    std::vector<float> XT(static_cast<size_t>(H) * R);
    for (int64_t r0 = 0; r0 < ntr; r0 += R) {
      const int rn = static_cast<int>(std::min<int64_t>(R, ntr - r0));
      std::fill(XT.begin(), XT.end(), 0.f);
      pfor(H / 64, threads, [&](int cb) {
        for (int r = 0; r < rn; ++r)
          for (int c = cb * 64; c < cb * 64 + 64; ++c) XT[static_cast<size_t>(c) * R + r] = bf(xb[static_cast<size_t>(r0 + r) * H + c]);
      });
      pfor(H, threads, [&](int i) {
        const float *xi = XT.data() + static_cast<size_t>(i) * R;
        double *hi = Hs.data() + static_cast<size_t>(i) * H;
        for (int j = i; j < H; ++j) hi[j] += dot16(xi, XT.data() + static_cast<size_t>(j) * R, R);
      });
    }
    double dmean = 0;
    for (int i = 0; i < H; ++i) {
      for (int j = 0; j < i; ++j) Hs[static_cast<size_t>(i) * H + j] = Hs[static_cast<size_t>(j) * H + i];
      dmean += Hs[static_cast<size_t>(i) * H + i];
    }
    dmean /= H;
    for (int i = 0; i < H; ++i) Hs[static_cast<size_t>(i) * H + i] += damp * dmean;
  }
  xb.clear();
  xb.shrink_to_fit();
  std::vector<float> dg(static_cast<size_t>(H));
  for (int i = 0; i < H; ++i) dg[static_cast<size_t>(i)] = static_cast<float>(Hs[static_cast<size_t>(i) * H + i]);
  std::printf("Hessian: %lld rows, damp %.3g x mean diag (%.0f s)\n", static_cast<long long>(ntr), damp, secs(t1));
  t1 = clk::now();
  auto make_U = [&](const std::vector<int> &perm) {
    std::vector<double> Hp(static_cast<size_t>(H) * H);
    for (int i = 0; i < H; ++i)
      for (int j = 0; j < H; ++j) Hp[static_cast<size_t>(i) * H + j] = Hs[static_cast<size_t>(perm[static_cast<size_t>(i)]) * H + perm[static_cast<size_t>(j)]];
    cholesky(Hp, H, threads);
    std::vector<double> Hi = chol_inverse(Hp, H, threads);
    cholesky(Hi, H, threads);
    std::vector<float> Uo(static_cast<size_t>(H) * H);
    for (int i = 0; i < H; ++i)
      for (int j = 0; j < H; ++j) Uo[static_cast<size_t>(i) * H + j] = j >= i ? static_cast<float>(Hi[static_cast<size_t>(j) * H + i]) : 0.f;
    return Uo;
  };
  std::vector<int> ident(static_cast<size_t>(H)), act(static_cast<size_t>(H));
  for (int i = 0; i < H; ++i) ident[static_cast<size_t>(i)] = act[static_cast<size_t>(i)] = i;
  std::stable_sort(act.begin(), act.end(), [&](int x, int y) { return dg[static_cast<size_t>(x)] > dg[static_cast<size_t>(y)]; });
  const std::vector<float> U = make_U(ident);
  std::printf("H^-1 Cholesky (%.0f s)\n", secs(t1));

  // ---- the heads ----
  std::vector<float> Wq(W.size());
  auto report = [&](const char *name) {
    const auto tr = clk::now();
    const HeadStats s = compare_heads(W, Wq, V, H, Xev, E, threads);
    std::printf("  %-26s top-1 == BF16 %6.2f %%   KL(BF16 || q) mean %.6f max %.4f nats   logit rel. error %.4f   (%.0f s)\n", name, s.top1, s.kl, s.kl_max,
                logit_err(Wq, W, V, H, Xev, std::min(E, 256), threads), secs(tr));
    std::fflush(stdout);
  };
  {  // int8 per row (the served head today: quantize_i8_rows, clips 1)
    pfor(V, threads, [&](int v) {
      const float *w = W.data() + static_cast<size_t>(v) * H;
      float amax = 0;
      for (int i = 0; i < H; ++i) amax = std::max(amax, std::fabs(w[i]));
      const float s = amax > 0 ? amax / 127.f : 1.f;
      for (int i = 0; i < H; ++i) Wq[static_cast<size_t>(v) * H + i] = std::nearbyint(w[i] / s) * s;
    });
    report("int8 per row (served)");
  }
  std::vector<int8_t> codes(static_cast<size_t>(V) * H);
  std::vector<float> scale(static_cast<size_t>(G) * V);
  auto deq = [&] {
    pfor(V, threads, [&](int v) {
      for (int c = 0; c < H; ++c) Wq[static_cast<size_t>(v) * H + c] = codes[static_cast<size_t>(v) * H + c] * scale[static_cast<size_t>(c / 128) * V + v];
    });
  };
  auto rtn = [&](bool search) {
    pfor(V, threads, [&](int v) {
      const float *w = W.data() + static_cast<size_t>(v) * H;
      for (int g = 0; g < G; ++g) {
        const float s = group_scale4(w + g * 128, search ? dg.data() + g * 128 : nullptr, 128, search);
        scale[static_cast<size_t>(g) * V + v] = s;
        for (int i = g * 128; i < g * 128 + 128; ++i)
          codes[static_cast<size_t>(v) * H + i] = static_cast<int8_t>(s != 0.f ? std::clamp(std::nearbyint(w[i] / s), -8.f, 7.f) : 0.f);
      }
    });
    deq();
  };
  std::map<std::string, std::pair<std::vector<int8_t>, std::vector<float>>> keep;
  rtn(false);
  report("int4 RTN (extreme on -8)");
  if (wants("rtn")) keep["rtn"] = {codes, scale};
  rtn(true);
  report("int4 RTN, weighted-MSE clip");
  if (wants("mse")) keep["mse"] = {codes, scale};
  // GPTQ: columns in order, the group's scale at its start on the error-updated weights
  t1 = clk::now();
  pfor(V, threads, [&](int v) {
    std::vector<float> w(W.begin() + static_cast<std::ptrdiff_t>(v) * H, W.begin() + static_cast<std::ptrdiff_t>(v + 1) * H);
    float err[128];
    for (int g = 0; g < G; ++g) {
      const int b0 = g * 128, b1 = b0 + 128;
      const float s = group_scale4(w.data() + b0, dg.data() + b0, 128, true);
      scale[static_cast<size_t>(g) * V + v] = s;
      for (int i = b0; i < b1; ++i) {
        const float qv = s != 0.f ? std::clamp(std::nearbyint(w[static_cast<size_t>(i)] / s), -8.f, 7.f) : 0.f;
        codes[static_cast<size_t>(v) * H + i] = static_cast<int8_t>(qv);
        const float e = (w[static_cast<size_t>(i)] - qv * s) / U[static_cast<size_t>(i) * H + i];
        err[i - b0] = e;
        const float *ui = U.data() + static_cast<size_t>(i) * H;
        for (int j = i + 1; j < b1; ++j) w[static_cast<size_t>(j)] -= e * ui[j];
      }
      for (int i = b0; i < b1; ++i) {
        const float e = err[i - b0];
        const float *ui = U.data() + static_cast<size_t>(i) * H;
        for (int j = b1; j < H; ++j) w[static_cast<size_t>(j)] -= e * ui[j];
      }
    }
  });
  std::printf("GPTQ int4: %d rows (%.0f s)\n", V, secs(t1));
  deq();
  report("int4 GPTQ");
  if (wants("gptq")) keep["gptq"] = {codes, scale};
  if (a.num("act", 1)) {  // act order, static weighted-MSE scales of the original rows
    const std::vector<float> Ua = make_U(act);
    t1 = clk::now();
    pfor(V, threads, [&](int v) {
      const float *w0 = W.data() + static_cast<size_t>(v) * H;
      for (int g = 0; g < G; ++g) scale[static_cast<size_t>(g) * V + v] = group_scale4(w0 + g * 128, dg.data() + g * 128, 128, true);
      std::vector<float> w(static_cast<size_t>(H));
      for (int i = 0; i < H; ++i) w[static_cast<size_t>(i)] = w0[act[static_cast<size_t>(i)]];
      for (int i = 0; i < H; ++i) {
        const int col = act[static_cast<size_t>(i)];
        const float s = scale[static_cast<size_t>(col / 128) * V + v];
        const float qv = s != 0.f ? std::clamp(std::nearbyint(w[static_cast<size_t>(i)] / s), -8.f, 7.f) : 0.f;
        codes[static_cast<size_t>(v) * H + col] = static_cast<int8_t>(qv);
        const float e = (w[static_cast<size_t>(i)] - qv * s) / Ua[static_cast<size_t>(i) * H + i];
        const float *ui = Ua.data() + static_cast<size_t>(i) * H;
        for (int j = i + 1; j < H; ++j) w[static_cast<size_t>(j)] -= e * ui[j];
      }
    });
    std::printf("GPTQ int4 act order: %d rows (%.0f s)\n", V, secs(t1));
    deq();
    report("int4 GPTQ act order");
    if (wants("act")) keep["act"] = {codes, scale};
  }
  for (const std::string &write : writes) {
  if (!keep.count(write)) throw std::runtime_error("head4-calib: variant " + write + " was not computed (--act 0?)");
  codes = keep.at(write).first;
  scale = keep.at(write).second;
  const std::string od = writes.size() == 1 ? out : out + "-" + write;

  // ---- write NEW_DIR ----
  if (fs::exists(od) && fs::equivalent(base, od)) throw std::runtime_error("--out must not be --base");
  fs::create_directories(od);
  for (const auto &e : fs::directory_iterator(base)) {
    const std::string name = e.path().filename().string();
    if (!e.is_regular_file() || name == "lm_head-w4.safetensors") continue;
    std::error_code ec;
    fs::remove(od + "/" + name);
    if (name == "model.safetensors.index.json") {  // a copy (edited below), never a link into the served dir
      fs::copy_file(e.path(), od + "/" + name, fs::copy_options::overwrite_existing);
      continue;
    }
    fs::create_hard_link(e.path(), od + "/" + name, ec);
    if (ec) fs::copy_file(e.path(), od + "/" + name, fs::copy_options::overwrite_existing);
  }
  std::vector<uint32_t> qw(static_cast<size_t>(H / 8) * V);
  q::gptq_pack(codes.data(), V, H, qw.data());
  std::vector<uint16_t> sh(scale.size());
  for (size_t i = 0; i < scale.size(); ++i) sh[i] = half_bits(scale[i]);
  std::vector<uint32_t> qz(static_cast<size_t>(G) * (V / 8), 0x77777777u);
  q::write_safetensors(od + "/lm_head-w4.safetensors",
                       {{"lm_head.qweight", "I32", {H / 8, V}, qw.data(), qw.size() * 4},
                        {"lm_head.qzeros", "I32", {G, V / 8}, qz.data(), qz.size() * 4},
                        {"lm_head.scales", "F16", {G, V}, sh.data(), sh.size() * 2}},
                       {{"format", "pt"}, {"xenithium_scheme", "lm_head-int4-g128-sym-" + write}, {"calibration", cap}});
  {  // the index lists the new tensors
    const std::string ip = od + "/model.safetensors.index.json";
    std::ifstream in(ip);
    std::string s((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    const size_t at = s.find("\"weight_map\"");
    const size_t br = at == std::string::npos ? at : s.find('{', at);
    if (br == std::string::npos) throw std::runtime_error("head4-calib: no weight_map in " + ip);
    s.insert(br + 1, "\n    \"lm_head.qweight\": \"lm_head-w4.safetensors\",\n    \"lm_head.qzeros\": \"lm_head-w4.safetensors\",\n    \"lm_head.scales\": \"lm_head-w4.safetensors\",");
    std::ofstream o(ip, std::ios::trunc);
    o << s;
    if (!o) throw std::runtime_error("cannot write " + ip);
  }
  std::printf("lm_head int4 (%s) -> %s/lm_head-w4.safetensors (%.0f s)\n", write.c_str(), od.c_str(), secs(t0));
  }
  return 0;
}

// ============================================================================================================
// xenithium-bake gdn4-calib: qwen38's Gated DeltaNet projections (in_proj_qkv, in_proj_z, out_proj of
// every linear-attention layer) as int4 g128 symmetric (the W4 kernels' format: code q - 8 in [-8, 7] times a signed fp16
// scale per (output row, 128-k group)), GPTQ on the layers' decode-time inputs.
// xenithium-bake gdn4-calib --capture FILE --src DIR --out NEW_DIR [--damp 0.01] [--threads N] [--clip 0|1]
// FILE = an XGC1 capture (`xenithium-probe model --rows 4 --gdn-cap FILE`): per linear layer, rows of its input x [hidden]
// (in_proj_qkv | z read it) and of out_proj's input [value width]. Per layer and input one Hessian H = X^T X (damped by
// --damp x its mean diagonal), the upper Cholesky factor U of H^-1, then per output row the columns in order: a group's
// scale at the group's start on the error-updated weights (-extreme / 8: the RTN convention; --clip 1: head4-calib's
// diag(H)-weighted MSE clip), each column rounded and its error pushed along U (inside the group at once, past it per
// group). Prints per matrix the relative output error ||(W - Wq) X|| / ||W X|| on the layer's rows for RTN and GPTQ.
// NEW_DIR = DIR hard-linked + gdn-w4.safetensors (`<module>.qweight` I32 [k/8][n].scales` F16 [k/128][n].qzeros`
// I32 [k/128][n/8] = 0x77777777: the GPTQ orientation) and an index copy listing them; the BF16 tensors stay (the engine's
// `gdn_w4` switch, server --gdn-w4, picks).
int cmd_gdn4_calib(const Args &a) {
  const std::string cap = a.str("capture"), src = a.str("src", home() + "/models/qwen38-w4a16"), out = a.str("out");
  if (cap.empty() || out.empty()) throw std::runtime_error("--capture and --out are required");
  const int threads = static_cast<int>(a.num("threads", std::thread::hardware_concurrency()));
  const double damp = std::strtod(a.str("damp", "0.01").c_str(), nullptr);
  const bool clip = a.num("clip", 0) != 0;
  const auto t0 = clk::now();
  // ---- the capture: rows per layer ----
  std::ifstream f(cap, std::ios::binary);
  uint32_t magic = 0;
  int32_t hdr[3] = {};
  f.read(reinterpret_cast<char *>(&magic), 4).read(reinterpret_cast<char *>(hdr), 12);
  const int H = hdr[0], VW = hdr[1];
  if (!f || magic != 0x31434758u || H % 128 || VW % 128) throw std::runtime_error("gdn4-calib: not an XGC1 capture: " + cap);
  std::map<int, std::vector<uint16_t>> xs, cs;  // layer -> rows
  for (int32_t rec[2]; f.read(reinterpret_cast<char *>(rec), 8);) {
    auto &x = xs[rec[0]];
    auto &c = cs[rec[0]];
    const size_t o1 = x.size(), o2 = c.size();
    x.resize(o1 + static_cast<size_t>(rec[1]) * H);
    c.resize(o2 + static_cast<size_t>(rec[1]) * VW);
    f.read(reinterpret_cast<char *>(x.data() + o1), static_cast<std::streamsize>(rec[1]) * H * 2);
    f.read(reinterpret_cast<char *>(c.data() + o2), static_cast<std::streamsize>(rec[1]) * VW * 2);
    if (!f) throw std::runtime_error("gdn4-calib: short capture");
  }
  std::printf("capture: %zu layers, %zu rows of layer %d (%.0f s)\n", xs.size(), xs.empty() ? 0 : xs.begin()->second.size() / H, xs.empty() ? -1 : xs.begin()->first,
              secs(t0));
  const auto ck = m::Checkpoint::open(src);
  const std::string P = "model.language_model.layers.";
  // H = X^T X of rows [n][K] (bf16), damped; returns U (upper Cholesky of H^-1, float) and diag H
  auto hessian = [&](const std::vector<uint16_t> &xb, int K, std::vector<float> &U, std::vector<float> &dg) {
    const int64_t n = static_cast<int64_t>(xb.size() / K);
    std::vector<double> Hs(static_cast<size_t>(K) * K, 0.0);
    const int R = 512;
    std::vector<float> XT(static_cast<size_t>(K) * R);
    for (int64_t r0 = 0; r0 < n; r0 += R) {
      const int rn = static_cast<int>(std::min<int64_t>(R, n - r0));
      std::fill(XT.begin(), XT.end(), 0.f);
      pfor(K / 64, threads, [&](int cb) {
        for (int r = 0; r < rn; ++r)
          for (int c = cb * 64; c < cb * 64 + 64; ++c) XT[static_cast<size_t>(c) * R + r] = bf(xb[static_cast<size_t>(r0 + r) * K + c]);
      });
      pfor(K, threads, [&](int i) {
        const float *xi = XT.data() + static_cast<size_t>(i) * R;
        double *hi = Hs.data() + static_cast<size_t>(i) * K;
        for (int j = i; j < K; ++j) hi[j] += dot16(xi, XT.data() + static_cast<size_t>(j) * R, R);
      });
    }
    double dmean = 0;
    for (int i = 0; i < K; ++i) {
      for (int j = 0; j < i; ++j) Hs[static_cast<size_t>(i) * K + j] = Hs[static_cast<size_t>(j) * K + i];
      dmean += Hs[static_cast<size_t>(i) * K + i];
    }
    dmean /= K;
    for (int i = 0; i < K; ++i) Hs[static_cast<size_t>(i) * K + i] += damp * dmean + 1e-12;
    dg.resize(static_cast<size_t>(K));
    for (int i = 0; i < K; ++i) dg[static_cast<size_t>(i)] = static_cast<float>(Hs[static_cast<size_t>(i) * K + i]);
    cholesky(Hs, K, threads);
    std::vector<double> Hi = chol_inverse(Hs, K, threads);
    cholesky(Hi, K, threads);
    U.assign(static_cast<size_t>(K) * K, 0.f);
    for (int i = 0; i < K; ++i)
      for (int j = i; j < K; ++j) U[static_cast<size_t>(i) * K + j] = static_cast<float>(Hi[static_cast<size_t>(j) * K + i]);
  };
  // relative output error of codes x scale vs W on rows X (a subsample of 256 rows)
  auto out_err = [&](const std::vector<float> &W, const std::vector<int8_t> &codes, const std::vector<float> &scale, int N, int K, const std::vector<uint16_t> &xb) {
    const int64_t n = std::min<int64_t>(256, static_cast<int64_t>(xb.size() / K));
    std::vector<float> X(static_cast<size_t>(n) * K);
    for (int64_t r = 0; r < n; ++r)
      for (int c = 0; c < K; ++c) X[static_cast<size_t>(r) * K + c] = bf(xb[static_cast<size_t>(r * (static_cast<int64_t>(xb.size() / K) / n)) * K + c]);
    std::vector<double> num(static_cast<size_t>(N)), den(static_cast<size_t>(N));
    pfor(N, threads, [&](int v) {
      std::vector<float> wq(static_cast<size_t>(K));
      for (int c = 0; c < K; ++c) wq[static_cast<size_t>(c)] = codes[static_cast<size_t>(v) * K + c] * scale[static_cast<size_t>(c / 128) * N + v];
      double a2 = 0, b2 = 0;
      for (int64_t r = 0; r < n; ++r) {
        const double y = dot16(W.data() + static_cast<size_t>(v) * K, X.data() + static_cast<size_t>(r) * K, K), yq = dot16(wq.data(), X.data() + static_cast<size_t>(r) * K, K);
        a2 += (yq - y) * (yq - y), b2 += y * y;
      }
      num[static_cast<size_t>(v)] = a2, den[static_cast<size_t>(v)] = b2;
    });
    double a2 = 0, b2 = 0;
    for (int v = 0; v < N; ++v) a2 += num[static_cast<size_t>(v)], b2 += den[static_cast<size_t>(v)];
    return std::sqrt(a2 / std::max(b2, 1e-30));
  };
  struct OutT {
    std::string name;
    std::vector<uint32_t> qw, qz;
    std::vector<uint16_t> sh;
    int n, k;
  };
  std::vector<OutT> outs;
  double rtn_sum = 0, gptq_sum = 0;
  int nm = 0;
  for (const auto &[l, xb] : xs) {
    const auto tl = clk::now();
    std::vector<float> Ui, di, Uo, dd;
    hessian(xb, H, Ui, di);
    hessian(cs.at(l), VW, Uo, dd);
    for (const char *mod : {"in_proj_qkv", "in_proj_z", "out_proj"}) {
      const std::string name = P + std::to_string(l) + ".linear_attn." + mod;
      const auto &t = ck.at(name + ".weight");
      if (t.dtype != m::DType::bf16 || t.shape.size() != 2) throw std::runtime_error("gdn4-calib: " + name + ".weight is not BF16 [n, k]");
      const int N = static_cast<int>(t.dim(0)), K = static_cast<int>(t.dim(1)), G = K / 128;
      const bool o = std::string(mod) == "out_proj";
      if (K != (o ? VW : H) || N % 8) throw std::runtime_error("gdn4-calib: " + name + " shape does not match the capture");
      const std::vector<float> &U = o ? Uo : Ui, &dg = o ? dd : di;
      std::vector<uint16_t> wb(static_cast<size_t>(N) * K);
      read_tensor_range(ck, t, 0, t.bytes, wb.data());
      std::vector<float> W(wb.size());
      for (size_t i = 0; i < W.size(); ++i) W[i] = bf(wb[i]);
      std::vector<int8_t> codes(static_cast<size_t>(N) * K);
      std::vector<float> scale(static_cast<size_t>(G) * N);
      pfor(N, threads, [&](int v) {  // RTN for the report
        const float *w = W.data() + static_cast<size_t>(v) * K;
        for (int g = 0; g < G; ++g) {
          const float s = group_scale4(w + g * 128, nullptr, 128, false);
          scale[static_cast<size_t>(g) * N + v] = s;
          for (int i = g * 128; i < g * 128 + 128; ++i)
            codes[static_cast<size_t>(v) * K + i] = static_cast<int8_t>(s != 0.f ? std::clamp(std::nearbyint(w[i] / s), -8.f, 7.f) : 0.f);
        }
      });
      const double e_rtn = out_err(W, codes, scale, N, K, o ? cs.at(l) : xb);
      pfor(N, threads, [&](int v) {  // GPTQ
        std::vector<float> w(W.begin() + static_cast<std::ptrdiff_t>(v) * K, W.begin() + static_cast<std::ptrdiff_t>(v + 1) * K);
        float err[128];
        for (int g = 0; g < G; ++g) {
          const int b0 = g * 128, b1 = b0 + 128;
          const float s = group_scale4(w.data() + b0, clip ? dg.data() + b0 : nullptr, 128, clip);
          scale[static_cast<size_t>(g) * N + v] = s;
          for (int i = b0; i < b1; ++i) {
            const float qv = s != 0.f ? std::clamp(std::nearbyint(w[static_cast<size_t>(i)] / s), -8.f, 7.f) : 0.f;
            codes[static_cast<size_t>(v) * K + i] = static_cast<int8_t>(qv);
            const float e = (w[static_cast<size_t>(i)] - qv * s) / U[static_cast<size_t>(i) * K + i];
            err[i - b0] = e;
            const float *ui = U.data() + static_cast<size_t>(i) * K;
            for (int j = i + 1; j < b1; ++j) w[static_cast<size_t>(j)] -= e * ui[j];
          }
          for (int i = b0; i < b1; ++i) {
            const float e = err[i - b0];
            const float *ui = U.data() + static_cast<size_t>(i) * K;
            for (int j = b1; j < K; ++j) w[static_cast<size_t>(j)] -= e * ui[j];
          }
        }
      });
      const double e_gptq = out_err(W, codes, scale, N, K, o ? cs.at(l) : xb);
      rtn_sum += e_rtn, gptq_sum += e_gptq, ++nm;
      std::printf("  layer %2d %-12s %5d x %4d: output rel. error RTN %.4f -> GPTQ %.4f\n", l, mod, N, K, e_rtn, e_gptq);
      std::fflush(stdout);
      OutT ot{name, std::vector<uint32_t>(static_cast<size_t>(K / 8) * N), std::vector<uint32_t>(static_cast<size_t>(G) * (N / 8), 0x77777777u),
              std::vector<uint16_t>(scale.size()), N, K};
      q::gptq_pack(codes.data(), N, K, ot.qw.data());
      for (size_t i = 0; i < scale.size(); ++i) ot.sh[i] = half_bits(scale[i]);
      outs.push_back(std::move(ot));
    }
    std::printf("layer %d done (%.0f s)\n", l, secs(tl));
  }
  std::printf("mean output rel. error over %d matrices: RTN %.4f, GPTQ %.4f\n", nm, rtn_sum / std::max(1, nm), gptq_sum / std::max(1, nm));
  // ---- NEW_DIR ----
  if (fs::exists(out) && fs::equivalent(src, out)) throw std::runtime_error("--out must not be --src");
  fs::create_directories(out);
  for (const auto &e : fs::directory_iterator(src)) {
    const std::string name = e.path().filename().string();
    if (!e.is_regular_file() || name == "gdn-w4.safetensors") continue;
    std::error_code ec;
    fs::remove(out + "/" + name);
    if (name == "model.safetensors.index.json") {
      fs::copy_file(e.path(), out + "/" + name, fs::copy_options::overwrite_existing);
      continue;
    }
    fs::create_hard_link(e.path(), out + "/" + name, ec);
    if (ec) fs::copy_file(e.path(), out + "/" + name, fs::copy_options::overwrite_existing);
  }
  std::vector<q::StTensor> ts;
  std::string idx;
  for (const auto &o : outs) {
    ts.push_back({o.name + ".qweight", "I32", {o.k / 8, o.n}, o.qw.data(), o.qw.size() * 4});
    ts.push_back({o.name + ".qzeros", "I32", {o.k / 128, o.n / 8}, o.qz.data(), o.qz.size() * 4});
    ts.push_back({o.name + ".scales", "F16", {o.k / 128, o.n}, o.sh.data(), o.sh.size() * 2});
    for (const char *sfx : {".qweight", ".qzeros", ".scales"}) idx += "\n    \"" + o.name + sfx + "\": \"gdn-w4.safetensors\",";
  }
  q::write_safetensors(out + "/gdn-w4.safetensors", ts,
                       {{"format", "pt"}, {"xenithium_scheme", std::string("gdn-int4-g128-sym-gptq") + (clip ? "-clip" : "")}, {"calibration", cap}});
  {
    const std::string ip = out + "/model.safetensors.index.json";
    std::ifstream in(ip);
    std::string s((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    const size_t at = s.find("\"weight_map\"");
    const size_t br = at == std::string::npos ? at : s.find('{', at);
    if (br == std::string::npos) throw std::runtime_error("gdn4-calib: no weight_map in " + ip);
    s.insert(br + 1, idx);
    std::ofstream o(ip, std::ios::trunc);
    o << s;
    if (!o) throw std::runtime_error("cannot write " + ip);
  }
  std::printf("GDN int4 GPTQ (%zu matrices) -> %s/gdn-w4.safetensors (%.0f s)\n", outs.size(), out.c_str(), secs(t0));
  return 0;
}

}  // namespace xenithium:bake
