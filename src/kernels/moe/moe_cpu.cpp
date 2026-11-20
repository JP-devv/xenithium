// Xenithium · the decode MoE's optional CPU path for cold experts (moe_cpu.hpp).
#include "xenithium/kernels/moe/moe_cpu.hpp"
#include "xenithium/kernels/rows.hpp"  // kStepRows32: the rows a request can carry

#include <immintrin.h>
#include <pthread.h>
#include <sched.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace xenithium::kernels {
namespace {

constexpr int kW = 16, kGroup = 128, kHalf = 64, kBlock = 1024, kHalfBlock = 512;

inline float bf2f(uint16_t b) {
  uint32_t u = static_cast<uint32_t>(b) << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}
inline uint16_t f2bf(float f) {  // round to nearest even (NaN stays NaN)
  uint32_t u;
  std::memcpy(&u, &f, 4);
  if ((u & 0x7fffffffu) > 0x7f800000u) return static_cast<uint16_t>((u >> 16) | 0x40u);
  u += 0x7fffu + ((u >> 16) & 1u);
  return static_cast<uint16_t>(u >> 16);
}
inline float half2f(uint16_t h) {  // IEEE fp16 -> fp32 (exact)
  const uint32_t s = (h & 0x8000u) << 16, e = (h >> 10) & 0x1fu, m = h & 0x3ffu;
  uint32_t u;
  if (e == 0) {
    if (m == 0) u = s;
    else {  // subnormal: normalise
      int ee = -1;
      uint32_t mm = m;
      do { ++ee; mm <<= 1; } while (!(mm & 0x400u));
      u = s | static_cast<uint32_t>(127 - 15 - ee) << 23 | (mm & 0x3ffu) << 13;
    }
  } else if (e == 31) {
    u = s | 0x7f800000u | m << 13;
  } else {
    u = s | (e + 112u) << 23 | m << 13;
  }
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}
// h = bf16(silu(bf16 g) bf16 u), the GPU path's form gb * sigmoid(gb) * ub with sigmoid = 1 / (1 + exp(-x))
inline uint16_t h_of(float g, float u) {
  const float gb = bf2f(f2bf(g)), ub = bf2f(f2bf(u));
  const float sg = 1.f / (1.f + std::exp(-gb));
  return f2bf(gb * sg * ub);
}

// Rows of a mask, ascending.
int rows_of(uint32_t mask, int *r) {
  int n = 0;
  for (int i = 0; i < kStepRows32; ++i)
    if ((mask >> i) & 1u) r[n++] = i;
  return n;
}

// ---- scalar (the canonical order: per (k, k+1) pair, acc = fma(q_{k+1} - 8, x_{k+1}, acc) then the even k, which is
// vdpbf16ps's order; AVX2 follows it, so all three agree bit for bit on normal inputs) -------------------------------
template <int R>
void gu_tile_scalar(const MoeCpuShape &s, const uint8_t *blob, const uint16_t *const *xr, int j, uint16_t *const *hr) {
  const int Ic = s.inter_c, tic = Ic / kW, G = s.hidden / kGroup, KS = s.gu_ks;
  const uint16_t *scl = reinterpret_cast<const uint16_t *>(blob + s.gu_scales);
  float tot[2][R][kW] = {};
  for (int p = 0; p < KS; ++p) {
    float acc[2][R][kW] = {};
    for (int g = p * G / KS; g < (p + 1) * G / KS; ++g)
      for (int proj = 0; proj < 2; ++proj) {
        const uint8_t *c = blob + s.gu_codes + (static_cast<size_t>(g) * 2 * tic + proj * tic + j) * kBlock;
        float gs[R][kW] = {};
        for (int kp = 0; kp < kGroup / 2; ++kp)
          for (int n = 0; n < kW; ++n) {
            const uint8_t b = c[kp * kW + n];
            const float lo = static_cast<float>(static_cast<int>(b & 15) - 8), hi = static_cast<float>(static_cast<int>(b >> 4) - 8);
            for (int r = 0; r < R; ++r) {
              const uint16_t *x = xr[r] + g * kGroup + 2 * kp;
              gs[r][n] = std::fma(hi, bf2f(x[1]), gs[r][n]);
              gs[r][n] = std::fma(lo, bf2f(x[0]), gs[r][n]);
            }
          }
        for (int n = 0; n < kW; ++n) {
          const float sc = half2f(scl[static_cast<size_t>(g) * 2 * Ic + proj * Ic + j * kW + n]);
          for (int r = 0; r < R; ++r) acc[proj][r][n] = std::fma(gs[r][n], sc, acc[proj][r][n]);
        }
      }
    for (int proj = 0; proj < 2; ++proj)
      for (int r = 0; r < R; ++r)
        for (int n = 0; n < kW; ++n) tot[proj][r][n] += acc[proj][r][n];
  }
  for (int r = 0; r < R; ++r)
    for (int n = 0; n < kW; ++n) hr[r][j * kW + n] = h_of(tot[0][r][n], tot[1][r][n]);
}

template <int R>
void dn_tile_scalar(const MoeCpuShape &s, const uint8_t *blob, const uint16_t *const *hr, int t, float *const *yr) {
  const int H = s.hidden, tiles = H / kW, NH = s.inter_c / kHalf, KS = s.dn_ks, hg0 = s.card * NH;
  const uint16_t *scl = reinterpret_cast<const uint16_t *>(blob + s.dn_scales);
  float tot[R][kW] = {};
  for (int p = 0; p < KS; ++p) {
    float acc[R][kW] = {};
    for (int g = p * NH / KS; g < (p + 1) * NH / KS; ++g) {
      const uint8_t *c = blob + s.dn_codes + (static_cast<size_t>(g) * tiles + t) * kHalfBlock;
      const int row = (hg0 + g) / 2 - hg0 / 2;
      float gs[R][kW] = {};
      for (int kp = 0; kp < kHalf / 2; ++kp)
        for (int n = 0; n < kW; ++n) {
          const uint8_t b = c[kp * kW + n];
          const float lo = static_cast<float>(static_cast<int>(b & 15) - 8), hi = static_cast<float>(static_cast<int>(b >> 4) - 8);
          for (int r = 0; r < R; ++r) {
            const uint16_t *h = hr[r] + g * kHalf + 2 * kp;
            gs[r][n] = std::fma(hi, bf2f(h[1]), gs[r][n]);
            gs[r][n] = std::fma(lo, bf2f(h[0]), gs[r][n]);
          }
        }
      for (int n = 0; n < kW; ++n) {
        const float sc = half2f(scl[static_cast<size_t>(row) * H + t * kW + n]);
        for (int r = 0; r < R; ++r) acc[r][n] = std::fma(gs[r][n], sc, acc[r][n]);
      }
    }
    for (int r = 0; r < R; ++r)
      for (int n = 0; n < kW; ++n) tot[r][n] += acc[r][n];
  }
  for (int r = 0; r < R; ++r) std::memcpy(yr[r] + t * kW, tot[r], sizeof tot[r]);
}

// ---- AVX-512 BF16 ----------------------------------------------------------------------------------------------
#define XEN_AVX512 __attribute__((target("avx512f,avx512bw,avx512vl,avx512bf16,f16c,fma")))
#define XEN_AVX2 __attribute__((target("avx2,fma,f16c")))

XEN_AVX512 inline __m512i lut512() {  // words 0..15: bf16(q - 8) (exact); the rest unused
  alignas(64) uint16_t w[32] = {};
  for (int q = 0; q < 16; ++q) w[q] = f2bf(static_cast<float>(q - 8));
  return _mm512_load_si512(w);
}
// 16 code bytes (one per column, low nibble k even, high nibble k odd) -> 16 (bf16 even, bf16 odd) pairs
XEN_AVX512 inline __m512bh pairs512(const uint8_t *p, __m512i lut) {
  const __m512i d = _mm512_cvtepu8_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i *>(p)));
  const __m512i idx = _mm512_or_si512(_mm512_and_si512(d, _mm512_set1_epi32(15)), _mm512_slli_epi32(_mm512_and_si512(d, _mm512_set1_epi32(0xf0)), 12));
  return reinterpret_cast<__m512bh>(_mm512_permutexvar_epi16(idx, lut));
}
XEN_AVX512 inline __m512bh xpair512(const uint16_t *x) {
  int32_t v;
  std::memcpy(&v, x, 4);
  return reinterpret_cast<__m512bh>(_mm512_set1_epi32(v));
}
XEN_AVX512 inline __m512 scale512(const uint16_t *p) { return _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(p))); }

template <int R>
XEN_AVX512 void gu_tile_512(const MoeCpuShape &s, const uint8_t *blob, const uint16_t *const *xr, int j, uint16_t *const *hr) {
  const int Ic = s.inter_c, tic = Ic / kW, G = s.hidden / kGroup, KS = s.gu_ks;
  const uint16_t *scl = reinterpret_cast<const uint16_t *>(blob + s.gu_scales);
  const __m512i lut = lut512();
  __m512 tg[R], tu[R];
  for (int r = 0; r < R; ++r) tg[r] = tu[r] = _mm512_setzero_ps();
  for (int p = 0; p < KS; ++p) {
    __m512 ag[R], au[R];
    for (int r = 0; r < R; ++r) ag[r] = au[r] = _mm512_setzero_ps();
    for (int g = p * G / KS; g < (p + 1) * G / KS; ++g) {
      const uint8_t *cg = blob + s.gu_codes + (static_cast<size_t>(g) * 2 * tic + j) * kBlock;
      const uint8_t *cu = cg + static_cast<size_t>(tic) * kBlock;
      if (g + 1 < G) {  // the next group's two blocks (40 KB away: not a stream the prefetcher follows)
        for (int o = 0; o < kBlock; o += 64) {
          _mm_prefetch(reinterpret_cast<const char *>(cg + 2 * static_cast<size_t>(tic) * kBlock + o), _MM_HINT_T0);
          _mm_prefetch(reinterpret_cast<const char *>(cu + 2 * static_cast<size_t>(tic) * kBlock + o), _MM_HINT_T0);
        }
      }
      __m512 gg[R], gu[R];
      for (int r = 0; r < R; ++r) gg[r] = gu[r] = _mm512_setzero_ps();
      for (int kp = 0; kp < kGroup / 2; ++kp) {
        const __m512bh bg = pairs512(cg + kp * kW, lut), bu = pairs512(cu + kp * kW, lut);
        for (int r = 0; r < R; ++r) {
          const __m512bh xp = xpair512(xr[r] + g * kGroup + 2 * kp);
          gg[r] = _mm512_dpbf16_ps(gg[r], bg, xp);
          gu[r] = _mm512_dpbf16_ps(gu[r], bu, xp);
        }
      }
      const __m512 sg = scale512(scl + static_cast<size_t>(g) * 2 * Ic + j * kW), su = scale512(scl + static_cast<size_t>(g) * 2 * Ic + Ic + j * kW);
      for (int r = 0; r < R; ++r) {
        ag[r] = _mm512_fmadd_ps(gg[r], sg, ag[r]);
        au[r] = _mm512_fmadd_ps(gu[r], su, au[r]);
      }
    }
    for (int r = 0; r < R; ++r) {
      tg[r] = _mm512_add_ps(tg[r], ag[r]);
      tu[r] = _mm512_add_ps(tu[r], au[r]);
    }
  }
  for (int r = 0; r < R; ++r) {
    alignas(64) float g16[kW], u16[kW];
    _mm512_store_ps(g16, tg[r]);
    _mm512_store_ps(u16, tu[r]);
    for (int n = 0; n < kW; ++n) hr[r][j * kW + n] = h_of(g16[n], u16[n]);
  }
}

template <int R>
XEN_AVX512 void dn_tile_512(const MoeCpuShape &s, const uint8_t *blob, const uint16_t *const *hr, int t, float *const *yr) {
  const int H = s.hidden, tiles = H / kW, NH = s.inter_c / kHalf, KS = s.dn_ks, hg0 = s.card * NH;
  const uint16_t *scl = reinterpret_cast<const uint16_t *>(blob + s.dn_scales);
  const __m512i lut = lut512();
  __m512 tot[R];
  for (int r = 0; r < R; ++r) tot[r] = _mm512_setzero_ps();
  for (int p = 0; p < KS; ++p) {
    __m512 acc[R];
    for (int r = 0; r < R; ++r) acc[r] = _mm512_setzero_ps();
    for (int g = p * NH / KS; g < (p + 1) * NH / KS; ++g) {
      const uint8_t *c = blob + s.dn_codes + (static_cast<size_t>(g) * tiles + t) * kHalfBlock;
      __m512 gs[R];
      for (int r = 0; r < R; ++r) gs[r] = _mm512_setzero_ps();
      for (int kp = 0; kp < kHalf / 2; ++kp) {
        const __m512bh b = pairs512(c + kp * kW, lut);
        for (int r = 0; r < R; ++r) gs[r] = _mm512_dpbf16_ps(gs[r], b, xpair512(hr[r] + g * kHalf + 2 * kp));
      }
      const int row = (hg0 + g) / 2 - hg0 / 2;
      const __m512 sc = scale512(scl + static_cast<size_t>(row) * H + t * kW);
      for (int r = 0; r < R; ++r) acc[r] = _mm512_fmadd_ps(gs[r], sc, acc[r]);
    }
    for (int r = 0; r < R; ++r) tot[r] = _mm512_add_ps(tot[r], acc[r]);
  }
  for (int r = 0; r < R; ++r) _mm512_storeu_ps(yr[r] + t * kW, tot[r]);
}

// ---- AVX2 + FMA (16 columns = two 8-lane halves; gate and up in two passes to keep the accumulators in registers) --
XEN_AVX2 inline void codes256(const uint8_t *p, __m256 *lo, __m256 *hi) {  // columns 0-7 -> [0], 8-15 -> [1]
  const __m128i b = _mm_loadu_si128(reinterpret_cast<const __m128i *>(p));
  const __m256 eight = _mm256_set1_ps(8.f);
  const __m256i d0 = _mm256_cvtepu8_epi32(b), d1 = _mm256_cvtepu8_epi32(_mm_srli_si128(b, 8));
  const __m256i m = _mm256_set1_epi32(15);
  lo[0] = _mm256_sub_ps(_mm256_cvtepi32_ps(_mm256_and_si256(d0, m)), eight);
  lo[1] = _mm256_sub_ps(_mm256_cvtepi32_ps(_mm256_and_si256(d1, m)), eight);
  hi[0] = _mm256_sub_ps(_mm256_cvtepi32_ps(_mm256_srli_epi32(d0, 4)), eight);
  hi[1] = _mm256_sub_ps(_mm256_cvtepi32_ps(_mm256_srli_epi32(d1, 4)), eight);
}
XEN_AVX2 inline __m256 bcast_bf(uint16_t v) { return _mm256_castsi256_ps(_mm256_set1_epi32(static_cast<int32_t>(static_cast<uint32_t>(v) << 16))); }
XEN_AVX2 inline void scale256(const uint16_t *p, __m256 *s) {
  s[0] = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(p)));
  s[1] = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(p + 8)));
}

// One projection's 16 columns of tile nt over groups [g0, g1) for R rows: acc += per group (sum) x scale
template <int R>
XEN_AVX2 void gu_part_256(const MoeCpuShape &s, const uint8_t *blob, const uint16_t *const *xr, int proj, int j, int g0, int g1, __m256 (*acc)[2]) {
  const int Ic = s.inter_c, tic = Ic / kW;
  const uint16_t *scl = reinterpret_cast<const uint16_t *>(blob + s.gu_scales);
  for (int g = g0; g < g1; ++g) {
    const uint8_t *c = blob + s.gu_codes + (static_cast<size_t>(g) * 2 * tic + proj * tic + j) * kBlock;
    __m256 gs[R][2];
    for (int r = 0; r < R; ++r) gs[r][0] = gs[r][1] = _mm256_setzero_ps();
    for (int kp = 0; kp < kGroup / 2; ++kp) {
      __m256 lo[2], hi[2];
      codes256(c + kp * kW, lo, hi);
      for (int r = 0; r < R; ++r) {
        const uint16_t *x = xr[r] + g * kGroup + 2 * kp;
        const __m256 xa = bcast_bf(x[0]), xb = bcast_bf(x[1]);
        for (int h = 0; h < 2; ++h) {
          gs[r][h] = _mm256_fmadd_ps(hi[h], xb, gs[r][h]);
          gs[r][h] = _mm256_fmadd_ps(lo[h], xa, gs[r][h]);
        }
      }
    }
    __m256 sc[2];
    scale256(scl + static_cast<size_t>(g) * 2 * Ic + proj * Ic + j * kW, sc);
    for (int r = 0; r < R; ++r)
      for (int h = 0; h < 2; ++h) acc[r][h] = _mm256_fmadd_ps(gs[r][h], sc[h], acc[r][h]);
  }
}

template <int R>
XEN_AVX2 void gu_tile_256(const MoeCpuShape &s, const uint8_t *blob, const uint16_t *const *xr, int j, uint16_t *const *hr) {
  const int G = s.hidden / kGroup, KS = s.gu_ks;
  alignas(32) float tot[2][R][kW];
  for (int proj = 0; proj < 2; ++proj) {
    __m256 t[R][2];
    for (int r = 0; r < R; ++r) t[r][0] = t[r][1] = _mm256_setzero_ps();
    for (int p = 0; p < KS; ++p) {
      __m256 acc[R][2];
      for (int r = 0; r < R; ++r) acc[r][0] = acc[r][1] = _mm256_setzero_ps();
      gu_part_256<R>(s, blob, xr, proj, j, p * G / KS, (p + 1) * G / KS, acc);
      for (int r = 0; r < R; ++r)
        for (int h = 0; h < 2; ++h) t[r][h] = _mm256_add_ps(t[r][h], acc[r][h]);
    }
    for (int r = 0; r < R; ++r) {
      _mm256_store_ps(tot[proj][r], t[r][0]);
      _mm256_store_ps(tot[proj][r] + 8, t[r][1]);
    }
  }
  for (int r = 0; r < R; ++r)
    for (int n = 0; n < kW; ++n) hr[r][j * kW + n] = h_of(tot[0][r][n], tot[1][r][n]);
}

template <int R>
XEN_AVX2 void dn_tile_256(const MoeCpuShape &s, const uint8_t *blob, const uint16_t *const *hr, int t, float *const *yr) {
  const int H = s.hidden, tiles = H / kW, NH = s.inter_c / kHalf, KS = s.dn_ks, hg0 = s.card * NH;
  const uint16_t *scl = reinterpret_cast<const uint16_t *>(blob + s.dn_scales);
  __m256 tot[R][2];
  for (int r = 0; r < R; ++r) tot[r][0] = tot[r][1] = _mm256_setzero_ps();
  for (int p = 0; p < KS; ++p) {
    __m256 acc[R][2];
    for (int r = 0; r < R; ++r) acc[r][0] = acc[r][1] = _mm256_setzero_ps();
    for (int g = p * NH / KS; g < (p + 1) * NH / KS; ++g) {
      const uint8_t *c = blob + s.dn_codes + (static_cast<size_t>(g) * tiles + t) * kHalfBlock;
      __m256 gs[R][2];
      for (int r = 0; r < R; ++r) gs[r][0] = gs[r][1] = _mm256_setzero_ps();
      for (int kp = 0; kp < kHalf / 2; ++kp) {
        __m256 lo[2], hi[2];
        codes256(c + kp * kW, lo, hi);
        for (int r = 0; r < R; ++r) {
          const uint16_t *h = hr[r] + g * kHalf + 2 * kp;
          const __m256 ha = bcast_bf(h[0]), hb = bcast_bf(h[1]);
          for (int q = 0; q < 2; ++q) {
            gs[r][q] = _mm256_fmadd_ps(hi[q], hb, gs[r][q]);
            gs[r][q] = _mm256_fmadd_ps(lo[q], ha, gs[r][q]);
          }
        }
      }
      const int row = (hg0 + g) / 2 - hg0 / 2;
      __m256 sc[2];
      scale256(scl + static_cast<size_t>(row) * H + t * kW, sc);
      for (int r = 0; r < R; ++r)
        for (int q = 0; q < 2; ++q) acc[r][q] = _mm256_fmadd_ps(gs[r][q], sc[q], acc[r][q]);
    }
    for (int r = 0; r < R; ++r)
      for (int q = 0; q < 2; ++q) tot[r][q] = _mm256_add_ps(tot[r][q], acc[r][q]);
  }
  for (int r = 0; r < R; ++r) {
    _mm256_storeu_ps(yr[r] + t * kW, tot[r][0]);
    _mm256_storeu_ps(yr[r] + t * kW + 8, tot[r][1]);
  }
}

// Rows in chunks of up to 4 (the register budget), per ISA.
template <typename F1, typename F2, typename F3, typename F4>
void by_rows(int nr, F1 f1, F2 f2, F3 f3, F4 f4) {
  for (int c = 0; c < nr; c += 4) {
    switch (std::min(4, nr - c)) {
      case 1: f1(c); break;
      case 2: f2(c); break;
      case 3: f3(c); break;
      default: f4(c); break;
    }
  }
}

}  // namespace

MoeCpuIsa moe_cpu_detect(const std::string &want) {
  __builtin_cpu_init();
  const bool a512 = __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") && __builtin_cpu_supports("avx512vl") &&
                    __builtin_cpu_supports("avx512bf16");
  const bool a2 = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") && __builtin_cpu_supports("f16c");
  if (want == "avx512") return a512 ? MoeCpuIsa::avx512bf16 : MoeCpuIsa::none;
  if (want == "avx2") return a2 ? MoeCpuIsa::avx2 : MoeCpuIsa::none;
  return a512 ? MoeCpuIsa::avx512bf16 : a2 ? MoeCpuIsa::avx2 : MoeCpuIsa::none;
}

const char *moe_cpu_isa_name(MoeCpuIsa isa) {
  switch (isa) {
    case MoeCpuIsa::avx512bf16: return "avx512-bf16";
    case MoeCpuIsa::avx2: return "avx2";
    default: return "scalar";
  }
}

void moe_cpu_gate_up(MoeCpuIsa isa, const MoeCpuShape &s, const uint8_t *blob, const uint16_t *x, int ldx, uint32_t mask, int t0, int t1, uint16_t *h) {
  int rr[kStepRows32];
  const int nr = rows_of(mask, rr);
  const uint16_t *xr[kStepRows32];
  uint16_t *hr[kStepRows32];
  for (int i = 0; i < nr; ++i) xr[i] = x + static_cast<size_t>(rr[i]) * ldx, hr[i] = h + static_cast<size_t>(rr[i]) * s.inter_c;
  for (int j = t0; j < t1; ++j) {
    if (isa == MoeCpuIsa::avx512bf16)
      by_rows(nr, [&](int c) { gu_tile_512<1>(s, blob, xr + c, j, hr + c); }, [&](int c) { gu_tile_512<2>(s, blob, xr + c, j, hr + c); },
              [&](int c) { gu_tile_512<3>(s, blob, xr + c, j, hr + c); }, [&](int c) { gu_tile_512<4>(s, blob, xr + c, j, hr + c); });
    else if (isa == MoeCpuIsa::avx2)
      by_rows(nr, [&](int c) { gu_tile_256<1>(s, blob, xr + c, j, hr + c); }, [&](int c) { gu_tile_256<2>(s, blob, xr + c, j, hr + c); },
              [&](int c) { gu_tile_256<3>(s, blob, xr + c, j, hr + c); }, [&](int c) { gu_tile_256<4>(s, blob, xr + c, j, hr + c); });
    else
      by_rows(nr, [&](int c) { gu_tile_scalar<1>(s, blob, xr + c, j, hr + c); }, [&](int c) { gu_tile_scalar<2>(s, blob, xr + c, j, hr + c); },
              [&](int c) { gu_tile_scalar<3>(s, blob, xr + c, j, hr + c); }, [&](int c) { gu_tile_scalar<4>(s, blob, xr + c, j, hr + c); });
  }
}

void moe_cpu_down(MoeCpuIsa isa, const MoeCpuShape &s, const uint8_t *blob, const uint16_t *h, uint32_t mask, int t0, int t1, float *y, int ldy) {
  int rr[kStepRows32];
  const int nr = rows_of(mask, rr);
  const uint16_t *hr[kStepRows32];
  float *yr[kStepRows32];
  for (int i = 0; i < nr; ++i) hr[i] = h + static_cast<size_t>(rr[i]) * s.inter_c, yr[i] = y + static_cast<size_t>(rr[i]) * ldy;
  for (int t = t0; t < t1; ++t) {
    if (isa == MoeCpuIsa::avx512bf16)
      by_rows(nr, [&](int c) { dn_tile_512<1>(s, blob, hr + c, t, yr + c); }, [&](int c) { dn_tile_512<2>(s, blob, hr + c, t, yr + c); },
              [&](int c) { dn_tile_512<3>(s, blob, hr + c, t, yr + c); }, [&](int c) { dn_tile_512<4>(s, blob, hr + c, t, yr + c); });
    else if (isa == MoeCpuIsa::avx2)
      by_rows(nr, [&](int c) { dn_tile_256<1>(s, blob, hr + c, t, yr + c); }, [&](int c) { dn_tile_256<2>(s, blob, hr + c, t, yr + c); },
              [&](int c) { dn_tile_256<3>(s, blob, hr + c, t, yr + c); }, [&](int c) { dn_tile_256<4>(s, blob, hr + c, t, yr + c); });
    else
      by_rows(nr, [&](int c) { dn_tile_scalar<1>(s, blob, hr + c, t, yr + c); }, [&](int c) { dn_tile_scalar<2>(s, blob, hr + c, t, yr + c); },
              [&](int c) { dn_tile_scalar<3>(s, blob, hr + c, t, yr + c); }, [&](int c) { dn_tile_scalar<4>(s, blob, hr + c, t, yr + c); });
  }
}

// ---- the pool ------------------------------------------------------------------------------------------------------
namespace {
constexpr int kDnChunk = 8;  // down tiles per job (2.5 KB of codes each; a gate_up job reads 40 KB)
inline void cpu_relax() { _mm_pause(); }
using clk = std::chrono::steady_clock;
inline uint32_t vload(const uint32_t *p) { return __atomic_load_n(p, __ATOMIC_ACQUIRE); }
inline void vstore(uint32_t *p, uint32_t v) { __atomic_store_n(p, v, __ATOMIC_RELEASE); }
void nap() {
  const timespec ts{0, 50000};
  nanosleep(&ts, nullptr);
}
}  // namespace

MoeCpuPool::MoeCpuPool(const MoeCpuShape &s, const MoeCpuOptions &o, MoeCpuMail *mail, const uint16_t *x, float *y)
    : s_(s), o_(o), mail_(mail), x_(x), y_(y) {
  if (s.hidden % kGroup || s.inter_c % kHalf || s.gu_ks < 1 || s.dn_ks < 1) throw std::invalid_argument("MoeCpuPool: shape");
  if (o.threads < 1 || o.threads > 64) throw std::invalid_argument("MoeCpuPool: threads");
  h_.assign(static_cast<size_t>(kMoeCpuMax) * kStepRows32 * s.inter_c, 0);
  seen0_ = vload(&mail_->req);  // before the threads start: a request may arrive before the poller first runs
  vstore(&mail_->done, seen0_);
  for (int t = 0; t < o.threads; ++t) th_.emplace_back([this, t] { worker(t); });
}

MoeCpuPool::~MoeCpuPool() {
  stop_.store(true);
  for (auto &t : th_) t.join();
}

void MoeCpuPool::compute(Req &q) {
  const int tic = s_.inter_c / kW, dn_jobs = s_.hidden / kW / kDnChunk;
  const int n = q.n, gu_total = n * tic, dn_total = n * dn_jobs;
  for (int j; (j = q.next_gu.fetch_add(1)) < gu_total;) {
    const int e = j / tic, tile = j % tic;
    moe_cpu_gate_up(o_.isa, s_, q.blob[e], x_, s_.hidden, q.mask[e], tile, tile + 1, h_.data() + static_cast<size_t>(e) * kStepRows32 * s_.inter_c);
    q.left_gu.fetch_sub(1);
  }
  while (q.left_gu.load() > 0) cpu_relax();  // every h written
  for (int j; (j = q.next_dn.fetch_add(1)) < dn_total;) {
    const int e = j / dn_jobs, t0 = (j % dn_jobs) * kDnChunk;
    moe_cpu_down(o_.isa, s_, q.blob[e], h_.data() + static_cast<size_t>(e) * kStepRows32 * s_.inter_c, q.mask[e], t0, t0 + kDnChunk,
                 y_ + static_cast<size_t>(e) * kStepRows32 * s_.hidden, s_.hidden);
    q.left_dn.fetch_sub(1);
  }
}

void MoeCpuPool::serve(uint32_t r) {
  const auto t0 = clk::now();
  const uint32_t g = gen_.load() + 1;
  Req &q = req_[g & 1u];
  while (q.users.load() > 0) cpu_relax();  // a worker still leaving request g - 2
  const int n = std::min<int>(std::max(mail_->n, 0), kMoeCpuMax);
  q.n = n;
  for (int i = 0; i < n; ++i) {
    q.mask[i] = mail_->mask[i];
    q.blob[i] = reinterpret_cast<const uint8_t *>(mail_->blob[i]);
  }
  if (o_.stall_us > 0) std::this_thread::sleep_for(std::chrono::microseconds(o_.stall_us));
  const int tic = s_.inter_c / kW, dn_jobs = s_.hidden / kW / kDnChunk;
  q.next_gu.store(0);
  q.next_dn.store(0);
  q.left_gu.store(n * tic);
  q.left_dn.store(n * dn_jobs);
  gen_.store(g);  // the workers go
  compute(q);
  while (q.left_dn.load() > 0) cpu_relax();
  vstore(&mail_->done, r);
  served_.fetch_add(1, std::memory_order_relaxed);
  experts_.fetch_add(static_cast<uint64_t>(n), std::memory_order_relaxed);
  busy_ns_.fetch_add(static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(clk::now() - t0).count()), std::memory_order_relaxed);
}

void MoeCpuPool::worker(int t) {
  if (o_.first_core >= 0) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(o_.first_core + t, &set);
    pthread_setaffinity_np(pthread_self(), sizeof set, &set);
  }
  auto last = clk::now();
  const auto spin = std::chrono::microseconds(o_.spin_us);
  if (t == 0) {  // the poller: the card's requests
    uint32_t seen = seen0_;
    while (!stop_.load(std::memory_order_relaxed)) {
      const uint32_t r = vload(&mail_->req);
      if (r == seen) {
        if (clk::now() - last > spin) nap();
        else cpu_relax();
        continue;
      }
      seen = r;
      serve(r);
      last = clk::now();
    }
    return;
  }
  uint32_t done_g = gen_.load();
  while (!stop_.load(std::memory_order_relaxed)) {
    const uint32_t g = gen_.load();
    if (g == done_g) {
      if (clk::now() - last > spin) nap();
      else cpu_relax();
      continue;
    }
    Req &q = req_[g & 1u];
    q.users.fetch_add(1);
    if (gen_.load() != g) {  // moved on meanwhile: the slot may be refilled; look again
      q.users.fetch_sub(1);
      continue;
    }
    compute(q);
    q.users.fetch_sub(1);
    done_g = g;
    last = clk::now();
  }
}

void MoeCpuPool::run_once(int n, int rows, const uint32_t *mask, const uint8_t *const *blob) {
  n = std::min(n, kMoeCpuMax);
  mail_->n = n;
  mail_->rows = rows;
  for (int i = 0; i < n; ++i) {
    mail_->mask[i] = mask[i];
    mail_->blob[i] = reinterpret_cast<uint64_t>(blob[i]);
  }
  const uint32_t r = vload(&mail_->req) + 1;
  vstore(&mail_->req, r);
  while (vload(&mail_->done) != r) cpu_relax();
}

}  // namespace xenithium:kernels
