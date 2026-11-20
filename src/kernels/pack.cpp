#include "xenithium/kernels/pack.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace xenithium::kernels {

void pack_weights_vnni(const int8_t *w, int n, int k, int8_t *out) {
  const int ntiles = n / kI8Width;
  const int chunks = k / kI8KPerCall;
  for (int c = 0; c < chunks; ++c) {
    for (int t = 0; t < ntiles; ++t) {
      int8_t *block = out + static_cast<size_t>(c * ntiles + t) * kI8BlockBytes;
      for (int lane = 0; lane < kI8Width; ++lane) {
        const int8_t *src = w + static_cast<size_t>(t * kI8Width + lane) * k +
                            static_cast<size_t>(c) * kI8KPerCall;
        for (int kk = 0; kk < kI8KPerCall; ++kk) {
          // VNNI: one dword per 4 consecutive k of a column, and the 16 lanes'
          const int dword = (kk / 4) * kI8Width + lane;
          block[dword * 4 + (kk % 4)] = src[kk];
        }
      }
    }
  }
}

void gemm_int8_ref(const int8_t *a, const int8_t *w, int m, int n, int k, int32_t *c) {
  for (int row = 0; row < m; ++row) {
    for (int col = 0; col < n; ++col) {
      int32_t acc = 0;
      const int8_t *wr = w + static_cast<size_t>(col) * k;
      const int8_t *ar = a + static_cast<size_t>(row) * k;
      for (int kk = 0; kk < k; ++kk) acc += static_cast<int32_t>(ar[kk]) * static_cast<int32_t>(wr[kk]);
      c[static_cast<size_t>(row) * n + col] = acc;
    }
  }
}

// Output byte (k pair kp of group g, column c of tile t) holds nibbles k = g * 128 + 2 kp
// (low) and k + 1 (high): exactly byte (k % 8) / 2 of the GPTQ word qweight[k / 8][t * 16 + c]
// (little-endian, nibble k % 8 at bits 4 (k % 8)). So the pack is a strided byte gather,
// written in output order (the model load packs ~24K experts with it).
void pack_w4_gptq_cat(const uint32_t *a, const uint32_t *b, int n_a, int n, int k, uint8_t *out) {
  const int ntiles = n / kW4Width, groups = k / kW4Group, tiles_a = n_a / kW4Width;
  for (int g = 0; g < groups; ++g)
    for (int kp = 0; kp < kW4Group / 2; ++kp) {
      const int kk = g * kW4Group + 2 * kp;
      const int sub = (kk % 8) / 2;
      const uint8_t *row_a = reinterpret_cast<const uint8_t *>(a + static_cast<size_t>(kk / 8) * n_a) + sub;
      const uint8_t *row_b = b ? reinterpret_cast<const uint8_t *>(b + static_cast<size_t>(kk / 8) * (n - n_a)) + sub : nullptr;
      for (int t = 0; t < ntiles; ++t) {
        uint8_t *o = out + (static_cast<size_t>(g) * ntiles + t) * kW4BlockBytes + kp * kW4Width;
        const uint8_t *s = t < tiles_a ? row_a + static_cast<size_t>(t) * kW4Width * 4
                                       : row_b + static_cast<size_t>(t - tiles_a) * kW4Width * 4;
        for (int c = 0; c < kW4Width; ++c) o[c] = s[4 * c];
      }
    }
}

void pack_w4_gptq(const uint32_t *qweight, int n, int k, uint8_t *out) { pack_w4_gptq_cat(qweight, nullptr, n, n, k, out); }

void pack_w4_gptq_rows(const uint32_t *qweight, int n, int k, int r0, int r1, uint8_t *out) {
  const int ntiles = n / kW4Width, groups = k / kW4Group;
  for (int g = 0; g < groups; ++g)
    for (int kp = 0; kp < kW4Group / 2; ++kp) {
      const int kk = g * kW4Group + 2 * kp;
      const uint8_t *row = reinterpret_cast<const uint8_t *>(qweight + static_cast<size_t>(kk / 8) * n) + (kk % 8) / 2;
      for (int t = r0 / kW4Width; t < r1 / kW4Width; ++t) {
        uint8_t *o = out + (static_cast<size_t>(g) * ntiles + t) * kW4BlockBytes + kp * kW4Width;
        const uint8_t *s = row + static_cast<size_t>(t) * kW4Width * 4;
        for (int c = 0; c < kW4Width; ++c) o[c] = s[4 * c];
      }
    }
}

void w4a16_ref(const float *a, const uint32_t *qweight, const float *scales, int m, int n, int k,
               double *c) {
  const int groups = k / kW4Group;
  for (int row = 0; row < m; ++row) {
    for (int col = 0; col < n; ++col) {
      double acc = 0;
      for (int g = 0; g < groups; ++g) {
        double gsum = 0;
        for (int kg = 0; kg < kW4Group; ++kg) {
          const int kk = g * kW4Group + kg;
          const int q = static_cast<int>((qweight[static_cast<size_t>(kk / 8) * n + col] >> (4 * (kk % 8))) & 15u);
          gsum += static_cast<double>(a[static_cast<size_t>(row) * k + kk]) * (q - kW4Zero);
        }
        acc += gsum * scales[static_cast<size_t>(g) * n + col];
      }
      c[static_cast<size_t>(row) * n + col] = acc;
    }
  }
}

void w4zp_ref(const float *a, const uint32_t *packed, const float *scales, const uint8_t *zeros, int m, int n, int k, double *c) {
  const int groups = k / kW4ZpGroup;
  for (int row = 0; row < m; ++row)
    for (int col = 0; col < n; ++col) {
      double acc = 0;
      for (int g = 0; g < groups; ++g) {
        double gsum = 0;
        const int z = zeros[static_cast<size_t>(col) * groups + g];
        for (int kg = 0; kg < kW4ZpGroup; ++kg) {
          const int kk = g * kW4ZpGroup + kg;
          gsum += static_cast<double>(a[static_cast<size_t>(row) * k + kk]) * (ct_code(packed, k, col, kk) - z);
        }
        acc += gsum * scales[static_cast<size_t>(col) * groups + g];
      }
      c[static_cast<size_t>(row) * n + col] = acc;
    }
}

void w4aff_ref(const float *a, const uint32_t *packed, const float *scales, const float *biases, int m, int n, int k, double *c) {
  const int groups = k / kW4AffGroup;
  for (int row = 0; row < m; ++row)
    for (int col = 0; col < n; ++col) {
      double acc = 0;
      for (int g = 0; g < groups; ++g) {
        double qsum = 0, asum = 0;
        for (int kg = 0; kg < kW4AffGroup; ++kg) {
          const int kk = g * kW4AffGroup + kg;
          const double av = a[static_cast<size_t>(row) * k + kk];
          qsum += av * ct_code(packed, k, col, kk);
          asum += av;
        }
        const size_t gi = static_cast<size_t>(col) * groups + g;
        acc += qsum * scales[gi] + asum * biases[gi];
      }
      c[static_cast<size_t>(row) * n + col] = acc;
    }
}

void pack_w4_zeros(const uint8_t *z, int n, int k, uint8_t *out) {
  const int groups = k / kW4ZpGroup;
  for (int h = 0; h < groups / 2; ++h)
    for (int col = 0; col < n; ++col)
      out[static_cast<size_t>(h) * n + col] = static_cast<uint8_t>((z[static_cast<size_t>(col) * groups + 2 * h] & 15) |
                                                                   ((z[static_cast<size_t>(col) * groups + 2 * h + 1] & 15) << 4));
}

// compressed-tensors [n][k/8] -> GPTQ [k/8][n]: the same words (code i at bits 4i for k = 8j + i),
// only their order changes, so a word moves whole.
void ct_to_gptq_qweight(const uint32_t *packed, int n, int k, uint32_t *gptq_out) {
  const int words = k / 8;
  for (int row = 0; row < n; ++row)
    for (int j = 0; j < words; ++j) gptq_out[static_cast<size_t>(j) * n + row] = packed[static_cast<size_t>(row) * words + j];
}

// Block (g, t) byte (kg / 2) * 16 + lane = codes (k = g * 128 + kg, k + 1) of column t * 16 + lane: both sit in
// one compressed-tensors word (8 codes per word, kg even), at bits 4 * (kg % 8).
void pack_w4_ct_rows(const uint32_t *packed, int n, int k, int r0, int r1, uint8_t *out) {
  const int ntiles = n / kW4Width, groups = k / kW4Group, words = k / 8;
  for (int g = 0; g < groups; ++g)
    for (int t = r0 / kW4Width; t < r1 / kW4Width; ++t) {
      uint8_t *o = out + (static_cast<size_t>(g) * ntiles + t) * kW4BlockBytes;
      for (int lane = 0; lane < kW4Width; ++lane) {
        const uint32_t *row = packed + static_cast<size_t>(t * kW4Width + lane) * words + static_cast<size_t>(g) * (kW4Group / 8);
        for (int kp = 0; kp < kW4Group / 2; ++kp) o[kp * kW4Width + lane] = static_cast<uint8_t>(row[kp / 4] >> (8 * (kp % 4)));
      }
    }
}

void ct_to_gptq_scales(const uint16_t *scales, int n, int k, uint16_t *scales_out) {
  const int groups = k / kW4Group;
  for (int row = 0; row < n; ++row)
    for (int g = 0; g < groups; ++g) scales_out[static_cast<size_t>(g) * n + row] = scales[static_cast<size_t>(row) * groups + g];
}


// ---- dense ------------------------------------------------------------------------------
size_t dense_packed_elems(int n, int k) { return static_cast<size_t>(n) * k; }

namespace {
// Element offset of W[row][col] in the packed dense layout.
size_t dense_offset(int n, int k, int row, int col) {
  const int tiles = n / 16, full = k / 128, t = row / 16, lane = row % 16;
  const int g = col / 128, kk = col % 128, j = kk / 16, kin = kk % 16;
  const size_t in_call = static_cast<size_t>(j) * kDenseCallElems + (kin / 2) * 32 + lane * 2 + (kin & 1);
  if (g < full) return (static_cast<size_t>(g) * tiles + t) * (kDenseCalls * kDenseCallElems) + in_call;
  const int calls = (k % 128) / 16;
  return static_cast<size_t>(full) * tiles * kDenseCalls * kDenseCallElems +
         static_cast<size_t>(t) * calls * kDenseCallElems + in_call;
}
template <typename T>
void pack_dense(const T *w, int n, int k, int r0, int r1, T *out) {
  for (int r = r0; r < r1; ++r)
    for (int c = 0; c < k; ++c) out[dense_offset(n, k, r, c)] = w[static_cast<size_t>(r) * k + c];
}
}  // namespace

void pack_dense_bf16(const uint16_t *w, int n, int k, uint16_t *out) { pack_dense(w, n, k, 0, n, out); }
void pack_dense_i8(const int8_t *codes, int n, int k, int8_t *out) { pack_dense(codes, n, k, 0, n, out); }
void pack_dense_bf16_rows(const uint16_t *w, int n, int k, int r0, int r1, uint16_t *out) { pack_dense(w, n, k, r0, r1, out); }
void pack_dense_i8_rows(const int8_t *codes, int n, int k, int r0, int r1, int8_t *out) { pack_dense(codes, n, k, r0, r1, out); }

float bf16_value(uint16_t bits) {
  const uint32_t u = static_cast<uint32_t>(bits) << 16;
  float v;
  std::memcpy(&v, &u, 4);
  return v;
}

uint16_t bf16_bits(float v) {
  uint32_t u;
  std::memcpy(&u, &v, 4);
  if ((u & 0x7fffffffu) > 0x7f800000u) return static_cast<uint16_t>((u >> 16) | 0x40);  // NaN stays NaN
  u += 0x7fffu + ((u >> 16) & 1u);
  return static_cast<uint16_t>(u >> 16);
}

void quantize_i8_rows(const uint16_t *w, int n, int k, int8_t *codes, float *scale, int clips) {
  for (int r = 0; r < n; ++r) {
    const uint16_t *row = w + static_cast<size_t>(r) * k;
    float amax = 0.f;
    for (int i = 0; i < k; ++i) amax = std::max(amax, std::fabs(bf16_value(row[i])));
    float best_sc = amax > 0.f ? amax / 127.f : 1.f;
    if (clips > 1 && amax > 0.f) {
      double best_err = -1;
      for (int c = 0; c < clips; ++c) {
        const float ratio = 1.f - 0.25f * static_cast<float>(c) / static_cast<float>(clips - 1);  // 1.0.. 0.75
        const float sc = amax * ratio / 127.f;
        double err = 0;
        for (int i = 0; i < k; ++i) {
          const float v = bf16_value(row[i]);
          const float q = std::clamp(std::nearbyint(v / sc), -127.f, 127.f);
          const double d = static_cast<double>(v) - static_cast<double>(q) * sc;
          err += d * d;
        }
        if (best_err < 0 || err < best_err) {
          best_err = err;
          best_sc = sc;
        }
      }
    }
    scale[r] = best_sc;
    for (int i = 0; i < k; ++i)
      codes[static_cast<size_t>(r) * k + i] =
          static_cast<int8_t>(std::clamp(std::nearbyint(bf16_value(row[i]) / best_sc), -127.f, 127.f));
  }
}

float e4m3_value(uint8_t code) {
  const int s = code >> 7, e = (code >> 3) & 15, m = code & 7;
  if (e == 15 && m == 7) return NAN;
  const float v = e == 0 ? std::ldexp(static_cast<float>(m), -9) : std::ldexp(1.f + m / 8.f, e - 7);
  return s ? -v : v;
}

void quantize_e4m3_rows(const uint16_t *w, int n, int k, uint8_t *codes, float *scale) {
  // Nearest finite code by search over the 127 non-negative finite codes (ties: even code).
  float table[127];
  for (int c = 0; c < 127; ++c) table[c] = e4m3_value(static_cast<uint8_t>(c));
  for (int r = 0; r < n; ++r) {
    const uint16_t *row = w + static_cast<size_t>(r) * k;
    float amax = 0.f;
    for (int i = 0; i < k; ++i) amax = std::max(amax, std::fabs(bf16_value(row[i])));
    const float sc = amax > 0.f ? amax / 448.f : 1.f;
    scale[r] = sc;
    for (int i = 0; i < k; ++i) {
      const float v = bf16_value(row[i]) / sc, a = std::fabs(v);
      int lo = static_cast<int>(std::upper_bound(table, table + 127, a) - table) - 1;
      lo = std::clamp(lo, 0, 126);
      int c = lo;
      if (lo + 1 < 127) {
        const float dl = a - table[lo], dh = table[lo + 1] - a;
        if (dh < dl || (dh == dl && ((lo + 1) % 2 == 0))) c = lo + 1;
      }
      codes[static_cast<size_t>(r) * k + i] = static_cast<uint8_t>(c | (v < 0 ? 0x80 : 0));
    }
  }
}

void dense_ref_bf16(const float *x, const uint16_t *w, int m, int n, int k, double *y) {
  for (int r = 0; r < m; ++r)
    for (int c = 0; c < n; ++c) {
      double acc = 0;
      for (int i = 0; i < k; ++i) acc += static_cast<double>(x[static_cast<size_t>(r) * k + i]) * bf16_value(w[static_cast<size_t>(c) * k + i]);
      y[static_cast<size_t>(r) * n + c] = acc;
    }
}

void dense_ref_i8(const float *x, const int8_t *codes, const float *scale, int m, int n, int k, double *y) {
  for (int r = 0; r < m; ++r)
    for (int c = 0; c < n; ++c) {
      double acc = 0;
      for (int i = 0; i < k; ++i) acc += static_cast<double>(x[static_cast<size_t>(r) * k + i]) * codes[static_cast<size_t>(c) * k + i];
      y[static_cast<size_t>(r) * n + c] = acc * scale[c];
    }
}

void dense_ref_a8(const int8_t *aq, const float *as, int lds, const int8_t *codes, const float *scale, int m, int n, int k, double *y) {
  const int gpr = (k + 127) / 128;
  for (int r = 0; r < m; ++r)
    for (int c = 0; c < n; ++c) {
      double acc = 0;
      for (int g = 0; g < gpr; ++g) {
        long long s = 0;
        const int k1 = std::min(k, (g + 1) * 128);
        for (int i = g * 128; i < k1; ++i) s += static_cast<int>(aq[static_cast<size_t>(r) * k + i]) * codes[static_cast<size_t>(c) * k + i];
        acc += static_cast<double>(static_cast<float>(s)) * as[static_cast<size_t>(g) * lds + r];
      }
      y[static_cast<size_t>(r) * n + c] = acc * scale[c];
    }
}

// ---- i2 -------------------------------------------------------------------------
size_t dense_i2_bytes(int n, int k) { return static_cast<size_t>(n) * k / 4; }

void pack_dense_i2(const uint8_t *codes, int n, int k, uint8_t *out) {
  if (k % 128) throw std::invalid_argument("pack_dense_i2: k % 128");
  std::memset(out, 0, dense_i2_bytes(n, k));
  for (int r = 0; r < n; ++r)
    for (int c = 0; c < k; ++c) {
      const size_t e = dense_offset(n, k, r, c);
      out[e / 4] |= static_cast<uint8_t>((codes[static_cast<size_t>(r) * k + c] & 3u) << (2 * (e % 4)));
    }
}

void quantize_i2_groups(const uint16_t *w, int n, int k, uint8_t *codes, float *scale, int r0, int r1) {
  if (k % 128) throw std::invalid_argument("quantize_i2_groups: k % 128");
  if (r1 < 0) r1 = n;
  const int G = k / 128;
  for (int r = r0; r < r1; ++r)
    for (int g = 0; g < G; ++g) {
      float v[128];
      double ss = 0;
      for (int i = 0; i < 128; ++i) {
        v[i] = bf16_value(w[static_cast<size_t>(r) * k + g * 128 + i]);
        ss += static_cast<double>(v[i]) * v[i];
      }
      const double rms = std::sqrt(ss / 128);
      float best_s = 0.f;
      double best_e = -1;
      if (rms > 0)
        for (int t = 0; t <= 24; ++t) {  // 0.5.. 1.7 x rms
          const float sc = static_cast<float>(rms * (0.5 + 0.05 * t));
          double e = 0;
          for (int i = 0; i < 128; ++i) {
            const float q = std::clamp(std::floor(v[i] / sc) + 2.f, 0.f, 3.f);  // nearest of the levels (q - 1.5) s: boundaries -1, 0, 1
            const double dlt = static_cast<double>(v[i]) - (q - 1.5) * sc;
            e += dlt * dlt;
          }
          if (best_e < 0 || e < best_e) best_e = e, best_s = sc;
        }
      scale[static_cast<size_t>(g) * n + r] = best_s;
      for (int i = 0; i < 128; ++i) {
        const float q = best_s > 0.f ? std::clamp(std::floor(v[i] / best_s) + 2.f, 0.f, 3.f) : 1.f;
        codes[static_cast<size_t>(r) * k + g * 128 + i] = static_cast<uint8_t>(q);
      }
    }
}

void dense_ref_i2(const float *x, const uint8_t *codes, const float *scale, int m, int n, int k, double *y) {
  const int G = k / 128;
  for (int r = 0; r < m; ++r)
    for (int c = 0; c < n; ++c) {
      double acc = 0;
      for (int g = 0; g < G; ++g) {
        double gs = 0;
        for (int i = g * 128; i < (g + 1) * 128; ++i) gs += static_cast<double>(x[static_cast<size_t>(r) * k + i]) * (codes[static_cast<size_t>(c) * k + i] - 1.5);
        acc += gs * scale[static_cast<size_t>(g) * n + c];
      }
      y[static_cast<size_t>(r) * n + c] = acc;
    }
}

// ----: the W4A8 reference ----
void w4a8_ref(const int8_t *aq, const float *as, int lds, const uint32_t *qweight, const float *scales, int m, int n, int k, double *c) {
  const int groups = k / kW4Group;
  for (int r = 0; r < m; ++r)
    for (int col = 0; col < n; ++col) {
      double acc = 0;
      for (int g = 0; g < groups; ++g) {
        int64_t gs = 0;
        for (int kk = g * kW4Group; kk < (g + 1) * kW4Group; ++kk) {
          const int q = static_cast<int>((qweight[static_cast<size_t>(kk / 8) * n + col] >> (4 * (kk % 8))) & 15u) - kW4Zero;
          gs += static_cast<int64_t>(aq[static_cast<size_t>(r) * k + kk]) * q;
        }
        acc += static_cast<double>(gs) * as[static_cast<size_t>(g) * lds + r] * scales[static_cast<size_t>(g) * n + col];
      }
      c[static_cast<size_t>(r) * n + col] = acc;
    }
}

}  // namespace xenithium:kernels
