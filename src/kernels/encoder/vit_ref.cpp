// Xenithium · host references of the vision encoder's own kernels (xenithium/kernels/encoder/vit.hpp), bf16 as bits.
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "xenithium/kernels/encoder/vit_ref.hpp"
#include "xenithium/kernels/pack.hpp"

namespace xenithium::kernels {

void vit_pos_add_ref(int n, int hidden, const uint16_t *table, const int32_t *idx, const uint16_t *w, uint16_t *x) {
  for (int p = 0; p < n; ++p)
    for (int h = 0; h < hidden; ++h) {
      float s = 0.f;
      for (int k = 0; k < 4; ++k) {
        const float e = bf16_value(bf16_bits(bf16_value(table[static_cast<size_t>(idx[p * 4 + k]) * hidden + h]) * bf16_value(w[p * 4 + k])));
        s = k == 0 ? e : s + e;
      }
      uint16_t &o = x[static_cast<size_t>(p) * hidden + h];
      o = bf16_bits(bf16_value(o) + bf16_value(bf16_bits(s)));
    }
}

void vit_rope_ref(int n, int heads, int hd, const float *cs, const float *sn, uint16_t *qkv) {
  const int half = hd / 2;
  for (int p = 0; p < n; ++p)
    for (int h = 0; h < 2 * heads; ++h) {
      uint16_t *b = qkv + static_cast<size_t>(p) * 3 * heads * hd + static_cast<size_t>(h) * hd;
      for (int i = 0; i < half; ++i) {
        const float x1 = bf16_value(b[i]), x2 = bf16_value(b[i + half]);
        const float c = cs[static_cast<size_t>(p) * half + i], s = sn[static_cast<size_t>(p) * half + i];
        b[i] = bf16_bits(std::fma(x1, c, -(x2 * s)));
        b[i + half] = bf16_bits(std::fma(x2, c, x1 * s));
      }
    }
}

void vit_layer_norm_ref(int m, int n, const uint16_t *x, const float *gamma, const float *beta, float eps, uint16_t *y) {
  constexpr int L = 16;
  for (int r = 0; r < m; ++r) {
    const uint16_t *xr = x + static_cast<size_t>(r) * n;
    float part[L];
    for (int l = 0; l < L; ++l) {
      float s = 0.f;
      for (int c = l; c < n; c += L) s += bf16_value(xr[c]);
      part[l] = s;
    }
    float sum = 0.f;
    for (int l = 0; l < L; ++l) sum += part[l];
    const float mean = sum / static_cast<float>(n);
    for (int l = 0; l < L; ++l) {
      float s = 0.f;
      for (int c = l; c < n; c += L) {
        const float d = bf16_value(xr[c]) - mean;
        s += d * d;
      }
      part[l] = s;
    }
    float ss = 0.f;
    for (int l = 0; l < L; ++l) ss += part[l];
    const float rstd = 1.f / std::sqrt(ss / static_cast<float>(n) + eps);
    for (int c = 0; c < n; ++c) y[static_cast<size_t>(r) * n + c] = bf16_bits(std::fma((bf16_value(xr[c]) - mean) * rstd, gamma[c], beta[c]));
  }
}

}  // namespace xenithium::kernels
