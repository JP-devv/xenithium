// Xenithium · QSA rotary table and host reference (host-only, precise floating point).
#include "xenithium/kernels/attention/qsa_ref.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

#include "xenithium/kernels/pack.hpp"

namespace xenithium::kernels {
namespace {
float bfv(double v) { return bf16_value(bf16_bits(static_cast<float>(v))); }
}  // namespace

QsaYarnBounds qsa_yarn_bounds(int rot_dim, double theta, int64_t orig, double beta_fast, double beta_slow) {
  auto dim_of = [&](double rot) { return rot_dim * std::log(static_cast<double>(orig) / (rot * 2 * M_PI)) / (2 * std::log(theta)); };
  QsaYarnBounds b;
  b.low = std::max(0, static_cast<int>(std::floor(dim_of(beta_fast))));
  b.high = std::min(rot_dim - 1, static_cast<int>(std::ceil(dim_of(beta_slow))));
  return b;
}

QsaYarn qsa_yarn_for(int64_t context, int64_t trained) {
  if (trained < 1) throw std::invalid_argument("qsa_yarn_for: trained positions must be >= 1");
  QsaYarn y;
  y.orig = trained;
  if (context <= trained) return y;
  const int64_t f100 = (context * 100 + trained - 1) / trained;  // ceil(100 x context / trained): exact in integers
  y.factor = static_cast<double>(f100) / 100.0;  // whole factors (2, 4) come out exact
  return y;
}

std::vector<uint16_t> qsa_rope_table(int64_t positions, int rot_dim, double theta, const QsaYarn &y) {
  // rot_dim 64 (qwen38 / qwen35: kQsaRopeRow) or any even width up to 256 (llama: 128); the row = rot_dim bf16 (cos | sin)
  if (rot_dim < 2 || rot_dim % 2 || rot_dim > kQsaDim) throw std::invalid_argument("qsa_rope_table: rotary dim must be even, 2..256");
  if (!(y.factor >= 1.0)) throw std::invalid_argument("qsa_rope_table: YaRN factor must be >= 1");
  const int half = rot_dim / 2;
  std::vector<float> inv(half);
  const float base = static_cast<float>(theta);
  for (int i = 0; i < half; ++i) {
    const float e = static_cast<float>(2 * i) / static_cast<float>(rot_dim);
    const float pf = std::pow(base, e);
    inv[i] = 1.0f / pf;
    if (y.factor > 1.0) {
      const QsaYarnBounds b = qsa_yarn_bounds(rot_dim, theta, y.orig, y.beta_fast, y.beta_slow);
      float lo = static_cast<float>(b.low), hi = static_cast<float>(b.high);
      if (lo == hi) hi += 0.001f;
      const float ramp = std::clamp((static_cast<float>(i) - lo) / (hi - lo), 0.f, 1.f);
      const float mask = 1.f - ramp;  // 1 = keep the original frequency (extrapolation)
      const float inter = 1.0f / (static_cast<float>(y.factor) * pf), extra = 1.0f / pf;
      inv[i] = inter * (1.f - mask) + extra * mask;
    }
  }
  const float mscale = y.factor > 1.0 ? static_cast<float>(y.attn_factor > 0 ? y.attn_factor : 0.1 * std::log(y.factor) + 1.0)
                                      : (y.attn_factor > 0 ? static_cast<float>(y.attn_factor) : 1.f);
  std::vector<uint16_t> t(static_cast<size_t>(positions) * rot_dim);
  for (int64_t p = 0; p < positions; ++p) {
    uint16_t *row = &t[static_cast<size_t>(p) * rot_dim];
    for (int i = 0; i < half; ++i) {
      const float ang = static_cast<float>(p) * inv[i];  // the angle in fp32
      row[i] = bf16_bits(static_cast<float>(std::cos(static_cast<double>(ang))) * mscale);
      row[half + i] = bf16_bits(static_cast<float>(std::sin(static_cast<double>(ang))) * mscale);
    }
  }
  return t;
}

void qsa_ref_norm_rope(const float *x, const float *w, int dim, const uint16_t *rope_row, float eps, float *out) {
  double ss = 0;
  for (int d = 0; d < dim; ++d) ss += static_cast<double>(x[d]) * x[d];
  const double r = 1.0 / std::sqrt(ss / dim + eps);
  std::vector<float> n(dim);
  for (int d = 0; d < dim; ++d) n[d] = bfv(x[d] * r * (1.0 + w[d]));
  const int half = kQsaRot / 2;
  for (int d = 0; d < dim; ++d) {
    if (d >= kQsaRot) {
      out[d] = n[d];
      continue;
    }
    const int i = d % half;
    const double c = bf16_value(rope_row[i]), s = bf16_value(rope_row[half + i]);
    out[d] = d < half ? bfv(n[d] * c - n[d + half] * s) : bfv(n[d] * c + n[d - half] * s);
  }
}

namespace {
void ref_sym(const float *x, int n, int8_t *codes, float *scale, float q) {
  float amax = 0.f;
  for (int i = 0; i < n; ++i) amax = std::max(amax, std::fabs(x[i]));
  if (!(amax > 0.f)) {
    *scale = 0.f;
    for (int i = 0; i < n; ++i) codes[i] = 0;
    return;
  }
  const float inv = q / amax;
  *scale = amax / q;
  for (int i = 0; i < n; ++i) {
    const float t = std::nearbyint(x[i] * inv);
    codes[i] = static_cast<int8_t>(std::max(-q, std::min(q, t)));
  }
}
}  // namespace

void qsa_ref_i8(const float *x, int n, int8_t *codes, float *scale) { ref_sym(x, n, codes, scale, 127.f); }
void qsa_ref_v4(const float *x, int8_t *codes, uint16_t *scales) {
  for (int g = 0; g < kQsaV4Scales; ++g) {
    const float *xg = x + g * kQsaV4Group;
    int8_t *cg = codes + g * kQsaV4Group;
    float amax = 0.f;
    for (int i = 0; i < kQsaV4Group; ++i) amax = std::max(amax, std::fabs(xg[i]));
    const uint16_t sb = amax > 0.f ? bf16_bits(amax / 7.f) : 0;
    scales[g] = sb;
    const float s = bf16_value(sb);
    const float inv = s > 0.f ? 1.f / s : 0.f;
    for (int i = 0; i < kQsaV4Group; ++i) {
      const float t = std::nearbyint(xg[i] * inv);
      cg[i] = static_cast<int8_t>(std::max(-7.f, std::min(7.f, t)));
    }
  }
}

void qsa_ref_compress(const float *raw4, const float *w, const uint16_t *rope_row, float eps, float *kc) {
  float mean[kQsaIdxDim];
  for (int d = 0; d < kQsaIdxDim; ++d) {
    float s = raw4[d];
    for (int j = 1; j < kQsaCompress; ++j) s += raw4[j * kQsaIdxDim + d];
    mean[d] = bfv(s / static_cast<float>(kQsaCompress));
  }
  qsa_ref_norm_rope(mean, w, kQsaIdxDim, rope_row, eps, kc);
}

double qsa_ref_score(const float *iq, const float *kc) {
  double s = 0;
  for (int h = 0; h < kQsaIdxHeads; ++h) {
    double d = 0;
    for (int i = 0; i < kQsaIdxDim; ++i) d += static_cast<double>(iq[h * kQsaIdxDim + i]) * kc[i];
    s += std::max(d, 0.0);
  }
  return s / std::sqrt(static_cast<double>(kQsaIdxDim));
}

std::vector<int> qsa_ref_select(const uint32_t *keys, int n, int k) {
  std::vector<int> idx(n);
  std::iota(idx.begin(), idx.end(), 0);
  if (n > k) {
    std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) { return keys[a] > keys[b]; });
    idx.resize(k);
    std::sort(idx.begin(), idx.end());
  }
  return idx;
}

std::vector<int> qsa_ref_tokens(const std::vector<int> &blocks, int p) {
  std::vector<int> t;
  for (int b : blocks)
    for (int j = 0; j < kQsaCompress; ++j) t.push_back(b * kQsaCompress + j);
  const int nblk = (p + 1) / kQsaCompress;
  for (int q = nblk * kQsaCompress; q <= p; ++q) t.push_back(q);
  return t;
}

void qsa_ref_attend(const float *q, int ntok, const float *K, const float *V, double *o) {
  std::vector<double> s(ntok);
  double m = -INFINITY;
  for (int t = 0; t < ntok; ++t) {
    double d = 0;
    for (int i = 0; i < kQsaDim; ++i) d += static_cast<double>(q[i]) * K[static_cast<size_t>(t) * kQsaDim + i];
    s[t] = d / 16.0;
    m = std::max(m, s[t]);
  }
  double l = 0;
  for (int t = 0; t < ntok; ++t) l += (s[t] = std::exp(s[t] - m));
  for (int i = 0; i < kQsaDim; ++i) o[i] = 0;
  for (int t = 0; t < ntok; ++t)
    for (int i = 0; i < kQsaDim; ++i) o[i] += s[t] / l * V[static_cast<size_t>(t) * kQsaDim + i];
}

void qsa_ref_gate(const double *o, const float *gate, float *out) {
  for (int i = 0; i < kQsaDim; ++i) {
    const float g = bfv(1.0 / (1.0 + std::exp(-static_cast<double>(gate[i]))));
    out[i] = bfv(static_cast<double>(bfv(o[i])) * g);
  }
}

}  // namespace xenithium::kernels
