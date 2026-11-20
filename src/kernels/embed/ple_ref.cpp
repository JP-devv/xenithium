#include "xenithium/kernels/embed/ple_ref.hpp"

#include <cmath>
#include <stdexcept>

#include "xenithium/kernels/pack.hpp"

namespace xenithium::kernels {

namespace {
float bfr(double v) { return bf16_value(bf16_bits(static_cast<float>(v))); }
float bff(float v) { return bf16_value(bf16_bits(v)); }
}  // namespace

uint64_t ple_splitmix64(uint64_t v) {
  v += 0x9E3779B97F4A7C15ull;
  v = (v ^ (v >> 30)) * 0xBF58476D1CE4E5B9ull;
  v = (v ^ (v >> 27)) * 0x94D049BB133111EBull;
  return v ^ (v >> 31);
}

bool ple_is_prime(int64_t v) {
  if (v < 2) return false;
  if (v % 2 == 0) return v == 2;
  for (int64_t d = 3; d * d <= v; d += 2)
    if (v % d == 0) return false;
  return true;
}

PleLayout PleLayout::derive(int64_t vocab, int ngram, int heads_per_ngram, int64_t base, int64_t seed, int layer_index, int64_t divisor,
                            int64_t eos) {
  if (ngram < 2 || ngram > kPleMaxNgram || heads_per_ngram < 1 || vocab < 1 || divisor < 1)
    throw std::runtime_error("ple layout: unsupported ngram / heads / vocab / divisor");
  PleLayout L;
  L.ngram = ngram;
  L.heads_per_ngram = heads_per_ngram;
  L.heads = (ngram - 1) * heads_per_ngram;
  L.eos = eos;
  const int64_t max_mult = INT64_MAX / vocab;
  const uint64_t half = static_cast<uint64_t>(std::max<int64_t>(1, max_mult / 2));
  const uint64_t base_seed = static_cast<uint64_t>(seed) + 10007ull * static_cast<uint64_t>(layer_index);
  for (int i = 0; i < ngram; ++i)
    L.mult.push_back(static_cast<int64_t>(2 * (ple_splitmix64(base_seed + 0x9E3779B97F4A7C15ull * static_cast<uint64_t>(i + 1)) % half) + 1));
  // The (global head + 1)-th prime after base - 1, global head = layer_index * heads + h.
  int64_t p = base - 1, found = 0;
  const int64_t first = static_cast<int64_t>(layer_index) * L.heads;
  for (int h = 0; h < L.heads; ++h) {
    while (found < first + h + 1) {
      ++p;
      while (!ple_is_prime(p)) ++p;
      ++found;
    }
    L.size.push_back(p);
    L.offset.push_back(L.rows);
    L.rows += p;
  }
  L.padded_rows = (L.rows + divisor - 1) / divisor * divisor;
  return L;
}

void ple_ngram_ids(const PleLayout &L, int64_t tok, const int64_t *prev, int64_t *ids) {
  // Candidates t - 1, t - 2, ...: once an eos is passed, older positions read as eos.
  int64_t cand[kPleMaxNgram];
  cand[0] = tok;
  bool crossed = false;
  for (int s = 1; s < L.ngram; ++s) {
    int64_t c = crossed ? L.eos : prev[s - 1];
    crossed = crossed || c == L.eos;
    cand[s] = c;
  }
  for (int h = 0; h < L.heads; ++h) {
    const int n = h / L.heads_per_ngram + 2;
    uint64_t mixed = static_cast<uint64_t>(cand[0]) * static_cast<uint64_t>(L.mult[0]);
    for (int s = 1; s < n; ++s) mixed ^= static_cast<uint64_t>(cand[s]) * static_cast<uint64_t>(L.mult[s]);
    const int64_t m = static_cast<int64_t>(mixed);
    int64_t r = m % L.size[h];
    if (r < 0) r += L.size[h];
    ids[h] = r + L.offset[h];
  }
}

void PleContext::push(const PleLayout &L, int64_t tok) {
  for (int s = L.ngram - 2; s > 0; --s) prev[s] = prev[s - 1];
  prev[0] = tok;
}

std::array<uint16_t, 256> ple_fp8_lut(uint16_t scale_bf16) {
  std::array<uint16_t, 256> lut{};
  const float sc = bf16_value(scale_bf16);
  for (int c = 0; c < 256; ++c) lut[c] = bf16_bits(e4m3_value(static_cast<uint8_t>(c)) * sc);
  return lut;
}

void ple_ref_gate(int H, int S, const float *key, const float *value, const float *res, const float *nk, const float *nq, const float *ncw, float eps,
                  float *gated, float *normed) {
  std::vector<float> kn(H), qn(H);
  for (int s = 0; s < S; ++s) {
    const float *k = key + static_cast<size_t>(s) * H, *q = res + static_cast<size_t>(s) * H;
    double kk = 0, qq = 0;
    for (int i = 0; i < H; ++i) {
      kk += static_cast<double>(k[i]) * k[i];
      qq += static_cast<double>(q[i]) * q[i];
    }
    const double rk = 1.0 / std::sqrt(kk / H + eps), rq = 1.0 / std::sqrt(qq / H + eps);
    double dot = 0;
    for (int i = 0; i < H; ++i) {
      kn[i] = bfr(k[i] * rk * (1.0 + nk[s * H + i]));
      qn[i] = bfr(q[i] * rq * (1.0 + nq[s * H + i]));
      dot += bff(kn[i] * qn[i]);
    }
    const float d = bfr(static_cast<double>(bfr(dot)) / std::sqrt(static_cast<double>(H)));
    const float sign = d > 0 ? 1.f : (d < 0 ? -1.f : 0.f);
    const float mag = bfr(std::sqrt(std::max(static_cast<double>(std::fabs(d)), 1e-6)));
    const float g = bfr(1.0 / (1.0 + std::exp(-static_cast<double>(sign * mag))));
    double gg = 0;
    float *go = gated + static_cast<size_t>(s) * H, *no = normed + static_cast<size_t>(s) * H;
    for (int i = 0; i < H; ++i) {
      go[i] = bff(g * value[i]);
      gg += static_cast<double>(go[i]) * go[i];
    }
    const double rg = 1.0 / std::sqrt(gg / H + eps);
    for (int i = 0; i < H; ++i) no[i] = bfr(go[i] * rg * (1.0 + ncw[s * H + i]));
  }
}

void ple_ref_conv(int C, const float *const taps[kPleTaps], const float *w, const float *gated, float *res) {
  for (int c = 0; c < C; ++c) {
    double acc = 0;
    for (int k = 0; k < kPleTaps; ++k)
      if (taps[k]) acc += static_cast<double>(w[c * kPleTaps + k]) * taps[k][c];
    const double conv = bfr(acc);
    const float y = bfr(conv / (1.0 + std::exp(-conv)));
    res[c] = bff(res[c] + bff(gated[c] + y));
  }
}

}  // namespace xenithium::kernels
