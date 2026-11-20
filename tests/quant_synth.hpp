// Shared by test_quant_host and test_autoround: a small synthetic MoE layer for AutoRound gates.
#pragma once

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include "xenithium/quant/autoround.hpp"

namespace quant_test {
namespace q = xenithium::quant;

// A small synthetic layer: random weights with heavy-tailed rows, x with outlier channels.
struct Synth {
  q::ArShape s;
  std::vector<uint16_t> gu, dn, x, t;
  std::vector<int32_t> ids;
  std::vector<float> w;
  q::LayerData d;
  q::ExpertSource src;
};

inline Synth make_synth(int samples, int seqlen, uint64_t seed) {
  Synth S;
  S.s.hidden = 256, S.s.inter = 128, S.s.experts = 8, S.s.topk = 2;
  const int H = S.s.hidden, I = S.s.inter, E = S.s.experts, K = S.s.topk;
  std::mt19937_64 rng(seed);
  std::normal_distribution<float> nd(0.f, 1.f);
  S.gu.resize(static_cast<size_t>(E) * 2 * I * H);
  S.dn.resize(static_cast<size_t>(E) * H * I);
  for (auto &v : S.gu) v = q::ar_bf16_bits(0.05f * nd(rng) * (1.f + 0.5f * std::fabs(nd(rng))));
  for (auto &v : S.dn) v = q::ar_bf16_bits(0.05f * nd(rng) * (1.f + 0.5f * std::fabs(nd(rng))));
  const int64_t R = static_cast<int64_t>(samples) * seqlen;
  std::vector<float> ch(static_cast<size_t>(H));
  for (auto &c : ch) c = std::exp(0.6f * nd(rng));
  S.x.resize(static_cast<size_t>(R) * H);
  for (int64_t r = 0; r < R; ++r)
    for (int c = 0; c < H; ++c) S.x[static_cast<size_t>(r) * H + c] = q::ar_bf16_bits(nd(rng) * ch[static_cast<size_t>(c)]);
  S.ids.resize(static_cast<size_t>(R) * K);
  S.w.resize(S.ids.size());
  for (int64_t r = 0; r < R; ++r) {
    const int a = static_cast<int>(rng() % E);
    int b = static_cast<int>(rng() % (E - 1));
    if (b >= a) ++b;
    const float wa = 0.3f + 0.4f * static_cast<float>(rng() % 1000) / 1000.f;
    S.ids[static_cast<size_t>(r) * K] = a, S.ids[static_cast<size_t>(r) * K + 1] = b;
    S.w[static_cast<size_t>(r) * K] = wa, S.w[static_cast<size_t>(r) * K + 1] = 1.f - wa;
  }
  // target: the bf16 experts on the unquantised x
  S.t.resize(S.x.size());
  std::vector<float> h(static_cast<size_t>(I));
  for (int64_t r = 0; r < R; ++r) {
    std::vector<double> acc(static_cast<size_t>(H), 0.0);
    for (int k = 0; k < K; ++k) {
      const size_t e = static_cast<size_t>(S.ids[static_cast<size_t>(r) * K + k]);
      for (int j = 0; j < I; ++j) {
        double g = 0, u = 0;
        for (int c = 0; c < H; ++c) {
          const double xv = q::ar_bf16(S.x[static_cast<size_t>(r) * H + c]);
          g += xv * q::ar_bf16(S.gu[(e * 2 * I + j) * H + c]);
          u += xv * q::ar_bf16(S.gu[(e * 2 * I + I + j) * H + c]);
        }
        const float gf = q::ar_round_bf16(static_cast<float>(g)), uf = q::ar_round_bf16(static_cast<float>(u));
        h[static_cast<size_t>(j)] = q::ar_round_bf16(gf / (1.f + std::exp(-gf)) * uf);
      }
      for (int n = 0; n < H; ++n) {
        double y = 0;
        for (int j = 0; j < I; ++j) y += h[static_cast<size_t>(j)] * q::ar_bf16(S.dn[(e * H + n) * I + j]);
        acc[static_cast<size_t>(n)] += S.w[static_cast<size_t>(r) * K + k] * y;
      }
    }
    for (int n = 0; n < H; ++n) S.t[static_cast<size_t>(r) * H + n] = q::ar_bf16_bits(static_cast<float>(acc[static_cast<size_t>(n)]));
  }
  S.d.samples = samples, S.d.seqlen = seqlen;
  S.d.x = S.x.data(), S.d.ids = S.ids.data(), S.d.w = S.w.data(), S.d.t = S.t.data();
  S.src = {S.gu.data(), S.dn.data()};
  return S;
}

}  // namespace quant_test
