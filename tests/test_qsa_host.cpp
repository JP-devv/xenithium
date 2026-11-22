// Host-only gate for QSA (precise floating point, like the code under test): YaRN correction
// bounds, the rotary table, the int8 KV codes.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "xenithium/kernels/pack.hpp"
#include "xenithium/kernels/attention/qsa_ref.hpp"
#include "xenithium/kernels/attention/trained_qsa_pages.hpp"

namespace {
namespace k = xenithium::kernels;
int g_failures = 0;
void expect(bool ok, const std::string &what) {
  std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
  if (!ok) ++g_failures;
}
float bfr(float v) { return k::bf16_value(k::bf16_bits(v)); }

void test_host() {
  {
    bool ok = true;
    const int dims[3] = {64, 128, 256}, lo[3] = {14, 28, 56}, hi[3] = {22, 43, 85};
    for (int i = 0; i < 3; ++i) {
      const auto b = k::qsa_yarn_bounds(dims[i], 1e7, 262144, 32, 1);
      ok &= b.low == lo[i] && b.high == hi[i];
    }
    expect(ok, "YaRN correction bounds at factor 2, theta 1e7, 262,144 trained: 14/22, 28/43, 56/85 (rotary 64 / 128 / 256)");
  }
  {
    const auto t = k::qsa_rope_table(300000, 64, 1e7, k::QsaYarn{});
    bool ok = true;
    for (int64_t p : {0LL, 1LL, 777LL, 262143LL, 299999LL})
      for (int i = 0; i < 32; ++i) {
        const float inv = 1.0f / std::pow(1e7f, static_cast<float>(2 * i) / 64.f);
        const float ang = static_cast<float>(p) * inv;
        ok &= t[p * 64 + i] == k::bf16_bits(static_cast<float>(std::cos(static_cast<double>(ang))));
        ok &= t[p * 64 + 32 + i] == k::bf16_bits(static_cast<float>(std::sin(static_cast<double>(ang))));
      }
    k::QsaYarn y2;
    y2.factor = 2;
    const auto t2 = k::qsa_rope_table(4, 64, 1e7, y2);
    const auto b = k::qsa_yarn_bounds(64, 1e7, 262144, 32, 1);
    const float mscale = static_cast<float>(1 + 0.1 * std::log(2.0));
    ok &= t2[0] == k::bf16_bits(mscale);  // cos(0) = attention factor
    for (int i = 0; i < 32; ++i) {
      const float pf = std::pow(1e7f, static_cast<float>(2 * i) / 64.f);
      const float want = i < b.low ? 1.0f / pf : (i > b.high ? 1.0f / (2.f * pf) : -1.f);
      if (want < 0) continue;
      ok &= t2[64 + i] == k::bf16_bits(static_cast<float>(std::cos(static_cast<double>(1.f * want))) * mscale);
    }
    expect(ok, "rotary table: factor 1 == the plain fp32-angle formula; YaRN 2: cos(0) = 1 + 0.1 ln 2, pairs below / above the ramp keep / halve");
  }
  {
    // The load's automatic factor for a max context (qsa_yarn_for): 1 up to the trained positions, past them the
    // smallest factor in steps of 0.01 reaching it; whole factors exact, so their tables equal the integer rule's.
    struct Case {
      int64_t ctx, trained;
      double factor;
    };
    const Case cs[] = {{64, 262144, 1.0},        {262144, 262144, 1.0},    {262208, 262144, 1.01},  {333000, 262144, 1.28},
                       {524288, 262144, 2.0},    {524352, 262144, 2.01},   {1048576, 262144, 4.0},  {1048576, 524288, 2.0},
                       {600000, 524288, 1.15}};
    bool ok = true;
    for (const Case &c : cs) {
      const k::QsaYarn y = k::qsa_yarn_for(c.ctx, c.trained);
      const bool hit = y.factor == c.factor && y.orig == c.trained && y.attn_factor == 0;
      const bool reaches = static_cast<double>(c.trained) * y.factor >= static_cast<double>(c.ctx);
      const bool smallest = y.factor == 1.0 || static_cast<double>(c.trained) * (y.factor - 0.01) < static_cast<double>(c.ctx);
      if (!(hit && reaches && smallest))
        std::printf("  qsa_yarn_for(%lld, %lld) = %.6f (want %.2f)\n", static_cast<long long>(c.ctx), static_cast<long long>(c.trained), y.factor, c.factor);
      ok &= hit && reaches && smallest;
    }
    k::QsaYarn whole;  // the integer rule: factor ceil(context / trained), orig 262,144
    whole.factor = 4;
    ok &= k::qsa_rope_table(4096, 64, 1e7, k::qsa_yarn_for(1048576, 262144)) == k::qsa_rope_table(4096, 64, 1e7, whole);
    ok &= k::qsa_rope_table(4096, 64, 1e7, k::qsa_yarn_for(262144, 262144)) == k::qsa_rope_table(4096, 64, 1e7, k::QsaYarn{});
    expect(ok, "YaRN auto factor: 1 to the trained positions, then the smallest 0.01 step reaching the context (333,000 -> 1.28); "
               "whole factors give the integer rule's table, factor 1 the plain table");
  }
  {
    // int8 KV codes: the amax element at +-127, every code the nearest multiple of the scale, an
    // all-zero row = 0 codes / 0 scale.
    std::mt19937 rng(9);
    std::normal_distribution<float> nd(0.f, 3.f);
    bool ok = true;
    double e8 = 0, ss = 0;
    for (int rep = 0; rep < 200 && ok; ++rep) {
      std::vector<float> x(256);
      const float sc = std::ldexp(1.f, rep % 40 - 20);
      for (auto &v : x) v = bfr(nd(rng) * sc);
      std::vector<int8_t> c(256);
      float s;
      k::qsa_ref_i8(x.data(), 256, c.data(), &s);
      int top = 0;
      for (int i = 0; i < 256; ++i) {
        const double q = static_cast<double>(c[i]) * s;
        // nearest up to the rounding of 127 / amax vs amax / 127 (a tie can go either way)
        ok &= c[i] >= -127 && c[i] <= 127 && std::fabs(q - x[i]) <= 0.5 * s * (1 + 1e-4);
        top = std::max(top, std::abs(static_cast<int>(c[i])));
        e8 += (q - x[i]) * (q - x[i]);
        ss += static_cast<double>(x[i]) * x[i];
      }
      ok &= top == 127;
    }
    std::vector<float> z(256, 0.f);
    std::vector<int8_t> cz(256, 1);
    float sz = 1.f;
    k::qsa_ref_i8(z.data(), 256, cz.data(), &sz);
    ok &= sz == 0.f && std::all_of(cz.begin(), cz.end(), [](int8_t v) { return v == 0; });
    char what[200];
    std::snprintf(what, sizeof what, "int8 KV codes: amax at +-127, every code the nearest, zero row = 0 (relative RMS error %.2e)",
                  std::sqrt(e8 / ss));
    expect(ok, what);
  }
}

void test_trained_pages() {
  std::vector<int32_t> ids(k::kTrainedQsaTopBlocks);
  std::iota(ids.begin(), ids.end(), 0);
  int32_t entries[k::kTrainedQsaPages]{};
  expect(k::trained_qsa_pack_row(2047, ids.data(), 512, entries) == 0 &&
         k::trained_qsa_pack_row(2048, ids.data(), 512, entries) == 0,
         "trained page pack: all-visible prefix and first partial block stay dense");

  // 512 selected blocks end on page 31; the current block on page 33 is
  // mandatory even though the intermediate page wasn't selected.
  const int pos = 33 * k::kQsaPage + 4;
  const int n = k::trained_qsa_pack_row(pos, ids.data(), 512, entries);
  const uint32_t last = k::trained_qsa_mask(entries[32]);
  expect(n == 33 && k::trained_qsa_page(entries[31]) == 31 && k::trained_qsa_mask(entries[31]) == 0xffffu &&
         k::trained_qsa_page(entries[32]) == 33 && last == 2 &&
         !k::trained_qsa_keeps(last, 3) && k::trained_qsa_keeps(last, 4) && !k::trained_qsa_keeps(last, 8),
         "trained page pack: ordered logical pages include the current block without adding skipped blocks");

  int32_t physical[34]{};
  physical[33] = 7;
  expect(k::trained_qsa_page_base(physical, k::trained_qsa_page(entries[32]), 3, 1) == (7 * 3 + 1) * k::kQsaPage,
         "trained page walk uses the slot's nonidentity physical page table");
  ids.back() = (pos + 1) / k::kQsaCompress;
  const bool rejects_future = k::trained_qsa_pack_row(pos, ids.data(), 512, entries) == -1;
  ids.back() = ids[510];
  expect(rejects_future && k::trained_qsa_pack_row(pos, ids.data(), 512, entries) == -1,
         "trained page pack rejects future and duplicate blocks");
}

}  // namespace

int main() {
  test_host();
  test_trained_pages();
  if (g_failures) {
    std::printf("FAILED: %d failure(s)\n", g_failures);
    return 1;
  }
  std::printf("OK: 0 failure(s)\n");
  return 0;
}
