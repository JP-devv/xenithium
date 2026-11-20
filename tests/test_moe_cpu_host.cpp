// Xenithium · gate moe-cpu-host: the decode MoE's CPU path for cold experts (kernels/moe/moe_cpu.hpp).
// Written for this engine. Host only, on random experts of qwen38's per-card TP half-blob layout (hidden 2560, inter 640,
// two cards): the three ISAs agree bit for bit (h and y, cards 0 / 1, 1 / 2 / 8 rows); the result is within 1e-4 (rel L2)
// of a double reference with the same rounding points; the pool (mailbox request -> done) gives exactly the direct calls'
// y for several experts and rows, run after run; the ISA detection names what the CPU has.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "xenithium/kernels/moe/moe_cpu.hpp"
#include "xenithium/kernels/rows.hpp"  // kStepRows32: the pool's rows per expert

namespace k = xenithium::kernels;

namespace {

int g_checks = 0, g_failures = 0;
#define CHECK(cond, ...)                                \
  do {                                                  \
    ++g_checks;                                         \
    if (!(cond)) {                                      \
      ++g_failures;                                     \
      std::printf("[FAIL] %s:%d: ", __FILE__, __LINE__); \
      std::printf(__VA_ARGS__);                         \
      std::printf("\n");                                \
    }                                                   \
  } while (0)

constexpr int H = 2560, I = 640, C = 2, Ic = I / C;

uint16_t f2bf(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  u += 0x7fffu + ((u >> 16) & 1u);
  return static_cast<uint16_t>(u >> 16);
}
float bf2f(uint16_t b) {
  uint32_t u = static_cast<uint32_t>(b) << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}
uint16_t f2h(float f) {  // positive normal range only (the scales)
  uint32_t u;
  std::memcpy(&u, &f, 4);
  const int e = static_cast<int>((u >> 23) & 0xff) - 127 + 15;
  return static_cast<uint16_t>((e << 10) | ((u >> 13) & 0x3ff));
}
float h2f(uint16_t h) { return std::ldexp(1.f + static_cast<float>(h & 0x3ff) / 1024.f, ((h >> 10) & 0x1f) - 15); }

k::MoeCpuShape shape_for(int card) {  // kernels:moe_tp_layout(2560, 640, 2)
  k::MoeCpuShape s;
  s.hidden = H, s.inter_c = Ic, s.card = card, s.cards = C, s.gu_ks = 8, s.dn_ks = 2;
  s.gu_codes = 0;
  s.dn_codes = static_cast<size_t>(H) * 2 * Ic / 2;
  s.gu_scales = s.dn_codes + static_cast<size_t>(Ic) * H / 2;
  s.dn_scales = s.gu_scales + static_cast<size_t>(H / 128) * 2 * Ic * 2;
  return s;
}
size_t blob_bytes(const k::MoeCpuShape &s) { return (s.dn_scales + 3ull * H * 2 + 1023) / 1024 * 1024; }

void fill_blob(uint8_t *b, const k::MoeCpuShape &s, std::mt19937 &rng) {
  std::uniform_int_distribution<int> byte(0, 255);
  std::uniform_real_distribution<float> sc(0.002f, 0.02f);
  for (size_t i = 0; i < s.gu_scales; ++i) b[i] = static_cast<uint8_t>(byte(rng));
  for (size_t i = s.gu_scales; i + 1 < blob_bytes(s); i += 2) {
    const uint16_t h = f2h(sc(rng));
    std::memcpy(b + i, &h, 2);
  }
}

// double reference with the same rounding points (g, u, h bf16; y from double)
std::vector<double> ref_expert(const k::MoeCpuShape &s, const uint8_t *b, const uint16_t *x) {
  const int tic = Ic / 16, G = H / 128, tiles = H / 16, NH = Ic / 64, hg0 = s.card * NH;
  auto code = [](const uint8_t *blk, int kk, int n) {
    const uint8_t v = blk[(kk / 2) * 16 + n];
    return static_cast<int>((kk & 1) ? (v >> 4) : (v & 15)) - 8;
  };
  const uint16_t *gsc = reinterpret_cast<const uint16_t *>(b + s.gu_scales);
  std::vector<float> h(Ic);
  for (int col = 0; col < Ic; ++col) {
    const int j = col / 16, n = col % 16;
    double gu[2] = {0, 0};
    for (int proj = 0; proj < 2; ++proj)
      for (int g = 0; g < G; ++g) {
        const uint8_t *blk = b + s.gu_codes + (static_cast<size_t>(g) * 2 * tic + proj * tic + j) * 1024;
        double sum = 0;
        for (int kk = 0; kk < 128; ++kk) sum += code(blk, kk, n) * static_cast<double>(bf2f(x[g * 128 + kk]));
        gu[proj] += sum * h2f(gsc[static_cast<size_t>(g) * 2 * Ic + proj * Ic + col]);
      }
    const float gb = bf2f(f2bf(static_cast<float>(gu[0]))), ub = bf2f(f2bf(static_cast<float>(gu[1])));
    h[col] = bf2f(f2bf(gb / (1.f + std::exp(-gb)) * ub));
  }
  const uint16_t *dsc = reinterpret_cast<const uint16_t *>(b + s.dn_scales);
  std::vector<double> y(H, 0.0);
  for (int col = 0; col < H; ++col) {
    const int t = col / 16, n = col % 16;
    for (int g = 0; g < NH; ++g) {
      const uint8_t *blk = b + s.dn_codes + (static_cast<size_t>(g) * tiles + t) * 512;
      double sum = 0;
      for (int kk = 0; kk < 64; ++kk) sum += code(blk, kk, n) * static_cast<double>(h[g * 64 + kk]);
      y[col] += sum * h2f(dsc[static_cast<size_t>((hg0 + g) / 2 - hg0 / 2) * H + col]);
    }
  }
  return y;
}

}  // namespace

int main() {
  std::mt19937 rng(170);
  std::normal_distribution<float> nd(0.f, 1.f);
  const k::MoeCpuShape s0 = shape_for(0);
  const size_t bb = blob_bytes(s0);
  constexpr int NB = 6;
  std::vector<uint8_t *> blobs(NB);
  for (auto &p : blobs) {
    p = static_cast<uint8_t *>(std::aligned_alloc(4096, bb));
    fill_blob(p, s0, rng);
  }
  std::vector<uint16_t> x(16 * H);  // 16-row requests
  for (auto &v : x) v = f2bf(nd(rng));

  const k::MoeCpuIsa best = k::moe_cpu_detect("auto");
  const bool has512 = k::moe_cpu_detect("avx512") == k::MoeCpuIsa::avx512bf16, has2 = k::moe_cpu_detect("avx2") == k::MoeCpuIsa::avx2;
  std::printf("CPU: %s (avx512-bf16 %s, avx2 %s)\n", k::moe_cpu_isa_name(best), has512 ? "yes" : "no", has2 ? "yes" : "no");
  CHECK(best == (has512 ? k::MoeCpuIsa::avx512bf16 : has2 ? k::MoeCpuIsa::avx2 : k::MoeCpuIsa::none), "auto picks the best ISA");

  // 1. the ISAs agree bit for bit; 2. the double reference
  for (int card = 0; card < 2; ++card) {
    const k::MoeCpuShape s = shape_for(card);
    for (const uint32_t mask : {0x1u, 0x5u, 0xffu, 0xff00u, 0xffffu}) {
      std::vector<uint16_t> h[3];
      std::vector<float> y[3];
      const k::MoeCpuIsa isas[3] = {k::MoeCpuIsa::none, k::MoeCpuIsa::avx2, k::MoeCpuIsa::avx512bf16};
      for (int i = 0; i < 3; ++i) {
        if ((i == 1 && !has2) || (i == 2 && !has512)) continue;
        h[i].assign(16 * Ic, 0);
        y[i].assign(16 * H, 0.f);
        k::moe_cpu_gate_up(isas[i], s, blobs[1], x.data(), H, mask, 0, Ic / 16, h[i].data());
        k::moe_cpu_down(isas[i], s, blobs[1], h[i].data(), mask, 0, H / 16, y[i].data(), H);
      }
      for (int i = 1; i < 3; ++i)
        if (!h[i].empty())
          CHECK(h[i] == h[0] && std::memcmp(y[i].data(), y[0].data(), y[0].size() * 4) == 0, "card %d mask %04x: %s == scalar bit for bit", card, mask,
                k::moe_cpu_isa_name(isas[i]));
      for (int r = 0; r < 16; ++r) {
        if (!((mask >> r) & 1u)) continue;
        const auto yr = ref_expert(s, blobs[1], x.data() + static_cast<size_t>(r) * H);
        double num = 0, den = 0;
        for (int c = 0; c < H; ++c) {
          const double d = y[0][static_cast<size_t>(r) * H + c] - yr[c];
          num += d * d, den += yr[c] * yr[c];
        }
        CHECK(std::sqrt(num / den) < 1e-4, "card %d row %d: rel L2 vs the double reference %.2e < 1e-4 (an h bf16 flip near a tie: ~1e-6)", card, r, std::sqrt(num / den));
      }
    }
  }

  // 3. the pool: a mailbox request gives the direct calls' y, every run
  if (best != k::MoeCpuIsa::none) {
    k::MoeCpuMail *mail = static_cast<k::MoeCpuMail *>(std::aligned_alloc(4096, 4096));
    std::memset(mail, 0, 4096);
    std::vector<float> ypool(static_cast<size_t>(k::kMoeCpuMax) * k::kStepRows32 * H, 0.f);
    k::MoeCpuOptions o;
    o.isa = best, o.threads = 4, o.first_core = -1;
    k::MoeCpuPool pool(s0, o, mail, x.data(), ypool.data());
    const uint32_t masks[4] = {0x1u, 0x3u, 0x8001u, 0xffffu};
    const uint8_t *bl[4] = {blobs[2], blobs[3], blobs[4], blobs[5]};
    std::vector<float> want(4 * static_cast<size_t>(k::kStepRows32) * H, 0.f);
    std::vector<uint16_t> hh(16 * Ic);
    for (int e = 0; e < 4; ++e) {
      k::moe_cpu_gate_up(best, s0, bl[e], x.data(), H, masks[e], 0, Ic / 16, hh.data());
      k::moe_cpu_down(best, s0, bl[e], hh.data(), masks[e], 0, H / 16, want.data() + static_cast<size_t>(e) * k::kStepRows32 * H, H);
    }
    for (int run = 0; run < 50; ++run) {
      std::fill(ypool.begin(), ypool.end(), 0.f);
      const int n = 1 + run % 4;
      pool.run_once(n, 16, masks, bl);
      bool same = mail->done == mail->req;
      for (int e = 0; e < n && same; ++e)
        for (int r = 0; r < 16; ++r)
          if ((masks[e] >> r) & 1u)
            same = same && std::memcmp(ypool.data() + (static_cast<size_t>(e) * k::kStepRows32 + r) * H, want.data() + (static_cast<size_t>(e) * k::kStepRows32 + r) * H, H * 4) == 0;
      CHECK(same, "pool run %d (%d experts): y == the direct calls' y, done == req", run, n);
    }
    CHECK(pool.served() == 50 && pool.experts() == 123, "pool counters: %llu requests, %llu experts", static_cast<unsigned long long>(pool.served()),
          static_cast<unsigned long long>(pool.experts()));
    std::free(mail);
  }
  for (auto p : blobs) std::free(p);
  std::printf("%s: %d checks, %d failures\n", g_failures ? "FAIL" : "PASS", g_checks, g_failures);
  return g_failures ? 1 : 0;
}
