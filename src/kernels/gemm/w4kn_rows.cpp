// Xenithium · W4A16 decode GEMMs on oneDNN's weight layout (see w4kn_gemm.hpp). The nib-block kernels
// (w4a16_gemm.cpp) stay as they are; this file is the kn layout's decode path.
// Per (128-k group, 16-column tile): two transposed 2D block loads of 8 dwords x 16 rows read the tile's 16 rows x 64
// bytes as tw[d][column] (dword d = bytes 4d..4d+3 of the column's row). The nib order the bf16 B operand needs (byte
// p x 16 + column) is then a stride-4 byte region per p (p / 4 x 64 + p % 4): no byte transposition beyond the region
// reads. ^ 0x88 back to the unsigned codes, then w4_tile's widen (q - 8 exact in bf16) and dpas<8, M>(acc, B, A).
#include "xenithium/kernels/gemm/w4kn_gemm.hpp"

#include <algorithm>
#include <stdexcept>

#include <sycl/ext/intel/esimd.hpp>

namespace xenithium::kernels {
namespace {

namespace esimd = sycl::ext::intel::esimd;
namespace xmx = esimd::xmx;
using bf16 = sycl::ext::oneapi::bfloat16;

constexpr int kW = 16, kG = 128, kCalls = 8, kKC = 16;

template <int M, int KS>
class W4KnRows;

template <int M, int KS>
sycl::event launch(sycl::queue &q, const bf16 *a, const uint8_t *w, const sycl::half *sc, float *c, int n, int k) {
  const int ntiles = n / kW, groups = k / kG;
  constexpr int kAcc = M * kW;
  const auto *w32 = reinterpret_cast<const uint32_t *>(w);
  return q.submit([&](sycl::handler &h) {
    h.parallel_for<W4KnRows<M, KS>>(sycl::nd_range<1>(static_cast<size_t>(ntiles) * KS, KS), [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
      if constexpr (KS > 1) esimd::slm_init<KS * kAcc * sizeof(float)>();
      const int tile = static_cast<int>(it.get_group(0)), part = static_cast<int>(it.get_local_id(0));
      const int g0 = part * groups / KS, g1 = (part + 1) * groups / KS;
      const unsigned pitch = static_cast<unsigned>(k / 2 - 1), height = static_cast<unsigned>(n - 1);
      esimd::simd<float, kAcc> acc(0.f);
      for (int g = g0; g < g1; ++g) {
        esimd::simd<uint32_t, 256> tw;
        tw.template select<128, 1>(0) = esimd::load_2d<uint32_t, 8, 16, 1, true, false>(w32, pitch, height, pitch, g * 16, tile * kW);
        tw.template select<128, 1>(128) = esimd::load_2d<uint32_t, 8, 16, 1, true, false>(w32, pitch, height, pitch, g * 16 + 8, tile * kW);
        const esimd::simd<float, kW> scale = esimd::block_load<sycl::half, kW>(sc + static_cast<size_t>(g) * n + tile * kW);
        esimd::simd<bf16, M * kG> ag;
#pragma unroll
        for (int r = 0; r < M; ++r) ag.template select<kG, 1>(r * kG) = esimd::block_load<bf16, kG>(a + static_cast<size_t>(r) * k + g * kG);
        esimd::simd<uint8_t, 1024> tb = tw.template bit_cast_view<uint8_t>();
        esimd::simd<float, kAcc> gs(0.f);
#pragma unroll
        for (int j = 0; j < kCalls; ++j) {
          esimd::simd<uint8_t, 128> bytes;
#pragma unroll
          for (int pp = 0; pp < 8; ++pp) {
            const int p = 8 * j + pp;
            bytes.template select<16, 1>(pp * 16) = tb.template select<16, 4>((p / 4) * 64 + p % 4);
          }
          bytes ^= 0x88;
          const esimd::simd<int16_t, 128> lo = esimd::simd<int16_t, 128>(bytes & 15) - 8, hi = esimd::simd<int16_t, 128>(bytes >> 4) - 8;
          esimd::simd<bf16, 256> b;
          b.template select<128, 2>(0) = esimd::simd<float, 128>(lo);
          b.template select<128, 2>(1) = esimd::simd<float, 128>(hi);
          esimd::simd<bf16, M * kKC> at;
#pragma unroll
          for (int r = 0; r < M; ++r) at.template select<kKC, 1>(r * kKC) = ag.template select<kKC, 1>(r * kG + j * kKC);
          gs = xmx::dpas<8, M, float, float, bf16, bf16>(gs, b, at);
        }
#pragma unroll
        for (int r = 0; r < M; ++r) acc.template select<kW, 1>(r * kW) += gs.template select<kW, 1>(r * kW) * scale;
      }
      float *out = c + static_cast<size_t>(tile) * kW;
      if constexpr (KS == 1) {
#pragma unroll
        for (int r = 0; r < M; ++r) {
          const esimd::simd<float, kW> o = acc.template select<kW, 1>(r * kW);
          o.copy_to(out + static_cast<size_t>(r) * n);
        }
      } else {
        esimd::slm_block_store<float, kAcc>(static_cast<uint32_t>(part * kAcc * sizeof(float)), acc);
        esimd::barrier();
        for (int r = part; r < M; r += KS) {  // parts summed in order: deterministic
          esimd::simd<float, kW> s(0.f);
#pragma unroll
          for (int p = 0; p < KS; ++p) s += esimd::slm_block_load<float, kW>(static_cast<uint32_t>((p * kAcc + r * kW) * sizeof(float)));
          s.copy_to(out + static_cast<size_t>(r) * n);
        }
      }
    });
  });
}

template <int M>
sycl::event by_split(sycl::queue &q, int ks, const bf16 *a, const uint8_t *w, const sycl::half *sc, float *c, int n, int k) {
  if (ks >= 16) return launch<M, 16>(q, a, w, sc, c, n, k);
  if (ks >= 8) return launch<M, 8>(q, a, w, sc, c, n, k);
  if (ks >= 4) return launch<M, 4>(q, a, w, sc, c, n, k);
  if (ks >= 2) return launch<M, 2>(q, a, w, sc, c, n, k);
  return launch<M, 1>(q, a, w, sc, c, n, k);
}

class W4KnRepack;

}  // namespace

sycl::event w4kn_gemm_rows(sycl::queue &q, int rows, const bf16 *a, const uint8_t *w_kn, const sycl::half *scales, float *c, int n, int k, int ksplit) {
  if (rows < 1 || rows > 8 || n % kW || k % kG) throw std::invalid_argument("w4kn_gemm_rows: rows 1..8, n % 16, k % 128");
  if (ksplit <= 0) ksplit = n >= 8192 ? 16 : (rows == 1 ? 2 : 4);  // the default split per shape class
  ksplit = std::min(ksplit, k / kG);
  switch (rows) {  // the DPAS repeat count = the live rows (as w4a16_gemm_rows' 1 / 2 / 4 / 8, here every count)
    case 1: return by_split<1>(q, ksplit, a, w_kn, scales, c, n, k);
    case 2: return by_split<2>(q, ksplit, a, w_kn, scales, c, n, k);
    case 3: return by_split<3>(q, ksplit, a, w_kn, scales, c, n, k);
    case 4: return by_split<4>(q, ksplit, a, w_kn, scales, c, n, k);
    case 5: return by_split<5>(q, ksplit, a, w_kn, scales, c, n, k);
    case 6: return by_split<6>(q, ksplit, a, w_kn, scales, c, n, k);
    case 7: return by_split<7>(q, ksplit, a, w_kn, scales, c, n, k);
    default: return by_split<8>(q, ksplit, a, w_kn, scales, c, n, k);
  }
}

sycl::event w4kn_repack(sycl::queue &q, const uint8_t *w, int n, int k, uint8_t *dst) {
  if (n % kW || k % kG) throw std::invalid_argument("w4kn_repack: n % 16, k % 128");
  const int tiles = n / kW;
  const size_t blocks = static_cast<size_t>(k / kG) * tiles;
  return q.submit([&](sycl::handler &h) {
    sycl::local_accessor<uint32_t, 1> slm(256, h);
    h.parallel_for<W4KnRepack>(sycl::nd_range<1>(blocks * 16, 16), [=](sycl::nd_item<1> it) {
      const size_t blk = it.get_group(0);
      const int i = static_cast<int>(it.get_local_id(0));
      const auto *src = reinterpret_cast<const sycl::vec<uint32_t, 4> *>(w + blk * 1024 + i * 64);
#pragma unroll
      for (int v = 0; v < 4; ++v) {
        const sycl::vec<uint32_t, 4> x = src[v];
#pragma unroll
        for (int e = 0; e < 4; ++e) slm[i * 16 + v * 4 + e] = x[e] ^ 0x88888888u;
      }
      sycl::group_barrier(it.get_group());
      const auto *sb = reinterpret_cast<const uint8_t *>(&slm[0]);
      const int g = static_cast<int>(blk / tiles), t = static_cast<int>(blk % tiles);
      auto *d = reinterpret_cast<sycl::vec<uint32_t, 4> *>(dst + (static_cast<size_t>(t) * 16 + i) * (k / 2) + static_cast<size_t>(g) * 64);
#pragma unroll
      for (int v = 0; v < 4; ++v) {
        sycl::vec<uint32_t, 4> o;
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const int p0 = v * 16 + e * 4;
          o[e] = static_cast<uint32_t>(sb[(p0 + 0) * 16 + i]) | static_cast<uint32_t>(sb[(p0 + 1) * 16 + i]) << 8 | static_cast<uint32_t>(sb[(p0 + 2) * 16 + i]) << 16 |
                 static_cast<uint32_t>(sb[(p0 + 3) * 16 + i]) << 24;
        }
        d[v] = o;
      }
    });
  });
}

}  // namespace xenithium:kernels
