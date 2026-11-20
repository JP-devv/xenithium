#include "xenithium/kernels/gemm/w4a16_gemm.hpp"
#include "xenithium/kernels/gemm/mlp.hpp"

#include <algorithm>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

#include "xenithium/kernels/tile/w4_tile.hpp"
#include "xenithium/rt/card.hpp"
#include "w4a16_c32.hpp"
#include "w4a8_rows.hpp"

namespace xenithium::kernels {
namespace {

namespace esimd = sycl::ext::intel::esimd;
namespace xmx = esimd::xmx;
using bf16 = sycl::ext::oneapi::bfloat16;
int g_rows8 = 1;  // w4a16_rows8_mode
int g_min_items = kW4MinWorkItems;  // w4a16_set_min_items


// One 128-k block's group terms into acc: ZP 0 = symmetric g128 (one fp16 scale), 1 = zero points (four 32-k groups),
// 2 = affine (two 64-k groups; `sc` holds bf16 scale bits, `zp` the bf16 biases).
template <int kRows, int ZP>
inline void block_into(esimd::simd<float, kRows * kW4Width> &acc, const uint8_t *w, const sycl::half *sc, const uint8_t *zp, const bf16 *a, int n,
                       int k, int ntiles, int tile, int g) SYCL_ESIMD_FUNCTION {
  const auto raw = w4::load_block(w + (static_cast<size_t>(g) * ntiles + tile) * kW4BlockBytes);
  auto a_rows = [&]() SYCL_ESIMD_FUNCTION {
    esimd::simd<bf16, kRows * kW4Group> a_group;
#pragma unroll
    for (int row = 0; row < kRows; ++row)
      a_group.template select<kW4Group, 1>(row * kW4Group) = esimd::block_load<bf16, kW4Group>(a + static_cast<size_t>(row) * k + g * kW4Group);
    return a_group;
  };
  if constexpr (ZP == 2) {
    const auto *sb = reinterpret_cast<const uint16_t *>(sc), *bb = reinterpret_cast<const uint16_t *>(zp);
    esimd::simd<uint32_t, 2 * kW4Width> su, bu;
#pragma unroll
    for (int s = 0; s < 2; ++s) {
      su.template select<kW4Width, 1>(s * kW4Width) = esimd::block_load<uint16_t, kW4Width>(sb + static_cast<size_t>(2 * g + s) * n + tile * kW4Width);
      bu.template select<kW4Width, 1>(s * kW4Width) = esimd::block_load<uint16_t, kW4Width>(bb + static_cast<size_t>(2 * g + s) * n + tile * kW4Width);
    }
    esimd::simd<uint32_t, 2 * kW4Width> sw = su << 16, bw = bu << 16;  // bf16 bits -> fp32 (exact)
    w4::block_dot_aff<kRows>(acc, raw, a_rows(), sw.template bit_cast_view<float>().read(), bw.template bit_cast_view<float>().read());
  } else if constexpr (ZP == 1) {
    esimd::simd<float, 4 * kW4Width> scs;
#pragma unroll
    for (int s = 0; s < 4; ++s)
      scs.template select<kW4Width, 1>(s * kW4Width) = esimd::block_load<sycl::half, kW4Width>(sc + static_cast<size_t>(4 * g + s) * n + tile * kW4Width);
    esimd::simd<uint8_t, 2 * kW4Width> zb;
    zb.template select<kW4Width, 1>(0) = esimd::block_load<uint8_t, kW4Width>(zp + static_cast<size_t>(2 * g) * n + tile * kW4Width);
    zb.template select<kW4Width, 1>(kW4Width) = esimd::block_load<uint8_t, kW4Width>(zp + static_cast<size_t>(2 * g + 1) * n + tile * kW4Width);
    w4::block_dot_zp<kRows>(acc, raw, a_rows(), zb, scs);
  } else {  // symmetric g128, in the order block, scale, A
    const esimd::simd<float, kW4Width> scale = esimd::block_load<sycl::half, kW4Width>(sc + static_cast<size_t>(g) * n + tile * kW4Width);
    w4::apply_scale<kRows>(acc, w4::group_dot<kRows>(raw, a_rows()), scale);
  }
}

template <int M, int KS, int ZP>
class W4A16M4;

template <int kRows, int KS, int ZP>
sycl::event launch(sycl::queue &q, const bf16 *a, const uint8_t *w, const sycl::half *sc, const uint8_t *zp, float *c, int n,
                   int k) {
  const int ntiles = n / kW4Width;
  const int groups = k / kW4Group;
  constexpr int kAcc = kRows * kW4Width;

  return q.submit([&](sycl::handler &h) {
    h.parallel_for<W4A16M4<kRows, KS, ZP>>(
        sycl::nd_range<1>(static_cast<size_t>(ntiles) * KS, KS),
        [=](sycl::nd_item<1> it) [[intel::sycl_explicit_simd]] {
          if constexpr (KS > 1) esimd::slm_init<KS * kAcc * sizeof(float)>();
          const int tile = static_cast<int>(it.get_group(0));
          const int part = static_cast<int>(it.get_local_id(0));
          const int g_begin = part * groups / KS;
          const int g_end = (part + 1) * groups / KS;

          esimd::simd<float, kAcc> acc(0.f);
          for (int g = g_begin; g < g_end; ++g) block_into<kRows, ZP>(acc, w, sc, zp, a, n, k, ntiles, tile, g);

          float *out_tile = c + static_cast<size_t>(tile) * kW4Width;
          if constexpr (KS == 1) {
#pragma unroll
            for (int row = 0; row < kRows; ++row) {
              const esimd::simd<float, kW4Width> out = acc.template select<kW4Width, 1>(row * kW4Width);
              out.copy_to(out_tile + static_cast<size_t>(row) * n);
            }
          } else {
            esimd::slm_block_store<float, kAcc>(static_cast<uint32_t>(part * kAcc * sizeof(float)), acc);
            esimd::barrier();
            // Rows dealt round-robin (KS = 2 has fewer work-items than rows). The
            // partials are summed in part order, so the result is deterministic.
            for (int row = part; row < kRows; row += KS) {
              esimd::simd<float, kW4Width> sum(0.f);
#pragma unroll
              for (int p = 0; p < KS; ++p) {
                sum += esimd::slm_block_load<float, kW4Width>(
                    static_cast<uint32_t>((p * kAcc + row * kW4Width) * sizeof(float)));
              }
              sum.copy_to(out_tile + static_cast<size_t>(row) * n);
            }
          }
        });
  });
}

// gate | up with swiglu in the epilogue. Work-group = 2 KS items: [0, KS) the k parts of
// gate tile t, [KS, 2 KS) those of up tile t (= tile t + n_half / 16 of the [gate | up] matrix). Each item's loop is
// launch's; the partials meet in SLM and are summed in part order exactly as launch sums them (KS 1: the item's own
// accumulator, no sum), then h = bf16(s(g) u) with g, u = bf16 of the sums and s(g) = g sigmoid(g) read from silu_tab
// (65,536 fp32 values of g by its bf16 bits, written by kernels:swiglu_table with swiglu's own expression). h is
// therefore swiglu's output bit for bit.
template <int M, int KS, int ZP>
class W4A16Glu;

template <int kRows, int KS, int ZP>
sycl::event launch_glu(sycl::queue &q, int live, const bf16 *a, const uint8_t *w, const sycl::half *sc, const uint8_t *zp, const float *tab, bf16 *h,
                       int n_half, int k) {
  const int n = 2 * n_half, ntiles = n / kW4Width, htiles = n_half / kW4Width;
  const int groups = k / kW4Group;
  constexpr int kAcc = kRows * kW4Width, kItems = 2 * KS;

  return q.submit([&](sycl::handler &hd) {
    hd.parallel_for<W4A16Glu<kRows, KS, ZP>>(
        sycl::nd_range<1>(static_cast<size_t>(htiles) * kItems, kItems), [=](sycl::nd_item<1> it) [[intel::sycl_explicit_simd]] {
          esimd::slm_init<kItems * kAcc * sizeof(float)>();
          const int ht = static_cast<int>(it.get_group(0));
          const int lid = static_cast<int>(it.get_local_id(0));
          const int half = lid / KS, part = lid % KS;
          const int tile = ht + half * htiles;
          const int g_begin = part * groups / KS;
          const int g_end = (part + 1) * groups / KS;

          esimd::simd<float, kAcc> acc(0.f);
          for (int g = g_begin; g < g_end; ++g) block_into<kRows, ZP>(acc, w, sc, zp, a, n, k, ntiles, tile, g);
          esimd::slm_block_store<float, kAcc>(static_cast<uint32_t>(lid * kAcc * sizeof(float)), acc);
          esimd::barrier();
          for (int row = lid; row < live; row += kItems) {  // live rows only (swiglu's rows)
            esimd::simd<float, kW4Width> gs, us;
            if constexpr (KS == 1) {
              gs = esimd::slm_block_load<float, kW4Width>(static_cast<uint32_t>(row * kW4Width * sizeof(float)));
              us = esimd::slm_block_load<float, kW4Width>(static_cast<uint32_t>((kAcc + row * kW4Width) * sizeof(float)));
            } else {
              gs = 0.f;
              us = 0.f;
#pragma unroll
              for (int p = 0; p < KS; ++p) {
                gs += esimd::slm_block_load<float, kW4Width>(static_cast<uint32_t>((p * kAcc + row * kW4Width) * sizeof(float)));
                us += esimd::slm_block_load<float, kW4Width>(static_cast<uint32_t>(((KS + p) * kAcc + row * kW4Width) * sizeof(float)));
              }
            }
            esimd::simd<bf16, kW4Width> gb = gs;
            const esimd::simd<float, kW4Width> uv = esimd::simd<bf16, kW4Width>(us);
            const esimd::simd<uint32_t, kW4Width> off = esimd::simd<uint32_t, kW4Width>(gb.template bit_cast_view<uint16_t>().read()) * 4u;
            const esimd::simd<float, kW4Width> sg = esimd::gather<float, kW4Width>(tab, off);
            const esimd::simd<bf16, kW4Width> hv = sg * uv;
            hv.copy_to(h + static_cast<size_t>(row) * n_half + static_cast<size_t>(ht) * kW4Width);
          }
        });
  });
}

// ---- 9-16 rows ----
// Row r's bits are its 8-row bits in both forms: every row's group sum is its own DPAS chain over the block's 8 calls (the
// same widened B, the same k order), then acc += sum x scale per group, the parts summed in part order (the 8-row k split).
// - C16: one item per (tile, part) widens each call once for two DPAS chains (rows 0-7, rows 8..8 + M2); A by one 2D
// block load per call straight into the DPAS A layout (row-major [rows][16 k]: no per-row copies). gate|up's k split
// stays the 8-row one (a row's bits).
// - H16: the halves form (a work-group = two halves of KS items, each the 8-row body on its rows): twice the 8-row
// items; used when 2 x the 8-row items fit the card's thread slots.
// The call loop of C16 is unrolled by 4 (fully unrolled it spills). 9-12 rows: C16's second chain on 4 rows; the halves
// keep 8-row halves.
template <int M2>
inline void c16_groups(esimd::simd<float, (8 + M2) * kW4Width> &acc, const bf16 *a, int k, int live, const uint8_t *w, const sycl::half *sc, int n,
                       int ntiles, int tile, int g0, int g1) SYCL_ESIMD_FUNCTION {
  constexpr int R = 8 + M2, kK = w4::kKPerCall;
  const auto *a16 = reinterpret_cast<const uint16_t *>(a);
  for (int g = g0; g < g1; ++g) {
    const auto raw = w4::load_block(w + (static_cast<size_t>(g) * ntiles + tile) * kW4BlockBytes);
    const esimd::simd<float, kW4Width> scale = esimd::block_load<sycl::half, kW4Width>(sc + static_cast<size_t>(g) * n + tile * kW4Width);
    esimd::simd<float, 8 * kW4Width> s0(0.f);
    esimd::simd<float, M2 * kW4Width> s1(0.f);
    auto call = [&](int j) SYCL_ESIMD_FUNCTION {
      const auto b = w4::call_codes(raw, j);
      // rows >= live read as 0 (the surface's height): their outputs are not stored
      esimd::simd<uint16_t, R * kK> u = esimd::load_2d<uint16_t, kK, R>(a16, k * 2 - 1, live - 1, k * 2 - 1, g * kW4Group + j * kK, 0);
      esimd::simd<bf16, R * kK> at = u.template bit_cast_view<bf16>().read();
      s0 = xmx::dpas<8, 8, float, float, bf16, bf16>(s0, b, at.template select<8 * kK, 1>(0).read());
      s1 = xmx::dpas<8, M2, float, float, bf16, bf16>(s1, b, at.template select<M2 * kK, 1>(8 * kK).read());
    };
#pragma unroll 4
    for (int j = 0; j < w4::kCalls; ++j) call(j);
#pragma unroll
    for (int row = 0; row < 8; ++row) acc.template select<kW4Width, 1>(row * kW4Width) += s0.template select<kW4Width, 1>(row * kW4Width) * scale;
#pragma unroll
    for (int row = 0; row < M2; ++row) acc.template select<kW4Width, 1>((8 + row) * kW4Width) += s1.template select<kW4Width, 1>(row * kW4Width) * scale;
  }
}

// The live rows of a tile's KS parts: KS 1 the item's own accumulator, else summed through SLM in part order.
template <int R, int KS>
inline void store_rows(esimd::simd<float, R * kW4Width> &acc, float *out_tile, int n, int live, int part, uint32_t slm0) SYCL_ESIMD_FUNCTION {
  constexpr int kAcc = R * kW4Width;
  if constexpr (KS == 1) {
#pragma unroll
    for (int row = 0; row < R; ++row)
      if (row < live) {
        const esimd::simd<float, kW4Width> o = acc.template select<kW4Width, 1>(row * kW4Width);
        o.copy_to(out_tile + static_cast<size_t>(row) * n);
      }
  } else {
    esimd::slm_block_store<float, kAcc>(slm0 + static_cast<uint32_t>(part * kAcc * sizeof(float)), acc);
    esimd::barrier();
    for (int row = part; row < live; row += KS) {
      esimd::simd<float, kW4Width> sum(0.f);
#pragma unroll
      for (int p = 0; p < KS; ++p) sum += esimd::slm_block_load<float, kW4Width>(slm0 + static_cast<uint32_t>((p * kAcc + row * kW4Width) * sizeof(float)));
      sum.copy_to(out_tile + static_cast<size_t>(row) * n);
    }
  }
}

template <int KS, int M2>
class W4A16C16;
template <int KS, int M2>
class W4A16H16;

// C16: one item per (tile, part), rows 0.. 8 + M2 (live of them stored)
template <int KS, int M2>
sycl::event launch_c16(sycl::queue &q, int live, const bf16 *a, const uint8_t *w, const sycl::half *sc, float *c, int n, int k) {
  const int ntiles = n / kW4Width, groups = k / kW4Group;
  constexpr int R = 8 + M2, kAcc = R * kW4Width;
  return q.submit([&](sycl::handler &h) {
    h.parallel_for<W4A16C16<KS, M2>>(sycl::nd_range<1>(static_cast<size_t>(ntiles) * KS, KS), [=](sycl::nd_item<1> it) [[intel::sycl_explicit_simd]] {
      if constexpr (KS > 1) esimd::slm_init<KS * kAcc * sizeof(float)>();
      const int tile = static_cast<int>(it.get_group(0)), part = static_cast<int>(it.get_local_id(0));
      esimd::simd<float, kAcc> acc(0.f);
      c16_groups<M2>(acc, a, k, live, w, sc, n, ntiles, tile, part * groups / KS, (part + 1) * groups / KS);
      store_rows<R, KS>(acc, c + static_cast<size_t>(tile) * kW4Width, n, live, part, 0);
    });
  });
}

// H16: a work-group = two halves of KS items (items [half][part]): half 0 the 8-row body on rows 0-7, half 1 the M2-row
// body on rows 8.. 8 + M2 (A rows up to row_shape(live) are read: the callers' buffers hold kMaxRows rows)
template <int KS, int M2>
sycl::event launch_h16(sycl::queue &q, int live, const bf16 *a, const uint8_t *w, const sycl::half *sc, float *c, int n, int k) {
  const int ntiles = n / kW4Width, groups = k / kW4Group;
  constexpr int kAcc8 = 8 * kW4Width;
  return q.submit([&](sycl::handler &h) {
    h.parallel_for<W4A16H16<KS, M2>>(sycl::nd_range<1>(static_cast<size_t>(ntiles) * KS * 2, KS * 2), [=](sycl::nd_item<1> it) [[intel::sycl_explicit_simd]] {
      if constexpr (KS > 1) esimd::slm_init<2 * KS * kAcc8 * sizeof(float)>();
      const int tile = static_cast<int>(it.get_group(0)), lid = static_cast<int>(it.get_local_id(0)), half = lid / KS, part = lid % KS;
      const int g_begin = part * groups / KS, g_end = (part + 1) * groups / KS;
      const uint32_t slm0 = static_cast<uint32_t>(half * KS * kAcc8 * sizeof(float));
      float *out_tile = c + static_cast<size_t>(tile) * kW4Width;
      if (half == 0) {
        esimd::simd<float, kAcc8> acc(0.f);
        for (int g = g_begin; g < g_end; ++g) block_into<8, 0>(acc, w, sc, nullptr, a, n, k, ntiles, tile, g);
        store_rows<8, KS>(acc, out_tile, n, 8, part, slm0);
      } else {
        esimd::simd<float, M2 * kW4Width> acc(0.f);
        const bf16 *a1 = a + static_cast<size_t>(8) * k;
        for (int g = g_begin; g < g_end; ++g) block_into<M2, 0>(acc, w, sc, nullptr, a1, n, k, ntiles, tile, g);
        store_rows<M2, KS>(acc, out_tile + static_cast<size_t>(8) * n, n, live - 8, part, slm0);
      }
    });
  });
}

// The gate | up GEMM with swiglu at 9-16 rows: launch_glu's work-group (items [gate / up][part]) on the C16 body.
template <int KS, int M2>
class W4A16Glu16;

template <int KS, int M2>
sycl::event launch_glu16(sycl::queue &q, int live, const bf16 *a, const uint8_t *w, const sycl::half *sc, const float *tab, bf16 *h, int n_half, int k) {
  const int n = 2 * n_half, ntiles = n / kW4Width, htiles = n_half / kW4Width;
  const int groups = k / kW4Group;
  constexpr int R = 8 + M2, kAcc = R * kW4Width, kItems = 2 * KS;
  return q.submit([&](sycl::handler &hd) {
    hd.parallel_for<W4A16Glu16<KS, M2>>(sycl::nd_range<1>(static_cast<size_t>(htiles) * kItems, kItems), [=](sycl::nd_item<1> it) [[intel::sycl_explicit_simd]] {
      esimd::slm_init<kItems * kAcc * sizeof(float)>();
      const int ht = static_cast<int>(it.get_group(0)), lid = static_cast<int>(it.get_local_id(0));
      const int half = lid / KS, part = lid % KS, tile = ht + half * htiles;
      esimd::simd<float, kAcc> acc(0.f);
      c16_groups<M2>(acc, a, k, live, w, sc, n, ntiles, tile, part * groups / KS, (part + 1) * groups / KS);
      esimd::slm_block_store<float, kAcc>(static_cast<uint32_t>(lid * kAcc * sizeof(float)), acc);
      esimd::barrier();
      for (int row = lid; row < live; row += kItems) {  // launch_glu's epilogue
        esimd::simd<float, kW4Width> gs, us;
        if constexpr (KS == 1) {
          gs = esimd::slm_block_load<float, kW4Width>(static_cast<uint32_t>(row * kW4Width * sizeof(float)));
          us = esimd::slm_block_load<float, kW4Width>(static_cast<uint32_t>((kAcc + row * kW4Width) * sizeof(float)));
        } else {
          gs = 0.f;
          us = 0.f;
#pragma unroll
          for (int p = 0; p < KS; ++p) {
            gs += esimd::slm_block_load<float, kW4Width>(static_cast<uint32_t>((p * kAcc + row * kW4Width) * sizeof(float)));
            us += esimd::slm_block_load<float, kW4Width>(static_cast<uint32_t>(((KS + p) * kAcc + row * kW4Width) * sizeof(float)));
          }
        }
        esimd::simd<bf16, kW4Width> gb = gs;
        const esimd::simd<float, kW4Width> uv = esimd::simd<bf16, kW4Width>(us);
        const esimd::simd<uint32_t, kW4Width> off = esimd::simd<uint32_t, kW4Width>(gb.template bit_cast_view<uint16_t>().read()) * 4u;
        const esimd::simd<float, kW4Width> sg = esimd::gather<float, kW4Width>(tab, off);
        const esimd::simd<bf16, kW4Width> hv = sg * uv;
        hv.copy_to(h + static_cast<size_t>(row) * n_half + static_cast<size_t>(ht) * kW4Width);
      }
    });
  });
}

// Thread slots of the card (XVEs x hardware threads at 128 GRF): the H16 / C16 choice. Read from the first queue's device
// (the cards of a fabric are the same model); the cores are rt:xe_cores (the emulated card's under XEN_EMU_CARD).
int thread_slots(sycl::queue &q) {
  static const int slots = [&] {
    const auto d = q.get_device();
    if (!d.has(sycl::aspect::ext_intel_gpu_hw_threads_per_eu)) return 2048;
    return rt::xe_cores(d) * rt::kXvePerCore * static_cast<int>(d.get_info<sycl::ext::intel::info::device::gpu_hw_threads_per_eu>());
  }();
  return slots;
}

}  // namespace

int w4a16_ksplit_for(int n, int k, int items) {
  const int ntiles = n / kW4Width;
  const int groups = k / kW4Group;
  int ks = 1;
  while (ks < kW4MaxKSplit && ntiles * ks < items && ks * 2 <= groups) ks *= 2;
  return ks;
}

int w4a16_auto_ksplit(int n, int k) { return w4a16_ksplit_for(n, k, g_min_items); }

template <int M, int ZP>
sycl::event gemm_mz(sycl::queue &q, const bf16 *a, const uint8_t *w_packed, const sycl::half *scales, const uint8_t *zp, float *c, int n, int k, int ksplit) {
  if (ksplit <= 0) ksplit = w4a16_auto_ksplit(n, k);
  ksplit = std::min(ksplit, k / kW4Group);
  if (ksplit >= 16) return launch<M, 16, ZP>(q, a, w_packed, scales, zp, c, n, k);
  if (ksplit >= 8) return launch<M, 8, ZP>(q, a, w_packed, scales, zp, c, n, k);
  if (ksplit >= 4) return launch<M, 4, ZP>(q, a, w_packed, scales, zp, c, n, k);
  if (ksplit >= 2) return launch<M, 2, ZP>(q, a, w_packed, scales, zp, c, n, k);
  return launch<M, 1, ZP>(q, a, w_packed, scales, zp, c, n, k);
}

template <int M>
sycl::event gemm_m(sycl::queue &q, const bf16 *a, const uint8_t *w_packed, const sycl::half *scales, float *c, int n, int k, int ksplit) {
  if (const uint16_t *b = w4_affine(w_packed)) return gemm_mz<M, 2>(q, a, w_packed, scales, reinterpret_cast<const uint8_t *>(b), c, n, k, ksplit);
  if (const uint8_t *zp = w4_zero_points(w_packed)) return gemm_mz<M, 1>(q, a, w_packed, scales, zp, c, n, k, ksplit);
  return gemm_mz<M, 0>(q, a, w_packed, scales, nullptr, c, n, k, ksplit);
}

sycl::event w4a16_gemm_m4(sycl::queue &q, const bf16 *a, const uint8_t *w_packed, const sycl::half *scales,
                          float *c, int n, int k, int ksplit) {
  return gemm_m<kGemmM4Rows>(q, a, w_packed, scales, c, n, k, ksplit);
}


namespace {
// 9-16 rows, symmetric g128 only (the archs with zero points / affine groups run 8-row steps: two 8-row launches here)
template <int M2>
sycl::event gemm16(sycl::queue &q, int live, const bf16 *a, const uint8_t *w, const sycl::half *sc, float *c, int n, int k, int ksplit) {
  if (ksplit <= 0) ksplit = w4a16_auto_ksplit(n, k);
  ksplit = std::min(ksplit, k / kW4Group);
  const int ks = ksplit >= 16 ? 16 : ksplit >= 8 ? 8 : ksplit >= 4 ? 4 : ksplit >= 2 ? 2 : 1;
  if (2 * (n / kW4Width) * ks <= thread_slots(q)) {  // the halves fit one wave (the second half is the 8-row body)
    switch (ks) {
      case 16: return launch_h16<16, 8>(q, live, a, w, sc, c, n, k);
      case 8: return launch_h16<8, 8>(q, live, a, w, sc, c, n, k);
      case 4: return launch_h16<4, 8>(q, live, a, w, sc, c, n, k);
      case 2: return launch_h16<2, 8>(q, live, a, w, sc, c, n, k);
      default: return launch_h16<1, 8>(q, live, a, w, sc, c, n, k);
    }
  }
  switch (ks) {
    case 16: return launch_c16<16, M2>(q, live, a, w, sc, c, n, k);
    case 8: return launch_c16<8, M2>(q, live, a, w, sc, c, n, k);
    case 4: return launch_c16<4, M2>(q, live, a, w, sc, c, n, k);
    case 2: return launch_c16<2, M2>(q, live, a, w, sc, c, n, k);
    default: return launch_c16<1, M2>(q, live, a, w, sc, c, n, k);
  }
}
template <int M2>
sycl::event glu16(sycl::queue &q, int live, const bf16 *a, const uint8_t *w, const sycl::half *s, const float *tab, bf16 *h, int n_half, int k, int ksplit) {
  if (ksplit <= 0) ksplit = w4a16_auto_ksplit(2 * n_half, k);  // the plain call's split of the whole [gate | up] matrix
  ksplit = std::min(ksplit, k / kW4Group);
  if (ksplit >= 16) return launch_glu16<16, M2>(q, live, a, w, s, tab, h, n_half, k);
  if (ksplit >= 8) return launch_glu16<8, M2>(q, live, a, w, s, tab, h, n_half, k);
  if (ksplit >= 4) return launch_glu16<4, M2>(q, live, a, w, s, tab, h, n_half, k);
  if (ksplit >= 2) return launch_glu16<2, M2>(q, live, a, w, s, tab, h, n_half, k);
  return launch_glu16<1, M2>(q, live, a, w, s, tab, h, n_half, k);
}
// 17-32 rows: C32 at the 8-row k split (SLM per work-group <= 16 parts x 2 KB, the swiglu form 2 x that);
// the instance of the live rows rounded up to 20 / 24 / 32 (c32:rows_of)
template <int R>
sycl::event gemm32_r(sycl::queue &q, int live, const bf16 *a, const uint8_t *w, const sycl::half *sc, float *c, int n, int k, int ksplit) {
  if (ksplit >= 16) return c32::launch<16, 4, R>(q, live, a, w, sc, c, n, k);
  if (ksplit >= 8) return c32::launch<8, 4, R>(q, live, a, w, sc, c, n, k);
  if (ksplit >= 4) return c32::launch<4, 4, R>(q, live, a, w, sc, c, n, k);
  if (ksplit >= 2) return c32::launch<2, 4, R>(q, live, a, w, sc, c, n, k);
  return c32::launch<1, 4, R>(q, live, a, w, sc, c, n, k);
}
sycl::event gemm32(sycl::queue &q, int live, const bf16 *a, const uint8_t *w, const sycl::half *sc, float *c, int n, int k, int ksplit) {
  if (ksplit <= 0) ksplit = w4a16_auto_ksplit(n, k);
  ksplit = std::min(ksplit, k / kW4Group);
  const int r = c32::rows_of(live);
  return r == 20 ? gemm32_r<20>(q, live, a, w, sc, c, n, k, ksplit) : r == 24 ? gemm32_r<24>(q, live, a, w, sc, c, n, k, ksplit)
                                                                     : gemm32_r<32>(q, live, a, w, sc, c, n, k, ksplit);
}
template <int R>
sycl::event glu32_r(sycl::queue &q, int live, const bf16 *a, const uint8_t *w, const sycl::half *s, const float *tab, bf16 *h, int n_half, int k, int ksplit) {
  if (ksplit >= 16) return c32::launch_glu<16, 4, R>(q, live, a, w, s, tab, h, n_half, k);  // 16 parts x 2 halves x 2 KB: 64 KB of SLM
  if (ksplit >= 8) return c32::launch_glu<8, 4, R>(q, live, a, w, s, tab, h, n_half, k);
  if (ksplit >= 4) return c32::launch_glu<4, 4, R>(q, live, a, w, s, tab, h, n_half, k);
  if (ksplit >= 2) return c32::launch_glu<2, 4, R>(q, live, a, w, s, tab, h, n_half, k);
  return c32::launch_glu<1, 4, R>(q, live, a, w, s, tab, h, n_half, k);
}
sycl::event glu32(sycl::queue &q, int live, const bf16 *a, const uint8_t *w, const sycl::half *s, const float *tab, bf16 *h, int n_half, int k, int ksplit) {
  if (ksplit <= 0) ksplit = w4a16_auto_ksplit(2 * n_half, k);  // the plain call's split of the whole [gate | up] matrix
  ksplit = std::min(ksplit, k / kW4Group);
  const int r = c32::rows_of(live);
  return r == 20 ? glu32_r<20>(q, live, a, w, s, tab, h, n_half, k, ksplit) : r == 24 ? glu32_r<24>(q, live, a, w, s, tab, h, n_half, k, ksplit)
                                                                            : glu32_r<32>(q, live, a, w, s, tab, h, n_half, k, ksplit);
}
// The 2D A loads want a 64-byte aligned base and a row pitch of 64+ bytes in 16-byte steps (k % 128 holds the pitch)
bool a2d_ok(const bf16 *a) { return reinterpret_cast<uintptr_t>(a) % 64 == 0; }
// A matrix registered for W4A8 decode (w4a8_rows_register) while the switch is on, plain
// symmetric g128 only
bool a8_rows(const uint8_t *w, int n, int k) {
  return w4a8_rows_of(w) && n % kW4Width == 0 && k % kW4Group == 0 && !w4_affine(w) && !w4_zero_points(w);
}
}  // namespace

// (w4_set_zero_points / w4_zero_points / w4_clear_zero_points live in w4a16_rows.cpp: the shared xl library holds the
// registry, so every binary that links it resolves them, whichever static objects it pulls in)

void w4a16_rows8_mode(int on) { g_rows8 = on; }
void w4a16_set_min_items(int items) { g_min_items = items > 0 ? items : kW4MinWorkItems; }
int w4a16_min_items() { return g_min_items; }

sycl::event w4a16_gemm_rows(sycl::queue &q, int rows, const bf16 *a, const uint8_t *w_packed, const sycl::half *scales, float *c, int n, int k,
                            int ksplit) {
  if (rows < 1 || rows > kStepRows32) throw std::invalid_argument("w4a16_gemm_rows: rows 1..32");
  if (a8_rows(w_packed, n, k)) {  // A's codes into the device's scratch, then the W4A8 body
    int8_t *codes = nullptr;
    float *as = nullptr;
    w4a8_rows_scratch(q, k, &codes, &as);
    if (ksplit <= 0) ksplit = w4a16_auto_ksplit(n, k);
    a8r::quant(q, rows, a, k, codes, as);
    return a8r::gemm(q, rows, reinterpret_cast<const uint8_t *>(codes), as, w_packed, scales, c, n, k, std::min(ksplit, k / kW4Group));
  }
  if (rows > kStepRows16) {  // 17-32 rows: C32, else two launches (rows 0-15, then the rest)
    const bool plain = !w4_affine(w_packed) && !w4_zero_points(w_packed);
    if (plain && a2d_ok(a)) return gemm32(q, rows, a, w_packed, scales, c, n, k, ksplit);
    w4a16_gemm_rows(q, kStepRows16, a, w_packed, scales, c, n, k, ksplit);
    return w4a16_gemm_rows(q, rows - kStepRows16, a + static_cast<size_t>(kStepRows16) * k, w_packed, scales, c + static_cast<size_t>(kStepRows16) * n, n, k,
                           ksplit);
  }
  if (rows > 2 * kGemmM4Rows) {  // 9-16 rows
    const bool plain = !w4_affine(w_packed) && !w4_zero_points(w_packed);
    if (plain && a2d_ok(a)) return rows <= 12 ? gemm16<4>(q, rows, a, w_packed, scales, c, n, k, ksplit) : gemm16<8>(q, rows, a, w_packed, scales, c, n, k, ksplit);
    // zero points / affine groups: two launches (rows 0-7, then the rest; a row's bits do not depend on its launch)
    w4a16_gemm_rows(q, 2 * kGemmM4Rows, a, w_packed, scales, c, n, k, ksplit);
    return w4a16_gemm_rows(q, rows - 2 * kGemmM4Rows, a + static_cast<size_t>(2 * kGemmM4Rows) * k, w_packed, scales, c + static_cast<size_t>(2 * kGemmM4Rows) * n, n, k,
                           ksplit);
  }
  // The DPAS repeat count follows the live rows (1, 2, 4, 8): a 1-row step loads one A row per group, not four.
  if (rows == 1) return gemm_m<1>(q, a, w_packed, scales, c, n, k, ksplit);
  if (rows == 2) return gemm_m<2>(q, a, w_packed, scales, c, n, k, ksplit);
  if (rows > kGemmM4Rows && g_rows8) return gemm_m<2 * kGemmM4Rows>(q, a, w_packed, scales, c, n, k, ksplit);
  sycl::event e = w4a16_gemm_m4(q, a, w_packed, scales, c, n, k, ksplit);
  if (rows > kGemmM4Rows) e = w4a16_gemm_m4(q, a + static_cast<size_t>(kGemmM4Rows) * k, w_packed, scales, c + static_cast<size_t>(kGemmM4Rows) * n, n, k, ksplit);
  return e;
}

namespace {
template <int M, int ZP>
sycl::event glu_mz(sycl::queue &q, int live, const bf16 *a, const uint8_t *w, const sycl::half *s, const uint8_t *zp, const float *tab, bf16 *h, int n_half,
                   int k, int ksplit) {
  if (ksplit <= 0) ksplit = w4a16_auto_ksplit(2 * n_half, k);  // the plain call's split of the whole [gate | up] matrix
  ksplit = std::min(ksplit, k / kW4Group);
  if (ksplit >= 16) return launch_glu<M, 16, ZP>(q, live, a, w, s, zp, tab, h, n_half, k);
  if (ksplit >= 8) return launch_glu<M, 8, ZP>(q, live, a, w, s, zp, tab, h, n_half, k);
  if (ksplit >= 4) return launch_glu<M, 4, ZP>(q, live, a, w, s, zp, tab, h, n_half, k);
  if (ksplit >= 2) return launch_glu<M, 2, ZP>(q, live, a, w, s, zp, tab, h, n_half, k);
  return launch_glu<M, 1, ZP>(q, live, a, w, s, zp, tab, h, n_half, k);
}

template <int M>
sycl::event glu_m(sycl::queue &q, int live, const bf16 *a, const uint8_t *w, const sycl::half *s, const float *tab, bf16 *h, int n_half, int k, int ksplit) {
  if (const uint16_t *b = w4_affine(w)) return glu_mz<M, 2>(q, live, a, w, s, reinterpret_cast<const uint8_t *>(b), tab, h, n_half, k, ksplit);
  if (const uint8_t *zp = w4_zero_points(w)) return glu_mz<M, 1>(q, live, a, w, s, zp, tab, h, n_half, k, ksplit);
  return glu_mz<M, 0>(q, live, a, w, s, nullptr, tab, h, n_half, k, ksplit);
}
}  // namespace

sycl::event w4a16_gemm_rows_swiglu(sycl::queue &q, int rows, const bf16 *a, const uint8_t *w_packed, const sycl::half *scales, const float *silu_tab,
                                   bf16 *h, int n_half, int k, int ksplit) {
  if (rows < 1 || rows > kStepRows32) throw std::invalid_argument("w4a16_gemm_rows_swiglu: rows 1..32");
  if (silu_tab && a8_rows(w_packed, 2 * n_half, k) && n_half % kW4Width == 0) {  // W4A8 decode
    int8_t *codes = nullptr;
    float *as = nullptr;
    w4a8_rows_scratch(q, k, &codes, &as);
    if (ksplit <= 0) ksplit = w4a16_auto_ksplit(2 * n_half, k);  // the plain call's split of the whole [gate | up] matrix
    a8r::quant(q, rows, a, k, codes, as);
    return a8r::glu(q, rows, reinterpret_cast<const uint8_t *>(codes), as, w_packed, scales, silu_tab, h, n_half, k, std::min(ksplit, k / kW4Group));
  }
  if (rows > kStepRows16) {  // 17-32 rows: C32, else two launches (as w4a16_gemm_rows)
    const bool plain = !w4_affine(w_packed) && !w4_zero_points(w_packed);
    if (plain && a2d_ok(a)) {
      if (n_half % kW4Width || k % kW4Group || !silu_tab) throw std::invalid_argument("w4a16_gemm_rows_swiglu: n_half % 16, k % 128, a silu table");
      return glu32(q, rows, a, w_packed, scales, silu_tab, h, n_half, k, ksplit);
    }
    w4a16_gemm_rows_swiglu(q, kStepRows16, a, w_packed, scales, silu_tab, h, n_half, k, ksplit);
    return w4a16_gemm_rows_swiglu(q, rows - kStepRows16, a + static_cast<size_t>(kStepRows16) * k, w_packed, scales, silu_tab,
                                  h + static_cast<size_t>(kStepRows16) * n_half, n_half, k, ksplit);
  }
  if (rows > 2 * kGemmM4Rows) {  // 9-16 rows
    const bool plain = !w4_affine(w_packed) && !w4_zero_points(w_packed);
    if (plain && a2d_ok(a)) {
      if (n_half % kW4Width || k % kW4Group || !silu_tab) throw std::invalid_argument("w4a16_gemm_rows_swiglu: n_half % 16, k % 128, a silu table");
      return rows <= 12 ? glu16<4>(q, rows, a, w_packed, scales, silu_tab, h, n_half, k, ksplit) : glu16<8>(q, rows, a, w_packed, scales, silu_tab, h, n_half, k, ksplit);
    }
    w4a16_gemm_rows_swiglu(q, 2 * kGemmM4Rows, a, w_packed, scales, silu_tab, h, n_half, k, ksplit);  // two launches (as w4a16_gemm_rows)
    return w4a16_gemm_rows_swiglu(q, rows - 2 * kGemmM4Rows, a + static_cast<size_t>(2 * kGemmM4Rows) * k, w_packed, scales, silu_tab,
                                  h + static_cast<size_t>(2 * kGemmM4Rows) * n_half, n_half, k, ksplit);
  }
  if (n_half % kW4Width || k % kW4Group || !silu_tab) throw std::invalid_argument("w4a16_gemm_rows_swiglu: n_half % 16, k % 128, a silu table");
  if (rows == 1) return glu_m<1>(q, rows, a, w_packed, scales, silu_tab, h, n_half, k, ksplit);
  if (rows == 2) return glu_m<2>(q, rows, a, w_packed, scales, silu_tab, h, n_half, k, ksplit);
  if (rows <= kGemmM4Rows) return glu_m<kGemmM4Rows>(q, rows, a, w_packed, scales, silu_tab, h, n_half, k, ksplit);
  return glu_m<2 * kGemmM4Rows>(q, rows, a, w_packed, scales, silu_tab, h, n_half, k, ksplit);
}

namespace {
// the many-row library's short calls run these (w4a16_pf_set_short_rows)
[[maybe_unused]] const bool g_short_registered = [] {  // (runs at load: the registration is its effect)
  w4a16_pf_short_hooks(&w4a16_gemm_rows, &w4a16_gemm_rows_swiglu, &swiglu_table);
  return true;
}();
}  // namespace

}  // namespace xenithium:kernels
