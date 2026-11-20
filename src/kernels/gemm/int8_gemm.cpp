#include "xenithium/kernels/gemm/int8_gemm.hpp"

#include <algorithm>

#include "xenithium/kernels/tile/dpas_int8.hpp"

namespace xenithium::kernels {
namespace {

namespace esimd = sycl::ext::intel::esimd;
constexpr int kRows = kGemmM4Rows;  // these GEMMs are 4-row only

template <int KS>
class GemmInt8M4;

// One work-group of KS work-items per 16-column tile; work-item `part` reduces
// chunks [part*chunks/KS, (part+1)*chunks/KS) and the partials meet in SLM. int32
// addition is associative, so every KS is bit-exact against the reference.
template <int KS>
sycl::event launch(sycl::queue &q, const int8_t *a, const int8_t *w_vnni, int32_t *c, int n, int k) {
  const int ntiles = n / dpas::kExecWidth;
  const int chunks = k / dpas::kKPerCall;
  const size_t chunk_stride = static_cast<size_t>(ntiles) * dpas::kBlockBytes;
  constexpr int kAcc = kRows * dpas::kExecWidth;

  return q.submit([&](sycl::handler &h) {
    h.parallel_for<GemmInt8M4<KS>>(
        sycl::nd_range<1>(static_cast<size_t>(ntiles) * KS, KS),
        [=](sycl::nd_item<1> it) [[intel::sycl_explicit_simd]] {
          if constexpr (KS > 1) esimd::slm_init<KS * kAcc * sizeof(int32_t)>();
          const int tile = static_cast<int>(it.get_group(0));
          const int part = static_cast<int>(it.get_local_id(0));
          const int c_begin = part * chunks / KS;
          const int c_end = (part + 1) * chunks / KS;

          esimd::simd<int32_t, kAcc> acc(0);
#pragma unroll 2
          for (int ci = c_begin; ci < c_end; ++ci) {
            // A: 4 rows x 32 int8, rows `k` bytes apart (tiny, served from cache).
            esimd::simd<int8_t, kRows * dpas::kKPerCall> a_tile;
#pragma unroll
            for (int row = 0; row < kRows; ++row) {
              a_tile.template select<dpas::kKPerCall, 1>(row * dpas::kKPerCall) =
                  esimd::block_load<int8_t, dpas::kKPerCall>(
                      a + static_cast<size_t>(row) * k + static_cast<size_t>(ci) * dpas::kKPerCall);
            }
            // B: one contiguous 512-byte VNNI block per (chunk, tile).
            const esimd::simd<int8_t, dpas::kBlockBytes> b_tile = esimd::block_load<int8_t, dpas::kBlockBytes>(
                w_vnni + static_cast<size_t>(ci) * chunk_stride + static_cast<size_t>(tile) * dpas::kBlockBytes);
            acc = dpas::accumulate<kRows>(acc, b_tile, a_tile);
          }

          int32_t *out_tile = c + static_cast<size_t>(tile) * dpas::kExecWidth;
          if constexpr (KS == 1) {
#pragma unroll
            for (int row = 0; row < kRows; ++row) {
              const esimd::simd<int32_t, dpas::kExecWidth> out =
                  acc.template select<dpas::kExecWidth, 1>(row * dpas::kExecWidth);
              out.copy_to(out_tile + static_cast<size_t>(row) * n);
            }
          } else {
            esimd::slm_block_store<int32_t, kAcc>(static_cast<uint32_t>(part * kAcc * sizeof(int32_t)), acc);
            esimd::barrier();
            // Rows are dealt round-robin: KS = 2 has fewer work-items than rows.
            for (int row = part; row < kRows; row += KS) {
              esimd::simd<int32_t, dpas::kExecWidth> sum(0);
#pragma unroll
              for (int p = 0; p < KS; ++p) {
                sum += esimd::slm_block_load<int32_t, dpas::kExecWidth>(
                    static_cast<uint32_t>((p * kAcc + row * dpas::kExecWidth) * sizeof(int32_t)));
              }
              sum.copy_to(out_tile + static_cast<size_t>(row) * n);
            }
          }
        });
  });
}

}  // namespace

int gemm_int8_auto_ksplit(int n, int k) {
  const int ntiles = n / dpas::kExecWidth;
  const int chunks = k / dpas::kKPerCall;
  int ks = 1;
  while (ks < kMaxKSplit && ntiles * ks < kMinWorkItems && chunks >= ks * 2) ks *= 2;
  return ks;
}

sycl::event gemm_int8_m4(sycl::queue &q, const int8_t *a, const int8_t *w_vnni, int32_t *c, int n, int k,
                         int ksplit) {
  if (ksplit <= 0) ksplit = gemm_int8_auto_ksplit(n, k);
  ksplit = std::min(ksplit, k / dpas::kKPerCall);
  if (ksplit >= 16) return launch<16>(q, a, w_vnni, c, n, k);
  if (ksplit >= 8) return launch<8>(q, a, w_vnni, c, n, k);
  if (ksplit >= 4) return launch<4>(q, a, w_vnni, c, n, k);
  if (ksplit >= 2) return launch<2>(q, a, w_vnni, c, n, k);
  return launch<1>(q, a, w_vnni, c, n, k);
}

}  // namespace xenithium::kernels
