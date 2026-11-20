#include "xenithium/kernels/head/logprob.hpp"

#include <cstring>
#include <limits>
#include <stdexcept>

namespace xenithium::kernels {
namespace {

constexpr int kLpItems = 1024;

template <class T, bool XT_DEV>
class LogprobStats;
class LogprobTargets;

// Order-preserving 16-bit image of a bf16 (bits b), as the pick's: -0 -> +0, NaN -> lowest.
inline uint32_t lp_key(float v) {
  const sycl::ext::oneapi::bfloat16 h(v);
  uint16_t b;
  std::memcpy(&b, &h, 2);
  if ((b & 0x7fff) > 0x7f80) return 0;
  if (b == 0x8000) b = 0;
  return (b & 0x8000) ? static_cast<uint32_t>(~b & 0xffff) : static_cast<uint32_t>(b | 0x8000);
}

// T = float (fp32 logits, rounded to bf16 here) or bf16 (already rounded: the same values). XT_DEV:
// the targets' logits from a device array (any row count) instead of LpArgs (<= 4 rows).
template <class T, bool XT_DEV>
sycl::event stats(sycl::queue &q, const T *logits, int ld, int rows, int n, int col_base, int top, const LpArgs &a, const float *xt_dev, float *rec) {
  const LpArgs args = a;
  const int K = top < n ? top : n;
  return q.submit([&](sycl::handler &h) {
    sycl::local_accessor<int, 1> hist(256, h);
    sycl::local_accessor<int, 1> misc(4, h);
    sycl::local_accessor<float, 1> imax(kLpItems, h);
    h.parallel_for<LogprobStats<T, XT_DEV>>(sycl::nd_range<1>(static_cast<size_t>(rows) * kLpItems, kLpItems), [=](sycl::nd_item<1> it) {
      const int r = static_cast<int>(it.get_group(0)), lid = static_cast<int>(it.get_local_id(0));
      const auto g = it.get_group();
      const T *x = logits + static_cast<size_t>(r) * ld;
      float *out = rec + static_cast<size_t>(r) * kLpRec;
      // contiguous ranges per item: ties at the threshold are then taken in index order by a scan
      const int per = (n + kLpItems - 1) / kLpItems, beg = sycl::min(n, lid * per), end = sycl::min(n, beg + per);
      float xt;
      if constexpr (XT_DEV) xt = xt_dev[r];
      else xt = args.xt[r];
      if (lid < 256) hist[lid] = 0;
      sycl::group_barrier(g);
      // pass 1: max, count above the target
      float m = -std::numeric_limits<float>::infinity();
      int cnt = 0;
      for (int i = beg; i < end; ++i) {
        const float v = static_cast<float>(sycl::ext::oneapi::bfloat16(x[i]));
        m = sycl::fmax(m, v);
        cnt += v > xt;
      }
      // A cutoff that keeps the histogram small: the K-th largest of the items' maxima. K items each
      // hold a value >= it, so at least K values are >= it (exact), and few values are. A histogram
      // over all values would serialise on the few bins most logits share.
      float lo = -std::numeric_limits<float>::infinity();
      if (K > 0) {
        imax[lid] = m;
        sycl::group_barrier(g);
        int greater = 0;
        for (int j = 0; j < kLpItems; ++j) greater += imax[j] > m;
        lo = sycl::reduce_over_group(g, greater < K ? m : std::numeric_limits<float>::infinity(), sycl::minimum<float>());
      }
      m = sycl::reduce_over_group(g, m, sycl::maximum<float>());
      cnt = sycl::reduce_over_group(g, cnt, sycl::plus<int>());
      // pass 2: sum of exponentials; the high-byte histogram of the values >= lo
      float s = 0.f;
      for (int i = beg; i < end; ++i) {
        const float v = static_cast<float>(sycl::ext::oneapi::bfloat16(x[i]));
        s += sycl::exp(v - m);
        if (K > 0 && v >= lo)
          sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::work_group, sycl::access::address_space::local_space>(hist[lp_key(v) >> 8]) += 1;
      }
      s = sycl::reduce_over_group(g, s, sycl::plus<float>());
      const float floor_v = lo;
      sycl::group_barrier(g);
      if (lid == 0) {
        int acc = 0, b1 = 0;
        for (int b = 255; b >= 0 && K > 0; --b) {
          if (acc + hist[b] >= K) {
            b1 = b;
            break;
          }
          acc += hist[b];
        }
        misc[0] = b1;
        misc[1] = acc;
      }
      sycl::group_barrier(g);
      const int b1 = misc[0];
      if (lid < 256) hist[lid] = 0;
      sycl::group_barrier(g);
      // pass 3: low-byte histogram inside the threshold bin
      for (int i = beg; i < end && K > 0; ++i) {
        const float v = static_cast<float>(sycl::ext::oneapi::bfloat16(x[i]));
        if (!(v >= floor_v)) continue;
        const uint32_t k = lp_key(v);
        if (static_cast<int>(k >> 8) == b1)
          sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::work_group, sycl::access::address_space::local_space>(hist[k & 255]) += 1;
      }
      sycl::group_barrier(g);
      if (lid == 0) {
        int acc = misc[1], t = 0;
        for (int b = 255; b >= 0 && K > 0; --b) {
          if (acc + hist[b] >= K) {
            t = (b1 << 8) | b;
            break;
          }
          acc += hist[b];
        }
        misc[2] = t;  // the top-th largest key
        misc[1] = acc;  // keys above it (< K)
        misc[3] = 0;  // collection counter
        out[0] = m;
        out[1] = s;
        out[2] = static_cast<float>(cnt);
        out[3] = static_cast<float>(K);
      }
      sycl::group_barrier(g);
      if (K == 0) return;
      const uint32_t T_ = static_cast<uint32_t>(misc[2]);
      const int above = misc[1], need = K - above;
      // pass 3: every key above T (any order), then the first `need` ties at T in index order
      int ties = 0;
      for (int i = beg; i < end; ++i) {
        const float v = static_cast<float>(sycl::ext::oneapi::bfloat16(x[i]));
        const uint32_t k = lp_key(v);
        if (k > T_) {
          const int at = sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::work_group, sycl::access::address_space::local_space>(misc[3]).fetch_add(1);
          out[4 + 2 * at] = static_cast<float>(col_base + i);
          out[5 + 2 * at] = v;
        }
        ties += k == T_;
      }
      int off = sycl::exclusive_scan_over_group(g, ties, sycl::plus<int>());
      for (int i = beg; i < end && off < need; ++i) {
        const float v = static_cast<float>(sycl::ext::oneapi::bfloat16(x[i]));
        if (lp_key(v) == T_) {
          const int at = above + off++;
          out[4 + 2 * at] = static_cast<float>(col_base + i);
          out[5 + 2 * at] = v;
        }
      }
    });
  });
}

}  // namespace

sycl::event logprob_stats(sycl::queue &q, const float *logits, int ld, int rows, int n, int col_base, int top, const LpArgs &a, float *rec) {
  if (rows < 1 || rows > kLpMaxRows || n < 1 || ld < n || top < 0 || top > kLpMaxTop) throw std::invalid_argument("logprob_stats: shape");
  return stats<float, false>(q, logits, ld, rows, n, col_base, top, a, nullptr, rec);
}

sycl::event logprob_stats_rows(sycl::queue &q, const sycl::ext::oneapi::bfloat16 *logits, int ld, int rows, int n, int col_base, int top, const float *xt,
                               float *rec) {
  if (rows < 1 || n < 1 || ld < n || top < 0 || top > kLpMaxTop || !xt) throw std::invalid_argument("logprob_stats_rows: shape");
  return stats<sycl::ext::oneapi::bfloat16, true>(q, logits, ld, rows, n, col_base, top, LpArgs{}, xt, rec);
}

sycl::event logprob_targets(sycl::queue &q, const sycl::ext::oneapi::bfloat16 *logits, int ld, int rows, int n, int col_base, const int32_t *targets, float *xt) {
  if (rows < 1 || n < 1 || ld < n) throw std::invalid_argument("logprob_targets: shape");
  return q.parallel_for<LogprobTargets>(sycl::range<1>(static_cast<size_t>(rows)), [=](sycl::id<1> id) {
    const int r = static_cast<int>(id[0]), t = targets[r];
    xt[r] = t >= col_base && t < col_base + n ? static_cast<float>(logits[static_cast<size_t>(r) * ld + (t - col_base)]) : 0.f;
  });
}

}  // namespace xenithium:kernels
