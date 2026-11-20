// Xenithium · the select kernels' exact top-k of one work-group, shared by qsa.cpp's select and the dense
// GQA's page selection (gqa_sel.cpp). Private to src/kernels.
#pragma once

#include <algorithm>
#include <cstdint>
#include <sycl/sycl.hpp>

namespace xenithium::kernels::topk {

inline constexpr int kSelItems = 1024;  // select work-group

// ---- select: exact top-k of up to N keys in one work-group of 1024 items (SIMT).
// Radix select from the highest bit where the keys differ, 8 bits per pass (histogram in SLM),
// then an ordered compaction (contiguous index ranges per item + group scans): ties go to the
// lowest index and the output is ascending.
// Two histogram buffers (the next pass's is cleared while sub-group 0 scans this one), so
// two barriers per pass; and an early stop: when the digit's bin holds exactly the
// keys still needed, every key >= the bin's floor is taken (K of them) and no tie is left to break,
// so T = floor - 1, need = 0 select the same set as the full descent.
struct SelSlm {
  uint32_t *hist;  // 2 x 256
  int32_t *sh;     // 2 x 4
};
template <class Valid, class Key, class Id>
void topk_wg(sycl::nd_item<1> it, const SelSlm &m, int N, Valid valid, Key key, Id id, int K, int32_t *out, int32_t *out_n) {
  const auto grp = it.get_group();
  const int li = static_cast<int>(it.get_local_id(0)), L = kSelItems;
  int nv = 0;
  uint32_t kmax = 0, kmin = 0xFFFFFFFFu;
  for (int i = li; i < N; i += L)
    if (valid(i)) {
      ++nv;
      const uint32_t k = key(i);
      kmax = k > kmax ? k : kmax;
      kmin = k < kmin ? k : kmin;
    }
  nv = sycl::reduce_over_group(grp, nv, sycl::plus<int>());
  bool all = nv <= K;
  uint32_t T = 0;
  int need = 0;
  if (!all) {
    kmax = sycl::reduce_over_group(grp, kmax, sycl::maximum<uint32_t>());
    kmin = sycl::reduce_over_group(grp, kmin, sycl::minimum<uint32_t>());
    const uint32_t diff = kmax ^ kmin;
    need = K;
    if (diff == 0) {
      T = kmax;
    } else {
      int left = 32 - sycl::clz(diff);  // bits below and including the highest differing bit
      uint32_t prefix = left == 32 ? 0u : (kmax >> left) << left;
      for (int i = li; i < 512; i += L) m.hist[i] = 0;
      sycl::group_barrier(grp);
      int buf = 0;
      bool stop = false;
      while (left > 0) {
        const int wdt = left < 8 ? left : 8, shift = left - wdt;
        const uint32_t dmask = (1u << wdt) - 1u;
        const uint32_t himask = left == 32 ? 0u : ~((1u << left) - 1u);
        uint32_t *H = m.hist + buf * 256, *Hn = m.hist + (buf ^ 1) * 256;
        int32_t *S = m.sh + buf * 4;
        for (int i = li; i < N; i += L)
          if (valid(i)) {
            const uint32_t k = key(i);
            if ((k & himask) == prefix)
              sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::work_group, sycl::access::address_space::local_space>(
                  H[(k >> shift) & dmask])
                  .fetch_add(1u);
          }
        sycl::group_barrier(grp);
        {
          // the digit: sub-group 0 scans the bins from the top (lane j: a run of bins per lane); the
          // other items clear the next pass's buffer (sub-groups are <= 32 items)
          const auto sg = it.get_sub_group();
          if (sg.get_group_linear_id() == 0) {
            const int SG = static_cast<int>(sg.get_local_linear_range()), ln = static_cast<int>(sg.get_local_linear_id());
            const int nb = static_cast<int>(dmask) + 1, run = (nb + SG - 1) / SG;
            int part = 0;
            for (int j = 0; j < run; ++j) {
              const int dg = static_cast<int>(dmask) - (ln * run + j);
              part += dg >= 0 ? static_cast<int>(H[dg]) : 0;
            }
            const int cum = sycl::inclusive_scan_over_group(sg, part, sycl::plus<int>());
            if (cum >= need && cum - part < need) {
              int c = cum - part;
              for (int j = 0; j < run; ++j) {
                const int dg = static_cast<int>(dmask) - (ln * run + j);
                const int h = dg >= 0 ? static_cast<int>(H[dg]) : 0;
                if (c + h >= need) {
                  S[0] = dg;
                  S[1] = c;
                  S[2] = c + h == need;
                  break;
                }
                c += h;
              }
            }
          } else {
            for (int i = li - 32; i >= 0 && i < 256; i += L - 32) Hn[i] = 0;
          }
        }
        sycl::group_barrier(grp);
        need -= S[1];
        prefix |= static_cast<uint32_t>(S[0]) << shift;
        left = shift;
        stop = S[2] != 0;
        buf ^= 1;
        if (stop) break;
      }
      if (stop && left > 0) {
        T = prefix - 1u;  // prefix > 0: a zero floor would take every key, more than K
        need = 0;
      } else {
        T = prefix;
      }
    }
  }
  // ordered compaction: one scan of (keys == T, keys > T) packed in 16-bit halves (counts per
  // group <= 32,768); taken before an item = gt_before + min(eq_before, need).
  const int per = (N + L - 1) / L, i0 = std::min(N, li * per), i1 = std::min(N, i0 + per);
  int ceq = 0, cgt = 0;
  for (int i = i0; i < i1; ++i) {
    if (!valid(i)) continue;
    if (all) {
      ++cgt;
      continue;
    }
    const uint32_t k = key(i);
    cgt += k > T;
    ceq += k == T;
  }
  const int packed = sycl::exclusive_scan_over_group(grp, (ceq << 16) | cgt, sycl::plus<int>());
  const int eq_before = packed >> 16, gt_before = packed & 0xFFFF;
  int o = gt_before + (eq_before < need ? eq_before : need);
  {
    int e = eq_before;
    for (int i = i0; i < i1; ++i) {
      if (!valid(i)) continue;
      bool take = all;
      if (!all) {
        const uint32_t k = key(i);
        take = k > T || (k == T && e++ < need);
      }
      if (take) out[o++] = id(i);
    }
  }
  if (li == 0) *out_n = all ? nv : K;
}

}  // namespace xenithium::kernels::topk
