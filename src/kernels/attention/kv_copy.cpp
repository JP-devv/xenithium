#include "xenithium/kernels/attention/kv_copy.hpp"

#include <stdexcept>

namespace xenithium::kernels {

class KvCopyPagesK;

sycl::event kv_copy_pages(sycl::queue &q, const std::vector<KvPageBuf> &bufs, const int32_t *src, const int32_t *dst, int n) {
  sycl::event last;
  if (n <= 0) return last;
  for (const KvPageBuf &b : bufs) {
    if (!b.base || b.page_bytes == 0) continue;
    if (b.page_bytes % 16 || reinterpret_cast<uintptr_t>(b.base) % 16) throw std::invalid_argument("kv_copy_pages: page bytes and bases are 16-byte multiples");
    const size_t lanes = b.page_bytes / 16;  // 16-byte items per page
    auto *base = static_cast<sycl::uint4 *>(b.base);
    const size_t total = lanes * static_cast<size_t>(n);
    constexpr size_t kWg = 256;
    last = q.parallel_for<KvCopyPagesK>(sycl::nd_range<1>((total + kWg - 1) / kWg * kWg, kWg), [=](sycl::nd_item<1> it) {
      const size_t g = it.get_global_id(0);
      if (g >= total) return;
      const size_t i = g / lanes, j = g % lanes;
      base[static_cast<size_t>(dst[i]) * lanes + j] = base[static_cast<size_t>(src[i]) * lanes + j];
    });
  }
  return last;
}

}  // namespace xenithium::kernels
