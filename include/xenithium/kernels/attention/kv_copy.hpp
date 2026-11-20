// Xenithium · whole-page KV copies on one card (the prefix cache's fork).
//
// A paged KV buffer holds physical page p at base + p x page_bytes (the K / V codes of a layer, its per-token scales,
// the drafter's). A fork gives a new sequence the KV of another sequence's first positions: logical page i of the source
// (physical src[i]) is copied to the new sequence's physical page dst[i], for every buffer, in one launch per buffer.
// Plain byte copies (16 bytes per item); page_bytes a multiple of 16, every base 16-byte aligned. The pairs are device
// memory ([n] src then [n] dst). Exact by construction: the gate is byte equality of the copied pages (test-kernels).
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include <sycl/sycl.hpp>

namespace xenithium::kernels {

struct KvPageBuf {
  void *base = nullptr;
  size_t page_bytes = 0;
};

// For each buffer: pages src[i] -> dst[i], i < n (src / dst device ints). Submits only (one launch per buffer).
sycl::event kv_copy_pages(sycl::queue &q, const std::vector<KvPageBuf> &bufs, const int32_t *src, const int32_t *dst, int n);

}  // namespace xenithium::kernels
