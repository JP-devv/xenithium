// Level Zero gate, every card: host -> device -> device -> host bit-exact, pinned H2D rate,
// and a device allocation that really is device memory.
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "xenithium/rt/device.hpp"

int main() {
  int failures = 0;
  auto expect = [&](bool ok, const char *what) {
    std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) ++failures;
  };
  try {
    const auto loader = xenithium::l0::Loader::open();
    expect(loader.size() > 0, "level zero device present");
    for (size_t d = 0; d < loader.size(); ++d) {
      auto dev = xenithium::Device::open(loader, d);
      const size_t bytes = 64ull << 20;
      auto *host = dev.alloc<uint32_t>(bytes / 4, xenithium::l0::MemKind::Host);
      auto *a = dev.alloc<uint8_t>(bytes, xenithium::l0::MemKind::Device);
      auto *b = dev.alloc<uint8_t>(bytes, xenithium::l0::MemKind::Device);
      std::mt19937 rng(static_cast<unsigned>(99 + d));
      for (size_t i = 0; i < bytes / 4; ++i) host[i] = rng();
      auto &cq = dev.copy_queue();
      cq.copy(a, host, bytes, true);
      cq.copy(b, a, bytes, true);
      std::vector<uint8_t> back(bytes);
      cq.copy(back.data(), b, bytes, true);
      char what[160];
      std::snprintf(what, sizeof what, "device %zu (%s): host->device->device->host 64 MiB bit-exact", d,
                    dev.info().pci_bdf().c_str());
      expect(std::memcmp(back.data(), host, bytes) == 0, what);

      const double us = cq.time_copy(a, host, bytes, 10);
      const double gbs = static_cast<double>(bytes) / (us * 1e3);
      std::snprintf(what, sizeof what, "device %zu: pinned host->device %.1f GB/s (> 5)", d, gbs);
      expect(gbs > 5.0, what);

      ze_memory_type_t type{};
      const uint64_t size = dev.alloc_size(a, &type);
      std::snprintf(what, sizeof what, "device %zu: allocation is %s memory, %llu MiB", d, xenithium::l0::mem_kind_name(type),
                    static_cast<unsigned long long>(size >> 20));
      expect(type == ZE_MEMORY_TYPE_DEVICE && size >= bytes, what);

      // Regression: back-to-back non-blocking copies on the one-list queue must all land
      // (resetting the list under a running copy would drop it), and so must
      // a batch of copies in one submit.
      cq.fill(a, 0, bytes, true);
      const size_t piece = 8ull << 20;
      for (size_t off = 0; off < bytes; off += piece) cq.copy(a + off, reinterpret_cast<uint8_t *>(host) + off, piece, false);
      cq.wait();
      cq.copy(back.data(), a, bytes, true);
      std::snprintf(what, sizeof what, "device %zu: 8 back-to-back non-blocking copies all land", d);
      expect(std::memcmp(back.data(), host, bytes) == 0, what);
      cq.fill(b, 0, bytes, true);
      std::vector<xenithium::l0::Queue::CopyOp> ops;
      for (size_t off = 0; off < bytes; off += piece) ops.push_back({b + off, reinterpret_cast<uint8_t *>(host) + off, piece});
      cq.copy_batch(ops, true);
      cq.copy(back.data(), b, bytes, true);
      std::snprintf(what, sizeof what, "device %zu: copy_batch of 8 copies lands", d);
      expect(std::memcmp(back.data(), host, bytes) == 0, what);
      dev.release(host);
      dev.release(a);
      dev.release(b);
    }
  } catch (const std::exception &e) {
    std::printf("[FAIL] exception: %s\n", e.what());
    return 1;
  }
  std::printf("%s: %d failure(s)\n", failures ? "FAILED" : "OK", failures);
  return failures == 0 ? 0 : 1;
}
