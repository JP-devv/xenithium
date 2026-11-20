// Xenithium · one GPU: the Level Zero objects the engine owns plus the SYCL
// view that runs ESIMD kernels on top of them.
// Two contexts would be a bug: USM allocated from our ze_context is only
// visible to kernels built on a SYCL context wrapping that same ze_context.
#pragma once

#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <sycl/sycl.hpp>

#include <memory>
#include <string>
#include <vector>

#include "xenithium/rt/l0.hpp"

namespace xenithium {

class Device {
 public:
  // Opens device `index` of the (ZE_AFFINITY_MASK-filtered) L0 device list.
  static Device open(const l0::Loader &loader, size_t index);
  static std::vector<Device> open_all();
  static std::vector<Device> open_all(const l0::Loader &loader);

  Device(const Device &) = delete;
  Device &operator=(const Device &) = delete;
  Device(Device &&) = default;
  Device &operator=(Device &&) = default;

  size_t index() const { return index_; }
  const l0::DeviceInfo &info() const { return info_; }
  ze_device_handle_t ze_device() const { return ze_device_; }
  ze_context_handle_t ze_context() const { return ctx_.handle(); }

  sycl::device &sycl_device() { return sycl_device_; }
  sycl::context &sycl_context() { return sycl_context_; }
  // Compute queue: kernels, in-order so a recorded step is deterministic.
  sycl::queue &compute() { return compute_; }
  // Copy queue: Level Zero, used for host transfers.
  l0::Queue &copy_queue() { return copy_queue_; }
  // SYCL copy queue on the same context, for comparisons in the probe.
  sycl::queue &sycl_copy() { return sycl_copy_; }

  template <typename T>
  T *alloc(size_t count, l0::MemKind kind) {
    return static_cast<T *>(ctx_.alloc(count * sizeof(T), kind, ze_device_));
  }
  void release(void *ptr) { ctx_.free(ptr); }
  uint64_t alloc_size(void *ptr, ze_memory_type_t *type = nullptr) const {
    return ctx_.alloc_size(ptr, type);
  }

  // Memory module size of this card (a ceiling, not free space).
  uint64_t budget_vram() const;

 private:
  Device(const l0::Loader &loader, size_t index);

  size_t index_ = 0;
  l0::DeviceInfo info_;
  ze_device_handle_t ze_device_ = nullptr;
  l0::Context ctx_;
  sycl::device sycl_device_;
  sycl::context sycl_context_;
  sycl::queue compute_;
  sycl::queue sycl_copy_;
  l0::Queue copy_queue_;
};

// Silence: a SYCL device matching a Level Zero handle, or throws.
sycl::device find_sycl_device(ze_device_handle_t ze_device);

}  // namespace xenithium