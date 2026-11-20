#include "xenithium/rt/device.hpp"

#include <stdexcept>

#include "xenithium/rt/card.hpp"

namespace xenithium {

sycl::device find_sycl_device(ze_device_handle_t target) {
  for (auto &platform : sycl::platform::get_platforms()) {
    for (auto &dev : platform.get_devices()) {
      if (dev.get_backend() != sycl::backend::ext_oneapi_level_zero) continue;
      if (sycl::get_native<sycl::backend::ext_oneapi_level_zero>(dev) == target) return dev;
    }
  }
  throw std::runtime_error("xenithium: no SYCL device matches the Level Zero device");
}

Device::Device(const l0::Loader &loader, size_t index) : index_(index) {
  info_ = loader.info(index);
  ze_device_ = loader.device(index);

  sycl_device_ = find_sycl_device(ze_device_);
  // One context for everything: SYCL builds it, Level Zero borrows the native
  // handle. USM allocated through Level Zero is then visible to ESIMD kernels,
  // and the engine still owns its memory and its queue objects.
  sycl_context_ = sycl::context(std::vector<sycl::device>{sycl_device_});
  ctx_ = l0::Context::borrow(sycl::get_native<sycl::backend::ext_oneapi_level_zero>(sycl_context_));

  // In-order keeps a recorded step deterministic; profiling lets the probes time
  // kernels without a host-side stopwatch.
  sycl::property_list props{sycl::property::queue::in_order(), sycl::property::queue::enable_profiling()};
  compute_ = sycl::queue(sycl_context_, sycl_device_, props);
  sycl_copy_ = sycl::queue(sycl_context_, sycl_device_, props);

  const l0::QueueFamily families = l0::query_queue_families(ze_device_);
  copy_queue_ = l0::Queue(ctx_.handle(), ze_device_, static_cast<uint32_t>(families.copy), "copy");
}

Device Device::open(const l0::Loader &loader, size_t index) { return Device(loader, index); }

std::vector<Device> Device::open_all(const l0::Loader &loader) {
  std::vector<Device> devices;
  devices.reserve(loader.size());
  for (size_t i = 0; i < loader.size(); ++i) devices.push_back(Device(loader, i));
  return devices;
}

std::vector<Device> Device::open_all() {
  const l0::Loader loader = l0::Loader::open();
  return open_all(loader);
}

uint64_t Device::budget_vram() const {
  // Level Zero exposes the memory module size, not what the driver has left;
  // callers treat this as a budget ceiling, never as free space. XEN_EMU_CARD: the emulated card's.
  const uint64_t cut = rt::emulated_vram_cut(info_);
  return info_.vram_bytes > cut ? info_.vram_bytes - cut : 0;
}

}  // namespace xenithium