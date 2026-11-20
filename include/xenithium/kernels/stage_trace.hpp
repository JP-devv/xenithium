// Xenithium · per-stage event traces of a kernel family's launches (prefill profiling).
// A model passes a trace in a module's scratch when it profiles a prefill call (qwen38: XEN_PF_PROFILE=1 on profiling
// queues, probe --queue-prof 1); the module adds the event of each launch under its stage, and the model sums the
// device time per stage after the call. Without a trace nothing is recorded and the launches are the same.
#pragma once

#include <vector>
#include <sycl/sycl.hpp>

namespace xenithium::kernels {

template <int Stages>
struct StageTrace {
  std::vector<sycl::event> ev[Stages];

  void add(int stage, const sycl::event &e) { ev[stage].push_back(e); }
  void clear() {
    for (auto &v : ev) v.clear();
  }
  // device ms of the stage's launches: the sum of each launch's start -> end (the queue must profile)
  double ms(int stage) const {
    double ns = 0;
    for (const auto &e : ev[stage])
      ns += static_cast<double>(e.template get_profiling_info<sycl::info::event_profiling::command_end>() -
                                e.template get_profiling_info<sycl::info::event_profiling::command_start>());
    return ns / 1e6;
  }
};

}  // namespace xenithium:kernels
