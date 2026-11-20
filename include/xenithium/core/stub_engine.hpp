// A GPU-less engine with the real engine's threading and event contract, for
// developing and gating server/.
//
// Each step takes `step_ms`; a request's first step is its prefill and yields one
// token, every later step yields `tokens_per_step` (as MTP k=3 does when every draft
// is accepted), and the speculative counters move with them, so ms/step and acceptance
// arithmetic over the counters can be exercised end to end. Output is the script
// from its start (or from seed % length when seeded with temperature > 0).
#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

#include "xenithium/core/admission.hpp"
#include "xenithium/core/engine.hpp"

namespace xenithium::core {

struct StubConfig {
  ModelInfo model;
  std::vector<int32_t> script;  // tokens emitted in a loop
  std::vector<int32_t> eos_ids;
  int32_t eos_after = 0;  // > 0: emit eos_ids[0] after this many tokens unless ignore_eos
  double step_ms = 2.0;
  int tokens_per_step = 4;
  int slots = 0;  // requests decoded at once; 0 = all of them (no queue). Others wait, as in GpuEngine
  AdmissionConfig admission;
};

class StubEngine final : public Engine {
 public:
  explicit StubEngine(StubConfig cfg);
  ~StubEngine() override;

  void start(EventSink* sink) override;
  void stop() override;
  void submit(uint64_t id, std::vector<int32_t> prompt, const SamplingParams& params) override;
  void cancel(uint64_t id) override;
  Stats stats() const override;
  const ModelInfo& model() const override { return cfg_.model; }

 private:
  struct Req {
    SamplingParams params;
    size_t prompt_len = 0;
    size_t pos = 0;  // into the script
    int32_t generated = 0;
    bool prefilled = false;
    std::chrono::steady_clock::time_point t_submit, t_admit;
  };
  struct Submit {
    uint64_t id;
    size_t prompt_len;
    SamplingParams params;
    std::chrono::steady_clock::time_point t_submit;
  };

  void loop();
  bool is_stop_token(const Req& r, int32_t tok) const;

  StubConfig cfg_;
  EventSink* sink_ = nullptr;
  std::thread thread_;

  std::mutex mu_;  // guards the inbox and stop_
  std::condition_variable cv_;
  std::vector<Submit> submits_;
  std::vector<uint64_t> cancels_;
  bool stop_ = false;

  mutable std::mutex stats_mu_;
  Stats stats_;

  std::deque<Submit> waiting_;                // admitted in arrival order or by policy; engine thread only
  std::unordered_map<uint64_t, Req> active_;  // engine thread only
};

}  // namespace xenithium::core
