#pragma once
#include <chrono>
#include <cstdlib>
#include <cstdint>
#include <string_view>
#include <functional>
#include <vector>

namespace xenithium::util::ruby_round {
inline int mode() {
  static const int x=[] { const char *s=std::getenv("XEN_BENCH_RUBY_ROUNDS");
    return !s?0:std::string_view(s)=="2"?2:std::string_view(s)=="1"?1:0; }();
  return x;
}
using clock=std::chrono::steady_clock;
inline int64_t ns() { return std::chrono::duration_cast<std::chrono::nanoseconds>(clock::now().time_since_epoch()).count(); }
struct Round {
  int64_t step=0, pos=0, verify0=0,verify1=0,pick1=0,accept0=0,pack0=0,pack1=0,draft0=0,draft1=0,accept1=0,mtp_ns=0,block_ns=0;
  double dflash_pass_ms=0;
  int width=0, device_sample=0;
  bool tree=false;
};
inline thread_local Round r{};
inline thread_local std::vector<std::function<void()>> context_events;
inline int context_rows() {
  static const int n=[] { const char *s=std::getenv("XEN_BENCH_CONTEXT_ROWS"); return s?std::atoi(s):0; }();return n;
}
}
