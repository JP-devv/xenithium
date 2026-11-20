// Xenithium · the serving backend of Qwen3.8 Flash-Next (MoE + Gated DeltaNet, TP2 + EP2): src/arch/qwen38/serve.cpp, listed in src/serve/registry.cpp.
#pragma once

#include "xenithium/serve/backend.hpp"

namespace xenithium::arch::qwen38 {

serve::BackendEntry serve_entry();

}  // namespace xenithium::arch::qwen38
