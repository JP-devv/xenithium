// Xenithium · the serving backend of the qwen35 runtime: the dense Qwen3.8-27B, and the llama (Nanbeige4.3) and flint (Recursive Flint INT4) archs on it: src/arch/qwen35/serve.cpp, listed in src/serve/registry.cpp.
#pragma once

#include "xenithium/serve/backend.hpp"

namespace xenithium::arch::qwen35 {

serve::BackendEntry serve_entry();

}  // namespace xenithium::arch::qwen35
