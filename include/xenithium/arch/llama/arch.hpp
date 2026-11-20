// Xenithium · arch module "llama": dense Llama-family decoders (HF model_type llama, LlamaForCausalLM).
//
// Served checkpoint: Nanbeige4.3-33B-INT4: 52 layers, hidden 7168, 56 q / 8 kv
// heads x 128, full rotary (NeoX halves, theta 1.2e8, no scaling), plain RMSNorm (w x norm(x), eps 1e-5), SwiGLU MLP
// (inter 22016), untied BF16 embedding and lm_head (vocab 166912). Every decoder Linear is compressed-tensors
// "pack-quantized" int4, ASYMMETRIC, group 32: `weight_packed` I32 [n][k/8] (code i of a word at bits 4i, q in 0..15),
// `weight_scale` BF16 [n][k/32], `weight_zero_point` I32 [n/8][k/32] (eight 4-bit zero points per word, along n),
// `weight_shape` I32 [2]; w = (q - z) x scale.
//
// The runtime is the dense decoder of arch qwen35 (arch::qwen35::Model): this module fills its Config with the
// Llama semantics (every layer full attention, no output gate, no q / k norms, the plain norm, zero points); the
// names, the validation, the placement and the budget below are this family's own, read from config.json and the
// checkpoint.
#pragma once

#include "xenithium/arch/qwen35/arch.hpp"
#include "xenithium/model/arch.hpp"

namespace xenithium::arch::llama {

// config.json -> the dense decoder's typed configuration (qwen35::Config with the llama fields set). Throws
// std::runtime_error naming the key or the broken invariant.
qwen35::Config parse_config(const model::ModelConfig &mc);

inline constexpr const char *kPrefix = "model.";

model::ArchEntry entry();

}  // namespace xenithium::arch::llama
