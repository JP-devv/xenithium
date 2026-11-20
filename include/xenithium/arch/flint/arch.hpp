// Xenithium · arch module "flint": Recursive Flint (HF model_type flint, FlintForCausalLM), a sparse MoE decoder.
//
// Served checkpoint: Flint INT4: 49 layers, hidden 2048, vocab 150144, 262,144 positions. Attention: 32 q / 4 kv heads
// x 128 in every layer, per-head q / k RMSNorm (plain w x norm(x)), a learned
// per-head sink logit (`self_attn.softmax_offset`, it enters the softmax denominator and attends to no value);
// `layer_types` makes most layers sliding-window (the current token and the 512 before it, `sliding_window` 513) with
// NeoX rotary over all 128 dims (theta 1e4), and 14 layers full attention with no positional encoding
// (`no_rope_layer_indices`). MLP: layers 0 and 1 (`mlp_only_layers`) a dense SwiGLU (inter 6144); every other layer 256
// routed SwiGLU experts (inter 512) by sigmoid score, top 6, the selected scores renormalised (no router bias), plus an
// ungated shared SwiGLU expert (inter 1024). Untied lm_head.
//
// Weights ("affine" quantisation, `quantization` {bits 4, group_size 64}): every Linear `M` is `M.weight` U8
// [n][k x bits / 8] (codes little-endian, element 2j in the low nibble of byte j: the compressed-tensors word order),
// `M.scales` BF16 [n][k/64] and `M.biases` BF16 [n][k/64]: w = q x scale + bias, q in 0..2^bits - 1 (the bias is not a
// multiple of the scale: no integer zero point). The routed experts are stacked per layer (`mlp.switch_mlp.*` [E][n][..]).
// The embedding is 8-bit (U8 [vocab][hidden], same scales / biases form), lm_head 4-bit, norms / router / sinks BF16.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "xenithium/arch/qwen35/arch.hpp"
#include "xenithium/model/arch.hpp"

namespace xenithium::arch::flint {

struct Config {
  // trunk
  int hidden = 0;
  int layers = 0;
  int64_t vocab = 0;
  int64_t max_positions = 0;
  double rms_eps = 0;

  // attention (every layer): GQA with per-head q / k norms and sinks
  int q_heads = 0, kv_heads = 0, head_dim = 0;
  std::vector<bool> sliding;  // per layer: true = sliding window (rotary), false = full attention (no rotary)
  int window = 0;             // tokens a sliding layer attends: the current one and window - 1 before it
  double rope_theta = 0;      // NeoX halves over all head_dim channels
  bool sinks = false;         // a learned sink logit per q head (softmax_offset)

  // MLP
  std::vector<bool> dense;  // per layer: true = dense SwiGLU (mlp_only_layers), false = MoE
  int inter = 0;            // dense SwiGLU intermediate
  int experts = 0, topk = 0, moe_inter = 0, shared_inter = 0;
  bool norm_topk = true;  // the selected sigmoid scores renormalised to sum 1
  std::string scoring;    // "sigmoid"
  std::string hidden_act; // "silu"

  // Affine quantisation: Linears w_bits / w_group, the embedding embed_bits (same group)
  int w_bits = 0, w_group = 0, embed_bits = 0;

  int64_t bos = -1, pad = -1;
  std::vector<int64_t> eos;

  int full_layers() const;
  int sliding_layers() const { return layers - full_layers(); }
  int moe_layers() const;
  int q_dim() const { return q_heads * head_dim; }
  int kv_dim() const { return kv_heads * head_dim; }

  // Throws std::runtime_error naming the key or the broken invariant.
  static Config parse(const model::ModelConfig &mc);
};

inline constexpr const char *kPrefix = "model.";

// Tokens a sliding layer keeps per sequence slot: the window plus a step's rows (up to `rows` new tokens attend
// window - 1 keys back each), in whole 64-token pages, plus one page of slack for a page-unaligned start.
int64_t ring_tokens(const Config &c, int rows);

model::ArchEntry entry();

// The runtime's configuration (arch::qwen35::Model serves flint: Config::flint and its fields) from config.json.
qwen35::Config runtime_config(const model::ModelConfig &mc);

}  // namespace xenithium::arch::flint
