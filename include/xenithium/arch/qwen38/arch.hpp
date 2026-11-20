// Xenithium · arch module "qwen38": Qwen3.8-Flash-Next (HF model_type qwen4_exp).
//
// The typed configuration, read from config.json and checked for the invariants the
// engine relies on. Semantics follow the model's own configuration and checkpoint.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "xenithium/model/arch.hpp"

namespace xenithium::arch::qwen38 {

struct Config {
  // trunk
  int hidden = 0;
  int layers = 0;
  int64_t vocab = 0;
  int64_t max_positions = 0;
  double rms_eps = 0;
  bool tie_embeddings = false;
  std::vector<bool> full_attention;  // per layer: true = QSA (softmax), false = GDN (linear)
  int full_interval = 0;

  // full attention (QSA): gated GQA + a block indexer
  int q_heads = 0, kv_heads = 0, head_dim = 0;
  double partial_rotary = 0;  // fraction of head_dim that is rotated
  double rope_theta = 0;
  std::string output_gate;    // "sigmoid"
  int idx_heads = 0, idx_kv_heads = 0, idx_head_dim = 0;
  int idx_compress = 0;       // tokens per compressed key block
  int idx_budget = 0;         // tokens attended per query (selected blocks x compress)

  // linear attention (Gated DeltaNet)
  int lin_k_heads = 0, lin_v_heads = 0, lin_k_dim = 0, lin_v_dim = 0, conv_kernel = 0;

  // MoE
  int experts = 0, topk = 0, moe_inter = 0, shared_inter = 0;

  // hyper-connections
  int hc = 0, hc_lowrank = 0;

  // per-layer n-gram embedding (PLE)
  std::vector<int> ple_layer_ids;  // as written in config.json
  int ple_layer = -1;              // the decoder layer that holds the block (from the checkpoint)
  int ngram = 0, heads_per_ngram = 0, ple_dim = 0, ple_conv_kernel = 0, ple_split_parts = 0;
  int64_t ngram_vocab_base = 0;
  int64_t ple_seed = 0;            // config `seed` (hash multipliers); 1234 when absent
  int64_t ple_divisor = 0;         // make_ngram_vocab_size_divisible_by (table rows padded to it)

  // speculative drafter
  int mtp_layers = 0;

  // quantisation of the routed experts
  int w_bits = 0, w_group = 0;
  bool w_sym = false;
  std::string w_packing;
  bool w_unquantized = false;
  // quantization_config.act_bits: 8 = the experts were tuned for int8 activations (the AutoRound W4A8
  // bake); 16 (absent) = bf16 activations (the GPTQ / AutoRound W4A16 export). Picks the decode MoE numerics by default.
  int act_bits = 16;
  bool w_rotate = false;  // quantization_config.xenithium_rotation == "hadamard-x128-h64" // no quantization_config: a BF16 source (fused BF16 experts), the bake's input

  int64_t bos = -1;
  std::vector<int64_t> eos;

  int full_layers() const;
  int ple_heads() const { return (ngram - 1) * heads_per_ngram; }
  int ple_head_dim() const { return ple_heads() > 0 ? ple_dim / ple_heads() : 0; }
  int lin_qkv() const { return 2 * lin_k_heads * lin_k_dim + lin_v_heads * lin_v_dim; }

  // Throws std::runtime_error naming the key or the broken invariant.
  static Config parse(const model::ModelConfig &cfg);
};

// Tensor-name prefix of the language model in this checkpoint family.
inline constexpr const char *kPrefix = "model.language_model.";

model::ArchEntry entry();

}  // namespace xenithium::arch::qwen38
