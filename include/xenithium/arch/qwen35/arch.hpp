// Xenithium · arch module "qwen35": the dense Qwen3.5 / Qwen3.8-27B family (HF model_type qwen3_5).
//
// A hybrid decoder of 48 Gated DeltaNet layers and 16 gated full-attention (GQA) layers with a dense
// SwiGLU MLP in every layer, plain residual connections (no hyper-connections, no PLE, no MoE).
// The served checkpoints store every decoder Linear except `in_proj_a` / `in_proj_b` and `lm_head`
// as int4, group 128, symmetric, in one of two layouts: compressed-tensors "pack-quantized"
// (`weight_packed` I32 [n, k/8], 8 codes per word, code i at bits 4i, stored q + 8; `weight_scale`
// F16 [n, k/128]; `weight_shape` I64 [2]: dbirks/Qwen3.8-27B-W4A16-AutoRound) or AutoRound's
// "auto_round:auto_gptq" (`qweight` I32 [k/8, n], `scales` F16 [k/128, n], `qzeros` I32 [k/128, n/8]:
// ukisai Swift-1.5-Qwen3.8-27b W4A16). The typed configuration below is read from config.json and checked for the
// invariants the engine relies on.
#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "xenithium/model/arch.hpp"

namespace xenithium::arch::qwen35 {

struct Config {
  // trunk
  int hidden = 0;
  int layers = 0;
  int64_t vocab = 0;
  int64_t max_positions = 0;
  double rms_eps = 0;
  bool tie_embeddings = false;
  std::vector<bool> full_attention;  // per layer: true = gated GQA (softmax), false = GDN (linear)
  int full_interval = 0;

  // full attention: gated GQA (q_proj = query | gate per head when attn_output_gate), q/k RMS norms,
  // partial rotary; the rope is mrope (interleaved) in the multimodal config: plain 1-D for text, 3-D positions for image
  // tokens with image input (Model::set_vision, model::Mrope).
  int q_heads = 0, kv_heads = 0, head_dim = 0;
  bool attn_output_gate = false;
  double partial_rotary = 0;  // fraction of head_dim that is rotated
  double rope_theta = 0;
  std::string rope_type;            // "default"
  std::vector<int> mrope_section;   // [t, h, w] rotary sections (text positions use all of them)
  bool mrope_interleaved = false;

  // linear attention (Gated DeltaNet); output gate `output_gate_type` ("swish": o = norm(o) * silu(z))
  int lin_k_heads = 0, lin_v_heads = 0, lin_k_dim = 0, lin_v_dim = 0, conv_kernel = 0;
  std::string output_gate;

  // dense MLP (SwiGLU)
  int inter = 0;
  std::string hidden_act;  // "silu"

  // speculative drafter
  int mtp_layers = 0;
  bool mtp_dedicated_embeddings = false;

  // quantisation of the decoder Linears: compressed-tensors "pack-quantized" (weight_packed [n][k/8],
  // weight_scale [n][k/128]) or AutoRound's "auto_round:auto_gptq" (qweight [k/8][n], scales [k/128][n],
  // qzeros [k/128][n/8] all 0x77777777 = symmetric, stored zero 7 -> effective 8;)
  int w_bits = 0, w_group = 0;
  bool w_sym = false;
  std::string w_method;  // "compressed-tensors" | "auto-round" | "gptq" (gptqmodel)
  std::string w_format;  // "pack-quantized" | "auto_round:auto_gptq" | "gptq"
  bool w_gptq() const { return w_method == "auto-round" || w_method == "gptq"; }
  // Names of the tensors of one int4 Linear `module` (e.g. "...mlp.gate_proj") in this checkpoint's format.
  std::string zero_name(const std::string &module) const { return module + ".weight_zero_point"; }
  std::string packed_name(const std::string &module) const {
    return module + (w_mlx ? ".weight" : w_gptq() ? ".qweight" : ".weight_packed");
  }
  std::string scale_name(const std::string &module) const { return module + (w_mlx || w_gptq() ? ".scales" : ".weight_scale"); }

  int64_t bos = -1;
  std::vector<int64_t> eos;

  // The dense decoder family this runtime serves beyond Qwen3.5 (arch llama, filled by arch::llama::parse_config):
  // every layer full attention, no output gate, no q / k norms, the plain RMSNorm, compressed-tensors int4 with zero
  // points (asymmetric) in groups of w_group, bf16 scales.
  std::string prefix = "model.language_model.";  // tensor-name prefix of the decoder (llama: "model.")
  bool norm_offset = true;   // RMSNorm weight applied as (1 + w) (Qwen3.5) or as w (Llama)
  bool qk_norm = true;       // per-head q / k RMS norms before the rope
  bool w_zero = false;       // int4 codes with a zero point per (row, group): weight_zero_point I32 [n/8][k/group]
  bool w_scale_bf16 = false; // weight_scale stored BF16 (else F16)
  bool llama() const { return !qk_norm; }
  // Arch flint (Recursive Flint INT4, filled by arch::flint::runtime_config): every layer GQA (head_dim 128, no gate)
  // with per-head plain q / k norms and a learned sink logit per q head; `sliding` layers attend `window` tokens with
  // rotary over all head dims, the others the whole context without positional encoding; the MLP of each `moe` layer is
  // the sigmoid top-k MoE (experts x moe_inter, an ungated shared expert of shared_inter), the others the dense SwiGLU of
  // `inter`; affine int4 g64 Linears (`.weight` U8 codes, `.scales` / `.biases` BF16), an 8-bit embedding.
  bool flint = false;
  std::vector<bool> sliding, moe;
  int window = 0;
  int experts = 0, topk = 0, moe_inter = 0, shared_inter = 0;
  bool w_mlx = false;

  int full_layers() const;
  int lin_qkv() const { return 2 * lin_k_heads * lin_k_dim + lin_v_heads * lin_v_dim; }
  int q_rows() const { return (attn_output_gate ? 2 : 1) * q_heads * head_dim; }

  // Throws std::runtime_error naming the key or the broken invariant.
  static Config parse(const model::ModelConfig &cfg);
};

// Tensor-name prefix of the language model in this checkpoint family.
inline constexpr const char *kPrefix = "model.language_model.";

// The `weight_scale` / `scales` tensor beside a packed one (either format), for callers that hold the packed name.
std::string scale_name_of(const std::string &packed_name);

model::ArchEntry entry();

// The MTP drafter's tensors (`mtp.*`, bf16 in the checkpoint) with their placement, as one more
// full-attention decoder layer: q_proj (+ k, v appended) split by head rows, o_proj and down_proj along k, gate | up by
// intermediate rows, fc [hidden][2 hidden] along k (card 0 gets the embedding half, card 1 the hidden half: each card
// norms one input and the two partial sums meet in the all-reduce), norms replicated. Matrices int8 per row
// (dense_bytes 1) or bf16 (2). The embedding and lm_head are the target's (mtp_use_dedicated_embeddings false).
// head_rows (the draft vocabulary): non-empty = also the target's lm_head rows in these ranges as one more
// row-split matrix under the key kMtpHeadKey (same dtype as the drafter's matrices).
inline constexpr const char *kMtpHeadKey = "mtp.draft_head";
std::vector<model::DenseSpec> mtp_dense_plan(const model::Checkpoint &ck, int dense_bytes, const std::vector<std::pair<int64_t, int64_t>> &head_rows = {},
                                             bool head_w4 = false);  // head rows from lm_head.qweight

// The pages a kv head of that weight keeps at the common count `pages` (weights of mean 1, SelectShare::weights):
// they share out the pages past kSelectFloor, so no head keeps fewer than that count gives every head.
inline constexpr int kSelectFloor = 128;
int select_head_pages(float weight, int pages);

// The page-count weights from "attention mass lost per fraction of the pages kept" curves, one curve per (attention
// layer, kv head) unit. The pages go where they remove the most lost mass: every unit starts at the smallest fraction,
// and the next pages go to the unit and jump ahead with the best lost mass removed per page, until the mean fraction is
// kMean; a unit's weight is its fraction over the mean, pulled towards 1 by kSoften. Used by the model for a sequence
// that measures itself (ModelOptions::select_self) and by the probe's host check of it.
struct SelectShare {
  static constexpr int kFractions = 111;  // fraction(i) = (i + 2) / 128 of the context's pages: 1.6 % .. 87.5 %
  static constexpr double kMean = 0.25, kSoften = 0.75;
  static constexpr double fraction(int i) { return (i + 2) / 128.0; }
  // the pages kept of `pages` at fraction f: never fewer than the forced ones (sink + local) and one more
  static int count(int pages, double f, int forced);
  // lost: [units][kFractions], row-major, each row non-increasing; returns [units], mean 1
  static std::vector<float> weights(std::span<const double> lost, int units);
};
// The selected steps a sequence measures its page counts on (ModelOptions::select_self as served).
inline constexpr int kSelectSelfSteps = 8;
// The score of a page from the select launch's order key (gqa_sel.cpp order_keys: larger score, larger key).
float select_key_score(uint32_t key);
// One unit's curve from its pages' scores alone (no attention read back): a page's mass is exp(scale x (score - the
// largest)) x its tokens, normalised over the pages; the pages are kept in the engine's order (the first `sink`, the
// last `local`, then by score, ties to the lower page). Adds the mass lost at each fraction to out[kFractions].
// scores: one per page of the context; tokens: the context's tokens (the last page may be partial).
void select_curve_add(std::span<const float> scores, int64_t tokens, int page_tokens, int sink, int local, float scale, std::span<double> out);

}  // namespace xenithium::arch::qwen35
