#include "xenithium/arch/qwen38/arch.hpp"
#include "xenithium/kernels/attention/qsa_ref.hpp"  // kQsaV4Scales (host-only constants)

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <sstream>
#include <stdexcept>

namespace xenithium::arch::qwen38 {

using model::Component;
using model::DType;
using model::ModelConfig;

int Config::full_layers() const {
  return static_cast<int>(std::count(full_attention.begin(), full_attention.end(), true));
}

namespace {

[[noreturn]] void invariant(const std::string &what) { throw std::runtime_error("qwen38 config: " + what); }

int as_int(int64_t v, const char *key) {
  if (v < 0 || v > (1ll << 31) - 1) invariant(std::string(key) + " out of range");
  return static_cast<int>(v);
}

}  // namespace

Config Config::parse(const ModelConfig &mc) {
  const json::Value &t = mc.text();
  Config c;
  c.hidden = as_int(ModelConfig::need_int(t, "hidden_size"), "hidden_size");
  c.layers = as_int(ModelConfig::need_int(t, "num_hidden_layers"), "num_hidden_layers");
  c.vocab = ModelConfig::need_int(t, "vocab_size");
  c.max_positions = ModelConfig::need_int(t, "max_position_embeddings");
  c.rms_eps = ModelConfig::need_num(t, "rms_norm_eps");
  c.tie_embeddings = ModelConfig::need_bool(t, "tie_word_embeddings");
  for (const auto &s : ModelConfig::need_strs(t, "layer_types")) {
    if (s == "full_attention") c.full_attention.push_back(true);
    else if (s == "linear_attention") c.full_attention.push_back(false);
    else invariant("unknown layer type '" + s + "'");
  }
  c.full_interval = as_int(ModelConfig::need_int(t, "full_attention_interval"), "full_attention_interval");

  c.q_heads = as_int(ModelConfig::need_int(t, "num_attention_heads"), "num_attention_heads");
  c.kv_heads = as_int(ModelConfig::need_int(t, "num_key_value_heads"), "num_key_value_heads");
  c.head_dim = as_int(ModelConfig::need_int(t, "head_dim"), "head_dim");
  c.partial_rotary = ModelConfig::need_num(t, "partial_rotary_factor");
  const json::Value &rope = ModelConfig::need(t, "rope_parameters");
  c.rope_theta = ModelConfig::need_num(rope, "rope_theta");
  c.output_gate = ModelConfig::need_str(t, "output_gate_type");
  c.idx_heads = as_int(ModelConfig::need_int(t, "indexer_n_heads"), "indexer_n_heads");
  c.idx_kv_heads = as_int(ModelConfig::need_int(t, "indexer_kv_heads"), "indexer_kv_heads");
  c.idx_head_dim = as_int(ModelConfig::need_int(t, "indexer_head_dim"), "indexer_head_dim");
  c.idx_compress = as_int(ModelConfig::need_int(t, "indexer_compress_ratio"), "indexer_compress_ratio");
  c.idx_budget = as_int(ModelConfig::need_int(t, "indexer_budget"), "indexer_budget");

  c.lin_k_heads = as_int(ModelConfig::need_int(t, "linear_num_key_heads"), "linear_num_key_heads");
  c.lin_v_heads = as_int(ModelConfig::need_int(t, "linear_num_value_heads"), "linear_num_value_heads");
  c.lin_k_dim = as_int(ModelConfig::need_int(t, "linear_key_head_dim"), "linear_key_head_dim");
  c.lin_v_dim = as_int(ModelConfig::need_int(t, "linear_value_head_dim"), "linear_value_head_dim");
  c.conv_kernel = as_int(ModelConfig::need_int(t, "linear_conv_kernel_dim"), "linear_conv_kernel_dim");

  c.experts = as_int(ModelConfig::need_int(t, "num_experts"), "num_experts");
  c.topk = as_int(ModelConfig::need_int(t, "num_experts_per_tok"), "num_experts_per_tok");
  c.moe_inter = as_int(ModelConfig::need_int(t, "moe_intermediate_size"), "moe_intermediate_size");
  c.shared_inter = as_int(ModelConfig::need_int(t, "shared_expert_intermediate_size"), "shared_expert_intermediate_size");

  c.hc = as_int(ModelConfig::need_int(t, "hc_count"), "hc_count");
  c.hc_lowrank = as_int(ModelConfig::need_int(t, "hc_lowrank"), "hc_lowrank");

  for (const int64_t v : ModelConfig::need_ints(t, "ple_layer_ids")) c.ple_layer_ids.push_back(as_int(v, "ple_layer_ids"));
  c.ngram = as_int(ModelConfig::need_int(t, "ngram_size"), "ngram_size");
  c.heads_per_ngram = as_int(ModelConfig::need_int(t, "heads_per_ngram"), "heads_per_ngram");
  c.ple_dim = as_int(ModelConfig::need_int(t, "ple_embed_dim"), "ple_embed_dim");
  c.ple_conv_kernel = as_int(ModelConfig::need_int(t, "ple_conv_kernel_size"), "ple_conv_kernel_size");
  c.ple_split_parts = as_int(ModelConfig::need_int(t, "split_ngram_parts"), "split_ngram_parts");
  c.ngram_vocab_base = ModelConfig::need_int(t, "ngram_vocab_size_base");
  c.ple_seed = ModelConfig::int_or(t, "seed", 1234);
  c.ple_divisor = ModelConfig::int_or(t, "make_ngram_vocab_size_divisible_by", 1);
  {  // the PLE table's rows (PleTable reads the shards' own dtype; adds the baked int8 rows)
    const std::string pd = ModelConfig::str_or(t, "ple_embedding_dtype", "float8_e4m3fn");
    if (pd != "float8_e4m3fn" && pd != "int8" && pd != "bfloat16") invariant("ple_embedding_dtype (float8_e4m3fn, int8 or bfloat16)");
  }

  c.mtp_layers = as_int(ModelConfig::int_or(t, "mtp_num_hidden_layers", 0), "mtp_num_hidden_layers");

  // A BF16 source (the bake's input) has no quantization_config: its experts are fused BF16 tensors
  // the bake quantises to this engine's format (int4 g128 sym, auto_gptq packing), so that format is
  // assumed; validate() checks the fused tensors instead of the per-expert ones.
  if (const json::Value *q = mc.root().get("quantization_config")) {
    c.w_bits = as_int(ModelConfig::need_int(*q, "bits"), "quantization_config.bits");
    c.w_group = as_int(ModelConfig::need_int(*q, "group_size"), "quantization_config.group_size");
    c.w_sym = ModelConfig::need_bool(*q, "sym");
    c.w_packing = ModelConfig::str_or(*q, "packing_format", "");
    const std::string rot = ModelConfig::str_or(*q, "xenithium_rotation", "");
    if (!rot.empty() && rot != "hadamard-x128-h64") invariant("quantization_config.xenithium_rotation (only hadamard-x128-h64)");
    c.w_rotate = !rot.empty();
    c.act_bits = as_int(ModelConfig::int_or(*q, "act_bits", 16), "quantization_config.act_bits");  // 
  } else {
    c.w_bits = 4, c.w_group = 128, c.w_sym = true, c.w_packing = "auto_round:auto_gptq";
    c.w_unquantized = true;
  }

  c.bos = ModelConfig::int_or(t, "bos_token_id", -1);
  if (const json::Value *e = t.get("eos_token_id")) {
    if (e->is_array())
      for (const auto &v : e->arr) c.eos.push_back(static_cast<int64_t>(v.num));
    else if (e->is_number())
      c.eos.push_back(static_cast<int64_t>(e->num));
  }

  // ---- invariants the engine's kernels and budget rely on ----
  if (static_cast<int>(c.full_attention.size()) != c.layers) invariant("layer_types has a different length than num_hidden_layers");
  for (int l = 0; l < c.layers; ++l)
    if (c.full_attention[l] != ((l + 1) % c.full_interval == 0)) invariant("layer_types is not 'full every full_attention_interval'");
  if (c.q_heads % c.kv_heads) invariant("num_attention_heads not a multiple of num_key_value_heads");
  if (c.lin_v_heads % c.lin_k_heads) invariant("linear value heads not a multiple of key heads");
  if (c.output_gate != "sigmoid") invariant("output_gate_type '" + c.output_gate + "' (only sigmoid is implemented)");
  if (c.w_bits != 4 || c.w_group != 128 || !c.w_sym) invariant("experts must be symmetric int4, group 128");
  if (c.w_packing != "auto_round:auto_gptq") invariant("packing_format '" + c.w_packing + "' (want auto_round:auto_gptq)");
  if (c.hidden % c.w_group || c.moe_inter % c.w_group) invariant("hidden / moe_intermediate_size not multiples of the group");
  if (c.ngram < 2 || c.ple_heads() * c.ple_head_dim() != c.ple_dim) invariant("ple_embed_dim is not heads x head_dim");
  if (c.ple_layer_ids.size() != 1) invariant("exactly one PLE layer is supported");
  if (c.idx_budget % c.idx_compress) invariant("indexer_budget not a multiple of indexer_compress_ratio");
  if (c.tie_embeddings) invariant("tied embeddings are not expected in this family");
  c.ple_layer = c.ple_layer_ids.empty() ? -1 : c.ple_layer_ids[0] - 1;  // config numbers PLE layers from 1
  return c;
}

namespace {

// ---------------------------------------------------------------- names
struct Name {
  std::string_view rest;  // after "model.language_model." [+ "layers.N."]
  int layer = -1;
};

bool eat(std::string_view &s, std::string_view p) {
  if (s.substr(0, p.size()) != p) return false;
  s.remove_prefix(p.size());
  return true;
}

int eat_int(std::string_view &s) {
  int v = 0;
  size_t i = 0;
  while (i < s.size() && s[i] >= '0' && s[i] <= '9') v = v * 10 + (s[i++] - '0');
  if (i == 0) return -1;
  s.remove_prefix(i);
  return v;
}

std::string L(int layer, std::string_view rest) {
  return std::string(kPrefix) + "layers." + std::to_string(layer) + "." + std::string(rest);
}
std::string T(std::string_view rest) { return std::string(kPrefix) + std::string(rest); }

// The decoder layer that holds the PLE block, found in the checkpoint; -1 when absent.
int find_ple_layer(const model::Checkpoint *ck, const Config &c) {
  // config.json numbers PLE layers from 1 (ple_layer_ids [2] = the second decoder layer).
  if (!ck) return c.ple_layer_ids.empty() ? -1 : c.ple_layer_ids[0] - 1;
  for (int l = 0; l < c.layers; ++l)
    if (ck->find(L(l, "ple.key_proj.weight"))) return l;
  return -1;
}

class Qwen38 final : public model::Arch {
 public:
  explicit Qwen38(Config c) : c_(std::move(c)) {}

  std::string_view id() const override { return "qwen38"; }

  std::string describe() const override {
    std::ostringstream o;
    o << "qwen38 (Qwen3.8-Flash-Next, qwen4_exp): " << c_.layers << " layers (" << c_.full_layers() << " full-attention QSA, "
      << c_.layers - c_.full_layers() << " Gated DeltaNet), hidden " << c_.hidden << ", vocab " << c_.vocab << "\n"
      << "  QSA: " << c_.q_heads << " q / " << c_.kv_heads << " kv heads x " << c_.head_dim << ", rotary "
      << static_cast<int>(c_.head_dim * c_.partial_rotary) << " dims, theta " << c_.rope_theta << ", gate " << c_.output_gate
      << "; indexer " << c_.idx_heads << "+" << c_.idx_kv_heads << " heads x " << c_.idx_head_dim << ", 1 key per "
      << c_.idx_compress << " tokens, budget " << c_.idx_budget << " tokens\n"
      << "  GDN: " << c_.lin_k_heads << " k / " << c_.lin_v_heads << " v heads x " << c_.lin_k_dim << "/" << c_.lin_v_dim
      << ", conv " << c_.conv_kernel << "\n"
      << "  MoE: " << c_.experts << " experts top-" << c_.topk << ", inter " << c_.moe_inter << ", shared " << c_.shared_inter
      << "; int" << c_.w_bits << " group " << c_.w_group << (c_.w_sym ? " sym" : " asym") << " (" << c_.w_packing << ")\n"
      << "  hyper-connections: " << c_.hc << " streams, low rank " << c_.hc_lowrank << "\n"
      << "  PLE: config ple_layer_ids [" << c_.ple_layer_ids[0] << "] -> decoder layer " << c_.ple_layer << "; " << c_.ngram
      << "-gram, " << c_.ple_heads() << " heads x " << c_.ple_head_dim() << " = " << c_.ple_dim << ", conv " << c_.ple_conv_kernel
      << ", table split " << c_.ple_split_parts << "\n"
      << "  positions " << c_.max_positions << ", rms eps " << c_.rms_eps << ", MTP layers in config " << c_.mtp_layers;
    return o.str();
  }

  model::TensorRole classify(std::string_view name) const override {
    model::TensorRole r;
    std::string_view s = name;
    if (eat(s, "lm_head.")) return {Component::lm_head, -1, -1};
    if (eat(s, "model.visual.")) return {Component::vision, -1, -1};
    if (eat(s, "mtp.") || eat(s, "model.mtp.")) return {Component::mtp, -1, -1};
    if (!eat(s, kPrefix)) return r;
    if (eat(s, "embed_tokens.")) return {Component::embedding, -1, -1};
    if (eat(s, "hyper_connection_mixer.")) return {Component::hyper_connection, -1, -1};
    if (eat(s, "norm.")) return {Component::norm, -1, -1};
    if (!eat(s, "layers.")) return r;
    r.layer = eat_int(s);
    if (r.layer < 0 || !eat(s, ".")) return {Component::other, -1, -1};
    if (eat(s, "mlp.experts.")) {
      r.comp = Component::routed_experts;
      r.expert = eat_int(s);
      return r;
    }
    if (eat(s, "mlp.shared_expert_gate.") || eat(s, "mlp.shared_expert.")) r.comp = Component::shared_expert;
    else if (eat(s, "mlp.gate.")) r.comp = Component::router;
    else if (eat(s, "attn_hyper_connection.") || eat(s, "mlp_hyper_connection.")) r.comp = Component::hyper_connection;
    else if (eat(s, "linear_attn.")) r.comp = Component::linear_attention;
    else if (eat(s, "self_attn.")) r.comp = Component::attention;
    else if (eat(s, "ple.ple_embedding.ngram_embedding.shard_")) r.comp = Component::ple_table;
    else if (eat(s, "ple.")) r.comp = Component::ple;
    else if (s.find("norm") != std::string_view::npos) r.comp = Component::norm;
    return r;
  }

  std::vector<std::string> validate(const model::Checkpoint &ck, std::vector<std::string> *warnings) const override {
    std::vector<std::string> bad;
    auto want = [&](const std::string &name, DType dt, std::vector<int64_t> shape) {
      const model::TensorInfo *t = ck.find(name);
      if (!t) {
        bad.push_back("missing " + name);
        return;
      }
      if (t->dtype != dt || t->shape != shape) {
        auto fmt = [](const std::vector<int64_t> &v) {
          std::string s = "[";
          for (size_t i = 0; i < v.size(); ++i) s += (i ? ", " : "") + std::to_string(v[i]);
          return s + "]";
        };
        bad.push_back(name + ": " + t->dtype_str + " " + fmt(t->shape) + ", want " + std::string(model::dtype_name(dt)) + " " +
                      fmt(shape));
      }
    };
    const DType bf = DType::bf16;
    const int64_t H = c_.hidden, HC = static_cast<int64_t>(c_.hc) * c_.hidden;
    want(T("embed_tokens.weight"), bf, {c_.vocab, H});
    want("lm_head.weight", bf, {c_.vocab, H});
    want(T("hyper_connection_mixer.hc_norm.weight"), bf, {HC});
    want(T("hyper_connection_mixer.input_mix_weight_down.weight"), bf, {c_.hc_lowrank, HC});
    want(T("hyper_connection_mixer.input_mix_weight_up.weight"), bf, {HC, c_.hc_lowrank});
    const int64_t G = c_.w_group, I = c_.moe_inter, pack = 32 / c_.w_bits;
    for (int l = 0; l < c_.layers; ++l) {
      for (const char *mix : {"attn_hyper_connection.", "mlp_hyper_connection."}) {
        const std::string m = mix;
        want(L(l, m + "block_inject_weight.weight"), bf, {c_.hc, HC});
        want(L(l, m + "hc_norm.weight"), bf, {HC});
        want(L(l, m + "input_mix_weight_down.weight"), bf, {c_.hc_lowrank, HC});
        want(L(l, m + "input_mix_weight_up.weight"), bf, {HC, c_.hc_lowrank});
      }
      if (c_.full_attention[l]) {
        const int64_t QD = static_cast<int64_t>(c_.q_heads) * c_.head_dim, KD = static_cast<int64_t>(c_.kv_heads) * c_.head_dim;
        want(L(l, "self_attn.q_proj.weight"), bf, {2 * QD, H});  // query | gate per head
        want(L(l, "self_attn.k_proj.weight"), bf, {KD, H});
        want(L(l, "self_attn.v_proj.weight"), bf, {KD, H});
        want(L(l, "self_attn.o_proj.weight"), bf, {H, QD});
        want(L(l, "self_attn.q_norm.weight"), bf, {c_.head_dim});
        want(L(l, "self_attn.k_norm.weight"), bf, {c_.head_dim});
        want(L(l, "self_attn.indexer.index_qk_proj.weight"), bf,
             {static_cast<int64_t>(c_.idx_heads + c_.idx_kv_heads) * c_.idx_head_dim, H});
        want(L(l, "self_attn.indexer.q_layernorm.weight"), bf, {c_.idx_head_dim});
        want(L(l, "self_attn.indexer.k_layernorm.weight"), bf, {c_.idx_head_dim});
      } else {
        const int64_t V = static_cast<int64_t>(c_.lin_v_heads) * c_.lin_v_dim;
        want(L(l, "linear_attn.in_proj_qkv.weight"), bf, {c_.lin_qkv(), H});
        want(L(l, "linear_attn.in_proj_z.weight"), bf, {V, H});
        want(L(l, "linear_attn.in_proj_a.weight"), bf, {c_.lin_v_heads, H});
        want(L(l, "linear_attn.in_proj_b.weight"), bf, {c_.lin_v_heads, H});
        want(L(l, "linear_attn.conv1d.weight"), bf, {c_.lin_qkv(), 1, c_.conv_kernel});
        want(L(l, "linear_attn.A_log"), bf, {c_.lin_v_heads});
        want(L(l, "linear_attn.dt_bias"), bf, {c_.lin_v_heads});
        want(L(l, "linear_attn.norm.weight"), bf, {c_.lin_v_dim});
        want(L(l, "linear_attn.out_proj.weight"), bf, {H, V});
      }
      want(L(l, "mlp.gate.weight"), bf, {c_.experts, H});
      want(L(l, "mlp.shared_expert.gate_proj.weight"), bf, {c_.shared_inter, H});
      want(L(l, "mlp.shared_expert.up_proj.weight"), bf, {c_.shared_inter, H});
      want(L(l, "mlp.shared_expert.down_proj.weight"), bf, {H, c_.shared_inter});
      want(L(l, "mlp.shared_expert_gate.weight"), bf, {1, H});
      if (c_.w_unquantized) {  // a BF16 source: the fused experts (the bake reads them)
        want(L(l, "mlp.experts.gate_up_proj"), bf, {c_.experts, 2 * I, H});
        want(L(l, "mlp.experts.down_proj"), bf, {c_.experts, H, I});
      }
      for (int e = 0; !c_.w_unquantized && e < c_.experts; ++e) {
        const std::string p = "mlp.experts." + std::to_string(e) + ".";
        for (const char *proj : {"gate_proj.", "up_proj."}) {
          want(L(l, p + proj + "qweight"), DType::i32, {H / pack, I});
          want(L(l, p + proj + "qzeros"), DType::i32, {H / G, I / pack});
          want(L(l, p + proj + "scales"), DType::f16, {H / G, I});
        }
        want(L(l, p + "down_proj.qweight"), DType::i32, {I / pack, H});
        want(L(l, p + "down_proj.qzeros"), DType::i32, {I / G, H / pack});
        want(L(l, p + "down_proj.scales"), DType::f16, {I / G, H});
      }
    }
    // PLE: the config's 1-based id must name the layer the checkpoint holds it under.
    const int pl = find_ple_layer(&ck, c_);
    if (pl < 0) {
      bad.push_back("no PLE block found under any layer (config ple_layer_ids [" + std::to_string(c_.ple_layer_ids[0]) + "])");
    } else if (pl != c_.ple_layer) {
      bad.push_back("PLE block is under layer " + std::to_string(pl) + " but config ple_layer_ids [" +
                    std::to_string(c_.ple_layer_ids[0]) + "] (1-based) names layer " + std::to_string(c_.ple_layer));
    } else {
      want(L(pl, "ple.key_proj.weight"), bf, {HC, H});
      want(L(pl, "ple.value_proj.weight"), bf, {H, H});
      want(L(pl, "ple.conv1d.weight"), bf, {HC, 1, c_.ple_conv_kernel});
      want(L(pl, "ple.norm_conv.weight"), bf, {HC});
      want(L(pl, "ple.norm_key.weight"), bf, {HC});
      want(L(pl, "ple.norm_query.weight"), bf, {HC});
      want(L(pl, "ple.ple_embedding.layer_multipliers"), DType::i64, {c_.ngram});
      want(L(pl, "ple.ple_embedding.ngram_heads_offsets"), DType::i64, {c_.ple_heads()});
      want(L(pl, "ple.ple_embedding.ngram_heads_vocab_sizes"), DType::i64, {c_.ple_heads()});
      // Row formats (PleTable): F8_E4M3 [rows, hd] + a bf16 weight_scale, U8 [rows, hd + 4] int8 rows
      // (the bake), or BF16 [rows, hd] (a BF16 source); every shard like shard 0.
      const model::TensorInfo *t0 = ck.find(L(pl, "ple.ple_embedding.ngram_embedding.shard_0.weight"));
      const DType pdt = t0 ? t0->dtype : DType::f8_e4m3;
      const int64_t hd = c_.ple_head_dim(), want_w = pdt == DType::u8 ? hd + 4 : hd;
      const char *fmt = pdt == DType::u8 ? "int8 rows" : pdt == DType::bf16 ? "BF16" : "FP8";
      if (pdt == DType::f8_e4m3) want(L(pl, "ple.ple_embedding.ngram_embedding.weight_scale"), bf, {1});
      else if (pdt != DType::u8 && pdt != DType::bf16) bad.push_back("PLE table shard_0: " + t0->dtype_str + " (want F8_E4M3, U8 or BF16)");
      int64_t rows = 0;
      for (int s = 0; s < c_.ple_split_parts; ++s) {
        const std::string n = L(pl, "ple.ple_embedding.ngram_embedding.shard_" + std::to_string(s) + ".weight");
        const model::TensorInfo *t = ck.find(n);
        if (!t) {
          bad.push_back("missing " + n);
          continue;
        }
        if (t->dtype != pdt || t->shape.size() != 2 || t->shape[1] != want_w)
          bad.push_back(n + ": want " + std::string(model::dtype_name(pdt)) + " [rows, " + std::to_string(want_w) + "] like shard 0");
        rows += t->dim(0);
      }
      if (warnings) {
        char buf[200];
        const double per = static_cast<double>(want_w) * (pdt == DType::bf16 ? 2 : 1);
        std::snprintf(buf, sizeof buf, "PLE table: %lld rows x %lld (%s) = %.2f GiB, host/NVMe resident", static_cast<long long>(rows),
                      static_cast<long long>(hd), fmt, static_cast<double>(rows) * per / (1ull << 30));
        warnings->push_back(buf);
      }
    }
    // MTP drafter: config may declare layers the checkpoint does not ship.
    size_t mtp = 0, other = 0, vision = 0;
    std::map<std::string, int> other_names;
    for (const auto &t : ck.tensors()) {
      const auto role = classify(t.name);
      if (role.comp == Component::mtp) ++mtp;
      if (role.comp == Component::vision) ++vision;
      if (role.comp == Component::other) {
        ++other;
        if (other_names.size() < 8) other_names[t.name] = 1;
      }
    }
    if (warnings) {
      if (c_.mtp_layers > 0 && mtp == 0)
        warnings->push_back("config declares " + std::to_string(c_.mtp_layers) +
                            " MTP layer(s) but the checkpoint has no MTP tensors: a drafter must come from another directory");
      if (vision > 0) warnings->push_back(std::to_string(vision) + " vision-tower tensors present (loaded with --vision 1)");
    }
    for (const auto &kv : other_names) bad.push_back("unclassified tensor " + kv.first);
    if (other > other_names.size()) bad.push_back("... " + std::to_string(other - other_names.size()) + " more unclassified");
    return bad;
  }

  model::Budget budget(const model::Checkpoint &ck, const model::DeployParams &p) const override;

  // Placement of every non-expert tensor (TP over the cards, by head):
  //   GDN: in_proj_qkv / conv1d rows = q | k | v sections, each split by head; z, a, b, A_log,
  //        dt_bias split by value head; out_proj split along k (value heads); norm replicated.
  //   QSA: q_proj rows are per head (query | gate of head h, 2 x head_dim) -> contiguous split,
  //        k/v split by kv head and appended to it (one GEMM); o_proj split along k; q/k norms
  //        and the indexer (bf16) replicated.
  //   MoE: router (+ shared gate row) and shared expert (gate | up, down) replicated; each card
  // computes half of the shared expert's intermediate columns.
  // hyper-connections: replicated (a split would need a cross-card exchange per
  //        mixer, which costs more than the weight reads it saves). Per layer the injection rows
  //        follow the down rows in one matrix (320 + 4 rows, padded to 336), one GEMM.
  //   lm_head: vocab rows split; embedding: pinned host (a step gathers a few rows).
  //   PLE block weights: replicated, key_proj | value_proj one matrix; its n-gram table stays on
  //        NVMe (not in this plan: arch/qwen38/ple_table reads it per step).
  std::vector<model::DenseSpec> dense_plan(const model::Checkpoint &ck, const model::DeployParams &p) const override {
    using S = model::DenseSpec;
    std::vector<S> out;
    const S::Store mat = p.dense_bytes == 1 ? S::Store::i8 : S::Store::bf16;
    auto add = [&](std::string name, S::Split split, S::Store store, Component comp, int layer, std::vector<int64_t> sections = {}) {
      if (!ck.find(name)) throw std::runtime_error("qwen38 dense plan: missing " + name);
      S sp;
      sp.name = std::move(name);
      sp.split = split;
      sp.store = store;
      sp.sections = std::move(sections);
      sp.comp = comp;
      sp.layer = layer;
      out.push_back(std::move(sp));
    };
    add(T("embed_tokens.weight"), S::Split::host, S::Store::raw, Component::embedding, -1);
    add("lm_head.weight", S::Split::rows, mat, Component::lm_head, -1);
    for (const char *t : {"hc_norm.weight", "input_mix_weight_down.weight", "input_mix_weight_up.weight"})
      add(T(std::string("hyper_connection_mixer.") + t), S::Split::replicate,
          std::string(t) == "hc_norm.weight" ? S::Store::raw : mat, Component::hyper_connection, -1);
    const int64_t kq = static_cast<int64_t>(c_.lin_k_heads) * c_.lin_k_dim, vq = static_cast<int64_t>(c_.lin_v_heads) * c_.lin_v_dim;
    for (int l = 0; l < c_.layers; ++l) {
      for (const char *mix : {"attn_hyper_connection.", "mlp_hyper_connection."}) {
        const std::string m = mix;
        add(L(l, m + "hc_norm.weight"), S::Split::replicate, S::Store::raw, Component::hyper_connection, l);
        add(L(l, m + "input_mix_weight_down.weight"), S::Split::replicate, mat, Component::hyper_connection, l);
        if (!ck.find(L(l, m + "block_inject_weight.weight"))) throw std::runtime_error("qwen38 dense plan: missing block_inject_weight");
        out.back().append = {L(l, m + "block_inject_weight.weight")};
        add(L(l, m + "input_mix_weight_up.weight"), S::Split::replicate, mat, Component::hyper_connection, l);
      }
      if (c_.full_attention[l]) {
        const auto A = Component::attention;
        // q | gate per head with this card's k and v rows appended (int8, one GEMM;).
        add(L(l, "self_attn.q_proj.weight"), S::Split::rows, mat, A, l);
        for (const char *t : {"self_attn.k_proj.weight", "self_attn.v_proj.weight"})
          if (!ck.find(L(l, t))) throw std::runtime_error("qwen38 dense plan: missing " + L(l, t));
        out.back().append = {L(l, "self_attn.k_proj.weight"), L(l, "self_attn.v_proj.weight")};
        out.back().append_heads = c_.kv_heads;  // TP4 = a kv head on two cards (each takes it whole)
        if (kv_by_tokens(p)) {  // the KV split by tokens: + the partner's q heads (q of each q | gate pair)
          out.back().peer_block = 2 * c_.head_dim;
          out.back().peer_keep = c_.head_dim;
        }
        add(L(l, "self_attn.o_proj.weight"), S::Split::cols, mat, A, l);
        add(L(l, "self_attn.q_norm.weight"), S::Split::replicate, S::Store::raw, A, l);
        add(L(l, "self_attn.k_norm.weight"), S::Split::replicate, S::Store::raw, A, l);
        add(L(l, "self_attn.indexer.index_qk_proj.weight"), S::Split::replicate, S::Store::bf16, A, l);
        add(L(l, "self_attn.indexer.q_layernorm.weight"), S::Split::replicate, S::Store::raw, A, l);
        add(L(l, "self_attn.indexer.k_layernorm.weight"), S::Split::replicate, S::Store::raw, A, l);
      } else {
        const auto G = Component::linear_attention;
        // q | k | v (by head) with z appended (int8, one GEMM); a and b raw bf16 rows (two dots
        // per head inside the GDN core kernel).
        add(L(l, "linear_attn.in_proj_qkv.weight"), S::Split::rows, mat, G, l, {kq, kq, vq});
        out.back().append = {L(l, "linear_attn.in_proj_z.weight")};
        add(L(l, "linear_attn.conv1d.weight"), S::Split::rows, S::Store::raw, G, l, {kq, kq, vq});
        add(L(l, "linear_attn.in_proj_a.weight"), S::Split::rows, S::Store::raw, G, l);
        add(L(l, "linear_attn.in_proj_b.weight"), S::Split::rows, S::Store::raw, G, l);
        add(L(l, "linear_attn.A_log"), S::Split::rows, S::Store::raw, G, l);
        add(L(l, "linear_attn.dt_bias"), S::Split::rows, S::Store::raw, G, l);
        add(L(l, "linear_attn.norm.weight"), S::Split::replicate, S::Store::raw, G, l);
        add(L(l, "linear_attn.out_proj.weight"), S::Split::cols, mat, G, l);
      }
      // Router with the shared-expert gate as one more row (one GEMM, bf16: routing must
      // match on every card); shared gate | up as one matrix.
      add(L(l, "mlp.gate.weight"), S::Split::replicate, S::Store::bf16, Component::router, l);
      out.back().append = {L(l, "mlp.shared_expert_gate.weight")};
      add(L(l, "mlp.shared_expert.gate_proj.weight"), S::Split::replicate, mat, Component::shared_expert, l);
      out.back().append = {L(l, "mlp.shared_expert.up_proj.weight")};
      add(L(l, "mlp.shared_expert.down_proj.weight"), S::Split::replicate, mat, Component::shared_expert, l);
    }
    const int pl = c_.ple_layer;
    // key_proj with value_proj appended (one GEMM, [S*H + H][E];).
    add(L(pl, "ple.key_proj.weight"), S::Split::replicate, mat, Component::ple, pl);
    if (!ck.find(L(pl, "ple.value_proj.weight"))) throw std::runtime_error("qwen38 dense plan: missing " + L(pl, "ple.value_proj.weight"));
    out.back().append = {L(pl, "ple.value_proj.weight")};
    for (const char *t : {"ple.conv1d.weight", "ple.norm_conv.weight", "ple.norm_key.weight", "ple.norm_query.weight",
                          "ple.ple_embedding.layer_multipliers", "ple.ple_embedding.ngram_heads_offsets",
                          "ple.ple_embedding.ngram_heads_vocab_sizes", "ple.ple_embedding.ngram_embedding.weight_scale"})
      // weight_scale: only the FP8 table has one (int8-row and BF16 tables do not;)
      if (std::string(t).find("weight_scale") == std::string::npos || ck.find(L(pl, t))) add(L(pl, t), S::Split::replicate, S::Store::raw, Component::ple, pl);
    return out;
  }

  Config c_;
  // The KV split by tokens applies (DeployParams::KvShard::tokens with fewer kv heads than cards: qwen38 TP4).
  bool kv_by_tokens(const model::DeployParams &p) const { return p.kv_shard == model::DeployParams::KvShard::tokens && c_.kv_heads < p.cards; }
};

// ---------------------------------------------------------------- budget
// Per-card bytes of the deployment the engine is built for: TP over the cards for the
// dense trunk, attention heads and vocab; EP for the routed experts; router, indexer and
// PLE weights replicated (every card must compute bit-identical routing / selection);
// the shared expert replicated so a step can split it by that step's routing.
model::Budget Qwen38::budget(const model::Checkpoint &ck, const model::DeployParams &p) const {
  // fewer kv heads than cards = each kv head on cards / kv_heads cards (TP4: 2 kv heads, 4 cards)
  if (p.cards < 1 || (c_.kv_heads % p.cards && p.cards % c_.kv_heads) || c_.q_heads % p.cards || c_.lin_v_heads % p.cards ||
      c_.experts % p.cards || c_.vocab % p.cards)
    throw std::runtime_error("qwen38 budget: the head/expert/vocab counts do not split over " + std::to_string(p.cards) + " cards");
  if (p.dense_bytes != 1 && p.dense_bytes != 2) throw std::runtime_error("dense_bytes must be 1 or 2");
  if (p.kv_bytes != 1 && p.kv_bytes != 2) throw std::runtime_error("kv_bytes must be 1 or 2");
  if (p.kv_v4 && p.kv_bytes != 1) throw std::runtime_error("kv_v4 (4-bit V) needs kv_bytes 1 (int8 K)");

  const double C = p.cards, db = p.dense_bytes;
  const double KVC = std::min(C, static_cast<double>(c_.kv_heads));  // the kv heads' split (TP4: 2, each head on two cards)
  // A dense [rows][cols] weight split over `split` cards at the dense precision;
  // 8-bit adds one fp32 scale per output row.
  auto dense = [&](double rows, double cols, double split) { return rows * cols * db / split + (db == 1 ? rows / split * 4 : 0); };
  auto bf16 = [&](double rows, double cols, double split) { return rows * cols * 2 / split; };

  model::Budget b;
  const double H = c_.hidden, HC = static_cast<double>(c_.hc) * c_.hidden;
  const int full = c_.full_layers(), lin = c_.layers - full;
  auto row = [&](std::string name, double resident, double step, std::string note) {
    b.rows.push_back({std::move(name), resident, step, std::move(note)});
  };

  // embedding: one table row per token, so the table may sit in pinned host memory.
  {
    const double table = bf16(static_cast<double>(c_.vocab), H, 1);
    if (p.embed_on_host) row("embedding", 0, 0, "pinned host table, a step gathers its rows over PCIe");
    else row("embedding", table, 0, "replicated table");
  }
  {
    const double lm = dense(static_cast<double>(c_.vocab), H, C);
    row("lm_head", lm, lm, "vocab-parallel, each card a contiguous half of the rows");
  }
  {
    const double QD = static_cast<double>(c_.q_heads) * c_.head_dim, KD = static_cast<double>(c_.kv_heads) * c_.head_dim;
    const double per = dense(2 * QD, H, C) + dense(KD, H, KVC) * 2 + dense(H, QD, C) +
                       bf16((c_.idx_heads + c_.idx_kv_heads) * static_cast<double>(c_.idx_head_dim), H, 1);
    row("attention (QSA) x" + std::to_string(full), per * full, per * full, "heads split; indexer projection replicated, bf16");
  }
  {
    const double V = static_cast<double>(c_.lin_v_heads) * c_.lin_v_dim;
    const double per = dense(c_.lin_qkv(), H, C) + dense(V, H, C) + bf16(2.0 * c_.lin_v_heads, H, C) + dense(H, V, C) +
                       bf16(c_.lin_qkv(), c_.conv_kernel, C);
    row("linear attention (GDN) x" + std::to_string(lin), per * lin, per * lin, "heads split; a/b projections bf16");
  }
  {
    // Replicated (a split needs a cross-card exchange per mixer, which costs more than the
    // reads it saves). The injection rows ride in the down matrix (int8).
    const double mixer = dense(c_.hc_lowrank, HC, 1) + dense(HC, c_.hc_lowrank, 1);
    const double inject = dense(c_.hc, HC, 1);
    const double total = (2.0 * c_.layers) * (mixer + inject) + mixer;
    row("hyper-connections x" + std::to_string(2 * c_.layers + 1), total, total, "replicated (each card runs every mixer)");
  }
  {
    const double router = bf16(c_.experts, H, 1) * c_.layers;
    row("router x" + std::to_string(c_.layers), router, router, "replicated, bf16 (routing must match on every card)");
  }
  {
    const double shared = (dense(c_.shared_inter, H, 1) * 2 + dense(H, c_.shared_inter, 1) + bf16(1, H, 1)) * c_.layers;
    row("shared expert x" + std::to_string(c_.layers), shared, shared / C, "replicated; each step splits it between the cards");
  }
  {
    const double ple = dense(HC, H, 1) + dense(H, H, 1) + bf16(HC, c_.ple_conv_kernel, 1);
    row("PLE block (layer " + std::to_string(c_.ple_layer) + ")", ple, ple,
        "replicated; its n-gram table is read from NVMe (" + std::to_string(c_.ple_heads()) + " rows per token)");
  }

  // KV per token per card per slot. K and V of every KV head (+ one fp32 scale per token and
  // head for 8-bit KV), plus the indexer's compressed key (bf16, replicated: every card selects). Split over the cards:
  //   heads:  each card stores all tokens of its kv heads; with fewer kv heads than cards (TP4) a kv head's cards each
  //           store all of them (KV per card as at kv_heads cards).
  //   tokens: (fewer kv heads than cards) a kv head's cards split its tokens: each stores 1 / C of the K / V.
  const bool by_tokens = kv_by_tokens(p);
  const int kv_layers = full + (p.drafter ? 1 : 0);
  // K + V (int8-v4: V at half a byte per element) + the two fp32 scales of an 8-bit KV
  // (int8-v4: 4-bit V + one bf16 scale per 32 dims)
  const double kv_all = (p.kv_v4 ? 1.5 : 2.0) * c_.kv_heads * c_.head_dim * p.kv_bytes +
                        (p.kv_bytes == 1 ? (4.0 + (p.kv_v4 ? 2.0 * kernels::kQsaV4Scales : 4.0)) * c_.kv_heads : 0);
  const double idx_all = static_cast<double>(c_.idx_head_dim) * 2 / c_.idx_compress;
  const double kv_tok_layer = by_tokens ? kv_all / C : kv_all / KVC;
  const double idx_tok_layer = idx_all;
  b.kv_bytes_per_token = kv_layers * (kv_tok_layer + idx_tok_layer);
  {
    const double ctx = static_cast<double>(p.context);
    const double resident = b.kv_bytes_per_token * ctx * p.seqs;
    // Sparse attention: a query reads the selected blocks' K/V (at most the budget plus the
    // open block) and scores every compressed key once. The split by tokens halves the K / V read.
    const double attended = std::min(ctx, static_cast<double>(c_.idx_budget + c_.idx_compress - 1));
    const double step = kv_layers * (attended * kv_tok_layer + ctx * idx_tok_layer) * p.seqs;
    row("KV cache (" + std::to_string(p.context) + " x " + std::to_string(p.seqs) + " slot)", resident, step,
        std::to_string(kv_layers) + " layers" + (p.drafter ? " incl. the drafter's" : "") + ", " +
            (p.kv_v4 ? "8-bit K, 4-bit V + scales" : p.kv_bytes == 1 ? "8-bit + scales" : "bf16") + ", split by " + (by_tokens ? "token" : "KV head") +
            "; a step reads <= " + std::to_string(static_cast<int>(attended)) + " tokens + all compressed keys");
  }
  if (by_tokens) {
    // Every card attends its tokens with all q heads of its kv head: the partner's q heads (no gates) are replicated
    // rows of the card's q_proj slice (an exchange of q per layer costs more, round 279's micro). o_proj stays split by
    // its own heads' columns after the pair's merge: one partial-softmax exchange per attention layer per step; the
    // selection needs none (the indexer is replicated).
    const double QD = static_cast<double>(c_.q_heads) * c_.head_dim;
    const double extra = dense(QD, H, C) * kv_layers;
    row("q_proj: the partner's q heads (KV split by tokens)", extra, extra, "each card attends with every q head of its kv head");
    b.notes.push_back("KV split by tokens: one cross-card exchange per attention layer per step (the pair's softmax merge)");
  }

  // Routed experts as the card stores them: int4 codes + fp16 group scales. The checkpoint's
  // qzeros are not stored (symmetric, zero fixed by the format) - the loader checks them.
  {
    const std::string e0 = L(0, "mlp.experts.0.");
    double eb = 0;
    for (const char *proj : {"gate_proj.", "up_proj.", "down_proj."})
      eb += static_cast<double>(ck.at(e0 + proj + "qweight").bytes + ck.at(e0 + proj + "scales").bytes);
    b.expert_bytes = eb;
  }
  b.experts_per_card = static_cast<int64_t>(c_.layers) * c_.experts / p.cards;
  b.unique_experts = p.unique_experts > 0
                         ? p.unique_experts
                         : c_.experts * (1.0 - std::pow(1.0 - static_cast<double>(c_.topk) / c_.experts, p.rows * p.seqs));
  b.expert_step_bytes = b.unique_experts / C * c_.layers * b.expert_bytes;

  double resident = 0, step = 0;
  for (const auto &r : b.rows) {
    resident += r.resident_bytes;
    step += r.step_bytes;
  }
  b.resident_non_expert = resident;
  const double vram = p.vram_gib * (1ull << 30), reserve = p.reserve_gib * (1ull << 30);
  const double room = vram - reserve - resident;
  b.hot_experts_fit = room <= 0 ? 0 : std::min<int64_t>(b.experts_per_card, static_cast<int64_t>(room / b.expert_bytes));
  b.floor_ms = (step + b.expert_step_bytes) / (p.bandwidth_gbs * 1e9) * 1e3;
  if (p.unique_experts <= 0)
    b.notes.push_back("unique experts per layer per step from independent routing (" + std::to_string(p.rows * p.seqs) +
                      " rows); real routing is correlated and touches fewer - pass a measured value to replace it");
  if (b.hot_experts_fit < b.experts_per_card)
    b.notes.push_back(std::to_string(b.experts_per_card - b.hot_experts_fit) +
                      " experts per card stay cold (host memory) at this context");
  b.notes.push_back("not modelled: activation scratch, recorded graphs, mailboxes (covered by the reserve)");
  return b;
}

}  // namespace

model::ArchEntry entry() {
  return {
      "qwen38",
      "Qwen3.8-Flash-Next (qwen4_exp): hybrid GDN + sparse-indexed GQA, 512-expert int4 MoE, hyper-connections, PLE",
      "Flash-Next MoE",
      "qwen4_exp",
      [](const ModelConfig &mc) {
        if (mc.model_type() == "qwen4_exp") return true;
        for (const auto &a : mc.architectures())
          if (a.rfind("Qwen4Exp", 0) == 0) return true;
        return false;
      },
      [](const ModelConfig &mc) -> std::unique_ptr<model::Arch> {
        Config c = Config::parse(mc);  // ple_layer from config; validate() checks it against the checkpoint
        return std::make_unique<Qwen38>(std::move(c));
      },
  };
}

}  // namespace xenithium::arch::qwen38
