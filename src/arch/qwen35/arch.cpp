#include "xenithium/arch/qwen35/arch.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace xenithium::arch::qwen35 {

using model::Component;
using model::DType;
using model::ModelConfig;

int Config::full_layers() const {
  return static_cast<int>(std::count(full_attention.begin(), full_attention.end(), true));
}

namespace {

[[noreturn]] void invariant(const std::string &what) { throw std::runtime_error("qwen35 config: " + what); }

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
  c.attn_output_gate = ModelConfig::need_bool(t, "attn_output_gate");
  c.partial_rotary = ModelConfig::need_num(t, "partial_rotary_factor");
  const json::Value &rope = ModelConfig::need(t, "rope_parameters");
  c.rope_theta = ModelConfig::need_num(rope, "rope_theta");
  c.rope_type = ModelConfig::str_or(rope, "rope_type", "default");
  if (rope.get("mrope_section"))
    for (const int64_t v : ModelConfig::need_ints(rope, "mrope_section")) c.mrope_section.push_back(as_int(v, "mrope_section"));
  if (const json::Value *mi = rope.get("mrope_interleaved")) c.mrope_interleaved = mi->is_bool() && mi->b;
  if (const json::Value *bias = t.get("attention_bias"))
    if (bias->is_bool() && bias->b) invariant("attention_bias true is not implemented");

  c.lin_k_heads = as_int(ModelConfig::need_int(t, "linear_num_key_heads"), "linear_num_key_heads");
  c.lin_v_heads = as_int(ModelConfig::need_int(t, "linear_num_value_heads"), "linear_num_value_heads");
  c.lin_k_dim = as_int(ModelConfig::need_int(t, "linear_key_head_dim"), "linear_key_head_dim");
  c.lin_v_dim = as_int(ModelConfig::need_int(t, "linear_value_head_dim"), "linear_value_head_dim");
  c.conv_kernel = as_int(ModelConfig::need_int(t, "linear_conv_kernel_dim"), "linear_conv_kernel_dim");
  c.output_gate = ModelConfig::need_str(t, "output_gate_type");

  c.inter = as_int(ModelConfig::need_int(t, "intermediate_size"), "intermediate_size");
  c.hidden_act = ModelConfig::need_str(t, "hidden_act");

  c.mtp_layers = as_int(ModelConfig::int_or(t, "mtp_num_hidden_layers", 0), "mtp_num_hidden_layers");
  if (const json::Value *d = t.get("mtp_use_dedicated_embeddings")) c.mtp_dedicated_embeddings = d->is_bool() && d->b;

  // quantization_config: compressed-tensors (config_groups.<one group>.weights), auto-round (top-level keys) or gptq
  // (gptqmodel's GPTQ: top-level bits / group_size / sym / format, the tensors in the auto_gptq orientation)
  const json::Value *q = mc.root().get("quantization_config");
  if (!q) q = t.get("quantization_config");
  if (!q) invariant("quantization_config missing (only int4 W4A16 checkpoints are served)");
  c.w_method = ModelConfig::str_or(*q, "quant_method", "");
  if (c.w_method == "compressed-tensors") {
    c.w_format = ModelConfig::str_or(*q, "format", "");
    const json::Value &groups = ModelConfig::need(*q, "config_groups");
    if (!groups.is_object() || groups.obj.size() != 1) invariant("quantization_config.config_groups (want exactly one group)");
    const json::Value &w = ModelConfig::need(groups.obj[0].second, "weights");
    c.w_bits = as_int(ModelConfig::need_int(w, "num_bits"), "weights.num_bits");
    c.w_group = as_int(ModelConfig::need_int(w, "group_size"), "weights.group_size");
    c.w_sym = ModelConfig::need_bool(w, "symmetric");
    if (ModelConfig::need_str(w, "type") != "int" || ModelConfig::need_str(w, "strategy") != "group") invariant("weights.type / strategy (want int, group)");
    if (const json::Value *ia = groups.obj[0].second.get("input_activations"); ia && !ia->is_null()) invariant("input_activations set: not a weight-only scheme");
    if (c.w_format != "pack-quantized") invariant("quantization_config.format '" + c.w_format + "' (want pack-quantized)");
  } else if (c.w_method == "auto-round") {
    c.w_format = ModelConfig::str_or(*q, "packing_format", "");
    c.w_bits = as_int(ModelConfig::need_int(*q, "bits"), "bits");
    c.w_group = as_int(ModelConfig::need_int(*q, "group_size"), "group_size");
    c.w_sym = ModelConfig::need_bool(*q, "sym");
    if (ModelConfig::str_or(*q, "data_type", "int") != "int") invariant("quantization_config.data_type (want int)");
    if (c.w_format != "auto_round:auto_gptq") invariant("quantization_config.packing_format '" + c.w_format + "' (want auto_round:auto_gptq)");
    // extra_config may exempt modules from quantisation: the engine expects exactly in_proj_a / in_proj_b, mtp.*, visual.*
    // (bf16 in every served checkpoint); validate() checks the tensors themselves.
  } else if (c.w_method == "gptq") {
    // gptqmodel writes format "gptq"; a config without the key (the older AutoGPTQ layout) means the same packing (v1
    // zero points, which validate() checks)
    c.w_format = ModelConfig::str_or(*q, "format", "gptq");
    c.w_bits = as_int(ModelConfig::need_int(*q, "bits"), "bits");
    c.w_group = as_int(ModelConfig::need_int(*q, "group_size"), "group_size");
    c.w_sym = ModelConfig::need_bool(*q, "sym");
    if (ModelConfig::str_or(*q, "pack_dtype", "int32") != "int32") invariant("quantization_config.pack_dtype (want int32)");
    if (c.w_format.empty()) c.w_format = "gptq";  // gptqmodel may omit format when quant_method is already gptq
    if (c.w_format != "gptq") invariant("quantization_config.format '" + c.w_format + "' (want gptq)");
    if (const json::Value *act = q->get("desc_act"); act && act->is_bool() && act->b) invariant("quantization_config.desc_act true (act order) is not implemented");
    // The tensors are in the auto_gptq orientation (qweight [k/8][n], scales [k/128][n], qzeros [k/128][n/8]);
    // validate() checks them.
  } else {
    invariant("quantization_config.quant_method '" + c.w_method + "' (want compressed-tensors, auto-round or gptq)");
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
  if (c.output_gate != "swish") invariant("output_gate_type '" + c.output_gate + "' (only swish is implemented)");
  if (c.hidden_act != "silu") invariant("hidden_act '" + c.hidden_act + "' (only silu is implemented)");
  if (c.rope_type != "default") invariant("rope_type '" + c.rope_type + "' (only default is implemented)");
  {
    const int rot = static_cast<int>(std::lround(c.head_dim * c.partial_rotary));
    if (rot <= 0 || rot % 2 || std::fabs(c.head_dim * c.partial_rotary - rot) > 1e-9) invariant("head_dim x partial_rotary_factor is not an even integer");
    if (!c.mrope_section.empty()) {
      int sum = 0;
      for (const int s : c.mrope_section) sum += s;
      if (2 * sum != rot) invariant("mrope_section does not sum to half the rotary dims");
    }
  }
  if (c.w_bits != 4 || c.w_group != 128 || !c.w_sym) invariant("weights must be symmetric int4, group 128");
  if (c.hidden % c.w_group || c.inter % c.w_group || c.lin_qkv() % c.w_group || (c.q_heads * c.head_dim) % c.w_group)
    invariant("a Linear's input dim is not a multiple of the group");
  if (c.tie_embeddings) invariant("tied embeddings are not expected in this family");
  return c;
}

namespace {

// ---------------------------------------------------------------- names
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

class Qwen35 final : public model::Arch {
 public:
  explicit Qwen35(Config c) : c_(std::move(c)) {}

  std::string_view id() const override { return "qwen35"; }

  std::string describe() const override {
    std::ostringstream o;
    o << "qwen35 (Qwen3.8-27B dense, HF qwen3_5): " << c_.layers << " layers (" << c_.full_layers() << " gated full attention, "
      << c_.layers - c_.full_layers() << " Gated DeltaNet), hidden " << c_.hidden << ", vocab " << c_.vocab << "\n"
      << "  attention: " << c_.q_heads << " q / " << c_.kv_heads << " kv heads x " << c_.head_dim << ", rotary "
      << static_cast<int>(std::lround(c_.head_dim * c_.partial_rotary)) << " dims, theta " << c_.rope_theta
      << (c_.mrope_section.empty() ? "" : ", mrope sections") << (c_.mrope_interleaved ? " interleaved" : "")
      << ", output gate " << (c_.attn_output_gate ? "yes (q | gate per head)" : "no") << ", q/k norm\n"
      << "  GDN: " << c_.lin_k_heads << " k / " << c_.lin_v_heads << " v heads x " << c_.lin_k_dim << "/" << c_.lin_v_dim
      << ", conv " << c_.conv_kernel << ", gate " << c_.output_gate << "\n"
      << "  MLP: dense SwiGLU, inter " << c_.inter << "; Linears int" << c_.w_bits << " group " << c_.w_group
      << (c_.w_sym ? " sym" : " asym") << " (" << c_.w_method << " " << c_.w_format
      << "; in_proj_a/b and lm_head bf16)\n"
      << "  positions " << c_.max_positions << ", rms eps " << c_.rms_eps << ", MTP layers in config " << c_.mtp_layers
      << (c_.mtp_dedicated_embeddings ? " (own embeddings)" : " (shared embeddings)");
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
    if (eat(s, "norm.")) return {Component::norm, -1, -1};
    if (!eat(s, "layers.")) return r;
    r.layer = eat_int(s);
    if (r.layer < 0 || !eat(s, ".")) return {Component::other, -1, -1};
    if (eat(s, "mlp.")) r.comp = Component::dense_mlp;
    else if (eat(s, "linear_attn.")) r.comp = Component::linear_attention;
    else if (eat(s, "self_attn.")) r.comp = Component::attention;
    else if (s.find("norm") != std::string_view::npos) r.comp = Component::norm;
    return r;
  }

  std::vector<std::string> validate(const model::Checkpoint &ck, std::vector<std::string> *warnings) const override {
    std::vector<std::string> bad;
    auto fmt = [](const std::vector<int64_t> &v) {
      std::string s = "[";
      for (size_t i = 0; i < v.size(); ++i) s += (i ? ", " : "") + std::to_string(v[i]);
      return s + "]";
    };
    auto want = [&](const std::string &name, DType dt, std::vector<int64_t> shape) {
      const model::TensorInfo *t = ck.find(name);
      if (!t) {
        bad.push_back("missing " + name);
        return;
      }
      if (t->dtype != dt || t->shape != shape)
        bad.push_back(name + ": " + t->dtype_str + " " + fmt(t->shape) + ", want " + std::string(model::dtype_name(dt)) + " " + fmt(shape));
    };
    // A dense tensor the engine reads as bf16: a checkpoint may store it f16 (converted at load).
    auto want2 = [&](const std::string &name, std::vector<int64_t> shape) {
      const model::TensorInfo *t = ck.find(name);
      if (!t) {
        bad.push_back("missing " + name);
        return;
      }
      if ((t->dtype != DType::bf16 && t->dtype != DType::f16) || t->shape != shape)
        bad.push_back(name + ": " + t->dtype_str + " " + fmt(t->shape) + ", want BF16 / F16 " + fmt(shape));
    };
    const int64_t pack = 32 / c_.w_bits, G = c_.w_group;
    // An int4 Linear of n output rows and k inputs, in the checkpoint's layout.
    auto want_w4 = [&](const std::string &module, int64_t n, int64_t k) {
      if (c_.w_gptq()) {
        want(module + ".qweight", DType::i32, {k / pack, n});
        want(module + ".scales", DType::f16, {k / G, n});
        want(module + ".qzeros", DType::i32, {k / G, n / pack});
      } else {
        want(module + ".weight_packed", DType::i32, {n, k / pack});
        want(module + ".weight_scale", DType::f16, {n, k / G});
        want(module + ".weight_shape", DType::i64, {2});
      }
    };
    const int64_t H = c_.hidden, I = c_.inter;
    want2(T("embed_tokens.weight"), {c_.vocab, H});
    // The head: a bf16 / f16 weight, or an int4-only head (the checkpoint's lm_head.qweight, no bf16 rows)
    if (ck.find("lm_head.qweight"))
      want_w4("lm_head", c_.vocab, H);
    else
      want2("lm_head.weight", {c_.vocab, H});
    want2(T("norm.weight"), {H});
    for (int l = 0; l < c_.layers; ++l) {
      want2(L(l, "input_layernorm.weight"), {H});
      want2(L(l, "post_attention_layernorm.weight"), {H});
      if (c_.full_attention[l]) {
        const int64_t QD = static_cast<int64_t>(c_.q_heads) * c_.head_dim, KD = static_cast<int64_t>(c_.kv_heads) * c_.head_dim;
        want_w4(L(l, "self_attn.q_proj"), c_.q_rows(), H);  // query | gate per head
        want_w4(L(l, "self_attn.k_proj"), KD, H);
        want_w4(L(l, "self_attn.v_proj"), KD, H);
        want_w4(L(l, "self_attn.o_proj"), H, QD);
        want2(L(l, "self_attn.q_norm.weight"), {c_.head_dim});
        want2(L(l, "self_attn.k_norm.weight"), {c_.head_dim});
      } else {
        const int64_t V = static_cast<int64_t>(c_.lin_v_heads) * c_.lin_v_dim;
        want_w4(L(l, "linear_attn.in_proj_qkv"), c_.lin_qkv(), H);
        want_w4(L(l, "linear_attn.in_proj_z"), V, H);
        want2(L(l, "linear_attn.in_proj_a.weight"), {c_.lin_v_heads, H});
        want2(L(l, "linear_attn.in_proj_b.weight"), {c_.lin_v_heads, H});
        want2(L(l, "linear_attn.conv1d.weight"), {c_.lin_qkv(), 1, c_.conv_kernel});
        want2(L(l, "linear_attn.A_log"), {c_.lin_v_heads});
        want2(L(l, "linear_attn.dt_bias"), {c_.lin_v_heads});
        want2(L(l, "linear_attn.norm.weight"), {c_.lin_v_dim});
        want_w4(L(l, "linear_attn.out_proj"), H, V);
      }
      want_w4(L(l, "mlp.gate_proj"), I, H);
      want_w4(L(l, "mlp.up_proj"), I, H);
      want_w4(L(l, "mlp.down_proj"), H, I);
    }
    // MTP drafter: one layer with its own norms, a single fc over
    // [norm(embedding) | norm(hidden)], and no embedding / lm_head of its own when shared. Its matrices are
    // bf16 / f16 (stored at the dense precision) or a baked int4 GPTQ (dequantised at load).
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
    const bool mtp_layer = ck.find("mtp.layers.0.input_layernorm.weight") != nullptr;
    if (mtp > 0 && !mtp_layer) {  // an export that kept only the fc / norms: no drafter from this directory
      if (warnings) warnings->push_back("MTP tensors present but no mtp.layers.0.* layer: the MTP drafter cannot be served from this checkpoint");
      mtp = 0;
    }
    if (mtp > 0) {
      const int64_t QD = static_cast<int64_t>(c_.q_heads) * c_.head_dim, KD = static_cast<int64_t>(c_.kv_heads) * c_.head_dim;
      auto want_mat = [&](const std::string &module, int64_t n, int64_t k) {
        if (ck.find(module + ".qweight"))
          want_w4(module, n, k);
        else
          want2(module + ".weight", {n, k});
      };
      want_mat("mtp.fc", H, 2 * H);
      want2("mtp.pre_fc_norm_embedding.weight", {H});
      want2("mtp.pre_fc_norm_hidden.weight", {H});
      want2("mtp.norm.weight", {H});
      for (int l = 0; l < std::max(c_.mtp_layers, 1); ++l) {
        const std::string P = "mtp.layers." + std::to_string(l) + ".";
        want2(P + "input_layernorm.weight", {H});
        want2(P + "post_attention_layernorm.weight", {H});
        want_mat(P + "self_attn.q_proj", c_.q_rows(), H);
        want_mat(P + "self_attn.k_proj", KD, H);
        want_mat(P + "self_attn.v_proj", KD, H);
        want_mat(P + "self_attn.o_proj", H, QD);
        want2(P + "self_attn.q_norm.weight", {c_.head_dim});
        want2(P + "self_attn.k_norm.weight", {c_.head_dim});
        want_mat(P + "mlp.gate_proj", I, H);
        want_mat(P + "mlp.up_proj", I, H);
        want_mat(P + "mlp.down_proj", H, I);
      }
      if (c_.mtp_dedicated_embeddings) {
        want2("mtp.embed_tokens.weight", {c_.vocab, H});
        if (ck.find("mtp.lm_head.qweight"))
          want_w4("mtp.lm_head", c_.vocab, H);
        else
          want2("mtp.lm_head.weight", {c_.vocab, H});
      }
    }
    if (warnings) {
      if (c_.mtp_layers > 0 && mtp == 0)
        warnings->push_back("config declares " + std::to_string(c_.mtp_layers) +
                            " MTP layer(s) but the checkpoint has no MTP tensors: a drafter must come from another directory");
      if (mtp > 0) warnings->push_back(std::to_string(mtp) + " MTP drafter tensors present (one layer, shared embeddings / lm_head)");
      if (vision > 0) warnings->push_back(std::to_string(vision) + " vision-tower tensors present (loaded with --vision 1)");
    }
    for (const auto &kv : other_names) bad.push_back("unclassified tensor " + kv.first);
    if (other > other_names.size()) bad.push_back("... " + std::to_string(other - other_names.size()) + " more unclassified");
    return bad;
  }

  model::Budget budget(const model::Checkpoint &ck, const model::DeployParams &p) const override;

  // Placement of every tensor (TP over the cards, by head / by intermediate column):
  //   attention: q_proj rows are per head (query | gate of head h) -> contiguous split, k / v
  //              split by kv head and appended (one GEMM); o_proj split along k; q/k norms replicated.
  //   GDN: in_proj_qkv rows = q | k | v sections, each split by head, z appended; conv1d rows the
  //        same sections; a, b, A_log, dt_bias split by value head; out_proj split along k; norm
  //        replicated.
  //   MLP: gate | up split by intermediate row (one GEMM), down split along k; the cards' partial
  //        sums meet in the residual all-reduce.
  //   norms replicated; lm_head: vocab rows split; embedding: pinned host.
  // int4 Linears are Store::w4 under their packed name (`weight_packed` or `qweight`; the loader reads
  // the scales beside it, `w4_layout` says which orientation); bf16 matrices follow `dense_bytes`.
  std::vector<model::DenseSpec> dense_plan(const model::Checkpoint &ck, const model::DeployParams &p) const override {
    using S = model::DenseSpec;
    std::vector<S> out;
    const S::Store mat = p.dense_bytes == 1 ? S::Store::i8 : S::Store::bf16;
    const S::Store lin = S::Store::w4;
    auto add = [&](std::string name, S::Split split, S::Store store, Component comp, int layer, std::vector<int64_t> sections = {}) {
      if (!ck.find(name)) throw std::runtime_error("qwen35 dense plan: missing " + name);
      S sp;
      sp.name = std::move(name);
      sp.split = split;
      sp.store = store == S::Store::w4 ? lin : store;
      sp.w4_layout = c_.w_gptq() ? S::W4Layout::gptq : S::W4Layout::ct;
      sp.sections = std::move(sections);
      sp.comp = comp;
      sp.layer = layer;
      out.push_back(std::move(sp));
    };
    auto append = [&](std::vector<std::string> names) {
      for (const auto &n : names)
        if (!ck.find(n)) throw std::runtime_error("qwen35 dense plan: missing " + n);
      out.back().append = std::move(names);
    };
    add(T("embed_tokens.weight"), S::Split::host, S::Store::raw, Component::embedding, -1);
    if (ck.find("lm_head.weight"))
      add("lm_head.weight", S::Split::rows, mat, Component::lm_head, -1);
    else if (ck.find("lm_head.qweight"))  // an int4-only head: kept int4 under the bf16 head's name
      add("lm_head.qweight", S::Split::rows, S::Store::w4, Component::lm_head, -1);
    else
      throw std::runtime_error("qwen35 dense plan: missing lm_head (weight or qweight)");
    if (out.back().name == "lm_head.qweight") out.back().key = "lm_head.weight";
    add(T("norm.weight"), S::Split::replicate, S::Store::raw, Component::norm, -1);
    const int64_t kq = static_cast<int64_t>(c_.lin_k_heads) * c_.lin_k_dim, vq = static_cast<int64_t>(c_.lin_v_heads) * c_.lin_v_dim;
    for (int l = 0; l < c_.layers; ++l) {
      add(L(l, "input_layernorm.weight"), S::Split::replicate, S::Store::raw, Component::norm, l);
      add(L(l, "post_attention_layernorm.weight"), S::Split::replicate, S::Store::raw, Component::norm, l);
      if (c_.full_attention[l]) {
        const auto A = Component::attention;
        add(c_.packed_name(L(l, "self_attn.q_proj")), S::Split::rows, S::Store::w4, A, l);
        append({c_.packed_name(L(l, "self_attn.k_proj")), c_.packed_name(L(l, "self_attn.v_proj"))});
        add(c_.packed_name(L(l, "self_attn.o_proj")), S::Split::cols, S::Store::w4, A, l);
        add(L(l, "self_attn.q_norm.weight"), S::Split::replicate, S::Store::raw, A, l);
        add(L(l, "self_attn.k_norm.weight"), S::Split::replicate, S::Store::raw, A, l);
      } else {
        const auto G = Component::linear_attention;
        add(c_.packed_name(L(l, "linear_attn.in_proj_qkv")), S::Split::rows, S::Store::w4, G, l, {kq, kq, vq});
        append({c_.packed_name(L(l, "linear_attn.in_proj_z"))});
        add(L(l, "linear_attn.conv1d.weight"), S::Split::rows, S::Store::raw, G, l, {kq, kq, vq});
        add(L(l, "linear_attn.in_proj_a.weight"), S::Split::rows, S::Store::raw, G, l);
        add(L(l, "linear_attn.in_proj_b.weight"), S::Split::rows, S::Store::raw, G, l);
        add(L(l, "linear_attn.A_log"), S::Split::rows, S::Store::raw, G, l);
        add(L(l, "linear_attn.dt_bias"), S::Split::rows, S::Store::raw, G, l);
        add(L(l, "linear_attn.norm.weight"), S::Split::replicate, S::Store::raw, G, l);
        add(c_.packed_name(L(l, "linear_attn.out_proj")), S::Split::cols, S::Store::w4, G, l);
      }
      add(c_.packed_name(L(l, "mlp.gate_proj")), S::Split::rows, S::Store::w4, Component::dense_mlp, l);
      append({c_.packed_name(L(l, "mlp.up_proj"))});
      add(c_.packed_name(L(l, "mlp.down_proj")), S::Split::cols, S::Store::w4, Component::dense_mlp, l);
    }
    return out;
  }

  Config c_;
};

// ---------------------------------------------------------------- budget
// Per-card bytes of the deployment: TP over the cards for every matrix (attention and GDN by
// head, the MLP by intermediate column, lm_head by vocab row); norms replicated; no experts.
model::Budget Qwen35::budget(const model::Checkpoint &ck, const model::DeployParams &p) const {
  if (p.cards < 1 || c_.kv_heads % p.cards || c_.q_heads % p.cards || c_.lin_v_heads % p.cards || c_.lin_k_heads % p.cards ||
      c_.vocab % p.cards || c_.inter % (static_cast<int64_t>(p.cards) * c_.w_group))
    throw std::runtime_error("qwen35 budget: the head / vocab / intermediate counts do not split over " + std::to_string(p.cards) + " cards");
  if (p.dense_bytes != 1 && p.dense_bytes != 2) throw std::runtime_error("dense_bytes must be 1 or 2");
  if (p.kv_bytes != 1 && p.kv_bytes != 2) throw std::runtime_error("kv_bytes must be 1 or 2");
  if (p.kv_v4 && p.kv_bytes != 1) throw std::runtime_error("kv_v4 (4-bit V) needs kv_bytes 1 (int8 K)");
  if (p.kv_shard == model::DeployParams::KvShard::tokens) throw std::runtime_error("qwen35 budget: token-sharded KV is not modelled (heads only)");

  const double C = p.cards, db = p.dense_bytes;
  // A bf16 [rows][cols] weight stored at the dense precision (8-bit adds one fp32 scale per row).
  auto dense = [&](double rows, double cols, double split) { return rows * cols * db / split + (db == 1 ? rows / split * 4 : 0); };
  auto bf16 = [&](double rows, double cols, double split) { return rows * cols * 2 / split; };
  // A pack-quantized Linear as stored: int4 codes + one fp16 scale per (row, group of k).
  auto w4 = [&](double rows, double cols, double split) { return rows * cols / split * (c_.w_bits / 8.0 + 2.0 / c_.w_group); };

  model::Budget b;
  const double H = c_.hidden, I = c_.inter;
  const int full = c_.full_layers(), lin = c_.layers - full;
  auto row = [&](std::string name, double resident, double step, std::string note) {
    b.rows.push_back({std::move(name), resident, step, std::move(note)});
  };

  {
    const double table = bf16(static_cast<double>(c_.vocab), H, 1);
    if (p.embed_on_host) row("embedding", 0, 0, "pinned host table, a step gathers its rows over PCIe");
    else row("embedding", table, 0, "replicated table");
  }
  {
    const double lm = dense(static_cast<double>(c_.vocab), H, C);
    row("lm_head", lm, lm, "vocab-parallel, each card a contiguous half of the rows (bf16 in the checkpoint)");
  }
  {
    const double QD = static_cast<double>(c_.q_heads) * c_.head_dim, KD = static_cast<double>(c_.kv_heads) * c_.head_dim;
    const double per = w4(c_.q_rows(), H, C) + w4(KD, H, C) * 2 + w4(H, QD, C);
    row("attention x" + std::to_string(full), per * full, per * full, "int4 g128; heads split (q | gate, k, v one GEMM)");
  }
  {
    const double V = static_cast<double>(c_.lin_v_heads) * c_.lin_v_dim;
    const double per = w4(c_.lin_qkv(), H, C) + w4(V, H, C) + bf16(2.0 * c_.lin_v_heads, H, C) + w4(H, V, C) + bf16(c_.lin_qkv(), c_.conv_kernel, C);
    row("linear attention (GDN) x" + std::to_string(lin), per * lin, per * lin, "int4 g128; heads split; a / b projections bf16");
  }
  {
    const double per = w4(2 * I, H, C) + w4(H, I, C);
    row("dense MLP x" + std::to_string(c_.layers), per * c_.layers, per * c_.layers, "int4 g128; gate | up by intermediate row, down along k");
  }
  {
    const double norms = bf16(2.0 * c_.layers + 1, H, 1) + bf16(2.0 * full, c_.head_dim, 1) + bf16(lin, c_.lin_v_dim, 1);
    row("norms x" + std::to_string(2 * c_.layers + 1 + 2 * full + lin), norms, norms, "replicated bf16 vectors");
  }

  // KV per token per card per slot: K and V of this card's KV heads (+ one fp32 scale per token
  // and head for 8-bit KV). Dense softmax attention reads the whole context every step.
  const int kv_layers = full + (p.drafter ? 1 : 0);
  // int8-v4 = 4-bit V codes + a bf16 scale per 32 dims on the target's layers (the drafter's stays int8)
  const double kv_all = 2.0 * c_.kv_heads * c_.head_dim * p.kv_bytes + (p.kv_bytes == 1 ? 2.0 * c_.kv_heads * 4 : 0);
  const double kv_v4 = 1.5 * c_.kv_heads * c_.head_dim + (4.0 + 2.0 * c_.head_dim / 32) * c_.kv_heads;
  b.kv_bytes_per_token = (p.kv_v4 ? full * kv_v4 + (kv_layers - full) * kv_all : kv_layers * kv_all) / C;
  {
    const double ctx = static_cast<double>(p.context);
    const double resident = b.kv_bytes_per_token * ctx * p.seqs;
    row("KV cache (" + std::to_string(p.context) + " x " + std::to_string(p.seqs) + " slot)", resident, resident,
        std::to_string(kv_layers) + " layers" + (p.drafter ? " incl. the drafter's" : "") + ", " +
            (p.kv_v4 ? "8-bit K, 4-bit V + scales" : p.kv_bytes == 1 ? "8-bit + scales" : "bf16") + ", split by KV head; dense attention reads every token per step");
  }
  if (p.drafter) {
    // The bf16 drafter layer at the dense precision: q | gate, k, v, o, gate | up, down and fc.
    const double QD = static_cast<double>(c_.q_heads) * c_.head_dim, KD = static_cast<double>(c_.kv_heads) * c_.head_dim;
    const double d = dense(c_.q_rows(), H, C) + dense(KD, H, C) * 2 + dense(H, QD, C) + dense(2 * I, H, C) + dense(H, I, C) + dense(H, 2 * H, 1);
    row("MTP drafter (1 layer)", d, d, "bf16 in the checkpoint, stored at the dense precision; fc replicated");
  }

  b.expert_bytes = 0;
  b.experts_per_card = 0;
  b.expert_step_bytes = 0;
  b.unique_experts = 0;
  (void)ck;

  double resident = 0, step = 0;
  for (const auto &r : b.rows) {
    resident += r.resident_bytes;
    step += r.step_bytes;
  }
  b.resident_non_expert = resident;
  const double vram = p.vram_gib * (1ull << 30), reserve = p.reserve_gib * (1ull << 30);
  b.hot_experts_fit = 0;
  b.floor_ms = step / (p.bandwidth_gbs * 1e9) * 1e3;
  if (resident + reserve > vram) {
    char buf[160];
    std::snprintf(buf, sizeof buf, "does NOT fit: %.2f GiB resident + %.2f reserve > %.2f GiB VRAM per card", resident / (1ull << 30),
                  p.reserve_gib, p.vram_gib);
    b.notes.push_back(buf);
  } else {
    char buf[160];
    std::snprintf(buf, sizeof buf, "fits: %.2f GiB free per card after the reserve (room for %.0f more context tokens per slot)",
                  (vram - reserve - resident) / (1ull << 30), (vram - reserve - resident) / std::max(1.0, b.kv_bytes_per_token * p.seqs));
    b.notes.push_back(buf);
  }
  b.notes.push_back("no routed experts: every weight is resident; the step reads them all once");
  b.notes.push_back("not modelled: activation scratch, recorded graphs, mailboxes (covered by the reserve)");
  return b;
}

}  // namespace

std::string scale_name_of(const std::string &packed_name) {
  for (const auto &[p, sc] : {std::pair<const char *, const char *>{".weight_packed", ".weight_scale"}, std::pair<const char *, const char *>{".qweight", ".scales"}}) {
    const std::string suffix = p;
    if (packed_name.size() > suffix.size() && packed_name.compare(packed_name.size() - suffix.size(), suffix.size(), suffix) == 0)
      return packed_name.substr(0, packed_name.size() - suffix.size()) + sc;
  }
  throw std::runtime_error("qwen35: " + packed_name + " is not a packed int4 tensor name");
}

std::vector<model::DenseSpec> mtp_dense_plan(const model::Checkpoint &ck, int dense_bytes, const std::vector<std::pair<int64_t, int64_t>> &head_rows, bool head_w4) {
  using S = model::DenseSpec;
  if (dense_bytes != 1 && dense_bytes != 2) throw std::invalid_argument("qwen35 mtp plan: dense_bytes 1 (int8) or 2 (bf16)");
  const S::Store mat = dense_bytes == 1 ? S::Store::i8 : S::Store::bf16;
  std::vector<S> out;
  auto add = [&](std::string name, S::Split split, S::Store store, std::vector<std::string> append = {}) {
    if (!ck.find(name)) throw std::runtime_error("qwen35 mtp plan: missing " + name);
    for (const auto &a : append)
      if (!ck.find(a)) throw std::runtime_error("qwen35 mtp plan: missing " + a);
    S sp;
    sp.name = std::move(name);
    sp.split = split;
    sp.store = store;
    sp.append = std::move(append);
    sp.comp = Component::mtp;
    sp.layer = 0;
    out.push_back(std::move(sp));
  };
  // A drafter matrix: the checkpoint's bf16 / f16 .weight stored at the dense precision, or a baked int4 GPTQ .qweight
  // dequantised at load to the same precision.
  auto add_mat = [&](std::string module, S::Split split, std::vector<std::string> append_mods = {}) {
    std::vector<std::string> ap;
    for (const auto &m : append_mods) ap.push_back(m + (ck.find(m + ".weight") ? ".weight" : ".qweight"));
    if (ck.find(module + ".weight")) {
      add(module + ".weight", split, mat, std::move(ap));
    } else {
      add(module + ".qweight", split, S::Store::w4d, std::move(ap));
      out.back().w4_layout = S::W4Layout::gptq;
      out.back().dense_bytes = dense_bytes;
      out.back().key = module + ".weight";  // the drafter reads its matrices by the canonical name
    }
  };
  const std::string M = "mtp.", L = "mtp.layers.0.";
  if (!ck.find(L + "input_layernorm.weight"))
    throw std::runtime_error("qwen35 mtp plan: this checkpoint has no mtp.layers.0.* layer (an export that kept only mtp.fc): no MTP drafter");
  add(M + "pre_fc_norm_embedding.weight", S::Split::replicate, S::Store::raw);
  add(M + "pre_fc_norm_hidden.weight", S::Split::replicate, S::Store::raw);
  add_mat(M + "fc", S::Split::cols);
  add(M + "norm.weight", S::Split::replicate, S::Store::raw);
  add(L + "input_layernorm.weight", S::Split::replicate, S::Store::raw);
  add(L + "post_attention_layernorm.weight", S::Split::replicate, S::Store::raw);
  add_mat(L + "self_attn.q_proj", S::Split::rows, {L + "self_attn.k_proj", L + "self_attn.v_proj"});
  add_mat(L + "self_attn.o_proj", S::Split::cols);
  add(L + "self_attn.q_norm.weight", S::Split::replicate, S::Store::raw);
  add(L + "self_attn.k_norm.weight", S::Split::replicate, S::Store::raw);
  add_mat(L + "mlp.gate_proj", S::Split::rows, {L + "mlp.up_proj"});
  add_mat(L + "mlp.down_proj", S::Split::cols);
  if (!head_rows.empty()) {
    if (head_w4) {
      add("lm_head.qweight", S::Split::rows, S::Store::w4);
      out.back().w4_layout = S::W4Layout::gptq;
    } else if (ck.find("lm_head.weight")) {
      add("lm_head.weight", S::Split::rows, mat);
    } else {  // an int4-only head: dequantise its rows to the dense precision
      add("lm_head.qweight", S::Split::rows, S::Store::w4d);
      out.back().w4_layout = S::W4Layout::gptq;
      out.back().dense_bytes = dense_bytes;
    }
    out.back().row_ranges = head_rows;
    out.back().key = kMtpHeadKey;
  }
  return out;
}

model::ArchEntry entry() {
  return {
      "qwen35",
      "Qwen3.8-27B / Qwen3.5 dense (qwen3_5): hybrid GDN + gated GQA, dense SwiGLU MLP, compressed-tensors int4 W4A16",
      "dense Qwen3.8-27B",
      "qwen3_5",
      [](const ModelConfig &mc) {
        if (mc.model_type() == "qwen3_5") return true;
        for (const auto &a : mc.architectures())
          if (a.rfind("Qwen3_5", 0) == 0) return true;
        return false;
      },
      [](const ModelConfig &mc) -> std::unique_ptr<model::Arch> { return std::make_unique<Qwen35>(Config::parse(mc)); },
  };
}

int select_head_pages(float weight, int pages) {
  if (pages <= kSelectFloor) return pages;
  return kSelectFloor + static_cast<int>(std::lround(static_cast<double>(weight) * (pages - kSelectFloor)));
}

int SelectShare::count(int pages, double f, int forced) {
  return std::min(pages, std::max(forced + 1, static_cast<int>(std::lround(pages * f))));
}

std::vector<float> SelectShare::weights(std::span<const double> lost, int units) {
  if (units < 1 || lost.size() != static_cast<size_t>(units) * kFractions) throw std::invalid_argument("SelectShare::weights: a curve of kFractions values per unit");
  auto curve = [&](int u) { return lost.subspan(static_cast<size_t>(u) * kFractions, kFractions); };
  std::vector<int> at(static_cast<size_t>(units), 0);  // each unit's fraction index
  long left = std::lround((kMean - fraction(0)) * 128) * static_cast<long>(units);  // steps of 1 / 128 to hand out
  while (left > 0) {
    int best = -1, jump = 0;
    double gain = -1;
    for (int u = 0; u < units; ++u) {
      const auto c = curve(u);
      const int i = at[static_cast<size_t>(u)];
      // the best average over any jump ahead: a curve is flat where the forced pages are the whole count. Equal
      // gains (curves that have nothing left to remove) go to the unit with the fewest pages, so what is left over
      // is spread evenly instead of landing on the first unit.
      for (int j = 1; j <= left && i + j < kFractions; ++j) {
        const double g = (c[static_cast<size_t>(i)] - c[static_cast<size_t>(i + j)]) / j;
        if (g > gain || (g == gain && best >= 0 && best != u && i < at[static_cast<size_t>(best)])) best = u, gain = g, jump = j;
      }
    }
    if (best < 0) break;
    at[static_cast<size_t>(best)] += jump, left -= jump;
  }
  double mean = 0;
  for (const int i : at) mean += fraction(i) / units;
  std::vector<float> w(static_cast<size_t>(units));
  for (int u = 0; u < units; ++u) w[static_cast<size_t>(u)] = static_cast<float>(1.0 + kSoften * (fraction(at[static_cast<size_t>(u)]) / mean - 1.0));
  return w;
}

float select_key_score(uint32_t key) {
  // order_keys: a non-negative float's bits with the sign bit set, a negative one's bits inverted
  return std::bit_cast<float>((key & 0x80000000u) != 0u ? key & 0x7fffffffu : ~key);
}

void select_curve_add(std::span<const float> scores, int64_t tokens, int page_tokens, int sink, int local, float scale, std::span<double> out) {
  const int pages = static_cast<int>(scores.size());
  if (pages < 1 || out.size() != static_cast<size_t>(SelectShare::kFractions) || tokens <= static_cast<int64_t>(pages - 1) * page_tokens ||
      tokens > static_cast<int64_t>(pages) * page_tokens)
    throw std::invalid_argument("select_curve_add: one score per page of the context, kFractions outputs");
  const float top = *std::max_element(scores.begin(), scores.end());
  struct Page {
    float score, mass;
    int page;
  };
  std::vector<Page> rest;  // the pages the score decides
  rest.reserve(static_cast<size_t>(pages));
  double sum = 0, forced_mass = 0;
  const int local0 = std::max(sink, pages - local), forced = std::min(pages, sink) + (pages - local0);
  for (int p = 0; p < pages; ++p) {
    const int n = p + 1 < pages ? page_tokens : static_cast<int>(tokens - static_cast<int64_t>(pages - 1) * page_tokens);
    const float mass = std::exp((scores[static_cast<size_t>(p)] - top) * scale) * static_cast<float>(n);
    sum += mass;
    if (p < sink || p >= local0) forced_mass += mass;
    else rest.push_back({scores[static_cast<size_t>(p)], mass, p});
  }
  std::sort(rest.begin(), rest.end(), [](const Page &a, const Page &b) { return a.score != b.score ? a.score > b.score : a.page < b.page; });
  double kept = forced_mass;
  size_t taken = 0;
  for (int f = 0; f < SelectShare::kFractions; ++f) {
    const size_t want = static_cast<size_t>(SelectShare::count(pages, SelectShare::fraction(f), forced) - forced);
    for (; taken < want && taken < rest.size(); ++taken) kept += rest[taken].mass;
    out[static_cast<size_t>(f)] += std::max(0.0, 1.0 - kept / sum);
  }
}

}  // namespace xenithium::arch::qwen35
