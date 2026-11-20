#include "xenithium/arch/llama/arch.hpp"

#include <cmath>
#include <cstdio>
#include <map>
#include <sstream>
#include <stdexcept>

namespace xenithium::arch::llama {

using model::Component;
using model::DType;
using model::ModelConfig;
using qwen35::Config;

namespace {

[[noreturn]] void invariant(const std::string &what) { throw std::runtime_error("llama config: " + what); }

int as_int(int64_t v, const char *key) {
  if (v < 0 || v > (1ll << 31) - 1) invariant(std::string(key) + " out of range");
  return static_cast<int>(v);
}

bool flag_true(const json::Value &obj, std::string_view key) {
  const json::Value *v = obj.get(key);
  return v && v->is_bool() && v->b;
}

}  // namespace

Config parse_config(const ModelConfig &mc) {
  const json::Value &t = mc.text();
  Config c;
  c.prefix = kPrefix;
  c.norm_offset = false;
  c.qk_norm = false;
  c.hidden = as_int(ModelConfig::need_int(t, "hidden_size"), "hidden_size");
  c.layers = as_int(ModelConfig::need_int(t, "num_hidden_layers"), "num_hidden_layers");
  c.vocab = ModelConfig::need_int(t, "vocab_size");
  c.max_positions = ModelConfig::need_int(t, "max_position_embeddings");
  c.rms_eps = ModelConfig::need_num(t, "rms_norm_eps");
  c.tie_embeddings = ModelConfig::need_bool(t, "tie_word_embeddings");
  c.full_attention.assign(static_cast<size_t>(c.layers), true);
  c.full_interval = 1;
  c.q_heads = as_int(ModelConfig::need_int(t, "num_attention_heads"), "num_attention_heads");
  c.kv_heads = as_int(ModelConfig::int_or(t, "num_key_value_heads", c.q_heads), "num_key_value_heads");
  c.head_dim = as_int(ModelConfig::int_or(t, "head_dim", c.q_heads > 0 ? c.hidden / c.q_heads : 0), "head_dim");
  c.attn_output_gate = false;
  c.partial_rotary = 1.0;
  c.rope_theta = ModelConfig::need_num(t, "rope_theta");
  c.rope_type = "default";
  if (const json::Value *rs = t.get("rope_scaling"); rs && !rs->is_null()) invariant("rope_scaling is not implemented (only plain rope)");
  if (flag_true(t, "attention_bias")) invariant("attention_bias true is not implemented");
  if (flag_true(t, "mlp_bias")) invariant("mlp_bias true is not implemented");
  c.inter = as_int(ModelConfig::need_int(t, "intermediate_size"), "intermediate_size");
  c.hidden_act = ModelConfig::need_str(t, "hidden_act");
  c.mtp_layers = 0;

  // quantization_config: compressed-tensors pack-quantized int4 (one config group over every Linear)
  const json::Value *q = mc.root().get("quantization_config");
  if (!q) invariant("quantization_config missing (only int4 compressed-tensors checkpoints are served)");
  c.w_method = ModelConfig::str_or(*q, "quant_method", "");
  if (c.w_method != "compressed-tensors") invariant("quantization_config.quant_method '" + c.w_method + "' (want compressed-tensors)");
  c.w_format = ModelConfig::str_or(*q, "format", "");
  if (c.w_format != "pack-quantized") invariant("quantization_config.format '" + c.w_format + "' (want pack-quantized)");
  const json::Value &groups = ModelConfig::need(*q, "config_groups");
  if (!groups.is_object() || groups.obj.size() != 1) invariant("quantization_config.config_groups (want exactly one group)");
  const json::Value &w = ModelConfig::need(groups.obj[0].second, "weights");
  c.w_bits = as_int(ModelConfig::need_int(w, "num_bits"), "weights.num_bits");
  c.w_group = as_int(ModelConfig::need_int(w, "group_size"), "weights.group_size");
  c.w_sym = ModelConfig::need_bool(w, "symmetric");
  c.w_zero = !c.w_sym;
  if (ModelConfig::need_str(w, "type") != "int" || ModelConfig::need_str(w, "strategy") != "group") invariant("weights.type / strategy (want int, group)");
  if (const json::Value *ao = w.get("actorder"); ao && !ao->is_null() && !(ao->is_bool() && !ao->b)) invariant("weights.actorder set (want none)");
  if (flag_true(w, "dynamic")) invariant("weights.dynamic true (want static)");
  if (const json::Value *ia = groups.obj[0].second.get("input_activations"); ia && !ia->is_null()) invariant("input_activations set: not a weight-only scheme");

  c.bos = ModelConfig::int_or(t, "bos_token_id", -1);
  if (const json::Value *e = t.get("eos_token_id")) {
    if (e->is_array())
      for (const auto &v : e->arr) c.eos.push_back(static_cast<int64_t>(v.num));
    else if (e->is_number())
      c.eos.push_back(static_cast<int64_t>(e->num));
  }

  // ---- invariants the engine's kernels and budget rely on ----
  if (c.q_heads <= 0 || c.kv_heads <= 0 || c.q_heads % c.kv_heads) invariant("num_attention_heads not a multiple of num_key_value_heads");
  if (c.head_dim % 64) invariant("head_dim not a multiple of 64");
  if (c.hidden_act != "silu") invariant("hidden_act '" + c.hidden_act + "' (only silu is implemented)");
  if (c.w_bits != 4 || (c.w_group != 32 && c.w_group != 128)) invariant("weights must be int4 in groups of 32 or 128");
  if (c.hidden % c.w_group || c.inter % c.w_group || (c.q_heads * c.head_dim) % c.w_group) invariant("a Linear's input dim is not a multiple of the group");
  if (c.tie_embeddings) invariant("tied embeddings are not implemented");
  return c;
}

namespace {

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

std::string L(int layer, std::string_view rest) { return std::string(kPrefix) + "layers." + std::to_string(layer) + "." + std::string(rest); }
std::string T(std::string_view rest) { return std::string(kPrefix) + std::string(rest); }

class Llama final : public model::Arch {
 public:
  explicit Llama(Config c) : c_(std::move(c)) {}

  std::string_view id() const override { return "llama"; }

  std::string describe() const override {
    std::ostringstream o;
    o << "llama (dense Llama family, HF llama): " << c_.layers << " layers, hidden " << c_.hidden << ", vocab " << c_.vocab << "\n"
      << "  attention: " << c_.q_heads << " q / " << c_.kv_heads << " kv heads x " << c_.head_dim << " (group " << c_.q_heads / c_.kv_heads
      << "), full rotary, theta " << c_.rope_theta << ", no output gate, no q/k norm\n"
      << "  MLP: dense SwiGLU, inter " << c_.inter << "; Linears int" << c_.w_bits << " group " << c_.w_group << (c_.w_sym ? " sym" : " asym (zero points)")
      << " (" << c_.w_method << " " << c_.w_format << "; embedding and lm_head bf16)\n"
      << "  positions " << c_.max_positions << ", rms eps " << c_.rms_eps << " (plain RMSNorm), untied embeddings";
    return o.str();
  }

  model::TensorRole classify(std::string_view name) const override {
    model::TensorRole r;
    std::string_view s = name;
    if (eat(s, "lm_head.")) return {Component::lm_head, -1, -1};
    if (!eat(s, kPrefix)) return r;
    if (eat(s, "embed_tokens.")) return {Component::embedding, -1, -1};
    if (eat(s, "norm.")) return {Component::norm, -1, -1};
    if (!eat(s, "layers.")) return r;
    r.layer = eat_int(s);
    if (r.layer < 0 || !eat(s, ".")) return {Component::other, -1, -1};
    if (eat(s, "mlp.")) r.comp = Component::dense_mlp;
    else if (eat(s, "self_attn.")) r.comp = Component::attention;
    else if (eat(s, "input_layernorm.") || eat(s, "post_attention_layernorm.")) r.comp = Component::norm;
    return r;
  }

  std::vector<std::string> validate(const model::Checkpoint &ck, std::vector<std::string> *warnings) const override {
    std::vector<std::string> bad;
    auto fmt = [](const std::vector<int64_t> &v) {
      std::string s = "[";
      for (size_t i = 0; i < v.size(); ++i) s += (i ? ", " : "") + std::to_string(v[i]);
      return s + "]";
    };
    auto want = [&](const std::string &name, std::initializer_list<DType> dts, std::vector<int64_t> shape) {
      const model::TensorInfo *t = ck.find(name);
      if (!t) {
        bad.push_back("missing " + name);
        return;
      }
      bool dt_ok = false;
      for (DType d : dts) dt_ok |= t->dtype == d;
      if (!dt_ok || t->shape != shape) bad.push_back(name + ": " + t->dtype_str + " " + fmt(t->shape) + ", want " + std::string(model::dtype_name(*dts.begin())) + " " + fmt(shape));
    };
    const int64_t pack = 32 / c_.w_bits, G = c_.w_group;
    // An int4 Linear of n output rows and k inputs (compressed-tensors pack-quantized, zero points when asymmetric).
    auto want_w4 = [&](const std::string &module, int64_t n, int64_t k) {
      want(module + ".weight_packed", {DType::i32}, {n, k / pack});
      want(module + ".weight_scale", {DType::bf16, DType::f16}, {n, k / G});
      want(module + ".weight_shape", {DType::i32, DType::i64}, {2});
      if (c_.w_zero) want(module + ".weight_zero_point", {DType::i32}, {n / pack, k / G});
    };
    const int64_t H = c_.hidden, I = c_.inter, QD = static_cast<int64_t>(c_.q_heads) * c_.head_dim, KD = static_cast<int64_t>(c_.kv_heads) * c_.head_dim;
    want(T("embed_tokens.weight"), {DType::bf16, DType::f16}, {c_.vocab, H});
    want("lm_head.weight", {DType::bf16, DType::f16}, {c_.vocab, H});
    want(T("norm.weight"), {DType::bf16, DType::f16}, {H});
    for (int l = 0; l < c_.layers; ++l) {
      want(L(l, "input_layernorm.weight"), {DType::bf16, DType::f16}, {H});
      want(L(l, "post_attention_layernorm.weight"), {DType::bf16, DType::f16}, {H});
      want_w4(L(l, "self_attn.q_proj"), QD, H);
      want_w4(L(l, "self_attn.k_proj"), KD, H);
      want_w4(L(l, "self_attn.v_proj"), KD, H);
      want_w4(L(l, "self_attn.o_proj"), H, QD);
      want_w4(L(l, "mlp.gate_proj"), I, H);
      want_w4(L(l, "mlp.up_proj"), I, H);
      want_w4(L(l, "mlp.down_proj"), H, I);
    }
    // Every Linear of one layer must use one scale dtype (the loader converts bf16 scales to the kernels' fp16).
    size_t bf16_scales = 0, f16_scales = 0, other = 0;
    std::map<std::string, int> other_names;
    for (const auto &t : ck.tensors()) {
      if (t.name.size() > 13 && t.name.compare(t.name.size() - 13, 13, ".weight_scale") == 0) (t.dtype == DType::bf16 ? bf16_scales : f16_scales)++;
      if (classify(t.name).comp == Component::other) {
        ++other;
        if (other_names.size() < 8) other_names[t.name] = 1;
      }
    }
    if (warnings) {
      warnings->push_back(std::to_string(bf16_scales) + " bf16 / " + std::to_string(f16_scales) + " f16 weight scales");
      if (!c_.w_sym) warnings->push_back("asymmetric int4 (zero points): w = (q - z) x scale, groups of " + std::to_string(c_.w_group));
    }
    for (const auto &kv : other_names) bad.push_back("unclassified tensor " + kv.first);
    if (other > other_names.size()) bad.push_back("... " + std::to_string(other - other_names.size()) + " more unclassified");
    return bad;
  }

  model::Budget budget(const model::Checkpoint &ck, const model::DeployParams &p) const override;

  // Placement (TP over the cards): q_proj by whole heads with k_proj and v_proj's kv-head slices appended (one W4
  // matrix [q x hq | k x hkv | v x hkv] per card), o_proj along k; gate | up by intermediate row (one matrix), down along
  // k; norms replicated; lm_head by vocabulary row; the embedding in pinned host memory.
  std::vector<model::DenseSpec> dense_plan(const model::Checkpoint &ck, const model::DeployParams &p) const override {
    using S = model::DenseSpec;
    std::vector<S> out;
    const S::Store mat = p.dense_bytes == 1 ? S::Store::i8 : S::Store::bf16;
    auto add = [&](std::string name, S::Split split, S::Store store, Component comp, int layer) {
      if (!ck.find(name)) throw std::runtime_error("llama dense plan: missing " + name);
      S sp;
      sp.name = std::move(name);
      sp.split = split;
      sp.store = store;
      if (store == S::Store::w4) {  // the int4 Linears' layout (other specs keep the defaults)
        sp.w4_layout = c_.w_zero ? S::W4Layout::ct_zp : S::W4Layout::ct;
        sp.w4_group = c_.w_group;
      }
      sp.comp = comp;
      sp.layer = layer;
      out.push_back(std::move(sp));
    };
    auto append = [&](std::vector<std::string> names) {
      for (const auto &n : names)
        if (!ck.find(n)) throw std::runtime_error("llama dense plan: missing " + n);
      out.back().append = std::move(names);
    };
    add(T("embed_tokens.weight"), S::Split::host, S::Store::raw, Component::embedding, -1);
    add("lm_head.weight", S::Split::rows, mat, Component::lm_head, -1);
    add(T("norm.weight"), S::Split::replicate, S::Store::raw, Component::norm, -1);
    for (int l = 0; l < c_.layers; ++l) {
      add(L(l, "input_layernorm.weight"), S::Split::replicate, S::Store::raw, Component::norm, l);
      add(L(l, "post_attention_layernorm.weight"), S::Split::replicate, S::Store::raw, Component::norm, l);
      const auto A = Component::attention;
      add(c_.packed_name(L(l, "self_attn.q_proj")), S::Split::rows, S::Store::w4, A, l);
      append({c_.packed_name(L(l, "self_attn.k_proj")), c_.packed_name(L(l, "self_attn.v_proj"))});
      add(c_.packed_name(L(l, "self_attn.o_proj")), S::Split::cols, S::Store::w4, A, l);
      add(c_.packed_name(L(l, "mlp.gate_proj")), S::Split::rows, S::Store::w4, Component::dense_mlp, l);
      append({c_.packed_name(L(l, "mlp.up_proj"))});
      add(c_.packed_name(L(l, "mlp.down_proj")), S::Split::cols, S::Store::w4, Component::dense_mlp, l);
    }
    return out;
  }

  Config c_;
};

// Per-card bytes: TP over the cards for every matrix, norms replicated, the KV of every layer split by kv head.
model::Budget Llama::budget(const model::Checkpoint &ck, const model::DeployParams &p) const {
  if (p.cards < 1 || c_.kv_heads % p.cards || c_.q_heads % p.cards || c_.vocab % p.cards || c_.inter % (static_cast<int64_t>(p.cards) * c_.w_group) ||
      (static_cast<int64_t>(c_.q_heads) * c_.head_dim) % (static_cast<int64_t>(p.cards) * c_.w_group))
    throw std::runtime_error("llama budget: the head / vocab / intermediate counts do not split over " + std::to_string(p.cards) + " cards");
  if (p.dense_bytes != 1 && p.dense_bytes != 2) throw std::runtime_error("dense_bytes must be 1 or 2");
  if (p.kv_bytes != 1 && p.kv_bytes != 2) throw std::runtime_error("kv_bytes must be 1 or 2");
  if (p.kv_shard == model::DeployParams::KvShard::tokens) throw std::runtime_error("llama budget: token-sharded KV is not modelled (heads only)");
  const double C = p.cards, db = p.dense_bytes;
  auto dense = [&](double rows, double cols, double split) { return rows * cols * db / split + (db == 1 ? rows / split * 4 : 0); };
  auto bf16 = [&](double rows, double cols, double split) { return rows * cols * 2 / split; };
  // An int4 Linear on the card: codes + one fp16 scale per (row, group) + one 4-bit zero point per (row, group).
  const double per_w = c_.w_bits / 8.0 + 2.0 / c_.w_group + (c_.w_zero ? 0.5 / c_.w_group : 0.0);
  auto w4 = [&](double rows, double cols, double split) { return rows * cols / split * per_w; };

  model::Budget b;
  const double H = c_.hidden, I = c_.inter, QD = static_cast<double>(c_.q_heads) * c_.head_dim, KD = static_cast<double>(c_.kv_heads) * c_.head_dim;
  auto row = [&](std::string name, double resident, double step, std::string note) { b.rows.push_back({std::move(name), resident, step, std::move(note)}); };
  char gnote[64];
  std::snprintf(gnote, sizeof gnote, "int4 g%d%s", c_.w_group, c_.w_zero ? " + zero points" : "");
  if (p.embed_on_host) row("embedding", 0, 0, "pinned host table, a step gathers its rows over PCIe");
  else row("embedding", bf16(static_cast<double>(c_.vocab), H, 1), 0, "replicated table");
  {
    const double lm = dense(static_cast<double>(c_.vocab), H, C);
    row("lm_head", lm, lm, "vocab-parallel, each card a contiguous 1/C of the rows (bf16 in the checkpoint)");
  }
  {
    const double per = w4(QD, H, C) + w4(KD, H, C) * 2 + w4(H, QD, C);
    row("attention x" + std::to_string(c_.layers), per * c_.layers, per * c_.layers, std::string(gnote) + "; heads split (q, k, v one GEMM)");
  }
  {
    const double per = w4(2 * I, H, C) + w4(H, I, C);
    row("dense MLP x" + std::to_string(c_.layers), per * c_.layers, per * c_.layers, std::string(gnote) + "; gate | up by intermediate row, down along k");
  }
  {
    const double norms = bf16(2.0 * c_.layers + 1, H, 1);
    row("norms x" + std::to_string(2 * c_.layers + 1), norms, norms, "replicated bf16 vectors");
  }
  const double kv_all = 2.0 * c_.kv_heads * c_.head_dim * p.kv_bytes + (p.kv_bytes == 1 ? 2.0 * c_.kv_heads * 4 : 0);
  b.kv_bytes_per_token = c_.layers * kv_all / C;
  {
    const double resident = b.kv_bytes_per_token * static_cast<double>(p.context) * p.seqs;
    row("KV cache (" + std::to_string(p.context) + " x " + std::to_string(p.seqs) + " slot)", resident, resident,
        std::to_string(c_.layers) + " layers, " + (p.kv_bytes == 1 ? "8-bit + scales" : "bf16") + ", split by KV head; dense attention reads every token per step");
  }
  (void)ck;
  double resident = 0, step = 0;
  for (const auto &r : b.rows) {
    resident += r.resident_bytes;
    step += r.step_bytes;
  }
  b.resident_non_expert = resident;
  const double vram = p.vram_gib * (1ull << 30), reserve = p.reserve_gib * (1ull << 30);
  b.floor_ms = step / (p.bandwidth_gbs * 1e9) * 1e3;
  char buf[160];
  if (resident + reserve > vram) {
    std::snprintf(buf, sizeof buf, "does NOT fit: %.2f GiB resident + %.2f reserve > %.2f GiB VRAM per card", resident / (1ull << 30), p.reserve_gib, p.vram_gib);
  } else {
    std::snprintf(buf, sizeof buf, "fits: %.2f GiB free per card after the reserve (room for %.0f more context tokens per slot)", (vram - reserve - resident) / (1ull << 30),
                  (vram - reserve - resident) / std::max(1.0, b.kv_bytes_per_token * p.seqs));
  }
  b.notes.push_back(buf);
  if (p.drafter) b.notes.push_back("no drafter in this family: --drafter ignored");
  b.notes.push_back("no routed experts: every weight is resident; the step reads them all once");
  b.notes.push_back("not modelled: activation scratch, recorded graphs, mailboxes (covered by the reserve)");
  return b;
}

}  // namespace

model::ArchEntry entry() {
  return {
      "llama",
      "dense Llama family (llama, e.g. Nanbeige4.3-33B): GQA with full rotary, dense SwiGLU MLP, compressed-tensors int4 (zero points)",
      "dense Nanbeige4.3-33B",
      "llama",
      [](const ModelConfig &mc) {
        if (mc.model_type() == "llama") return true;
        for (const auto &a : mc.architectures())
          if (a == "LlamaForCausalLM") return true;
        return false;
      },
      [](const ModelConfig &mc) -> std::unique_ptr<model::Arch> { return std::make_unique<Llama>(parse_config(mc)); },
  };
}

}  // namespace xenithium::arch::llama
