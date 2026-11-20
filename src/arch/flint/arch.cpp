#include "xenithium/arch/flint/arch.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>

namespace xenithium::arch::flint {

using model::Component;
using model::DType;
using model::ModelConfig;

namespace {

[[noreturn]] void invariant(const std::string &what) { throw std::runtime_error("flint config: " + what); }

int as_int(int64_t v, const char *key) {
  if (v < 0 || v > (1ll << 31) - 1) invariant(std::string(key) + " out of range");
  return static_cast<int>(v);
}

bool flag_true(const json::Value &obj, std::string_view key) {
  const json::Value *v = obj.get(key);
  return v && v->is_bool() && v->b;
}

std::vector<int64_t> int_list(const json::Value &obj, std::string_view key) {
  std::vector<int64_t> out;
  const json::Value *v = obj.get(key);
  if (!v || v->is_null()) return out;
  if (!v->is_array()) invariant(std::string(key) + " is not a list");
  for (const auto &e : v->arr) {
    if (!e.is_int()) invariant(std::string(key) + " holds a non-integer");
    out.push_back(static_cast<int64_t>(e.num));
  }
  return out;
}

}  // namespace

int Config::full_layers() const { return static_cast<int>(std::count(sliding.begin(), sliding.end(), false)); }
int Config::moe_layers() const { return static_cast<int>(std::count(dense.begin(), dense.end(), false)); }

int64_t ring_tokens(const Config &c, int rows) {
  const int64_t need = static_cast<int64_t>(c.window) - 1 + std::max(1, rows);
  return ((need + 63) / 64 + 1) * 64;
}

Config Config::parse(const ModelConfig &mc) {
  const json::Value &t = mc.text();
  Config c;
  c.hidden = as_int(ModelConfig::need_int(t, "hidden_size"), "hidden_size");
  c.layers = as_int(ModelConfig::need_int(t, "num_hidden_layers"), "num_hidden_layers");
  c.vocab = ModelConfig::need_int(t, "vocab_size");
  c.max_positions = ModelConfig::need_int(t, "max_position_embeddings");
  c.rms_eps = ModelConfig::need_num(t, "rms_norm_eps");
  if (ModelConfig::need_bool(t, "tie_word_embeddings")) invariant("tied embeddings are not implemented");
  c.q_heads = as_int(ModelConfig::need_int(t, "num_attention_heads"), "num_attention_heads");
  c.kv_heads = as_int(ModelConfig::need_int(t, "num_key_value_heads"), "num_key_value_heads");
  c.head_dim = as_int(ModelConfig::need_int(t, "head_dim"), "head_dim");
  if (flag_true(t, "attention_bias")) invariant("attention_bias true is not implemented");

  // layer_types: sliding_attention (rotary, window) | full_attention (no rotary)
  const json::Value &lt = ModelConfig::need(t, "layer_types");
  if (!lt.is_array() || static_cast<int>(lt.arr.size()) != c.layers) invariant("layer_types: want one entry per layer");
  for (const auto &v : lt.arr) {
    if (!v.is_string() || (v.str != "sliding_attention" && v.str != "full_attention")) invariant("layer_types: only sliding_attention / full_attention");
    c.sliding.push_back(v.str == "sliding_attention");
  }
  if (const json::Value *u = t.get("use_sliding_window"); u && u->is_bool() && !u->b) invariant("use_sliding_window false is not implemented");
  c.window = as_int(ModelConfig::need_int(t, "sliding_window"), "sliding_window");
  // Every full-attention layer is exactly a no-rope layer (the model card: rotary on sliding layers only).
  {
    std::set<int64_t> nope;
    for (int64_t l : int_list(t, "no_rope_layer_indices")) nope.insert(l);
    for (int l = 0; l < c.layers; ++l)
      if (nope.count(l) != (c.sliding[l] ? 0u : 1u)) invariant("no_rope_layer_indices differ from the full-attention layers at layer " + std::to_string(l));
    if (const json::Value *mx = mc.root().get("mlx_export"); mx && mx->is_object()) {
      std::set<int64_t> full;
      for (int64_t l : int_list(*mx, "full_attention_layers")) full.insert(l);
      if (!full.empty() && full != nope) invariant("mlx_export.full_attention_layers differ from layer_types");
    }
  }
  // rope_parameters (or the flat rope_theta): default type only
  if (const json::Value *rp = t.get("rope_parameters"); rp && rp->is_object()) {
    const std::string ty = ModelConfig::str_or(*rp, "rope_type", "default");
    if (ty != "default") invariant("rope_parameters.rope_type '" + ty + "' (only default)");
    c.rope_theta = ModelConfig::need_num(*rp, "rope_theta");
  } else {
    c.rope_theta = ModelConfig::need_num(t, "rope_theta");
  }
  if (const json::Value *rs = t.get("rope_scaling"); rs && !rs->is_null()) invariant("rope_scaling is not implemented");
  if (const json::Value *pr = t.get("partial_rotary_factor"); pr && pr->is_number() && pr->num != 1.0) invariant("partial_rotary_factor != 1 is not implemented");
  c.sinks = flag_true(t, "attention_sink");
  if (c.sinks && ModelConfig::str_or(t, "softmax_type", "learnable") != "learnable") invariant("softmax_type: only learnable sinks");

  // MLP: mlp_only_layers dense, the rest MoE (decoder_sparse_step 1)
  c.inter = as_int(ModelConfig::need_int(t, "intermediate_size"), "intermediate_size");
  c.hidden_act = ModelConfig::need_str(t, "hidden_act");
  c.experts = as_int(ModelConfig::need_int(t, "num_experts"), "num_experts");
  c.topk = as_int(ModelConfig::need_int(t, "num_experts_per_tok"), "num_experts_per_tok");
  c.moe_inter = as_int(ModelConfig::need_int(t, "moe_intermediate_size"), "moe_intermediate_size");
  c.shared_inter = as_int(ModelConfig::need_int(t, "shared_expert_intermediate_size"), "shared_expert_intermediate_size");
  c.norm_topk = ModelConfig::need_bool(t, "norm_topk_prob");
  c.scoring = ModelConfig::str_or(t, "scoring_func", ModelConfig::str_or(t, "router_score_function", ""));
  if (ModelConfig::int_or(t, "num_local_experts", c.experts) != c.experts) invariant("num_local_experts != num_experts");
  if (ModelConfig::int_or(t, "decoder_sparse_step", 1) != 1) invariant("decoder_sparse_step != 1 is not implemented");
  if (flag_true(t, "router_expert_bias")) invariant("router_expert_bias true is not implemented");
  c.dense.assign(static_cast<size_t>(c.layers), false);
  for (int64_t l : int_list(t, "mlp_only_layers")) {
    if (l < 0 || l >= c.layers) invariant("mlp_only_layers: layer out of range");
    c.dense[static_cast<size_t>(l)] = true;
  }

  // Affine quantisation: one {bits, group_size} for every Linear; the embedding's bits from mlx_export (default = bits)
  const json::Value *q = mc.root().get("quantization");
  if (!q || !q->is_object()) invariant("quantization (MLX {bits, group_size}) missing: only the MLX int4 export is served");
  c.w_bits = as_int(ModelConfig::need_int(*q, "bits"), "quantization.bits");
  c.w_group = as_int(ModelConfig::need_int(*q, "group_size"), "quantization.group_size");
  c.embed_bits = c.w_bits;
  if (const json::Value *mx = mc.root().get("mlx_export"); mx && mx->is_object()) {
    c.embed_bits = as_int(ModelConfig::int_or(*mx, "embedding_bits", c.w_bits), "mlx_export.embedding_bits");
    if (ModelConfig::int_or(*mx, "routed_expert_bits", c.w_bits) != c.w_bits) invariant("mlx_export.routed_expert_bits != quantization.bits");
    if (ModelConfig::int_or(*mx, "bits", c.w_bits) != c.w_bits) invariant("mlx_export.bits != quantization.bits");
  }
  for (const auto &kv : q->obj)  // per-module overrides would be objects keyed by module name
    if (kv.second.is_object()) invariant("quantization: per-module overrides (" + kv.first + ") are not implemented");

  c.bos = ModelConfig::int_or(t, "bos_token_id", -1);
  c.pad = ModelConfig::int_or(t, "pad_token_id", -1);
  if (const json::Value *e = t.get("eos_token_id")) {
    if (e->is_array())
      for (const auto &v : e->arr) c.eos.push_back(static_cast<int64_t>(v.num));
    else if (e->is_number())
      c.eos.push_back(static_cast<int64_t>(e->num));
  }

  // ---- invariants the engine's kernels and budget rely on ----
  if (c.q_heads <= 0 || c.kv_heads <= 0 || c.q_heads % c.kv_heads) invariant("num_attention_heads not a multiple of num_key_value_heads");
  if (c.head_dim != 128) invariant("head_dim " + std::to_string(c.head_dim) + " (only 128 is implemented)");
  if (c.window < 2) invariant("sliding_window < 2");
  if (c.hidden_act != "silu") invariant("hidden_act '" + c.hidden_act + "' (only silu is implemented)");
  if (c.scoring != "sigmoid") invariant("scoring_func '" + c.scoring + "' (only sigmoid is implemented)");
  if (!c.norm_topk) invariant("norm_topk_prob false is not implemented");
  if (c.topk < 1 || c.topk > 16 || c.experts < c.topk || c.experts > 1024) invariant("num_experts / num_experts_per_tok out of the kernels' range");
  if (c.w_bits != 4 || c.w_group != 64) invariant("weights must be MLX int4 in groups of 64");
  if (c.embed_bits != 8 && c.embed_bits != 4) invariant("embedding_bits must be 8 or 4");
  for (int k : {c.hidden, c.inter, c.moe_inter, c.shared_inter, c.q_dim()})
    if (k % 128) invariant("a Linear's input dim is not a multiple of 128 (the W4 block)");
  if (c.full_layers() == 0) invariant("no full-attention layer");
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

class Flint final : public model::Arch {
 public:
  explicit Flint(Config c) : c_(std::move(c)) {}

  std::string_view id() const override { return "flint"; }

  std::string describe() const override {
    std::ostringstream o;
    o << "flint (Recursive Flint, HF flint): " << c_.layers << " layers, hidden " << c_.hidden << ", vocab " << c_.vocab << "\n"
      << "  attention: " << c_.q_heads << " q / " << c_.kv_heads << " kv heads x " << c_.head_dim << " (group " << c_.q_heads / c_.kv_heads
      << "), q/k RMSNorm, " << (c_.sinks ? "learned sinks" : "no sinks") << "; " << c_.sliding_layers() << " sliding (window " << c_.window
      << ", NeoX rotary theta " << c_.rope_theta << ") + " << c_.full_layers() << " full (no rotary)\n"
      << "  MLP: " << c_.layers - c_.moe_layers() << " dense SwiGLU (inter " << c_.inter << ") + " << c_.moe_layers() << " MoE (" << c_.experts
      << " experts x inter " << c_.moe_inter << ", " << c_.scoring << " top-" << c_.topk << " renormalised, shared expert inter " << c_.shared_inter
      << ")\n"
      << "  weights: MLX affine int" << c_.w_bits << " group " << c_.w_group << " (scale + bias per group), embedding int" << c_.embed_bits
      << "; positions " << c_.max_positions << ", rms eps " << c_.rms_eps << " (plain RMSNorm), untied lm_head";
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
    if (r.layer < 0 || r.layer >= c_.layers || !eat(s, ".")) return {Component::other, -1, -1};
    if (eat(s, "self_attn.")) r.comp = Component::attention;
    else if (eat(s, "input_layernorm.") || eat(s, "post_attention_layernorm.")) r.comp = Component::norm;
    else if (eat(s, "mlp.")) {
      if (c_.dense[static_cast<size_t>(r.layer)]) r.comp = Component::dense_mlp;
      else if (eat(s, "switch_mlp.")) r.comp = Component::routed_experts;  // stacked [E][..]: expert stays -1
      else if (eat(s, "shared_expert.")) r.comp = Component::shared_expert;
      else if (eat(s, "gate.")) r.comp = Component::router;
    }
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
      if (t->dtype != dt || t->shape != shape) bad.push_back(name + ": " + t->dtype_str + " " + fmt(t->shape) + ", want " + std::string(model::dtype_name(dt)) + " " + fmt(shape));
    };
    const int64_t G = c_.w_group;
    // One affine-quantised Linear (optionally stacked over `lead` experts): weight U8 [.., n, k bits / 8], scales and
    // biases BF16 [.., n, k / G].
    auto want_q = [&](const std::string &module, std::vector<int64_t> lead, int64_t n, int64_t k, int bits) {
      auto sh = [&](int64_t last) {
        std::vector<int64_t> v = lead;
        v.push_back(n);
        v.push_back(last);
        return v;
      };
      want(module + ".weight", DType::u8, sh(k * bits / 8));
      want(module + ".scales", DType::bf16, sh(k / G));
      want(module + ".biases", DType::bf16, sh(k / G));
    };
    const int64_t H = c_.hidden, QD = c_.q_dim(), KD = c_.kv_dim(), E = c_.experts, b = c_.w_bits;
    want_q(T("embed_tokens"), {}, c_.vocab, H, c_.embed_bits);
    want_q("lm_head", {}, c_.vocab, H, c_.w_bits);
    want(T("norm.weight"), DType::bf16, {H});
    for (int l = 0; l < c_.layers; ++l) {
      want(L(l, "input_layernorm.weight"), DType::bf16, {H});
      want(L(l, "post_attention_layernorm.weight"), DType::bf16, {H});
      want_q(L(l, "self_attn.q_proj"), {}, QD, H, b);
      want_q(L(l, "self_attn.k_proj"), {}, KD, H, b);
      want_q(L(l, "self_attn.v_proj"), {}, KD, H, b);
      want_q(L(l, "self_attn.o_proj"), {}, H, QD, b);
      want(L(l, "self_attn.q_norm.weight"), DType::bf16, {c_.head_dim});
      want(L(l, "self_attn.k_norm.weight"), DType::bf16, {c_.head_dim});
      if (c_.sinks) want(L(l, "self_attn.softmax_offset"), DType::bf16, {c_.q_heads});
      if (c_.dense[static_cast<size_t>(l)]) {
        want_q(L(l, "mlp.gate_proj"), {}, c_.inter, H, b);
        want_q(L(l, "mlp.up_proj"), {}, c_.inter, H, b);
        want_q(L(l, "mlp.down_proj"), {}, H, c_.inter, b);
      } else {
        want(L(l, "mlp.gate.weight"), DType::bf16, {E, H});
        want_q(L(l, "mlp.switch_mlp.gate_proj"), {E}, c_.moe_inter, H, b);
        want_q(L(l, "mlp.switch_mlp.up_proj"), {E}, c_.moe_inter, H, b);
        want_q(L(l, "mlp.switch_mlp.down_proj"), {E}, H, c_.moe_inter, b);
        want_q(L(l, "mlp.shared_expert.gate_proj"), {}, c_.shared_inter, H, b);
        want_q(L(l, "mlp.shared_expert.up_proj"), {}, c_.shared_inter, H, b);
        want_q(L(l, "mlp.shared_expert.down_proj"), {}, H, c_.shared_inter, b);
      }
    }
    size_t other = 0;
    std::map<std::string, int> other_names;
    for (const auto &t : ck.tensors())
      if (classify(t.name).comp == Component::other) {
        ++other;
        if (other_names.size() < 8) other_names[t.name] = 1;
      }
    for (const auto &kv : other_names) bad.push_back("unclassified tensor " + kv.first);
    if (other > other_names.size()) bad.push_back("... " + std::to_string(other - other_names.size()) + " more unclassified");
    if (warnings) {
      warnings->push_back("MLX affine int" + std::to_string(c_.w_bits) + " g" + std::to_string(G) + ": w = q x scale + bias (bf16 scale and bias per group)");
      warnings->push_back("routed experts stacked per layer (mlp.switch_mlp.*, " + std::to_string(E) + " experts)");
    }
    return bad;
  }

  model::Budget budget(const model::Checkpoint &ck, const model::DeployParams &p) const override;

  // Placement (TP over the cards): q_proj by whole heads with k_proj and v_proj's kv-head slices appended (one W4 matrix
  // [q x hq | k x hkv | v x hkv] per card), o_proj along k; gate | up by intermediate row (one matrix), down along k, for
  // the dense MLP and the shared expert alike; the router bf16 replicated (routing must match on every card); norms, q / k
  // norms and sinks replicated; lm_head by vocabulary row; the 8-bit embedding replicated with its scales and biases (a
  // step gathers and dequantises its rows). The routed experts have their own store (not part of this plan).
  std::vector<model::DenseSpec> dense_plan(const model::Checkpoint &ck, const model::DeployParams &p) const override {
    using S = model::DenseSpec;
    (void)p;
    std::vector<S> out;
    auto add = [&](std::string name, S::Split split, S::Store store, Component comp, int layer) {
      if (!ck.find(name)) throw std::runtime_error("flint dense plan: missing " + name);
      S sp;
      sp.name = std::move(name);
      sp.split = split;
      sp.store = store;
      if (store == S::Store::w4) {
        sp.w4_layout = S::W4Layout::mlx;
        sp.w4_group = c_.w_group;
      }
      sp.comp = comp;
      sp.layer = layer;
      out.push_back(std::move(sp));
    };
    auto append = [&](std::vector<std::string> names) {
      for (const auto &n : names)
        if (!ck.find(n)) throw std::runtime_error("flint dense plan: missing " + n);
      out.back().append = std::move(names);
    };
    const auto R = S::Split::replicate;
    const auto raw = S::Store::raw, w4 = S::Store::w4;
    for (const char *part : {".weight", ".scales", ".biases"}) add(T(std::string("embed_tokens") + part), R, raw, Component::embedding, -1);
    add("lm_head.weight", S::Split::rows, w4, Component::lm_head, -1);
    add(T("norm.weight"), R, raw, Component::norm, -1);
    for (int l = 0; l < c_.layers; ++l) {
      add(L(l, "input_layernorm.weight"), R, raw, Component::norm, l);
      add(L(l, "post_attention_layernorm.weight"), R, raw, Component::norm, l);
      const auto A = Component::attention;
      add(L(l, "self_attn.q_proj.weight"), S::Split::rows, w4, A, l);
      append({L(l, "self_attn.k_proj.weight"), L(l, "self_attn.v_proj.weight")});
      add(L(l, "self_attn.o_proj.weight"), S::Split::cols, w4, A, l);
      add(L(l, "self_attn.q_norm.weight"), R, raw, A, l);
      add(L(l, "self_attn.k_norm.weight"), R, raw, A, l);
      if (c_.sinks) add(L(l, "self_attn.softmax_offset"), R, raw, A, l);
      if (c_.dense[static_cast<size_t>(l)]) {
        add(L(l, "mlp.gate_proj.weight"), S::Split::rows, w4, Component::dense_mlp, l);
        append({L(l, "mlp.up_proj.weight")});
        add(L(l, "mlp.down_proj.weight"), S::Split::cols, w4, Component::dense_mlp, l);
      } else {
        // (the shared expert is packed beside the routed experts by the MoE store, arch/flint/moe.hpp)
        add(L(l, "mlp.gate.weight"), R, S::Store::bf16, Component::router, l);
      }
    }
    return out;
  }

  Config c_;
};

// Per-card bytes: TP over the cards for every matrix (each routed expert split by its intermediate columns, so every card
// reads 1/C of every touched expert), norms / router / embedding replicated, the KV of every layer split by kv head.
// Sliding layers keep a fixed ring per slot (ring_tokens); only the full-attention layers' KV grows with the context.
model::Budget Flint::budget(const model::Checkpoint &ck, const model::DeployParams &p) const {
  if (p.cards < 1 || c_.kv_heads % p.cards || c_.q_heads % p.cards || c_.vocab % (16ll * p.cards) || c_.inter % (p.cards * 128) ||
      c_.moe_inter % (p.cards * 128) || c_.shared_inter % (p.cards * 128) || c_.q_dim() % (p.cards * 128))
    throw std::runtime_error("flint budget: the head / vocab / intermediate counts do not split over " + std::to_string(p.cards) + " cards");
  if (p.kv_bytes != 1 && p.kv_bytes != 2) throw std::runtime_error("kv_bytes must be 1 or 2");
  if (p.kv_shard == model::DeployParams::KvShard::tokens) throw std::runtime_error("flint budget: token-sharded KV is not modelled (heads only)");
  const double C = p.cards;
  auto bf16 = [&](double rows, double cols, double split) { return rows * cols * 2 / split; };
  // An affine int4 Linear on the card: codes + an fp16 scale and an fp16 bias per (row, group).
  const double per_w = c_.w_bits / 8.0 + 4.0 / c_.w_group;
  auto w4 = [&](double rows, double cols, double split) { return rows * cols / split * per_w; };

  model::Budget b;
  const double H = c_.hidden, QD = c_.q_dim(), KD = c_.kv_dim();
  const int moe = c_.moe_layers(), dense_l = c_.layers - moe, full = c_.full_layers(), slide = c_.sliding_layers();
  auto row = [&](std::string name, double resident, double step, std::string note) { b.rows.push_back({std::move(name), resident, step, std::move(note)}); };
  char gnote[64];
  std::snprintf(gnote, sizeof gnote, "MLX int4 g%d (scale + bias)", c_.w_group);
  {
    const double table = static_cast<double>(c_.vocab) * H * (c_.embed_bits / 8.0 + 4.0 / c_.w_group);
    if (p.embed_on_host) row("embedding", 0, 0, "pinned host table, a step gathers its rows over PCIe");
    else row("embedding", table, 0, "replicated int" + std::to_string(c_.embed_bits) + " table + group scales / biases");
  }
  {
    const double lm = w4(static_cast<double>(c_.vocab), H, C);
    row("lm_head", lm, lm, std::string(gnote) + "; vocab-parallel, each card a contiguous 1/C of the rows");
  }
  {
    const double per = w4(QD, H, C) + w4(KD, H, C) * 2 + w4(H, QD, C) + bf16(1, 2.0 * c_.head_dim + c_.q_heads, 1);
    row("attention x" + std::to_string(c_.layers), per * c_.layers, per * c_.layers, std::string(gnote) + "; heads split (q, k, v one GEMM)");
  }
  if (dense_l) {
    const double per = w4(2.0 * c_.inter, H, C) + w4(H, c_.inter, C);
    row("dense MLP x" + std::to_string(dense_l), per * dense_l, per * dense_l, std::string(gnote) + "; gate | up by intermediate row, down along k");
  }
  {
    const double router = bf16(c_.experts, H, 1) * moe;
    row("router x" + std::to_string(moe), router, router, "replicated, bf16 (routing must match on every card)");
  }
  {
    const double per = w4(2.0 * c_.shared_inter, H, C) + w4(H, c_.shared_inter, C);
    row("shared expert x" + std::to_string(moe), per * moe, per * moe, std::string(gnote) + "; split like the dense MLP");
  }
  {
    const double norms = bf16(2.0 * c_.layers + 1, H, 1);
    row("norms x" + std::to_string(2 * c_.layers + 1), norms, norms, "replicated bf16 vectors");
  }
  // KV per token per layer per card: K and V of the card's kv heads (+ one fp32 scale per token and head for 8-bit KV).
  const double kv_layer = (2.0 * c_.kv_heads * c_.head_dim * p.kv_bytes + (p.kv_bytes == 1 ? 2.0 * c_.kv_heads * 4 : 0)) / C;
  b.kv_bytes_per_token = full * kv_layer;
  const double ctx = static_cast<double>(p.context);
  {
    const double resident = b.kv_bytes_per_token * ctx * p.seqs;
    row("KV full attention (" + std::to_string(p.context) + " x " + std::to_string(p.seqs) + " slot)", resident, resident,
        std::to_string(full) + " layers, " + (p.kv_bytes == 1 ? "8-bit + scales" : "bf16") + ", split by KV head; a step reads every token");
  }
  {
    const double ring = static_cast<double>(ring_tokens(c_, p.rows));
    const double resident = slide * kv_layer * ring * p.seqs;
    const double step = slide * kv_layer * std::min(ctx, static_cast<double>(c_.window)) * p.seqs;
    row("KV sliding window (ring " + std::to_string(static_cast<int64_t>(ring)) + " x " + std::to_string(p.seqs) + " slot)", resident, step,
        std::to_string(slide) + " layers; a step reads <= " + std::to_string(c_.window) + " tokens per layer");
  }
  // Routed experts as the card stores them (gate | up | down, codes + fp16 scale and bias per group).
  {
    double eb = 0;
    for (const char *proj : {"gate_proj", "up_proj", "down_proj"}) {
      const std::string m = L(static_cast<int>(std::find(c_.dense.begin(), c_.dense.end(), false) - c_.dense.begin()), std::string("mlp.switch_mlp.") + proj);
      const model::TensorInfo &w = ck.at(m + ".weight");
      eb += static_cast<double>(w.bytes) + static_cast<double>(ck.at(m + ".scales").bytes) + static_cast<double>(ck.at(m + ".biases").bytes);
    }
    b.expert_bytes = eb / c_.experts;  // bf16 scales / biases become fp16: the same bytes
  }
  b.experts_per_card = static_cast<int64_t>(moe) * c_.experts / p.cards;
  b.unique_experts = p.unique_experts > 0 ? p.unique_experts
                                          : c_.experts * (1.0 - std::pow(1.0 - static_cast<double>(c_.topk) / c_.experts, p.rows * p.seqs));
  b.expert_step_bytes = b.unique_experts / C * moe * b.expert_bytes;

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
                      " rows); real routing is correlated and touches fewer");
  if (b.hot_experts_fit < b.experts_per_card)
    b.notes.push_back(std::to_string(b.experts_per_card - b.hot_experts_fit) + " experts per card do not fit in VRAM at this context");
  else
    b.notes.push_back("every routed expert fits in VRAM: " + std::to_string((room - b.experts_per_card * b.expert_bytes) / (1ull << 30)).substr(0, 5) +
                      " GiB left per card");
  if (p.drafter) b.notes.push_back("no drafter in this checkpoint: --drafter ignored");
  b.notes.push_back("not modelled: activation scratch, recorded graphs, mailboxes (covered by the reserve)");
  return b;
}

}  // namespace

qwen35::Config runtime_config(const ModelConfig &mc) {
  const Config f = Config::parse(mc);
  qwen35::Config c;
  c.prefix = kPrefix;
  c.norm_offset = false;  // w x norm(x)
  c.qk_norm = true;       // per-head q / k norms (plain)
  c.hidden = f.hidden;
  c.layers = f.layers;
  c.vocab = f.vocab;
  c.max_positions = f.max_positions;
  c.rms_eps = f.rms_eps;
  c.tie_embeddings = false;
  c.full_attention.assign(static_cast<size_t>(f.layers), true);  // every layer GQA (`sliding` says which kind)
  c.full_interval = 1;
  c.q_heads = f.q_heads;
  c.kv_heads = f.kv_heads;
  c.head_dim = f.head_dim;
  c.attn_output_gate = false;
  c.partial_rotary = 1.0;
  c.rope_theta = f.rope_theta;
  c.rope_type = "default";
  c.inter = f.inter;
  c.hidden_act = f.hidden_act;
  c.mtp_layers = 0;
  c.w_bits = f.w_bits;
  c.w_group = f.w_group;
  c.w_sym = false;
  c.w_method = "mlx";
  c.w_format = "affine";
  c.w_mlx = true;
  c.bos = f.bos;
  c.eos = f.eos;
  c.flint = true;
  c.sliding = f.sliding;
  c.window = f.window;
  c.moe.resize(f.dense.size());
  for (size_t l = 0; l < f.dense.size(); ++l) c.moe[l] = !f.dense[l];
  c.experts = f.experts;
  c.topk = f.topk;
  c.moe_inter = f.moe_inter;
  c.shared_inter = f.shared_inter;
  if (!f.sinks) invariant("the runtime is built for learned sinks (attention_sink true)");
  if (f.scoring != "sigmoid" || !f.norm_topk) invariant("the runtime's MoE is sigmoid scores, top-k renormalised");
  if (f.embed_bits != 8) invariant("the runtime gathers an 8-bit embedding");
  return c;
}

model::ArchEntry entry() {
  return {
      "flint",
      "Recursive Flint (flint): sliding-window + full GQA with sinks, 256-expert sigmoid MoE + shared expert, MLX affine int4",
      "Recursive Flint INT4 MoE",
      "flint",
      [](const ModelConfig &mc) {
        if (mc.model_type() == "flint") return true;
        for (const auto &a : mc.architectures())
          if (a == "FlintForCausalLM") return true;
        return false;
      },
      [](const ModelConfig &mc) -> std::unique_ptr<model::Arch> { return std::make_unique<Flint>(Config::parse(mc)); },
  };
}

}  // namespace xenithium::arch::flint
