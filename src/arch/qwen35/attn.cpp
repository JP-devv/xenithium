#include "xenithium/arch/qwen35/attn.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace xenithium::arch::qwen35 {

AttnSet AttnSet::from(const load::DenseStore &dense, const Config &cfg, int cards, kernels::QsaKv kv, const std::vector<int> &only) {
  const bool qwen = cfg.head_dim == kernels::kQsaDim && cfg.attn_output_gate && cfg.qk_norm && cfg.partial_rotary * cfg.head_dim == kernels::kQsaRot;
  const bool llama = cfg.head_dim == 128 && !cfg.attn_output_gate && !cfg.qk_norm && cfg.partial_rotary == 1.0 && cfg.q_heads == 7 * cfg.kv_heads;
  const bool flint = cfg.flint && cfg.head_dim == 128 && !cfg.attn_output_gate && cfg.qk_norm && cfg.partial_rotary == 1.0 && cfg.q_heads == 8 * cfg.kv_heads &&
                     static_cast<int>(cfg.sliding.size()) == cfg.layers;
  if (!qwen && !llama && !flint)
    throw std::runtime_error("qwen35 attention: the kernels are built for head_dim 256 + output gate + q/k norms + 64 rotary dims (qwen35), "
                             "head_dim 128, 7 q heads per kv head, full rotary, no gate / norms (llama), or head_dim 128, 8 q heads per kv head, "
                             "plain q / k norms, sinks, sliding rotary + full NoPE layers (flint)");
  AttnSet a;
  a.shape = kernels::GqaShape{cfg.hidden, cfg.q_heads, cfg.kv_heads, cards, 0, static_cast<float>(cfg.rms_eps), kv};
  if (llama) a.shape.head_dim = 128, a.shape.gate = false, a.shape.qk_norm = false, a.shape.rot = 128;
  if (flint) {
    a.shape.head_dim = 128, a.shape.gate = false, a.shape.qk_norm = true, a.shape.norm_plus1 = false, a.shape.rot = 128;
    a.shape.split_merge = true;  // the merge across its items
    for (int l = 0; l < cfg.layers; ++l) {
      a.window.push_back(cfg.sliding[static_cast<size_t>(l)] ? cfg.window : 0);
      a.rot.push_back(cfg.sliding[static_cast<size_t>(l)] ? 128 : 0);
    }
  }
  a.w.assign(static_cast<size_t>(cards), std::vector<kernels::GqaWeights>(static_cast<size_t>(cfg.layers)));
  for (int l = 0; l < cfg.layers; ++l) {
    if (!cfg.full_attention[l]) continue;
    if (!only.empty() && std::find(only.begin(), only.end(), l) == only.end()) continue;
    a.layers.push_back(l);
    for (int c = 0; c < cards; ++c) {
      const auto s = a.shape_of(c);
      const std::string L = cfg.prefix + "layers." + std::to_string(l) + ".self_attn.";
      auto w4 = [&](const char *mod) {
        const load::W4Weights v = dense.at(c, cfg.packed_name(L + mod)).w4();
        kernels::W4Mat m;
        m.w = v.w;
        m.scales = v.scales;
        m.n = v.n;
        m.k = v.k;
        return m;
      };
      kernels::GqaWeights w;
      w.qkv = w4("q_proj");
      w.o = w4("o_proj");
      if (cfg.qk_norm) {
        w.q_norm = reinterpret_cast<const kernels::gqa_bf16 *>(dense.at(c, L + "q_norm.weight").ptr);
        w.k_norm = reinterpret_cast<const kernels::gqa_bf16 *>(dense.at(c, L + "k_norm.weight").ptr);
      }
      if (flint)  // the replicated [q_heads] sinks: this card's heads
        w.sinks = reinterpret_cast<const kernels::gqa_bf16 *>(dense.at(c, L + "softmax_offset").ptr) + static_cast<size_t>(c) * s.hq();
      if (w.qkv.n != s.qkv_n() || w.qkv.k != s.hidden || w.o.n != s.hidden || w.o.k != s.o_k())
        throw std::runtime_error("qwen35 attention: unexpected shapes for layer " + std::to_string(l) + " (q_proj needs k | v appended)");
      a.w[c][l] = w;
    }
  }
  return a;
}

}  // namespace xenithium::arch::qwen35
