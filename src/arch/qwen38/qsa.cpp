#include "xenithium/arch/qwen38/qsa.hpp"

#include <stdexcept>
#include <string>

namespace xenithium::arch::qwen38 {

QsaSet QsaSet::from(const load::DenseStore &dense, const Config &cfg, int cards, kernels::QsaKv kv) {
  if (cfg.head_dim != kernels::kQsaDim || cfg.idx_head_dim != kernels::kQsaIdxDim || cfg.idx_heads != kernels::kQsaIdxHeads ||
      cfg.idx_kv_heads != 1 || cfg.idx_compress != kernels::kQsaCompress || cfg.idx_budget != kernels::kQsaTopBlocks * kernels::kQsaCompress ||
      static_cast<int>(cfg.partial_rotary * cfg.head_dim + 0.5) != kernels::kQsaRot || cfg.output_gate != "sigmoid")
    throw std::runtime_error("qwen38 qsa: the kernels are built for head_dim 256, rotary 64, indexer 4 x 128 / compress 4 / budget 2048, "
                             "sigmoid output gate");
  QsaSet g;
  g.shape = kernels::QsaShape{cfg.hidden, cfg.q_heads, cfg.kv_heads, cards, 0, static_cast<float>(cfg.rms_eps), kv};
  g.w.assign(static_cast<size_t>(cards), std::vector<kernels::QsaWeights>(static_cast<size_t>(cfg.layers)));
  for (int l = 0; l < cfg.layers; ++l) {
    if (!cfg.full_attention[l]) continue;
    g.layers.push_back(l);
    for (int c = 0; c < cards; ++c) {
      const auto s = g.shape_of(c);
      const std::string L = std::string(kPrefix) + "layers." + std::to_string(l) + ".self_attn.";
      auto raw = [&](const std::string &t) { return reinterpret_cast<const kernels::qsa_bf16 *>(dense.at(c, L + t).ptr); };
      kernels::QsaWeights w;
      w.qkv = dense.at(c, L + "q_proj.weight").dense();
      w.idx = dense.at(c, L + "indexer.index_qk_proj.weight").dense();
      w.o = dense.at(c, L + "o_proj.weight").dense();
      w.q_norm = raw("q_norm.weight");
      w.k_norm = raw("k_norm.weight");
      w.iq_norm = raw("indexer.q_layernorm.weight");
      w.ik_norm = raw("indexer.k_layernorm.weight");
      if (dense.at(c, L + "q_proj.weight").rows != s.qkv_n() || w.qkv.n != s.qkv_n() || w.idx.n != s.idx_n() ||
          w.idx.kind != kernels::DenseKind::bf16 || w.o.n != s.hidden || w.o.k != s.o_k())
        throw std::runtime_error("qwen38 qsa: unexpected shapes for layer " + std::to_string(l) + " (q_proj needs k and v appended)");
      g.w[c][l] = w;
    }
  }
  return g;
}

}  // namespace xenithium::arch::qwen38
