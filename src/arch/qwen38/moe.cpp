#include "xenithium/arch/qwen38/moe.hpp"

#include <stdexcept>
#include <string>

namespace xenithium::arch::qwen38 {

kernels::MoeBlobLayout moe_blob_layout(const ExpertLayout &lay) {
  kernels::MoeBlobLayout b;
  b.gu_codes = lay.gu_codes;
  b.dn_codes = lay.dn_codes;
  b.gu_scales = lay.gu_scales;
  b.dn_scales = lay.dn_scales;
  return b;
}

MoeSet MoeSet::from(const load::DenseStore &dense, const Config &cfg, int cards, bool tp) {
  MoeSet m;
  m.shape = kernels::MoeShape{cfg.hidden, cfg.moe_inter, cfg.experts, cfg.topk, cfg.shared_inter, cards, 0};
  m.shape.tp = tp;
  const auto blob = tp ? kernels::moe_tp_layout(cfg.hidden, cfg.moe_inter, cards, nullptr) : moe_blob_layout(ExpertLayout::of(cfg));
  m.w.resize(static_cast<size_t>(cards));
  for (int c = 0; c < cards; ++c)
    for (int l = 0; l < cfg.layers; ++l) {
      const std::string L = std::string(kPrefix) + "layers." + std::to_string(l) + ".mlp.";
      kernels::MoeWeights w;
      const auto &r = dense.at(c, L + "gate.weight");
      w.router = r.dense();
      w.shared_gu = dense.at(c, L + "shared_expert.gate_proj.weight").dense();
      w.shared_down = dense.at(c, L + "shared_expert.down_proj.weight").dense();
      w.blob = blob;
      if (r.rows != cfg.experts + 1 || w.router.kind != kernels::DenseKind::bf16 || w.shared_gu.n != 2 * cfg.shared_inter ||
          w.shared_down.n != cfg.hidden || w.shared_down.k != cfg.shared_inter)
        throw std::runtime_error("qwen38 moe: unexpected shapes for layer " + std::to_string(l) +
                                 " (router needs the shared gate row, shared gate needs up appended)");
      m.w[c].push_back(w);
    }
  return m;
}

}  // namespace xenithium::arch::qwen38
