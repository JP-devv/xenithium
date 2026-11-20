#include "xenithium/arch/qwen35/gdn.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace xenithium::arch::qwen35 {

GdnSet GdnSet::from(const load::DenseStore &dense, const Config &cfg, int cards, const std::vector<int> &only) {
  GdnSet g;
  g.shape = kernels::GdnShape{cfg.hidden, cfg.lin_k_heads, cfg.lin_v_heads, cfg.lin_k_dim, cfg.lin_v_dim, cards, 0, static_cast<float>(cfg.rms_eps),
                              cfg.output_gate == "swish"};
  g.w.assign(static_cast<size_t>(cards), std::vector<kernels::GdnWeights>(static_cast<size_t>(cfg.layers)));
  for (int l = 0; l < cfg.layers; ++l) {
    if (cfg.full_attention[l]) continue;
    if (!only.empty() && std::find(only.begin(), only.end(), l) == only.end()) continue;
    g.layers.push_back(l);
    for (int c = 0; c < cards; ++c) {
      const auto s = g.shape_of(c);
      const std::string L = std::string(kPrefix) + "layers." + std::to_string(l) + ".linear_attn.";
      auto raw = [&](const char *t) { return reinterpret_cast<const kernels::gdn_bf16 *>(dense.at(c, L + t).ptr); };
      auto w4 = [&](const char *mod) {
        const load::W4Weights v = dense.at(c, cfg.packed_name(L + mod)).w4();
        kernels::W4Mat m;
        m.w = v.w;
        m.scales = v.scales;
        m.n = v.n;
        m.k = v.k;
        return m;
      };
      kernels::GdnWeights w;
      w.qkvz4 = w4("in_proj_qkv");
      w.out4 = w4("out_proj");
      w.qkvz.n = w.qkvz4.n;  // strides for the core / state layouts
      w.qkvz.k = w.qkvz4.k;
      w.out.n = w.out4.n;
      w.out.k = w.out4.k;
      w.a = raw("in_proj_a.weight");
      w.b = raw("in_proj_b.weight");
      w.conv = raw("conv1d.weight");
      w.a_log = raw("A_log");
      w.dt_bias = raw("dt_bias");
      w.norm = raw("norm.weight");
      if (w.qkvz4.n != s.qkvz_n() || w.qkvz4.k != s.hidden || dense.at(c, L + "in_proj_a.weight").rows != s.vh() ||
          dense.at(c, L + "in_proj_b.weight").rows != s.vh() || w.out4.n != s.hidden || w.out4.k != s.val_w() ||
          dense.at(c, L + "conv1d.weight").rows != s.conv_ch())
        throw std::runtime_error("qwen35 gdn: unexpected shapes for layer " + std::to_string(l) + " (qkv needs z appended; a, b raw)");
      g.w[c][l] = w;
    }
  }
  return g;
}

}  // namespace xenithium::arch::qwen35
