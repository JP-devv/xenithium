#include "xenithium/arch/qwen38/gdn.hpp"

#include <stdexcept>
#include <string>

namespace xenithium::arch::qwen38 {

GdnSet GdnSet::from(const load::DenseStore &dense, const Config &cfg, int cards) {
  GdnSet g;
  g.shape = kernels::GdnShape{cfg.hidden, cfg.lin_k_heads, cfg.lin_v_heads, cfg.lin_k_dim, cfg.lin_v_dim, cards, 0, static_cast<float>(cfg.rms_eps)};
  g.w.assign(static_cast<size_t>(cards), std::vector<kernels::GdnWeights>(static_cast<size_t>(cfg.layers)));
  for (int l = 0; l < cfg.layers; ++l) {
    if (cfg.full_attention[l]) continue;
    g.layers.push_back(l);
    for (int c = 0; c < cards; ++c) {
      const auto s = g.shape_of(c);
      const std::string L = std::string(kPrefix) + "layers." + std::to_string(l) + ".linear_attn.";
      auto raw = [&](const char *t) { return reinterpret_cast<const kernels::gdn_bf16 *>(dense.at(c, L + t).ptr); };
      kernels::GdnWeights w;
      const load::DeviceTensor &tq = dense.at(c, L + "in_proj_qkv.weight"), &to = dense.at(c, L + "out_proj.weight");
      // an int4 matrix (ModelOptions:w4_tensors / gdn_w4) runs the W4 kernels; the int8 fields then
      // carry only (n, k) for the strides of the core / state layouts
      auto w4 = [](const load::DeviceTensor &t, kernels::DenseWeights &d) {
        const load::W4Weights v = t.w4();
        kernels::W4Mat m;
        m.w = v.w;
        m.scales = v.scales;
        m.n = v.n;
        m.k = v.k;
        d.n = v.n;
        d.k = v.k;
        d.kind = kernels::DenseKind::i8;
        return m;
      };
      if (tq.store == model::DenseSpec::Store::w4) w.qkvz4 = w4(tq, w.qkvz);
      else w.qkvz = tq.dense();
      if (to.store == model::DenseSpec::Store::w4) w.out4 = w4(to, w.out);
      else w.out = to.dense();
      w.a = raw("in_proj_a.weight");
      w.b = raw("in_proj_b.weight");
      w.conv = raw("conv1d.weight");
      w.a_log = raw("A_log");
      w.dt_bias = raw("dt_bias");
      w.norm = raw("norm.weight");
      if (dense.at(c, L + "in_proj_qkv.weight").rows != s.qkvz_n() || dense.at(c, L + "in_proj_a.weight").rows != s.vh() ||
          dense.at(c, L + "in_proj_b.weight").rows != s.vh() || w.qkvz.kind != kernels::DenseKind::i8 || w.out.n != s.hidden || w.out.k != s.val_w() ||
          dense.at(c, L + "conv1d.weight").rows != s.conv_ch())
        throw std::runtime_error("qwen38 gdn: unexpected shapes for layer " + std::to_string(l) + " (qkv needs z appended; a, b raw)");
      g.w[c][l] = w;
    }
  }
  return g;
}

}  // namespace xenithium::arch::qwen38
