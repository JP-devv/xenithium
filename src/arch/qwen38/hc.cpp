#include "xenithium/arch/qwen38/hc.hpp"

#include <stdexcept>
#include <string>

namespace xenithium::arch::qwen38 {

HcSet HcSet::from(const load::DenseStore &dense, const Config &cfg, int cards) {
  HcSet h;
  h.shape = kernels::HcShape{cfg.hc, cfg.hidden, cfg.hc_lowrank, static_cast<float>(cfg.rms_eps)};
  h.down_ld = (cfg.hc_lowrank + cfg.hc + 15) / 16 * 16;
  h.ple_layer = cfg.ple_layer;
  const std::string P = "model.language_model.";
  auto mixer = [&](int c, const std::string &pre, bool inject) {
    kernels::HcWeights w;
    w.norm = reinterpret_cast<const kernels::hc_bf16 *>(dense.at(c, pre + "hc_norm.weight").ptr);
    const auto &d = dense.at(c, pre + "input_mix_weight_down.weight");
    w.down = d.dense();
    w.up = dense.at(c, pre + "input_mix_weight_up.weight").dense();
    const int want_rows = inject ? cfg.hc_lowrank + cfg.hc : cfg.hc_lowrank;
    if (d.rows != want_rows || w.down.k != h.shape.width() || w.up.n != h.shape.width() || w.up.k != cfg.hc_lowrank)
      throw std::runtime_error("qwen38 hc: unexpected shapes for " + pre);
    if (inject && w.down.n != h.down_ld) throw std::runtime_error("qwen38 hc: " + pre + " down|inject is not padded to " + std::to_string(h.down_ld));
    return w;
  };
  h.attn.resize(static_cast<size_t>(cards));
  h.mlp.resize(static_cast<size_t>(cards));
  for (int c = 0; c < cards; ++c) {
    for (int l = 0; l < cfg.layers; ++l) {
      const std::string L = P + "layers." + std::to_string(l) + ".";
      h.attn[c].push_back(mixer(c, L + "attn_hyper_connection.", true));
      h.mlp[c].push_back(mixer(c, L + "mlp_hyper_connection.", true));
    }
    h.final_mixer.push_back(mixer(c, P + "hyper_connection_mixer.", false));
  }
  return h;
}

}  // namespace xenithium::arch::qwen38
