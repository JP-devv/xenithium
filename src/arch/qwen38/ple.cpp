#include "xenithium/arch/qwen38/ple.hpp"

#include <stdexcept>
#include <string>

namespace xenithium::arch::qwen38 {

PleSet PleSet::from(const load::DenseStore &dense, const Config &cfg, int cards) {
  if (cfg.ple_layer < 0) throw std::runtime_error("qwen38 ple: no PLE layer");
  if (cfg.ple_conv_kernel != kernels::kPleTaps || cfg.ngram != kernels::kPleDil)
    throw std::runtime_error("qwen38 ple: the kernel is built for conv kernel 4, dilation (ngram) 3");
  PleSet p;
  p.layer = cfg.ple_layer;
  p.shape = kernels::PleShape{cfg.hidden, cfg.hc, cfg.ple_dim, static_cast<float>(cfg.rms_eps)};
  const std::string L = std::string(kPrefix) + "layers." + std::to_string(cfg.ple_layer) + ".ple.";
  for (int c = 0; c < cards; ++c) {
    auto raw = [&](const char *t) { return reinterpret_cast<const kernels::ple_bf16 *>(dense.at(c, L + t).ptr); };
    kernels::PleWeights w;
    const auto &kv = dense.at(c, L + "key_proj.weight");
    w.kv = kv.dense();
    w.nk = raw("norm_key.weight");
    w.nq = raw("norm_query.weight");
    w.ncw = raw("norm_conv.weight");
    w.conv = raw("conv1d.weight");
    if (kv.rows != p.shape.kv_n() || w.kv.k != p.shape.emb || dense.at(c, L + "conv1d.weight").rows != p.shape.width())
      throw std::runtime_error("qwen38 ple: unexpected shapes (key_proj needs value_proj appended)");
    p.w.push_back(w);
  }
  return p;
}

}  // namespace xenithium::arch::qwen38
