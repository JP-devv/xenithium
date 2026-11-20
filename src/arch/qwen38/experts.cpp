#include "xenithium/arch/qwen38/experts.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <stdexcept>

#include "xenithium/kernels/pack.hpp"

namespace xenithium::arch::qwen38 {

namespace k = xenithium::kernels;

ExpertLayout ExpertLayout::of(const Config &c) {
  ExpertLayout l;
  l.hidden = c.hidden;
  l.inter = c.moe_inter;
  const size_t H = static_cast<size_t>(c.hidden), I = static_cast<size_t>(c.moe_inter);
  l.gu_codes = 0;
  l.dn_codes = l.gu_codes + H * 2 * I / 2;
  l.gu_scales = l.dn_codes + I * H / 2;
  l.dn_scales = l.gu_scales + H / k::kW4Group * 2 * I * 2;
  l.bytes = l.dn_scales + I / k::kW4Group * H * 2;
  l.stride = (l.bytes + 4095) / 4096 * 4096;
  if (l.dn_codes % 4096 || l.gu_scales % 512) throw std::runtime_error("qwen38 expert layout: sections not aligned for these dims");
  return l;
}

std::string expert_name(int layer, int expert, const char *proj, const char *part) {
  return std::string(kPrefix) + "layers." + std::to_string(layer) + ".mlp.experts." + std::to_string(expert) + "." + proj + "." +
         part;
}

std::vector<const model::TensorInfo *> expert_tensors(const model::Checkpoint &ck, int layer, int expert) {
  std::vector<const model::TensorInfo *> v;
  for (const char *proj : {"gate_proj", "up_proj", "down_proj"})
    for (const char *part : {"qweight", "qzeros", "scales"}) v.push_back(&ck.at(expert_name(layer, expert, proj, part)));
  return v;
}

void pack_expert(const ExpertLayout &lay, const model::Checkpoint &ck, const model::BulkBuffer &buf, int layer, int expert,
                 uint8_t *out) {
  const int H = lay.hidden, I = lay.inter;
  auto t = [&](const char *proj, const char *part) -> const model::TensorInfo & {
    return ck.at(expert_name(layer, expert, proj, part));
  };
  for (const char *proj : {"gate_proj", "up_proj", "down_proj"}) {
    const auto &z = t(proj, "qzeros");
    const uint32_t *w = buf.as<uint32_t>(z);
    for (size_t i = 0; i < z.bytes / 4; ++i)
      if (w[i] != k::kW4ZeroWord) throw std::runtime_error(z.name + ": qzeros word " + std::to_string(i) + " is not symmetric");
  }
  k::pack_w4_gptq_cat(buf.as<uint32_t>(t("gate_proj", "qweight")), buf.as<uint32_t>(t("up_proj", "qweight")), I, 2 * I, H,
                      out + lay.gu_codes);
  k::pack_w4_gptq(buf.as<uint32_t>(t("down_proj", "qweight")), H, I, out + lay.dn_codes);
  // gate|up scales: row g = gate row g | up row g (fp16 bytes copied, never converted).
  const uint8_t *gs = buf.data(t("gate_proj", "scales")), *us = buf.data(t("up_proj", "scales"));
  const size_t row = static_cast<size_t>(I) * 2;
  for (int g = 0; g < H / k::kW4Group; ++g) {
    std::memcpy(out + lay.gu_scales + g * 2 * row, gs + g * row, row);
    std::memcpy(out + lay.gu_scales + g * 2 * row + row, us + g * row, row);
  }
  std::memcpy(out + lay.dn_scales, buf.data(t("down_proj", "scales")), t("down_proj", "scales").bytes);
  if (lay.stride > lay.bytes) std::memset(out + lay.bytes, 0, lay.stride - lay.bytes);
}

}  // namespace xenithium::arch::qwen38
