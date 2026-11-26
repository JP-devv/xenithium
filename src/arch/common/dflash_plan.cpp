// Xenithium · a DFlash 2 / DSpine drafter's tensor placement (see dflash_plan.hpp).
#include "xenithium/arch/common/dflash_plan.hpp"

#include <stdexcept>
#include <string>
#include <utility>

namespace xenithium::arch {

std::vector<model::DenseSpec> dflash_dense_plan(const model::Checkpoint &ck, int layers, int dense_bytes, bool dspine) {
  using S = model::DenseSpec;
  if (dense_bytes != 1 && dense_bytes != 2 && dense_bytes != 4) throw std::invalid_argument("dflash plan: dense_bytes 1 (int8), 2 (bf16) or 4 (int4 by RTN)");
  const S::Store mat = dense_bytes == 2 ? S::Store::bf16 : S::Store::i8;
  const S::Store big = dense_bytes == 4 ? S::Store::w4r : mat;  // q, o, gate | up, down
  std::vector<S> out;
  auto add = [&](std::string name, S::Split split, S::Store store, int layer, std::vector<std::string> append = {}) {
    if (!ck.find(name)) throw std::runtime_error("dflash plan: missing " + name);
    for (const auto &a : append)
      if (!ck.find(a)) throw std::runtime_error("dflash plan: missing " + a);
    S sp;
    sp.name = std::move(name);
    sp.split = split;
    sp.store = store;
    sp.append = std::move(append);
    sp.comp = model::Component::mtp;
    sp.layer = layer;
    out.push_back(std::move(sp));
  };
  // A Linear: `.weight` (bf16 / f16) keeps `intended`; `.qweight` is the GPTQ int4 path the 27B body
  // and MTP already use. The store is keyed as `<module>.weight` so the drafter's lookups stay the same.
  auto add_mat = [&](const std::string &module, S::Split split, S::Store intended, int layer, std::vector<std::string> append_mods = {}) {
    const bool gptq = ck.find(module + ".qweight") && !ck.find(module + ".weight");
    if (!gptq && !ck.find(module + ".weight")) throw std::runtime_error("dflash plan: missing " + module + ".weight or .qweight");
    std::vector<std::string> ap;
    for (const auto &m : append_mods) {
      const bool a_q = ck.find(m + ".qweight") && !ck.find(m + ".weight");
      if (!a_q && !ck.find(m + ".weight")) throw std::runtime_error("dflash plan: missing " + m + ".weight or .qweight");
      if (a_q != gptq) throw std::runtime_error("dflash plan: mixed .weight / .qweight append on " + module);
      ap.push_back(m + (a_q ? ".qweight" : ".weight"));
    }
    if (!gptq) {
      add(module + ".weight", split, intended, layer, std::move(ap));
      return;
    }
    if (intended == S::Store::w4r) {
      add(module + ".qweight", split, S::Store::w4, layer, std::move(ap));  // already int4: pack into the W4 GEMM layout
    } else {
      add(module + ".qweight", split, S::Store::w4d, layer, std::move(ap));  // fc (and other dense-only mats): dequant to mat precision
      out.back().dense_bytes = intended == S::Store::bf16 ? 2 : 1;
    }
    out.back().w4_layout = S::W4Layout::gptq;
    out.back().key = module + ".weight";
  };
  add_mat("fc", S::Split::cols, mat, 0);
  add("hidden_norm.weight", S::Split::replicate, S::Store::raw, 0);
  add("norm.weight", S::Split::replicate, S::Store::raw, layers - 1);
  if (!dspine) add_mat("candidate_selector.hidden_projection", S::Split::replicate, mat, layers - 1);
  for (int l = 0; l < layers; ++l) {
    const std::string L = "layers." + std::to_string(l) + ".";
    add(L + "input_layernorm.weight", S::Split::replicate, S::Store::raw, l);
    add(L + "post_attention_layernorm.weight", S::Split::replicate, S::Store::raw, l);
    add_mat(L + "self_attn.q_proj", S::Split::rows, big, l);
    add_mat(L + "self_attn.k_proj", S::Split::rows, mat, l, {L + "self_attn.v_proj"});
    add_mat(L + "self_attn.o_proj", S::Split::cols, big, l);
    add(L + "self_attn.q_norm.weight", S::Split::replicate, S::Store::raw, l);
    add(L + "self_attn.k_norm.weight", S::Split::replicate, S::Store::raw, l);
    add_mat(L + "mlp.gate_proj", S::Split::rows, big, l, {L + "mlp.up_proj"});
    add_mat(L + "mlp.down_proj", S::Split::cols, big, l);
    if (dspine) continue;  // DSpine: no convolution (its bridge loads in DFlash::load_dspine)
    add_mat(L + "attention_conv.kernel_projection", S::Split::replicate, mat, l);
    add(L + "attention_conv.base_kernel", S::Split::replicate, S::Store::raw, l);
    add_mat(L + "mlp_conv.kernel_projection", S::Split::replicate, mat, l);
    add(L + "mlp_conv.base_kernel", S::Split::replicate, S::Store::raw, l);
  }
  return out;
}

}  // namespace xenithium::arch
