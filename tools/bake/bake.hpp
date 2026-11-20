// xenithium-bake: shared pieces of the tool's subcommands.
#pragma once

#include <cstdlib>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "xenithium/model/checkpoint.hpp"
#include "xenithium/quant/autoround.hpp"

namespace xenithium::bake {

struct Args {
  std::map<std::string, std::string> kv;
  bool has(const char *k) const { return kv.count(k) != 0; }
  std::string str(const char *k, const std::string &d = "") const { return has(k) ? kv.at(k) : d; }
  long num(const char *k, long d) const { return has(k) ? std::strtol(kv.at(k).c_str(), nullptr, 10) : d; }
};

inline std::string home() {
  const char *h = std::getenv("HOME");
  return h ? h : ".";
}

// Bytes [off, off + bytes) of a tensor's data (pread on its shard).
void read_tensor_range(const model::Checkpoint &ck, const model::TensorInfo &t, uint64_t off, uint64_t bytes, void *dst);

// --gpus N (default: every GPU found). Tuning runs on the GPUs only; the host cores do the
// streaming, packing and per-step reduction around them.
std::vector<std::unique_ptr<quant::ArWorker>> make_workers(const Args &a);

int cmd_synth(const Args &a);
int cmd_rtn(const Args &a);
int cmd_check_layout(const Args &a);
int cmd_ple(const Args &a);
int cmd_selftest_calib(const Args &a);
int cmd_tune(const Args &a);
int cmd_assemble(const Args &a);
int cmd_pack_native(const Args &a);
int cmd_mtp(const Args &a);
int cmd_dense_tune(const Args &a);
int cmd_dense_assemble(const Args &a);  // dense.cpp: AutoRound W4A8 of a dense model's MLPs from its W4A16 export

// The layer's fused BF16 experts: gate_up [E][2I][H], down [E][H][I] (parallel preads).
void read_layer_experts(const model::Checkpoint &ck, int layer, const quant::ArShape &s, std::vector<uint16_t> &gu,
                        std::vector<uint16_t> &dn);
// The same from any fused pair under prefix P (P + "gate_up_proj", P + "down_proj"; the drafter's
// "mtp.layers.0.mlp.experts.").
// Fused or per-expert BF16 experts into read_fused_experts' layout.
void read_mtp_experts_any(const model::Checkpoint &ck, const std::string &P, const quant::ArShape &s, std::vector<uint16_t> &gu,
                          std::vector<uint16_t> &dn);
void read_fused_experts(const model::Checkpoint &ck, const std::string &P, const quant::ArShape &s, std::vector<uint16_t> &gu,
                        std::vector<uint16_t> &dn);

int cmd_mtp_tune(const Args &a);
int cmd_mtp_states(const Args &a);
int cmd_head_capture(const Args &a);
int cmd_head_calib(const Args &a);
int cmd_prompts(const Args &a);
int cmd_head4_calib(const Args &a);
int cmd_gdn4_calib(const Args &a);
int cmd_kld(const Args &a);

}  // namespace xenithium:bake
