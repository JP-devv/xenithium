// Xenithium · qwen38 MTP drafter, host side: its dense plan and its routed experts.
//
// The drafter's weights live in their own checkpoint directory (runtime/mtp-int4-g32: one MTP
// layer + its front and final mixer, routed experts in compressed-tensors int4 group 32). Its
// embedding and lm_head are the target's (byte-identical in the snapshot), so they are not loaded.
//
// Placement follows the target's: dense matrices int8 per row (the
// router bf16), attention heads split over the cards as the target's QSA layers, the mixers and
// the front replicated; routed experts requantised g32 -> g128 into the target's blob layout, so
// the target's MoE kernel (and its expert TP split) runs them unchanged.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "xenithium/arch/qwen38/experts.hpp"
#include "xenithium/model/arch.hpp"
#include "xenithium/model/checkpoint.hpp"

namespace xenithium::arch::qwen38 {

inline constexpr const char *kMtpPrefix = "mtp.";
inline constexpr const char *kMtpLayer = "mtp.layers.0.";

// Every non-expert tensor of the drafter with its placement (the DenseStore input). kv_by_tokens: the KV split by
// tokens (TP4, kernels/attention/qsa.hpp): its q_proj slices carry the partner's q heads too.
std::vector<model::DenseSpec> mtp_dense_plan(const model::Checkpoint &ck, bool kv_by_tokens = false);

// One int4 matrix [n][k]: codes in [-8, 7] and one scale per (row, group of G).
struct W4Rows {
  int n = 0, k = 0, G = 32;
  std::vector<int8_t> q;   // [n][k]
  std::vector<float> s;    // [n][k / G]
};
// compressed-tensors pack-quantized: int32 [n][k / 8], nibble j = column 8 i + j, stored q + 8;
// scales bf16 [n][k / 32].
W4Rows mtp_unpack_g32(const uint8_t *packed, const uint16_t *scale_bf16, int n, int k);
// Re-quantise to groups of 128 (the target's expert format: symmetric int4): per group the scale
// with the least squared error among bf16(amax / 7 x {1, 0.98 .. 0.80}), codes clamped to [-8, 7].
// xenithium-mtp-ref uses this same function, so the drafter on the cards and its reference run the
// same codes.
W4Rows mtp_requant_g128(const W4Rows &a);
// The full blob (ExpertLayout: GPTQ tiles, fp16 scales) of one expert from its g128 matrices
// (gate, up [inter][hidden], down [hidden][inter]). Returns how many scales are not exact in fp16.
int64_t mtp_pack_expert(const ExpertLayout &lay, const W4Rows &gate, const W4Rows &up, const W4Rows &down, uint8_t *out);
// fp16 bits of a float (round to nearest even; overflow -> inf).
uint16_t fp16_bits(float v);
float fp16_value(uint16_t h);

std::string mtp_expert_name(int expert, const char *proj, const char *part);

}  // namespace xenithium::arch::qwen38
