// Xenithium · qwen38 routed experts: the on-card blob layout and its packer.
//
// One expert = one contiguous blob, so placement (hot / cold / lent) moves whole blobs:
//
//   [gate|up codes]  W4 tiles, k = hidden, n = 2 x inter (gate columns, then up columns)
//   [down codes]     W4 tiles, k = inter,  n = hidden
//   [gate|up scales] fp16 [hidden / 128][2 x inter]  (row g = gate scales | up scales)
//   [down scales]    fp16 [inter / 128][hidden]
//
// Codes first keeps every tile block 4 KiB-aligned inside the blob for this model's dims;
// the stride between blobs is rounded up to 4 KiB. The checkpoint's qzeros are not stored:
// the packer checks that every qzeros word is the symmetric zero and throws otherwise.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include "xenithium/arch/qwen38/arch.hpp"
#include "xenithium/model/bulk_read.hpp"

namespace xenithium::arch::qwen38 {

struct ExpertLayout {
  int hidden = 0, inter = 0;
  size_t gu_codes = 0, dn_codes = 0, gu_scales = 0, dn_scales = 0;  // section offsets
  size_t bytes = 0;   // payload
  size_t stride = 0;  // bytes rounded up to 4 KiB: blob i starts at i * stride
  static ExpertLayout of(const Config &c);
};

// The 9 checkpoint tensors of one expert (gate/up/down x qweight/qzeros/scales), in a
// fixed order; for bulk reads.
std::vector<const model::TensorInfo *> expert_tensors(const model::Checkpoint &ck, int layer, int expert);

// Packs expert (layer, e) from bytes already read into `buf` into `out` (layout.stride
// bytes, the padding zeroed). Throws on a non-symmetric qzeros word.
void pack_expert(const ExpertLayout &lay, const model::Checkpoint &ck, const model::BulkBuffer &buf, int layer, int expert,
                 uint8_t *out);

std::string expert_name(int layer, int expert, const char *proj, const char *part);

}  // namespace xenithium::arch::qwen38
