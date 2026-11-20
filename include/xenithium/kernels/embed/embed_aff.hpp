// Xenithium · the affine 8-bit embedding gather (arch flint): out[r][c] = bf16(fma(q, s, b)) with q = table[id][c]
// (0..255), s / b the bf16 scale and bias of the row's 64-column group. The table stays 8-bit on the card (V x H bytes +
// 4 bytes per 64 columns); a step gathers its rows.
#pragma once

#include <cstdint>
#include <sycl/sycl.hpp>

namespace xenithium::kernels {

inline constexpr int kEmbedAffGroup = 64;

// ids: device int32 [rows]; table U8 [vocab][hidden]; scales / biases bf16 bits [vocab][hidden / 64]; out bf16 [rows][hidden]
// at row stride ldo. An id outside [0, vocab) writes zeros.
sycl::event embed_aff8(sycl::queue &q, int rows, const int32_t *ids, const uint8_t *table, const uint16_t *scales, const uint16_t *biases,
                       int64_t vocab, int hidden, sycl::ext::oneapi::bfloat16 *out, int ldo);

// Host reference (the same expression and rounding): one row.
void embed_aff8_ref(int32_t id, const uint8_t *table, const uint16_t *scales, const uint16_t *biases, int64_t vocab, int hidden, uint16_t *out);

}  // namespace xenithium::kernels
