// Xenithium · host references of the vision encoder's own kernels (semantics in xenithium/kernels/encoder/vit.hpp);
// bf16 as raw bits. Host-only (xenithium_pack, -fp-model=precise).
#pragma once

#include <cstdint>

namespace xenithium::kernels {

void vit_pos_add_ref(int n, int hidden, const uint16_t *table, const int32_t *idx, const uint16_t *w, uint16_t *x);
void vit_rope_ref(int n, int heads, int hd, const float *cs, const float *sn, uint16_t *qkv);
void vit_layer_norm_ref(int m, int n, const uint16_t *x, const float *gamma, const float *beta, float eps, uint16_t *y);

}  // namespace xenithium::kernels
