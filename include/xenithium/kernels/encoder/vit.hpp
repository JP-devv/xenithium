// Xenithium · the vision encoder's ops (the Qwen3-VL family's ViT: 27 pre-norm blocks of bidirectional attention +
// gelu-tanh MLP over an image's patches, then a 2 x 2 patch merger into the language model's hidden size).
//
// Own kernels (vit.cpp, compiled -fp-model=precise; bit-exact against the host references in vit_ref.cpp):
//   vit_pos_add:  x[p] = bf16(x[p] + bf16(((e0 + e1) + e2) + e3)),  e_k = bf16(table[idx[p][k]] x w[p][k])
//                 (the learned position grid sampled bilinearly: 4 rows, bf16 weights, each product rounded to bf16,
//                 the sum in fp32)
//   vit_rope:     q and k of qkv [n][3][heads][hd] rotated in place, NeoX halves over the whole head (hd / 2 angles per
//                 patch: hd / 4 row frequencies, then hd / 4 column ones; cos / sin from the host, fp32):
//                   x'[i]        = bf16(fma(x[i], c[i], -(x[i + hd/2] x s[i])))
//                   x'[i + hd/2] = bf16(fma(x[i + hd/2], c[i], x[i] x s[i]))
//   vit_layer_norm: y = bf16(fma((x - mean) x rstd, gamma, beta)), two passes in fp32 (mean, then the mean of
//                 (x - mean)^2: no cancellation on the tower's large-activation tokens), rstd = 1 / sqrt(var + eps)
//                 IEEE; each row's 16 lanes sum the columns c = lane + 16 j in order, then lane 0 the 16 partials in order
// oneDNN (vit_onednn.cpp, the XL library beside the dnnl link; not bit-exact against a host loop: the gate is the
// whole tower against golden outputs, tests/test_vision.cpp):
//   vit_linear:     out = bf16(act(a . w^T + bias) [+ res]): bf16 inputs, fp32 accumulation and post-ops, one rounding
//                   (none between the bias and the activation / residual add); w is the
//                   checkpoint's own [n][k] rows (read as a k x n column-major matrix: no repack, one stored copy)
//   vit_attention:  one image's bidirectional attention, oneDNN Graph's fused SDPA: s = (q . k^T) x hd^-0.5 in fp32,
//                   softmax -> bf16 p, o = p . v (bf16); q, k, v strided views of qkv, o [seq][heads][hd]
#pragma once

#include <cstddef>
#include <cstdint>

#include <sycl/sycl.hpp>

#include "xenithium/kernels/encoder/vit_ref.hpp"

namespace xenithium::kernels {

using vit_bf16 = sycl::ext::oneapi::bfloat16;

sycl::event vit_pos_add(sycl::queue &q, int n, int hidden, const vit_bf16 *table, const int32_t *idx, const vit_bf16 *w, vit_bf16 *x);
sycl::event vit_rope(sycl::queue &q, int n, int heads, int hd, const float *cs, const float *sn, vit_bf16 *qkv);
sycl::event vit_layer_norm(sycl::queue &q, int m, int n, const vit_bf16 *x, const float *gamma, const float *beta, float eps, vit_bf16 *y);

enum class VitAct : int { none = 0, gelu_tanh = 1, gelu_erf = 2 };

sycl::event vit_linear(sycl::queue &q, int m, int n, int k, const vit_bf16 *a, int lda, const vit_bf16 *w, const vit_bf16 *bias, VitAct act,
                       const vit_bf16 *res, vit_bf16 *out, int ldo);
sycl::event vit_attention(sycl::queue &q, int seq, int heads, int hd, const vit_bf16 *qkv, vit_bf16 *o);

}  // namespace xenithium::kernels
