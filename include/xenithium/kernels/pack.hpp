// Xenithium · weight packers and exact CPU references (host-only: no SYCL).
// The packed layouts are what the DPAS kernels read directly; they are defined here once
// and used by the kernels, the loader (which packs ~24K experts per load) and the tests.
// Built with precise floating point: host references must round like IEEE.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace xenithium::kernels {

// ---- int8 VNNI (W8A8) --------------------------------------------------------------
inline constexpr int kI8Width = 16;  // DPAS exec width on Xe2: columns per block
inline constexpr int kI8KPerCall = 32;  // int8 k per DPAS call
inline constexpr int kI8BlockBytes = 512;  // one VNNI B block

// W (n x k, row-major int8) -> VNNI blocks of 512 bytes, [k/32][n/16][512].
// Inside a block: dword index = (k%32)/4 * 16 + lane (lane = column t*16+lane),
// byte 0..3 of that dword = k%4. This is the layout DPAS reads directly.
void pack_weights_vnni(const int8_t *w, int n, int k, int8_t *out);
// Exact CPU reference: C (m x n int32) = A (m x k) * W^T (W n x k, row-major).
void gemm_int8_ref(const int8_t *a, const int8_t *w, int m, int n, int k, int32_t *c);

// ---- int4 GPTQ (W4A16) --------------------------------------------------------------
// Checkpoint format (auto_round:auto_gptq, symmetric, group 128): qweight uint32 [k/8][n],
// nibble i of row r is k = 8r + i; w[k][n] = (q - 8) * scale[k / 128][n].
inline constexpr int kW4Group = 128;  // GPTQ group size (k)
inline constexpr int kW4Zero = 8;  // stored zero 7, +1 (GPTQ v1)
inline constexpr uint32_t kW4ZeroWord = 0x77777777u;  // qzeros word of a symmetric checkpoint
inline constexpr int kW4Width = 16;  // DPAS exec width, columns per tile
inline constexpr int kW4BlockBytes = kW4Group * kW4Width / 2;  // 1024: one (group, tile)

// GPTQ qweight -> uint8 [k/128][n/16][1024]: inside a block, byte (kg/2)*16 + lane holds
// column t*16 + lane at kg (low nibble, kg even) and kg+1 (high nibble), kg = k % 128.
// Each 128-byte slice is one bf16 VNNI DPAS operand after widening.
void pack_w4_gptq(const uint32_t *qweight, int n, int k, uint8_t *out);
// The same pack of two [k/8][n_a] and [k/8][n - n_a] matrices concatenated along N
// (gate | up), without building the concatenation. b may be null when n_a == n.
void pack_w4_gptq_cat(const uint32_t *a, const uint32_t *b, int n_a, int n, int k, uint8_t *out);
// pack_w4_gptq for the output columns [r0, r1) only (multiples of 16: whole tiles), so a slice packs on threads.
void pack_w4_gptq_rows(const uint32_t *qweight, int n, int k, int r0, int r1, uint8_t *out);

// CPU reference from the unpacked checkpoint tensors, accumulated in double per group.
// With integer activations and power-of-two scales every intermediate is exact in fp32,
// so the device result must match bit for bit; with real data tests bound the error.
void w4a16_ref(const float *a, const uint32_t *qweight, const float *scales, int m, int n, int k, double *c);

// ---- int4 with zero points, group 32 (compressed-tensors asymmetric: arch llama, Nanbeige4.3) ---------
// w = (q - z) x s per (row n, 32-k group): q 0..15 as stored (no offset), z 0..15, s fp16. On the card the codes are
// the W4 blocks of pack_w4_ct_rows (raw q), the scales fp16 [k/32][n] and the zero points uint8 [k/64][n] (low nibble =
// group 2h, high nibble = group 2h + 1): pack_w4_zeros. The decode GEMM widens q - z per 32-k group (exact in bf16:
// |q - z| <= 15) and applies the group's scale to its fp32 sum.
inline constexpr int kW4ZpGroup = 32;
// CPU reference: packed = compressed-tensors words [n][k/8] (raw q), scales [n][k/32], zeros [n][k/32] (one per byte);
// c[m][n] = sum over the groups in order of (double) (sum_{k in g} a (q - z)) x s.
void w4zp_ref(const float *a, const uint32_t *packed, const float *scales, const uint8_t *zeros, int m, int n, int k, double *c);
// zero points uint8 [n][k/32] (one per byte) -> the card's nibbles [k/64][n].
void pack_w4_zeros(const uint8_t *z, int n, int k, uint8_t *out);

// ---- affine int4, group 64 (arch flint) ----------------------------------------------------------
// w = q x s + b per (row n, 64-k group): q 0..15 raw, s and b bf16. On the card: the W4 blocks of pack_w4_ct_rows (the
// checkpoint's U8 [n][k/2] rows are ct words), scales and biases bf16 [k/64][n].
inline constexpr int kW4AffGroup = 64;
// CPU reference: packed = ct words [n][k/8] (raw q), scales / biases fp32 [n][k/64] (bf16 values);
// c[m][n] = sum over the groups in order of (double) s x (sum_{k in g} a q) + (double) b x (sum_{k in g} a).
void w4aff_ref(const float *a, const uint32_t *packed, const float *scales, const float *biases, int m, int n, int k, double *c);

// ---- W4A8 ----------------------------------------------------------------------------------
// W4A8 CPU reference: c[m][n] = sum over the 128-groups g in order of (double) (sum_{k in g} aq[r][k] (q - 8)) x as[g][r]
// x scale[g][n]; aq int8 [m][k], as group-major [k/128][lds] (row_quant_rows' layout), qweight / scales as w4a16_ref.
void w4a8_ref(const int8_t *aq, const float *as, int lds, const uint32_t *qweight, const float *scales, int m, int n, int k, double *c);

// ---- int4 compressed-tensors "pack-quantized" (the qwen35 checkpoints) ---------
// The same codes as GPTQ (stored q + 8, code i of a word at bits 4i, k contiguous) in the other
// orientation: weight_packed uint32 [n][k/8] (word j of row n holds k = 8j..8j+7) and weight_scale
// fp16 [n][k/128]. These bridges re-express a matrix in the GPTQ orientation the packers and
// references above take, so one block layout serves both checkpoint families.
// gptq_out uint32 [k/8][n]; scales_out fp16 bits [k/128][n].
void ct_to_gptq_qweight(const uint32_t *packed, int n, int k, uint32_t *gptq_out);
void ct_to_gptq_scales(const uint16_t *scales, int n, int k, uint16_t *scales_out);
// Direct packer: compressed-tensors words -> the W4 blocks of pack_w4_gptq (bytes identical to
// pack_w4_gptq(ct_to_gptq_qweight(...))), without the intermediate matrix. Rows [r0, r1) of the
// output columns only (r1 - r0 and r0 multiples of 16: whole tiles), so slices pack on threads.
void pack_w4_ct_rows(const uint32_t *packed, int n, int k, int r0, int r1, uint8_t *out);
// Code q (0..15) of element (row n, k) straight from the compressed-tensors words.
inline int ct_code(const uint32_t *packed, int k_dim, int row, int kk) {
  return static_cast<int>((packed[static_cast<size_t>(row) * (k_dim / 8) + kk / 8] >> (4 * (kk % 8))) & 15u);
}

// ---- dense weights (W8A16 / W16A16 decode GEMM) -------------------------------------
// The checkpoint's [n][k] matrices (n = outputs) packed into DPAS tiles: block (128-k group
// g, 16-column tile t) holds 8 calls of 16 k x 16 columns in bf16 VNNI order, element
// (kin, lane) of call j at j*256 + (kin/2)*32 + lane*2 + (kin&1). bf16: 2 bytes per element
// (4096-byte blocks); int8: 1 byte per element (2048-byte blocks), widened to bf16 exactly
// on the card (|q| <= 127 has <= 7 significant bits). Full groups come first, [g][t];
// a k % 128 tail of `calls` x 16 k follows as [t][calls x 256 elements] (no padding read).
// n must be a multiple of 16 and k of 16.
inline constexpr int kDenseCalls = 8;  // 16-k DPAS calls per 128-k group
inline constexpr int kDenseCallElems = 256;  // 16 k x 16 columns
size_t dense_packed_elems(int n, int k);  // elements (bytes for int8)
void pack_dense_bf16(const uint16_t *w, int n, int k, uint16_t *out);
void pack_dense_i8(const int8_t *codes, int n, int k, int8_t *out);
// Rows [r0, r1) only (disjoint outputs: row ranges may be packed on separate threads).
void pack_dense_bf16_rows(const uint16_t *w, int n, int k, int r0, int r1, uint16_t *out);
void pack_dense_i8_rows(const int8_t *codes, int n, int k, int r0, int r1, int8_t *out);

// Symmetric per-output-row int8: scale = amax / 127 (or the clip that minimises the row's
// squared error over `clips` candidate ratios in [0.75, 1]; 1 = plain amax), q = rne(w / scale)
// clamped to [-127, 127]. w is bf16 bits; scale[n] fp32.
void quantize_i8_rows(const uint16_t *w, int n, int k, int8_t *codes, float *scale, int clips = 1);
// FP8 E4M3 per-row (scale = amax / 448, round to nearest even, saturating); for comparison.
void quantize_e4m3_rows(const uint16_t *w, int n, int k, uint8_t *codes, float *scale);
float e4m3_value(uint8_t code);
float bf16_value(uint16_t bits);
uint16_t bf16_bits(float v);  // round to nearest even
// dst[i] = bf16_value(bf16_bits(src[i])) for n floats, bit for bit (NaN stays NaN with the quiet bit set), branch-free
// so the loop vectorises. dst may equal src.
// Engine (DenseSpec:Store:w4r): one BF16 row of k values (k % 128 == 0) -> compressed-tensors words (code i
// of word j = column 8 j + i at bits 4 i, stored q + 8) and fp16 scale bits per 128-group. Plain (mse false): s =
// fp16(-extreme / 8), the group's largest |w| on code -8 (the AutoRound sign convention, == xenithium-bake head4-calib's
// RTN); mse: s = fp16(-extreme / 8 x alpha) for the alpha in {1.00, 0.99.., 0.70} with the least squared error over the
// group (ties: the larger alpha). q = clamp(rint(w / s), -8, 7); an all-zero group gets scale 0 and code 8.
inline void quantize_w4_row(const uint16_t *row, int64_t k, uint32_t *words, uint16_t *scales, bool mse = false) {
  constexpr int G = 128;
  for (int64_t g = 0; g < k / G; ++g) {
    float v[G];
    int im = 0;
    for (int i = 0; i < G; ++i) {
      const uint32_t u = static_cast<uint32_t>(row[g * G + i]) << 16;
      std::memcpy(&v[i], &u, 4);
      if (std::fabs(v[i]) > std::fabs(v[im])) im = i;
    }
    _Float16 h = static_cast<_Float16>(-v[im] / 8.f);
    if (mse && v[im] != 0.f) {
      double best = -1;
      for (int t = 0; t <= 30; ++t) {
        const _Float16 ht = static_cast<_Float16>(-v[im] / 8.f * (1.f - 0.01f * static_cast<float>(t)));
        const float sc = static_cast<float>(ht);
        if (sc == 0.f) continue;
        double e = 0;
        for (int i = 0; i < G; ++i) {
          const float q = std::clamp(std::nearbyint(v[i] / sc), -8.f, 7.f);
          const double d = static_cast<double>(v[i]) - static_cast<double>(q * sc);
          e += d * d;
        }
        if (best < 0 || e < best) best = e, h = ht;
      }
    }
    const float sc = static_cast<float>(h);
    std::memcpy(&scales[g], &h, 2);
    for (int j = 0; j < G / 8; ++j) {
      uint32_t w = 0;
      for (int i = 0; i < 8; ++i) {
        const float q = sc == 0.f ? 0.f : std::clamp(std::nearbyint(v[j * 8 + i] / sc), -8.f, 7.f);
        w |= static_cast<uint32_t>(static_cast<int>(q) + 8) << (4 * i);
      }
      words[g * (G / 8) + j] = w;
    }
  }
}

inline void bf16_round_floats(const float *src, float *dst, size_t n) {
  const auto *s = reinterpret_cast<const uint32_t *>(src);
  auto *d = reinterpret_cast<uint32_t *>(dst);
  for (size_t i = 0; i < n; ++i) {
    const uint32_t u = s[i];
    const uint32_t rne = (u + 0x7fffu + ((u >> 16) & 1u)) & 0xffff0000u;
    const uint32_t nan = (u & 0xffff0000u) | 0x00400000u;
    d[i] = (u & 0x7fffffffu) > 0x7f800000u ? nan : rne;
  }
}

// i2 (DenseKind:i2): codes [n][k] in 0..3 -> packed bytes [n k / 4] (the int8 layout's
// element order, 4 per byte). quantize_i2_groups: per (row, 128-group) the scale minimising the
// group's squared error over a grid a.0 x rms (Lloyd-Max for a Gaussian: levels +-0.45, +-1.51
// sigma), q = clamp(floor(w / s) + 2, 0, 3) (the nearest level); scale [k / 128][n].
size_t dense_i2_bytes(int n, int k);
void pack_dense_i2(const uint8_t *codes, int n, int k, uint8_t *out);
void quantize_i2_groups(const uint16_t *w, int n, int k, uint8_t *codes, float *scale, int r0 = 0, int r1 = -1);
void dense_ref_i2(const float *x, const uint8_t *codes, const float *scale, int m, int n, int k, double *y);
// y[m][n] = x[m][k]. W^T in double: bf16 weights, or int8 codes x per-row scale.
void dense_ref_bf16(const float *x, const uint16_t *w, int m, int n, int k, double *y);
void dense_ref_i8(const float *x, const int8_t *codes, const float *scale, int m, int n, int k, double *y);
// W8A8 prefill: y[m][n] = (sum over the 128-k groups g in order of float(sum_{k in g} aq[r][k] *
// codes[c][k]) * as[g][r]) * scale[c]; aq [m][k] int8 activations, as their group scales, group-major
// [gpr][lds] (gpr = ceil(k / 128), a short last group; dense.hpp's PfA8 layout). The device sums each group in
// int32 (exact) and accumulates the scaled groups in fp32; here in double. With integer data under
// power-of-two scales both are exact.
void dense_ref_a8(const int8_t *aq, const float *as, int lds, const int8_t *codes, const float *scale, int m, int n, int k, double *y);

}  // namespace xenithium:kernels
