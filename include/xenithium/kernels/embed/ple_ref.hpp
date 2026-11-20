// Xenithium · PLE (per-layer n-gram embedding) constants, n-gram layout and ids, the FP8 table
// dequantisation and the host reference of the block (host-only: no SYCL; precise fp in
// xenithium_pack). The device side is ple.hpp; the semantics are written there.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace xenithium::kernels {

inline constexpr int kPleTaps = 4;      // conv kernel
inline constexpr int kPleDil = 3;       // conv dilation (= ngram): taps t - 9, t - 6, t - 3, t
inline constexpr int kPleRing = 32;  // conv inputs kept per sequence (by position % 32)
                                        // (an 8-row segment + the conv's 9 earlier positions need 17)
inline constexpr int kPleMaxNgram = 4;  // n-gram order the id code supports

// ---- n-gram layout (derived from the config; the checkpoint stores the same buffers) ----------
// heads = (ngram - 1) * heads_per_ngram; head h belongs to order n = h / heads_per_ngram + 2.
// size[h] = the (layer_index * heads + h + 1)-th prime after base - 1; offset[h] = the sum of the
// sizes before it. mult[i] = 2 (splitmix64(seed + 10007 layer_index + gamma (i + 1)) % half) + 1,
// half = ((2^63 - 1) / vocab) / 2.
struct PleLayout {
  int ngram = 0, heads_per_ngram = 0, heads = 0;
  int64_t eos = 0;
  std::vector<int64_t> mult;            // [ngram]
  std::vector<int64_t> size, offset;    // [heads]
  int64_t rows = 0;                     // sum of the sizes
  int64_t padded_rows = 0;              // rounded up to the divisor
  static PleLayout derive(int64_t vocab, int ngram, int heads_per_ngram, int64_t base, int64_t seed, int layer_index, int64_t divisor,
                          int64_t eos);
};
uint64_t ple_splitmix64(uint64_t v);
bool ple_is_prime(int64_t v);

// The ids of one token: tok at position t, prev[0] = token t - 1, prev[1] = t - 2 (... up to
// ngram - 1). Walking back from t - 1, everything older than the first eos reads as eos; before
// the sequence start the context is eos. mixed = xor_i(token[t - i] * mult[i]) (int64, wrapping)
// over the order's i < n; id[h] = offset[h] + mixed mod size[h] (non-negative remainder).
void ple_ngram_ids(const PleLayout &L, int64_t tok, const int64_t *prev, int64_t *ids);
// A sequence's context: the last ngram - 1 tokens, newest first; starts as eos.
struct PleContext {
  std::array<int64_t, kPleMaxNgram - 1> prev{};
  void reset(const PleLayout &L) { prev.fill(L.eos); }
  void push(const PleLayout &L, int64_t tok);  // tok becomes t - 1 of the next token
};

// ---- the FP8 table ------------------------------------------------------------------------------
// The stock dequantisation bf16(e4m3) * bf16(scale) in bf16: the product of a 4-bit and an 8-bit
// mantissa is exact in fp32, so one table of bf16(code value * scale) per code is bit-identical.
std::array<uint16_t, 256> ple_fp8_lut(uint16_t scale_bf16);

// ---- host reference of the block, one row ------------------------------------------------------
// Rounding points of the gate and the conv, sums in double:
//   per stream s:  kn = bf16(rms(key_s) (1 + nk)), qn = bf16(rms(res_s) (1 + nq))  (fp32 math)
//                  d  = bf16(bf16(sum_i bf16(kn qn)) / sqrt(H));  g = bf16(sigmoid(sign(d) bf16(sqrt(max(|d|, 1e-6)))))
//                  gated_s = bf16(g v);  normed_s = bf16(rms(gated_s) (1 + ncw))
//   per channel:   conv = bf16(sum_k w[c][k] x[t - 9 + 3k]) over the normed inputs (zero before 0)
//                  res' = bf16(res + bf16(gated + bf16(silu(conv))))
// key [S*H], value [H], res [S*H]: bf16 values; gated / normed out [S*H] bf16 values.
void ple_ref_gate(int H, int S, const float *key, const float *value, const float *res, const float *nk, const float *nq, const float *ncw, float eps,
                  float *gated, float *normed);
// taps[k] = the normed input at t - 9 + 3k ([C] each, null = zero); w [C][4]; res in place.
void ple_ref_conv(int C, const float *const taps[kPleTaps], const float *w, const float *gated, float *res);

}  // namespace xenithium::kernels
