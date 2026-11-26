#include "xenithium/arch/qwen38/mtp.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

#include "xenithium/kernels/attention/qsa_ref.hpp"
#include "xenithium/kernels/pack.hpp"

namespace xenithium::arch::qwen38 {

namespace k = xenithium::kernels;

std::vector<model::DenseSpec> mtp_dense_plan(const model::Checkpoint &ck, bool kv_by_tokens) {
  using S = model::DenseSpec;
  using model::Component;
  std::vector<S> out;
  auto add = [&](std::string name, S::Split split, S::Store store, int layer, std::vector<std::string> append = {}) {
    if (!ck.find(name)) throw std::runtime_error("qwen38 mtp plan: missing " + name);
    for (const auto &a : append)
      if (!ck.find(a)) throw std::runtime_error("qwen38 mtp plan: missing " + a);
    S sp;
    sp.name = std::move(name);
    sp.split = split;
    sp.store = store;
    sp.append = std::move(append);
    sp.comp = Component::mtp;
    sp.layer = layer;
    out.push_back(std::move(sp));
  };
  const std::string P = kMtpPrefix, L = kMtpLayer;
  // front: two Gemma norms and the two projections (replicated: every card builds the whole state)
  add(P + "pre_fc_norm_embedding.weight", S::Split::replicate, S::Store::raw, -1);
  add(P + "pre_fc_norm_hidden.weight", S::Split::replicate, S::Store::raw, -1);
  add(P + "fc_embedding.weight", S::Split::replicate, S::Store::i8, -1);
  add(P + "fc_hidden.weight", S::Split::replicate, S::Store::i8, -1);
  // the MTP's own final mixer (no injection rows)
  add(P + "hyper_connection_mixer.hc_norm.weight", S::Split::replicate, S::Store::raw, -1);
  add(P + "hyper_connection_mixer.input_mix_weight_down.weight", S::Split::replicate, S::Store::i8, -1);
  add(P + "hyper_connection_mixer.input_mix_weight_up.weight", S::Split::replicate, S::Store::i8, -1);
  // the layer: as one of the target's full-attention layers
  for (const char *mix : {"attn_hyper_connection.", "mlp_hyper_connection."}) {
    const std::string m = L + mix;
    add(m + "hc_norm.weight", S::Split::replicate, S::Store::raw, 0);
    add(m + "input_mix_weight_down.weight", S::Split::replicate, S::Store::i8, 0, {m + "block_inject_weight.weight"});
    add(m + "input_mix_weight_up.weight", S::Split::replicate, S::Store::i8, 0);
  }
  const std::string A = L + "self_attn.";
  add(A + "q_proj.weight", S::Split::rows, S::Store::i8, 0, {A + "k_proj.weight", A + "v_proj.weight"});
  // TP4 = a kv head on two cards (each takes it whole): heads of kQsaDim rows
  out.back().append_heads = static_cast<int>(ck.at(A + "k_proj.weight").dim(0) / kernels::kQsaDim);
  if (kv_by_tokens) {  // the KV split by tokens (TP4): + the partner's q heads, as the target's QSA layers
    out.back().peer_block = 2 * kernels::kQsaDim;
    out.back().peer_keep = kernels::kQsaDim;
  }
  add(A + "o_proj.weight", S::Split::cols, S::Store::i8, 0);
  add(A + "q_norm.weight", S::Split::replicate, S::Store::raw, 0);
  add(A + "k_norm.weight", S::Split::replicate, S::Store::raw, 0);
  add(A + "indexer.index_qk_proj.weight", S::Split::replicate, S::Store::bf16, 0);
  add(A + "indexer.q_layernorm.weight", S::Split::replicate, S::Store::raw, 0);
  add(A + "indexer.k_layernorm.weight", S::Split::replicate, S::Store::raw, 0);
  add(L + "mlp.gate.weight", S::Split::replicate, S::Store::bf16, 0, {L + "mlp.shared_expert_gate.weight"});
  add(L + "mlp.shared_expert.gate_proj.weight", S::Split::replicate, S::Store::i8, 0, {L + "mlp.shared_expert.up_proj.weight"});
  add(L + "mlp.shared_expert.down_proj.weight", S::Split::replicate, S::Store::i8, 0);
  return out;
}

std::string mtp_expert_name(int expert, const char *proj, const char *part) {
  return std::string(kMtpLayer) + "mlp.experts." + std::to_string(expert) + "." + proj + "." + part;
}

W4Rows mtp_unpack_g32(const uint8_t *pk, const uint16_t *sc, int n, int kd) {
  W4Rows m;
  m.n = n;
  m.k = kd;
  m.G = 32;
  const size_t bytes = static_cast<size_t>(n) * kd / 2;
  m.q.resize(static_cast<size_t>(n) * kd);
  for (size_t i = 0; i < bytes; ++i) {
    m.q[2 * i] = static_cast<int8_t>((pk[i] & 15) - 8);
    m.q[2 * i + 1] = static_cast<int8_t>((pk[i] >> 4) - 8);
  }
  m.s.resize(static_cast<size_t>(n) * kd / 32);
  for (size_t i = 0; i < m.s.size(); ++i) m.s[i] = k::bf16_value(sc[i]);
  return m;
}

W4Rows mtp_requant_g128(const W4Rows &a) {
  if (a.G != 32 || a.k % 128) throw std::invalid_argument("mtp_requant_g128: needs group 32 input and k % 128 == 0");
  W4Rows m;
  m.n = a.n;
  m.k = a.k;
  m.G = 128;
  m.q.resize(a.q.size());
  m.s.resize(static_cast<size_t>(a.n) * (a.k / 128));
  float w[128];
  for (int r = 0; r < a.n; ++r)
    for (int g = 0; g < a.k / 128; ++g) {
      float amax = 0;
      for (int i = 0; i < 128; ++i) {
        const int c = g * 128 + i;
        w[i] = a.q[static_cast<size_t>(r) * a.k + c] * a.s[static_cast<size_t>(r) * (a.k / 32) + c / 32];
        amax = std::max(amax, std::fabs(w[i]));
      }
      float best = amax > 0 ? amax / 7.f : 1.f;
      double best_err = 1e300;
      for (int t = 0; t <= 10 && amax > 0; ++t) {
        const float s = k::bf16_value(k::bf16_bits(amax / 7.f * (1.f - 0.02f * t)));
        double err = 0;
        for (int i = 0; i < 128; ++i) {
          const float q = std::clamp(std::nearbyint(w[i] / s), -8.f, 7.f);
          err += (q * s - w[i]) * (q * s - w[i]);
        }
        if (err < best_err) {
          best_err = err;
          best = s;
        }
      }
      m.s[static_cast<size_t>(r) * (a.k / 128) + g] = best;
      for (int i = 0; i < 128; ++i)
        m.q[static_cast<size_t>(r) * a.k + g * 128 + i] = static_cast<int8_t>(std::clamp(std::nearbyint(w[i] / best), -8.f, 7.f));
    }
  return m;
}

uint16_t fp16_bits(float v) {
  uint32_t x;
  std::memcpy(&x, &v, 4);
  const uint32_t sign = (x >> 16) & 0x8000u;
  const uint32_t ax = x & 0x7FFFFFFFu;
  if (ax >= 0x7F800000u) return static_cast<uint16_t>(sign | 0x7C00u | (ax > 0x7F800000u ? 0x200u : 0));
  if (ax >= 0x477FF000u) return static_cast<uint16_t>(sign | 0x7C00u);  // rounds to >= 65520: inf
  if (ax < 0x38800000u) {  // below 2^-14: fp16 subnormal (or zero)
    if (ax < 0x33000000u) return static_cast<uint16_t>(sign);  // < 2^-25: rounds to 0
    const uint32_t mant = (ax & 0x7FFFFFu) | 0x800000u;
    const uint32_t drop = 126u - (ax >> 23);  // value = mant 2^(e - 150) = h 2^-24 -> h = mant >> (126 - e)
    uint32_t h = mant >> drop;
    const uint32_t rem = mant & ((1u << drop) - 1), half = 1u << (drop - 1);
    if (rem > half || (rem == half && (h & 1))) ++h;
    return static_cast<uint16_t>(sign | h);
  }
  uint32_t h = ((ax - 0x38000000u) >> 13);  // rebias 127 -> 15
  const uint32_t rem = ax & 0x1FFFu;
  if (rem > 0x1000u || (rem == 0x1000u && (h & 1))) ++h;
  return static_cast<uint16_t>(sign | h);
}

float fp16_value(uint16_t h) {
  const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
  const uint32_t e = (h >> 10) & 31, m = h & 0x3FFu;
  float v;
  if (e == 0) v = std::ldexp(static_cast<float>(m), -24);
  else if (e == 31) v = m ? NAN : INFINITY;
  else v = std::ldexp(static_cast<float>(m | 0x400u), static_cast<int>(e) - 25);
  uint32_t x;
  std::memcpy(&x, &v, 4);
  x |= sign;
  std::memcpy(&v, &x, 4);
  return v;
}

namespace {
// GPTQ qweight uint32 [k / 8][n] from codes [n][k]: nibble i of row r is k = 8 r + i, stored q + 8.
std::vector<uint32_t> gptq_qweight(const W4Rows &m) {
  std::vector<uint32_t> qw(static_cast<size_t>(m.k / 8) * m.n, 0);
  for (int c = 0; c < m.n; ++c)
    for (int kk = 0; kk < m.k; ++kk) {
      const uint32_t nib = static_cast<uint32_t>(m.q[static_cast<size_t>(c) * m.k + kk] + k::kW4Zero) & 15u;
      qw[static_cast<size_t>(kk / 8) * m.n + c] |= nib << (4 * (kk % 8));
    }
  return qw;
}
}  // namespace

int64_t mtp_pack_expert(const ExpertLayout &lay, const W4Rows &gate, const W4Rows &up, const W4Rows &down, uint8_t *out) {
  const int H = lay.hidden, I = lay.inter;
  if (gate.G != k::kW4Group || up.G != k::kW4Group || down.G != k::kW4Group || gate.n != I || gate.k != H || up.n != I || up.k != H || down.n != H ||
      down.k != I)
    throw std::invalid_argument("mtp_pack_expert: matrices are not g128 [inter][hidden] / [hidden][inter]");
  const auto gq = gptq_qweight(gate), uq = gptq_qweight(up), dq = gptq_qweight(down);
  k::pack_w4_gptq_cat(gq.data(), uq.data(), I, 2 * I, H, out + lay.gu_codes);
  k::pack_w4_gptq(dq.data(), H, I, out + lay.dn_codes);
  int64_t inexact = 0;
  auto put = [&](size_t off, float v) {
    const uint16_t h = fp16_bits(v);
    inexact += fp16_value(h) != v;
    std::memcpy(out + off, &h, 2);
  };
  // gate|up scales fp16 [hidden / 128][2 inter]: row g = gate scales | up scales
  for (int g = 0; g < H / k::kW4Group; ++g)
    for (int c = 0; c < I; ++c) {
      put(lay.gu_scales + (static_cast<size_t>(g) * 2 * I + c) * 2, gate.s[static_cast<size_t>(c) * (H / 128) + g]);
      put(lay.gu_scales + (static_cast<size_t>(g) * 2 * I + I + c) * 2, up.s[static_cast<size_t>(c) * (H / 128) + g]);
    }
  // down scales fp16 [inter / 128][hidden]
  for (int g = 0; g < I / k::kW4Group; ++g)
    for (int c = 0; c < H; ++c) put(lay.dn_scales + (static_cast<size_t>(g) * H + c) * 2, down.s[static_cast<size_t>(c) * (I / 128) + g]);
  if (lay.stride > lay.bytes) std::memset(out + lay.bytes, 0, lay.stride - lay.bytes);
  return inexact;
}

}  // namespace xenithium::arch::qwen38
