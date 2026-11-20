// xenithium-mtp-ref: host reference of the qwen38 MTP drafter, teacher-forced on the target's own
// multi-stream hiddens (probe model --hidden-out), scored by agreement with the target's picks.
// xenithium-mtp-ref --target DIR --mtp DIR --hidden F.xmh [--depth 1..3] [--eval-from P]
// [--experts g32|g128] [--dense bf16|i8] [--attn full|qsa] [--eval-n N]
// [--drafts-out F] [--ablate X] [--head bf16|i8] [--threads T]
// --drafts-out: int32 records (chunk, position, d1, d2, d3). --ablate (debug): hidden | embed |
// attn | moe | routed zero that input or output; target-final uses the target's final mixer;
// check-target-head runs the target's final mixer (no block) + lm_head on the dumped hidden and
// counts == the target's picks (must be ~98 %: the check that found the stale replay buffer).
// Semantics (Qwen3.8-Flash-Next MTP, from the snapshot's config.json and tensors; the reference
// engine's model definition read for semantics only). A draft pass at position p takes the target's
// pre-final-mixer multi-stream hidden h_p [hc][hidden] and the token t_{p+1}:
// e = bf16(gemma_rmsnorm(embed[t], pre_fc_norm_embedding)) (embed = the target's)
// hn = bf16(gemma_rmsnorm(h_p, pre_fc_norm_hidden)) (one norm over all hc*hidden)
// x_s = bf16(bf16(e. Wfe^T) + bf16(hn_s. Wfh^T)) (fc_hidden shared by the streams)
// then one decoder layer with no pending block (attn mixer -> full attention -> mlp mixer -> MoE),
// the MTP's own final mixer (no injection) and the target's lm_head. Pass k (k >= 2) at position
// p + k - 1 takes the previous pass's state after its final combine and its draft token.
// Full attention (the snapshot config has no indexer: plain gated GQA, not QSA): [q_h | gate_h] =
// bf16(x. Wq^T), q, k = bf16(rope(bf16(gemma_rmsnorm(.)))) (NeoX, first 64 dims, theta from the
// config), softmax(q. K / 16). Wo^T.
// --attn qsa: the layer's attention as the target's QSA layers (kernels/attention/qsa.hpp: indexer from
// index_qk_proj + q/k_layernorm, compressed keys of 4 tokens, top-512 blocks + the open block's
// tail); the runtime checkpoint's config names the indexer, the served snapshot's does not. The two
// are identical up to 2,048 tokens (every block selected). Every pass selects for itself.
// MoE: the target's block semantics (kernels/moe/moe.hpp), with the snapshot's experts.
// KV at positions <= p comes from pass 1 at those positions (what a decode keeps once they are
// accepted); a chain's passes 2.. write their own positions. Agreement: pass k at p predicts the
// token at p + k + 1, accepted while every earlier pass of the chain was too and d_k == the
// target's pick at p + k (on greedy text the picks are the text, so this is the real acceptance).
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

#include "xenithium/arch/qwen38/mtp.hpp"
#include "xenithium/kernels/residual/hc.hpp"
#include "xenithium/kernels/pack.hpp"
#include "xenithium/kernels/attention/qsa_ref.hpp"
#include "xenithium/model/checkpoint.hpp"
#include "xenithium/model/config.hpp"
#include "xenithium/util/json.hpp"

namespace k = xenithium::kernels;
namespace xm = xenithium::model;

namespace {

constexpr int H = 2560, S = 4, W = S * H, R = 320, QH = 24, KVH = 2, HD = 256, E = 512, TOPK = 10, INTER = 640, SINTER = 640;
constexpr int QN = QH * 2 * HD;  // q | gate per head
constexpr float kEps = 1e-6f;

float bf(uint16_t b) { return k::bf16_value(b); }
uint16_t rb(float v) { return k::bf16_bits(v); }
float rbf(float v) { return bf(rb(v)); }
float sig(float x) { return 1.f / (1.f + std::exp(-x)); }

std::vector<uint8_t> read_bytes(const xm::Checkpoint &ck, const std::string &name, uint64_t off = 0, uint64_t n = ~0ull) {
  const auto &t = ck.at(name);
  if (n == ~0ull) n = t.bytes - off;
  std::vector<uint8_t> out(n);
  const int fd = ::open(ck.shards()[t.shard].path.c_str(), O_RDONLY);
  if (fd < 0) throw std::runtime_error("open " + ck.shards()[t.shard].path);
  size_t got = 0;
  while (got < n) {
    const ssize_t r = ::pread(fd, out.data() + got, n - got, static_cast<off_t>(t.offset + off + got));
    if (r <= 0) {
      ::close(fd);
      throw std::runtime_error("read " + name);
    }
    got += static_cast<size_t>(r);
  }
  ::close(fd);
  return out;
}
std::vector<uint16_t> read_bf16(const xm::Checkpoint &ck, const std::string &name, int64_t numel) {
  const auto &t = ck.at(name);
  if (t.dtype_str != "BF16" || t.numel() != numel) throw std::runtime_error(name + ": expected BF16 x " + std::to_string(numel));
  const auto b = read_bytes(ck, name);
  std::vector<uint16_t> v(static_cast<size_t>(numel));
  std::memcpy(v.data(), b.data(), b.size());
  return v;
}

// A dense matrix [n][k] for the reference: bf16 values, or int8 codes x a per-row scale.
struct Mat {
  int n = 0, kd = 0;
  std::vector<float> w;  // [n][k] values (bf16) or codes (i8)
  std::vector<float> scale;  // [n] (i8) or empty
  k::RefMatrix ref() const { return {w.data(), scale.empty() ? nullptr : scale.data(), n, kd}; }
};
Mat make_mat(const std::vector<uint16_t> &b, int n, int kd, bool i8) {
  Mat m;
  m.n = n;
  m.kd = kd;
  m.w.resize(static_cast<size_t>(n) * kd);
  if (!i8) {
    for (size_t i = 0; i < m.w.size(); ++i) m.w[i] = bf(b[i]);
  } else {
    std::vector<int8_t> codes(m.w.size());
    m.scale.resize(static_cast<size_t>(n));
    k::quantize_i8_rows(b.data(), n, kd, codes.data(), m.scale.data());
    for (size_t i = 0; i < m.w.size(); ++i) m.w[i] = codes[i];
  }
  return m;
}
// y[n] = sum_k x w (double), then * scale.
void gemv(const Mat &m, const float *x, double *y) {
  for (int r = 0; r < m.n; ++r) {
    const float *w = m.w.data() + static_cast<size_t>(r) * m.kd;
    double acc = 0;
    for (int i = 0; i < m.kd; ++i) acc += static_cast<double>(x[i]) * w[i];
    y[r] = m.scale.empty() ? acc : acc * m.scale[static_cast<size_t>(r)];
  }
}

// A routed expert's matrix [n][k]: codes in [-8, 7] and one scale per (row, group of G); the
// unpack and the g128 requant are the engine's own (arch/qwen38/mtp.hpp), so both run the same codes.
using QMat = xenithium::arch::qwen38::W4Rows;
double qdot(const QMat &m, int r, const float *x) {
  const int8_t *q = m.q.data() + static_cast<size_t>(r) * m.k;
  const float *s = m.s.data() + static_cast<size_t>(r) * (m.k / m.G);
  double acc = 0;
  for (int g = 0; g < m.k / m.G; ++g) {
    double a = 0;
    for (int i = g * m.G; i < (g + 1) * m.G; ++i) a += static_cast<double>(x[i]) * q[i];
    acc += a * s[g];
  }
  return acc;
}
QMat load_g32(const xm::Checkpoint &ck, const std::string &base, int n, int kd) {
  const auto pk = read_bytes(ck, base + ".weight_packed");
  const auto sc = read_bf16(ck, base + ".weight_scale", static_cast<int64_t>(n) * kd / 32);
  if (pk.size() != static_cast<size_t>(n) * kd / 2) throw std::runtime_error(base + ": packed size");
  return xenithium::arch::qwen38::mtp_unpack_g32(pk.data(), sc.data(), n, kd);
}
QMat requant_g128(const QMat &a) { return xenithium::arch::qwen38::mtp_requant_g128(a); }

struct Expert {
  QMat gate, up, down;  // [inter][hidden], [inter][hidden], [hidden][inter]
};

struct Drafter {
  std::vector<uint16_t> embed;  // target [V][H] (rows read on demand would do; kept whole)
  std::vector<uint16_t> lm;  // target lm_head [V][H]
  int64_t V = 0;
  std::vector<float> norm_e, norm_h;  // pre_fc norms [H], [W]
  Mat fe, fh;
  std::vector<uint16_t> attn_norm, mlp_norm, fin_norm;  // [W]
  Mat attn_down, attn_up, mlp_down, mlp_up, fin_down, fin_up;
  Mat qp, kp, vp, op;
  std::vector<float> qn, kn;  // [256]
  Mat idx;  // qsa: index_qk_proj [4 x 128 | 128][H] (bf16, as the target's)
  std::vector<float> iqn, ikn;  // [128]
  bool qsa = false;
  std::string ablate;  // debug: "hidden" (zero h), "embed" (zero e), "attn" (zero attention out), "moe" (zero MoE out)
  Mat router;  // [E + 1][H]: row E = shared_expert_gate (bf16 always)
  Mat sg, su, sd;  // shared expert
  std::vector<Expert> ex;
  std::vector<uint16_t> rope;  // bf16 [positions][64]
  bool head_i8 = false;  // --head i8: the device's int8 lm_head and bf16 logit pick
  std::vector<int8_t> lm8;  // [V][H] codes
  std::vector<float> lm8s;  // [V] row scales
};

std::vector<uint16_t> cat_rows(const std::vector<uint16_t> &a, const std::vector<uint16_t> &b) {
  std::vector<uint16_t> c(a);
  c.insert(c.end(), b.begin(), b.end());
  return c;
}
std::vector<float> to_f(const std::vector<uint16_t> &v) {
  std::vector<float> f(v.size());
  for (size_t i = 0; i < v.size(); ++i) f[i] = bf(v[i]);
  return f;
}

// gemma RMSNorm over n values -> bf16 bits.
void gemma_norm(const float *x, const float *w, int n, uint16_t *out) {
  double ss = 0;
  for (int i = 0; i < n; ++i) ss += static_cast<double>(x[i]) * x[i];
  const float r = 1.f / std::sqrt(static_cast<float>(ss / n) + kEps);
  for (int i = 0; i < n; ++i) {
    const float y = x[i] * r;
    out[i] = rb(std::fma(y, w[i], y));
  }
}

struct Kv {  // one position's K and V (bf16 values) [kvh][256]
  std::vector<float> k, v;
};

struct Pre {  // a pass up to the attention: state + q, gate, K, V at its position
  std::vector<uint16_t> res;  // [W] after the attn mixer (unchanged by a mix with no pending block)
  std::vector<float> inj;  // [S] the attn mixer's injection logits
  std::vector<float> q, gate;  // [QH][256] roped q values, raw gate values
  Kv kv;
  std::vector<float> iq, ik;  // qsa: [4][128] normed + roped index queries, the raw index key [128]
};

Pre pass_front(const Drafter &d, const uint16_t *h, int32_t tok, int64_t pos) {
  Pre p;
  std::vector<float> e(H), hf(W), xb(H);
  for (int i = 0; i < H; ++i) e[i] = bf(d.embed[static_cast<size_t>(tok) * H + i]);
  for (int i = 0; i < W; ++i) hf[i] = d.ablate == "hidden" ? 0.f : bf(h[i]);
  if (d.ablate == "embed") std::fill(e.begin(), e.end(), 0.f);
  std::vector<uint16_t> en(H), hn(W);
  gemma_norm(e.data(), d.norm_e.data(), H, en.data());
  gemma_norm(hf.data(), d.norm_h.data(), W, hn.data());
  std::vector<float> enf(H), hnf(W);
  for (int i = 0; i < H; ++i) enf[i] = bf(en[i]);
  for (int i = 0; i < W; ++i) hnf[i] = bf(hn[i]);
  std::vector<double> y(H);
  gemv(d.fe, enf.data(), y.data());
  std::vector<float> fe(H);
  for (int i = 0; i < H; ++i) fe[i] = rbf(static_cast<float>(y[i]));
  p.res.resize(W);
  for (int s = 0; s < S; ++s) {
    gemv(d.fh, hnf.data() + static_cast<size_t>(s) * H, y.data());
    for (int i = 0; i < H; ++i) p.res[static_cast<size_t>(s) * H + i] = rb(fe[i] + rbf(static_cast<float>(y[i])));
  }
  // attn mixer (no pending block): block input + the injection logits for the next combine
  const k::HcShape hs{S, H, R, kEps};
  std::vector<float> lora(static_cast<size_t>(d.attn_down.n));
  std::vector<uint16_t> bin(H);
  k::hc_mix_ref(hs, 1, p.res.data(), nullptr, nullptr, d.attn_norm.data(), d.attn_down.ref(), d.attn_up.ref(), nullptr, lora.data(), bin.data());
  p.inj.assign(lora.begin() + R, lora.begin() + R + S);
  for (int i = 0; i < H; ++i) xb[i] = bf(bin[i]);
  // q | gate, k, v
  std::vector<double> yq(QN), yk(KVH * HD), yv(KVH * HD);
  gemv(d.qp, xb.data(), yq.data());
  gemv(d.kp, xb.data(), yk.data());
  gemv(d.vp, xb.data(), yv.data());
  const uint16_t *rr = d.rope.data() + static_cast<size_t>(pos) * k::kQsaRopeRow;
  p.q.resize(QH * HD);
  p.gate.resize(QH * HD);
  std::vector<float> t(HD);
  for (int hh = 0; hh < QH; ++hh) {
    for (int i = 0; i < HD; ++i) t[i] = rbf(static_cast<float>(yq[static_cast<size_t>(hh) * 2 * HD + i]));
    k::qsa_ref_norm_rope(t.data(), d.qn.data(), HD, rr, kEps, p.q.data() + static_cast<size_t>(hh) * HD);
    for (int i = 0; i < HD; ++i) p.gate[static_cast<size_t>(hh) * HD + i] = rbf(static_cast<float>(yq[static_cast<size_t>(hh) * 2 * HD + HD + i]));
  }
  p.kv.k.resize(KVH * HD);
  p.kv.v.resize(KVH * HD);
  for (int hh = 0; hh < KVH; ++hh) {
    for (int i = 0; i < HD; ++i) t[i] = rbf(static_cast<float>(yk[static_cast<size_t>(hh) * HD + i]));
    k::qsa_ref_norm_rope(t.data(), d.kn.data(), HD, rr, kEps, p.kv.k.data() + static_cast<size_t>(hh) * HD);
    for (int i = 0; i < HD; ++i) p.kv.v[static_cast<size_t>(hh) * HD + i] = rbf(static_cast<float>(yv[static_cast<size_t>(hh) * HD + i]));
  }
  if (d.qsa) {
    std::vector<double> yi(5 * 128);
    gemv(d.idx, xb.data(), yi.data());
    p.iq.resize(4 * 128);
    p.ik.resize(128);
    std::vector<float> t2(128);
    for (int hh = 0; hh < 4; ++hh) {
      for (int i = 0; i < 128; ++i) t2[i] = rbf(static_cast<float>(yi[static_cast<size_t>(hh) * 128 + i]));
      k::qsa_ref_norm_rope(t2.data(), d.iqn.data(), 128, rr, kEps, p.iq.data() + static_cast<size_t>(hh) * 128);
    }
    for (int i = 0; i < 128; ++i) p.ik[i] = rbf(static_cast<float>(yi[4 * 128 + i]));
  }
  return p;
}
// Compressed key of block b from four raw index keys.
std::vector<float> compress(const Drafter &d, const float *const raw[4], int b) {
  std::vector<float> r4(4 * 128), kc(128);
  for (int j = 0; j < 4; ++j) std::memcpy(&r4[static_cast<size_t>(j) * 128], raw[j], 128 * 4);
  k::qsa_ref_compress(r4.data(), d.ikn.data(), &d.rope[static_cast<size_t>(b) * 4 * k::kQsaRopeRow], kEps, kc.data());
  return kc;
}

// MoE block (kernels/moe/moe.hpp semantics), one row: x bf16 values [H] -> out bf16 bits [H].
void moe(const Drafter &d, const float *x, uint16_t *out) {
  std::vector<double> lg(E + 1);
  gemv(d.router, x, lg.data());
  std::vector<std::pair<float, int>> v(E);
  for (int e = 0; e < E; ++e) v[static_cast<size_t>(e)] = {rbf(static_cast<float>(lg[e])), e};
  std::partial_sort(v.begin(), v.begin() + TOPK, v.end(), [](const auto &a, const auto &b) { return a.first != b.first ? a.first > b.first : a.second < b.second; });
  float z = 0;
  float wv[TOPK];
  for (int j = 0; j < TOPK; ++j) z += (wv[j] = std::exp(v[static_cast<size_t>(j)].first - v[0].first));
  std::vector<std::pair<int, float>> sel(TOPK);
  for (int j = 0; j < TOPK; ++j) sel[static_cast<size_t>(j)] = {v[static_cast<size_t>(j)].second, wv[j] / z};
  std::sort(sel.begin(), sel.end());  // ascending expert id (the device's sum order)
  const float sgate = sig(rbf(static_cast<float>(lg[E])));
  std::vector<float> p(H, 0.f), h(INTER);
  for (const auto &[e, w] : sel) {
    if (d.ablate == "routed") break;
    const Expert &X = d.ex[static_cast<size_t>(e)];
    for (int n = 0; n < INTER; ++n) {
      const float g = rbf(static_cast<float>(qdot(X.gate, n, x))), u = rbf(static_cast<float>(qdot(X.up, n, x)));
      h[static_cast<size_t>(n)] = rbf(g * sig(g) * u);
    }
    for (int n = 0; n < H; ++n) p[static_cast<size_t>(n)] = std::fma(static_cast<float>(qdot(X.down, n, h.data())), w, p[static_cast<size_t>(n)]);
  }
  std::vector<double> yg(SINTER), yu(SINTER), yd(H);
  gemv(d.sg, x, yg.data());
  gemv(d.su, x, yu.data());
  for (int n = 0; n < SINTER; ++n) {
    const float g = rbf(static_cast<float>(yg[n])), u = rbf(static_cast<float>(yu[n]));
    h[static_cast<size_t>(n)] = rbf(g * sig(g) * u);
  }
  gemv(d.sd, h.data(), yd.data());
  for (int n = 0; n < H; ++n) out[n] = rb(std::fma(static_cast<float>(yd[n]), sgate, p[static_cast<size_t>(n)]));
}

// The rest of a pass at position kvs.size - 1: attention over the cached positions (full, or the
// QSA selection over the complete blocks' keys kcs), then mlp mixer, MoE, final mixer, lm_head.
// res is updated to the state after the final combine (the next pass's hidden).
int32_t pass_back(const Drafter &d, Pre &p, const std::vector<const Kv *> &all, const std::vector<const float *> &kcs) {
  std::vector<const Kv *> kvs;
  if (d.qsa) {
    const int pos = static_cast<int>(all.size()) - 1;
    std::vector<uint32_t> keys(kcs.size());
    for (size_t b = 0; b < kcs.size(); ++b) {
      const float sc = static_cast<float>(k::qsa_ref_score(p.iq.data(), kcs[b]));
      std::memcpy(&keys[b], &sc, 4);
    }
    const auto sel = k::qsa_ref_select(keys.data(), static_cast<int>(keys.size()), k::kQsaTopBlocks);
    for (const int t : k::qsa_ref_tokens(sel, pos)) kvs.push_back(all[static_cast<size_t>(t)]);
  } else {
    kvs = all;
  }
  const int nt = static_cast<int>(kvs.size());
  std::vector<float> K(static_cast<size_t>(nt) * HD), Vv(static_cast<size_t>(nt) * HD), att(QH * HD);
  std::vector<double> o(HD);
  for (int g = 0; g < KVH; ++g) {
    for (int t = 0; t < nt; ++t) {
      std::memcpy(&K[static_cast<size_t>(t) * HD], kvs[static_cast<size_t>(t)]->k.data() + g * HD, HD * 4);
      std::memcpy(&Vv[static_cast<size_t>(t) * HD], kvs[static_cast<size_t>(t)]->v.data() + g * HD, HD * 4);
    }
    for (int hh = g * (QH / KVH); hh < (g + 1) * (QH / KVH); ++hh) {
      k::qsa_ref_attend(p.q.data() + static_cast<size_t>(hh) * HD, nt, K.data(), Vv.data(), o.data());
      k::qsa_ref_gate(o.data(), p.gate.data() + static_cast<size_t>(hh) * HD, att.data() + static_cast<size_t>(hh) * HD);
    }
  }
  std::vector<double> y(H);
  gemv(d.op, att.data(), y.data());
  std::vector<uint16_t> blk(H), bin(H), mo(H), out(H);
  for (int i = 0; i < H; ++i) blk[i] = d.ablate == "attn" ? 0 : rb(static_cast<float>(y[i]));
  const k::HcShape hs{S, H, R, kEps};
  std::vector<float> lora(static_cast<size_t>(d.mlp_down.n)), lf(static_cast<size_t>(d.fin_down.n));
  k::hc_mix_ref(hs, 1, p.res.data(), blk.data(), p.inj.data(), d.mlp_norm.data(), d.mlp_down.ref(), d.mlp_up.ref(), nullptr, lora.data(), bin.data());
  std::vector<float> xb(H);
  for (int i = 0; i < H; ++i) xb[i] = bf(bin[i]);
  moe(d, xb.data(), mo.data());
  if (d.ablate == "moe") std::fill(mo.begin(), mo.end(), 0);
  k::hc_mix_ref(hs, 1, p.res.data(), mo.data(), lora.data() + R, d.fin_norm.data(), d.fin_down.ref(), d.fin_up.ref(), nullptr, lf.data(), out.data());
  for (int i = 0; i < H; ++i) xb[i] = bf(out[i]);
  int32_t best = 0;
  double bv = -1e300;
  for (int64_t t = 0; t < d.V; ++t) {
    double l;
    if (d.head_i8) {  // the device's head: int8 rows x scale, the logit rounded to bf16, ties to the lowest id
      const int8_t *w = d.lm8.data() + static_cast<size_t>(t) * H;
      double acc = 0;
      for (int i = 0; i < H; ++i) acc += static_cast<double>(xb[i]) * w[i];
      l = rbf(static_cast<float>(acc) * d.lm8s[static_cast<size_t>(t)]);
    } else {
      const uint16_t *w = d.lm.data() + static_cast<size_t>(t) * H;
      double acc = 0;
      for (int i = 0; i < H; ++i) acc += static_cast<double>(xb[i]) * bf(w[i]);
      l = acc;  // unrounded: bf16 logits tie too often for an argmax
    }
    if (l > bv) {
      bv = l;
      best = static_cast<int32_t>(t);
    }
  }
  return best;
}

template <class F>
void parallel(int n, int threads, F &&f) {
  std::atomic<int> next{0};
  std::vector<std::thread> pool;
  std::mutex em;
  std::string err;
  for (int t = 0; t < threads; ++t)
    pool.emplace_back([&] {
      try {
        for (int i; (i = next.fetch_add(1)) < n;) f(i);
      } catch (const std::exception &e) {
        std::lock_guard<std::mutex> l(em);
        err = e.what();
        next = n;
      }
    });
  for (auto &t : pool) t.join();
  if (!err.empty()) throw std::runtime_error(err);
}

struct Chunk {
  std::vector<int32_t> tok, pick;
  std::vector<uint16_t> hid;  // [len][W]
};
std::vector<Chunk> read_xmh(const std::string &path) {
  std::FILE *f = std::fopen(path.c_str(), "rb");
  if (!f) throw std::runtime_error("cannot open " + path);
  uint32_t magic = 0;
  int32_t width = 0, nch = 0;
  if (std::fread(&magic, 4, 1, f) != 1 || std::fread(&width, 4, 1, f) != 1 || std::fread(&nch, 4, 1, f) != 1 || magic != 0x31484D58u || width != W)
    throw std::runtime_error(path + ": not an XMH1 file of width 10240");
  std::vector<Chunk> out(static_cast<size_t>(nch));
  for (auto &c : out) {
    int32_t len = 0;
    if (std::fread(&len, 4, 1, f) != 1) throw std::runtime_error(path + ": short");
    c.tok.resize(static_cast<size_t>(len));
    c.pick.resize(static_cast<size_t>(len));
    c.hid.resize(static_cast<size_t>(len) * W);
    for (int i = 0; i < len; ++i)
      if (std::fread(&c.tok[static_cast<size_t>(i)], 4, 1, f) != 1 || std::fread(&c.pick[static_cast<size_t>(i)], 4, 1, f) != 1 ||
          std::fread(&c.hid[static_cast<size_t>(i) * W], 2, W, f) != static_cast<size_t>(W))
        throw std::runtime_error(path + ": short");
  }
  std::fclose(f);
  return out;
}

}  // namespace

int main(int argc, char **argv) {
  std::string target, mtp, hidden, experts = "g32", dense = "bf16", attn = "full", drafts_out, ablate, head = "bf16";
  int depth = 3, threads = static_cast<int>(std::thread::hardware_concurrency()), eval_from = 0, eval_n = 1 << 30;
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string a = argv[i], v = argv[i + 1];
    if (a == "--target") target = v;
    else if (a == "--mtp") mtp = v;
    else if (a == "--hidden") hidden = v;
    else if (a == "--depth") depth = std::stoi(v);
    else if (a == "--threads") threads = std::stoi(v);
    else if (a == "--eval-from") eval_from = std::stoi(v);
    else if (a == "--eval-n") eval_n = std::stoi(v);
    else if (a == "--experts") experts = v;
    else if (a == "--dense") dense = v;
    else if (a == "--attn") attn = v;
    else if (a == "--drafts-out") drafts_out = v;
    else if (a == "--ablate") ablate = v;
    else if (a == "--head") head = v;
    else {
      std::fprintf(stderr, "unknown option %s\n", a.c_str());
      return 2;
    }
  }
  if (target.empty() || mtp.empty() || hidden.empty() || depth < 1 || depth > 3 || (experts != "g32" && experts != "g128") || (dense != "bf16" && dense != "i8") || (attn != "full" && attn != "qsa") || (head != "bf16" && head != "i8")) {
    std::fprintf(stderr, "usage: xenithium-mtp-ref --target DIR --mtp DIR --hidden F.xmh [--depth 1..3] [--eval-from P] [--experts g32|g128] [--dense bf16|i8] [--attn full|qsa] [--eval-n N] [--drafts-out F] [--ablate X] [--head bf16|i8] [--threads T]\n");
    return 2;
  }
  try {
    const auto t0 = std::chrono::steady_clock::now();
    const auto tck = xm::Checkpoint::open(target);
    const auto mck = xm::Checkpoint::open(mtp);
    // The snapshot's config: one MTP layer, no indexer (plain full attention), rope theta.
    const auto mc = xm::ModelConfig::load(mtp);
    const auto &tc = mc.text();
    const double theta = xm::ModelConfig::need_num(xm::ModelConfig::need(tc, "rope_parameters"), "rope_theta");
    const bool i8 = dense == "i8";
    Drafter d;
    const std::string P = "mtp.", L = "mtp.layers.0.";
    d.V = tck.at("lm_head.weight").dim(0);
    d.embed = read_bf16(tck, "model.language_model.embed_tokens.weight", d.V * H);
    d.lm = read_bf16(tck, "lm_head.weight", d.V * H);
    if (head == "i8") {  // as the target's dense plan stores lm_head (int8 per row, plain amax)
      d.head_i8 = true;
      d.lm8.resize(d.lm.size());
      d.lm8s.resize(static_cast<size_t>(d.V));
      parallel(static_cast<int>((d.V + 4095) / 4096), threads, [&](int b) {
        const int64_t r0 = static_cast<int64_t>(b) * 4096, n = std::min<int64_t>(4096, d.V - r0);
        k::quantize_i8_rows(d.lm.data() + r0 * H, static_cast<int>(n), H, d.lm8.data() + r0 * H, d.lm8s.data() + r0, 1);
      });
    }
    d.norm_e = to_f(read_bf16(mck, P + "pre_fc_norm_embedding.weight", H));
    d.norm_h = to_f(read_bf16(mck, P + "pre_fc_norm_hidden.weight", W));
    d.fe = make_mat(read_bf16(mck, P + "fc_embedding.weight", int64_t{H} * H), H, H, i8);
    d.fh = make_mat(read_bf16(mck, P + "fc_hidden.weight", int64_t{H} * H), H, H, i8);
    auto mixer = [&](const std::string &pre, bool inject, std::vector<uint16_t> &norm, Mat &down, Mat &up) {
      norm = read_bf16(mck, pre + "hc_norm.weight", W);
      auto dn = read_bf16(mck, pre + "input_mix_weight_down.weight", int64_t{R} * W);
      if (inject) dn = cat_rows(dn, read_bf16(mck, pre + "block_inject_weight.weight", int64_t{S} * W));
      down = make_mat(dn, inject ? R + S : R, W, i8);
      up = make_mat(read_bf16(mck, pre + "input_mix_weight_up.weight", int64_t{W} * R), W, R, i8);
    };
    mixer(L + "attn_hyper_connection.", true, d.attn_norm, d.attn_down, d.attn_up);
    mixer(L + "mlp_hyper_connection.", true, d.mlp_norm, d.mlp_down, d.mlp_up);
    if (ablate == "target-final") {  // debug: the target's final mixer instead of the MTP's own
      d.fin_norm = read_bf16(tck, "model.language_model.hyper_connection_mixer.hc_norm.weight", W);
      d.fin_down = make_mat(read_bf16(tck, "model.language_model.hyper_connection_mixer.input_mix_weight_down.weight", int64_t{R} * W), R, W, i8);
      d.fin_up = make_mat(read_bf16(tck, "model.language_model.hyper_connection_mixer.input_mix_weight_up.weight", int64_t{W} * R), W, R, i8);
    } else {
      mixer(P + "hyper_connection_mixer.", false, d.fin_norm, d.fin_down, d.fin_up);
    }
    d.qp = make_mat(read_bf16(mck, L + "self_attn.q_proj.weight", int64_t{QN} * H), QN, H, i8);
    d.kp = make_mat(read_bf16(mck, L + "self_attn.k_proj.weight", int64_t{KVH} * HD * H), KVH * HD, H, i8);
    d.vp = make_mat(read_bf16(mck, L + "self_attn.v_proj.weight", int64_t{KVH} * HD * H), KVH * HD, H, i8);
    d.op = make_mat(read_bf16(mck, L + "self_attn.o_proj.weight", int64_t{H} * QH * HD), H, QH * HD, i8);
    d.qn = to_f(read_bf16(mck, L + "self_attn.q_norm.weight", HD));
    d.kn = to_f(read_bf16(mck, L + "self_attn.k_norm.weight", HD));
    d.qsa = attn == "qsa";
    d.ablate = ablate;
    if (d.qsa) {
      d.idx = make_mat(read_bf16(mck, L + "self_attn.indexer.index_qk_proj.weight", int64_t{5} * 128 * H), 5 * 128, H, false);
      d.iqn = to_f(read_bf16(mck, L + "self_attn.indexer.q_layernorm.weight", 128));
      d.ikn = to_f(read_bf16(mck, L + "self_attn.indexer.k_layernorm.weight", 128));
    }
    d.router = make_mat(cat_rows(read_bf16(mck, L + "mlp.gate.weight", int64_t{E} * H), read_bf16(mck, L + "mlp.shared_expert_gate.weight", H)), E + 1, H, false);
    d.sg = make_mat(read_bf16(mck, L + "mlp.shared_expert.gate_proj.weight", int64_t{SINTER} * H), SINTER, H, i8);
    d.su = make_mat(read_bf16(mck, L + "mlp.shared_expert.up_proj.weight", int64_t{SINTER} * H), SINTER, H, i8);
    d.sd = make_mat(read_bf16(mck, L + "mlp.shared_expert.down_proj.weight", int64_t{H} * SINTER), H, SINTER, i8);
    d.ex.resize(E);
    parallel(E, threads, [&](int e) {
      const std::string b = L + "mlp.experts." + std::to_string(e) + ".";
      Expert &X = d.ex[static_cast<size_t>(e)];
      X.gate = load_g32(mck, b + "gate_proj", INTER, H);
      X.up = load_g32(mck, b + "up_proj", INTER, H);
      X.down = load_g32(mck, b + "down_proj", H, INTER);
      if (experts == "g128") {
        X.gate = requant_g128(X.gate);
        X.up = requant_g128(X.up);
        X.down = requant_g128(X.down);
      }
    });
    const auto chunks = read_xmh(hidden);
    if (ablate == "check-target-head") {  // debug: the target's final mixer (block-free) + lm_head on the dump == its picks?
      std::vector<uint16_t> tn = read_bf16(tck, "model.language_model.hyper_connection_mixer.hc_norm.weight", W);
      const Mat td = make_mat(read_bf16(tck, "model.language_model.hyper_connection_mixer.input_mix_weight_down.weight", int64_t{R} * W), R, W, false);
      const Mat tu = make_mat(read_bf16(tck, "model.language_model.hyper_connection_mixer.input_mix_weight_up.weight", int64_t{W} * R), W, R, false);
      const auto &c = chunks[0];
      const int n = std::min<int>(256, static_cast<int>(c.tok.size()));
      std::atomic<int> same{0};
      parallel(n, threads, [&](int p) {
        std::vector<uint16_t> res(c.hid.begin() + static_cast<size_t>(p) * W, c.hid.begin() + static_cast<size_t>(p + 1) * W), out(H);
        std::vector<float> lo(R);
        k::hc_mix_ref({S, H, R, kEps}, 1, res.data(), nullptr, nullptr, tn.data(), td.ref(), tu.ref(), nullptr, lo.data(), out.data());
        int32_t best = 0;
        double bv = -1e300;
        for (int64_t t = 0; t < d.V; ++t) {
          double acc = 0;
          for (int i = 0; i < H; ++i) acc += static_cast<double>(bf(out[i])) * bf(d.lm[static_cast<size_t>(t) * H + i]);
          if (acc > bv) {
            bv = acc;
            best = static_cast<int32_t>(t);
          }
        }
        same += best == c.pick[static_cast<size_t>(p)];
      });
      std::printf("check-target-head: target final mixer + lm_head on the dumped hidden == the target's pick in %d / %d positions\n", same.load(), n);
      return 0;
    }
    size_t longest = 0;
    for (const auto &c : chunks) longest = std::max(longest, c.tok.size());
    d.rope = k::qsa_rope_table(static_cast<int64_t>(longest) + 4, k::kQsaRot, theta, {});
    std::printf("mtp-ref: loaded in %.1f s (experts %s, dense %s, attention %s, head %s, rope theta %.0f); %zu chunks from %s, depth %d, %d threads\n",
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), experts.c_str(), dense.c_str(), attn.c_str(), head.c_str(), theta, chunks.size(),
                hidden.c_str(), depth, threads);
    int64_t n1 = 0, a1 = 0;  // pass 1 vs the target's next pick, every position
    int64_t chains = 0, acc_sum = 0;  // chains on greedy positions: accepted drafts
    int64_t acc_at[4] = {0, 0, 0, 0};
    for (size_t ci = 0; ci < chunks.size(); ++ci) {
      const auto &c = chunks[ci];
      const int len = static_cast<int>(c.tok.size());
      const auto tc0 = std::chrono::steady_clock::now();
      // pass 1 at every position p < len - 1 (input token t_{p+1}); its KV is the cache.
      std::vector<Pre> pre(static_cast<size_t>(len - 1));
      parallel(len - 1, threads, [&](int p) { pre[static_cast<size_t>(p)] = pass_front(d, &c.hid[static_cast<size_t>(p) * W], c.tok[static_cast<size_t>(p) + 1], p); });
      // qsa: pass 1's compressed keys of every complete block
      std::vector<std::vector<float>> kc1;
      if (d.qsa) {
        kc1.resize(static_cast<size_t>(len - 1) / 4);
        parallel(static_cast<int>(kc1.size()), threads, [&](int b) {
          const float *raw[4];
          for (int j = 0; j < 4; ++j) raw[j] = pre[static_cast<size_t>(b) * 4 + j].ik.data();
          kc1[static_cast<size_t>(b)] = compress(d, raw, b);
        });
      }
      // greedy region: from where every later pick is the next text token (generated text)
      int greedy_from = len - 1;
      while (greedy_from > 0 && c.pick[static_cast<size_t>(greedy_from) - 1] == c.tok[static_cast<size_t>(greedy_from)]) --greedy_from;
      const int p0 = std::max(eval_from, 0);
      const int np = std::min(eval_n, std::max(0, len - 1 - p0));
      std::vector<int32_t> dr(static_cast<size_t>(np) * 3, -1);
      parallel(np, threads, [&](int i) {
        const int p = p0 + i;
        std::vector<const Kv *> kvs;
        kvs.reserve(static_cast<size_t>(p) + 4);
        for (int q = 0; q <= p; ++q) kvs.push_back(&pre[static_cast<size_t>(q)].kv);
        Pre cur = pre[static_cast<size_t>(p)];
        std::vector<Pre> chain;
        chain.reserve(4);  // kvs points into it
        // complete blocks at position pos: pass 1's below p + 1, recompressed where a chain position closes one
        std::vector<std::vector<float>> kcx;
        kcx.reserve(4);
        auto blocks = [&](int pos) {
          std::vector<const float *> kcs;
          if (!d.qsa) return kcs;
          for (int b = 0; b < (pos + 1) / 4; ++b) {
            if (4 * b + 3 <= p) {
              kcs.push_back(kc1[static_cast<size_t>(b)].data());
              continue;
            }
            const float *raw[4];
            for (int j = 0; j < 4; ++j) {
              const int t = 4 * b + j;
              raw[j] = t <= p ? pre[static_cast<size_t>(t)].ik.data() : chain[static_cast<size_t>(t - p - 1)].ik.data();
            }
            kcx.push_back(compress(d, raw, b));
            kcs.push_back(kcx.back().data());
          }
          return kcs;
        };
        int32_t tokd = pass_back(d, cur, kvs, blocks(p));
        dr[static_cast<size_t>(i) * 3] = tokd;
        for (int kk = 2; kk <= depth && p + kk < len; ++kk) {
          chain.push_back(pass_front(d, cur.res.data(), tokd, p + kk - 1));
          kvs.push_back(&chain.back().kv);
          Pre nx = chain.back();
          tokd = pass_back(d, nx, kvs, blocks(p + kk - 1));
          cur = std::move(nx);
          dr[static_cast<size_t>(i) * 3 + kk - 1] = tokd;
        }
      });
      if (!drafts_out.empty()) {
        std::FILE *f = std::fopen(drafts_out.c_str(), ci == 0 ? "wb" : "ab");
        if (!f) throw std::runtime_error("cannot write " + drafts_out);
        for (int i = 0; i < np; ++i) {
          const int32_t rec[5] = {static_cast<int32_t>(ci), p0 + i, dr[static_cast<size_t>(i) * 3], dr[static_cast<size_t>(i) * 3 + 1], dr[static_cast<size_t>(i) * 3 + 2]};
          std::fwrite(rec, 4, 5, f);
        }
        std::fclose(f);
      }
      int64_t cn1 = 0, ca1 = 0, cch = 0, cacc = 0;
      for (int i = 0; i < np; ++i) {
        const int p = p0 + i;
        if (p + 1 >= len) continue;
        ++cn1;
        ca1 += dr[static_cast<size_t>(i) * 3] == c.pick[static_cast<size_t>(p) + 1];
        if (p + 1 >= greedy_from && p + depth < len) {
          ++cch;
          int a = 0;
          while (a < depth && dr[static_cast<size_t>(i) * 3 + a] == c.pick[static_cast<size_t>(p + a) + 1]) ++a;
          cacc += a;
          ++acc_at[a];
        }
      }
      std::printf("  chunk %zu: %d positions (greedy from %d), evaluated from %d: pass-1 agreement %lld/%lld (%.1f %%); MTP-%d chains %lld, "
                  "accepted %.3f drafts per step (%.1f s)\n",
                  ci, len, greedy_from, p0, static_cast<long long>(ca1), static_cast<long long>(cn1), cn1 ? 100.0 * ca1 / cn1 : 0.0, depth,
                  static_cast<long long>(cch), cch ? static_cast<double>(cacc) / cch : 0.0,
                  std::chrono::duration<double>(std::chrono::steady_clock::now() - tc0).count());
      n1 += cn1;
      a1 += ca1;
      chains += cch;
      acc_sum += cacc;
    }
    std::printf("mtp-ref: pass-1 agreement %.2f %% (%lld positions); MTP-%d on greedy text: %.3f drafts accepted per step over %lld chains "
                "(0: %lld, 1: %lld, 2: %lld, 3: %lld) -> %.3f tokens per target step\n",
                n1 ? 100.0 * a1 / n1 : 0.0, static_cast<long long>(n1), depth, chains ? static_cast<double>(acc_sum) / chains : 0.0,
                static_cast<long long>(chains), static_cast<long long>(acc_at[0]), static_cast<long long>(acc_at[1]), static_cast<long long>(acc_at[2]),
                static_cast<long long>(acc_at[3]), chains ? 1.0 + static_cast<double>(acc_sum) / chains : 0.0);
  } catch (const std::exception &e) {
    std::fprintf(stderr, "mtp-ref: %s\n", e.what());
    return 1;
  }
  return 0;
}
