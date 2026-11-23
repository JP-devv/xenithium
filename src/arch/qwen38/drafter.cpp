// Xenithium · qwen38 MTP drafter on the cards: load and the draft passes (Model members).
//
// A pass (rows = 1..4 positions p .. p + rows - 1), per card, in the target's kernels:
//   embed_gather(ids) -> e;  front_norm(e, h) -> en, hn;  fe = en . Wfe^T;  fh_r = hn_r . Wfh^T
//   front_add -> x (into the drafter's state);  attn mixer -> qsa_decode (own KV, the pair wire)
//   -> mlp mixer -> moe_decode (own experts, the MoE wire) -> the MTP final mixer -> lm_head + pick
// Pass 1 takes the target's state after its final combine (Head::residual) at the verify step's
// rows; pass k >= 2 takes pass k - 1's state at one row and its pick.
#include <algorithm>
#include <cstdlib>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <unistd.h>

#include "model_internal.hpp"
#include "xenithium/arch/qwen38/mtp.hpp"
#include "xenithium/kernels/head/head.hpp"
#include "xenithium/kernels/drafter/mtp.hpp"
#include "xenithium/kernels/pack.hpp"
#include "xenithium/kernels/gemm/pf_dense.hpp"
#include "xenithium/kernels/gemm/w4a16_gemm.hpp"
#include "xenithium/model/config.hpp"
#include "xenithium/util/json.hpp"

namespace xenithium::arch::qwen38 {

namespace k = xenithium::kernels;
using bf16 = sycl::ext::oneapi::bfloat16;

namespace {

double since_s(std::chrono::steady_clock::time_point t0) { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); }

std::vector<uint8_t> read_tensor(const model::Checkpoint &ck, const model::TensorInfo &t) {
  std::vector<uint8_t> out(t.bytes);
  const int fd = ::open(ck.shards()[t.shard].path.c_str(), O_RDONLY);
  if (fd < 0) throw std::runtime_error("qwen38 mtp: cannot open " + ck.shards()[t.shard].path);
  size_t got = 0;
  while (got < out.size()) {
    const ssize_t r = ::pread(fd, out.data() + got, out.size() - got, static_cast<off_t>(t.offset + got));
    if (r <= 0) {
      ::close(fd);
      throw std::runtime_error("qwen38 mtp: short read of " + t.name);
    }
    got += static_cast<size_t>(r);
  }
  ::close(fd);
  return out;
}

template <class F>
void parallel_for_n(int n, int threads, F &&f) {
  std::atomic<int> next{0};
  std::mutex em;
  std::string err;
  std::vector<std::thread> pool;
  for (int t = 0; t < std::max(1, threads); ++t)
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

}  // namespace

kernels::MoeShape Model::mtp_moe_shape(int c) const {
  kernels::MoeShape s = ms_.shape_of(c);
  s.rot = mtp_ && mtp_->rot;  // the drafter's own basis, not the target's
  return s;
}

// The drafter's lm_head: its int2 draft head when baked, else the target's.
const std::vector<kernels::DenseWeights> *Model::mtp_head() {
  if (!mtp_ || !mtp_->cards[0].dhead.w) return nullptr;
  mtp_head_w_ = per_card([&](int c) { return mtp_->cards[static_cast<size_t>(c)].dhead; });
  return &mtp_head_w_;
}

// The draft vocabulary head (the target's lm_head rows of the subset, int8 like lm_head) when loaded: its GEMM, the
// pick over its rows (virtual indices, card c's from c * sub_n), then virtual -> token id. Else the head's lm_pick on
// the int2 draft head or the target's lm_head.
// With the draft vocabulary head, a sampled request's rows pick argmax(logit / T + Gumbel noise) (draft_gumbel;
// the slot from the pass's segment records, or slot_fixed for the prefill's last slice); greedy rows are untouched.
void Model::mtp_lm_pick(int rows, const std::vector<const Head::bf16 *> &in, const std::vector<int32_t *> &tok, const std::vector<float *> &logits,
                        int pass, int seg_rows, int slot_fixed) {
  Mtp &M = *mtp_;
  if (!M.cards[0].vhead.w && !M.cards[0].vhead4.w) {
    head_->lm_pick(rows, in, tok, logits, mtp_head());
    return;
  }
  const int n = M.sub_n, kk = M.sub_k, s0 = M.sub_s0;
  for (int c = 0; c < nc_; ++c) {
    const Mtp::Card &X = M.cards[static_cast<size_t>(c)];
    if (X.vhead4.w)  // int4 rows
      k::w4a16_gemm_rows(fab_.compute(c), rows, in[static_cast<size_t>(c)], X.vhead4.w, X.vhead4.scales, logits[static_cast<size_t>(c)], X.vhead4.n, X.vhead4.k);
    else
      k::dense_gemm(fab_.compute(c), rows, in[static_cast<size_t>(c)], X.vhead, logits[static_cast<size_t>(c)]);
  }
  for (int c = 0; c < nc_; ++c) {
    auto &q = fab_.compute(c);
    const auto &hc = head_->card(c);
    const Mtp::Card &X = M.cards[static_cast<size_t>(c)];
    k::draft_gumbel(q, logits[static_cast<size_t>(c)], n, rows, n, c * n, kk, s0, slot_fixed < 0 ? X.seg + static_cast<size_t>(pass) * k::kMaxSegs * k::kSegRec : nullptr,
                    k::kSegRec, seg_rows, slot_fixed, X.noise, pass);
    k::greedy_pick(q, logits[static_cast<size_t>(c)], n, rows, n, c * n, hc.scratch, nc_ > 1 ? &hc.wire : nullptr, tok[static_cast<size_t>(c)]);
    int32_t *t = tok[static_cast<size_t>(c)];
    q.parallel_for(sycl::range<1>(static_cast<size_t>(rows)), [=](sycl::id<1> r) {
      const int32_t v = t[r];
      t[r] = v < kk ? v : v - kk + s0;
    });
  }
}

void Model::set_draft_noise(int slot, uint64_t key, uint64_t pos, float inv_t) {
  if (dflash_) dflash_->set_noise(slot, key, pos, inv_t);  // the DFlash drafter couples too
  if (!mtp_ || slot < 0 || slot >= slots()) return;
  auto &h = mtp_->noise_host;
  if (h.size() != static_cast<size_t>(slots())) h.assign(static_cast<size_t>(slots()), k::DraftNoiseRec{});
  auto &r = h[static_cast<size_t>(slot)];
  if (r.key != key || r.pos != pos || r.inv_t != inv_t) mtp_->noise_dirty = true;
  r.key = key, r.pos = pos, r.inv_t = inv_t;
}

void Model::mtp_noise_upload() {
  Mtp &M = *mtp_;
  if (M.noise_host.size() != static_cast<size_t>(slots())) M.noise_host.assign(static_cast<size_t>(slots()), k::DraftNoiseRec{}), M.noise_dirty = true;
  if (!M.noise_dirty) return;  // the cards hold these values (no copy per pass)
  std::copy(M.noise_host.begin(), M.noise_host.end(), M.noise_pin);
  for (int c = 0; c < nc_; ++c)
    fab_.compute(c).memcpy(M.cards[static_cast<size_t>(c)].noise, M.noise_pin, M.noise_host.size() * sizeof(k::DraftNoiseRec));
  M.noise_dirty = false;
}

int Model::mtp_draft_vocab() const { return mtp_ && mtp_->sub_k > 0 ? mtp_->sub_k + 128 : 0; }

void Model::load_mtp(const std::string &dir) {
  const auto t0 = std::chrono::steady_clock::now();
  auto M = std::make_unique<Mtp>();
  {  // the passes' pinned copy sources (released with M->mem)
    auto pin = [&](size_t bytes) {
      void *p = fab_.alloc_host(std::max<size_t>(bytes, 4096));
      std::memset(p, 0, std::max<size_t>(bytes, 4096));
      M->mem.push_back(p);
      return p;
    };
    M->seg_host = static_cast<int32_t *>(pin(static_cast<size_t>(Mtp::kPasses) * k::kMaxSegs * k::kSegRec * 4));
    M->ids_host = static_cast<int32_t *>(pin(static_cast<size_t>(Head::kRows) * 4));
    M->noise_pin = static_cast<k::DraftNoiseRec *>(pin(static_cast<size_t>(slots()) * sizeof(k::DraftNoiseRec)));
  }
  M->ck = model::Checkpoint::open(dir);
  const auto &ck = M->ck;
  // A baked drafter says in its config whether its experts are rotated.
  if (std::filesystem::exists(dir + "/config.json")) {
    const auto mc = model::ModelConfig::load(dir);
    if (const json::Value *qc = mc.root().get("quantization_config")) {
      const std::string *rot = qc->get_string("xenithium_rotation");
      if (rot && *rot != "hadamard-x128-h64") throw std::runtime_error("qwen38 mtp: xenithium_rotation " + *rot + " (only hadamard-x128-h64)");
      M->rot = rot != nullptr;
    }
  }
  M->dense = std::make_unique<load::DenseStore>(fab_);
  auto plan = mtp_dense_plan(ck);
  for (auto &sp : plan)  // the calibration targets' BF16 drafter
    if (sp.store == model::DenseSpec::Store::i8)
      for (const auto &pat : opt_.mtp_bf16_tensors)
        if (!pat.empty() && sp.name.find(pat) != std::string::npos) {
          sp.store = model::DenseSpec::Store::bf16;
          ++ls_.mtp_bf16_overrides;
          break;
        }
  M->dense->load(ck, plan, opt_.threads);
  const auto &D = *M->dense;
  const int H = cfg_.hidden, W = hs_.shape.width(), E = cfg_.experts;
  const std::string P = kMtpPrefix, L = kMtpLayer, A = L + "self_attn.";
  uint64_t vram = 0;
  for (const auto &kv : D.stats().card_bytes[0]) vram += kv.second;
  M->cards.resize(static_cast<size_t>(nc_));
  for (int c = 0; c < nc_; ++c) {
    Mtp::Card &X = M->cards[static_cast<size_t>(c)];
    auto raw = [&](const std::string &n) { return reinterpret_cast<const bf16 *>(D.at(c, n).ptr); };
    X.norm_e = raw(P + "pre_fc_norm_embedding.weight");
    X.norm_h = raw(P + "pre_fc_norm_hidden.weight");
    X.fe = D.at(c, P + "fc_embedding.weight").dense();
    X.fh = D.at(c, P + "fc_hidden.weight").dense();
    if (X.fe.n != H || X.fe.k != H || X.fh.n != H || X.fh.k != H || D.at(c, P + "pre_fc_norm_hidden.weight").bytes != static_cast<size_t>(W) * 2)
      throw std::runtime_error("qwen38 mtp: unexpected front shapes");
    auto mixer = [&](const std::string &pre, bool inject) {
      k::HcWeights w;
      w.norm = raw(pre + "hc_norm.weight");
      w.down = D.at(c, pre + "input_mix_weight_down.weight").dense();
      w.up = D.at(c, pre + "input_mix_weight_up.weight").dense();
      if (w.down.k != W || w.up.n != W || w.up.k != cfg_.hc_lowrank || (inject ? w.down.n != hs_.down_ld : w.down.n != cfg_.hc_lowrank))
        throw std::runtime_error("qwen38 mtp: unexpected mixer shapes for " + pre);
      return w;
    };
    X.attn = mixer(L + "attn_hyper_connection.", true);
    X.mlp = mixer(L + "mlp_hyper_connection.", true);
    X.fin = mixer(P + "hyper_connection_mixer.", false);
    const auto qsh = mtp_shape(qs_, opt_.mtp_kv, c);
    X.qw.qkv = D.at(c, A + "q_proj.weight").dense();
    X.qw.idx = D.at(c, A + "indexer.index_qk_proj.weight").dense();
    X.qw.o = D.at(c, A + "o_proj.weight").dense();
    X.qw.q_norm = raw(A + "q_norm.weight");
    X.qw.k_norm = raw(A + "k_norm.weight");
    X.qw.iq_norm = raw(A + "indexer.q_layernorm.weight");
    X.qw.ik_norm = raw(A + "indexer.k_layernorm.weight");
    if (X.qw.qkv.n != qsh.qkv_n() || X.qw.idx.n != qsh.idx_n() || X.qw.idx.kind != k::DenseKind::bf16 || X.qw.o.n != H || X.qw.o.k != qsh.o_k())
      throw std::runtime_error("qwen38 mtp: unexpected attention shapes");
    const auto &r = D.at(c, L + "mlp.gate.weight");
    X.mw.router = r.dense();
    X.mw.shared_gu = D.at(c, L + "mlp.shared_expert.gate_proj.weight").dense();
    X.mw.shared_down = D.at(c, L + "mlp.shared_expert.down_proj.weight").dense();
    X.mw.blob = ms_.w[static_cast<size_t>(c)][0].blob;  // the target's blob layout (EP or TP)
    if (r.rows != E + 1 || X.mw.router.kind != k::DenseKind::bf16 || X.mw.shared_gu.n != 2 * cfg_.shared_inter || X.mw.shared_down.n != H ||
        X.mw.shared_down.k != cfg_.shared_inter)
      throw std::runtime_error("qwen38 mtp: unexpected MoE shapes");
  }

  // ---- states and per-pass buffers ----
  for (int c = 0; c < nc_; ++c) {
    Mtp::Card &X = M->cards[static_cast<size_t>(c)];
    auto &q = fab_.compute(c);
    auto dev = [&](size_t bytes) {
      bytes = std::max<size_t>(bytes, 64);
      void *p = fab_.alloc(c, bytes);
      q.memset(p, 0, bytes);
      M->mem.push_back(p);
      if (c == 0) vram += bytes;
      return p;
    };
    const size_t R = Head::kRows;
    for (int i = 0; i < 2; ++i) {
      X.st.res[i] = static_cast<bf16 *>(dev(R * W * 2));
      X.st.lora[i] = static_cast<float *>(dev(R * static_cast<size_t>(head_->lora_ld()) * 4));
    }
    X.e = static_cast<bf16 *>(dev(R * H * 2));
    X.en = static_cast<bf16 *>(dev(R * H * 2));
    X.hn = static_cast<bf16 *>(dev(R * W * 2));
    X.fe_y = static_cast<float *>(dev(R * H * 4));
    X.fh_y = static_cast<float *>(dev(R * W * 4));
    X.out = static_cast<bf16 *>(dev(R * H * 2));
    X.ids = static_cast<int32_t *>(dev(R * 4));  // a step's rows
    X.gh = static_cast<bf16 *>(dev(R * W * 2));
    X.gid = static_cast<int32_t *>(dev(R * 4));
    X.tok = static_cast<int32_t *>(dev(Mtp::kPasses * Mtp::kTokStride * 4));
    X.logits = static_cast<float *>(dev(head_->logits_floats() * 4));
    X.seg = static_cast<int32_t *>(dev(static_cast<size_t>(Mtp::kPasses) * k::kMaxSegs * k::kSegRec * 4));
    X.noise = static_cast<k::DraftNoiseRec *>(dev(static_cast<size_t>(slots()) * sizeof(k::DraftNoiseRec)));
    fab_.compute(c).memset(X.noise, 0, static_cast<size_t>(slots()) * sizeof(k::DraftNoiseRec)).wait();
    X.route = static_cast<k::MoeRoute *>(dev(sizeof(k::MoeRoute)));
    // KV of the drafter's attention layer, per slot as the target's; on the target's page
    // table and pool (a page holds the same 64 positions of a sequence in both), its own indexer ring per slot; the
    // kernels read the slot from the pass's segment records (the pool's pages: Model::kv_pool_alloc, once the VRAM left
    // for it is known)
    const size_t P = static_cast<size_t>(cap_) / k::kQsaPage, NS = static_cast<size_t>(slots());
    X.qc.ring = static_cast<bf16 *>(dev(NS * k::kQsaRing * k::kQsaIdxDim * 2));
    X.qc.pages = kv_table_[c];
    X.qc.max_tokens = static_cast<int>(cap_);
    X.qc.slot_pages = static_cast<int>(P);
    q.wait();
  }

  // ---- routed experts: g32 -> g128 requant, the target's blob (TP halves or EP owners) ----
  const ExpertLayout lay = ExpertLayout::of(cfg_);
  const bool tp = ms_.shape.tp;
  const int per = ms_.shape.per_card();
  size_t stride = lay.stride;
  k::MoeBlobLayout tl;
  if (tp) tl = k::moe_tp_layout(H, cfg_.moe_inter, nc_, &stride);
  std::vector<std::vector<uint8_t>> host(static_cast<size_t>(nc_), std::vector<uint8_t>(static_cast<size_t>(per) * stride, 0));
  std::atomic<int64_t> inexact{0};
  std::mutex em;
  double err2 = 0, ref2 = 0;
  const int I = cfg_.moe_inter;
  const bool gptq = ck.find(mtp_expert_name(0, "gate_proj", "qweight")) != nullptr;  // a baked drafter
  parallel_for_n(E, opt_.threads, [&](int e) {
    W4Rows m[3];
    const char *proj[3] = {"gate_proj", "up_proj", "down_proj"};
    const int n[3] = {I, I, H}, kd[3] = {H, H, I};
    double le = 0, lr = 0;
    if (gptq) {  // baked g128 experts (the target's GPTQ layout, symmetric: code q - 8, fp16 scales [k/128][n])
      for (int j = 0; j < 3; ++j) {
        const auto qw = read_tensor(ck, ck.at(mtp_expert_name(e, proj[j], "qweight")));
        const auto sc = read_tensor(ck, ck.at(mtp_expert_name(e, proj[j], "scales")));
        if (qw.size() != static_cast<size_t>(n[j]) * kd[j] / 2 || sc.size() != static_cast<size_t>(n[j]) * (kd[j] / 128) * 2)
          throw std::runtime_error("qwen38 mtp: unexpected g128 expert tensor shapes for expert " + std::to_string(e));
        const auto *qp = reinterpret_cast<const uint32_t *>(qw.data());
        const auto *sp = reinterpret_cast<const uint16_t *>(sc.data());
        W4Rows &w = m[j];
        w.n = n[j], w.k = kd[j], w.G = 128;
        w.q.resize(static_cast<size_t>(w.n) * w.k);
        w.s.resize(static_cast<size_t>(w.n) * (w.k / 128));
        for (int kk = 0; kk < w.k; ++kk)
          for (int col = 0; col < w.n; ++col)
            w.q[static_cast<size_t>(col) * w.k + kk] = static_cast<int8_t>(static_cast<int>((qp[static_cast<size_t>(kk / 8) * w.n + col] >> (4 * (kk % 8))) & 15u) - 8);
        for (int g = 0; g < w.k / 128; ++g)
          for (int col = 0; col < w.n; ++col) w.s[static_cast<size_t>(col) * (w.k / 128) + g] = fp16_value(sp[static_cast<size_t>(g) * w.n + col]);
      }
    }
    for (int j = 0; !gptq && j < 3; ++j) {
      const auto &tp_ = ck.at(mtp_expert_name(e, proj[j], "weight_packed"));
      const auto &ts = ck.at(mtp_expert_name(e, proj[j], "weight_scale"));
      if (tp_.bytes != static_cast<uint64_t>(n[j]) * kd[j] / 2 || ts.dtype_str != "BF16" || ts.numel() != static_cast<int64_t>(n[j]) * kd[j] / 32)
        throw std::runtime_error("qwen38 mtp: unexpected expert tensor shapes for expert " + std::to_string(e));
      const auto pk = read_tensor(ck, tp_);
      const auto sb = read_tensor(ck, ts);
      const W4Rows g32 = mtp_unpack_g32(pk.data(), reinterpret_cast<const uint16_t *>(sb.data()), n[j], kd[j]);
      m[j] = mtp_requant_g128(g32);
      for (size_t i = 0; i < g32.q.size(); ++i) {
        const double a = static_cast<double>(g32.q[i]) * g32.s[i / 32], b = static_cast<double>(m[j].q[i]) * m[j].s[i / 128];
        le += (a - b) * (a - b);
        lr += a * a;
      }
    }
    std::vector<uint8_t> full(lay.stride);
    inexact += mtp_pack_expert(lay, m[0], m[1], m[2], full.data());
    if (tp) {
      for (int c = 0; c < nc_; ++c)
        k::moe_split_tp(moe_blob_layout(lay), H, I, nc_, c, full.data(), tl, host[static_cast<size_t>(c)].data() + static_cast<size_t>(e) * stride);
    } else {
      std::memcpy(host[static_cast<size_t>(e / per)].data() + static_cast<size_t>(e % per) * stride, full.data(), lay.stride);
    }
    std::lock_guard<std::mutex> l(em);
    err2 += le;
    ref2 += lr;
  });
  for (int c = 0; c < nc_; ++c) {
    Mtp::Card &X = M->cards[static_cast<size_t>(c)];
    auto &q = fab_.compute(c);
    const size_t bytes = static_cast<size_t>(per) * stride;
    X.experts = static_cast<uint8_t *>(fab_.alloc(c, bytes));
    M->mem.push_back(X.experts);
    q.memcpy(X.experts, host[static_cast<size_t>(c)].data(), bytes).wait();
    std::vector<const uint8_t *> t(static_cast<size_t>(per));
    for (int i = 0; i < per; ++i) t[static_cast<size_t>(i)] = X.experts + static_cast<size_t>(i) * stride;
    X.tab = static_cast<const uint8_t **>(fab_.alloc(c, t.size() * 8));
    M->mem.push_back(X.tab);
    q.memcpy(X.tab, t.data(), t.size() * 8).wait();
    if (c == 0) vram += bytes + t.size() * 8;
  }
  // The int2 draft head (mtp.draft_head.codes U8 [vocab][hidden / 4], element c of a row at
  // bits 2 (c % 4) of byte c / 4; scales F32 [hidden / 128][vocab]), this card's vocab slice.
  if (const model::TensorInfo *dc = ck.find("mtp.draft_head.codes")) {
    const model::TensorInfo &ds = ck.at("mtp.draft_head.scales");
    const int V = cfg_.vocab, vpc = head_->vocab_per_card(), G = H / 128;
    if (dc->bytes != static_cast<uint64_t>(V) * H / 4 || ds.bytes != static_cast<uint64_t>(G) * V * 4)
      throw std::runtime_error("qwen38 mtp: draft head shapes (want codes [vocab][hidden / 4], scales [hidden / 128][vocab])");
    const auto codes = read_tensor(ck, *dc);
    const auto scl = read_tensor(ck, ds);
    for (int c = 0; c < nc_; ++c) {
      std::vector<uint8_t> un(static_cast<size_t>(vpc) * H), packed(k::dense_i2_bytes(vpc, H));
      for (int r = 0; r < vpc; ++r) {
        const uint8_t *row = codes.data() + static_cast<size_t>(c * vpc + r) * (H / 4);
        for (int i = 0; i < H; ++i) un[static_cast<size_t>(r) * H + i] = (row[i / 4] >> (2 * (i % 4))) & 3u;
      }
      k::pack_dense_i2(un.data(), vpc, H, packed.data());
      std::vector<float> sl(static_cast<size_t>(G) * vpc);
      const float *sf = reinterpret_cast<const float *>(scl.data());
      for (int g = 0; g < G; ++g)
        for (int r = 0; r < vpc; ++r) sl[static_cast<size_t>(g) * vpc + r] = sf[static_cast<size_t>(g) * V + c * vpc + r];
      Mtp::Card &X = M->cards[static_cast<size_t>(c)];
      auto &q = fab_.compute(c);
      void *dw = fab_.alloc(c, packed.size());
      void *dsc = fab_.alloc(c, sl.size() * 4);
      M->mem.push_back(dw);
      M->mem.push_back(dsc);
      q.memcpy(dw, packed.data(), packed.size());
      q.memcpy(dsc, sl.data(), sl.size() * 4).wait();
      X.dhead.w = dw;
      X.dhead.scale = static_cast<const float *>(dsc);
      X.dhead.kind = k::DenseKind::i2;
      X.dhead.n = vpc;
      X.dhead.k = H;
      if (c == 0) vram += packed.size() + sl.size() * 4;
    }
  }
  // ----: the draft vocabulary head (the target's lm_head rows, from the target checkpoint) ----
  if (opt_.mtp_head_w4 && opt_.mtp_vocab <= 0) throw std::invalid_argument("qwen38 mtp: mtp_head_w4 needs a draft vocabulary (mtp_vocab > 0)");
  if (opt_.mtp_vocab > 0) {
    if (M->cards[0].dhead.w) throw std::invalid_argument("qwen38 mtp: a draft vocabulary with the int2 draft head (use one)");
    if (opt_.mtp_vocab % 64 || cfg_.eos.empty()) throw std::invalid_argument("qwen38 mtp: the draft vocabulary K a multiple of 64, and an eos id");
    const int64_t e0 = *std::min_element(cfg_.eos.begin(), cfg_.eos.end()), s0 = e0 / 64 * 64;
    if (opt_.mtp_vocab > s0 || s0 + 128 > cfg_.vocab) throw std::invalid_argument("qwen38 mtp: the draft vocabulary reaches the special tokens");
    model::DenseSpec sp;
    sp.name = "lm_head.weight";
    sp.split = model::DenseSpec::Split::rows;
    // as the target's lm_head (DeployParams' default dense bytes). Engine (mtp_head_w4): int4 - the target
    // directory's GPTQ rows (`lm_head.qweight`, xenithium-bake head4-calib) when it has them, else RTN at load
    sp.store = model::DenseSpec::Store::i8;
    if (opt_.mtp_head_w4) {
      if (ck_.find("lm_head.qweight")) {
        sp.name = "lm_head.qweight";
        sp.store = model::DenseSpec::Store::w4;
        sp.w4_layout = model::DenseSpec::W4Layout::gptq;
      } else {
        sp.store = model::DenseSpec::Store::w4r;
      }
    }
    sp.comp = model::Component::lm_head;
    sp.row_ranges = {{0, opt_.mtp_vocab}, {s0, s0 + 128}};
    sp.key = "mtp.draft_vocab_head";
    M->vdense = std::make_unique<load::DenseStore>(fab_);
    M->vdense->load(ck_, {sp}, opt_.threads);
    M->sub_k = opt_.mtp_vocab;
    M->sub_s0 = static_cast<int>(s0);
    M->sub_n = (opt_.mtp_vocab + 128) / nc_;  // per card (TP1: all)
    for (int c = 0; c < nc_; ++c) {
      Mtp::Card &X = M->cards[static_cast<size_t>(c)];
      const load::DeviceTensor &t = M->vdense->at(c, sp.key);
      int n = 0, kk = 0;
      if (t.store == model::DenseSpec::Store::w4) {  // 
        const load::W4Weights w = t.w4();
        X.vhead4 = k::W4Mat{w.w, w.scales, w.n, w.k};
        n = w.n, kk = w.k;
      } else {
        X.vhead = t.dense();
        n = X.vhead.n, kk = X.vhead.k;
      }
      if (n != M->sub_n || kk != H) throw std::runtime_error("qwen38 mtp: draft vocabulary head slice shape");
    }
    for (const auto &kv : M->vdense->stats().card_bytes[0]) vram += kv.second;
  }
  ls_.mtp_scales_inexact = inexact.load();
  ls_.mtp_requant_err = ref2 > 0 ? std::sqrt(err2 / ref2) : 0;
  ls_.mtp_vram = vram;
  ls_.mtp_s = since_s(t0);
  mtp_ = std::move(M);
}

void Model::mtp_pass(int pass, int rows, const std::vector<const bf16 *> &hidden, const std::vector<const int32_t *> &ids,
                     const std::vector<std::pair<int, int64_t>> &segs, int seg_rows, bool kv_only) {
  namespace sx = sycl::ext::oneapi::experimental;
  Mtp &M = *mtp_;
  const int ns = static_cast<int>(segs.size());
  // seg_rows: rows per segment, or a seg_het word: unequal segments, as the verify it follows
  const bool layout_ok = seg_rows > 0 ? seg_rows * ns == rows : k::seg_rows_ok(rows, seg_rows, k::kSegRec) && k::seg_het_n(seg_rows) == ns;
  if (pass < 0 || pass >= Mtp::kPasses || rows < 1 || rows > Head::kRows || ns < 1 || ns > k::kMaxSegs || seg_rows == 0 || !layout_ok)
    throw std::invalid_argument("qwen38 mtp: pass / rows / segments out of range");
  // Segment records (rows.hpp): position and slot per segment, device words (outside any graph;
  // the source lives in the Mtp state)
  int32_t *w = M.seg_host + static_cast<size_t>(pass) * k::kMaxSegs * k::kSegRec;
  std::fill(w, w + k::kMaxSegs * k::kSegRec, 0);
  int64_t need = 0;
  for (int i = 0; i < ns; ++i) {
    const auto &[slot, pos] = segs[static_cast<size_t>(i)];
    int r0 = 0, nr = 0;
    k::seg_span(seg_rows, rows, i, r0, nr);
    if (slot < 0 || slot >= slots() || pos < 0 || pos + nr > cap_) throw std::invalid_argument("qwen38 mtp: slot / position out of range");
    kv_need(slot, pos + nr);  // the drafter KV rows this pass writes (shared pages with the target's)
    w[i * k::kSegRec + k::kSegPos] = static_cast<int32_t>(pos);
    w[i * k::kSegRec + k::kSegRope] = static_cast<int32_t>(seqs_[static_cast<size_t>(slot)].rope_delta);  // its YaRN tier
    w[i * k::kSegRec + k::kSegSlot] = slot;
    need = std::max(need, pos + nr);
  }
  for (int c = 0; c < nc_; ++c)
    fab_.compute(c).memcpy(M.cards[static_cast<size_t>(c)].seg + static_cast<size_t>(pass) * k::kMaxSegs * k::kSegRec, w, static_cast<size_t>(ns) * k::kSegRec * 4);
  if (!kv_only) mtp_noise_upload();
  if (!opt_.graphs || profiling_) {
    mtp_submit(pass, rows, hidden, ids, static_cast<int>(need), seg_rows, kv_only);
    return;
  }
  const int64_t bound = bound_bucket(need);
  const auto &st0 = M.cards[0].st;
  const Mtp::GraphKey key{pass, rows, seg_rows, hidden[0], ids[0], pair_parity_, moe_parity_, bound, st0.cur, st0.lcur, st0.lora_ld, kv_only ? 1 : 0};
  auto it = M.graphs.find(key);
  if (it == M.graphs.end()) {
    graph_room();  // as run_graph
    try {
      mtp_record_graph(pass, rows, hidden, ids, seg_rows, bound, kv_only);
    } catch (const GraphRecordError &) {
      graph_room(true);
      try {
        mtp_record_graph(pass, rows, hidden, ids, seg_rows, bound, kv_only);
      } catch (const GraphRecordError &y) {
        if (graphs_eager_++ == 0) std::fprintf(stderr, "qwen38: %s; this pass runs without a graph\n", y.what());
        mtp_submit(pass, rows, hidden, ids, static_cast<int>(bound), seg_rows, kv_only);
        return;
      }
    }
    it = M.graphs.find(key);
  }
  it->second.used = ++graph_tick_;
  // Both cards' graphs in flight together: the mailbox waits inside pair up across the cards.
  for (int c = 0; c < nc_; ++c) fab_.compute(c).ext_oneapi_graph(it->second.exec[static_cast<size_t>(c)]);
  pair_parity_ = it->second.end_pair;
  moe_parity_ = it->second.end_moe;
  for (size_t c = 0; c < static_cast<size_t>(nc_); ++c) {
    auto &st = M.cards[c].st;
    st.cur = it->second.end_hc[c].cur;
    st.lcur = it->second.end_hc[c].lcur;
    st.lora_ld = it->second.end_hc[c].lora_ld;
  }
}

void Model::mtp_record_graph(int pass, int rows, const std::vector<const bf16 *> &hidden, const std::vector<const int32_t *> &ids, int seg_rows,
                             int64_t bound, bool kv_only) {
  namespace sx = sycl::ext::oneapi::experimental;
  Mtp &M = *mtp_;
  const auto &st0 = M.cards[0].st;
  const Mtp::GraphKey key{pass, rows, seg_rows, hidden[0], ids[0], pair_parity_, moe_parity_, bound, st0.cur, st0.lcur, st0.lora_ld, kv_only ? 1 : 0};
  {
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<sx::command_graph<sx::graph_state::modifiable>> g;
    for (int c = 0; c < nc_; ++c) g.emplace_back(fab_.compute(c).get_context(), fab_.compute(c).get_device());
    const int pp = pair_parity_, mp = moe_parity_;
    std::vector<Mtp::Hc> start;
    for (const auto &X : M.cards) start.push_back({X.st.cur, X.st.lcur, X.st.lora_ld});
    for (int c = 0; c < nc_; ++c) g[static_cast<size_t>(c)].begin_recording(fab_.compute(c));
    try {
      mtp_submit(pass, rows, hidden, ids, static_cast<int>(bound), seg_rows, kv_only);
    } catch (...) {
      for (int c = 0; c < nc_; ++c) g[static_cast<size_t>(c)].end_recording();
      throw;
    }
    for (int c = 0; c < nc_; ++c) g[static_cast<size_t>(c)].end_recording();
    Mtp::Graph e;
    e.end_pair = pair_parity_;
    e.end_moe = moe_parity_;
    pair_parity_ = pp;
    moe_parity_ = mp;
    for (size_t c = 0; c < static_cast<size_t>(nc_); ++c) {
      auto &st = M.cards[c].st;
      e.end_hc.push_back({st.cur, st.lcur, st.lora_ld});
      st.cur = start[c].cur;
      st.lcur = start[c].lcur;
      st.lora_ld = start[c].lora_ld;
    }
    e.start_hc = start;
    e.hidden = hidden;
    e.ids = ids;
    try {
      for (int c = 0; c < nc_; ++c) e.exec.push_back(g[static_cast<size_t>(c)].finalize());
    } catch (const std::exception &x) {  // the host state is the start's again
      throw GraphRecordError(std::string("qwen38 model: drafter pass graph finalize: ") + x.what());
    }
    M.graphs.emplace(key, std::move(e));
    ++M.graphs_n;
    M.graphs_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  }
}

// Every pass graph recorded so far, at every other bound bucket, from its own start
// state (wire parities, the drafter's HC indices, input pointers); the host state is restored.
void Model::mtp_record_buckets(const std::vector<int64_t> &buckets) {
  Mtp &M = *mtp_;
  struct Base {
    int pass, rows, seg_rows, pp, mp, kv_only;
    std::vector<Mtp::Hc> hc;
    std::vector<const bf16 *> hidden;
    std::vector<const int32_t *> ids;
  };
  std::vector<Base> bases;
  {
    std::set<Mtp::GraphKey> seen;  // the key with the bound zeroed
    for (const auto &[key, e] : M.graphs) {
      auto k0 = key;
      std::get<7>(k0) = 0;
      if (!seen.insert(k0).second) continue;
      bases.push_back({std::get<0>(key), std::get<1>(key), std::get<2>(key), std::get<5>(key), std::get<6>(key), std::get<11>(key), e.start_hc, e.hidden, e.ids});
    }
  }
  const int pp0 = pair_parity_, mp0 = moe_parity_;
  std::vector<Mtp::Hc> hc0;
  for (const auto &X : M.cards) hc0.push_back({X.st.cur, X.st.lcur, X.st.lora_ld});
  auto set = [&](int pp, int mp, const std::vector<Mtp::Hc> &hc) {
    pair_parity_ = pp;
    moe_parity_ = mp;
    for (size_t c = 0; c < static_cast<size_t>(nc_); ++c) {
      auto &st = M.cards[c].st;
      st.cur = hc[c].cur;
      st.lcur = hc[c].lcur;
      st.lora_ld = hc[c].lora_ld;
    }
  };
  for (const auto &b : bases)
    for (const int64_t bound : buckets) {
      const Mtp::GraphKey key{b.pass, b.rows, b.seg_rows, b.hidden[0], b.ids[0], b.pp, b.mp, bound, b.hc[0].cur, b.hc[0].lcur, b.hc[0].lora_ld, b.kv_only};
      if (M.graphs.count(key)) continue;
      set(b.pp, b.mp, b.hc);
      mtp_record_graph(b.pass, b.rows, b.hidden, b.ids, b.seg_rows, bound, b.kv_only != 0);
    }
  set(pp0, mp0, hc0);
}

int Model::mtp_graphs_recorded() const { return mtp_ ? mtp_->graphs_n : dflash_ ? dflash_->graphs() : 0; }  // or DFlash's
double Model::mtp_graph_record_ms() const { return mtp_ ? mtp_->graphs_ms : 0; }

// The launches of one pass (eager, or inside a recording). bound: the QSA grid bound (>= pos + rows).
// kv_only: up to the drafter KV's writes (QSA projections + prep); no wire, no picks.
void Model::mtp_submit(int pass, int rows, const std::vector<const bf16 *> &hidden, const std::vector<const int32_t *> &ids, int bound, int seg_rows,
                       bool kv_only) {
  Mtp &M = *mtp_;
  const int H = cfg_.hidden, S = hs_.shape.streams, W = hs_.shape.width();
  if (S != k::kDenseRows) throw std::logic_error("qwen38 mtp: the front runs a row's streams as one 4-row GEMM");
  for (int c = 0; c < nc_; ++c) {
    auto &q = fab_.compute(c);
    Mtp::Card &X = M.cards[static_cast<size_t>(c)];
    Card &K = *cards_[static_cast<size_t>(c)];
    embed_rows(q, c, ids[static_cast<size_t>(c)], rows, 1, X.e);  // a prompt's next token may be an image row
    k::mtp_front_norm(q, H, S, rows, X.e, hidden[static_cast<size_t>(c)], X.norm_e, X.norm_h, hs_.shape.eps, X.en, X.hn);
    k::dense_gemm(q, rows, X.en, X.fe, X.fe_y);
    for (int r = 0; r < rows; ++r) k::dense_gemm_m4(q, X.hn + static_cast<size_t>(r) * W, X.fh, X.fh_y + static_cast<size_t>(r) * W);
    k::mtp_front_add(q, H, S, rows, X.fe_y, X.fh_y, X.st.res[X.st.cur ^ 1]);
    X.st.cur ^= 1;
    k::hc_mixer(q, hs_.shape, rows, X.st, nullptr, X.attn, head_->scratch(c), K.xa);
    if (kv_only) {
      k::qsa_store(q, mtp_shape(qs_, opt_.mtp_kv, c), rows, K.xa, X.qw, X.qc, K.rope,
                   k::QsaStep{X.seg + static_cast<size_t>(pass) * k::kMaxSegs * k::kSegRec, bound, k::kSegRec, seg_rows}, K.qsc);
      continue;
    }
    k::QsaStep qst{X.seg + static_cast<size_t>(pass) * k::kMaxSegs * k::kSegRec, bound, k::kSegRec, seg_rows};
    qst.every = bound <= kDecEveryBound;  // the first bound bucket selects every block
    k::qsa_decode(q, mtp_shape(qs_, opt_.mtp_kv, c), rows, K.xa, X.qw, X.qc, K.rope, qst, K.qsc, wire_of(K.pw), pair_parity_, K.ya);
  }
  if (kv_only) return;
  pair_parity_ ^= 1;
  for (int c = 0; c < nc_; ++c) {
    auto &q = fab_.compute(c);
    Mtp::Card &X = M.cards[static_cast<size_t>(c)];
    Card &K = *cards_[static_cast<size_t>(c)];
    k::hc_mixer(q, hs_.shape, rows, X.st, K.ya, X.mlp, head_->scratch(c), K.xm);
    k::MoeWeights w = X.mw;
    w.blobs = X.tab;
    k::MoeScratch sc = K.msc;
    sc.route = X.route;
    k::moe_decode(q, mtp_moe_shape(c), rows, K.xm, w, sc, wire_of(K.mw), moe_parity_, K.ym, opt_.moe_gu_ksplit, opt_.moe_dn_ksplit);
  }
  moe_parity_ ^= 1;
  for (int c = 0; c < nc_; ++c) {
    Mtp::Card &X = M.cards[static_cast<size_t>(c)];
    Card &K = *cards_[static_cast<size_t>(c)];
    k::hc_mixer(fab_.compute(c), hs_.shape, rows, X.st, K.ym, X.fin, head_->scratch(c), X.out);
  }
  mtp_lm_pick(rows, per_card([&](int c) -> const bf16 * { return M.cards[static_cast<size_t>(c)].out; }),
              per_card([&](int c) { return M.cards[static_cast<size_t>(c)].tok + pass * Mtp::kTokStride; }),
              per_card([&](int c) { return M.cards[static_cast<size_t>(c)].logits; }), pass, seg_rows, -1);
}

// Pass 1 over a prefill chunk: the drafter layer on every row at once, on the
// many-row kernels the target's chunk ran (front GEMMs as pf_dense, hc_mixer_rows, qsa_rows into the
// drafter's own KV, moe_rows on its experts), then its final mixer + lm_head + pick on the last slice
// only (the only pick the drafts after the prompt need). The target's combine of its last layer is
// materialised first (hc_combine over the chunk: the hidden pass 1 takes, as Head::residual is for a
// step). Buffers: the chunk's (the target's layers are done with them): the drafter's chunk state
// reuses the target's res / lora pair (the hidden is dead once the front has read it), e -> xa, en ->
// xm, hn -> hc.xn, fh_y -> hc.gate, fe_y -> hc.part. Numerics differ from the slice path by fp32
// summation order in the GEMMs only (front norm / add, mixers' elementwise steps per row).
void Model::mtp_prefill_rows(int m, int64_t pos0, std::vector<k::HcState> &cs, const std::vector<int32_t> &next, std::vector<k::HcState> *layer_only) {
  Mtp &M = *mtp_;
  const int H = cfg_.hidden, S = hs_.shape.streams, W = hs_.shape.width(), per = ms_.shape.per_card();
  if (!ms_.shape.tp) throw std::logic_error("qwen38 mtp: the many-row prefill pass needs expert TP (moe_rows)");
  if (m < 1 || static_cast<int>(next.size()) != m) throw std::invalid_argument("qwen38 mtp: prefill rows");
  (void)per;
  std::vector<k::HcState> ds(static_cast<size_t>(nc_));
  const int32_t *next_dev = vis_map(next.data(), m, vis_next_);  // image positions -> image rows
  for (int c = 0; c < nc_; ++c) {
    auto &q = fab_.compute(c);
    Card &K = *cards_[static_cast<size_t>(c)];
    Mtp::Card &X = M.cards[static_cast<size_t>(c)];
    k::HcState &st = cs[static_cast<size_t>(c)];
    k::hc_combine(q, hs_.shape, m, st, K.pf.ym);  // the target's hidden of every row -> st.residual()
    q.memcpy(K.pf.ids, next_dev, static_cast<size_t>(m) * 4);
    embed_rows(q, c, K.pf.ids, m, 1, K.pf.xa);
    k::mtp_front_norm(q, H, S, m, K.pf.xa, st.residual(), X.norm_e, X.norm_h, hs_.shape.eps, K.pf.xm, K.pf.hc.xn);
    k::pf_dense(q, m, X.fe, K.pf.xm, H, k::PfOut{K.pf.hc.part, nullptr, H}, K.pf.a8.on());
    k::pf_dense(q, m * S, X.fh, K.pf.hc.xn, H, k::PfOut{K.pf.hc.gate, nullptr, H}, K.pf.a8.on());  // m x S rows x hidden = rows x width
    k::HcState &d = ds[static_cast<size_t>(c)];
    d = st;
    k::mtp_front_add(q, H, S, m, K.pf.hc.part, K.pf.hc.gate, d.res[d.cur ^ 1]);
    d.cur ^= 1;
    k::hc_mixer_rows(q, hs_.shape, m, d, nullptr, X.attn, K.pf.hc, K.pf.xa);
    k::qsa_rows(q, mtp_shape(qs_, opt_.mtp_kv, c), m, K.pf.xa, X.qw, X.qc, K.rope, K.pf.words, k::kSegRec, pos0, K.qsc, K.pf.qsa, wire_of(K.pf.pw), pf_pair_parity_,
                K.pf.ya);
  }
  pf_pair_parity_ ^= 1;
  // calib_mtp's capture points (calib_layer_ == layers marks the drafter's layer)
  const bool cal = (calib_cap_ || calib_hook_) && calib_layer_ == cfg_.layers;
  // The rows before the last slice need only their KV (written by qsa_rows above); the MLP mixer + MoE run on the last
  // slice's rows alone unless every row is read (calibration, the picks trace, layer_only).
  const int r0 = !cal && !mtp_pf_picks_ && !layer_only ? 4 * ((m + 3) / 4 - 1) : 0, mn = m - r0;
  for (int c = 0; c < nc_; ++c) {
    Card &K = *cards_[static_cast<size_t>(c)];
    Mtp::Card &X = M.cards[static_cast<size_t>(c)];
    k::HcState &dd = ds[static_cast<size_t>(c)];
    k::hc_mixer_rows_part(fab_.compute(c), hs_.shape, r0, m, dd, K.pf.ya, X.mlp, K.pf.hc, K.pf.xm);  // rows [r0, m) (the API)
    k::hc_mixer_rows_flip(dd, X.mlp);
  }
  pf_tr("MTP.mlp_mix.res", ds[0].residual(), W);
  pf_tr("MTP.xm", cards_[0]->pf.xm, H);
  if (cal && calib_cap_) {
    for (int c = 0; c < nc_; ++c) fab_.compute(c).wait();
    calib_capture_mix(m, ds[0], cards_[0]->pf.xm);
  }
  for (int c = 0; c < nc_; ++c) {
    Card &K = *cards_[static_cast<size_t>(c)];
    Mtp::Card &X = M.cards[static_cast<size_t>(c)];
    k::MoeWeights w = X.mw;
    w.blobs = X.tab;
    k::moe_rows(fab_.compute(c), mtp_moe_shape(c), mn, K.pf.xm + static_cast<size_t>(r0) * H, w, K.pf.moe, wire_of(K.pf.mw), pf_moe_parity_,
                K.pf.ym + static_cast<size_t>(r0) * H, opt_.prefill_moe_a8);
  }
  pf_moe_parity_ ^= 1;
  if (cal) {
    for (int c = 0; c < nc_; ++c) fab_.compute(c).wait();
    calib_capture_moe(cfg_.layers, m);
  }
  pf_tr("MTP.moe.ym", cards_[0]->pf.ym, H);
  if (layer_only) {
    *layer_only = ds;
    return;
  }
  // The final mixer + pick on a slice (the last; every slice with the picks trace): its rows copied
  // into the drafter's 4-row state under the chunk's indices, as the head does for the target.
  const int slices = (m + 3) / 4;
  mtp_noise_upload();
  auto final_slice = [&](int s) {
    const int rows = std::min(4, m - 4 * s);
    for (int c = 0; c < nc_; ++c) {
      auto &q = fab_.compute(c);
      Card &K = *cards_[static_cast<size_t>(c)];
      Mtp::Card &X = M.cards[static_cast<size_t>(c)];
      const k::HcState &d = ds[static_cast<size_t>(c)];
      X.st.cur = d.cur;
      X.st.lcur = d.lcur;
      X.st.lora_ld = d.lora_ld;
      q.memcpy(X.st.res[X.st.cur], d.res[d.cur] + static_cast<size_t>(s) * 4 * W, static_cast<size_t>(k::kDecodeRows) * W * 2);
      q.memcpy(X.st.lora[X.st.lcur], d.lora[d.lcur] + static_cast<size_t>(s) * 4 * d.lora_ld, static_cast<size_t>(k::kDecodeRows) * d.lora_ld * 4);
      k::hc_mixer(q, hs_.shape, rows, X.st, K.pf.ym + static_cast<size_t>(s) * 4 * H, X.fin, head_->scratch(c), X.out);
    }
    mtp_lm_pick(rows, per_card([&](int c) -> const bf16 * { return M.cards[static_cast<size_t>(c)].out; }),
                per_card([&](int c) { return M.cards[static_cast<size_t>(c)].tok; }), per_card([&](int c) { return M.cards[static_cast<size_t>(c)].logits; }), 0, rows,
                slot_);
    return rows;
  };
  if (mtp_pf_picks_)
    for (int s = 0; s + 1 < slices; ++s) {
      const int rows = final_slice(s);
      const auto d = draft_picks(rows);
      mtp_pf_picks_->insert(mtp_pf_picks_->end(), d.begin(), d.end());
    }
  const int rows = final_slice(slices - 1);
  // The state draft_after_prefill / draft_chain continue from: the last slice as one pass-1 segment.
  for (int r = 0; r < rows; ++r) M.ids_host[r] = next[static_cast<size_t>(4 * (slices - 1) + r)];
  for (int c = 0; c < nc_; ++c) fab_.compute(c).memcpy(M.cards[static_cast<size_t>(c)].ids, M.ids_host, static_cast<size_t>(rows) * 4);
  M.p1_segs = {{slot_, pos0 + 4 * (slices - 1)}};
  M.p1_seg_rows = rows;
  M.last_rows = rows;
  if (mtp_pf_picks_) {
    const auto d = draft_picks(rows);
    mtp_pf_picks_->insert(mtp_pf_picks_->end(), d.begin(), d.end());
  }
}

void Model::draft_pass1(const int32_t *next, int rows, bool kv_only) {
  if (!mtp_) throw std::logic_error("qwen38 model: no drafter loaded (ModelOptions::mtp_dir)");
  // the verify's layout (batch_code_: rows per segment, or unequal segments as a seg_het word)
  const int verify_rows = batch_code_ > 0 ? static_cast<int>(run_slots_.size()) * batch_code_ : k::seg_het_rows(batch_code_);
  if (run_slots_.empty() || rows < 1 || rows != verify_rows) throw std::invalid_argument("qwen38 model: draft_pass1 runs over the last verify / step's rows");
  Mtp &M = *mtp_;
  std::vector<std::pair<int, int64_t>> segs;
  for (int s : run_slots_) segs.push_back({s, seqs_[static_cast<size_t>(s)].run_pos});
  std::memcpy(M.ids_host, next, static_cast<size_t>(rows) * 4);
  for (int c = 0; c < nc_; ++c) fab_.compute(c).memcpy(M.cards[static_cast<size_t>(c)].ids, M.ids_host, static_cast<size_t>(rows) * 4);
  mtp_pass(0, rows, per_card([&](int c) -> const bf16 * { return head_->residual(c); }),
           per_card([&](int c) -> const int32_t * { return M.cards[static_cast<size_t>(c)].ids; }), segs, batch_code_, kv_only);
  // a KV-only pass leaves no picks or chain state: nothing may draft from it
  M.last_rows = kv_only ? 0 : rows;
  M.p1_segs = std::move(segs);
  M.p1_seg_rows = batch_code_;
}

std::vector<int32_t> Model::draft_chain(int row, int depth) {
  if (!mtp_) throw std::logic_error("qwen38 model: no drafter loaded (ModelOptions::mtp_dir)");
  Mtp &M = *mtp_;
  if (row < 0 || row >= M.last_rows || depth < 1 || depth > Mtp::kPasses || M.p1_seg_rows == 0) throw std::invalid_argument("qwen38 model: draft_chain row / depth");
  const int W = hs_.shape.width();
  const auto [slot, pos1] = M.p1_row(row);
  int done = 1;
  for (int kk = 2; kk <= depth; ++kk) {
    const int64_t pos = pos1 + kk - 1;
    if (pos >= cap_) break;
    const size_t at = kk == 2 ? static_cast<size_t>(row) : 0;
    std::vector<const bf16 *> hid;
    std::vector<const int32_t *> ids;
    for (int c = 0; c < nc_; ++c) {
      const Mtp::Card &X = M.cards[static_cast<size_t>(c)];
      hid.push_back(X.st.res[X.st.cur] + at * W);
      ids.push_back(X.tok + (kk - 2) * Mtp::kTokStride + at);
    }
    mtp_pass(kk - 1, 1, hid, ids, {{slot, pos}}, 1);
    done = kk;
  }
  int32_t t[Mtp::kPasses * Mtp::kTokStride];
  fab_.compute(static_cast<size_t>(nc_ - 1)).wait_and_throw();
  fab_.compute(0).memcpy(t, M.cards[0].tok, sizeof t).wait();
  std::vector<int32_t> d;
  d.push_back(t[row]);
  for (int kk = 2; kk <= done; ++kk) d.push_back(t[(kk - 1) * Mtp::kTokStride]);
  return d;
}

// The chains of several segments at once. Pass k (k >= 2) runs one row per
// segment still in range (rows.hpp segments of 1 row): its inputs are the previous pass's state row
// and pick of each such segment, gathered into gh / gid first (a segment's row in pass 1 is any row of
// the verify; in later passes the previous pass's row i). A 2-row pass gives each row the bits of a
// 1-row pass (row-independent kernels; QSA per segment), so the drafts equal draft_chain's.
std::vector<std::vector<int32_t>> Model::draft_chains(const std::vector<int> &rows, int depth) {
  if (!mtp_) throw std::logic_error("qwen38 model: no drafter loaded (ModelOptions::mtp_dir)");
  if (rows.size() == 1) return {draft_chain(rows[0], depth)};
  Mtp &M = *mtp_;
  const int n = static_cast<int>(rows.size());
  if (n < 1 || n > Head::kRows || depth < 1 || depth > Mtp::kPasses || M.p1_seg_rows == 0) throw std::invalid_argument("qwen38 model: draft_chains rows / depth");
  for (int r : rows)
    if (r < 0 || r >= M.last_rows) throw std::invalid_argument("qwen38 model: draft_chains row");
  const int W = hs_.shape.width();
  // per chain: its slot, the position of its pass-1 row; src = its row in the previous pass's state / picks
  struct Chain {
    int slot;
    int64_t pos1;
    int src;
    int done;
  };
  std::vector<Chain> ch;
  for (int r : rows) {
    const auto [slot, pos1] = M.p1_row(r);
    ch.push_back({slot, pos1, r, 1});
  }
  for (int kk = 2; kk <= depth; ++kk) {
    std::vector<int> live;
    for (int i = 0; i < n; ++i)
      if (ch[static_cast<size_t>(i)].done == kk - 1 && ch[static_cast<size_t>(i)].pos1 + kk - 1 < cap_) live.push_back(i);
    if (live.empty()) break;
    std::vector<std::pair<int, int64_t>> segs;
    for (int c = 0; c < nc_; ++c) {
      auto &q = fab_.compute(c);
      const Mtp::Card &X = M.cards[static_cast<size_t>(c)];
      for (size_t j = 0; j < live.size(); ++j) {
        const Chain &h = ch[static_cast<size_t>(live[j])];
        q.memcpy(X.gh + j * W, X.st.res[X.st.cur] + static_cast<size_t>(h.src) * W, static_cast<size_t>(W) * 2);
        q.memcpy(X.gid + j, X.tok + (kk - 2) * Mtp::kTokStride + h.src, 4);
      }
    }
    for (int i : live) segs.push_back({ch[static_cast<size_t>(i)].slot, ch[static_cast<size_t>(i)].pos1 + kk - 1});
    mtp_pass(kk - 1, static_cast<int>(live.size()), per_card([&](int c) -> const bf16 * { return M.cards[static_cast<size_t>(c)].gh; }),
             per_card([&](int c) -> const int32_t * { return M.cards[static_cast<size_t>(c)].gid; }), segs, 1);
    for (size_t j = 0; j < live.size(); ++j) {
      Chain &h = ch[static_cast<size_t>(live[j])];
      h.src = static_cast<int>(j);
      h.done = kk;
    }
  }
  int32_t t[Mtp::kPasses * Mtp::kTokStride];
  fab_.compute(static_cast<size_t>(nc_ - 1)).wait_and_throw();
  fab_.compute(0).memcpy(t, M.cards[0].tok, sizeof t).wait();
  // each pass's picks sit at the chain's row of that pass: pass 1 at its verify row, pass k at its
  // index among the chains pass k ran
  std::vector<std::vector<int32_t>> out(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) out[static_cast<size_t>(i)].push_back(t[rows[static_cast<size_t>(i)]]);
  for (int kk = 2; kk <= depth; ++kk) {
    int j = 0;
    for (int i = 0; i < n; ++i)
      if (ch[static_cast<size_t>(i)].done >= kk && static_cast<int>(out[static_cast<size_t>(i)].size()) == kk - 1)
        out[static_cast<size_t>(i)].push_back(t[(kk - 1) * Mtp::kTokStride + j++]);
  }
  return out;
}

std::vector<int32_t> Model::draft_picks(int rows) {
  if (!mtp_) throw std::logic_error("qwen38 model: no drafter loaded");
  std::vector<int32_t> t(static_cast<size_t>(rows));
  fab_.compute(static_cast<size_t>(nc_ - 1)).wait_and_throw();
  fab_.compute(0).memcpy(t.data(), mtp_->cards[0].tok, t.size() * 4).wait();
  return t;
}

void Model::draft_logits(int rows, std::vector<int32_t> &ids, std::vector<float> &vals) {
  if (!mtp_) throw std::logic_error("qwen38 model: no drafter loaded");
  if (rows < 1 || rows > Head::kRows) throw std::invalid_argument("qwen38 model: draft_logits rows 1..kRows");
  Mtp &M = *mtp_;
  const bool sub = M.cards[0].vhead.w || M.cards[0].vhead4.w;
  const int n = sub ? M.sub_n : head_->vocab_per_card();  // columns per card (the pick's row stride)
  ids.resize(static_cast<size_t>(nc_) * n);
  for (int v = 0; v < nc_ * n; ++v) ids[static_cast<size_t>(v)] = !sub ? v : v < M.sub_k ? v : v - M.sub_k + M.sub_s0;
  vals.assign(static_cast<size_t>(rows) * nc_ * n, 0.f);
  std::vector<float> part(static_cast<size_t>(rows) * n);
  for (int c = 0; c < nc_; ++c) {
    fab_.compute(c).memcpy(part.data(), M.cards[static_cast<size_t>(c)].logits, part.size() * 4).wait();
    for (int r = 0; r < rows; ++r)
      std::copy(part.begin() + static_cast<std::ptrdiff_t>(r) * n, part.begin() + static_cast<std::ptrdiff_t>(r + 1) * n,
                vals.begin() + static_cast<std::ptrdiff_t>(r) * nc_ * n + static_cast<std::ptrdiff_t>(c) * n);
  }
}

// The drafter's experts from the calibration (TP half blobs, the target's TP stride).
void Model::calib_set_mtp_blobs(int card, const uint8_t *host, size_t bytes) {
  if (!mtp_ || !ms_.shape.tp) throw std::logic_error("qwen38 model: calib_set_mtp_blobs needs a drafter with expert TP");
  size_t st = ExpertLayout::of(cfg_).stride;
  k::moe_tp_layout(cfg_.hidden, cfg_.moe_inter, 2, &st);
  if (card < 0 || card > 1 || bytes > st * static_cast<size_t>(ms_.shape.per_card())) throw std::invalid_argument("qwen38 model: calib_set_mtp_blobs");
  fab_.compute(card).memcpy(mtp_->cards[static_cast<size_t>(card)].experts, host, bytes).wait();
}

}  // namespace xenithium::arch::qwen38
