// Xenithium · qwen35 MTP drafter on the cards: the checkpoint's one `mtp` layer.
//
// Semantics (HF qwen3_5 MTP, from config.json and the checkpoint's mtp.* tensors; written to the target's own kernels):
// a pass at position p takes the target's final-normed hidden h_p (the lm_head input whose pick is token p + 1) and the
// token t_{p+1} after p:
//   e   = rms_norm_(1+w)(embed[t_{p+1}], pre_fc_norm_embedding)      bf16, the target's (shared) embedding
//   hn  = rms_norm_(1+w)(h_p, pre_fc_norm_hidden)                    bf16
//   x   = bf16(fc . [e | hn])                                        fc [hidden][2 hidden]: card 0 holds the embedding
//                                                                    half, card 1 the hidden half; fp32 partials summed
//                                                                    by the pair all-reduce, one rounding (TP1: the
//                                                                    whole fc on the card over [e | hn] side by side)
//   one full-attention decoder layer at position p, the target's semantics: res = x; res += gated GQA(norm(res)) over the
//   drafter's own KV cache; res += SwiGLU MLP(norm(res)) (the residual adds bf16, the norms (1 + w))
//   h'  = rms_norm_(1+w)(res, mtp.norm)  -> the target's lm_head + two-card greedy pick = the draft of token p + 2
// A chained pass takes the previous pass's h' as its hidden and its pick as its token, one position further.
//
// Weights: the drafter's bf16 matrices stored int8 per row (dense_bytes 1, the default: qwen38's drafter choice) or bf16
// (2); decode passes on dense_gemm / gqa_decode's dense projections, the prompt's KV pass on pf_dense. KV cache: the
// target's page layout, bf16 or int8.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "xenithium/arch/qwen35/arch.hpp"
#include "xenithium/arch/qwen35/head.hpp"
#include "xenithium/kernels/gemm/dense.hpp"
#include "xenithium/kernels/attention/gqa.hpp"
#include "xenithium/kernels/attention/kv_copy.hpp"
#include "xenithium/kernels/comm/wire.hpp"
#include "xenithium/load/dense_store.hpp"
#include "xenithium/rt/fabric.hpp"

namespace xenithium::arch::qwen35 {

struct MtpOptions {
  int dense_bytes = 1;                       // drafter matrices: 1 = int8 per row, 2 = bf16
  kernels::QsaKv kv = kernels::QsaKv::i8;  // drafter KV (int8 accepts as bf16 at half the VRAM)
  int threads = 16;
  // The draft vocabulary: 0 = the target's whole lm_head; K = token ids [0, K) (byte-level BPE ids follow the
  // merge order, so low ids are the frequent tokens) + the 128 ids from the first eos id rounded down to 64 (the special
  // tokens), as the drafter's own lm_head rows (a draft outside it is simply never proposed: verification is the
  // target's full head). K a multiple of 32.
  int vocab = 0;
  bool head_w4 = false;  // the draft vocabulary's rows from the target's int4 lm_head.qweight
  int slots = 1;  // sequence slots, each with its own KV (pages of the target's table, attach_kv)
  int rows = 0;  // rows a pass holds (0 = kRows; the model's step rows, 16 unless 32-row steps)
  // KV-only passes deferred, up to this many rows per slot (0 = every KV-only pass runs; else 8..): defer() / settle()
  // below.
  int kv_defer = 0;
  // TP1 (one card): the most prompt rows a prefill_kv call takes (the fc's [rows][2 hidden] input rows; 0 = a pass's)
  int prefill_rows = 0;
};

class Mtp {
 public:
  using bf16 = sycl::ext::oneapi::bfloat16;
  static constexpr int kRows = kernels::kStepRows32;

  // capacity: tokens of drafter KV (the target's); rope: per card the target's table (>= capacity rows).
  Mtp(Fabric &fab, const std::string &dir, const Config &cfg, int64_t capacity, const std::vector<const bf16 *> &rope, const MtpOptions &opt);
  Mtp(const Mtp &) = delete;
  Mtp &operator=(const Mtp &) = delete;
  ~Mtp();

  // One decode pass of `rows` rows (1..8) in segments of seg_rows rows (rows.hpp: rec[c] the per-card device segment
  // records, kSegRec ints each: position of the segment's first row, slot), every position < pos_bound: h[c] bf16
  // [rows][hidden] on card c (rows past `rows` finite up to the row shape), ids[c] the rows' tokens (device or pinned
  // host). Writes the drafter KV there, the pass's hidden h' (hidden(c, which)) and the drafts (tokens(c, which)). which
  // (0 / 1) picks the output buffers: a chained pass reads the other one. kv_only: the KV and nothing after
  // the attention (no MLP, final norm or head: a pass whose drafts nobody reads). Submits only.
  // draft: the pass's draft index for coupled drafts (set_noise): a row's slot (its record's kSegSlot) with a
  // noise record picks argmax of logit / T + util::gumbel(key, pos + draft, token) (kernels::draft_gumbel).
  void pass(Head &head, int rows, const std::vector<const bf16 *> &h, const std::vector<const int32_t *> &ids, const std::vector<const int32_t *> &rec,
            int seg_rows, int pos_bound, int which, bool kv_only = false, int draft = 0);
  // Coupled drafts for sampled requests. set_noise keeps the slot's record (key, the output position of its first draft,
  // 1 / T; inv_t 0 = argmax, the same bits as a greedy draft); noise_upload copies the table to the cards before a call's
  // passes (no launch at all while no slot samples).
  void set_noise(int slot, uint64_t key, uint64_t pos, float inv_t);
  void noise_upload();
  // Rows src[i] of pass `which`'s hidden and drafts gathered into rows i of chain_hidden / chain_tokens (the first
  // chained pass of several segments reads them). Submits only.
  void gather_chain(int which, const std::vector<int> &src);
  const bf16 *chain_hidden(int c) const { return cards_[static_cast<size_t>(c)].gh; }
  const int32_t *chain_tokens(int c) const { return cards_[static_cast<size_t>(c)].gid; }
  // The drafter KV of m prompt rows at pos0 .. pos0 + m - 1 (KV only: fc, input norm, q|k|v and prep; no attention,
  // MLP or head): h[c] bf16 [m][hidden] the target's final-normed hidden, ids host [m] the token after each row, pos4[c]
  // the target's per-4-row position words (word j = pos0 + 4 j). The caller lends its chunk buffers (bf16 [m][hidden]
  // a / b, fp32 part [m][hidden], the many-row attention scratch, the rows wire and its parity). Submits only.
  struct Lend {
    bf16 *a = nullptr, *b = nullptr;
    float *part = nullptr;
    kernels::GqaRowsScratch sc;
    const kernels::PairRowsWire *wire = nullptr;
  };
  // rope_rows (M-RoPE): per card the target's rope rows of the call (GqaShape::rope_rows), null = by position.
  void prefill_kv(Head &head, int m, const std::vector<const bf16 *> &h, const int32_t *ids, const std::vector<const int32_t *> &pos4, int pos0,
                  const std::vector<Lend> &lend, int parity, int slot = 0, const std::vector<const bf16 *> *rope_rows = nullptr);
  // qwen35 image input: the pass and flush preps add the rows' M-RoPE delta (records' kSegRope, defer's rope).
  void set_mrope(bool on) {
    for (auto &s : shape_) s.mrope = on;
  }

  // MtpOptions::kv_defer: the rows of a KV-only pass wait in their slot's backlog instead (ns segments
  // of seg_rows rows: segment i at positions p0[i] .. of slot[i]; h1 = card 1's hidden rows [ns x seg_rows][hidden] (card
  // 0 reads only the tokens), ids pinned host [ns x seg_rows] the token after each row). settle(slot, pos) writes the
  // slot's deferred rows below pos as one many-row KV pass (prefill_kv's kernels on its own scratch and wire) and drops the
  // rest; drop(slot) forgets them. Rows of a later call that start inside the backlog replace the rows from there on (the
  // positions its last verify rejected), as the passes would have in order. Callers settle before anything reads the
  // slot's drafter KV below pos (a drafting pass, a prefill of the slot, a page copy). Submit only.
  bool defers() const { return opt_.kv_defer > 0; }
  void defer(Head &head, int ns, int seg_rows, const int *slot, const int64_t *p0, const bf16 *h1, const int32_t *ids, const int32_t *rope = nullptr);
  void settle(Head &head, int slot, int64_t pos);
  void drop(int slot);
  int deferred_rows(int slot) const { return pend_.empty() ? 0 : pend_[static_cast<size_t>(slot)].n; }
  int flushes() const { return flushes_; }
  const bf16 *hidden(int c, int which) const { return cards_[static_cast<size_t>(c)].hout[which]; }
  const int32_t *tokens(int c, int which) const { return cards_[static_cast<size_t>(c)].tok[which]; }
  std::vector<int32_t> read_tokens(int which, int rows);  // card 0's drafts (waits)
  std::vector<uint16_t> read_hidden(int which, int rows);  // card 0's output hidden (after mtp.norm; waits; checks)
  int32_t *pinned_ids() { return ids_host_; }               // kRows + prefill rows of pinned host ids for the callers
  size_t card_bytes(int c) const;                           // drafter weights + KV + scratch on card c
  // The drafter KV lives on the target's pages (the shared pool or the fixed per-slot layout): pool_pages
  // physical pages of K / V (+ scales) per card, read through the target's device table [slots][slot_pages] per card
  // (slot_pages x 64 = the capacity). Before the first pass; again when the target re-sizes its pool (the old KV is
  // freed). kv_page_bytes: one page's bytes on card c.
  void attach_kv(int pool_pages, const std::vector<const int32_t *> &table, int slot_pages);
  size_t kv_page_bytes(int c) const;
  // Prefix cache: the drafter's paged KV buffers on card c (K, V and their scales), for whole-page copies.
  std::vector<kernels::KvPageBuf> kv_page_bufs(int c) const;
  int64_t capacity() const { return cap_; }
  uint32_t wire_error();
  const kernels::GqaShape &shape(int c) const { return shape_[static_cast<size_t>(c)]; }

 private:
  struct Card {
    const bf16 *norm_e = nullptr, *norm_h = nullptr, *norm_out = nullptr, *in_ln = nullptr, *post_ln = nullptr;
    kernels::DenseWeights fc, gate_up, down;
    kernels::GqaWeights attn;
    kernels::GqaCache kv;
    kernels::GqaScratch asc;
    const bf16 *rope = nullptr;
    bf16 *e = nullptr, *xin = nullptr, *res = nullptr, *xn = nullptr, *mix = nullptr, *mlp = nullptr;
    bf16 *hout[2] = {nullptr, nullptr};
    bf16 *gh = nullptr;      // gathered chain inputs [kSegs16][hidden]
    int32_t *gid = nullptr;  // [kSegs16]
    float *part = nullptr, *gu = nullptr, *logits[2] = {nullptr, nullptr};
    bf16 *h = nullptr;
    int32_t *tok[2] = {nullptr, nullptr};
    kernels::PairWire wire;
    // the deferred KV rows (hidden [slots][kv_defer][hidden] on card 1, tokens [slots][kv_defer] on card 0), the
    // flush's scratch, 4-row position words and many-row wire
    bf16 *bl_h = nullptr, *fl_a = nullptr, *fl_b = nullptr;
    int32_t *bl_ids = nullptr, *fl_pos4 = nullptr;
    float *fl_part = nullptr;
    kernels::GqaRowsScratch fl_sc;
    kernels::PairRowsWire fl_wire;
    bf16 *cat = nullptr;  // TP1: [norm(e) | norm(h)] rows [cat_rows_][2 hidden], the whole fc's input
    kernels::DraftNoiseRec *noise = nullptr;  // [slots]
    std::vector<void *> mem;
    std::vector<std::pair<void *, size_t>> kv_mem;  // the KV buffers (attach_kv may replace them)
  };
  Fabric &fab_;
  Config cfg_;
  MtpOptions opt_;
  int64_t cap_ = 0;
  int ipc_ = 0, parity_ = 0;
  std::vector<kernels::DraftNoiseRec> noise_host_;  // [slots]
  bool noise_any_ = false;                          // a slot samples: the passes run draft_gumbel
  std::unique_ptr<load::DenseStore> dense_;
  std::vector<Card> cards_;
  std::vector<kernels::GqaShape> shape_;
  int32_t *ids_host_ = nullptr;
  int rows_ = kRows;  // MtpOptions::rows
  int pinned_n_ = 0;  // ids_host_'s length: kRows + the largest prefill call (MtpOptions::prefill_rows)
  std::vector<size_t> bytes_;
  struct Pend {  // a slot's deferred rows: positions p0 .. p0 + n - 1
    int64_t p0 = 0;
    int n = 0;
    int rope = 0;  // the rows' M-RoPE delta
  };
  std::vector<Pend> pend_;
  int fl_par_ = 0, flushes_ = 0;
  void flush(Head &head, int slot, int m);
  int cat_rows_ = 0;
  void concat(sycl::queue &q, int rows, const bf16 *e, const bf16 *hn);
  // the draft vocabulary (MtpOptions::vocab): its lm_head rows per card, virtual index -> token id = v < sub_k_ ? v : v - sub_k_ + sub_s0_
  std::vector<LmMat> head_w_;
  int head_n_ = 0, sub_k_ = 0, sub_s0_ = 0;

 public:
  int draft_vocab() const { return head_w_.empty() ? 0 : sub_k_ + 128; }
};

}  // namespace xenithium::arch::qwen35
