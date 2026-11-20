// Xenithium · qwen35 step ends on the cards: embedding in, final norm + lm_head + greedy out.
//
// Per card (TP over the vocabulary, everything else replicated):
//   embed:  res[row] = embed_tokens[id]            (table in pinned host, read over PCIe; arch flint: the 8-bit affine
//                                                   table on the card, kernels::embed_aff8)
//   norm:   xn = rms_norm(res, norm.weight)        ((1 + w) affine, rmsnorm.hpp)
//   head:   logits = xn . lm_head_slice^T (int8 W8A16 or bf16, dense_bytes), greedy pick across both cards
// Both cards compute the replicated parts on their own copy and end with identical residuals and
// the same tokens; the pick's two words per row are the only exchange. The same shape as the
// qwen38 head without the hyper-connection mixer (one residual stream, a plain norm).
#pragma once

#include <cstdint>
#include <functional>
#include <vector>

#include "xenithium/arch/common/lm_mat.hpp"
#include "xenithium/arch/qwen35/arch.hpp"
#include "xenithium/kernels/gemm/dense.hpp"
#include "xenithium/kernels/head/head.hpp"
#include "xenithium/kernels/rows.hpp"
#include "xenithium/kernels/gemm/w4a16_gemm.hpp"
#include "xenithium/load/dense_store.hpp"
#include "xenithium/rt/fabric.hpp"

namespace xenithium::arch::qwen35 {

// LmMat / lm_mat / lm_gemm: arch/common/lm_mat.hpp (shared with qwen38's DFlash drafter).

// The step's lm_head as loaded int8 rows, the int4 g128 rows, or the two-stage head:
// the int4 rows score the whole vocabulary, kernels::head_select_tiles keeps each row's top `tiles` 16-column tiles per
// card, and kernels::dense_gemm_tiles overwrites those tiles with the int8 rows' logits (the int8 head's bits there). The
// pick, the sampler's logits copy and the device logprobs then read the patched rows. int8 needs the int8 rows (the
// plain head or the second copy), int4 / rescore the int4 rows, rescore both. Run-time switchable (set_mode).
enum class HeadMode : uint8_t { int8, int4, rescore };
inline constexpr const char *kHeadI8Key = "lm_head.weight#i8";  // the store key of the int8 copy beside an int4 head
inline constexpr int kHeadTilesDefault = 32;  // the two-stage head's tiles per row and card (set_mode's default)

class Head {
 public:
  static constexpr int kRows = kernels::kStepRows32;  // rows a step can carry (the GEMMs run row_shape(rows))
  using bf16 = sycl::ext::oneapi::bfloat16;

  struct Card {
    bf16 *res = nullptr;      // [kRows][hidden] the residual stream
    bf16 *xn = nullptr;       // [kRows][hidden] the final norm's output
    float *logits = nullptr;  // [kRows][vocab / cards]
    int32_t *tokens = nullptr;
    void *scratch = nullptr;
    kernels::PickWire wire;
    void *sel_scratch = nullptr;  // head_select_tiles' keys (rescore)
    int32_t *tiles = nullptr;     // [kRows][kernels::kHeadTilesMax] the rows' selected tiles
  };
  struct Events {  // the last launch of each stage, per card (profiling); sel / rescore = lm when not rescoring
    sycl::event embed, norm, lm, selmax, sel, rescore, chunks, pick;
  };

  // int8_copy: the store also holds the int8 rows under kHeadI8Key beside an int4 lm_head.weight (rescore).
  // rows: the rows its buffers hold (<= kRows; the model's step / call rows, 16 unless 32-row steps)
  Head(Fabric &fab, const load::DenseStore &dense, const Config &cfg, bool int8_copy = false, int rows = kRows);
  int rows() const { return rows_; }
  Head(const Head &) = delete;
  Head &operator=(const Head &) = delete;
  ~Head();

  // ids: host array of `rows` token ids for the next embed, written into a pinned host buffer the
  // embed launches read; call it only after the earlier step's wait.
  void set_ids(const int32_t *ids, int rows);
  // Each stage submits on both cards and returns without waiting.
  void embed(int rows);
  // card c: the embedding rows of ids (device / pinned host int32 [rows]) into out bf16 [rows][hidden] (the bf16 table or
  // arch flint's 8-bit one); on card c's compute queue
  sycl::event gather(int c, const int32_t *ids, int rows, bf16 *out);
  void final_norm(int rows);
  // The same with the last block's output added into the residual first (add[c]: bf16 [rows][hidden] on card c;
  // the whole model).
  void final_norm(int rows, const std::vector<const bf16 *> &add);
  void lm_pick(int rows);
  // For the MTP drafter: lm_head + the two-card greedy pick of x[c] (bf16 [rows][hidden] on card c, rows past
  // `rows` finite up to the row shape) into the caller's logits[c] ([rows][vocab / cards]) / tokens[c]: the step's own
  // buffers stay untouched. The pick scratch and wire are shared (both calls are ordered on each card's queue).
  // before_pick (optional): submitted on card c's queue between the lm_head and the pick, on logits[c] (n = the
  // card's columns): the MTP drafter's coupled-draft noise.
  using PrePick = std::function<void(size_t c, sycl::queue &q, float *logits, int n)>;
  void lm_pick_to(int rows, const std::vector<const bf16 *> &x, const std::vector<float *> &logits, const std::vector<int32_t *> &tokens,
                  const PrePick &before_pick = {});
  void wait();
  std::vector<int32_t> tokens(int card, int rows);
  // The lm_head the step runs (throws when the rows it needs are not loaded); tiles 1..kHeadTilesMax.
  void set_mode(HeadMode m, int tiles = kHeadTilesDefault);
  HeadMode mode() const { return mode_; }
  int tiles() const { return tiles_; }
  bool can(HeadMode m) const;

  bf16 *residual(int c) const { return cards_[static_cast<size_t>(c)].res; }
  const Card &card(int c) const { return cards_[static_cast<size_t>(c)]; }
  const LmMat &lm_weights(int c) const { return lm_[static_cast<size_t>(c)]; }
  const bf16 *norm_weight(int c) const { return norm_[static_cast<size_t>(c)]; }
  const bf16 *embed_table(int c) const { return embed_[static_cast<size_t>(c)]; }
  // Image input: ids >= base read rows (id - base) of the card's pool (arch::VisionStore's image rows) in every
  // embedding gather (embed, gather, gather_on); empty pools = text only.
  void set_ext(const std::vector<const bf16 *> &pools, int32_t base) { ext_ = pools, ext_base_ = base; }
  // the embedding of `rows` ids (device or pinned host) on queue q from card c's table
  sycl::event gather_on(sycl::queue &q, int c, const int32_t *ids, int rows, bf16 *out) const;
  const Events &events(int c) const { return ev_[static_cast<size_t>(c)]; }
  int hidden() const { return hidden_; }
  int vocab_per_card() const { return vpc_; }
  uint32_t wire_error(int c);

 private:
  Fabric &fab_;
  int hidden_ = 0, vpc_ = 0, rows_ = kRows;
  float eps_ = 1e-6f;
  int plain_ = 0;  // the final norm (kernels::rms_norm plain): 1 = arch llama's w x norm(x), 2 = arch flint's (one rounding)
  std::vector<Card> cards_;
  int32_t *ids_host_ = nullptr;  // the next embed's token ids, pinned host [kPickMaxRows]
  std::vector<Events> ev_;
  std::vector<const bf16 *> embed_, norm_;
  std::vector<const bf16 *> ext_;  // the image rows per card (set_ext)
  int32_t ext_base_ = 0;
  // arch flint: the 8-bit affine embedding (kernels::embed_aff8) per card: codes, bf16 scales, biases
  bool embed8_ = false;
  int64_t vocab_ = 0;
  std::vector<const uint8_t *> e8_;
  std::vector<const uint16_t *> e8s_, e8b_;
  std::vector<LmMat> lm_;
  std::vector<kernels::DenseWeights> lm8_;  // the int8 rows beside an int4 lm_ (empty: none)
  HeadMode mode_ = HeadMode::int8;
  int tiles_ = kHeadTilesDefault;
  // lm_head (as set_mode says) of x [rows][hidden] on card c into logits [rows][vocab / cards]
  void lm_run(size_t c, int rows, const bf16 *x, float *logits);
};

}  // namespace xenithium::arch::qwen35
