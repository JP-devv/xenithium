// Xenithium · the sampling candidates of a step from the cards.
//
// A sampled request with top_k <= T and no logit modifier reads only the k best logits of each row (core::Sampler's
// candidate path). Instead of copying every row's 2 x 124,160 fp32 to the host, each card selects its T best 16-column
// tiles by the bf16-rounded tile maximum (kernels::head_cand_tiles) and copies those T x 16 raw values; the host
// rounds them like the full copy. The k best of the union are the row's k best (proof at head_cand_tiles), so the
// sampler's kept set and its draws are the same as from the full row.
// Arch-agnostic: any model whose head leaves fp32 logits [kRows][vpc] per card (vocabulary-parallel, card c holds
// global ids [c vpc, (c + 1) vpc)).
#pragma once

#include <cstdint>
#include <vector>

#include "xenithium/rt/fabric.hpp"

namespace xenithium::arch {

class HeadCands {
 public:
  // Allocates per-card scratch for up to max_rows rows (device buffers at construction: before any VRAM fit).
  HeadCands(Fabric &fab, int vocab_per_card, int max_rows);
  HeadCands(const HeadCands &) = delete;
  HeadCands &operator=(const HeadCands &) = delete;
  ~HeadCands();

  // The most tiles per card a call takes (kernels::kHeadTilesMax).
  static int max_tiles();
  // logits[c]: card c's device rows [rows][vpc] (stride vpc). Submits on each card's compute queue (after the step that
  // wrote them), waits, and fills ids / vals [rows][n] with n = cards x P x T x 16 per row (global ids, bf16-rounded values;
  // card 0's tiles first, ascending; P = kernels::head_select_parts(vpc): 1 under TP2, 2 for TP1's whole vocabulary). Returns n.
  int run(const std::vector<const float *> &logits, int rows, int T, std::vector<int32_t> &ids, std::vector<float> &vals);

 private:
  Fabric &fab_;
  int vpc_, rows_;
  std::vector<void *> scratch_;
  std::vector<int32_t *> tiles_;
  std::vector<float *> vals_;
  int32_t *host_tiles_ = nullptr;  // pinned [cards][rows][kHeadTilesMax]
  float *host_vals_ = nullptr;     // pinned [cards][rows][kHeadTilesMax][16]
};

}  // namespace xenithium::arch
