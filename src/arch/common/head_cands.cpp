#include "xenithium/arch/common/head_cands.hpp"

#include <stdexcept>

#include "xenithium/kernels/head/head.hpp"
#include "xenithium/kernels/pack.hpp"

namespace xenithium::arch {

namespace k = kernels;

HeadCands::HeadCands(Fabric &fab, int vocab_per_card, int max_rows) : fab_(fab), vpc_(vocab_per_card), rows_(max_rows) {
  // tiles per row and card: kHeadTilesMax per part (a row past 8,192 tiles - TP1's whole vocabulary - selects per part)
  const size_t C = fab_.size(), T = static_cast<size_t>(k::kHeadTilesMax * k::head_select_parts(vpc_)), R = static_cast<size_t>(rows_);
  for (size_t c = 0; c < C; ++c) {
    scratch_.push_back(fab_.alloc(c, k::head_select_scratch_bytes(vpc_)));
    tiles_.push_back(static_cast<int32_t *>(fab_.alloc(c, R * T * 4)));
    vals_.push_back(static_cast<float *>(fab_.alloc(c, R * T * 16 * 4)));
  }
  host_tiles_ = static_cast<int32_t *>(fab_.alloc_host(C * R * T * 4));
  host_vals_ = static_cast<float *>(fab_.alloc_host(C * R * T * 16 * 4));
}

HeadCands::~HeadCands() {
  for (void *p : scratch_) fab_.release(p);
  for (void *p : tiles_) fab_.release(p);
  for (void *p : vals_) fab_.release(p);
  if (host_tiles_) fab_.release(host_tiles_);
  if (host_vals_) fab_.release(host_vals_);
}

int HeadCands::max_tiles() { return k::kHeadTilesMax; }

int HeadCands::run(const std::vector<const float *> &logits, int rows, int T, std::vector<int32_t> &ids, std::vector<float> &vals) {
  const size_t C = fab_.size();
  if (logits.size() != C || rows < 1 || rows > rows_ || T < 1 || T > k::kHeadTilesMax) throw std::invalid_argument("HeadCands::run: shape");
  const int PT = k::head_select_parts(vpc_) * T;  // tiles per row and card
  const size_t TM = static_cast<size_t>(k::kHeadTilesMax * k::head_select_parts(vpc_)), R = static_cast<size_t>(rows_);
  std::vector<sycl::event> ev(C);
  for (size_t c = 0; c < C; ++c) {
    sycl::queue &q = fab_.compute(c);
    k::head_cand_tiles(q, logits[c], vpc_, rows, vpc_, T, scratch_[c], tiles_[c], vals_[c]);
    q.memcpy(host_tiles_ + c * R * TM, tiles_[c], static_cast<size_t>(rows) * PT * 4);
    ev[c] = q.memcpy(host_vals_ + c * R * TM * 16, vals_[c], static_cast<size_t>(rows) * PT * 16 * 4);
  }
  const int n = static_cast<int>(C) * PT * 16;
  ids.resize(static_cast<size_t>(rows) * n);
  vals.resize(ids.size());
  for (size_t c = 0; c < C; ++c) {
    ev[c].wait();
    const int32_t *ht = host_tiles_ + c * R * TM;
    const float *hv = host_vals_ + c * R * TM * 16;
    for (int r = 0; r < rows; ++r) {
      const size_t o = static_cast<size_t>(r) * n + c * static_cast<size_t>(PT) * 16;
      k::bf16_round_floats(hv + static_cast<size_t>(r) * PT * 16, vals.data() + o, static_cast<size_t>(PT) * 16);
      for (int i = 0; i < PT; ++i) {
        const int32_t base = static_cast<int32_t>(c) * vpc_ + ht[static_cast<size_t>(r) * PT + i] * 16;
        for (int j = 0; j < 16; ++j) ids[o + static_cast<size_t>(i) * 16 + j] = base + j;
      }
    }
  }
  return n;
}

}  // namespace xenithium::arch
