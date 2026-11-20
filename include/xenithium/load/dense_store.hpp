// Xenithium · the non-expert weights on the cards, from an arch module's dense plan.
// Model-agnostic: the arch declares every tensor's split (replicate / rows by sections /
// cols / host) and storage (int8 per-row, bf16 tiles, raw bytes); this store slices,
// quantises, packs and uploads it. Pipeline as the expert store: the next layer's tensors are
// read (O_DIRECT, pooled buffers) while the current one is sliced / packed into pinned staging
// on threads and copied to each card (one VRAM block per (card, layer group)).
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include "xenithium/kernels/gemm/dense.hpp"
#include "xenithium/kernels/gemm/w4a16_gemm.hpp"
#include "xenithium/model/arch.hpp"
#include "xenithium/model/bulk_read.hpp"
#include "xenithium/rt/fabric.hpp"

namespace xenithium::load {

// A W4 matrix on the card as w4a16_gemm_m4 reads it: blocks [k/128][n/16][1024] + fp16 scales [k/128][n].
struct W4Weights {
  const uint8_t *w = nullptr;
  const sycl::half *scales = nullptr;  // fp16 [k/group][n]
  int n = 0, k = 0;  // n padded to 16
  // DenseSpec:W4Layout:ct_zp (group 32): the zero points uint8 [k/64][n] (kernels:pack_w4_zeros), registered with
  // kernels:w4_set_zero_points under w for the W4 GEMMs; null for the symmetric g128 layouts.
  const uint8_t *zeros = nullptr;
  // DenseSpec:W4Layout:mlx (group 64): `scales` then holds bf16 bits [k/64][n] and `biases` bf16 [k/64][n] (w = q x s + b),
  // registered with kernels:w4_set_affine under w.
  const uint16_t *biases = nullptr;
  int group = 128;
};

struct DeviceTensor {
  model::DenseSpec::Store store = model::DenseSpec::Store::raw;
  const uint8_t *ptr = nullptr;  // device (or pinned host for Split:host)
  const float *scale = nullptr;  // int8: per row of the slice (device)
  const uint16_t *scale16 = nullptr;  // w4: fp16 group scales [k/group][n_pad] (device)
  const uint8_t *zeros = nullptr;  // w4 ct_zp: zero points [k/64][n_pad] (device)
  const uint16_t *biases = nullptr;  // w4 affine: bf16 group biases [k/64][n_pad] (device); scale16 then holds bf16 scales
  int group = 128;  // w4: k per scale
  int64_t rows = 0, cols = 0;  // this card's slice as a 2-D matrix (w4: cols = k)
  size_t bytes = 0;  // payload incl. scales
  kernels::DenseWeights dense() const;  // for i8 / bf16 matrices
  W4Weights w4() const;  // for w4 matrices
};

// The rows of a [d0][...] tensor that card `card` of `cards` holds under a rows split with
// these sections (empty = one section): each section's contiguous 1/cards, in section order.
std::vector<std::pair<int64_t, int64_t>> slice_rows(int64_t d0, const std::vector<int64_t> &sections, int card, int cards);

struct DenseStoreStats {
  double total_s = 0, read_wait_s = 0, pack_s = 0, h2d_wait_s = 0;
  uint64_t read_bytes = 0, host_bytes = 0;
  std::vector<std::map<model::Component, uint64_t>> card_bytes;  // [card][component] VRAM payload
  std::vector<uint64_t> card_alloc;  // [card] VRAM allocated (aligned)
};

class DenseStore {
 public:
  explicit DenseStore(Fabric &fabric);
  DenseStore(const DenseStore &) = delete;
  DenseStore &operator=(const DenseStore &) = delete;
  ~DenseStore();

  void load(const model::Checkpoint &ck, const std::vector<model::DenseSpec> &plan, int threads = 16);
  // Before load: read up to `ahead` groups (layers) ahead of the packing (default 1), and call
  // `done` on the reader thread once the last group is read (so the next reader can take the drive).
  void set_reads(int ahead, std::function<void()> done = {}) {
    read_ahead_ = ahead;
    reads_done_ = std::move(done);
  }
  const DeviceTensor &at(int card, const std::string &name) const;
  // The largest int8 matrix on `card` whose name contains none of `skip`: its codes' bytes (rows x cols) and the most rows
  // (output columns) of any (the oneDNN prefill route's scratch).
  void max_i8(int card, size_t *bytes, int64_t *rows, const std::vector<std::string> &skip = {}) const;
  const DenseStoreStats &stats() const { return stats_; }

 private:
  int read_ahead_ = 1;
  std::function<void()> reads_done_;
  Fabric &fab_;
  int cards_ = 0;
  std::vector<std::map<std::string, DeviceTensor>> t_;  // [card] name -> tensor
  std::vector<void *> dev_blocks_, host_blocks_;
  std::vector<const uint8_t *> zp_registered_;  // w4 ct_zp matrices registered with kernels:w4_set_zero_points (cleared at destruction)
  std::vector<const uint8_t *> aff_registered_;  // w4 affine matrices registered with kernels:w4_set_affine
  DenseStoreStats stats_;
};

}  // namespace xenithium:load
