// Xenithium · image input on the cards, for any arch whose checkpoint carries a Qwen3-VL-family vision tower
// (model::VisionConfig, kernels/encoder/vit.hpp).
//
// VisionTower: the tower's weights (model.visual.*, bf16, the checkpoint's own layouts) spread over the fabric's cards:
// the patch embedding and the first blocks on card 0, the rest of the blocks and the merger on the last card (split so
// both cards hold about the same bytes). encode() runs one image: patches + tables up, the patch embedding (a linear
// over [C][T][P][P]), the position grid, the blocks (pre-norm attention + MLP), the merger (LayerNorm per patch, 2 x 2
// patches concatenated, fc1 + GELU(erf), fc2); the hidden state crosses cards once through pinned host memory (no
// peer copy), and the merged tokens land in every card's rows.
//
// VisionStore: the image embeddings a running request's prompt names. An image registered by the server (any thread)
// gets `tokens` consecutive rows of a per-card pool and a range of virtual token ids [base, base + tokens): base >=
// the vocabulary, issued once and never reused (a content hash finds an image already held, so the same picture sent
// again gets the same ids and the prefix cache can continue a conversation over it; an evicted image registered again
// gets new ids, so a cached prefix can never meet another image's rows). The engine thread calls prepare() over a
// prefill chunk's tokens: images not yet encoded are encoded then, and the chunk's ids are mapped for the device to
// vocab + pool row (the embedding gather reads rows >= vocab from the pool). A Lease keeps an image's rows while its
// request lives; rows of images without a lease are evicted least recently used first when a new image needs room.
#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "xenithium/model/vision.hpp"
#include "xenithium/rt/fabric.hpp"

namespace xenithium::model {
class Checkpoint;
}

namespace xenithium::arch {

class VisionTower {
 public:
  // max_patches: the largest image's patches (scratch sizing; preprocess() never exceeds it at the server's cap).
  VisionTower(Fabric &fab, const model::Checkpoint &ck, const model::VisionConfig &vc, int max_patches);
  VisionTower(const VisionTower &) = delete;
  VisionTower &operator=(const VisionTower &) = delete;
  ~VisionTower();

  // Bytes the tower would hold on card c of `cards` (weights + scratch), without loading: the serving backend reserves
  // them before the hot experts fill the VRAM.
  static size_t bytes_on(const model::VisionConfig &vc, int max_patches, int cards, int c);
  size_t bytes(int c) const;

  // One image -> dst[c] (device rows [tokens][out_hidden] bf16 on card c, every card). Blocks until written.
  void encode(const model::VisionInput &in, const std::vector<void *> &dst);
  double last_ms() const { return last_ms_; }

 private:
  struct Block;
  struct Card;
  Fabric &fab_;
  model::VisionConfig vc_;
  int max_patches_ = 0;
  int split_ = 0;  // blocks [0, split_) on card 0, the rest on the last card
  std::vector<std::unique_ptr<Card>> cards_;
  std::vector<void *> owned_;  // device allocations (released in the destructor)
  void *host_ = nullptr;       // pinned staging (the hidden state crossing cards, the merged rows)
  double last_ms_ = 0;
};

class VisionStore {
 public:
  // rows: the pool's rows per card (the most image tokens held at once).
  VisionStore(Fabric &fab, std::unique_ptr<VisionTower> tower, const model::VisionConfig &vc, int64_t vocab, int64_t rows);
  VisionStore(const VisionStore &) = delete;
  VisionStore &operator=(const VisionStore &) = delete;
  ~VisionStore();

  static size_t pool_bytes(const model::VisionConfig &vc, int64_t rows) { return static_cast<size_t>(rows) * vc.out_hidden * 2 + 4096; }

  struct Lease {
    int64_t base = 0;  // the first virtual token id
    int tokens = 0;
    VisionStore *store = nullptr;
    uint64_t key = 0;
    ~Lease();
  };
  // Any thread. The image (preprocessed) under its content hash; nullptr + err when the pool cannot hold it now.
  std::shared_ptr<Lease> add(model::VisionInput &&in, std::string &err);

  // Engine thread: maps tokens[0, m) for the device (ids >= vocab -> vocab + pool row), encoding the images they name
  // first. out may equal tokens. An id of no held image maps to the image token (logged once).
  void prepare(const int32_t *tokens, int m, int32_t *out);
  // Any thread: the image a virtual id names (its first id, tokens and merged grid: the M-RoPE geometry); false = none.
  bool image(int64_t id, model::MropeImage &out) const;
  // The token every id-based input (PLE n-grams, penalties) reads at an image position.
  int32_t image_token() const { return vc_.image_token; }
  // That token for a virtual id: the video token at a video frame's ids, else the image token.
  int32_t pad_token(int64_t id) const;
  int64_t vocab() const { return vocab_; }
  const void *pool(int c) const { return pool_[static_cast<size_t>(c)]; }
  int64_t rows() const { return rows_; }
  const model::VisionConfig &config() const { return vc_; }
  int64_t max_pixels = 0;  // the server's cap (preprocess)

  struct Stats {
    uint64_t images = 0, encoded = 0, hits = 0, evicted = 0;
    double encode_ms = 0;
  };
  Stats stats() const;

 private:
  struct Entry {
    int64_t base = 0;   // virtual ids [base, base + tokens)
    int64_t row = -1;   // pool rows [row, row + tokens)
    int tokens = 0;
    int gh = 0, gw = 0;  // merged grid
    bool video = false;  // a video's temporal patch
    int refs = 0;
    bool encoded = false;
    uint64_t used = 0;  // LRU tick
    std::unique_ptr<model::VisionInput> input;  // until encoded
  };
  int64_t alloc_rows(int tokens);  // -1 = no room (caller evicts)
  void release(uint64_t key);

  Fabric &fab_;
  std::unique_ptr<VisionTower> tower_;
  model::VisionConfig vc_;
  int64_t vocab_ = 0, rows_ = 0;
  std::vector<void *> pool_;
  mutable std::mutex mu_;
  std::unordered_map<uint64_t, Entry> by_key_;  // content hash -> entry
  std::map<int64_t, uint64_t> by_base_;         // first virtual id -> content hash
  int64_t next_base_ = 0;
  uint64_t tick_ = 0;
  Stats stats_;
  bool warned_ = false;
};

}  // namespace xenithium::arch
