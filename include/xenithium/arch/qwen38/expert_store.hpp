// Xenithium · qwen38 routed experts on the cards: EP placement + the streaming loader.
//
// Expert parallel: card r owns experts [r * E / C, (r + 1) * E / C) of every layer. Of those,
// `hot` per card live in VRAM (one allocation per layer, blobs at slot * stride); the rest are
// cold and live in pinned host memory (readable in place by a kernel over PCIe, or by a CPU
// path later). The loader streams the checkpoint one layer at a time:
//
//   reader thread:  O_DIRECT bulk read of layers ahead (the drive: the critical path); started before
//                   the dense load by start_reads (the raw reads need no placement)
//   load thread:    pack layer L (threads) -> hot into pinned staging, cold into their blobs
//                   -> one H2D batch per card on the copy engine (double-buffered staging)
//
// Which experts are hot is a placement policy:
//   - no routing counts: an even split per layer by expert id (the placeholder);
//   - routing counts [layer][expert] (ExpertStoreOptions::rank, e.g. collected by our own MoE on
//     calibration text): per card, the `hot` (layer, expert) pairs with the most routes over all
//     layers ("global hot"), so skewed layers give slots to flat ones. Ties: lower layer, then
//     lower expert id. Within a layer the hot blobs sit in ascending expert id.
#pragma once

#include <cstddef>
#include <cstdint>
#include <future>
#include <memory>
#include <string>
#include <vector>

#include "xenithium/arch/qwen38/experts.hpp"
#include "xenithium/kernels/moe/moe.hpp"
#include "xenithium/rt/fabric.hpp"

namespace xenithium::arch::qwen38 {

struct ExpertStoreOptions {
  int64_t hot_per_card = -1;   // -1 = everything the card owns
  int pack_threads = 16;
  model::BulkOptions read;     // the reader's options (threads, pieces, O_DIRECT)
  int read_ahead = 2;          // layers the reader runs ahead of the packing (when load() starts it)
  bool hash = false;           // keep a 64-bit hash of every packed blob (for verify())
  const std::vector<double> *rank = nullptr;  // routing counts [layer][expert]; null = even split
  // Prepacked TP blobs (xenithium-bake): <dir>/experts-LNN.tp2, per layer card 0's
  // [experts][tp stride] then card 1's, read as they are instead of read + packed from the GPTQ
  // tensors. TP over 2 cards only; empty = pack at load.
  std::string native_dir;
};

struct ExpertStoreStats {
  double total_s = 0;          // wall clock of load()
  double alloc_s = 0;          // VRAM + pinned host allocations (pinning is not free)
  double first_read_s = 0;     // the first layer's read (nothing to overlap it with yet)
  double read_wait_s = 0;      // load thread waiting for the reader (drive-bound time)
  double free_s = 0;           // releasing each layer's read buffers
  double pack_s = 0;           // packing (on the load thread's critical path)
  double h2d_wait_s = 0;       // waiting for the previous layer's copies before reusing staging
  uint64_t read_bytes = 0;     // from the drive
  bool native = false;         // the blobs came prepacked (native_dir)
  uint64_t h2d_bytes = 0;      // to the cards, all cards
  uint64_t pinned_bytes = 0;   // cold blobs + staging
};

class ExpertStore {
 public:
  // tp: every card holds its 1/cards of every expert (kernels::moe_tp_layout), hot or cold together.
  ExpertStore(Fabric &fabric, const Config &cfg, bool tp = false);
  ExpertStore(const ExpertStore &) = delete;
  ExpertStore &operator=(const ExpertStore &) = delete;
  ~ExpertStore();

  void load(const model::Checkpoint &ck, const ExpertStoreOptions &opt);
  // Starts the reader before load(), e.g. while the dense weights load (their packing leaves the
  // drive idle): layers 0, 1, .. are read into host buffers, at most `ahead` layers beyond the one load()
  // packs; the first read waits for `gate` when given (the dense reads' end: two readers at once slow
  // the drive). Not for native_dir loads (load() drops the reader then). ck must outlive load().
  void start_reads(const model::Checkpoint &ck, const model::BulkOptions &read, int ahead, std::shared_future<void> gate = {});
  // Host bytes the reader holds now (layers read and not yet packed).
  size_t reader_bytes() const;

  const ExpertLayout &layout() const { return lay_; }
  int cards() const { return cards_; }
  int per_card() const { return per_card_; }      // blobs each card holds per layer (EP: owned experts; TP: all)
  bool tp() const { return tp_; }
  size_t stride() const { return stride_; }       // bytes per blob on a card
  int owner(int expert) const { return expert / per_card_; }
  // Number of hot experts of (card, layer).
  int hot_count(int card, int layer) const { return hot_[idx(card, layer)]; }
  int64_t hot_total(int card) const;
  // Where expert (layer, e) lives: device pointer on its owner card when hot, else a pinned
  // host pointer. `hot` reports which.
  // Under TP every card has a blob of every expert: `card` picks it (ignored under EP).
  const uint8_t *blob(int layer, int expert, bool *hot = nullptr, int card = 0) const;
  const ExpertStoreStats &stats() const { return stats_; }

  // Reads every hot blob back from the cards and compares its hash with the one taken at pack
  // time; cold blobs are re-hashed in host memory. Needs opt.hash. Returns mismatches.
  int64_t verify(int threads = 16) const;

 private:
  size_t idx(int card, int layer) const { return static_cast<size_t>(card) * layers_ + layer; }
  void place(int64_t want_per_card, const std::vector<double> *rank);
  void load_native(const ExpertStoreOptions &opt, std::vector<uint8_t *> &stage, std::vector<int> &inflight);
  struct Reader;
  std::unique_ptr<Reader> reader_;
  Fabric &fab_;
  Config cfg_;
  ExpertLayout lay_;
  bool tp_ = false;
  kernels::MoeBlobLayout tp_lay_;
  size_t stride_ = 0;
  int cards_ = 0, layers_ = 0, experts_ = 0, per_card_ = 0;
  std::vector<int> hot_;                 // [card][layer] hot count
  std::vector<uint8_t> is_hot_;          // [layer][expert]
  std::vector<int> slot_;                // [layer][expert]: index in its (card, layer) hot or cold array
  std::vector<uint8_t *> dev_;           // [card][layer] device allocation (hot blobs), may be null
  std::vector<uint8_t *> cold_;          // [card][layer] pinned host allocation (cold blobs), may be null
  std::vector<uint64_t> hash_;           // [layer][expert]
  ExpertStoreStats stats_;
};

// 64-bit hash of a byte range (multiplicative mixing over 8-byte words); for load verification,
// not security.
uint64_t blob_hash(const uint8_t *p, size_t n);

}  // namespace xenithium::arch::qwen38
