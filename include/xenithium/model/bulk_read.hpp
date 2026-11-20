// Xenithium · reading many tensors from a checkpoint at drive speed.
// A model load reads ~64 GB of experts from a checkpoint larger than the page cache, so it
// reads large merged ranges with O_DIRECT on several threads. Keep one bulk_read in flight at a time
// (two at once share the drive); SeqReader reads a sequence of them in order.
// Rules this reader keeps:
// * tensors of one shard closer than `merge_gap` are merged into one 4 KiB-aligned range;
// * EOF is tolerated only for the alignment padding after a range's last tensor byte;
// a short read inside tensor data throws with shard and offset (never a silent zero);
// * O_DIRECT may open fine and still fail pread with EINVAL (overlayfs, some btrfs/zfs,
// 4Kn devices): such a piece is retried once through a buffered fd;
// * every buffer is 4 KiB aligned (O_DIRECT needs it; so do device copies).
// Peak memory = the sum of the merged ranges (which can exceed the tensors' own bytes by up
// to merge_gap per gap): callers bound it by reading a layer at a time.
#pragma once

#include <cstddef>
#include <cstdint>
#include <condition_variable>
#include <functional>
#include <future>
#include <mutex>
#include <thread>
#include <string>
#include <vector>

#include "xenithium/model/checkpoint.hpp"

namespace xenithium::model {

struct BulkOptions {
  int threads = 8;
  uint64_t merge_gap = 64ull << 20;  // merge tensors of a shard closer than this
  uint64_t piece = 4ull << 20;  // one pread
  bool direct = true;  // O_DIRECT (falls back per piece on EINVAL)
  // Buffer allocator (4 KiB-aligned); default aligned_alloc / free. A loader can hand in
  // pinned host memory so the read lands where the device copy starts.
  std::function<void *(size_t)> alloc;
  std::function<void(void *)> release;
};

struct BulkStats {
  uint64_t tensor_bytes = 0;  // bytes the caller asked for
  uint64_t read_bytes = 0;  // bytes read from the drive (merged ranges, incl. gaps)
  size_t ranges = 0, pieces = 0;
  int buffered_retries = 0;  // pieces re-read without O_DIRECT
  bool direct = false;  // at least one piece used O_DIRECT
  double seconds = 0;
  double gbs() const { return seconds > 0 ? static_cast<double>(read_bytes) / seconds / 1e9 : 0; }
};

class BulkBuffer {
 public:
  BulkBuffer() = default;
  BulkBuffer(const BulkBuffer &) = delete;
  BulkBuffer &operator=(const BulkBuffer &) = delete;
  BulkBuffer(BulkBuffer &&o) noexcept;
  BulkBuffer &operator=(BulkBuffer &&o) noexcept;
  ~BulkBuffer();

  // Pointer to the tensor's bytes; throws when the tensor was not part of the read.
  const uint8_t *data(const TensorInfo &t) const;
  template <typename T>
  const T *as(const TensorInfo &t) const {
    return reinterpret_cast<const T *>(data(t));
  }
  const BulkStats &stats() const { return stats_; }

 private:
  friend BulkBuffer bulk_read(const Checkpoint &, const std::vector<const TensorInfo *> &, const BulkOptions &);
  struct Range {
    uint32_t shard = 0;
    uint64_t off = 0, len = 0;  // 4 KiB-aligned file range
    uint64_t end = 0;  // absolute end of the last tensor byte in it
    uint8_t *buf = nullptr;
  };
  void free_all();
  std::vector<Range> ranges_;
  std::function<void(void *)> release_;
  BulkStats stats_;
};

// Reusable 4 KiB-aligned buffers for repeated bulk reads (a layer at a time): freeing a read
// buffer per layer costs a munmap of its touched pages, and a fresh buffer is faulted in again
// by the next read. Blocks are kept and handed out again by best fit. Thread-safe. Hand
// `options` to bulk_read; the pool must outlive the buffers.
class BufferPool {
 public:
  BufferPool() = default;
  BufferPool(const BufferPool &) = delete;
  BufferPool &operator=(const BufferPool &) = delete;
  ~BufferPool();
  void *take(size_t bytes);
  void give(void *p);
  BulkOptions options(BulkOptions base = {});
  size_t held_bytes() const;

 private:
  struct Block {
    void *p;
    size_t cap;
    bool used;
  };
  mutable std::mutex mu_;
  std::vector<Block> blocks_;
};

BulkBuffer bulk_read(const Checkpoint &ck, const std::vector<const TensorInfo *> &tensors, const BulkOptions &opt = {});

// Reads items 0.. n-1 in order on one thread (read(i), e.g. one bulk_read per layer), at most
// `ahead` items past the last one taken; take(i) hands item i over in order, rethrowing its read error.
// `gate` (optional): the first read waits for it (another reader's last read, so the two do not share the
// drive); `done` (optional): called on the reader thread after the last read. The destructor stops the reader
// (a read in progress completes first).
class SeqReader {
 public:
  SeqReader(int n, int ahead, std::function<BulkBuffer(int)> read, std::function<void()> done = {}, std::shared_future<void> gate = {});
  SeqReader(const SeqReader &) = delete;
  SeqReader &operator=(const SeqReader &) = delete;
  ~SeqReader();
  BulkBuffer take(int i);

 private:
  std::mutex mu_;
  std::condition_variable cv_;
  std::vector<BulkBuffer> bufs_;
  std::vector<uint8_t> ready_;
  int taken_ = 0, ahead_ = 1;
  bool stop_ = false;
  std::exception_ptr err_;
  std::thread th_;
};

}  // namespace xenithium:model
