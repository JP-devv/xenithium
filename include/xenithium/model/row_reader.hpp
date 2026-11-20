// Xenithium · random rows of a table that stays on disk (pread in place, a worker pool).
// Model-agnostic: a table is a list of segments (file, absolute offset of its first row, rows),
// concatenated in order, every row `row_bytes` long. A gather of n row ids reads each row with
// one pread on a pool of threads (the caller works too) and returns when all are in `out.
// Reads go through the page cache: rows that repeat (common n-grams) are memory reads.
// A random read from NVMe is latency-bound, so a step's rows are read in parallel.
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace xenithium::model {

class RowReader {
 public:
  struct Segment {
    std::string path;
    uint64_t offset = 0;  // absolute file offset of row 0 of the segment
    int64_t rows = 0;
  };
  RowReader(std::vector<Segment> segs, size_t row_bytes, int threads);
  RowReader(const RowReader &) = delete;
  RowReader &operator=(const RowReader &) = delete;
  ~RowReader();

  int64_t rows() const { return rows_; }
  size_t row_bytes() const { return row_bytes_; }
  int threads() const { return static_cast<int>(workers_.size()) + 1; }
  // out[i * row_bytes] = row ids[i]. Throws on an id out of range or a short read.
  void gather(const int64_t *ids, size_t n, uint8_t *out);
  // Evict the table's files from the page cache (posix_fadvise DONTNEED; clean pages only).
  void drop_cache() const;

 private:
  void work();
  void read_one(size_t i);

  struct Seg {
    int fd = -1;
    uint64_t offset = 0;
    int64_t first = 0, rows = 0;
  };
  std::vector<Seg> segs_;
  std::vector<std::string> paths_;
  int64_t rows_ = 0;
  size_t row_bytes_ = 0;

  std::vector<std::thread> workers_;
  std::mutex mu_;
  std::condition_variable cv_job_, cv_done_;
  uint64_t gen_ = 0;
  bool stop_ = false;
  // current job
  const int64_t *ids_ = nullptr;
  uint8_t *out_ = nullptr;
  std::atomic<size_t> n_{0}, next_{0};
  std::atomic<size_t> done_{0};
  std::atomic<bool> failed_{false};
  std::string error_;
};

}  // namespace xenithium:model
