#include "xenithium/model/bulk_read.hpp"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <stdexcept>
#include <thread>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace xenithium::model {

namespace {
constexpr uint64_t kAlign = 4096;
}

BulkBuffer::BulkBuffer(BulkBuffer &&o) noexcept
    : ranges_(std::move(o.ranges_)), release_(std::move(o.release_)), stats_(o.stats_) {
  o.ranges_.clear();
}
BulkBuffer &BulkBuffer::operator=(BulkBuffer &&o) noexcept {
  if (this != &o) {
    free_all();
    ranges_ = std::move(o.ranges_);
    release_ = std::move(o.release_);
    stats_ = o.stats_;
    o.ranges_.clear();
  }
  return *this;
}
BulkBuffer::~BulkBuffer() { free_all(); }

void BulkBuffer::free_all() {
  for (auto &r : ranges_)
    if (r.buf) release_ ? release_(r.buf) : std::free(r.buf);
  ranges_.clear();
}

const uint8_t *BulkBuffer::data(const TensorInfo &t) const {
  // Ranges are sorted by (shard, off): binary search the last range starting at or before t.
  auto it = std::upper_bound(ranges_.begin(), ranges_.end(), std::make_pair(t.shard, t.offset),
                             [](const std::pair<uint32_t, uint64_t> &k, const Range &r) {
                               return k.first < r.shard || (k.first == r.shard && k.second < r.off);
                             });
  if (it != ranges_.begin()) {
    const Range &r = *std::prev(it);
    if (r.shard == t.shard && t.offset >= r.off && t.offset + t.bytes <= r.off + r.len) return r.buf + (t.offset - r.off);
  }
  throw std::runtime_error("bulk read: tensor " + t.name + " was not part of this read");
}

BufferPool::~BufferPool() {
  for (auto &b : blocks_) std::free(b.p);
}

void *BufferPool::take(size_t bytes) {
  std::lock_guard<std::mutex> lk(mu_);
  Block *best = nullptr;
  for (auto &b : blocks_)
    if (!b.used && b.cap >= bytes && (!best || b.cap < best->cap)) best = &b;
  if (best) {
    best->used = true;
    return best->p;
  }
  const size_t cap = (bytes + kAlign - 1) / kAlign * kAlign;
  void *p = std::aligned_alloc(kAlign, cap);
  if (!p) return nullptr;
  blocks_.push_back({p, cap, true});
  return p;
}

void BufferPool::give(void *p) {
  std::lock_guard<std::mutex> lk(mu_);
  for (auto &b : blocks_)
    if (b.p == p) {
      b.used = false;
      return;
    }
  std::free(p);  // not ours
}

BulkOptions BufferPool::options(BulkOptions base) {
  base.alloc = [this](size_t n) { return take(n); };
  base.release = [this](void *p) { give(p); };
  return base;
}

size_t BufferPool::held_bytes() const {
  std::lock_guard<std::mutex> lk(mu_);
  size_t s = 0;
  for (const auto &b : blocks_) s += b.cap;
  return s;
}

BulkBuffer bulk_read(const Checkpoint &ck, const std::vector<const TensorInfo *> &tensors, const BulkOptions &opt) {
  const auto t0 = std::chrono::steady_clock::now();
  BulkBuffer out;
  out.release_ = opt.release;
  auto alloc = [&](size_t n) -> uint8_t * {
    void *p = opt.alloc ? opt.alloc(n) : std::aligned_alloc(kAlign, n);
    if (!p) throw std::runtime_error("bulk read: cannot allocate " + std::to_string(n) + " bytes");
    if (reinterpret_cast<uintptr_t>(p) % kAlign) throw std::runtime_error("bulk read: allocator returned an unaligned buffer");
    return static_cast<uint8_t *>(p);
  };

  // ---- merge per shard ----
  std::map<uint32_t, std::vector<std::pair<uint64_t, uint64_t>>> by;
  for (const TensorInfo *t : tensors) {
    if (!t) throw std::runtime_error("bulk read: null tensor");
    out.stats_.tensor_bytes += t->bytes;
    if (t->bytes) by[t->shard].push_back({t->offset, t->offset + t->bytes});
  }
  for (auto &[shard, iv] : by) {
    std::sort(iv.begin(), iv.end());
    uint64_t b = iv[0].first, e = iv[0].second;
    auto flush = [&] {
      BulkBuffer::Range r;
      r.shard = shard;
      r.off = b / kAlign * kAlign;
      r.len = (e - r.off + kAlign - 1) / kAlign * kAlign;
      r.end = e;
      out.ranges_.push_back(r);
    };
    for (size_t i = 1; i < iv.size(); ++i) {
      if (iv[i].first > e + opt.merge_gap) {
        flush();
        b = iv[i].first;
      }
      e = std::max(e, iv[i].second);
    }
    flush();
  }

  // ---- open + size check before any buffer is trusted ----
  struct Fds {
    int direct = -1, buffered = -1;
  };
  std::map<uint32_t, Fds> fds;
  auto close_all = [&] {
    for (auto &[s, f] : fds) {
      if (f.direct >= 0) ::close(f.direct);
      if (f.buffered >= 0) ::close(f.buffered);
    }
  };
  struct Piece {
    size_t range;
    uint64_t at, len;
  };
  std::vector<Piece> pieces;
  try {
    for (const auto &r : out.ranges_) {
      if (fds.count(r.shard)) continue;
      const ShardInfo &s = ck.shards().at(r.shard);
      Fds f;
      f.buffered = ::open(s.path.c_str(), O_RDONLY | O_CLOEXEC);
      if (f.buffered < 0) throw std::runtime_error("bulk read: cannot open " + s.path + ": " + std::strerror(errno));
      fds[r.shard] = f;  // registered before the next call can throw
      if (opt.direct) fds[r.shard].direct = ::open(s.path.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECT);
      struct stat st{};
      if (::fstat(f.buffered, &st) != 0) throw std::runtime_error("bulk read: cannot stat " + s.path);
      if (static_cast<uint64_t>(st.st_size) != s.file_bytes)
        throw std::runtime_error("bulk read: " + s.file + " changed size since the catalog was built (" +
                                 std::to_string(s.file_bytes) + " -> " + std::to_string(st.st_size) + ")");
    }
    for (const auto &r : out.ranges_) {
      const uint64_t size = ck.shards()[r.shard].file_bytes;
      if (r.end > size)
        throw std::runtime_error("bulk read: " + ck.shards()[r.shard].file + " is " + std::to_string(size) + " bytes, tensors need " +
                                 std::to_string(r.end));
    }
    for (size_t i = 0; i < out.ranges_.size(); ++i) {
      auto &r = out.ranges_[i];
      r.buf = alloc(r.len);
      out.stats_.read_bytes += std::min(r.len, r.end - r.off);
      for (uint64_t a = 0; a < r.len; a += opt.piece) pieces.push_back({i, a, std::min(opt.piece, r.len - a)});
    }
  } catch (...) {
    close_all();
    throw;
  }

  // ---- read ----
  std::atomic<size_t> next{0};
  std::atomic<int> retries{0};
  std::atomic<bool> used_direct{false};
  std::mutex err_mu;
  std::string err;
  auto worker = [&] {
    for (size_t i; (i = next.fetch_add(1)) < pieces.size();) {
      {
        std::lock_guard<std::mutex> lk(err_mu);
        if (!err.empty()) return;
      }
      const Piece &p = pieces[i];
      const auto &r = out.ranges_[p.range];
      const Fds &f = fds.at(r.shard);
      int fd = f.direct >= 0 ? f.direct : f.buffered;
      bool retried = fd == f.buffered;
      const uint64_t at = r.off + p.at;
      // Bytes of this piece that hold tensor data (the rest is alignment padding, possibly past EOF).
      const uint64_t want = r.end > at ? std::min(p.len, r.end - at) : 0;
      uint64_t done = 0;
      // O_DIRECT needs aligned lengths: request the whole aligned piece, accept EOF only past `want.
      while (done < want) {
        const ssize_t got = ::pread(fd, r.buf + p.at + done, p.len - done, static_cast<off_t>(at + done));
        if (got < 0 && errno == EINTR) continue;
        if (got < 0 && errno == EINVAL && !retried) {
          retried = true;
          fd = f.buffered;
          retries.fetch_add(1);
          continue;
        }
        if (got <= 0) {
          std::lock_guard<std::mutex> lk(err_mu);
          if (err.empty())
            err = "bulk read: " + ck.shards()[r.shard].file + " offset " + std::to_string(at + done) +
                  (got < 0 ? std::string(": ") + std::strerror(errno) : std::string(": end of file inside tensor data"));
          return;
        }
        done += static_cast<uint64_t>(got);
      }
      if (fd == f.direct) used_direct = true;
      if (done < p.len) std::memset(r.buf + p.at + done, 0, p.len - done);
    }
  };
  const int nt = std::max(1, std::min<int>(opt.threads, static_cast<int>(pieces.size())));
  std::vector<std::thread> th;
  for (int t = 0; t < nt; ++t) th.emplace_back(worker);
  for (auto &t : th) t.join();
  close_all();
  if (!err.empty()) throw std::runtime_error(err);

  out.stats_.ranges = out.ranges_.size();
  out.stats_.pieces = pieces.size();
  out.stats_.buffered_retries = retries.load();
  out.stats_.direct = used_direct.load();
  out.stats_.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  return out;
}

SeqReader::SeqReader(int n, int ahead, std::function<BulkBuffer(int)> read, std::function<void()> done, std::shared_future<void> gate)
    : bufs_(static_cast<size_t>(std::max(0, n))), ready_(static_cast<size_t>(std::max(0, n)), 0), ahead_(std::max(1, ahead)) {
  th_ = std::thread([this, n, read = std::move(read), done = std::move(done), gate] {
    if (gate.valid())  // wait for the gate, or for the stop
      for (;;) {
        if (gate.wait_for(std::chrono::milliseconds(5)) == std::future_status::ready) break;
        std::lock_guard<std::mutex> lk(mu_);
        if (stop_) return;
      }
    for (int i = 0; i < n; ++i) {
      {
        std::unique_lock<std::mutex> lk(mu_);
        cv_.wait(lk, [&] { return stop_ || i < taken_ + ahead_; });
        if (stop_) return;
      }
      try {
        BulkBuffer b = read(i);
        std::lock_guard<std::mutex> lk(mu_);
        bufs_[static_cast<size_t>(i)] = std::move(b);
        ready_[static_cast<size_t>(i)] = 1;
      } catch (...) {
        std::lock_guard<std::mutex> lk(mu_);
        err_ = std::current_exception();
        cv_.notify_all();
        return;
      }
      cv_.notify_all();
    }
    if (done) done();
  });
}

SeqReader::~SeqReader() {
  {
    std::lock_guard<std::mutex> lk(mu_);
    stop_ = true;
  }
  cv_.notify_all();
  if (th_.joinable()) th_.join();
}

BulkBuffer SeqReader::take(int i) {
  if (i < 0 || static_cast<size_t>(i) >= ready_.size()) throw std::out_of_range("SeqReader::take");
  std::unique_lock<std::mutex> lk(mu_);
  cv_.wait(lk, [&] { return ready_[static_cast<size_t>(i)] || err_; });
  if (!ready_[static_cast<size_t>(i)]) std::rethrow_exception(err_);
  BulkBuffer b = std::move(bufs_[static_cast<size_t>(i)]);
  ready_[static_cast<size_t>(i)] = 0;
  taken_ = std::max(taken_, i + 1);
  cv_.notify_all();
  return b;
}

}  // namespace xenithium:model
