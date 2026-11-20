#include "xenithium/model/row_reader.hpp"

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <map>
#include <stdexcept>

namespace xenithium::model {

RowReader::RowReader(std::vector<Segment> segs, size_t row_bytes, int threads) : row_bytes_(row_bytes) {
  if (row_bytes == 0) throw std::runtime_error("row reader: row_bytes is 0");
  std::map<std::string, int> fds;  // one descriptor per file
  for (auto &s : segs) {
    auto it = fds.find(s.path);
    if (it == fds.end()) {
      const int fd = ::open(s.path.c_str(), O_RDONLY | O_CLOEXEC);
      if (fd < 0) throw std::runtime_error("row reader: open " + s.path + ": " + std::strerror(errno));
      it = fds.emplace(s.path, fd).first;
      paths_.push_back(s.path);
    }
    segs_.push_back(Seg{it->second, s.offset, rows_, s.rows});
    rows_ += s.rows;
  }
  for (int t = 1; t < std::max(1, threads); ++t) workers_.emplace_back([this] { work(); });
}

RowReader::~RowReader() {
  {
    std::lock_guard<std::mutex> g(mu_);
    stop_ = true;
  }
  cv_job_.notify_all();
  for (auto &t : workers_) t.join();
  std::vector<int> closed;
  for (const auto &s : segs_)
    if (std::find(closed.begin(), closed.end(), s.fd) == closed.end()) {
      ::close(s.fd);
      closed.push_back(s.fd);
    }
}

void RowReader::read_one(size_t i) {
  const int64_t id = ids_[i];
  if (id < 0 || id >= rows_) {
    if (!failed_.exchange(true)) error_ = "row reader: id " + std::to_string(id) + " out of range";
    return;
  }
  // Segments are few (one per table shard); binary search by first row.
  size_t lo = 0, hi = segs_.size();
  while (hi - lo > 1) {
    const size_t mid = (lo + hi) / 2;
    if (segs_[mid].first <= id) lo = mid;
    else hi = mid;
  }
  const Seg &s = segs_[lo];
  const uint64_t off = s.offset + static_cast<uint64_t>(id - s.first) * row_bytes_;
  uint8_t *dst = out_ + i * row_bytes_;
  size_t got = 0;
  while (got < row_bytes_) {
    const ssize_t r = ::pread(s.fd, dst + got, row_bytes_ - got, static_cast<off_t>(off + got));
    if (r <= 0) {
      if (r < 0 && errno == EINTR) continue;
      if (!failed_.exchange(true)) error_ = "row reader: short read of row " + std::to_string(id);
      return;
    }
    got += static_cast<size_t>(r);
  }
}

void RowReader::work() {
  uint64_t seen = 0;
  for (;;) {
    {
      std::unique_lock<std::mutex> lk(mu_);
      cv_job_.wait(lk, [&] { return stop_ || gen_ != seen; });
      if (stop_) return;
      seen = gen_;
    }
    size_t mine = 0;
    for (size_t i; (i = next_.fetch_add(1)) < n_;) {
      read_one(i);
      ++mine;
    }
    if (mine && done_.fetch_add(mine) + mine == n_.load()) {
      std::lock_guard<std::mutex> g(mu_);
      cv_done_.notify_all();
    }
  }
}

void RowReader::gather(const int64_t *ids, size_t n, uint8_t *out) {
  if (n == 0) return;
  {
    std::lock_guard<std::mutex> g(mu_);
    ids_ = ids;
    out_ = out;
    n_ = n;
    next_ = 0;
    done_ = 0;
    failed_ = false;
    ++gen_;
  }
  cv_job_.notify_all();
  size_t mine = 0;
  for (size_t i; (i = next_.fetch_add(1)) < n;) {
    read_one(i);
    ++mine;
  }
  done_.fetch_add(mine);
  {
    std::unique_lock<std::mutex> lk(mu_);
    cv_done_.wait(lk, [&] { return done_.load() >= n; });
    n_ = 0;  // late workers see no work for this generation
  }
  if (failed_) throw std::runtime_error(error_);
}

void RowReader::drop_cache() const {
  for (const auto &s : segs_) {
    (void)::posix_fadvise(s.fd, 0, 0, POSIX_FADV_DONTNEED);
  }
}

}  // namespace xenithium:model
