// Xenithium · KV lending, host side.
// The shared KV pool is one buffer per (attention layer, tensor) of pool_pages x page_bytes each. Pages at or above the
// pool's lend limit W (KvPages:limit) hold no KV, so the top of every buffer is free memory: KvLend cuts each buffer's
// top into slots of `stride` bytes (aligned down to `align` from the buffer's aligned end) and orders them by the page W
// must stay at or below for the slot to be untouched, its "min page" = floor(offset / page_bytes) (KV pages [0, W) end at
// W x page_bytes <= offset). slots is sorted by min page descending: at limit W the usable slots are the prefix
// usable(W), so the caller puts its most valuable data in the first slots and a rising W takes the last ones back first.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace xenithium::model {

class KvLend {
 public:
  struct Slot {
    int buf = 0;  // buffer index (the constructor's order)
    size_t off = 0;  // byte offset in the buffer
    int min_page = 0;  // usable while the lend limit <= min_page
  };

  KvLend() = default;
  // page_bytes[b]: one page of buffer b (pool_pages pages each); stride: bytes per slot; max_slots caps the count (the
  // slots with the highest min pages are kept)
  KvLend(const std::vector<size_t> &page_bytes, int pool_pages, size_t stride, size_t align, size_t max_slots) {
    if (pool_pages < 1 || stride == 0 || align == 0) throw std::invalid_argument("kv lend: pool pages, stride and alignment > 0");
    const size_t S = (stride + align - 1) / align * align;
    for (size_t b = 0; b < page_bytes.size(); ++b) {
      const size_t pb = page_bytes[b];
      if (pb == 0) continue;
      const size_t end = static_cast<size_t>(pool_pages) * pb / align * align;
      for (size_t j = 1; j * S <= end; ++j) {
        const size_t off = end - j * S;
        const int mp = static_cast<int>(std::min<size_t>(off / pb, static_cast<size_t>(pool_pages)));
        if (mp < 1) break;  // page 0 is never lent (unmapped table entries address it)
        slots_.push_back({static_cast<int>(b), off, mp});
      }
    }
    std::stable_sort(slots_.begin(), slots_.end(), [](const Slot &a, const Slot &b) { return a.min_page > b.min_page; });
    if (slots_.size() > max_slots) slots_.resize(max_slots);
  }

  const std::vector<Slot> &slots() const { return slots_; }
  size_t size() const { return slots_.size(); }
  // slots usable at lend limit W: a prefix of slots
  int usable(int limit) const {
    return static_cast<int>(std::partition_point(slots_.begin(), slots_.end(), [&](const Slot &s) { return s.min_page >= limit; }) - slots_.begin());
  }
  // the highest limit at which the first n slots stay usable (pool_pages for n == 0)
  int limit_for(int n, int pool_pages) const {
    if (n <= 0) return pool_pages;
    return slots_[static_cast<size_t>(std::min<int>(n, static_cast<int>(slots_.size())) - 1)].min_page;
  }

 private:
  std::vector<Slot> slots_;
};

// Regions of the caller's sizes (one per item, in the caller's order) instead of equal slots. Item
// i's region is cut from the top of the buffer where it leaves the most pages to the KV (the highest min page; the
// lowest buffer index among equals); an item no buffer holds above page 0 gets none (buf -1). A region is untouched by
// the KV while the lend limit <= its min_page, each on its own (no prefix order): lent(W) counts them, and
// limit_for(W) = the highest limit that keeps every region usable at W (pool_pages when none is).
class KvLendRegions {
 public:
  struct Region {
    int buf = -1;  // -1: the item has no region
    size_t off = 0;
    size_t bytes = 0;  // the item's size rounded up to the alignment
    int min_page = 0;
  };

  KvLendRegions() = default;
  KvLendRegions(const std::vector<size_t> &page_bytes, int pool_pages, const std::vector<size_t> &sizes, size_t align) {
    if (pool_pages < 1 || align == 0) throw std::invalid_argument("kv lend regions: pool pages and alignment > 0");
    std::vector<size_t> end(page_bytes.size());
    for (size_t b = 0; b < page_bytes.size(); ++b) end[b] = static_cast<size_t>(pool_pages) * page_bytes[b] / align * align;
    regions_.resize(sizes.size());
    for (size_t i = 0; i < sizes.size(); ++i) {
      const size_t S = (sizes[i] + align - 1) / align * align;
      int best = -1, best_page = 0;
      for (size_t b = 0; b < page_bytes.size() && S > 0; ++b) {
        if (page_bytes[b] == 0 || end[b] < S) continue;
        const int mp = static_cast<int>(std::min<size_t>((end[b] - S) / page_bytes[b], static_cast<size_t>(pool_pages)));
        if (mp > best_page) best = static_cast<int>(b), best_page = mp;  // (mp >= 1: page 0 is never lent)
      }
      if (best < 0) continue;
      end[static_cast<size_t>(best)] -= S;
      regions_[i] = Region{best, end[static_cast<size_t>(best)], S, best_page};
    }
  }

  const std::vector<Region> &regions() const { return regions_; }
  size_t size() const { return regions_.size(); }
  bool usable(size_t i, int limit) const { return regions_[i].buf >= 0 && regions_[i].min_page >= limit; }
  int lent(int limit) const {
    int n = 0;
    for (size_t i = 0; i < regions_.size(); ++i) n += usable(i, limit);
    return n;
  }
  int limit_for(int limit, int pool_pages) const {
    int w = pool_pages;
    for (size_t i = 0; i < regions_.size(); ++i)
      if (usable(i, limit)) w = std::min(w, regions_[i].min_page);
    return w;
  }

 private:
  std::vector<Region> regions_;
};

}  // namespace xenithium:model
