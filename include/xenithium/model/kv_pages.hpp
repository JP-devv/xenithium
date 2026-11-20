// Xenithium · the shared paged KV pool, host side.
// The attention kernels read and write K / V through a page table: GqaCache:pages is [slots][slot_pages] ints, entry
// (s, i) = the physical page holding tokens [64 i, 64 i + 64) of slot s's sequence (kernels/attention/gqa.hpp). KvPages
// keeps that table on the host for a pool of `pool_pages` physical pages shared by every slot:
// - reserve(slot, tokens) maps the slot's logical pages [0, ceil(tokens / 64)) (all or nothing; false = the pool
// cannot, nothing changed), lazily as a sequence grows;
// - release(slot) returns every page of the slot (a finished or reset sequence);
// - dirty is the flat table range changed since the last clear_dirty: the caller uploads it to the cards.
// Unmapped entries hold 0, a valid page: a kernel never reads past a row's own position, but a prefetch of the next
// tile's page id may, so every entry must address memory inside the pool.
// The fixed layout (shared false) is the identity table with every page owned from the start: reserve succeeds up
// to the slot's capacity and release keeps the pages (bit for bit).
// `scramble` hands free pages out in a fixed pseudo-random order instead of lowest-first (a gate: results must not
// depend on where pages live).
// Shared pages: share(dst, src, n) maps dst's logical pages [0, n) to src's physical
// ones (a prefix-cache fork: positions below a snapshot at a page boundary, which neither sequence writes again) with a
// reference count per physical page; release / trim return a page to the free list when its last holder lets go.
// unshare(slot, from) gives the slot private copies of its shared pages from logical page `from` on (copy-on-write: the
// caller copies the returned (old, new) physical pairs before it writes there).
// KV lending: a shared, unscrambled pool has a limit: pages >= limit are lent (their memory holds other
// data, model/kv_lend.hpp) and never handed out. reserve / unshare take pages below it only; set_limit raises it, or
// lowers it to no less than high_water (the pages from there up are free).
#pragma once

#include <algorithm>
#include <cstdint>
#include <functional>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

namespace xenithium::model {

class KvPages {
 public:
  static constexpr int kPageTokens = 64;

  KvPages() = default;
  // shared = false: the identity layout (pool_pages must be slots x slot_pages)
  KvPages(int slots, int slot_pages, int pool_pages, bool shared, bool scramble = false)
      : slots_(slots), slot_pages_(slot_pages), pool_pages_(pool_pages), limit_(pool_pages), shared_(shared), scramble_(scramble) {
    if (slots < 1 || slot_pages < 1 || pool_pages < 1) throw std::invalid_argument("kv pages: slots, slot pages and pool pages >= 1");
    if (!shared && static_cast<int64_t>(pool_pages) != static_cast<int64_t>(slots) * slot_pages)
      throw std::invalid_argument("kv pages: the fixed layout holds slots x slot_pages pages");
    if (shared && pool_pages < slot_pages) throw std::invalid_argument("kv pages: a shared pool holds at least one slot's capacity");
    table_.assign(static_cast<size_t>(slots) * slot_pages, 0);
    held_.assign(static_cast<size_t>(slots), 0);
    if (!shared) {
      std::iota(table_.begin(), table_.end(), 0);
      std::fill(held_.begin(), held_.end(), slot_pages);
      dirty_lo_ = 0;
      dirty_hi_ = table_.size();
      return;
    }
    // the free list is popped from the back: lowest page first, or a fixed shuffle
    ref_.assign(static_cast<size_t>(pool_pages), 0);
    free_.resize(static_cast<size_t>(pool_pages));
    std::iota(free_.rbegin(), free_.rend(), 0);
    if (scramble) {
      std::mt19937 rng(18);
      std::shuffle(free_.begin(), free_.end(), rng);
    }
    dirty_lo_ = 0;
    dirty_hi_ = table_.size();
  }

  bool shared() const { return shared_; }
  int slots() const { return slots_; }
  int slot_pages() const { return slot_pages_; }
  int pool_pages() const { return pool_pages_; }
  int64_t pool_tokens() const { return static_cast<int64_t>(pool_pages_) * kPageTokens; }
  int64_t free_tokens() const { return shared_ ? static_cast<int64_t>(free_.size()) * kPageTokens : 0; }
  // the lend limit: free pages below it, the lowest page from which every page up is free
  int limit() const { return limit_; }
  int free_below_limit() const {
    if (!shared_) return 0;
    if (limit_ >= pool_pages_) return static_cast<int>(free_.size());
    // free_ is sorted descending (unscrambled): the pages < limit are its tail
    return static_cast<int>(free_.end() - std::upper_bound(free_.begin(), free_.end(), limit_, std::greater<int32_t>()));
  }
  int high_water() const {
    if (!shared_) return pool_pages_;
    int h = pool_pages_;
    while (h > 0 && ref_[static_cast<size_t>(h - 1)] == 0) --h;
    return h;
  }
  // pages -> [1, pool_pages] (page 0 is what unmapped table entries address); false (unchanged): a scrambled or fixed
  // pool, or below high_water
  bool set_limit(int pages) {
    pages = std::clamp(pages, 1, pool_pages_);
    if (!shared_ || (scramble_ && pages < pool_pages_) || pages < high_water()) return false;
    limit_ = pages;
    return true;
  }
  // pages the slot still needs for positions [0, tokens) (0: it holds them)
  int pages_short(int slot, int64_t tokens) const {
    check(slot);
    return std::max(0, std::min(pages_for(tokens), slot_pages_) - held_[static_cast<size_t>(slot)]);
  }
  int64_t used_tokens() const {  // physical pages in use (a shared page once)
    if (shared_) return (static_cast<int64_t>(pool_pages_) - static_cast<int64_t>(free_.size())) * kPageTokens;
    int64_t n = 0;
    for (int h : held_) n += h;
    return n * kPageTokens;
  }
  int held_pages(int slot) const { return held_.at(static_cast<size_t>(slot)); }
  static int pages_for(int64_t tokens) { return static_cast<int>((std::max<int64_t>(tokens, 0) + kPageTokens - 1) / kPageTokens); }

  bool reserve(int slot, int64_t tokens) {
    check(slot);
    const int want = pages_for(tokens);
    if (want > slot_pages_) return false;
    int &h = held_[static_cast<size_t>(slot)];
    if (want <= h) return true;
    if (!shared_) return false;  // unreachable: the fixed layout holds every page
    if (want - h > free_below_limit()) return false;
    const size_t row = static_cast<size_t>(slot) * slot_pages_;
    for (int i = h; i < want; ++i) {
      table_[row + i] = free_.back();
      ref_[static_cast<size_t>(free_.back())] = 1;
      free_.pop_back();
    }
    mark(row + h, row + want);
    h = want;
    return true;
  }

  void release(int slot) {
    check(slot);
    if (!shared_) return;
    int &h = held_[static_cast<size_t>(slot)];
    const size_t row = static_cast<size_t>(slot) * slot_pages_;
    for (int i = h - 1; i >= 0; --i) {  // back on the free list so the lowest page comes out first again
      drop(table_[row + i]);
      table_[row + i] = 0;
    }
    if (h > 0) mark(row, row + h);
    h = 0;
    if (!scramble_) std::sort(free_.begin(), free_.end(), std::greater<int32_t>());
  }

  // (prefix cache): the slot keeps the pages of positions [0, tokens) and returns the rest (a
  // sequence resumed at an earlier position of its own history). The fixed layout keeps every page.
  void trim(int slot, int64_t tokens) {
    check(slot);
    if (!shared_) return;
    int &h = held_[static_cast<size_t>(slot)];
    const int keep = std::min(h, pages_for(tokens));
    if (keep == h) return;
    const size_t row = static_cast<size_t>(slot) * slot_pages_;
    for (int i = h - 1; i >= keep; --i) {
      drop(table_[row + i]);
      table_[row + i] = 0;
    }
    mark(row + keep, row + h);
    h = keep;
    if (!scramble_) std::sort(free_.begin(), free_.end(), std::greater<int32_t>());
  }
  // dst's logical pages [0, n) become src's physical pages [0, n) (n <= src's held pages); dst's own pages
  // there are let go, its pages past n kept. Returns false (nothing changed) on the fixed layout or a bad n.
  bool share(int dst, int src, int n) {
    check(dst);
    check(src);
    if (!shared_ || dst == src || n < 1 || n > held_[static_cast<size_t>(src)] || n > slot_pages_) return false;
    int &hd = held_[static_cast<size_t>(dst)];
    const size_t rd = static_cast<size_t>(dst) * slot_pages_, rs = static_cast<size_t>(src) * slot_pages_;
    for (int i = 0; i < n; ++i) {
      const int32_t p = table_[rs + i];
      ++ref_[static_cast<size_t>(p)];  // first: dst may already hold p
      if (i < hd) drop(table_[rd + i]);
      table_[rd + i] = p;
    }
    if (n > hd) {  // entries past dst's old pages: unmapped before
      hd = n;
    }
    mark(rd, rd + n);
    if (!scramble_) std::sort(free_.begin(), free_.end(), std::greater<int32_t>());
    return true;
  }
  bool page_shared(int slot, int i) const {
    return shared_ && i >= 0 && i < held_.at(static_cast<size_t>(slot)) && ref_[static_cast<size_t>(table_[static_cast<size_t>(slot) * slot_pages_ + i])] > 1;
  }
  int shared_pages(int slot) const {
    int n = 0;
    for (int i = 0; i < held_.at(static_cast<size_t>(slot)); ++i) n += page_shared(slot, i);
    return n;
  }
  // copy-on-write: every shared page of the slot from logical page `from` on gets a fresh private page; returns the
  // (old, new) physical pairs the caller copies (empty: nothing shared there). Throws when the pool cannot.
  std::vector<std::pair<int32_t, int32_t>> unshare(int slot, int from) {
    check(slot);
    std::vector<std::pair<int32_t, int32_t>> moved;
    if (!shared_) return moved;
    const size_t row = static_cast<size_t>(slot) * slot_pages_;
    for (int i = std::max(from, 0); i < held_[static_cast<size_t>(slot)]; ++i) {
      const int32_t p = table_[row + i];
      if (ref_[static_cast<size_t>(p)] < 2) continue;
      if (free_.empty() || free_.back() >= limit_) throw std::runtime_error("kv pages: no free page for a copy-on-write");
      const int32_t np = free_.back();
      free_.pop_back();
      ref_[static_cast<size_t>(np)] = 1;
      --ref_[static_cast<size_t>(p)];
      table_[row + i] = np;
      mark(row + i, row + i + 1);
      moved.emplace_back(p, np);
    }
    return moved;
  }
  // the physical page of the slot's logical page i (i < held_pages(slot))
  int32_t page(int slot, int i) const { return table_.at(static_cast<size_t>(slot) * slot_pages_ + static_cast<size_t>(i)); }

  const std::vector<int32_t> &table() const { return table_; }
  // the flat table range [lo, hi) changed since clear_dirty (lo == hi: nothing)
  size_t dirty_lo() const { return dirty_lo_; }
  size_t dirty_hi() const { return dirty_hi_; }
  bool dirty() const { return dirty_hi_ > dirty_lo_; }
  void clear_dirty() { dirty_lo_ = dirty_hi_ = 0; }

 private:
  void check(int slot) const {
    if (slot < 0 || slot >= slots_) throw std::invalid_argument("kv pages: no such slot");
  }
  void drop(int32_t p) {  // one holder lets go of physical page p
    if (--ref_[static_cast<size_t>(p)] == 0) free_.push_back(p);
  }
  void mark(size_t lo, size_t hi) {
    if (!dirty()) {
      dirty_lo_ = lo;
      dirty_hi_ = hi;
    } else {
      dirty_lo_ = std::min(dirty_lo_, lo);
      dirty_hi_ = std::max(dirty_hi_, hi);
    }
  }
  int slots_ = 0, slot_pages_ = 0, pool_pages_ = 0, limit_ = 0;
  bool shared_ = false, scramble_ = false;
  std::vector<int32_t> table_, free_;
  std::vector<int32_t> ref_;  // holders per physical page (shared pools)
  std::vector<int> held_;
  size_t dirty_lo_ = 0, dirty_hi_ = 0;
};

}  // namespace xenithium:model
