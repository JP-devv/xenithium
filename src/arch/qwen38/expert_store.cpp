#include "xenithium/arch/qwen38/expert_store.hpp"

#include "xenithium/arch/qwen38/moe.hpp"

#include <algorithm>
#include <cstdio>
#include <fcntl.h>
#include <filesystem>
#include <unistd.h>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <chrono>
#include <cstring>
#include <future>
#include <stdexcept>
#include <thread>

namespace xenithium::arch::qwen38 {

namespace {

using clock_t_ = std::chrono::steady_clock;
double secs(clock_t_::time_point t0) { return std::chrono::duration<double>(clock_t_::now() - t0).count(); }

// Runs f(i) for i in [0, n) on `threads` threads; rethrows the first error.
template <typename F>
void parallel_for(int n, int threads, F &&f) {
  std::atomic<int> next{0};
  std::vector<std::string> err(static_cast<size_t>(threads));
  std::vector<std::thread> th;
  for (int t = 0; t < threads; ++t)
    th.emplace_back([&, t] {
      try {
        for (int i; (i = next.fetch_add(1)) < n;) f(i);
      } catch (const std::exception &e) {
        err[static_cast<size_t>(t)] = e.what();
        next = n;
      }
    });
  for (auto &x : th) x.join();
  for (const auto &e : err)
    if (!e.empty()) throw std::runtime_error(e);
}

std::vector<l0::Queue::CopyOp> chunked(void *dst, const void *src, size_t bytes) {
  constexpr size_t kChunk = 64ull << 20;
  std::vector<l0::Queue::CopyOp> ops;
  for (size_t off = 0; off < bytes; off += kChunk)
    ops.push_back({static_cast<uint8_t *>(dst) + off, static_cast<const uint8_t *>(src) + off, std::min(kChunk, bytes - off)});
  return ops;
}

}  // namespace

// The raw layer reads: a SeqReader over the layers; the pool first (the buffers give their blocks
// back to it, so it must outlive them).
struct ExpertStore::Reader {
  model::BufferPool pool;
  std::unique_ptr<model::SeqReader> seq;
};

void ExpertStore::start_reads(const model::Checkpoint &ck, const model::BulkOptions &read, int ahead, std::shared_future<void> gate) {
  reader_.reset();
  reader_ = std::make_unique<Reader>();
  const model::BulkOptions ropt = reader_->pool.options(read);
  const int E = experts_;
  reader_->seq = std::make_unique<model::SeqReader>(
      layers_, ahead,
      [&ck, ropt, E](int l) {
        std::vector<const model::TensorInfo *> ts;
        ts.reserve(static_cast<size_t>(E) * 9);
        for (int e = 0; e < E; ++e)
          for (const auto *t : expert_tensors(ck, l, e)) ts.push_back(t);
        return model::bulk_read(ck, ts, ropt);
      },
      std::function<void()>{}, std::move(gate));
}

size_t ExpertStore::reader_bytes() const { return reader_ ? reader_->pool.held_bytes() : 0; }

uint64_t blob_hash(const uint8_t *p, size_t n) {
  uint64_t h = 0x9E3779B97F4A7C15ull ^ n;
  size_t i = 0;
  for (; i + 8 <= n; i += 8) {
    uint64_t w;
    std::memcpy(&w, p + i, 8);
    h = (h ^ (w * 0xBF58476D1CE4E5B9ull)) * 0x94D049BB133111EBull;
    h ^= h >> 29;
  }
  for (; i < n; ++i) h = (h ^ p[i]) * 0x100000001B3ull;
  return h;
}

ExpertStore::ExpertStore(Fabric &fabric, const Config &cfg, bool tp) : fab_(fabric), cfg_(cfg), lay_(ExpertLayout::of(cfg)), tp_(tp) {
  cards_ = static_cast<int>(fab_.size());
  layers_ = cfg.layers;
  experts_ = cfg.experts;
  if (experts_ % cards_) throw std::runtime_error("expert store: experts do not split over the cards");
  per_card_ = tp_ ? experts_ : experts_ / cards_;
  stride_ = lay_.stride;
  if (tp_) tp_lay_ = kernels::moe_tp_layout(cfg.hidden, cfg.moe_inter, cards_, &stride_);
  hot_.assign(static_cast<size_t>(cards_) * layers_, 0);
  dev_.assign(hot_.size(), nullptr);
  cold_.assign(hot_.size(), nullptr);
}

ExpertStore::~ExpertStore() {
  reader_.reset();
  for (int c = 0; c < cards_; ++c) fab_.copy(c).wait();
  for (auto *p : dev_)
    if (p) fab_.release(p);
  for (auto *p : cold_)
    if (p) fab_.release(p);
}

int64_t ExpertStore::hot_total(int card) const {
  int64_t s = 0;
  for (int l = 0; l < layers_; ++l) s += hot_[idx(card, l)];
  return s;
}

const uint8_t *ExpertStore::blob(int layer, int expert, bool *hot, int card) const {
  const int c = tp_ ? card : owner(expert);
  if (hot) *hot = is_hot_[static_cast<size_t>(layer) * experts_ + expert] != 0;
  return loc_[loc_idx(c, layer, expert)];
}

void ExpertStore::relocate(int card, int layer, int expert, uint8_t *to, bool hot) {
  if (!tp_) throw std::logic_error("expert store: relocate needs the TP layout");
  loc_[loc_idx(card, layer, expert)] = to;
  is_hot_[static_cast<size_t>(layer) * experts_ + expert] = hot ? 1 : 0;
  moved_ = true;
}

void ExpertStore::place(int64_t want, const std::vector<double> *rank) {
  const size_t LE = static_cast<size_t>(layers_) * experts_;
  if (rank && rank->size() != LE) throw std::runtime_error("expert store: routing counts must be [layers][experts]");
  is_hot_.assign(LE, 0);
  // EP: each card ranks its own experts; TP: one ranking over all experts, shared by the cards.
  const int base_of = tp_ ? 0 : per_card_;
  for (int c = 0; c < (tp_ ? 1 : cards_); ++c) {
    if (!rank) {  // even split per layer, the lowest local ids
      for (int l = 0; l < layers_; ++l) {
        const int h = std::min(static_cast<int>(want / layers_ + (l < want % layers_ ? 1 : 0)), per_card_);
        for (int i = 0; i < h; ++i) is_hot_[static_cast<size_t>(l) * experts_ + c * base_of + i] = 1;
      }
      continue;
    }
    std::vector<std::pair<int, int>> pairs;  // (layer, global expert) of this card
    pairs.reserve(static_cast<size_t>(layers_) * per_card_);
    for (int l = 0; l < layers_; ++l)
      for (int i = 0; i < per_card_; ++i) pairs.emplace_back(l, c * base_of + i);
    std::stable_sort(pairs.begin(), pairs.end(), [&](const auto &a, const auto &b) {
      return (*rank)[static_cast<size_t>(a.first) * experts_ + a.second] > (*rank)[static_cast<size_t>(b.first) * experts_ + b.second];
    });
    for (int64_t j = 0; j < want && j < static_cast<int64_t>(pairs.size()); ++j)
      is_hot_[static_cast<size_t>(pairs[static_cast<size_t>(j)].first) * experts_ + pairs[static_cast<size_t>(j)].second] = 1;
  }
  slot_.assign(LE, 0);
  for (int c = 0; c < cards_; ++c)
    for (int l = 0; l < layers_; ++l) {
      int nh = 0, nc = 0;
      for (int i = 0; i < per_card_; ++i) {
        const size_t k = static_cast<size_t>(l) * experts_ + c * base_of + i;
        slot_[k] = is_hot_[k] ? nh++ : nc++;
      }
      hot_[idx(c, l)] = nh;
    }
}

void ExpertStore::load(const model::Checkpoint &ck, const ExpertStoreOptions &opt) {
  const auto t_all = clock_t_::now();
  stats_ = {};
  const size_t S = stride_;

  // ---- placement (policy: see header) ----
  const int64_t want = opt.hot_per_card < 0 ? static_cast<int64_t>(per_card_) * layers_
                                            : std::min<int64_t>(opt.hot_per_card, static_cast<int64_t>(per_card_) * layers_);
  place(want, opt.rank);
  int max_hot = 0;
  for (const int h : hot_) max_hot = std::max(max_hot, h);

  const auto t_alloc = clock_t_::now();
  // ---- memory: hot per (card, layer) in VRAM, cold in pinned host, 2 staging slots per card ----
  for (int c = 0; c < cards_; ++c)
    for (int l = 0; l < layers_; ++l) {
      const int h = hot_[idx(c, l)], cold = per_card_ - h;
      if (h) dev_[idx(c, l)] = static_cast<uint8_t *>(fab_.alloc(c, static_cast<size_t>(h) * S));
      if (cold) {
        cold_[idx(c, l)] = static_cast<uint8_t *>(fab_.alloc_host(static_cast<size_t>(cold) * S));
        stats_.pinned_bytes += static_cast<uint64_t>(cold) * S;
      }
    }
  std::vector<uint8_t *> stage(static_cast<size_t>(cards_) * 2, nullptr);
  if (max_hot)
    for (auto &p : stage) {
      p = static_cast<uint8_t *>(fab_.alloc_host(static_cast<size_t>(max_hot) * S));
      stats_.pinned_bytes += static_cast<uint64_t>(max_hot) * S;
    }
  stats_.alloc_s = secs(t_alloc);
  // where every blob lives (EP: on its owner card only)
  loc_.assign(static_cast<size_t>(cards_) * layers_ * experts_, nullptr);
  moved_ = false;
  for (int c = 0; c < cards_; ++c)
    for (int l = 0; l < layers_; ++l)
      for (int e = 0; e < experts_; ++e) {
        if (!tp_ && owner(e) != c) continue;
        const size_t k = static_cast<size_t>(l) * experts_ + e;
        loc_[loc_idx(c, l, e)] = (is_hot_[k] ? dev_ : cold_)[idx(c, l)] + static_cast<size_t>(slot_[k]) * S;
      }
  if (opt.hash) hash_.assign(static_cast<size_t>(layers_) * experts_ * (tp_ ? cards_ : 1), 0);


  std::vector<int> inflight(static_cast<size_t>(cards_), -1);  // staging slot of the copy in flight
  if (!opt.native_dir.empty()) {
    if (!tp_ || cards_ != 2) throw std::runtime_error("expert store: native blobs are TP over 2 cards");
    stats_.native = true;
    reader_.reset();
    load_native(opt, stage, inflight);
  } else
  try {
    // the reader (started early by start_reads, else here); its buffers are reused across layers
    if (!reader_) start_reads(ck, opt.read, opt.read_ahead);
    for (int l = 0; l < layers_; ++l) {
      auto t0 = clock_t_::now();
      model::BulkBuffer buf = reader_->seq->take(l);
      stats_.read_wait_s += secs(t0);
      if (l == 0) stats_.first_read_s = secs(t0);
      stats_.read_bytes += buf.stats().read_bytes;

      const int s = l % 2;
      t0 = clock_t_::now();
      for (int c = 0; c < cards_; ++c)
        if (inflight[c] == s) {  // only when the previous layer submitted nothing for this card
          fab_.copy(c).wait();
          inflight[c] = -1;
        }
      stats_.h2d_wait_s += secs(t0);

      t0 = clock_t_::now();
      parallel_for(experts_, opt.pack_threads, [&](int e) {
        const size_t k = static_cast<size_t>(l) * experts_ + e;
        auto dst_of = [&](int c) { return (is_hot_[k] ? stage[static_cast<size_t>(c) * 2 + s] : cold_[idx(c, l)]) + static_cast<size_t>(slot_[k]) * S; };
        if (!tp_) {
          uint8_t *dst = dst_of(owner(e));
          pack_expert(lay_, ck, buf, l, e, dst);
          if (opt.hash) hash_[k] = blob_hash(dst, S);
          return;
        }
        // TP: pack the full blob once, then cut each card's columns out of it. The packing is
        // memory-bound beside the drive's DMA, so it moves only the bytes it must: `full` is zeroed once per
        // thread (pack_expert writes the same bytes of it for every expert; it stays in L3) and a card's
        // blob only past the split's content (moe_split_tp writes every byte before it).
        thread_local std::vector<uint8_t> full;
        if (full.size() != lay_.stride) full.assign(lay_.stride, 0);
        pack_expert(lay_, ck, buf, l, e, full.data());
        const auto fl = moe_blob_layout(lay_);
        const size_t content = tp_lay_.dn_scales + static_cast<size_t>(kernels::moe_tp_scale_rows(cfg_.moe_inter, cards_)) * cfg_.hidden * 2;
        for (int c = 0; c < cards_; ++c) {
          uint8_t *dst = dst_of(c);
          if (content < S) std::memset(dst + content, 0, S - content);
          kernels::moe_split_tp(fl, cfg_.hidden, cfg_.moe_inter, cards_, c, full.data(), tp_lay_, dst);
          if (opt.hash) hash_[k * cards_ + c] = blob_hash(dst, S);
        }
      });
      stats_.pack_s += secs(t0);

      t0 = clock_t_::now();
      for (int c = 0; c < cards_; ++c) {
        const int h = hot_[idx(c, l)];
        if (!h) continue;
        // copy_batch first waits for this card's previous batch (the other staging slot).
        fab_.copy(c).copy_batch(chunked(dev_[idx(c, l)], stage[static_cast<size_t>(c) * 2 + s], static_cast<size_t>(h) * S), false);
        inflight[c] = s;
        stats_.h2d_bytes += static_cast<uint64_t>(h) * S;
      }
      stats_.h2d_wait_s += secs(t0);
      t0 = clock_t_::now();
      { model::BulkBuffer gone = std::move(buf); }  // back to the pool
      stats_.free_s += secs(t0);
    }
    const auto t0 = clock_t_::now();
    for (int c = 0; c < cards_; ++c) fab_.copy(c).wait();
    stats_.h2d_wait_s += secs(t0);
    reader_.reset();  // its buffers back to the system
  } catch (...) {
    reader_.reset();
    for (int c = 0; c < cards_; ++c) fab_.copy(c).wait();
    for (auto *p : stage)
      if (p) fab_.release(p);
    throw;
  }
  for (auto *p : stage)
    if (p) fab_.release(p);
  stats_.pinned_bytes -= max_hot ? static_cast<uint64_t>(max_hot) * S * stage.size() : 0;  // staging is gone
  stats_.total_s = secs(t_all);
}

// The bake's per-card TP blobs, read straight into their destinations: each run of experts
// whose slots are consecutive in one place (staging for hot, the pinned cold region for cold) is one
// range of the file (the even split gives two runs per card and layer), cut into pieces. One queue
// of pieces over every layer keeps the drive busy across layer boundaries: a reader thread takes the
// next piece; a staging piece of layer l first waits until staging slot l % 2 is free (the H2D of
// layer l - 2 done), cold pieces never wait. O_DIRECT (buffered where refused).
void ExpertStore::load_native(const ExpertStoreOptions &opt, std::vector<uint8_t *> &stage, std::vector<int> &inflight) {
  const size_t S = stride_, region = S * static_cast<size_t>(experts_), file_bytes = region * 2;
  const int rthreads = std::max(1, opt.read.threads * 2);
  const uint64_t piece = 16ull << 20;  // larger pieces would leave the readers idle at each layer's tail
  auto path_of = [&](int l) {
    char n[32];
    std::snprintf(n, sizeof n, "/experts-L%02d.tp2", l);
    return opt.native_dir + n;
  };
  std::vector<int> fd_d(static_cast<size_t>(layers_), -1), fd_b(static_cast<size_t>(layers_), -1);
  auto close_all = [&] {
    for (int &f : fd_d)
      if (f >= 0) ::close(f), f = -1;
    for (int &f : fd_b)
      if (f >= 0) ::close(f), f = -1;
  };
  for (int l = 0; l < layers_; ++l) {
    std::error_code ec;
    const std::string path = path_of(l);
    const auto sz = std::filesystem::file_size(path, ec);
    if (ec || sz != file_bytes) {
      close_all();
      throw std::runtime_error("expert store: " + path + (ec ? " is missing" : " has " + std::to_string(sz) + " bytes, want " + std::to_string(file_bytes)));
    }
    fd_b[static_cast<size_t>(l)] = ::open(path.c_str(), O_RDONLY);
    if (opt.read.direct) fd_d[static_cast<size_t>(l)] = ::open(path.c_str(), O_RDONLY | O_DIRECT);
    if (fd_b[static_cast<size_t>(l)] < 0) {
      close_all();
      throw std::runtime_error("expert store: cannot open " + path);
    }
  }
  struct Job {
    uint8_t *dst;
    uint64_t off, len;
    int layer;
    bool staged;
  };
  std::vector<Job> jobs;
  for (int l = 0; l < layers_; ++l)
    for (int c = 0; c < cards_; ++c)
      for (int e = 0; e < experts_;) {
        const size_t k = static_cast<size_t>(l) * experts_ + e;
        const bool h = is_hot_[k] != 0;
        int n = 1;
        while (e + n < experts_ && (is_hot_[k + n] != 0) == h && slot_[k + n] == slot_[k] + n) ++n;
        uint8_t *dst = (h ? stage[static_cast<size_t>(c) * 2 + l % 2] : cold_[idx(c, l)]) + static_cast<size_t>(slot_[k]) * S;
        const uint64_t off = static_cast<uint64_t>(c) * region + static_cast<uint64_t>(e) * S, len = static_cast<uint64_t>(n) * S;
        for (uint64_t o = 0; o < len; o += piece) jobs.push_back({dst + o, off + o, std::min(piece, len - o), l, h});
        e += n;
      }
  std::mutex mu;
  std::condition_variable cv_slot, cv_done;
  int freed_upto = 1;  // staging pieces of layer l may start once l <= freed_upto (layers 0, 1: free)
  std::vector<int> left(static_cast<size_t>(layers_), 0);
  for (const Job &j : jobs) ++left[static_cast<size_t>(j.layer)];
  bool bad = false, stop = false;
  std::atomic<size_t> next{0};
  auto reader = [&] {
    for (size_t i; (i = next.fetch_add(1)) < jobs.size();) {
      const Job &J = jobs[i];
      if (J.staged) {
        std::unique_lock<std::mutex> lk(mu);
        cv_slot.wait(lk, [&] { return stop || J.layer <= freed_upto; });
        if (stop) return;
      }
      const int fdd = fd_d[static_cast<size_t>(J.layer)];
      int fd = fdd >= 0 && (reinterpret_cast<uintptr_t>(J.dst) | J.off | J.len) % 4096 == 0 ? fdd : fd_b[static_cast<size_t>(J.layer)];
      uint64_t done = 0;
      bool ok = true;
      while (done < J.len) {
        const ssize_t r = ::pread(fd, J.dst + done, J.len - done, static_cast<off_t>(J.off + done));
        if (r < 0 && fd == fdd) {  // O_DIRECT refused here: buffered
          fd = fd_b[static_cast<size_t>(J.layer)];
          continue;
        }
        if (r <= 0) {
          ok = false;
          break;
        }
        done += static_cast<uint64_t>(r);
      }
      std::lock_guard<std::mutex> lk(mu);
      if (!ok) bad = true;
      if (--left[static_cast<size_t>(J.layer)] == 0 || !ok) cv_done.notify_all();
    }
  };
  std::vector<std::thread> pool;
  auto shutdown = [&] {
    {
      std::lock_guard<std::mutex> lk(mu);
      stop = true;
    }
    cv_slot.notify_all();
    for (auto &t : pool) t.join();
    pool.clear();
    close_all();
  };
  try {
    for (int t = 0; t < rthreads; ++t) pool.emplace_back(reader);
    for (int l = 0; l < layers_; ++l) {
      const int s = l % 2;
      auto t0 = clock_t_::now();
      {
        std::unique_lock<std::mutex> lk(mu);
        cv_done.wait(lk, [&] { return bad || left[static_cast<size_t>(l)] == 0; });
        if (bad) throw std::runtime_error("expert store: short read of " + path_of(l));
      }
      const double rs = secs(t0);
      stats_.read_wait_s += rs;
      if (l == 0) stats_.first_read_s = rs;
      stats_.read_bytes += file_bytes;
      if (opt.hash) {
        t0 = clock_t_::now();
        parallel_for(experts_, opt.pack_threads, [&](int e) {
          const size_t k = static_cast<size_t>(l) * experts_ + e;
          for (int c = 0; c < cards_; ++c)
            hash_[k * cards_ + c] = blob_hash((is_hot_[k] ? stage[static_cast<size_t>(c) * 2 + s] : cold_[idx(c, l)]) + static_cast<size_t>(slot_[k]) * S, S);
        });
        stats_.pack_s += secs(t0);
      }
      t0 = clock_t_::now();
      for (int c = 0; c < cards_; ++c) {
        const int h = hot_[idx(c, l)];
        if (!h) continue;
        // copy_batch first waits for this card's previous batch (layer l - 1, the other slot)
        fab_.copy(c).copy_batch(chunked(dev_[idx(c, l)], stage[static_cast<size_t>(c) * 2 + s], static_cast<size_t>(h) * S), false);
        inflight[c] = s;
        stats_.h2d_bytes += static_cast<uint64_t>(h) * S;
      }
      stats_.h2d_wait_s += secs(t0);
      {  // the copies of layer l - 1 are done (copy_batch waited for them): slot (l + 1) % 2 is free
        std::lock_guard<std::mutex> lk(mu);
        freed_upto = l + 1;
      }
      cv_slot.notify_all();
    }
    const auto t0 = clock_t_::now();
    for (int c = 0; c < cards_; ++c) fab_.copy(c).wait();
    stats_.h2d_wait_s += secs(t0);
    shutdown();
  } catch (...) {
    shutdown();
    for (int c = 0; c < cards_; ++c) fab_.copy(c).wait();
    for (auto *p : stage)
      if (p) fab_.release(p);
    stage.assign(stage.size(), nullptr);
    throw;
  }
}

int64_t ExpertStore::verify(int threads) const {
  if (hash_.empty()) throw std::runtime_error("expert store: verify() needs load(..., hash = true)");
  if (moved_) throw std::logic_error("expert store: verify() after the live exchange moved blobs");
  const size_t S = stride_;
  std::atomic<int64_t> bad{0};
  std::vector<uint8_t> back;
  for (int c = 0; c < cards_; ++c)
    for (int l = 0; l < layers_; ++l) {
      const int h = hot_[idx(c, l)], cold = per_card_ - h;
      if (h) {
        back.resize(static_cast<size_t>(h) * S);
        fab_.copy(c).copy_batch(chunked(back.data(), dev_[idx(c, l)], back.size()), true);
      }
      parallel_for(per_card_, threads, [&](int local) {
        const int e = tp_ ? local : c * per_card_ + local;
        const size_t k = static_cast<size_t>(l) * experts_ + e;
        const uint8_t *p = (is_hot_[k] ? back.data() : cold_[idx(c, l)]) + static_cast<size_t>(slot_[k]) * S;
        if (blob_hash(p, S) != hash_[tp_ ? k * cards_ + c : k]) bad.fetch_add(1);
      });
      (void)cold;
    }
  return bad.load();
}

}  // namespace xenithium::arch::qwen38
