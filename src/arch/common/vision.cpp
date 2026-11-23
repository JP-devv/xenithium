// Xenithium · image input on the cards (xenithium/arch/common/vision.hpp): the vision tower and the store of image
// embeddings the prompts name.
#include "xenithium/arch/common/vision.hpp"

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <stdexcept>

#include "xenithium/kernels/encoder/vit.hpp"
#include "xenithium/kernels/pack.hpp"
#include "xenithium/model/checkpoint.hpp"

namespace xenithium::arch {

namespace k = xenithium::kernels;
using bf16 = k::vit_bf16;

namespace {

constexpr float kLnEps = 1e-6f;

// parameters (elements) of one block / the embedding side / the merger side
int64_t block_params(const model::VisionConfig &v) {
  const int64_t H = v.hidden, I = v.inter;
  return 3 * H * H + 3 * H + H * H + H + I * H + I + H * I + H + 4 * H;
}
int64_t embed_params(const model::VisionConfig &v) { return static_cast<int64_t>(v.hidden) * v.patch_dim() + v.hidden + static_cast<int64_t>(v.num_pos) * v.hidden; }
int64_t merger_params(const model::VisionConfig &v) {
  const int64_t M = v.merged_dim();
  return 2 * static_cast<int64_t>(v.hidden) + M * M + M + static_cast<int64_t>(v.out_hidden) * M + v.out_hidden;
}
// blocks on card 0 when the tower spans two cards: the split with the closest byte totals
int split_of(const model::VisionConfig &v, int cards) {
  if (cards < 2) return v.depth;
  int best = v.depth / 2;
  int64_t bd = INT64_MAX;
  for (int s = 1; s < v.depth; ++s) {
    const int64_t a = embed_params(v) + s * block_params(v), b = (v.depth - s) * block_params(v) + merger_params(v);
    if (std::llabs(a - b) < bd) bd = std::llabs(a - b), best = s;
  }
  return best;
}
size_t scratch_bytes(const model::VisionConfig &v, int N, bool first, bool last) {
  const size_t n = static_cast<size_t>(N), H = static_cast<size_t>(v.hidden);
  size_t b = (2 + 1 + 3 + 1) * n * H * 2 + n * static_cast<size_t>(v.inter) * 2 + 2 * n * static_cast<size_t>(v.head_dim() / 2) * 4;
  if (first) b += n * static_cast<size_t>(v.patch_dim()) * 2 + n * 4 * 4 + n * 4 * 2;
  if (last) b += n / 4 * static_cast<size_t>(v.merged_dim()) * 2;
  return b + 64 * 4096;  // per-allocation alignment slack
}

void read_tensor(const model::Checkpoint &ck, const std::string &name, std::vector<uint16_t> &out, int64_t expect) {
  const model::TensorInfo &t = ck.at(name);
  if (t.dtype != model::DType::bf16) throw std::runtime_error("vision tower: " + name + " is " + t.dtype_str + " (bf16 expected)");
  if (t.numel() != expect) throw std::runtime_error("vision tower: " + name + " has " + std::to_string(t.numel()) + " elements, " + std::to_string(expect) + " expected");
  out.resize(static_cast<size_t>(t.numel()));
  const int fd = ::open(ck.shards()[t.shard].path.c_str(), O_RDONLY);
  if (fd < 0) throw std::runtime_error("vision tower: cannot open " + ck.shards()[t.shard].path);
  size_t done = 0;
  while (done < t.bytes) {
    const ssize_t r = ::pread(fd, reinterpret_cast<char *>(out.data()) + done, t.bytes - done, static_cast<off_t>(t.offset + done));
    if (r <= 0) {
      ::close(fd);
      throw std::runtime_error("vision tower: read failed for " + name);
    }
    done += static_cast<size_t>(r);
  }
  ::close(fd);
}

uint64_t fnv1a(const void *p, size_t n, uint64_t h) {
  const auto *b = static_cast<const uint8_t *>(p);
  for (size_t i = 0; i < n; ++i) h = (h ^ b[i]) * 1099511628211ull;
  return h;
}

}  // namespace

struct VisionTower::Block {
  bf16 *qkv_w = nullptr, *qkv_b = nullptr, *proj_w = nullptr, *proj_b = nullptr, *fc1_w = nullptr, *fc1_b = nullptr, *fc2_w = nullptr, *fc2_b = nullptr;
  float *n1g = nullptr, *n1b = nullptr, *n2g = nullptr, *n2b = nullptr;
};

struct VisionTower::Card {
  int c = 0;
  int b0 = 0, b1 = 0;  // blocks [b0, b1)
  bool first = false, last = false;
  std::vector<Block> blocks;
  bf16 *patch_w = nullptr, *patch_b = nullptr, *pos = nullptr;                                 // first
  float *mn_g = nullptr, *mn_b = nullptr;                                                       // last
  bf16 *m1_w = nullptr, *m1_b = nullptr, *m2_w = nullptr, *m2_b = nullptr;                      // last
  bf16 *x[2] = {}, *xn = nullptr, *qkv = nullptr, *o = nullptr, *h = nullptr;                    // scratch
  bf16 *patches = nullptr, *pw = nullptr, *mrg = nullptr;
  int32_t *idx = nullptr;
  float *cs = nullptr, *sn = nullptr;
};

size_t VisionTower::bytes_on(const model::VisionConfig &v, int max_patches, int cards, int c) {
  cards = std::min(cards, kMaxCards);
  if (c >= cards) return 0;
  const int split = split_of(v, cards);
  const bool first = c == 0, last = c == cards - 1;
  int64_t params = 0;
  if (first) params += embed_params(v);
  if (last) params += merger_params(v);
  const int nb = cards < 2 ? v.depth : (first ? split : v.depth - split);
  params += nb * block_params(v);
  return static_cast<size_t>(params) * 2 + scratch_bytes(v, max_patches, first, last) + 16 * 4096 * static_cast<size_t>(nb + 2);
}

size_t VisionTower::bytes(int c) const { return bytes_on(vc_, max_patches_, static_cast<int>(cards_.size()), c); }

VisionTower::VisionTower(Fabric &fab, const model::Checkpoint &ck, const model::VisionConfig &vc, int max_patches) : fab_(fab), vc_(vc), max_patches_(max_patches) {
  if (!vc.present) throw std::invalid_argument("vision tower: the checkpoint has no vision_config");
  const int ncards = std::min(static_cast<int>(fab.size()), kMaxCards);
  split_ = split_of(vc, ncards);
  const int H = vc.hidden, I = vc.inter, M = vc.merged_dim(), N = max_patches;
  std::vector<uint16_t> buf;
  auto dev = [&](int c, size_t bytes) {
    void *p = fab_.alloc(static_cast<size_t>(c), std::max<size_t>(bytes, 4096));
    if (!p) throw std::runtime_error("vision tower: out of device memory");
    owned_.push_back(p);
    return p;
  };
  auto up = [&](int c, const std::string &name, int64_t n) {
    read_tensor(ck, "model.visual." + name, buf, n);
    auto *p = static_cast<bf16 *>(dev(c, buf.size() * 2));
    fab_.compute(static_cast<size_t>(c)).memcpy(p, buf.data(), buf.size() * 2).wait();
    return p;
  };
  auto up_f32 = [&](int c, const std::string &name, int64_t n) {
    read_tensor(ck, "model.visual." + name, buf, n);
    std::vector<float> f(buf.size());
    for (size_t i = 0; i < buf.size(); ++i) f[i] = k::bf16_value(buf[i]);
    auto *p = static_cast<float *>(dev(c, f.size() * 4));
    fab_.compute(static_cast<size_t>(c)).memcpy(p, f.data(), f.size() * 4).wait();
    return p;
  };
  for (int c = 0; c < ncards; ++c) {
    auto C = std::make_unique<Card>();
    C->c = c;
    C->first = c == 0, C->last = c == ncards - 1;
    C->b0 = c == 0 ? 0 : split_, C->b1 = c == ncards - 1 ? vc.depth : split_;
    if (C->first) {
      C->patch_w = up(c, "patch_embed.proj.weight", static_cast<int64_t>(H) * vc.patch_dim());
      C->patch_b = up(c, "patch_embed.proj.bias", H);
      C->pos = up(c, "pos_embed.weight", static_cast<int64_t>(vc.num_pos) * H);
    }
    for (int b = C->b0; b < C->b1; ++b) {
      const std::string p = "blocks." + std::to_string(b) + ".";
      Block B;
      B.n1g = up_f32(c, p + "norm1.weight", H), B.n1b = up_f32(c, p + "norm1.bias", H);
      B.qkv_w = up(c, p + "attn.qkv.weight", static_cast<int64_t>(3) * H * H), B.qkv_b = up(c, p + "attn.qkv.bias", 3 * H);
      B.proj_w = up(c, p + "attn.proj.weight", static_cast<int64_t>(H) * H), B.proj_b = up(c, p + "attn.proj.bias", H);
      B.n2g = up_f32(c, p + "norm2.weight", H), B.n2b = up_f32(c, p + "norm2.bias", H);
      B.fc1_w = up(c, p + "mlp.linear_fc1.weight", static_cast<int64_t>(I) * H), B.fc1_b = up(c, p + "mlp.linear_fc1.bias", I);
      B.fc2_w = up(c, p + "mlp.linear_fc2.weight", static_cast<int64_t>(H) * I), B.fc2_b = up(c, p + "mlp.linear_fc2.bias", H);
      C->blocks.push_back(B);
    }
    if (C->last) {
      C->mn_g = up_f32(c, "merger.norm.weight", H), C->mn_b = up_f32(c, "merger.norm.bias", H);
      C->m1_w = up(c, "merger.linear_fc1.weight", static_cast<int64_t>(M) * M), C->m1_b = up(c, "merger.linear_fc1.bias", M);
      C->m2_w = up(c, "merger.linear_fc2.weight", static_cast<int64_t>(vc.out_hidden) * M), C->m2_b = up(c, "merger.linear_fc2.bias", vc.out_hidden);
      C->mrg = static_cast<bf16 *>(dev(c, static_cast<size_t>(N / 4) * M * 2));
    }
    const size_t n = static_cast<size_t>(N);
    for (auto &x : C->x) x = static_cast<bf16 *>(dev(c, n * H * 2));
    C->xn = static_cast<bf16 *>(dev(c, n * H * 2));
    C->qkv = static_cast<bf16 *>(dev(c, n * 3 * H * 2));
    C->o = static_cast<bf16 *>(dev(c, n * H * 2));
    C->h = static_cast<bf16 *>(dev(c, n * I * 2));
    C->cs = static_cast<float *>(dev(c, n * (vc.head_dim() / 2) * 4));
    C->sn = static_cast<float *>(dev(c, n * (vc.head_dim() / 2) * 4));
    if (C->first) {
      C->patches = static_cast<bf16 *>(dev(c, n * vc.patch_dim() * 2));
      C->idx = static_cast<int32_t *>(dev(c, n * 4 * 4));
      C->pw = static_cast<bf16 *>(dev(c, n * 4 * 2));
    }
    cards_.push_back(std::move(C));
  }
  host_ = fab_.alloc_host(std::max(static_cast<size_t>(N) * H * 2, static_cast<size_t>(N / 4) * vc.out_hidden * 2) + 4096);
  if (!host_) throw std::runtime_error("vision tower: out of pinned host memory");
}

VisionTower::~VisionTower() {
  for (const auto &C : cards_) fab_.compute(static_cast<size_t>(C->c)).wait();
  for (void *p : owned_) fab_.release(p);
  if (host_) fab_.release(host_);
}

void VisionTower::encode(const model::VisionInput &in, const std::vector<void *> &dst) {
  const auto t0 = std::chrono::steady_clock::now();
  const int N = in.patches_n(), H = vc_.hidden, I = vc_.inter, M = vc_.merged_dim(), hd = vc_.head_dim(), heads = vc_.heads;
  if (N > max_patches_ || N % (vc_.merge * vc_.merge)) throw std::invalid_argument("vision tower: " + std::to_string(N) + " patches (at most " + std::to_string(max_patches_) + ")");
  const size_t half = static_cast<size_t>(hd / 2);
  int cur = 0;
  for (auto &Cp : cards_) {
    Card &C = *Cp;
    sycl::queue &q = fab_.compute(static_cast<size_t>(C.c));
    q.memcpy(C.cs, in.rope_cos.data(), static_cast<size_t>(N) * half * 4);
    q.memcpy(C.sn, in.rope_sin.data(), static_cast<size_t>(N) * half * 4);
    if (C.first) {
      q.memcpy(C.patches, in.patches.data(), in.patches.size() * 2);
      q.memcpy(C.idx, in.pos_idx.data(), in.pos_idx.size() * 4);
      q.memcpy(C.pw, in.pos_w.data(), in.pos_w.size() * 2);
      k::vit_linear(q, N, H, vc_.patch_dim(), C.patches, vc_.patch_dim(), C.patch_w, C.patch_b, k::VitAct::none, nullptr, C.x[0], H);
      k::vit_pos_add(q, N, H, C.pos, C.idx, C.pw, C.x[0]);
      cur = 0;
    } else {  // the hidden state from the previous card, through pinned host memory
      sycl::queue &qp = fab_.compute(static_cast<size_t>(C.c - 1));
      qp.memcpy(host_, cards_[static_cast<size_t>(C.c - 1)]->x[cur], static_cast<size_t>(N) * H * 2).wait();
      q.memcpy(C.x[0], host_, static_cast<size_t>(N) * H * 2);
      cur = 0;
    }
    for (const Block &B : C.blocks) {
      bf16 *x = C.x[cur], *y = C.x[cur ^ 1];
      k::vit_layer_norm(q, N, H, x, B.n1g, B.n1b, kLnEps, C.xn);
      k::vit_linear(q, N, 3 * H, H, C.xn, H, B.qkv_w, B.qkv_b, k::VitAct::none, nullptr, C.qkv, 3 * H);
      k::vit_rope(q, N, heads, hd, C.cs, C.sn, C.qkv);
      k::vit_attention(q, N, heads, hd, C.qkv, C.o);
      k::vit_linear(q, N, H, H, C.o, H, B.proj_w, B.proj_b, k::VitAct::none, x, y, H);
      k::vit_layer_norm(q, N, H, y, B.n2g, B.n2b, kLnEps, C.xn);
      k::vit_linear(q, N, I, H, C.xn, H, B.fc1_w, B.fc1_b, k::VitAct::gelu_tanh, nullptr, C.h, I);
      k::vit_linear(q, N, H, I, C.h, I, B.fc2_w, B.fc2_b, k::VitAct::none, y, x, H);
    }
    if (C.last) {
      const int T = N / (vc_.merge * vc_.merge);
      k::vit_layer_norm(q, N, H, C.x[cur], C.mn_g, C.mn_b, kLnEps, C.xn);  // per patch, then [T][merge^2 x hidden]
      k::vit_linear(q, T, M, M, C.xn, M, C.m1_w, C.m1_b, k::VitAct::gelu_erf, nullptr, C.mrg, M);
      k::vit_linear(q, T, vc_.out_hidden, M, C.mrg, M, C.m2_w, C.m2_b, k::VitAct::none, nullptr, static_cast<bf16 *>(dst[static_cast<size_t>(C.c)]), vc_.out_hidden);
      const size_t bytes = static_cast<size_t>(T) * vc_.out_hidden * 2;
      if (cards_.size() > 1 || dst.size() > 1) q.memcpy(host_, dst[static_cast<size_t>(C.c)], bytes).wait();
      for (size_t c = 0; c < dst.size(); ++c)
        if (static_cast<int>(c) != C.c) fab_.compute(c).memcpy(dst[c], host_, bytes);
    }
  }
  for (size_t c = 0; c < dst.size(); ++c) fab_.compute(c).wait();
  last_ms_ = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

// ---- the store ----------------------------------------------------------------------------------------------------

VisionStore::Lease::~Lease() {
  if (store) store->release(key);
}

VisionStore::VisionStore(Fabric &fab, std::unique_ptr<VisionTower> tower, const model::VisionConfig &vc, int64_t vocab, int64_t rows)
    : fab_(fab), tower_(std::move(tower)), vc_(vc), vocab_(vocab), rows_(rows) {
  next_base_ = vocab;
  for (size_t c = 0; c < fab.size(); ++c) {  // every card's embedding gather reads the rows
    void *p = fab_.alloc(c, pool_bytes(vc, rows));
    if (!p) {
      for (void *q : pool_) fab_.release(q);  // the destructor does not run for a constructor that throws
      throw std::runtime_error("vision store: out of device memory for the image rows on card " + std::to_string(c));
    }
    pool_.push_back(p);
  }
}

VisionStore::~VisionStore() {
  for (size_t c = 0; c < pool_.size(); ++c) fab_.compute(c).wait();
  for (void *p : pool_) fab_.release(p);
}

int64_t VisionStore::alloc_rows(int tokens) {
  // first fit over the gaps between held images (by_key_ rows, sorted)
  std::vector<std::pair<int64_t, int>> used;
  for (const auto &kv : by_key_) used.push_back({kv.second.row, kv.second.tokens});
  std::sort(used.begin(), used.end());
  int64_t at = 0;
  for (const auto &u : used) {
    if (u.first - at >= tokens) return at;
    at = std::max(at, u.first + u.second);
  }
  return rows_ - at >= tokens ? at : -1;
}

std::shared_ptr<VisionStore::Lease> VisionStore::add(model::VisionInput &&in, std::string &err) {
  const int tokens = in.tokens(vc_.merge);
  uint64_t key = fnv1a(&in.gh, sizeof(in.gh), 1469598103934665603ull);
  key = fnv1a(&in.gw, sizeof(in.gw), key);
  key = fnv1a(&in.video, sizeof(in.video), key);
  key = fnv1a(in.patches.data(), in.patches.size() * 2, key);
  std::lock_guard<std::mutex> lk(mu_);
  ++stats_.images;
  auto it = by_key_.find(key);
  if (it != by_key_.end()) {
    ++stats_.hits;
  } else {
    if (tokens > rows_) {
      err = "the image needs " + std::to_string(tokens) + " tokens, more than the server's image rows (" + std::to_string(rows_) + ")";
      return nullptr;
    }
    int64_t row = alloc_rows(tokens);
    while (row < 0) {  // evict the least recently used image no request holds
      auto victim = by_key_.end();
      for (auto e = by_key_.begin(); e != by_key_.end(); ++e)
        if (e->second.refs == 0 && (victim == by_key_.end() || e->second.used < victim->second.used)) victim = e;
      if (victim == by_key_.end()) {
        err = "the server's image rows are all held by running requests; retry later";
        return nullptr;
      }
      by_base_.erase(victim->second.base);
      by_key_.erase(victim);
      ++stats_.evicted;
      row = alloc_rows(tokens);
    }
    if (next_base_ + tokens > INT32_MAX) {
      err = "the server ran out of image token ids (restart it)";
      return nullptr;
    }
    Entry e;
    e.base = next_base_, e.row = row, e.tokens = tokens;
    e.gh = in.gh / vc_.merge, e.gw = in.gw / vc_.merge;
    e.video = in.video;
    e.input = std::make_unique<model::VisionInput>(std::move(in));
    next_base_ += tokens;
    by_base_[e.base] = key;
    it = by_key_.emplace(key, std::move(e)).first;
  }
  Entry &e = it->second;
  ++e.refs;
  e.used = ++tick_;
  auto lease = std::make_shared<Lease>();
  lease->base = e.base, lease->tokens = e.tokens, lease->store = this, lease->key = key;
  return lease;
}

int32_t VisionStore::pad_token(int64_t id) const {
  if (id < vocab_) return static_cast<int32_t>(id);
  std::lock_guard<std::mutex> lk(mu_);
  auto b = by_base_.upper_bound(id);
  if (b == by_base_.begin()) return vc_.image_token;
  --b;
  const Entry &e = by_key_.at(b->second);
  return e.video && id < e.base + e.tokens && vc_.video_token >= 0 ? vc_.video_token : vc_.image_token;
}

bool VisionStore::image(int64_t id, model::MropeImage &out) const {
  if (id < vocab_) return false;
  std::lock_guard<std::mutex> lk(mu_);
  auto b = by_base_.upper_bound(id);
  if (b == by_base_.begin()) return false;
  --b;
  const Entry &e = by_key_.at(b->second);
  if (id >= e.base + e.tokens) return false;
  out = model::MropeImage{e.base, e.tokens, e.gh, e.gw};
  return true;
}

void VisionStore::release(uint64_t key) {
  std::lock_guard<std::mutex> lk(mu_);
  auto it = by_key_.find(key);
  if (it != by_key_.end() && it->second.refs > 0) --it->second.refs;
}

void VisionStore::prepare(const int32_t *tokens, int m, int32_t *out) {
  std::vector<uint64_t> todo;
  {
    std::lock_guard<std::mutex> lk(mu_);
    for (int i = 0; i < m; ++i) {
      if (tokens[i] < vocab_) continue;
      auto b = by_base_.upper_bound(tokens[i]);
      if (b == by_base_.begin()) continue;
      --b;
      Entry &e = by_key_.at(b->second);
      if (tokens[i] < e.base + e.tokens && !e.encoded && std::find(todo.begin(), todo.end(), b->second) == todo.end()) {
        todo.push_back(b->second);
        ++e.refs;  // held while it encodes outside the lock
      }
    }
  }
  for (const uint64_t key : todo) {
    model::VisionInput *in;
    int64_t row;
    {
      std::lock_guard<std::mutex> lk(mu_);
      Entry &e = by_key_.at(key);
      in = e.input.get(), row = e.row;
    }
    std::vector<void *> dst;
    for (size_t c = 0; c < pool_.size(); ++c) dst.push_back(static_cast<char *>(pool_[c]) + static_cast<size_t>(row) * vc_.out_hidden * 2);
    tower_->encode(*in, dst);
    std::lock_guard<std::mutex> lk(mu_);
    Entry &e = by_key_.at(key);
    e.encoded = true;
    e.input.reset();
    --e.refs;
    ++stats_.encoded;
    stats_.encode_ms += tower_->last_ms();
  }
  std::lock_guard<std::mutex> lk(mu_);
  for (int i = 0; i < m; ++i) {
    const int32_t t = tokens[i];
    if (t < vocab_) {
      out[i] = t;
      continue;
    }
    auto b = by_base_.upper_bound(t);
    const Entry *e = nullptr;
    if (b != by_base_.begin()) {
      --b;
      const Entry &x = by_key_.at(b->second);
      if (t < x.base + x.tokens && x.encoded) e = &x;
    }
    if (!e) {
      if (!warned_) std::fprintf(stderr, "vision: token %d names no held image: fed as the image token (warned once)\n", t), warned_ = true;
      out[i] = vc_.image_token;
      continue;
    }
    out[i] = static_cast<int32_t>(vocab_ + e->row + (t - e->base));
  }
}

VisionStore::Stats VisionStore::stats() const {
  std::lock_guard<std::mutex> lk(mu_);
  return stats_;
}

}  // namespace xenithium::arch
