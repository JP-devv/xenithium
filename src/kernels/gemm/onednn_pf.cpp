// Xenithium · W4A8 prefill GEMMs on oneDNN (see w4a16_gemm.hpp w4a16_pf_set_onednn).
// oneDNN 3.11's matmul runs the engine's W4A8 semantics (s8 activations with an fp32 scale per (row, 128-k group) x s4
// weights with fp16 g128 scales, int32 group sums). It reads its own weight layout ([n][k] s4, two per byte, low nibble =
// even k), so each call repacks the matrix's nib blocks into a per-device scratch first: a byte permutation (a nib byte
// holds k 2p, 2p + 1 of one column) with q ^ 8 = q - 8 as s4; the fp16 scales [k/128][n] are read in place. The stored
// weights stay the nib blocks (decode reads them).
// GLU (gate | up, swiglu): one matmul over [gate | up] into a bf16 scratch, then the swiglu on the engine's table
// (w4a16_rows.cpp can run that epilogue itself with down's quantiser in it, glu_q8).
// The int8 dense matrices (pf_dense's oneDNN route, onednn_i8_pf below) are repacked per call from the VNNI blocks.
#include <dnnl.hpp>
#include <dnnl_sycl.hpp>
#include <sycl/sycl.hpp>

#include <algorithm>
#include <cstdlib>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "xenithium/kernels/gemm/onednn_pf.hpp"

namespace xenithium::kernels {

int onednn_dense_slots() {
  static const int n = [] {
    const char *v = std::getenv("XEN_PF_DENSE_ONEDNN_SLOTS");
    return v ? std::max(1, std::min(64, std::atoi(v))) : 4;
  }();
  return n;
}
size_t onednn_dense_scratch_bytes(size_t max_matrix_bytes) { return static_cast<size_t>(onednn_dense_slots()) * max_matrix_bytes + (1u << 20); }

namespace {

using dt = dnnl::memory::data_type;
using tag = dnnl::memory::format_tag;
using bf16 = sycl::ext::oneapi::bfloat16;

int g_min_n = [] {
  const char *v = std::getenv("XEN_PF_ONEDNN");  // the initial value of w4a16_pf_set_onednn
  return v ? std::atoi(v) : 0;
}();

struct Dev {
  sycl::queue q;
  dnnl::engine eng;
  dnnl::stream st;
  // the last kSlots repacked matrices (the split prefill calls each layer's GEMM twice in a row): key = nib pointer
  // XEN_PF_ONEDNN_SLOTS (experiment): a count past the model's wide matrices = every matrix repacked once and kept
  // (what a load-time repack would give, at its memory)
  static constexpr int kSlots = 512;
  int slots = [] { const char *v = std::getenv("XEN_PF_ONEDNN_SLOTS"); return v ? std::max(1, std::min(kSlots, std::atoi(v))) : 2; }();
  uint8_t *wscr[kSlots] = {};
  size_t wcap[kSlots] = {};
  const uint8_t *wkey[kSlots] = {};
  int wn[kSlots] = {}, wk[kSlots] = {};
  int next = 0;
  // matrices whose layout stays in memory the caller lends (onednn_pf_keep)
  struct Keep {
    uint8_t *dst = nullptr;
    int n = 0, k = 0;
    bool filled = false;
  };
  std::unordered_map<const uint8_t *, Keep> keep;
  bf16 *gscr = nullptr;  // GLU: [gate | up] in bf16 [m][n] (the ESIMD GLU's rounding points: g, u = bf16 of the fp32 sums)
  size_t gcap = 0;
  float *silu = nullptr;  // swiglu_table's values (s(g) by g's bf16 bits, mlp.cpp's expression)
  float *sscr = nullptr;  // the activation scales row-major [m][k/128] (the engine keeps them group-major)
  size_t scap = 0;
  sycl::half *wsscr = nullptr;  // GLU: the gate and up weight scales, each dense [k/128][n/2] (oneDNN reads grouped
  size_t wscap = 0;  // scales as dense, so a strided half of [k/128][n] cannot be passed)
  std::map<std::string, dnnl::matmul> prims;
  // the int8 dense matrices' [n][k] repacks, LRU by use stamp
  static constexpr int kDenseSlotsMax = 64;
  int dslots = onednn_dense_slots();
  int8_t *dscr[kDenseSlotsMax] = {};
  size_t dcap[kDenseSlotsMax] = {};
  const int8_t *dkey[kDenseSlotsMax] = {};
  int dn[kDenseSlotsMax] = {}, dk[kDenseSlotsMax] = {};
  uint64_t dstamp[kDenseSlotsMax] = {};
  uint64_t dclock = 0;
  uint32_t *diff = nullptr;  // onednn_words_differ's counter
};
std::mutex g_mu;
std::vector<std::unique_ptr<Dev>> g_devs;

Dev &dev_of(sycl::queue &q) {
  for (auto &d : g_devs)
    if (d->q == q) return *d;
  auto d = std::make_unique<Dev>();
  d->q = q;
  d->eng = dnnl::sycl_interop::make_engine(q.get_device(), q.get_context());
  d->st = dnnl::sycl_interop::make_stream(d->eng, q);
  g_devs.push_back(std::move(d));
  return *g_devs.back();
}

template <typename T>
T *grow(sycl::queue &q, T *&p, size_t &cap, size_t need) {
  if (need > cap) {
    q.wait();
    if (p) sycl::free(p, q);
    p = static_cast<T *>(sycl::malloc_device(need * sizeof(T), q));
    if (!p) throw std::runtime_error("onednn_pf: no VRAM for a scratch of " + std::to_string(need * sizeof(T)) + " bytes");
    cap = need;
  }
  return p;
}

class OneDnnRepack;
class OneDnnRepackI8;
class OneDnnDiffer;
class OneDnnScales;
class OneDnnSiluTab;
class OneDnnSwiglu;

// nib block (g, t) = 1024 bytes, byte p * 16 + lane = k 2p (low), 2p + 1 (high) of column t * 16 + lane, q in 0..15
// -> dst[n][k / 2] with q ^ 8 (s4 q - 8). One work-group of 16 per block: item i loads bytes [64 i, 64 i + 64) (16
// dwords) into SLM, then writes its column's 64 bytes (p = 0..63, SLM stride 16) as 16 dwords.
sycl::event repack(sycl::queue &q, const uint8_t *w, int n, int k, uint8_t *dst) {
  const int tiles = n / 16;
  const size_t blocks = static_cast<size_t>(k / 128) * tiles;
  return q.submit([&](sycl::handler &h) {
    sycl::local_accessor<uint32_t, 1> slm(256, h);
    h.parallel_for<OneDnnRepack>(sycl::nd_range<1>(blocks * 16, 16), [=](sycl::nd_item<1> it) {
      const size_t blk = it.get_group(0);
      const int i = static_cast<int>(it.get_local_id(0));
      const auto *src = reinterpret_cast<const sycl::vec<uint32_t, 4> *>(w + blk * 1024 + i * 64);
#pragma unroll
      for (int v = 0; v < 4; ++v) {
        const sycl::vec<uint32_t, 4> x = src[v];
        slm[i * 16 + v * 4 + 0] = x[0] ^ 0x88888888u, slm[i * 16 + v * 4 + 1] = x[1] ^ 0x88888888u;
        slm[i * 16 + v * 4 + 2] = x[2] ^ 0x88888888u, slm[i * 16 + v * 4 + 3] = x[3] ^ 0x88888888u;
      }
      sycl::group_barrier(it.get_group());
      const auto *sb = reinterpret_cast<const uint8_t *>(&slm[0]);
      const int g = static_cast<int>(blk / tiles), t = static_cast<int>(blk % tiles);
      auto *d = reinterpret_cast<sycl::vec<uint32_t, 4> *>(dst + (static_cast<size_t>(t) * 16 + i) * (k / 2) + static_cast<size_t>(g) * 64);
#pragma unroll
      for (int v = 0; v < 4; ++v) {
        sycl::vec<uint32_t, 4> o;
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const int p0 = v * 16 + e * 4;
          o[e] = static_cast<uint32_t>(sb[(p0 + 0) * 16 + i]) | static_cast<uint32_t>(sb[(p0 + 1) * 16 + i]) << 8 |
                 static_cast<uint32_t>(sb[(p0 + 2) * 16 + i]) << 16 | static_cast<uint32_t>(sb[(p0 + 3) * 16 + i]) << 24;
        }
        d[v] = o;
      }
    });
  });
}

// as[g][lds] -> dst[r][G]
sycl::event scales_rows(sycl::queue &q, const float *as, int lds, int m, int G, float *dst) {
  return q.parallel_for<OneDnnScales>(sycl::range<1>(static_cast<size_t>(m) * G), [=](sycl::id<1> i) {
    const int r = static_cast<int>(i[0] / G), g = static_cast<int>(i[0] % G);
    dst[i[0]] = as[static_cast<size_t>(g) * lds + r];
  });
}

// h[r][j] = bf16(s(g) u), g = gu[r][j], u = gu[r][nh + j] (bf16), s from the table: kernels:swiglu's arithmetic
sycl::event swiglu_bf16(sycl::queue &q, int m, int nh, const bf16 *gu, const float *tab, bf16 *h, int ldh) {
  return q.parallel_for<OneDnnSwiglu>(sycl::range<2>(static_cast<size_t>(m), static_cast<size_t>(nh)), [=](sycl::id<2> i) {
    const size_t r = i[0], j = i[1];
    const bf16 g = gu[r * 2 * nh + j], u = gu[r * 2 * nh + nh + j];
    h[r * ldh + j] = static_cast<bf16>(tab[sycl::bit_cast<uint16_t>(g)] * static_cast<float>(u));
  });
}

std::string key(const char *what, std::initializer_list<long long> v) {
  std::string s(what);
  for (long long x : v) s += ':' + std::to_string(x);
  return s;
}

// dense.hpp's int8 VNNI blocks -> [n][k] s8. Full groups: block (g, t) = 2048 B, element (kin, lane) of call j
// at j * 256 + (kin / 2) * 32 + lane * 2 + (kin & 1); a k % 128 tail of `calls` x 16 k follows as [t][calls x 256]. One
// work-group of 16 per block: the items stage the block into SLM (dword loads), then item = lane writes its column's k
// bytes as one 16-byte store per call.
sycl::event repack_i8(sycl::queue &q, const int8_t *w, int n, int k, int8_t *dst) {
  const int tiles = n / 16, full = k / 128, calls_t = (k % 128) / 16;
  const int nb = full + (calls_t ? 1 : 0);
  return q.submit([&](sycl::handler &h) {
    sycl::local_accessor<uint32_t, 1> slm(512, h);
    h.parallel_for<OneDnnRepackI8>(sycl::nd_range<1>(static_cast<size_t>(nb) * tiles * 16, 16), [=](sycl::nd_item<1> it) {
      const size_t blk = it.get_group(0);
      const int i = static_cast<int>(it.get_local_id(0));
      const int g = static_cast<int>(blk / tiles), t = static_cast<int>(blk % tiles);
      const bool tail = g == full;
      const int calls = tail ? calls_t : 8;
      const size_t base = tail ? static_cast<size_t>(full) * tiles * 2048 + static_cast<size_t>(t) * calls_t * 256 : (static_cast<size_t>(g) * tiles + t) * 2048;
      const auto *src = reinterpret_cast<const sycl::vec<uint32_t, 4> *>(w + base);
      for (int v = i; v < calls * 16; v += 16) {
        const sycl::vec<uint32_t, 4> x = src[v];
        slm[v * 4 + 0] = x[0], slm[v * 4 + 1] = x[1], slm[v * 4 + 2] = x[2], slm[v * 4 + 3] = x[3];
      }
      sycl::group_barrier(it.get_group());
      const auto *sb = reinterpret_cast<const uint8_t *>(&slm[0]);
      auto *d = reinterpret_cast<sycl::vec<uint32_t, 4> *>(dst + (static_cast<size_t>(t) * 16 + i) * k + static_cast<size_t>(g) * 128);
      for (int c = 0; c < calls; ++c) {
        sycl::vec<uint32_t, 4> o;
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          uint32_t u = 0;
#pragma unroll
          for (int b = 0; b < 4; ++b) {
            const int kin = e * 4 + b;
            u |= static_cast<uint32_t>(sb[c * 256 + (kin / 2) * 32 + i * 2 + (kin & 1)]) << (8 * b);
          }
          o[e] = u;
        }
        d[c] = o;
      }
    });
  });
}

// The [n][k] s8 copy of an int8 dense matrix in the device's LRU (repacked on a miss).
const int8_t *dense_slot(Dev &d, const int8_t *w, int n, int k) {
  int slot = -1, old = 0;
  for (int i = 0; i < d.dslots; ++i) {
    if (d.dkey[i] == w && d.dn[i] == n && d.dk[i] == k) slot = i;
    if (d.dstamp[i] < d.dstamp[old]) old = i;
  }
  if (slot < 0) {
    slot = old;
    grow(d.q, d.dscr[slot], d.dcap[slot], static_cast<size_t>(n) * k);
    repack_i8(d.q, w, n, k, d.dscr[slot]);
    d.dkey[slot] = w, d.dn[slot] = n, d.dk[slot] = k;
  }
  d.dstamp[slot] = ++d.dclock;
  return d.dscr[slot];
}

// mode 0: W8A16 (bf16 src, weight scale per column or none); 1: split-K raw sums (ks batched k slices)
dnnl::matmul &prim_i8(Dev &d, int mode, int m, int n, int k, int lda, bool scaled, dt dst_dt, int ld, int ks) {
  const std::string kk = key("i8", {mode, m, n, k, lda, scaled, static_cast<int>(dst_dt), ld, ks});
  auto it = d.prims.find(kk);
  if (it != d.prims.end()) return it->second;
  using namespace dnnl;
  primitive_attr attr;
  attr.set_fpmath_mode(fpmath_mode::bf16, true);  // the codes widened to bf16 (exact), bf16 DPAS, fp32 sums
  memory::desc smd, wmd, dmd;
  if (mode == 1) {
    const int kp = k / ks;
    smd = memory::desc({ks, m, kp}, dt::bf16, memory::dims{kp, lda, 1});
    wmd = memory::desc({ks, kp, n}, dt::s8, memory::dims{kp, 1, k});
    dmd = memory::desc({ks, m, n}, dst_dt, memory::dims{static_cast<long long>(m) * ld, ld, 1});
  } else {
    smd = memory::desc({m, k}, dt::bf16, memory::dims{lda, 1});
    wmd = memory::desc({k, n}, dt::s8, tag::ba);
    dmd = memory::desc({m, n}, dst_dt, memory::dims{ld, 1});
    if (scaled) attr.set_scales(DNNL_ARG_WEIGHTS, 2, {}, dt::f32);  // per column, on the finished sum
  }
  matmul::primitive_desc pd(d.eng, smd, wmd, dmd, attr);
  return d.prims.emplace(kk, matmul(pd)).first->second;
}

// C = A8 (m x k s8, scales [k/128][lds] fp32 group-major). W (n x k s4 at w, scales s [k/128] rows of stride lds_w)
// into dst (dst_dt, row stride ldd); post: 0 none, 1 swish, 2 multiply by mul (bf16 [m][ldmul]).
dnnl::matmul &prim(Dev &d, int m, int n, int k, int lds, int lds_w, dt dst_dt, int ldd, int post, int ldmul) {
  const std::string kk = key("mm", {m, n, k, lds, lds_w, static_cast<int>(dst_dt), ldd, post, ldmul});
  auto it = d.prims.find(kk);
  if (it != d.prims.end()) return it->second;
  dnnl::primitive_attr attr;
  attr.set_scales(DNNL_ARG_SRC, 3, {1, 128}, dt::f32);
  attr.set_scales(DNNL_ARG_WEIGHTS, 3, {128, 1}, dt::f16);
  dnnl::post_ops po;
  if (post == 1) po.append_eltwise(dnnl::algorithm::eltwise_swish, 1.f, 0.f);
  if (post == 2) po.append_binary(dnnl::algorithm::binary_mul, dnnl::memory::desc({m, n}, dt::bf16, dnnl::memory::dims{ldmul, 1}));
  if (post) attr.set_post_ops(po);
  const dnnl::memory::desc smd({m, k}, dt::s8, dnnl::memory::dims{k, 1});
  const dnnl::memory::desc wmd({k, n}, dt::s4, tag::ba);
  const dnnl::memory::desc dmd({m, n}, dst_dt, dnnl::memory::dims{ldd, 1});
  dnnl::matmul::primitive_desc pd(d.eng, smd, wmd, dmd, attr);
  (void)lds, (void)lds_w;
  return d.prims.emplace(kk, dnnl::matmul(pd)).first->second;
}

sycl::event run(Dev &d, dnnl::matmul &mm, int m, int n, int k, const int8_t *aq, const float *as, int lds, const uint8_t *w, const sycl::half *s, int lds_w,
                void *dst, dt dst_dt, int ldd, const bf16 *mul, int ldmul) {
  using namespace dnnl;
  auto mk = [&](const memory::desc &md, const void *p) { return sycl_interop::make_memory(md, d.eng, sycl_interop::memory_kind::usm, const_cast<void *>(p)); };
  std::unordered_map<int, memory> args{
      {DNNL_ARG_SRC, mk(memory::desc({m, k}, dt::s8, memory::dims{k, 1}), aq)},
      {DNNL_ARG_WEIGHTS, mk(memory::desc({k, n}, dt::s4, tag::ba), w)},
      {DNNL_ARG_DST, mk(memory::desc({m, n}, dst_dt, memory::dims{ldd, 1}), dst)},
      {DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, mk(memory::desc({m, k / 128}, dt::f32, tag::ab), as)},
      {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, mk(memory::desc({k / 128, n}, dt::f16, memory::dims{lds_w, 1}), s)}};
  if (mul) args[DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1] = mk(memory::desc({m, n}, dt::bf16, memory::dims{ldmul, 1}), mul);
  return sycl_interop::execute(mm, d.st, args);
}

}  // namespace

void w4a16_pf_set_onednn(int min_n) {
  g_min_n = min_n;
  std::lock_guard<std::mutex> lk(g_mu);
  for (auto &d : g_devs) {
    for (auto &key : d->wkey) key = nullptr;  // weights may have been reloaded at the same address
    d->keep.clear();
  }
}

void onednn_pf_keep(sycl::queue &q, const uint8_t *w_packed, int n, int k, uint8_t *dst) {
  if (dst && (reinterpret_cast<uintptr_t>(dst) % 4096 || n < 16 || n % 16 || k < 128 || k % 128)) throw std::invalid_argument("onednn_pf_keep: shape / alignment");
  std::lock_guard<std::mutex> lk(g_mu);
  Dev &d = dev_of(q);
  if (!dst) d.keep.erase(w_packed);
  else d.keep[w_packed] = Dev::Keep{dst, n, k, false};
}

void onednn_pf_reserve(sycl::queue &q, size_t matrix_bytes) {
  std::lock_guard<std::mutex> lk(g_mu);
  Dev &d = dev_of(q);
  for (int i = 0; i < d.slots && i < 2; ++i)
    if (d.wcap[i] < matrix_bytes) {
      q.wait();  // (grow frees the old scratch)
      grow(q, d.wscr[i], d.wcap[i], matrix_bytes);
      d.wkey[i] = nullptr;
    }
}

int onednn_pf_keep_fill(sycl::queue &q, int max_n) {
  std::lock_guard<std::mutex> lk(g_mu);
  Dev &d = dev_of(q);
  int done = 0;
  for (auto &[w, e] : d.keep) {
    if (done >= max_n) break;
    if (e.filled) continue;
    repack(q, w, e.n, e.k, e.dst);
    e.filled = true;
    ++done;
  }
  return done;
}

void onednn_pf_keep_count(sycl::queue &q, int *kept, int *filled) {
  std::lock_guard<std::mutex> lk(g_mu);
  Dev &d = dev_of(q);
  int f = 0;
  for (const auto &[w, e] : d.keep) f += e.filled;
  if (kept) *kept = static_cast<int>(d.keep.size());
  if (filled) *filled = f;
}
int w4a16_pf_onednn() { return g_min_n; }
int g_min_m = [] {
  const char *v = std::getenv("XEN_PF_ONEDNN_MIN_M");  // experiment: rows per call below which the ESIMD body stays
  return v ? std::atoi(v) : 1024;
}();
void w4a16_pf_set_onednn_rows(int min_m) { g_min_m = std::max(1, min_m); }
int w4a16_pf_onednn_rows() { return g_min_m; }
bool onednn_pf_takes(int m, int n_out) { return g_min_n > 0 && n_out >= g_min_n && m >= g_min_m; }

sycl::event onednn_w4a8_pf(sycl::queue &q, int m, int n, int k, const int8_t *aq, const float *as, int lds, const uint8_t *w_packed, const sycl::half *scales,
                           float *c, int ldc, bf16 *h, int ldh, bf16 **gu_out, const float **tab_out, const float *as_rows) {
  std::lock_guard<std::mutex> lk(g_mu);
  Dev &d = dev_of(q);
  uint8_t *wr = nullptr;
  if (const auto kept = d.keep.find(w_packed); kept != d.keep.end() && kept->second.n == n && kept->second.k == k) {
    // the layout stays in lent memory (onednn_pf_keep)
    if (!kept->second.filled) {
      repack(q, w_packed, n, k, kept->second.dst);
      kept->second.filled = true;
    }
    wr = kept->second.dst;
  } else {
    int slot = -1;
    for (int i = 0; i < d.slots; ++i)
      if (d.wkey[i] == w_packed && d.wn[i] == n && d.wk[i] == k) slot = i;
    if (slot < 0) {
      slot = d.next, d.next = (d.next + 1) % d.slots;
      grow(q, d.wscr[slot], d.wcap[slot], static_cast<size_t>(n) * k / 2);
      repack(q, w_packed, n, k, d.wscr[slot]);
      d.wkey[slot] = w_packed, d.wn[slot] = n, d.wk[slot] = k;
    }
    wr = d.wscr[slot];
  }
  if (as_rows) {  // the producer wrote them row-major too
    as = as_rows;
  } else {
    float *sr = grow(q, d.sscr, d.scap, static_cast<size_t>(m) * (k / 128));
    scales_rows(q, as, lds, m, k / 128, sr);
    as = sr;
  }
  if (!h) {
    auto &mm = prim(d, m, n, k, lds, n, dt::f32, ldc, 0, 0);
    return run(d, mm, m, n, k, aq, as, lds, wr, scales, n, c, dt::f32, ldc, nullptr, 0);
  }
  // GLU: one matmul over [gate | up] into bf16 gu, then swiglu with the ESIMD GLU's rounding points and table (swish /
  // multiply post-ops on the fp32 sums would not give the ESIMD path's values)
  const int nh = n / 2;
  if (!d.silu) {
    d.silu = sycl::malloc_device<float>(65536, q);
    float *tab = d.silu;
    q.parallel_for<OneDnnSiluTab>(sycl::range<1>(65536), [=](sycl::id<1> id) {
      const float g = static_cast<float>(sycl::bit_cast<bf16>(static_cast<uint16_t>(id[0])));
      tab[id[0]] = g * (1.f / (1.f + sycl::exp(-g)));  // mlp.cpp swiglu_table's expression
    });
  }
  bf16 *gu = grow(q, d.gscr, d.gcap, static_cast<size_t>(m) * n);
  auto &mm = prim(d, m, n, k, lds, n, dt::bf16, n, 0, 0);
  const sycl::event e = run(d, mm, m, n, k, aq, as, lds, wr, scales, n, gu, dt::bf16, n, nullptr, 0);
  if (gu_out) {  // the caller's epilogue (swiglu + down's quantiser)
    *gu_out = gu;
    if (tab_out) *tab_out = d.silu;
    return e;
  }
  return swiglu_bf16(q, m, nh, gu, d.silu, h, ldh);
}

sycl::event onednn_i8_pf(sycl::queue &q, int m, int n, int k, const int8_t *w_packed, const float *wscale, const bf16 *a, int lda, float *f32, bf16 *b16,
                         int ld, int ks) {
  using namespace dnnl;
  if ((f32 != nullptr) == (b16 != nullptr) || m < 1 || n % 16 || k % 16 || ld < n || ks < 1 || k % ks || (k / ks) % 16 || (ks > 1 && !f32))
    throw std::invalid_argument("onednn_i8_pf: shape / output");
  std::lock_guard<std::mutex> lk(g_mu);
  Dev &d = dev_of(q);
  const int8_t *wr = dense_slot(d, w_packed, n, k);
  const int mode = ks > 1 ? 1 : 0;
  const dt ddt = f32 ? dt::f32 : dt::bf16;
  void *dst = f32 ? static_cast<void *>(f32) : static_cast<void *>(b16);
  auto &mm = prim_i8(d, mode, m, n, k, lda, wscale != nullptr, ddt, ld, ks);
  auto mk = [&](const memory::desc &md, const void *p) { return sycl_interop::make_memory(md, d.eng, sycl_interop::memory_kind::usm, const_cast<void *>(p)); };
  std::unordered_map<int, memory> args;
  if (mode == 1) {
    const int kp = k / ks;
    args[DNNL_ARG_SRC] = mk(memory::desc({ks, m, kp}, dt::bf16, memory::dims{kp, lda, 1}), a);
    args[DNNL_ARG_WEIGHTS] = mk(memory::desc({ks, kp, n}, dt::s8, memory::dims{kp, 1, k}), wr);
    args[DNNL_ARG_DST] = mk(memory::desc({ks, m, n}, ddt, memory::dims{static_cast<long long>(m) * ld, ld, 1}), dst);
  } else {
    args[DNNL_ARG_SRC] = mk(memory::desc({m, k}, dt::bf16, memory::dims{lda, 1}), a);
    args[DNNL_ARG_WEIGHTS] = mk(memory::desc({k, n}, dt::s8, tag::ba), wr);
    args[DNNL_ARG_DST] = mk(memory::desc({m, n}, ddt, memory::dims{ld, 1}), dst);
    if (wscale) args[DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS] = mk(memory::desc({n}, dt::f32, tag::a), wscale);
  }
  return sycl_interop::execute(mm, d.st, args);
}

void onednn_dense_forget(sycl::queue &q, const int8_t *w_packed) {
  std::lock_guard<std::mutex> lk(g_mu);
  Dev &d = dev_of(q);
  for (int i = 0; i < d.dslots; ++i)
    if (d.dkey[i] == w_packed && d.dscr[i]) {
      q.wait();
      sycl::free(d.dscr[i], q);
      d.dscr[i] = nullptr, d.dcap[i] = 0, d.dkey[i] = nullptr, d.dstamp[i] = 0;
    }
}

size_t onednn_words_differ(sycl::queue &q, const void *x, const void *y, int parts, int rows, size_t row_bytes, size_t ld_bytes, size_t part_bytes) {
  if (row_bytes % 4 || ld_bytes % 4 || part_bytes % 4) throw std::invalid_argument("onednn_words_differ: alignment");
  uint32_t *cnt = nullptr;
  {
    std::lock_guard<std::mutex> lk(g_mu);
    Dev &d = dev_of(q);
    if (!d.diff) d.diff = sycl::malloc_device<uint32_t>(1, q);
    cnt = d.diff;
  }
  const size_t wpr = row_bytes / 4, ldw = ld_bytes / 4, pw = part_bytes / 4;
  const auto *xa = static_cast<const uint32_t *>(x);
  const auto *ya = static_cast<const uint32_t *>(y);
  q.memset(cnt, 0, 4);
  q.parallel_for<OneDnnDiffer>(sycl::range<3>(static_cast<size_t>(parts), static_cast<size_t>(rows), wpr), [=](sycl::id<3> i) {
    const size_t at = i[0] * pw + i[1] * ldw + i[2];
    if (xa[at] != ya[at]) sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device>(*cnt).fetch_add(1u);
  });
  uint32_t h = 0;
  q.memcpy(&h, cnt, 4).wait();
  return h;
}

}  // namespace xenithium:kernels
