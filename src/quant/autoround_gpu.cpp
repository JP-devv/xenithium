// Xenithium · AutoRound GPU worker (SYCL, one card): the CPU worker's math as kernels.
// Per step: x -> A8 fake quant (per row, 128-group) -> grouped gate|up GEMM over the step's
// expert segments (weights dequantised on the fly from bf16 + v / alpha / beta) -> h = bf16(silu(bf16 g)
// bf16 u) -> A8 fake quant (64-group) -> grouped down GEMM -> per-row weighted combine. Backward:
// dh = (w delta) Wd_q, the SwiGLU derivative, then one kernel per matrix computes dW for a
// (64 rows x one 128-k group) tile over the expert's rows and applies signSGD to v, alpha, beta in
// its epilogue (each tile owns its parameters, so no gradient buffers and no second pass).
// The tiled GEMMs are plain fp32 SIMT (64 x 64 tiles, 4 x 4 per work-item), kept simple: the bake
// runs once per model.
// Final codes come from the host (ar_codes) on the downloaded parameters, so they do not depend
// on device fp rounding.
#include <sycl/sycl.hpp>

#include <algorithm>
#include <cstdio>
#include <stdexcept>

#include "xenithium/quant/autoround.hpp"

namespace xenithium::quant {

namespace {

constexpr int TM = 64, TN = 64, TK = 32;

inline float d_bf16(uint16_t b) { return sycl::bit_cast<float>(static_cast<uint32_t>(b) << 16); }
inline float d_rbf16(float f) {
  uint32_t u = sycl::bit_cast<uint32_t>(f);
  u = (u + 0x7fffu + ((u >> 16) & 1u)) & 0xffff0000u;
  return sycl::bit_cast<float>(u);
}
inline float d_scale(float wmin, float wmax, float a, float b) {
  const float x = -wmin * a, y = wmax * b;
  const float s = static_cast<float>(sycl::half(x >= y ? x / kArQMax : -y / kArQMax));
  return s != 0.f ? s : 5.9604645e-8f;
}
inline float d_wq(float w, float s, float v) {
  float t = sycl::rint(w / s + v);
  t = sycl::fmin(7.f, sycl::fmax(-8.f, t));
  return s * t;
}
inline float d_silu(float x) { return x / (1.f + sycl::exp(-x)); }
// normalised Walsh-Hadamard of 64 values in place
inline void d_hadamard64(float *x) {
  for (int h = 1; h < 64; h <<= 1)
    for (int i = 0; i < 64; i += h << 1)
      for (int j = i; j < i + h; ++j) {
        const float a = x[j], b = x[j + h];
        x[j] = a + b, x[j + h] = a - b;
      }
  for (int i = 0; i < 64; ++i) x[i] *= 0.125f;
}

std::vector<sycl::device> gpu_devices() {
  std::vector<sycl::device> out;
  for (const auto &p : sycl::platform::get_platforms()) {
    if (p.get_backend() != sycl::backend::ext_oneapi_level_zero) continue;
    for (const auto &d : p.get_devices(sycl::info::device_type::gpu)) out.push_back(d);
  }
  return out;
}

template <class T>
T *dalloc(sycl::queue &q, size_t n) {
  if (n == 0) n = 1;
  T *p = static_cast<T *>(sycl::aligned_alloc_device(4096, n * sizeof(T), q));
  if (!p) throw std::runtime_error("autoround gpu: device allocation failed");
  return p;
}

// A device buffer that grows on demand.
template <class T>
struct DBuf {
  T *p = nullptr;
  size_t n = 0;
  void need(sycl::queue &q, size_t want) {
    if (want <= n) return;
    if (p) sycl::free(p, q);
    n = want + want / 4 + 1024;
    p = dalloc<T>(q, n);
  }
  void release(sycl::queue &q) {
    if (p) sycl::free(p, q);
    p = nullptr, n = 0;
  }
};

// One matrix kind's parameters for all local experts: [E][rows][cols], groups along cols.
struct DMat {
  int rows = 0, cols = 0, grp = 0;
  uint16_t *w = nullptr;
  int16_t *v = nullptr, *v_best = nullptr;
  float *a = nullptr, *b = nullptr, *a_best = nullptr, *b_best = nullptr, *wmin = nullptr, *wmax = nullptr;
  size_t elems(int E) const { return static_cast<size_t>(E) * rows * cols; }
  size_t gelems(int E) const { return static_cast<size_t>(E) * rows * (cols / grp); }
};

class GpuWorker final : public ArWorker {
 public:
  explicit GpuWorker(int dev) : dev_(dev) {
    const auto ds = gpu_devices();
    if (dev < 0 || dev >= static_cast<int>(ds.size())) throw std::runtime_error("autoround gpu: no device " + std::to_string(dev));
    q_ = sycl::queue(ds[static_cast<size_t>(dev)], sycl::property::queue::in_order());
  }
  ~GpuWorker() override {
    q_.wait();
    free_params();
    xb_.release(q_), xq_.release(q_), part_.release(q_), delta_.release(q_);
    asg_row_.release(q_), asg_w_.release(q_), tiles_.release(q_), seg_.release(q_), row_asg_.release(q_);
    gu_out_.release(q_), hq_.release(q_), y_.release(q_), dh_.release(q_), dgu_.release(q_);
  }
  void free_params() {
    for (DMat *m : {&gu_, &dn_}) {
      for (void *p : {static_cast<void *>(m->w), static_cast<void *>(m->v), static_cast<void *>(m->v_best), static_cast<void *>(m->a),
                      static_cast<void *>(m->b), static_cast<void *>(m->a_best), static_cast<void *>(m->b_best), static_cast<void *>(m->wmin),
                      static_cast<void *>(m->wmax)})
        if (p) sycl::free(p, q_);
      *m = DMat{};
    }
  }
  std::string name() const override { return "gpu" + std::to_string(dev_); }
  void exact(bool on) override { exact_ = on; }
  double speed_hint() const override { return 1.0; }
  void act_quant(bool on) override { aq_ = on; }

  void setup(const ArShape &s, const std::vector<int> &experts, const ExpertSource &src) override {
    q_.wait();
    if (s.rot_h && s.rot_h != 64) throw std::runtime_error("gpu worker: rot_h must be 64 (the h group)");
    if (s.rot_h && s.hgroup != 64) throw std::runtime_error("gpu worker: rot_h needs hgroup 64");
    free_params();  // the previous layer's (setup runs once or twice per layer)
    s_ = s;
    ex_ = experts;
    E_ = static_cast<int>(experts.size());
    src_ = src;
    const int H = s.hidden, I = s.inter;
    gu_ = {2 * I, H, s.wgroup};
    dn_ = {H, I, s.wgroup};
    for (DMat *m : {&gu_, &dn_}) {
      m->w = dalloc<uint16_t>(q_, m->elems(E_));
      m->v = dalloc<int16_t>(q_, m->elems(E_));
      m->v_best = dalloc<int16_t>(q_, m->elems(E_));
      for (float **p : {&m->a, &m->b, &m->a_best, &m->b_best, &m->wmin, &m->wmax}) *p = dalloc<float>(q_, m->gelems(E_));
      q_.memset(m->v, 0, m->elems(E_) * 2);
      q_.fill(m->a, 1.f, m->gelems(E_));
      q_.fill(m->b, 1.f, m->gelems(E_));
    }
    std::vector<float> lo, hi;
    for (int i = 0; i < E_; ++i) {
      const size_t e = static_cast<size_t>(experts[static_cast<size_t>(i)]);
      const uint16_t *wg = src.gate_up + e * 2 * I * H, *wd = src.down + e * H * I;
      q_.memcpy(gu_.w + static_cast<size_t>(i) * 2 * I * H, wg, static_cast<size_t>(2) * I * H * 2);
      q_.memcpy(dn_.w + static_cast<size_t>(i) * H * I, wd, static_cast<size_t>(H) * I * 2);
      for (auto [m, w] : {std::pair<DMat *, const uint16_t *>{&gu_, wg}, {&dn_, wd}}) {
        const size_t g = static_cast<size_t>(m->rows) * (m->cols / m->grp);
        lo.resize(g), hi.resize(g);
        ar_minmax(w, m->rows, m->cols, m->grp, lo.data(), hi.data());
        q_.memcpy(m->wmin + static_cast<size_t>(i) * g, lo.data(), g * 4).wait();
        q_.memcpy(m->wmax + static_cast<size_t>(i) * g, hi.data(), g * 4).wait();
      }
      q_.wait();
    }
    save_best();
    q_.wait();
  }

  void begin(const ArBatch &b) override {
    b_ = &b;
    const int H = s_.hidden, I = s_.inter, R = b.rows;
    na_ = static_cast<int>(b.asg_row.size());
    xb_.need(q_, static_cast<size_t>(R) * H);
    xq_.need(q_, static_cast<size_t>(R) * H);
    part_.need(q_, static_cast<size_t>(R) * H);
    delta_.need(q_, static_cast<size_t>(R) * H);
    row_asg_.need(q_, static_cast<size_t>(R) * s_.topk);
    asg_row_.need(q_, static_cast<size_t>(na_));
    asg_w_.need(q_, static_cast<size_t>(na_));
    seg_.need(q_, static_cast<size_t>(E_) + 1);
    gu_out_.need(q_, static_cast<size_t>(na_) * 2 * I);
    hq_.need(q_, static_cast<size_t>(na_) * I);
    y_.need(q_, static_cast<size_t>(na_) * H);
    dh_.need(q_, static_cast<size_t>(na_) * I);
    dgu_.need(q_, static_cast<size_t>(na_) * 2 * I);
    // 64-row tiles of every segment: (local expert, first assignment, rows)
    tiles_h_.clear();
    for (int e = 0; e < E_; ++e)
      for (int a = b.seg_start[static_cast<size_t>(e)]; a < b.seg_start[static_cast<size_t>(e) + 1]; a += TM)
        tiles_h_.insert(tiles_h_.end(), {e, a, std::min(TM, b.seg_start[static_cast<size_t>(e) + 1] - a)});
    tiles_.need(q_, tiles_h_.size() + 3);
    q_.memcpy(xb_.p, b.x, static_cast<size_t>(R) * H * 2);
    q_.memcpy(row_asg_.p, b.row_asg.data(), b.row_asg.size() * 4);
    if (na_) {
      q_.memcpy(asg_row_.p, b.asg_row.data(), static_cast<size_t>(na_) * 4);
      q_.memcpy(asg_w_.p, b.asg_w.data(), static_cast<size_t>(na_) * 4);
    }
    q_.memcpy(seg_.p, b.seg_start.data(), b.seg_start.size() * 4);
    if (!tiles_h_.empty()) q_.memcpy(tiles_.p, tiles_h_.data(), tiles_h_.size() * 4);
    // x -> xq
    {
      const uint16_t *xb = xb_.p;
      float *xq = xq_.p;
      const int G = s_.xgroup, ng = H / G;
      const bool aq = aq_ && !exact_;
      q_.parallel_for(sycl::range<1>(static_cast<size_t>(R) * ng), [=](sycl::id<1> id) {
        const size_t r = id[0] / ng, g = id[0] % ng;
        const uint16_t *src = xb + r * H + g * G;
        float *dst = xq + r * H + g * G;
        if (!aq) {
          for (int i = 0; i < G; ++i) dst[i] = d_bf16(src[i]);
          return;
        }
        float amax = 0.f;
        for (int i = 0; i < G; ++i) amax = sycl::fmax(amax, sycl::fabs(d_bf16(src[i])));
        if (amax == 0.f) {
          for (int i = 0; i < G; ++i) dst[i] = 0.f;
          return;
        }
        const float inv = 127.f / amax, sc = amax / 127.f;
        for (int i = 0; i < G; ++i) dst[i] = sycl::fmin(127.f, sycl::fmax(-127.f, sycl::rint(d_bf16(src[i]) * inv))) * sc;
      });
    }
  }

  void forward(float *part) override {
    const int H = s_.hidden, I = s_.inter, R = b_->rows;
    const int T = static_cast<int>(tiles_h_.size() / 3);
    const bool ex = exact_;
    if (T > 0) {
      // gate|up: gu[a][j] = sum_k xq[row a][k] Wq_gu[e][j][k]
      const float *xq = xq_.p;
      const int32_t *arow = asg_row_.p;
      const DMat W = gu_;
      float *gu = gu_out_.p;
      seg_gemm(T, 2 * I / TN, H,
               [=](float *As, int, int a0, int m, int k0, int l) {
                 const int mm = l / 4, kc = (l % 4) * 8;
                 if (mm < m) {
                   const float *src = xq + static_cast<size_t>(arow[a0 + mm]) * H + k0 + kc;
                   for (int i = 0; i < 8; ++i) As[(kc + i) * TM + mm] = src[i];
                 } else {
                   for (int i = 0; i < 8; ++i) As[(kc + i) * TM + mm] = 0.f;
                 }
               },
               [=](float *Bs, int e, int n0, int k0, int l) { fill_w_rows(W, Bs, e, n0, k0, l, ex); },
               [=](int, int a0, int m, int n0, int ty, int tx, const float (&acc)[4][4]) {
                 for (int i = 0; i < 4; ++i) {
                   const int mm = ty * 4 + i;
                   if (mm >= m) continue;
                   for (int j = 0; j < 4; ++j) gu[static_cast<size_t>(a0 + mm) * 2 * I + n0 + tx * 4 + j] = acc[i][j];
                 }
               });
      // h, A8 per 64-group
      {
        const int G = s_.hgroup, ng = I / G;
        float *hq = hq_.p;
        const bool aq = aq_ && !exact_;
        const bool rot = s_.rot_h != 0;
        q_.parallel_for(sycl::range<1>(static_cast<size_t>(na_) * ng), [=](sycl::id<1> id) {
          const size_t a = id[0] / ng, g = id[0] % ng;
          const float *gr = gu + a * 2 * I + g * G, *ur = gr + I;
          float h[64];
          float amax = 0.f;
          for (int i = 0; i < G; ++i) h[i] = d_rbf16(d_silu(d_rbf16(gr[i])) * d_rbf16(ur[i]));
          if (rot) {  // the down input in the rotated basis, bf16 as the engine stores it (moe_hadamard_rows)
            d_hadamard64(h);
            for (int i = 0; i < G; ++i) h[i] = d_rbf16(h[i]);
          }
          for (int i = 0; i < G; ++i) amax = sycl::fmax(amax, sycl::fabs(h[i]));
          float *dst = hq + a * I + g * G;
          if (!aq) {
            for (int i = 0; i < G; ++i) dst[i] = h[i];
            return;
          }
          if (amax == 0.f) {
            for (int i = 0; i < G; ++i) dst[i] = 0.f;
            return;
          }
          const float inv = 127.f / amax, sc = amax / 127.f;
          for (int i = 0; i < G; ++i) dst[i] = sycl::fmin(127.f, sycl::fmax(-127.f, sycl::rint(h[i] * inv))) * sc;
        });
      }
      // down: y[a][n] = sum_k hq[a][k] Wq_d[e][n][k]
      {
        const float *hq = hq_.p;
        const DMat D = dn_;
        float *y = y_.p;
        seg_gemm(T, H / TN, I,
                 [=](float *As, int, int a0, int m, int k0, int l) {
                   const int mm = l / 4, kc = (l % 4) * 8;
                   for (int i = 0; i < 8; ++i) As[(kc + i) * TM + mm] = mm < m ? hq[static_cast<size_t>(a0 + mm) * I + k0 + kc + i] : 0.f;
                 },
                 [=](float *Bs, int e, int n0, int k0, int l) { fill_w_rows(D, Bs, e, n0, k0, l, ex); },
                 [=](int, int a0, int m, int n0, int ty, int tx, const float (&acc)[4][4]) {
                   for (int i = 0; i < 4; ++i) {
                     const int mm = ty * 4 + i;
                     if (mm >= m) continue;
                     for (int j = 0; j < 4; ++j) y[static_cast<size_t>(a0 + mm) * H + n0 + tx * 4 + j] = acc[i][j];
                   }
                 });
      }
    }
    // combine: part[r][n] = sum_k w y
    {
      const int K = s_.topk;
      const int32_t *ra = row_asg_.p;
      const float *aw = asg_w_.p, *y = y_.p;
      float *p = part_.p;
      q_.parallel_for(sycl::range<2>(static_cast<size_t>(R), static_cast<size_t>(H)), [=](sycl::id<2> id) {
        const size_t r = id[0], n = id[1];
        float s = 0.f;
        for (int k = 0; k < K; ++k) {
          const int a = ra[r * K + k];
          if (a >= 0) s += aw[a] * y[static_cast<size_t>(a) * H + n];
        }
        p[r * H + n] = s;
      });
    }
    q_.memcpy(part, part_.p, static_cast<size_t>(R) * H * 4).wait();
  }

  void backward(const float *delta, float lr, float mm_lr, bool minmax) override {
    const int H = s_.hidden, I = s_.inter, R = b_->rows;
    const int T = static_cast<int>(tiles_h_.size() / 3);
    if (T == 0) return;
    q_.memcpy(delta_.p, delta, static_cast<size_t>(R) * H * 4);
    const float *dl = delta_.p, *aw = asg_w_.p;
    const int32_t *arow = asg_row_.p;
    // dh[a][k] = w_a sum_r delta[row a][r] Wq_d[e][r][k]
    {
      const DMat D = dn_;
      float *dh = dh_.p;
      seg_gemm(T, I / TN, H,
               [=](float *As, int, int a0, int m, int r0, int l) {
                 const int mm = l / 4, rc = (l % 4) * 8;
                 if (mm < m) {
                   const float wa = aw[a0 + mm];
                   const float *src = dl + static_cast<size_t>(arow[a0 + mm]) * H + r0 + rc;
                   for (int i = 0; i < 8; ++i) As[(rc + i) * TM + mm] = wa * src[i];
                 } else {
                   for (int i = 0; i < 8; ++i) As[(rc + i) * TM + mm] = 0.f;
                 }
               },
               [=](float *Bs, int e, int n0, int r0, int l) {
                 // Bs[rr][nn] = Wq_d[e][r0 + rr][n0 + nn]: 8 consecutive k (one group) per item
                 const int rr = l / 8, nc = (l % 8) * 8;
                 const size_t row = static_cast<size_t>(e) * D.rows + r0 + rr;
                 const int G = D.cols / D.grp;
                 const size_t gi = row * G + (n0 + nc) / D.grp;
                 const float s = d_scale(D.wmin[gi], D.wmax[gi], D.a[gi], D.b[gi]);
                 const size_t at = row * D.cols + n0 + nc;
                 for (int i = 0; i < 8; ++i) Bs[rr * TN + nc + i] = d_wq(d_bf16(D.w[at + i]), s, D.v[at + i] / kArVOne);
               },
               [=](int, int a0, int m, int n0, int ty, int tx, const float (&acc)[4][4]) {
                 for (int i = 0; i < 4; ++i) {
                   const int mm = ty * 4 + i;
                   if (mm >= m) continue;
                   for (int j = 0; j < 4; ++j) dh[static_cast<size_t>(a0 + mm) * I + n0 + tx * 4 + j] = acc[i][j];
                 }
               });
    }
    // rotated h: dh is the gradient wrt H h; back to h's basis (H symmetric orthogonal)
    if (s_.rot_h) {
      float *dh = dh_.p;
      const int ng = I / 64;
      q_.parallel_for(sycl::range<1>(static_cast<size_t>(na_) * ng), [=](sycl::id<1> id) {
        float t[64];
        float *p = dh + (id[0] / ng) * I + (id[0] % ng) * 64;
        for (int i = 0; i < 64; ++i) t[i] = p[i];
        d_hadamard64(t);
        for (int i = 0; i < 64; ++i) p[i] = t[i];
      });
    }
    // SwiGLU derivative (the rounding points are straight-through)
    {
      const float *gu = gu_out_.p, *dh = dh_.p;
      float *dgu = dgu_.p;
      q_.parallel_for(sycl::range<2>(static_cast<size_t>(na_), static_cast<size_t>(I)), [=](sycl::id<2> id) {
        const size_t a = id[0], j = id[1];
        const float g = d_rbf16(gu[a * 2 * I + j]), u = d_rbf16(gu[a * 2 * I + I + j]);
        const float sg = 1.f / (1.f + sycl::exp(-g));
        const float d = dh[a * I + j];
        dgu[a * 2 * I + j] = d * u * sg * (1.f + g * (1.f - sg));
        dgu[a * 2 * I + I + j] = d * g * sg;
      });
    }
    const int lv = static_cast<int>(sycl::round(lr * kArVOne));
    // down: dW[n][k] = sum_a (w_a delta[row a][n]) hq[a][k]
    {
      const float *hq = hq_.p;
      update(dn_, lv, mm_lr, minmax,
             [=](int a, int n) { return aw[a] * dl[static_cast<size_t>(arow[a]) * H + n]; },
             [=](int a, int k) { return hq[static_cast<size_t>(a) * I + k]; });
    }
    // gate|up: dW[j][k] = sum_a dgu[a][j] xq[row a][k]
    {
      const float *dgu = dgu_.p, *xq = xq_.p;
      update(gu_, lv, mm_lr, minmax, [=](int a, int j) { return dgu[static_cast<size_t>(a) * 2 * I + j]; },
             [=](int a, int k) { return xq[static_cast<size_t>(arow[a]) * H + k]; });
    }
    q_.wait();
  }

  void save_best() override {
    for (DMat *m : {&gu_, &dn_}) {
      q_.memcpy(m->v_best, m->v, m->elems(E_) * 2);
      q_.memcpy(m->a_best, m->a, m->gelems(E_) * 4);
      q_.memcpy(m->b_best, m->b, m->gelems(E_) * 4);
    }
    q_.wait();
  }
  void restore_best() override {
    for (DMat *m : {&gu_, &dn_}) {
      q_.memcpy(m->v, m->v_best, m->elems(E_) * 2);
      q_.memcpy(m->a, m->a_best, m->gelems(E_) * 4);
      q_.memcpy(m->b, m->b_best, m->gelems(E_) * 4);
    }
    q_.wait();
  }
  void result(int local, ExpertQuant &out) override {
    const int H = s_.hidden, I = s_.inter;
    const size_t e = static_cast<size_t>(ex_[static_cast<size_t>(local)]);
    auto one = [&](const DMat &m, const uint16_t *w, std::vector<int8_t> &q, std::vector<uint16_t> &s) {
      const size_t n = static_cast<size_t>(m.rows) * m.cols, g = static_cast<size_t>(m.rows) * (m.cols / m.grp);
      std::vector<int16_t> v(n);
      std::vector<float> a(g), b(g), lo(g), hi(g);
      q_.memcpy(v.data(), m.v + static_cast<size_t>(local) * n, n * 2);
      q_.memcpy(a.data(), m.a + static_cast<size_t>(local) * g, g * 4);
      q_.memcpy(b.data(), m.b + static_cast<size_t>(local) * g, g * 4);
      q_.wait();
      ar_minmax(w, m.rows, m.cols, m.grp, lo.data(), hi.data());
      ar_codes(w, v.data(), a.data(), b.data(), lo.data(), hi.data(), m.rows, m.cols, m.grp, q, s);
    };
    one(gu_, src_.gate_up + e * 2 * I * H, out.gu_q, out.gu_s);
    one(dn_, src_.down + e * H * I, out.dn_q, out.dn_s);
  }

 private:
  // Bs[kk][nn] = Wq[e][n0 + nn][k0 + kk] for a 64 x 32 tile (8 consecutive k per item, one group).
  static void fill_w_rows(const DMat &W, float *Bs, int e, int n0, int k0, int l, bool exact) {
    const int nn = l / 4, kc = (l % 4) * 8;
    const size_t row = static_cast<size_t>(e) * W.rows + n0 + nn;
    if (exact) {
      const size_t at = row * W.cols + k0 + kc;
      for (int i = 0; i < 8; ++i) Bs[(kc + i) * TN + nn] = d_bf16(W.w[at + i]);
      return;
    }
    const int G = W.cols / W.grp;
    const size_t gi = row * G + (k0 + kc) / W.grp;
    const float s = d_scale(W.wmin[gi], W.wmax[gi], W.a[gi], W.b[gi]);
    const size_t at = row * W.cols + k0 + kc;
    for (int i = 0; i < 8; ++i) Bs[(kc + i) * TN + nn] = d_wq(d_bf16(W.w[at + i]), s, W.v[at + i] / kArVOne);
  }

  // Grouped GEMM over the step's 64-row tiles: C[m][n] = sum_k A[m][k] B[n][k], ntiles x 64 columns.
  template <class FA, class FB, class ST>
  void seg_gemm(int T, int ntiles, int K, FA fa, FB fb, ST st) {
    const int32_t *tiles = tiles_.p;
    q_.submit([&](sycl::handler &h) {
      sycl::local_accessor<float, 1> As(sycl::range<1>(TK * TM), h), Bs(sycl::range<1>(TK * TN), h);
      h.parallel_for(sycl::nd_range<2>(sycl::range<2>(static_cast<size_t>(T) * 16, static_cast<size_t>(ntiles) * 16), sycl::range<2>(16, 16)),
                     [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(16)]] {
                       const int t = static_cast<int>(it.get_group(0)), n0 = static_cast<int>(it.get_group(1)) * TN;
                       const int ty = static_cast<int>(it.get_local_id(0)), tx = static_cast<int>(it.get_local_id(1));
                       const int l = ty * 16 + tx;
                       const int e = tiles[3 * t], a0 = tiles[3 * t + 1], m = tiles[3 * t + 2];
                       float *as = As.get_multi_ptr<sycl::access::decorated::no>().get();
                       float *bs = Bs.get_multi_ptr<sycl::access::decorated::no>().get();
                       float acc[4][4] = {};
                       for (int k0 = 0; k0 < K; k0 += TK) {
                         fa(as, e, a0, m, k0, l);
                         fb(bs, e, n0, k0, l);
                         sycl::group_barrier(it.get_group());
#pragma unroll 4
                         for (int kk = 0; kk < TK; ++kk) {
                           float av[4], bv[4];
                           for (int i = 0; i < 4; ++i) av[i] = as[kk * TM + ty * 4 + i];
                           for (int j = 0; j < 4; ++j) bv[j] = bs[kk * TN + tx * 4 + j];
                           for (int i = 0; i < 4; ++i)
                             for (int j = 0; j < 4; ++j) acc[i][j] += av[i] * bv[j];
                         }
                         sycl::group_barrier(it.get_group());
                       }
                       st(e, a0, m, n0, ty, tx, acc);
                     });
    });
  }

  // dW over each expert's assignments for (64 rows x one group) tiles + the signSGD epilogue.
  template <class FO, class FI>
  void update(DMat &W, int lv, float mm_lr, bool minmax, FO gout, FI gin) {
    const int32_t *seg = seg_.p;
    const DMat M = W;
    const int G = M.cols / M.grp;
    const int ntiles = M.rows / 64;
    if (M.grp != 128) throw std::runtime_error("autoround gpu: the update kernel assumes 128-k groups");
    q_.submit([&](sycl::handler &h) {
      sycl::local_accessor<float, 1> As(sycl::range<1>(16 * 64), h), Bs(sycl::range<1>(16 * 128), h);
      h.parallel_for(sycl::nd_range<2>(sycl::range<2>(static_cast<size_t>(E_) * ntiles * 16, static_cast<size_t>(G) * 16), sycl::range<2>(16, 16)),
                     [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(16)]] {
                       const int e = static_cast<int>(it.get_group(0)) / ntiles, n0 = (static_cast<int>(it.get_group(0)) % ntiles) * 64;
                       const int g = static_cast<int>(it.get_group(1)), k0 = g * 128;
                       const int ty = static_cast<int>(it.get_local_id(0)), tx = static_cast<int>(it.get_local_id(1));
                       const int l = ty * 16 + tx;
                       const int s0 = seg[e], s1 = seg[e + 1];
                       if (s0 == s1) return;  // the whole work-group: no barrier reached
                       float *as = As.get_multi_ptr<sycl::access::decorated::no>().get();
                       float *bs = Bs.get_multi_ptr<sycl::access::decorated::no>().get();
                       float acc[4][8] = {};
                       for (int m0 = s0; m0 < s1; m0 += 16) {
                         {  // As[mm][nn] 16 x 64: 4 per item
                           const int mm = l / 16, nc = (l % 16) * 4;
                           for (int i = 0; i < 4; ++i) as[mm * 64 + nc + i] = m0 + mm < s1 ? gout(m0 + mm, n0 + nc + i) : 0.f;
                         }
                         {  // Bs[mm][kk] 16 x 128: 8 per item
                           const int mm = l / 16, kc = (l % 16) * 8;
                           for (int i = 0; i < 8; ++i) bs[mm * 128 + kc + i] = m0 + mm < s1 ? gin(m0 + mm, k0 + kc + i) : 0.f;
                         }
                         sycl::group_barrier(it.get_group());
                         for (int mm = 0; mm < 16; ++mm) {
                           float av[4], bv[8];
                           for (int i = 0; i < 4; ++i) av[i] = as[mm * 64 + ty * 4 + i];
                           for (int j = 0; j < 8; ++j) bv[j] = bs[mm * 128 + tx * 8 + j];
                           for (int i = 0; i < 4; ++i)
                             for (int j = 0; j < 8; ++j) acc[i][j] += av[i] * bv[j];
                         }
                         sycl::group_barrier(it.get_group());
                       }
                       auto sg = it.get_sub_group();  // the 16 tx of one ty: the whole group of 128 k
                       for (int i = 0; i < 4; ++i) {
                         const size_t row = static_cast<size_t>(e) * M.rows + n0 + ty * 4 + i;
                         const size_t gi = row * G + g;
                         const float wmin = M.wmin[gi], wmax = M.wmax[gi], a = M.a[gi], b = M.b[gi];
                         const float s = d_scale(wmin, wmax, a, b);
                         float gs = 0.f;
                         for (int j = 0; j < 8; ++j) {
                           const size_t at = row * M.cols + k0 + tx * 8 + j;
                           const float w = d_bf16(M.w[at]);
                           const int16_t v0 = M.v[at];
                           const float t = sycl::rint(w / s + v0 / kArVOne);
                           const bool inr = t >= -8.f && t <= 7.f;
                           const float qv = sycl::fmin(7.f, sycl::fmax(-8.f, t));
                           const float gw = acc[i][j];
                           gs += gw * (inr ? qv - w / s : qv);
                           const float gv = gw * s;  // dL/dv = dL/dw_q * s: the scale is signed
                           if (inr && gv != 0.f) {
                             const int nv = v0 + (gv > 0.f ? -lv : lv);
                             M.v[at] = static_cast<int16_t>(nv < -32767 ? -32767 : nv > 32767 ? 32767 : nv);
                           }
                         }
                         gs = sycl::reduce_over_group(sg, gs, sycl::plus<float>());
                         if (tx == 0 && minmax && gs != 0.f) {
                           if (-wmin * a >= wmax * b) {
                             const float ga = gs * (-wmin);
                             if (ga != 0.f) M.a[gi] = sycl::fmin(1.f, sycl::fmax(0.f, a - (ga > 0.f ? mm_lr : -mm_lr)));
                           } else {
                             const float gb = gs * (-wmax);
                             if (gb != 0.f) M.b[gi] = sycl::fmin(1.f, sycl::fmax(0.f, b - (gb > 0.f ? mm_lr : -mm_lr)));
                           }
                         }
                       }
                     });
    });
  }

  int dev_;
  sycl::queue q_;
  ArShape s_;
  std::vector<int> ex_;
  int E_ = 0, na_ = 0;
  ExpertSource src_;
  bool aq_ = true, exact_ = false;
  DMat gu_, dn_;
  const ArBatch *b_ = nullptr;
  std::vector<int32_t> tiles_h_;
  DBuf<uint16_t> xb_;
  DBuf<float> xq_, part_, delta_, asg_w_, gu_out_, hq_, y_, dh_, dgu_;
  DBuf<int32_t> asg_row_, tiles_, seg_, row_asg_;
};

}  // namespace

int ar_gpu_count() {
  try {
    return static_cast<int>(gpu_devices().size());
  } catch (const std::exception &) {
    return 0;
  }
}

std::unique_ptr<ArWorker> make_gpu_worker(int device) { return std::make_unique<GpuWorker>(device); }

}  // namespace xenithium:quant
