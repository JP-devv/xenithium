#include "xenithium/quant/autoround.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <functional>
#include <numeric>
#include <stdexcept>
#include <thread>

namespace xenithium::quant {

// ---- fp16 ------------------------------------------------------------------------------------
uint16_t ar_half_bits(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  const uint32_t sign = (u >> 16) & 0x8000u;
  const uint32_t a = u & 0x7fffffffu;
  if (a > 0x7f800000u) return static_cast<uint16_t>(sign | 0x7e00u);
  if (a >= 0x477ff000u) return static_cast<uint16_t>(sign | 0x7c00u);  // rounds past 65504
  if (a < 0x38800000u) {  // subnormal half (or zero): value / 2^-24, round to nearest even
    float v;
    std::memcpy(&v, &a, 4);
    const float q = std::nearbyint(v * 16777216.f);
    return static_cast<uint16_t>(sign | static_cast<uint32_t>(q));
  }
  const uint32_t mant = a & 0x7fffffu;
  const uint32_t exp = (a >> 23) - 127 + 15;
  uint32_t h = (exp << 10) | (mant >> 13);
  const uint32_t rest = mant & 0x1fffu;
  if (rest > 0x1000u || (rest == 0x1000u && (h & 1u))) ++h;  // carries into the exponent correctly
  return static_cast<uint16_t>(sign | h);
}

float ar_half(uint16_t h) {
  const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
  const uint32_t exp = (h >> 10) & 0x1fu, mant = h & 0x3ffu;
  uint32_t u;
  if (exp == 0) {
    const float v = static_cast<float>(mant) * 5.9604645e-8f;
    std::memcpy(&u, &v, 4);
    u |= sign;
  } else if (exp == 31) {
    u = sign | 0x7f800000u | (mant << 13);
  } else {
    u = sign | ((exp - 15 + 127) << 23) | (mant << 13);
  }
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

void ar_fake_quant_rows(const float *src, int rows, int cols, int group, float *dst) {
  for (int r = 0; r < rows; ++r)
    for (int g = 0; g < cols / group; ++g) {
      const float *v = src + static_cast<size_t>(r) * cols + static_cast<size_t>(g) * group;
      float *o = dst + static_cast<size_t>(r) * cols + static_cast<size_t>(g) * group;
      float amax = 0.f;
      for (int i = 0; i < group; ++i) amax = std::max(amax, std::fabs(v[i]));
      if (amax == 0.f) {
        for (int i = 0; i < group; ++i) o[i] = 0.f;
        continue;
      }
      const float inv = 127.f / amax, s = amax / 127.f;
      for (int i = 0; i < group; ++i) {
        float t = std::nearbyint(v[i] * inv);
        t = std::min(127.f, std::max(-127.f, t));
        o[i] = t * s;
      }
    }
}

namespace {

void parallel_for(int n, int threads, const std::function<void(int)> &fn) {
  threads = std::max(1, std::min(threads, n));
  if (threads == 1) {
    for (int i = 0; i < n; ++i) fn(i);
    return;
  }
  std::atomic<int> next{0};
  std::vector<std::thread> pool;
  std::exception_ptr err;
  std::atomic<bool> failed{false};
  for (int t = 0; t < threads; ++t)
    pool.emplace_back([&] {
      try {
        for (int i; (i = next.fetch_add(1)) < n && !failed;) fn(i);
      } catch (...) {
        if (!failed.exchange(true)) err = std::current_exception();
      }
    });
  for (auto &t : pool) t.join();
  if (err) std::rethrow_exception(err);
}

// 16 independent accumulators: vectorises under precise fp (no reassociation needed).
inline float dot(const float *a, const float *b, int n) {
  float acc[16] = {};
  int i = 0;
  for (; i + 16 <= n; i += 16)
    for (int j = 0; j < 16; ++j) acc[j] += a[i + j] * b[i + j];
  float s = 0.f;
  for (int j = 0; j < 16; ++j) s += acc[j];
  for (; i < n; ++i) s += a[i] * b[i];
  return s;
}
inline void axpy(float a, const float *x, float *y, int n) {
  for (int i = 0; i < n; ++i) y[i] += a * x[i];
}

inline float silu(float x) { return x / (1.f + std::exp(-x)); }

// The parameters of one matrix [n][k] with groups of `grp` along k.
struct MatParams {
  int n = 0, k = 0, grp = 0;
  const uint16_t *w = nullptr;  // bf16 [n][k] (the source, not copied)
  std::vector<int16_t> v, v_best;
  std::vector<float> a, b, a_best, b_best, wmin, wmax;  // [n][k / grp]
  void init(const uint16_t *src, int n_, int k_, int g_) {
    n = n_, k = k_, grp = g_, w = src;
    const size_t ng = static_cast<size_t>(n) * (k / grp);
    v.assign(static_cast<size_t>(n) * k, 0);
    a.assign(ng, 1.f);
    b.assign(ng, 1.f);
    wmin.assign(ng, 0.f);
    wmax.assign(ng, 0.f);
    for (int r = 0; r < n; ++r)
      for (int g = 0; g < k / grp; ++g) {
        float lo = 0.f, hi = 0.f;
        for (int i = 0; i < grp; ++i) {
          const float x = ar_bf16(w[static_cast<size_t>(r) * k + g * grp + i]);
          lo = std::min(lo, x);
          hi = std::max(hi, x);
        }
        wmin[static_cast<size_t>(r) * (k / grp) + g] = lo;
        wmax[static_cast<size_t>(r) * (k / grp) + g] = hi;
      }
    v_best = v, a_best = a, b_best = b;
  }
  float scale(int r, int g) const {
    const size_t i = static_cast<size_t>(r) * (k / grp) + g;
    return ar_scale(wmin[i], wmax[i], a[i], b[i]);
  }
  // dequantised [n][k] fp32
  void dequant(std::vector<float> &out) const {
    out.resize(static_cast<size_t>(n) * k);
    for (int r = 0; r < n; ++r)
      for (int g = 0; g < k / grp; ++g) {
        const float s = scale(r, g);
        for (int i = 0; i < grp; ++i) {
          const size_t at = static_cast<size_t>(r) * k + g * grp + i;
          out[at] = s * static_cast<float>(ar_code(ar_bf16(w[at]), s, v[at] / kArVOne));
        }
      }
  }
  // one signSGD step from the gradient wrt w_q (grad [n][k])
  void step(const float *grad, float lr, float mm_lr, bool minmax) {
    const int G = k / grp;
    const int lv = static_cast<int>(std::lround(lr * kArVOne));
    for (int r = 0; r < n; ++r)
      for (int g = 0; g < G; ++g) {
        const size_t gi = static_cast<size_t>(r) * G + g;
        const float s = scale(r, g);
        double gs = 0;
        for (int i = 0; i < grp; ++i) {
          const size_t at = static_cast<size_t>(r) * k + g * grp + i;
          const float wf = ar_bf16(w[at]);
          bool inr = false;
          const int q = ar_code(wf, s, v[at] / kArVOne, &inr);
          const float gw = grad[at];
          gs += static_cast<double>(gw) * (inr ? (static_cast<float>(q) - wf / s) : static_cast<float>(q));
          const float gv = inr ? gw * s : 0.f;
          if (gv != 0.f) {
            int nv = v[at] + (gv > 0.f ? -lv : lv);
            v[at] = static_cast<int16_t>(std::max(-32767, std::min(32767, nv)));
          }
        }
        if (!minmax || gs == 0) continue;
        // s = a / 8 (a >= b) or -b / 8: ds/dalpha = -wmin / 8, ds/dbeta = -wmax / 8
        const float am = -wmin[gi] * a[gi], bm = wmax[gi] * b[gi];
        if (am >= bm) {
          const double ga = gs * (-wmin[gi]) / kArQMax;
          if (ga != 0) a[gi] = std::min(1.f, std::max(0.f, a[gi] - (ga > 0 ? mm_lr : -mm_lr)));
        } else {
          const double gb = gs * (-wmax[gi]) / kArQMax;
          if (gb != 0) b[gi] = std::min(1.f, std::max(0.f, b[gi] - (gb > 0 ? mm_lr : -mm_lr)));
        }
      }
  }
  void codes(std::vector<int8_t> &q, std::vector<uint16_t> &s) const {
    q.resize(static_cast<size_t>(n) * k);
    s.resize(static_cast<size_t>(n) * (k / grp));
    for (int r = 0; r < n; ++r)
      for (int g = 0; g < k / grp; ++g) {
        const float sc = scale(r, g);
        s[static_cast<size_t>(r) * (k / grp) + g] = ar_half_bits(sc);
        for (int i = 0; i < grp; ++i) {
          const size_t at = static_cast<size_t>(r) * k + g * grp + i;
          q[at] = static_cast<int8_t>(ar_code(ar_bf16(w[at]), sc, v[at] / kArVOne));
        }
      }
  }
};

class CpuWorker final : public ArWorker {
 public:
  explicit CpuWorker(int threads) : threads_(std::max(1, threads)) {}
  std::string name() const override { return "cpu x" + std::to_string(threads_); }
  double speed_hint() const override { return 0.004 * threads_; }  // relative to about 1 per GPU
  void act_quant(bool on) override { aq_ = on; }
  void exact(bool on) override { exact_ = on; }
  void setup(const ArShape &s, const std::vector<int> &experts, const ExpertSource &src) override {
    if (s.rot_h) throw std::runtime_error("cpu worker: no rotation (the GPU worker only)");
    s_ = s;
    ex_ = experts;
    const size_t H = s.hidden, I = s.inter;
    gu_.resize(experts.size());
    dn_.resize(experts.size());
    parallel_for(static_cast<int>(experts.size()), threads_, [&](int i) {
      const size_t e = static_cast<size_t>(experts[static_cast<size_t>(i)]);
      gu_[static_cast<size_t>(i)].init(src.gate_up + e * 2 * I * H, static_cast<int>(2 * I), static_cast<int>(H), s.wgroup);
      dn_[static_cast<size_t>(i)].init(src.down + e * H * I, static_cast<int>(H), static_cast<int>(I), s.wgroup);
    });
  }
  void begin(const ArBatch &b) override {
    b_ = &b;
    const int H = s_.hidden;
    xq_.resize(static_cast<size_t>(b.rows) * H);
    for (size_t i = 0; i < xq_.size(); ++i) xq_[i] = ar_bf16(b.x[i]);
    if (aq_ && !exact_) {
      std::vector<float> tmp(xq_);
      parallel_for(b.rows, threads_, [&](int r) {
        ar_fake_quant_rows(tmp.data() + static_cast<size_t>(r) * H, 1, H, s_.xgroup, xq_.data() + static_cast<size_t>(r) * H);
      });
    }
    const size_t na = b.asg_row.size();
    gu_out_.assign(na * 2 * s_.inter, 0.f);
    hq_.assign(na * s_.inter, 0.f);
    y_.assign(na * static_cast<size_t>(H), 0.f);
  }
  void forward(float *part) override {
    const ArBatch &b = *b_;
    const int H = s_.hidden, I = s_.inter;
    parallel_for(static_cast<int>(ex_.size()), threads_, [&](int e) {
      const int a0 = b.seg_start[static_cast<size_t>(e)], a1 = b.seg_start[static_cast<size_t>(e) + 1];
      if (a0 == a1) return;
      thread_local std::vector<float> wg, wd, h;
      if (exact_) {
        for (auto [m, out] : {std::pair<const MatParams *, std::vector<float> *>{&gu_[static_cast<size_t>(e)], &wg}, {&dn_[static_cast<size_t>(e)], &wd}}) {
          out->resize(static_cast<size_t>(m->n) * m->k);
          for (size_t i = 0; i < out->size(); ++i) (*out)[i] = ar_bf16(m->w[i]);
        }
      } else {
        gu_[static_cast<size_t>(e)].dequant(wg);
        dn_[static_cast<size_t>(e)].dequant(wd);
      }
      h.resize(static_cast<size_t>(I));
      for (int a = a0; a < a1; ++a) {
        const float *x = xq_.data() + static_cast<size_t>(b.asg_row[static_cast<size_t>(a)]) * H;
        float *gu = gu_out_.data() + static_cast<size_t>(a) * 2 * I;
        for (int j = 0; j < 2 * I; ++j) gu[j] = dot(wg.data() + static_cast<size_t>(j) * H, x, H);
        for (int j = 0; j < I; ++j) h[static_cast<size_t>(j)] = ar_round_bf16(silu(ar_round_bf16(gu[j])) * ar_round_bf16(gu[I + j]));
        float *hq = hq_.data() + static_cast<size_t>(a) * I;
        if (aq_ && !exact_) ar_fake_quant_rows(h.data(), 1, I, s_.hgroup, hq);
        else std::copy(h.begin(), h.end(), hq);
        float *y = y_.data() + static_cast<size_t>(a) * H;
        for (int n = 0; n < H; ++n) y[n] = dot(wd.data() + static_cast<size_t>(n) * I, hq, I);
      }
    });
    parallel_for(b.rows, threads_, [&](int r) {
      float *p = part + static_cast<size_t>(r) * H;
      std::fill(p, p + H, 0.f);
      for (int k = 0; k < s_.topk; ++k) {
        const int a = b.row_asg[static_cast<size_t>(r) * s_.topk + k];
        if (a < 0) continue;
        axpy(b.asg_w[static_cast<size_t>(a)], y_.data() + static_cast<size_t>(a) * H, p, H);
      }
    });
  }
  void backward(const float *delta, float lr, float mm_lr, bool minmax) override {
    const ArBatch &b = *b_;
    const int H = s_.hidden, I = s_.inter;
    parallel_for(static_cast<int>(ex_.size()), threads_, [&](int e) {
      const int a0 = b.seg_start[static_cast<size_t>(e)], a1 = b.seg_start[static_cast<size_t>(e) + 1];
      if (a0 == a1) return;
      thread_local std::vector<float> wg, wd, dwg, dwd, dy, dh, dgu;
      gu_[static_cast<size_t>(e)].dequant(wg);
      dn_[static_cast<size_t>(e)].dequant(wd);
      dwg.assign(wg.size(), 0.f);
      dwd.assign(wd.size(), 0.f);
      dy.resize(static_cast<size_t>(H));
      dh.resize(static_cast<size_t>(I));
      dgu.resize(static_cast<size_t>(2 * I));
      for (int a = a0; a < a1; ++a) {
        const int r = b.asg_row[static_cast<size_t>(a)];
        const float wa = b.asg_w[static_cast<size_t>(a)];
        for (int n = 0; n < H; ++n) dy[static_cast<size_t>(n)] = wa * delta[static_cast<size_t>(r) * H + n];
        // dh = Wd^T dy; dWd += dy hq^T
        std::fill(dh.begin(), dh.end(), 0.f);
        const float *hq = hq_.data() + static_cast<size_t>(a) * I;
        for (int n = 0; n < H; ++n) {
          const float g = dy[static_cast<size_t>(n)];
          axpy(g, wd.data() + static_cast<size_t>(n) * I, dh.data(), I);
          axpy(g, hq, dwd.data() + static_cast<size_t>(n) * I, I);
        }
        const float *gu = gu_out_.data() + static_cast<size_t>(a) * 2 * I;
        for (int j = 0; j < I; ++j) {
          const float g = ar_round_bf16(gu[j]), u = ar_round_bf16(gu[I + j]);
          const float sg = 1.f / (1.f + std::exp(-g));
          dgu[static_cast<size_t>(j)] = dh[static_cast<size_t>(j)] * u * sg * (1.f + g * (1.f - sg));
          dgu[static_cast<size_t>(I + j)] = dh[static_cast<size_t>(j)] * g * sg;
        }
        const float *x = xq_.data() + static_cast<size_t>(r) * H;
        for (int j = 0; j < 2 * I; ++j) axpy(dgu[static_cast<size_t>(j)], x, dwg.data() + static_cast<size_t>(j) * H, H);
      }
      dn_[static_cast<size_t>(e)].step(dwd.data(), lr, mm_lr, minmax);
      gu_[static_cast<size_t>(e)].step(dwg.data(), lr, mm_lr, minmax);
    });
  }
  void save_best() override {
    parallel_for(static_cast<int>(ex_.size()), threads_, [&](int e) {
      for (MatParams *m : {&gu_[static_cast<size_t>(e)], &dn_[static_cast<size_t>(e)]}) m->v_best = m->v, m->a_best = m->a, m->b_best = m->b;
    });
  }
  void restore_best() override {
    parallel_for(static_cast<int>(ex_.size()), threads_, [&](int e) {
      for (MatParams *m : {&gu_[static_cast<size_t>(e)], &dn_[static_cast<size_t>(e)]}) m->v = m->v_best, m->a = m->a_best, m->b = m->b_best;
    });
  }
  void result(int local, ExpertQuant &out) override {
    gu_[static_cast<size_t>(local)].codes(out.gu_q, out.gu_s);
    dn_[static_cast<size_t>(local)].codes(out.dn_q, out.dn_s);
  }

 private:
  int threads_;
  bool aq_ = true, exact_ = false;
  ArShape s_;
  std::vector<int> ex_;
  std::vector<MatParams> gu_, dn_;
  const ArBatch *b_ = nullptr;
  std::vector<float> xq_, gu_out_, hq_, y_;
};

using clk = std::chrono::steady_clock;
double secs(clk::time_point t) { return std::chrono::duration<double>(clk::now() - t).count(); }

struct Splitmix {
  uint64_t s;
  uint64_t next() {
    uint64_t z = (s += 0x9e3779b97f4a7c15ull);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
  }
};

// The batches of one step, one per worker, over the given samples.
struct StepRows {
  std::vector<uint16_t> x;  // [rows][hidden]
  std::vector<int64_t> src_row;  // [rows] data row
  int rows = 0;
};

void gather_rows(const ArShape &s, const LayerData &d, const std::vector<int> &samples, StepRows &st, int threads) {
  st.rows = static_cast<int>(samples.size()) * d.seqlen;
  st.x.resize(static_cast<size_t>(st.rows) * s.hidden);
  st.src_row.resize(static_cast<size_t>(st.rows));
  parallel_for(static_cast<int>(samples.size()), threads, [&](int i) {
    const int64_t r0 = static_cast<int64_t>(samples[static_cast<size_t>(i)]) * d.seqlen;
    std::memcpy(st.x.data() + static_cast<size_t>(i) * d.seqlen * s.hidden, d.x + r0 * s.hidden,
                static_cast<size_t>(d.seqlen) * s.hidden * 2);
    for (int j = 0; j < d.seqlen; ++j) st.src_row[static_cast<size_t>(i) * d.seqlen + j] = r0 + j;
  });
}

void build_batches(const ArShape &s, const LayerData &d, const StepRows &st, const std::vector<int> &owner,
                   const std::vector<int> &local, std::vector<ArBatch> &bs, const std::vector<int> &n_local) {
  const size_t W = bs.size();
  for (size_t w = 0; w < W; ++w) {
    ArBatch &b = bs[w];
    b.rows = st.rows;
    b.x = st.x.data();
    b.seg_start.assign(static_cast<size_t>(n_local[w]) + 1, 0);
    b.row_asg.assign(static_cast<size_t>(st.rows) * s.topk, -1);
  }
  // counts
  for (int r = 0; r < st.rows; ++r)
    for (int k = 0; k < s.topk; ++k) {
      const int e = d.ids[st.src_row[static_cast<size_t>(r)] * s.topk + k];
      ++bs[static_cast<size_t>(owner[static_cast<size_t>(e)])].seg_start[static_cast<size_t>(local[static_cast<size_t>(e)]) + 1];
    }
  std::vector<std::vector<int32_t>> fill(W);
  for (size_t w = 0; w < W; ++w) {
    ArBatch &b = bs[w];
    for (size_t i = 1; i < b.seg_start.size(); ++i) b.seg_start[i] += b.seg_start[i - 1];
    b.asg_row.assign(static_cast<size_t>(b.seg_start.back()), 0);
    b.asg_w.assign(b.asg_row.size(), 0.f);
    fill[w].assign(b.seg_start.begin(), b.seg_start.end() - 1);
  }
  for (int r = 0; r < st.rows; ++r)
    for (int k = 0; k < s.topk; ++k) {
      const int64_t dr = st.src_row[static_cast<size_t>(r)];
      const int e = d.ids[dr * s.topk + k];
      const size_t w = static_cast<size_t>(owner[static_cast<size_t>(e)]);
      ArBatch &b = bs[w];
      const int a = fill[w][static_cast<size_t>(local[static_cast<size_t>(e)])]++;
      b.asg_row[static_cast<size_t>(a)] = r;
      b.asg_w[static_cast<size_t>(a)] = d.w[dr * s.topk + k];
      b.row_asg[static_cast<size_t>(r) * s.topk + k] = a;
    }
}

// Sum the workers' partials, the loss, and (optionally) delta. Block loss per element of the
// 4-stream output: (G ||sum - T||^2 + c) / (rows * 4 * hidden).
double loss_and_delta(const ArShape &s, const LayerData &d, const StepRows &st, std::vector<std::vector<float>> &parts,
                      float *delta, int threads) {
  const int H = s.hidden;
  const double norm = static_cast<double>(st.rows) * 4.0 * H;
  std::vector<double> lr(static_cast<size_t>(st.rows), 0.0);
  parallel_for(st.rows, threads, [&](int r) {
    const int64_t dr = st.src_row[static_cast<size_t>(r)];
    const float G = d.g ? d.g[dr] : 1.f;
    double acc = 0;
    for (int n = 0; n < H; ++n) {
      float sum = 0.f;
      for (auto &p : parts) sum += p[static_cast<size_t>(r) * H + n];
      const float e = sum - ar_bf16(d.t[dr * H + n]);
      acc += static_cast<double>(e) * e;
      if (delta) delta[static_cast<size_t>(r) * H + n] = static_cast<float>(2.0 * G * e / norm);
    }
    lr[static_cast<size_t>(r)] = G * acc + (d.c ? d.c[dr] : 0.f);
  });
  double tot = 0;
  for (const double v : lr) tot += v;
  return tot / norm;
}

}  // namespace

std::unique_ptr<ArWorker> make_cpu_worker(int threads) { return std::make_unique<CpuWorker>(threads); }

void ar_minmax(const uint16_t *w, int n, int k, int grp, float *wmin, float *wmax) {
  for (int r = 0; r < n; ++r)
    for (int g = 0; g < k / grp; ++g) {
      float lo = 0.f, hi = 0.f;
      for (int i = 0; i < grp; ++i) {
        const float x = ar_bf16(w[static_cast<size_t>(r) * k + g * grp + i]);
        lo = std::min(lo, x);
        hi = std::max(hi, x);
      }
      wmin[static_cast<size_t>(r) * (k / grp) + g] = lo;
      wmax[static_cast<size_t>(r) * (k / grp) + g] = hi;
    }
}

void ar_codes(const uint16_t *w, const int16_t *v, const float *a, const float *b, const float *wmin, const float *wmax, int n, int k,
              int grp, std::vector<int8_t> &q, std::vector<uint16_t> &s) {
  const int G = k / grp;
  q.resize(static_cast<size_t>(n) * k);
  s.resize(static_cast<size_t>(n) * G);
  for (int r = 0; r < n; ++r)
    for (int g = 0; g < G; ++g) {
      const size_t gi = static_cast<size_t>(r) * G + g;
      const float sc = ar_scale(wmin[gi], wmax[gi], a[gi], b[gi]);
      s[gi] = ar_half_bits(sc);
      for (int i = 0; i < grp; ++i) {
        const size_t at = static_cast<size_t>(r) * k + g * grp + i;
        q[at] = static_cast<int8_t>(ar_code(ar_bf16(w[at]), sc, v[at] / kArVOne));
      }
    }
}

void rtn_expert(const ArShape &s, const ExpertSource &src, int expert, ExpertQuant &out) {
  const size_t H = s.hidden, I = s.inter, e = static_cast<size_t>(expert);
  MatParams gu, dn;
  gu.init(src.gate_up + e * 2 * I * H, static_cast<int>(2 * I), static_cast<int>(H), s.wgroup);
  dn.init(src.down + e * H * I, static_cast<int>(H), static_cast<int>(I), s.wgroup);
  gu.codes(out.gu_q, out.gu_s);
  dn.codes(out.dn_q, out.dn_s);
}

namespace {
struct Split {
  std::vector<int> owner, local, n_local;
};
Split split_experts(const ArShape &s, const LayerData &d, std::vector<std::unique_ptr<ArWorker>> &workers, int train_samples,
                    std::vector<int64_t> &routes) {
  routes.assign(static_cast<size_t>(s.experts), 0);
  const int64_t rows = static_cast<int64_t>(train_samples) * d.seqlen;
  for (int64_t i = 0; i < rows * s.topk; ++i) ++routes[static_cast<size_t>(d.ids[i])];
  std::vector<int> order(static_cast<size_t>(s.experts));
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return routes[static_cast<size_t>(a)] > routes[static_cast<size_t>(b)]; });
  const size_t W = workers.size();
  std::vector<double> load(W, 0.0), speed(W);
  for (size_t w = 0; w < W; ++w) speed[w] = std::max(1e-6, workers[w]->speed_hint());
  Split sp;
  sp.owner.assign(static_cast<size_t>(s.experts), 0);
  for (const int e : order) {
    size_t best = 0;
    double bv = 1e300;
    for (size_t w = 0; w < W; ++w) {
      const double v = (load[w] + static_cast<double>(routes[static_cast<size_t>(e)]) + 1.0) / speed[w];
      if (v < bv) bv = v, best = w;
    }
    sp.owner[static_cast<size_t>(e)] = static_cast<int>(best);
    load[best] += static_cast<double>(routes[static_cast<size_t>(e)]) + 1.0;
  }
  sp.local.assign(static_cast<size_t>(s.experts), 0);
  sp.n_local.assign(W, 0);
  for (int e = 0; e < s.experts; ++e) sp.local[static_cast<size_t>(e)] = sp.n_local[static_cast<size_t>(sp.owner[static_cast<size_t>(e)])]++;
  return sp;
}
}  // namespace

double ar_eval(const ArShape &s, const LayerData &d, std::vector<std::unique_ptr<ArWorker>> &workers, const std::vector<int> &owner,
               int s0, int s1, int threads) {
  std::vector<int> local(static_cast<size_t>(s.experts)), n_local(workers.size(), 0);
  for (int e = 0; e < s.experts; ++e) local[static_cast<size_t>(e)] = n_local[static_cast<size_t>(owner[static_cast<size_t>(e)])]++;
  double tot = 0;
  int64_t rows = 0;
  const int step = 8;
  StepRows st;
  std::vector<ArBatch> bs(workers.size());
  std::vector<std::vector<float>> parts(workers.size());
  for (int a = s0; a < s1; a += step) {
    std::vector<int> smp;
    for (int i = a; i < std::min(s1, a + step); ++i) smp.push_back(i);
    gather_rows(s, d, smp, st, threads);
    build_batches(s, d, st, owner, local, bs, n_local);
    std::vector<std::thread> th;
    for (size_t w = 0; w < workers.size(); ++w) {
      parts[w].assign(static_cast<size_t>(st.rows) * s.hidden, 0.f);
      th.emplace_back([&, w] {
        workers[w]->begin(bs[w]);
        workers[w]->forward(parts[w].data());
      });
    }
    for (auto &t : th) t.join();
    tot += loss_and_delta(s, d, st, parts, nullptr, threads) * st.rows;
    rows += st.rows;
  }
  return rows ? tot / static_cast<double>(rows) : 0.0;
}

ArLayerStats tune_layer(const ArShape &s, const ArConfig &cfg, const LayerData &d, const ExpertSource &src,
                        std::vector<std::unique_ptr<ArWorker>> &workers, int eval_samples, std::vector<ExpertQuant> &out) {
  const auto t0 = clk::now();
  if (workers.empty()) throw std::invalid_argument("autoround: no workers");
  if (eval_samples >= d.samples) throw std::invalid_argument("autoround: eval samples >= samples");
  const int threads = static_cast<int>(std::max(4u, std::thread::hardware_concurrency()));
  const int train = d.samples - eval_samples;
  ArLayerStats st;
  std::vector<int64_t> routes;
  const Split sp = split_experts(s, d, workers, train, routes);
  for (size_t w = 0; w < workers.size(); ++w) {
    std::vector<int> mine;
    int64_t na = 0;
    for (int e = 0; e < s.experts; ++e)
      if (sp.owner[static_cast<size_t>(e)] == static_cast<int>(w)) mine.push_back(e), na += routes[static_cast<size_t>(e)];
    st.workers.push_back(workers[w]->name());
    st.experts_per_worker.push_back(static_cast<int>(mine.size()));
    st.assignments_per_worker.push_back(na);
  }
  {
    std::vector<std::thread> th;
    for (size_t w = 0; w < workers.size(); ++w)
      th.emplace_back([&, w] {
        std::vector<int> mine;
        for (int e = 0; e < s.experts; ++e)
          if (sp.owner[static_cast<size_t>(e)] == static_cast<int>(w)) mine.push_back(e);
        workers[w]->setup(s, mine, src);
        workers[w]->act_quant(cfg.act_quant);
      });
    for (auto &t : th) t.join();
  }
  if (eval_samples > 0) st.rtn_loss = ar_eval(s, d, workers, sp.owner, train, d.samples, threads);

  const int iters = cfg.iters;
  const float lr0 = cfg.lr > 0 ? cfg.lr : (iters > 0 ? 1.f / iters : 0.f);
  const float mm0 = cfg.minmax_lr > 0 ? cfg.minmax_lr : lr0;
  Splitmix rng{cfg.seed};
  StepRows rows;
  std::vector<ArBatch> bs(workers.size());
  std::vector<std::vector<float>> parts(workers.size());
  std::vector<float> delta;
  double best = 1e300;
  for (int it = 0; it < iters; ++it) {
    // AutoRound: a random batch of samples per step (without replacement inside the step)
    std::vector<int> smp;
    const int nb = std::min(cfg.batch_samples, train);
    while (static_cast<int>(smp.size()) < nb) {
      const int c = static_cast<int>(rng.next() % static_cast<uint64_t>(train));
      if (std::find(smp.begin(), smp.end(), c) == smp.end()) smp.push_back(c);
    }
    gather_rows(s, d, smp, rows, threads);
    build_batches(s, d, rows, sp.owner, sp.local, bs, sp.n_local);
    {
      std::vector<std::thread> th;
      for (size_t w = 0; w < workers.size(); ++w) {
        parts[w].assign(static_cast<size_t>(rows.rows) * s.hidden, 0.f);
        th.emplace_back([&, w] {
          workers[w]->begin(bs[w]);
          workers[w]->forward(parts[w].data());
        });
      }
      for (auto &t : th) t.join();
    }
    delta.resize(static_cast<size_t>(rows.rows) * s.hidden);
    const double loss = loss_and_delta(s, d, rows, parts, delta.data(), threads);
    st.step_loss.push_back(loss);
    if (loss < best) {
      best = loss;
      st.best_iter = it;
      for (auto &w : workers) w->save_best();
    }
    const float decay = 1.f - static_cast<float>(it) / static_cast<float>(iters);
    {
      std::vector<std::thread> th;
      for (size_t w = 0; w < workers.size(); ++w)
        th.emplace_back([&, w] { workers[w]->backward(delta.data(), lr0 * decay, mm0 * decay, cfg.minmax); });
      for (auto &t : th) t.join();
    }
    if (cfg.log_every > 0 && (it % cfg.log_every == 0 || it + 1 == iters))
      std::fprintf(stderr, "  iter %4d  loss %.6e  best %.6e @ %d  %.1f s\n", it, loss, best, st.best_iter, secs(t0));
  }
  if (iters > 0) {
    for (auto &w : workers) w->restore_best();
    st.best_step_loss = best;
  }
  if (eval_samples > 0) st.tuned_loss = iters > 0 ? ar_eval(s, d, workers, sp.owner, train, d.samples, threads) : st.rtn_loss;
  if (eval_samples > 0 && cfg.floor) {  // the same rows through the exact bf16 experts (what no quant can beat)
    for (auto &w : workers) w->exact(true);
    st.floor_loss = ar_eval(s, d, workers, sp.owner, train, d.samples, threads);
    for (auto &w : workers) w->exact(false);
  }
  out.assign(static_cast<size_t>(s.experts), {});
  {
    std::vector<std::thread> th;
    for (size_t w = 0; w < workers.size(); ++w)
      th.emplace_back([&, w] {
        for (int e = 0; e < s.experts; ++e)
          if (sp.owner[static_cast<size_t>(e)] == static_cast<int>(w)) workers[w]->result(sp.local[static_cast<size_t>(e)], out[static_cast<size_t>(e)]);
      });
    for (auto &t : th) t.join();
  }
  st.seconds = secs(t0);
  return st;
}

void ar_hadamard_blocks(float *v, size_t len, int n) {
  if (n <= 1 || (n & (n - 1)) || len % static_cast<size_t>(n)) throw std::invalid_argument("ar_hadamard_blocks: n must be a power of 2 dividing len");
  const float sc = 1.f / std::sqrt(static_cast<float>(n));
  for (size_t b = 0; b < len; b += static_cast<size_t>(n)) {
    float *x = v + b;
    for (int hlen = 1; hlen < n; hlen <<= 1)
      for (int i = 0; i < n; i += hlen << 1)
        for (int j = i; j < i + hlen; ++j) {
          const float a = x[j], c2 = x[j + hlen];
          x[j] = a + c2, x[j + hlen] = a - c2;
        }
    for (int i = 0; i < n; ++i) x[i] *= sc;
  }
}

}  // namespace xenithium:quant
