// Xenithium · AutoRound for the routed experts (W4A8), block-output objective.
// The scheme (AutoRound's sym int4 defaults): per output row n and 128-k group g,
// a = -wmin * alpha, b = wmax * beta wmin = min(0, min w), wmax = max(0, max w)
// s = fp16( a >= b ? a / 8: -b / 8 ) (signed: the dominant extreme -> code -8)
// q = clamp( rne(w / s + v), -8, 7 ) v in [-0.5, 0.5] per weight (learned rounding)
// w_q = s * q (the bake stores q + 8 and the fp16 s)
// alpha, beta in [0, 1] (learned clipping, init 1), v init 0. Iters = 0 is plain RTN at the same
// format (the baseline). signSGD with lr = 1 / iters for v and for alpha / beta, linearly decayed
// to 0, 8 samples per step, the parameters of the lowest-loss step kept (AutoRound's best tracking).
// Activations are fake-quantised exactly where the W4A8 prefill kernels quantise them
// (row_quant.hpp: amax / 127 per (row, 128-group) of x before gate_up, per (row, 64-group) of h
// before down), and h = bf16(silu(bf16 g) * bf16 u) as the kernels round it, so the rounding is
// tuned against what moe_rows(a8) computes. Gradients pass the rounding and the activation
// quantisers straight through (STE); the clamp stops them.
// Objective: AutoRound's block-output MSE. In this model a layer's block output is the 4-stream
// residual after the MoE output m is combined in, out_s = res_s + g_s m (g_s = 2 sigmoid(inj_s / 4)
// from the MLP mixer, fixed by the inputs), and only the routed experts are trained, so
// sum_s || res_q,s + g_s m - out_fp,s ||^2 = G || m - t ||^2 + c, G = sum_s g_s^2,
// t = sum_s g_s (out_fp,s - res_q,s) / G, c = the rest (independent of the experts).
// With m = sum_k w_k y_k + shared(x) the routed sum's target is T = t - shared(x). The capture
// stage writes x, the routing, T, G and c per row; this file only sees that LayerData.
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace xenithium::quant {

struct ArShape {
  int hidden = 2560, inter = 640, experts = 512, topk = 10;
  int wgroup = 128;  // weight group (k)
  int xgroup = 128;  // A8 group of x (gate_up input)
  int hgroup = 64;  // A8 group of h (down input)
  // Block Hadamard rotation (0 = off). rot_h: the worker rotates h per block before
  // its A8 quant (== hgroup) and expects W_dn already rotated along k; rot_x is applied by the caller
  // to x and W_gu (the worker sees rotated data), recorded here.
  int rot_x = 0, rot_h = 0;
};
// In place: v[i * n.. (i + 1) * n) <- H_n v (normalised Walsh-Hadamard, n a power of 2) for each block.
void ar_hadamard_blocks(float *v, size_t len, int n);

struct ArConfig {
  int iters = 200;
  float lr = -1.f;  // -1 = 1 / iters
  float minmax_lr = -1.f;  // -1 = lr
  bool minmax = true;  // tune alpha / beta
  bool act_quant = true;  // A8 fake quant of x and h in the loop
  int batch_samples = 8;
  uint64_t seed = 42;
  int log_every = 20;
  bool floor = false;  // also score the exact bf16 experts on the eval samples (ArLayerStats:floor_loss)
};

// One layer's calibration rows (q path), row-major, rows = samples x seqlen.
struct LayerData {
  int samples = 0, seqlen = 0;
  const uint16_t *x = nullptr;  // bf16 [rows][hidden]: the MoE input
  const int32_t *ids = nullptr;  // [rows][topk]
  const float *w = nullptr;  // [rows][topk] routing weights
  const uint16_t *t = nullptr;  // bf16 [rows][hidden]: T, the routed sum's target
  const float *g = nullptr;  // [rows] G (null = 1)
  const float *c = nullptr;  // [rows] the constant part of the block loss (null = 0)
  int64_t rows() const { return static_cast<int64_t>(samples) * seqlen; }
};

// The layer's bf16 expert weights, the checkpoint's fused layout:
struct ExpertSource {
  const uint16_t *gate_up = nullptr;
  const uint16_t *down = nullptr;
};

// One expert's result: codes q in [-8, 7] and fp16 scales, row-major by output.
struct ExpertQuant {
  std::vector<int8_t> gu_q;  // [2 inter][hidden]
  std::vector<uint16_t> gu_s;  // fp16 [2 inter][hidden / wgroup]
  std::vector<int8_t> dn_q;  // [hidden][inter]
  std::vector<uint16_t> dn_s;  // fp16 [hidden][inter / wgroup]
};

// ---- the scalar math, shared by the CPU worker and the tests (the GPU kernels mirror it) ----
inline float ar_bf16(uint16_t b) {
  uint32_t u = static_cast<uint32_t>(b) << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}
inline uint16_t ar_bf16_bits(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  if ((u & 0x7fffffffu) > 0x7f800000u) return static_cast<uint16_t>((u >> 16) | 0x40);
  return static_cast<uint16_t>((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
inline float ar_round_bf16(float f) { return ar_bf16(ar_bf16_bits(f)); }
// float -> fp16 bits, round to nearest even (finite inputs; overflow saturates to inf).
uint16_t ar_half_bits(float f);
float ar_half(uint16_t h);
inline float ar_round_half(float f) { return ar_half(ar_half_bits(f)); }

inline constexpr float kArQMax = 8.f;  // 2^(bits - 1): AutoRound's sym int4 maxq
inline constexpr float kArVOne = 65536.f;  // v is stored as int16 in units of 1/65536 (|v| <= 0.49998)

// AutoRound's sym scale: the group's dominant extreme maps to code -8 exactly, the other side clamps
// at 7. a = -wmin alpha, b = wmax beta; a >= b: s = a / 8 (> 0, wmin -> -8); else s = -b / 8 (< 0,
// wmax -> -8). Stored as fp16 (the kernels multiply signed scales).
inline float ar_scale(float wmin, float wmax, float alpha, float beta) {
  const float a = -wmin * alpha, b = wmax * beta;
  const float s = ar_round_half(a >= b ? a / kArQMax : -b / kArQMax);
  return s != 0.f ? s : 5.9604645e-8f;  // an all-zero group: the smallest fp16, every q = 0
}
inline int ar_code(float w, float s, float v, bool *in_range = nullptr) {
  const float t = std::nearbyint(w / s + v);  // the default rounding mode: nearest even
  if (in_range) *in_range = t >= -8.f && t <= 7.f;
  return static_cast<int>(t < -8.f ? -8.f : t > 7.f ? 7.f : t);
}

// The kernels' activation quantiser, as fake quant: per G-group s = amax / 127,
// v' = rne(v * (127 / amax)) * s (0 for an all-zero group).
void ar_fake_quant_rows(const float *src, int rows, int cols, int group, float *dst);

// ---- workers ----------------------------------------------------------------------------------
// The rows of one step and this worker's (row, k) assignments, grouped by its local experts.
struct ArBatch {
  int rows = 0;
  const uint16_t *x = nullptr;  // bf16 [rows][hidden] (the step's rows)
  std::vector<int32_t> seg_start;  // [local experts + 1] into the assignment arrays
  std::vector<int32_t> asg_row;  // [assignments] row in the step
  std::vector<float> asg_w;  // [assignments] routing weight
  std::vector<int32_t> row_asg;  // [rows][topk] assignment index of (row, k) on this worker, -1 = elsewhere
};

class ArWorker {
 public:
  virtual ~ArWorker() = default;
  virtual std::string name() const = 0;
  // Takes the bf16 weights of `experts` (global ids, ascending) and starts from RTN.
  virtual void setup(const ArShape &s, const std::vector<int> &experts, const ExpertSource &src) = 0;
  virtual void begin(const ArBatch &b) = 0;
  // part = sum over this worker's assignments of w * y, fp32 [rows][hidden] (0 where none).
  virtual void forward(float *part) = 0;
  // delta = dLoss / d(routed sum), fp32 [rows][hidden]; one signSGD step of every parameter.
  virtual void backward(const float *delta, float lr, float minmax_lr, bool minmax) = 0;
  virtual void save_best() = 0;
  virtual void restore_best() = 0;
  virtual void result(int local, ExpertQuant &out) = 0;
  // set the activation fake quant (default on)
  virtual void act_quant(bool on) = 0;
  // exact: the bf16 weights as they are and no activation quant (the BF16 model's routed sum, for
  // the calibration targets); off = the parameterised W4A8 forward (default)
  virtual void exact(bool on) = 0;
  // rough relative throughput (assignments per second) for the expert split; 0 = unknown
  virtual double speed_hint() const = 0;
};

std::unique_ptr<ArWorker> make_cpu_worker(int threads);  // the reference for tests only (not a tuning option)
// SYCL GPU workers, one per device (index into the GPU device list); in autoround_gpu.cpp.
int ar_gpu_count();
std::unique_ptr<ArWorker> make_gpu_worker(int device);

struct ArLayerStats {
  double rtn_loss = 0, tuned_loss = 0;  // on the eval samples
  double floor_loss = 0;  // ArConfig:floor: the exact bf16 experts on the same rows
  double best_step_loss = 0;
  int best_iter = -1;
  double seconds = 0;
  std::vector<double> step_loss;
  std::vector<std::string> workers;
  std::vector<int> experts_per_worker;
  std::vector<int64_t> assignments_per_worker;
};

// Tunes one layer's routed experts on the workers (experts split by routed count / speed);
// eval_samples: samples scored before (RTN) and after (tuned), taken from the data's end.
// out: [experts] results. iters = 0 skips tuning (RTN).
ArLayerStats tune_layer(const ArShape &s, const ArConfig &cfg, const LayerData &d, const ExpertSource &src,
                        std::vector<std::unique_ptr<ArWorker>> &workers, int eval_samples, std::vector<ExpertQuant> &out);

// The loss of the current parameters on samples [s0, s1) (forward only).
double ar_eval(const ArShape &s, const LayerData &d, std::vector<std::unique_ptr<ArWorker>> &workers,
               const std::vector<int> &owner, int s0, int s1, int threads);

// RTN of one expert directly (no worker): the iters = 0 result.
void rtn_expert(const ArShape &s, const ExpertSource &src, int expert, ExpertQuant &out);

// Final codes + fp16 scales of one [n][k] matrix from its parameters (host, precise: every
// worker's result goes through this, so codes never depend on which device tuned them).
// v int16 [n][k] in 1/65536 units; a, b, wmin, wmax [n][k / grp].
void ar_codes(const uint16_t *w, const int16_t *v, const float *a, const float *b, const float *wmin, const float *wmax, int n, int k,
              int grp, std::vector<int8_t> &q, std::vector<uint16_t> &s);
// Per-group min(0, min w) / max(0, max w) of a bf16 [n][k] matrix.
void ar_minmax(const uint16_t *w, int n, int k, int grp, float *wmin, float *wmax);

}  // namespace xenithium:quant
