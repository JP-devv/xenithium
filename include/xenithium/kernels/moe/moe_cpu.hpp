// Xenithium · the decode MoE's optional CPU path for cold experts. Host code only (xenithium_pack).
// Why: a cold expert lives in pinned host RAM and the card reads it over PCIe, while the host's own cores can read
// that RAM at the same time. With the path on, the decode route launch hands part of each layer's cold
// experts (MoeCpuOptions:pct) to a per-card host thread pool: it writes the step's activation rows and the experts'
// blobs and row masks into a mailbox in host memory and bumps a request number; the card runs the hot and the other
// cold experts meanwhile; a wait launch before the combine polls the pool's done word with uncached loads (the only GPU
// load that sees a host store inside a recorded graph) and copies the pool's y rows into the step's partials. On a
// timeout the wait launch computes those experts itself (same tiles as the GPU path) and switches the path off.
// Numerics: the pool computes each expert at the GPU path's rounding points (per 128-k group sum, x fp16 scale, the
// gate_up k-split parts summed in order, g / u / h bf16, the down parts summed in order, y fp32) but its in-group
// summation order is not the DPAS order and silu uses libm's expf: not bit-equal to the GPU path. A numerics change,
// off by default; which experts take the CPU path depends on the placement.
// Batch invariance: which unit computes a (row, expert) pair depends on that row's own routing only, never on
// the step's other rows: a fixed set of each layer's cold experts is the pool's (MoeWeights:cold == 2, chosen at load),
// and a row hands its pool experts over when it routes to at least MoeCpuLink:min_cold of them; an expert other rows
// need on the card stays on the card for those rows too (the wait launch writes only the pool's rows). A sequence's
// tokens are then the same alone and in any batch.
// ISAs: AVX-512 BF16 (vdpbf16ps on the blob's (k, k+1) nibble pairs through a 16-entry vpermw table) or AVX2 + FMA
// (fp32 FMAs). The library is built for generic x86-64; only these functions carry the ISA (target attributes), and
// they run only after detect said the CPU has it.
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <string>
#include <thread>
#include <vector>

#include "xenithium/kernels/rows.hpp"  // kStepRows32 (kMoeCpuRows)


namespace xenithium::kernels {

enum class MoeCpuIsa { none = 0, avx2 = 1, avx512bf16 = 2 };
// The best ISA this CPU runs (cpuid); `want`: "auto" (the best), "avx512", "avx2" (that one or none).
MoeCpuIsa moe_cpu_detect(const std::string &want = "auto");
const char *moe_cpu_isa_name(MoeCpuIsa isa);

// CPU experts per request: every expert a 32-row step can route to (32 rows x top-10); the per-row rule cannot
// cap a request without making a row's unit depend on its batch-mates (Model:setup_moe_cpu checks rows x topk).
inline constexpr int kMoeCpuRows = kStepRows32;  // rows of a step the pool serves (the 64-row shape runs without it)
inline constexpr int kMoeCpuMax = 320;  // 32 rows x top-k 10

// The mailbox in host memory (one per card). The card writes x, the expert list and then `req`; the pool writes y and
// then `done = req. Field order is the device's view (moe.cpp writes it as raw words).
struct MoeCpuMail {
  uint32_t req;  // request number (a device counter: unique across graph replays)
  uint32_t pad0[15];
  uint32_t done;  // = req once y holds the request's rows
  uint32_t pad1[15];
  uint32_t timeouts;  // set by the card when a wait timed out (it computed those experts itself; the path is off)
  uint32_t pad3[15];
  int32_t n;  // CPU experts in this request
  int32_t rows;  // rows of the step (x rows [0, rows))
  int32_t pad2[14];
  uint32_t mask[kMoeCpuMax];  // per expert: bit r = the pool computes row r of it
  uint64_t blob[kMoeCpuMax];  // per expert: its blob (pinned host)
};
static_assert(sizeof(MoeCpuMail) % 64 == 0, "MoeCpuMail: cache-line multiple");

// One card's blob shape (MoeShape:tp, per-card half blobs) as the pool sees it.
struct MoeCpuShape {
  int hidden = 0, inter_c = 0;  // inter_c = inter / cards (this card's h columns)
  int card = 0, cards = 1;
  int gu_ks = 8, dn_ks = 2;  // the GPU path's k splits (the parts are summed in the same order)
  size_t gu_codes = 0, dn_codes = 0, gu_scales = 0, dn_scales = 0;  // MoeBlobLayout's offsets (moe_tp_layout)
};

// The expert math on one blob, for the rows set in `mask` (bit r -> x row r, y row r):
// gate_up tiles [t0, t1) of inter_c / 16: h[r][16 t + c] = bf16(silu(bf16 g) bf16 u), h row stride inter_c
// down tiles [t0, t1) of hidden / 16: y[r][16 t + c] (fp32), y row stride ldy
// x, h: bf16 bits. isa none = the scalar code (tests; same summation order as avx2).
void moe_cpu_gate_up(MoeCpuIsa isa, const MoeCpuShape &s, const uint8_t *blob, const uint16_t *x, int ldx, uint32_t mask, int t0, int t1, uint16_t *h);
void moe_cpu_down(MoeCpuIsa isa, const MoeCpuShape &s, const uint8_t *blob, const uint16_t *h, uint32_t mask, int t0, int t1, float *y, int ldy);

struct MoeCpuOptions {
  MoeCpuIsa isa = MoeCpuIsa::none;
  int threads = 8;  // per card, the poller included
  int first_core = -1;  // pin threads to [first_core, first_core + threads); -1 = unpinned
  int pct = 70;  // percent of a layer's cold experts in the pool's set (informational: the set is MoeWeights:cold)
  int min_rows = 1;  // steps with fewer rows keep every cold expert on the card
  int timeout_ms = 500;  // the card's wait bound per layer before it computes the experts itself
  int spin_us = 20000;  // the pool spins this long after a request before it naps (latency vs idle CPU)
  int stall_us = 0;  // tests: the pool sleeps this long per request (exercises the timeout path)
};

// A card's pool: `threads` threads; thread 0 polls the mailbox, all of them compute (gate_up jobs, a barrier, down
// jobs; jobs = (expert, 16-column tile) pairs taken from a shared counter). Host buffers (x, y, the mailbox) are given
// by the caller: pinned host USM the card reads and writes.
class MoeCpuPool {
 public:
  MoeCpuPool(const MoeCpuShape &s, const MoeCpuOptions &o, MoeCpuMail *mail, const uint16_t *x, float *y);
  ~MoeCpuPool();
  MoeCpuPool(const MoeCpuPool &) = delete;
  MoeCpuPool &operator=(const MoeCpuPool &) = delete;

  uint64_t served() const { return served_.load(std::memory_order_relaxed); }
  uint64_t experts() const { return experts_.load(std::memory_order_relaxed); }
  double busy_us() const { return static_cast<double>(busy_ns_.load(std::memory_order_relaxed)) * 1e-3; }
  // Runs one request now on the calling thread plus the pool (tests and the micro: no card involved).
  void run_once(int n, int rows, const uint32_t *mask, const uint8_t *const *blob);

 private:
  // One request; two of them (by the generation's parity) so a worker that is still leaving request G - 1 never reads
  // a slot thread 0 is filling for G + 1 (`users` counts the workers inside a slot).
  struct Req {
    int n = 0;
    uint32_t mask[kMoeCpuMax] = {};
    const uint8_t *blob[kMoeCpuMax] = {};
    alignas(64) std::atomic<int> next_gu{0};
    alignas(64) std::atomic<int> next_dn{0};
    alignas(64) std::atomic<int> left_gu{0};
    alignas(64) std::atomic<int> left_dn{0};
    alignas(64) std::atomic<int> users{0};
  };
  void worker(int t);
  void compute(Req &q);
  void serve(uint32_t r);  // thread 0: one request of the mailbox
  MoeCpuShape s_;
  MoeCpuOptions o_;
  MoeCpuMail *mail_;
  const uint16_t *x_;
  float *y_;
  std::vector<std::thread> th_;
  std::vector<uint16_t> h_;  // [kMoeCpuMax][kStepRows32][inter_c]
  Req req_[2];
  uint32_t seen0_ = 0;  // the mailbox's request number when the pool started
  alignas(64) std::atomic<uint32_t> gen_{0};
  alignas(64) std::atomic<bool> stop_{false};
  std::atomic<uint64_t> served_{0}, experts_{0}, busy_ns_{0};
};

}  // namespace xenithium:kernels
