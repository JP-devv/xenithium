// Gate for the AutoRound GPU worker (src/quant/autoround_gpu.cpp) against the CPU reference worker
// on a small synthetic layer: the same seed and data on both, so the losses of every step must
// agree to fp32 summation-order noise (codes can flip at exact rounding ties, which then moves a
// step's sign update: the bound widens with the steps), the final codes nearly all agree, and a
// zero-step tune on the GPU is RTN bit for bit. With two cards: the expert split over two workers
// against one.
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "quant_synth.hpp"
#include "xenithium/quant/autoround.hpp"

namespace {
namespace q = xenithium::quant;

int g_failures = 0;
void expect(bool ok, const std::string &what) {
  std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
  if (!ok) ++g_failures;
}

struct Run {
  q::ArLayerStats st;
  std::vector<q::ExpertQuant> out;
};

Run run(const quant_test::Synth &S, int iters, bool cpu, int gpus) {
  std::vector<std::unique_ptr<q::ArWorker>> ws;
  if (cpu) ws.push_back(q::make_cpu_worker(16));
  for (int i = 0; i < gpus; ++i) ws.push_back(q::make_gpu_worker(i));
  q::ArConfig cfg;
  cfg.iters = iters;
  cfg.log_every = 0;
  Run r;
  r.st = q::tune_layer(S.s, cfg, S.d, S.src, ws, 8, r.out);
  return r;
}

double code_agreement(const Run &a, const Run &b) {
  int64_t same = 0, tot = 0;
  for (size_t e = 0; e < a.out.size(); ++e) {
    for (size_t i = 0; i < a.out[e].gu_q.size(); ++i) same += a.out[e].gu_q[i] == b.out[e].gu_q[i];
    for (size_t i = 0; i < a.out[e].dn_q.size(); ++i) same += a.out[e].dn_q[i] == b.out[e].dn_q[i];
    tot += static_cast<int64_t>(a.out[e].gu_q.size() + a.out[e].dn_q.size());
  }
  return static_cast<double>(same) / static_cast<double>(tot);
}

}  // namespace

int main() {
  const int gpus = q::ar_gpu_count();
  if (gpus < 1) {
    std::printf("[SKIP] no GPU\nPASS: 0 failure(s)\n");
    return 0;
  }
  const auto S = quant_test::make_synth(40, 64, 3);
  {
    const Run g = run(S, 0, false, 1);
    bool same = true;
    for (int e = 0; e < S.s.experts; ++e) {
      q::ExpertQuant r;
      q::rtn_expert(S.s, S.src, e, r);
      same = same && r.gu_q == g.out[static_cast<size_t>(e)].gu_q && r.dn_q == g.out[static_cast<size_t>(e)].dn_q &&
             r.gu_s == g.out[static_cast<size_t>(e)].gu_s && r.dn_s == g.out[static_cast<size_t>(e)].dn_s;
    }
    expect(same, "gpu: iters 0 == rtn_expert");
  }
  const Run c = run(S, 40, true, 0), g = run(S, 40, false, 1);
  // Exact semantics show in the first steps: step 0 is the RTN forward, step 1 follows one update.
  // From then on signSGD amplifies sign flips of near-zero gradients (fp32 summation order) step
  // after step, so later steps are gated by quality, not by identity.
  std::printf("step loss rel diff gpu/cpu:");
  for (size_t i = 0; i < c.st.step_loss.size(); i += (i < 10 ? 1 : 5)) std::printf(" %zu:%.1e", i, std::fabs(g.st.step_loss[i] / c.st.step_loss[i] - 1.0));
  std::printf("\n");
  double early = 0;
  for (size_t i = 0; i < 2; ++i) early = std::max(early, std::fabs(g.st.step_loss[i] / c.st.step_loss[i] - 1.0) / (i == 0 ? 1e-5 : 1e-3));
  char b[200];
  // Step 0 (the RTN forward) within 1e-5, step 1 (after one update) within 1e-3. With s = absmax / 8
  // the ratio w / s of two bf16 values often lands on an exact.5 tie, so a few codes flip on fp32
  // order already at step 0; a wrong update sign reads 1e-1 at step 1.
  std::snprintf(b, sizeof b, "gpu vs cpu: step 0 / 1 loss rel diff %.1e / %.1e (< 1e-5 / 1e-3: same forward and same update)",
                std::fabs(g.st.step_loss[0] / c.st.step_loss[0] - 1.0), std::fabs(g.st.step_loss[1] / c.st.step_loss[1] - 1.0));
  expect(early < 1.0, b);
  const double q_rel = g.st.tuned_loss / c.st.tuned_loss - 1.0;
  std::snprintf(b, sizeof b, "gpu vs cpu: tuned held-out loss %.4e vs %.4e (%.2f%%, within 3%%); codes agree %.2f%%", g.st.tuned_loss,
                c.st.tuned_loss, 100.0 * q_rel, 100.0 * code_agreement(c, g));
  expect(std::fabs(q_rel) < 0.03, b);
  std::snprintf(b, sizeof b, "gpu: held-out loss tuned %.4e < RTN %.4e (cpu tuned %.4e)", g.st.tuned_loss, g.st.rtn_loss, c.st.tuned_loss);
  expect(g.st.tuned_loss < g.st.rtn_loss, b);
  if (gpus >= 2) {
    const Run g2 = run(S, 40, false, 2);
    double w2 = 0;
    for (size_t i = 0; i < g.st.step_loss.size(); ++i) w2 = std::max(w2, std::fabs(g2.st.step_loss[i] / g.st.step_loss[i] - 1.0));
    std::snprintf(b, sizeof b, "2 gpus vs 1: worst step loss rel diff %.2e (< 1e-6: the expert split changes no sum), codes agree %.3f%%", w2,
                  100.0 * code_agreement(g, g2));
    expect(w2 < 1e-6, b);
  } else {
    std::printf("[SKIP] one GPU: no split check\n");
  }
  std::printf("%s: %d failure(s)\n", g_failures ? "FAIL" : "PASS", g_failures);
  return g_failures ? 1 : 0;
}
