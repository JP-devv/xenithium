// xenithium-bake rtn / check-layout: the experts' RTN bake (the baseline) and a layout check.
// xenithium-bake rtn --src BF16_DIR --out DIR [--layers A-B] [--threads 32]
// Routed experts of every layer -> int4 g128 RTN (AutoRound's format at iters 0), written as
// DIR/experts-LNN.safetensors in the qwen38-w4a16 names and layout (the export).
// xenithium-bake check-layout --src BF16_DIR --ref W4A16_DIR [--layer 0] [--experts 8]
// Dequantised weights vs BF16 per expert: ours (RTN from BF16, through the export) and the
// reference checkpoint's. Both must be a few % off BF16; a layout error reads ~100 %.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <functional>
#include <stdexcept>
#include <thread>

#include "bake.hpp"
#include "xenithium/quant/export.hpp"

namespace xenithium::bake {

namespace q = xenithium::quant;
namespace m = xenithium::model;

namespace {

void pfor(int n, int threads, const std::function<void(int)> &fn) {
  std::atomic<int> next{0};
  std::vector<std::thread> pool;
  for (int t = 0; t < std::max(1, threads); ++t)
    pool.emplace_back([&] {
      for (int i; (i = next.fetch_add(1)) < n;) fn(i);
    });
  for (auto &t : pool) t.join();
}

std::pair<int, int> layer_range(const Args &a, int layers) {
  const std::string r = a.str("layers", "");
  if (r.empty()) return {0, layers};
  const size_t d = r.find('-');
  const int lo = std::stoi(r.substr(0, d)), hi = d == std::string::npos ? lo : std::stoi(r.substr(d + 1));
  if (lo < 0 || hi >= layers || lo > hi) throw std::runtime_error("--layers out of range");
  return {lo, hi + 1};
}

const std::string kPrefix = "model.language_model.";

}  // namespace

// The layer's fused BF16 experts: gate_up [E][2I][H], down [E][H][I].
void read_layer_experts(const m::Checkpoint &ck, int layer, const q::ArShape &s, std::vector<uint16_t> &gu, std::vector<uint16_t> &dn) {
  read_fused_experts(ck, kPrefix + "layers." + std::to_string(layer) + ".mlp.experts.", s, gu, dn);
}

void read_fused_experts(const m::Checkpoint &ck, const std::string &P, const q::ArShape &s, std::vector<uint16_t> &gu, std::vector<uint16_t> &dn) {
  const auto &tgu = ck.at(P + "gate_up_proj"), &tdn = ck.at(P + "down_proj");
  const std::vector<int64_t> wgu{s.experts, 2 * s.inter, s.hidden}, wdn{s.experts, s.hidden, s.inter};
  if (tgu.shape != wgu || tdn.shape != wdn || tgu.dtype != m::DType::bf16 || tdn.dtype != m::DType::bf16)
    throw std::runtime_error(P + ": unexpected fused expert tensors");
  gu.resize(tgu.bytes / 2);
  dn.resize(tdn.bytes / 2);
  // 4 readers per tensor: one stream does not reach the drive's rate
  const int parts = 8;
  pfor(parts, parts, [&](int i) {
    const m::TensorInfo &t = i < parts / 2 ? tgu : tdn;
    uint16_t *dst = i < parts / 2 ? gu.data() : dn.data();
    const int j = i % (parts / 2);
    const uint64_t chunk = (t.bytes / (parts / 2) + 4095) / 4096 * 4096;
    const uint64_t a0 = std::min<uint64_t>(t.bytes, chunk * j), a1 = std::min<uint64_t>(t.bytes, a0 + chunk);
    if (a1 > a0) read_tensor_range(ck, t, a0, a1 - a0, reinterpret_cast<uint8_t *>(dst) + a0);
  });
}

// The drafter's BF16 experts fused ([experts][2 x inter][hidden] gate rows then up rows, [experts][hidden][inter])
// from a checkpoint holding them fused (P + gate_up_proj / down_proj) or per expert (P + E.{gate,up,down}_proj.weight,
// Intel's W4A16 export's model_extra_tensors).
void read_mtp_experts_any(const m::Checkpoint &ck, const std::string &P, const q::ArShape &s, std::vector<uint16_t> &gu, std::vector<uint16_t> &dn) {
  if (ck.find(P + "gate_up_proj")) {
    read_fused_experts(ck, P, s, gu, dn);
    return;
  }
  const size_t gu_n = static_cast<size_t>(2) * s.inter * s.hidden, dn_n = static_cast<size_t>(s.hidden) * s.inter;
  gu.resize(gu_n * s.experts);
  dn.resize(dn_n * s.experts);
  pfor(s.experts, 16, [&](int e) {
    const std::string b = P + std::to_string(e) + ".";
    const auto &tg = ck.at(b + "gate_proj.weight"), &tu = ck.at(b + "up_proj.weight"), &td = ck.at(b + "down_proj.weight");
    if (tg.dtype != m::DType::bf16 || tg.bytes != gu_n || tu.bytes != gu_n || td.bytes != dn_n * 2) throw std::runtime_error(b + ": unexpected expert tensors");
    uint16_t *g = gu.data() + static_cast<size_t>(e) * gu_n;
    read_tensor_range(ck, tg, 0, gu_n, reinterpret_cast<uint8_t *>(g));
    read_tensor_range(ck, tu, 0, gu_n, reinterpret_cast<uint8_t *>(g + gu_n / 2));
    read_tensor_range(ck, td, 0, dn_n * 2, reinterpret_cast<uint8_t *>(dn.data() + static_cast<size_t>(e) * dn_n));
  });
}

int cmd_rtn(const Args &a) {
  const auto ck = m::Checkpoint::open(a.str("src", home() + "/models/Qwen3.8-Flash-Next-Uncensored"));
  const std::string out = a.str("out");
  if (out.empty()) throw std::runtime_error("--out DIR is required");
  std::filesystem::create_directories(out);
  const int threads = static_cast<int>(a.num("threads", std::thread::hardware_concurrency()));
  q::ArShape s;
  const auto [l0, l1] = layer_range(a, 48);
  std::vector<uint16_t> gu, dn;
  for (int l = l0; l < l1; ++l) {
    const auto t0 = std::chrono::steady_clock::now();
    read_layer_experts(ck, l, s, gu, dn);
    const double tr = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::vector<q::ExpertQuant> res(static_cast<size_t>(s.experts));
    const q::ExpertSource src{gu.data(), dn.data()};
    pfor(s.experts, threads, [&](int e) { q::rtn_expert(s, src, e, res[static_cast<size_t>(e)]); });
    std::vector<std::vector<uint8_t>> store;
    const auto ts = q::expert_export_tensors(s, kPrefix, l, res, store);
    char name[64];
    std::snprintf(name, sizeof name, "/experts-L%02d.safetensors", l);
    q::write_safetensors(out + name, ts, {{"format", "pt"}, {"xenithium_scheme", "w4g128-sym-rtn"}});
    const double tt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("layer %2d: read %.1f s, total %.1f s\n", l, tr, tt);
    std::fflush(stdout);
  }
  return 0;
}

int cmd_check_layout(const Args &a) {
  const auto src = m::Checkpoint::open(a.str("src", home() + "/models/Qwen3.8-Flash-Next"));
  const auto ref = m::Checkpoint::open(a.str("ref", home() + "/models/qwen38-w4a16"));
  const int layer = static_cast<int>(a.num("layer", 0)), n_ex = static_cast<int>(a.num("experts", 8));
  q::ArShape s;
  const int H = s.hidden, I = s.inter;
  std::vector<uint16_t> gu, dn;
  read_layer_experts(src, layer, s, gu, dn);
  const q::ExpertSource es{gu.data(), dn.data()};
  std::printf("%-6s %-5s %12s %12s %10s\n", "expert", "proj", "ours/bf16", "ref/bf16", "same code");
  double worst = 0;
  for (int e = 0; e < n_ex; ++e) {
    q::ExpertQuant mine;
    q::rtn_expert(s, es, e, mine);
    std::vector<std::vector<uint8_t>> store;
    const auto ts = q::expert_export_tensors(s, kPrefix, layer, std::vector<q::ExpertQuant>(static_cast<size_t>(e) + 1, mine), store);
    for (int p = 0; p < 3; ++p) {
      const char *proj = p == 0 ? "gate_proj" : p == 1 ? "up_proj" : "down_proj";
      const int n = p < 2 ? I : H, k = p < 2 ? H : I;
      const std::string b = kPrefix + "layers." + std::to_string(layer) + ".mlp.experts." + std::to_string(e) + "." + proj + ".";
      // ours through the export tensors (the last expert's 9 tensors are this one's)
      const size_t base = ts.size() - 9 + static_cast<size_t>(p) * 3;
      std::vector<int8_t> c1(static_cast<size_t>(n) * k), c2(c1.size());
      std::vector<uint16_t> s1(static_cast<size_t>(n) * (k / 128)), s2(s1.size());
      q::gptq_unpack(static_cast<const uint32_t *>(ts[base].data), static_cast<const uint16_t *>(ts[base + 2].data), n, k, c1.data(), s1.data());
      std::vector<uint32_t> rq(ref.at(b + "qweight").bytes / 4);
      std::vector<uint16_t> rs(ref.at(b + "scales").bytes / 2);
      read_tensor_range(ref, ref.at(b + "qweight"), 0, rq.size() * 4, rq.data());
      read_tensor_range(ref, ref.at(b + "scales"), 0, rs.size() * 2, rs.data());
      q::gptq_unpack(rq.data(), rs.data(), n, k, c2.data(), s2.data());
      const uint16_t *w = p == 0 ? gu.data() + static_cast<size_t>(e) * 2 * I * H
                          : p == 1 ? gu.data() + (static_cast<size_t>(e) * 2 * I + I) * H
                                   : dn.data() + static_cast<size_t>(e) * H * I;
      double e1 = 0, e2 = 0, r2 = 0;
      int64_t same = 0;
      for (int r = 0; r < n; ++r)
        for (int c = 0; c < k; ++c) {
          const size_t at = static_cast<size_t>(r) * k + c, g = static_cast<size_t>(r) * (k / 128) + c / 128;
          const double x = q::ar_bf16(w[at]);
          const double d1 = c1[at] * static_cast<double>(q::ar_half(s1[g])) - x, d2 = c2[at] * static_cast<double>(q::ar_half(s2[g])) - x;
          e1 += d1 * d1, e2 += d2 * d2, r2 += x * x;
          same += c1[at] == c2[at];
        }
      const double r1 = std::sqrt(e1 / r2), rr = std::sqrt(e2 / r2);
      worst = std::max(worst, std::max(r1, rr));
      std::printf("%-6d %-5.5s %12.4f %12.4f %9.2f%%\n", e, proj, r1, rr, 100.0 * static_cast<double>(same) / (static_cast<double>(n) * k));
    }
  }
  std::printf("worst relative error vs BF16: %.4f (%s)\n", worst, worst < 0.2 ? "layout consistent" : "LAYOUT MISMATCH");
  return worst < 0.2 ? 0 : 1;
}

}  // namespace xenithium:bake
