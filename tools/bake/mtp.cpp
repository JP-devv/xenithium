// xenithium-bake mtp: the MTP drafter of a BF16 checkpoint -> a drafter directory (ModelOptions:mtp_dir).
// xenithium-bake mtp --src BF16_DIR --out DIR [--rotate 1] [--draft-head 1] [--experts-only 0] [--threads N]
// The source's experts may be fused
// (mtp.layers.0.mlp.experts.{gate_up_proj,down_proj}) or one tensor per expert and projection
// (mtp.layers.0.mlp.experts.E.{gate,up,down}_proj.weight, as in Intel's W4A16 AutoRound export's
// model_extra_tensors.safetensors). --experts-only 1 writes mtp-experts.safetensors only (the dense trunk and the
// rest come from another drafter directory, hard-linked by the caller).
// Writes:
// mtp-dense.safetensors every mtp.* tensor but the routed experts, BF16 as in the source (the
// engine converts the drafter trunk to int8 per row at load: W8)
// mtp-experts.safetensors the routed experts, int4 g128 symmetric in the target's GPTQ layout
// (mtp.layers.0.mlp.experts.E.{gate,up,down}_proj.{qweight,qzeros,scales}),
// RTN (AutoRound's signed /8 rule), in the block-Hadamard basis with --rotate
// (the target's: W_gu H_128, W_dn H_64)
// draft-head.safetensors (--draft-head, default 1) the int2 draft head from the source's lm_head:
// mtp.draft_head.codes U8 [vocab][hidden / 4] (element c at bits 2 (c % 4)
// of byte c / 4, code q = level q - 1.5), mtp.draft_head.scales F32
// [hidden / 128][vocab] (per (row, 128-group), kernels:quantize_i2_groups)
// config.json the source's + quantization_config (xenithium_rotation with --rotate)
// The drafter's AutoRound (calibrated on the quantised target's hiddens) replaces the RTN experts later.
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <functional>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <thread>

#include "bake.hpp"
#include "xenithium/kernels/pack.hpp"
#include "xenithium/quant/autoround.hpp"
#include "xenithium/quant/export.hpp"

namespace xenithium::bake {

namespace q = xenithium::quant;
namespace m = xenithium::model;
namespace k = xenithium::kernels;
namespace fs = std::filesystem;

namespace {
void pfor(int n, int threads, const std::function<void(int)> &fn) {
  std::atomic<int> next{0};
  std::vector<std::thread> pool;
  for (int t = 0; t < std::max(1, std::min(threads, n)); ++t)
    pool.emplace_back([&] {
      for (int i; (i = next.fetch_add(1)) < n;) fn(i);
    });
  for (auto &t : pool) t.join();
}
}  // namespace

int cmd_mtp(const Args &a) {
  const std::string src = a.str("src", home() + "/models/Qwen3.8-Flash-Next-Uncensored"), out = a.str("out");
  if (out.empty()) throw std::runtime_error("--out is required");
  const bool experts_only = a.num("experts-only", 0) != 0;
  const bool rotate = a.num("rotate", 1) != 0, head = !experts_only && a.num("draft-head", 1) != 0;
  const int threads = static_cast<int>(a.num("threads", std::thread::hardware_concurrency()));
  fs::create_directories(out);
  const auto ck = m::Checkpoint::open(src);
  const auto t0 = std::chrono::steady_clock::now();
  auto secs = [&] { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); };

  // ---- dense ----
  if (!experts_only) {
    std::vector<std::vector<uint8_t>> store;
    std::vector<q::StTensor> ts;
    uint64_t bytes = 0;
    for (const auto &t : ck.tensors()) {
      if (t.name.rfind("mtp.", 0) != 0 || t.name.find(".mlp.experts.") != std::string::npos) continue;
      store.emplace_back(t.bytes);
      read_tensor_range(ck, t, 0, t.bytes, store.back().data());
      ts.push_back({t.name, t.dtype_str, t.shape, store.back().data(), t.bytes});
      bytes += t.bytes;
    }
    if (ts.empty()) throw std::runtime_error("mtp: no mtp.* tensors in " + src);
    q::write_safetensors(out + "/mtp-dense.safetensors", ts, {{"format", "pt"}, {"xenithium_scheme", "mtp-dense-bf16"}});
    std::printf("mtp-dense: %zu tensors, %.2f GiB (%.0f s)\n", ts.size(), static_cast<double>(bytes) / (1ull << 30), secs());
  }

  // ---- routed experts: RTN int4 g128 (rotated) ----
  {
    const std::string EX = "mtp.layers.0.mlp.experts.";
    const bool fused = ck.find(EX + "gate_up_proj") != nullptr;
    q::ArShape s;
    if (fused) {
      const auto &tgu = ck.at(EX + "gate_up_proj"), &tdn = ck.at(EX + "down_proj");
      if (tgu.dtype != m::DType::bf16 || tgu.shape.size() != 3 || tdn.shape.size() != 3) throw std::runtime_error("mtp: want BF16 experts");
      s.experts = static_cast<int>(tgu.dim(0));
      s.inter = static_cast<int>(tgu.dim(1) / 2);
      s.hidden = static_cast<int>(tgu.dim(2));
    } else {
      const auto &tg = ck.at(EX + "0.gate_proj.weight");
      if (tg.dtype != m::DType::bf16 || tg.shape.size() != 2) throw std::runtime_error("mtp: want BF16 experts (fused or per expert)");
      s.inter = static_cast<int>(tg.dim(0));
      s.hidden = static_cast<int>(tg.dim(1));
      while (ck.find(EX + std::to_string(s.experts) + ".gate_proj.weight")) ++s.experts;
    }
    // one expert at a time (RAM: the bake may be running beside this)
    const size_t gu_n = static_cast<size_t>(2) * s.inter * s.hidden, dn_n = static_cast<size_t>(s.hidden) * s.inter;
    std::vector<q::ExpertQuant> res(static_cast<size_t>(s.experts));
    q::ArShape s1 = s;
    s1.experts = 1;
    pfor(s.experts, std::min(threads, 16), [&](int e) {
      std::vector<uint16_t> gu(gu_n), dn(dn_n);
      if (fused) {
        read_tensor_range(ck, ck.at(EX + "gate_up_proj"), static_cast<uint64_t>(e) * gu_n * 2, gu_n * 2, gu.data());
        read_tensor_range(ck, ck.at(EX + "down_proj"), static_cast<uint64_t>(e) * dn_n * 2, dn_n * 2, dn.data());
      } else {  // gate rows then up rows, as the fused tensor
        const std::string b = EX + std::to_string(e) + ".";
        const auto &tg = ck.at(b + "gate_proj.weight"), &tu = ck.at(b + "up_proj.weight"), &td = ck.at(b + "down_proj.weight");
        if (tg.bytes != gu_n || tu.bytes != gu_n || td.bytes != dn_n * 2) throw std::runtime_error("mtp: unexpected expert shapes for expert " + std::to_string(e));
        read_tensor_range(ck, tg, 0, gu_n, gu.data());
        read_tensor_range(ck, tu, 0, gu_n, gu.data() + gu_n / 2);
        read_tensor_range(ck, td, 0, dn_n * 2, dn.data());
      }
      if (rotate) {
        auto rot = [](uint16_t *v, size_t rows, int cols, int blk) {
          std::vector<float> f(static_cast<size_t>(cols));
          for (size_t r = 0; r < rows; ++r) {
            uint16_t *row = v + r * cols;
            for (int i = 0; i < cols; ++i) f[static_cast<size_t>(i)] = q::ar_bf16(row[i]);
            q::ar_hadamard_blocks(f.data(), f.size(), blk);
            for (int i = 0; i < cols; ++i) row[i] = q::ar_bf16_bits(f[static_cast<size_t>(i)]);
          }
        };
        rot(gu.data(), static_cast<size_t>(2) * s.inter, s.hidden, 128);
        rot(dn.data(), static_cast<size_t>(s.hidden), s.inter, 64);
      }
      q::rtn_expert(s1, q::ExpertSource{gu.data(), dn.data()}, 0, res[static_cast<size_t>(e)]);
    });
    std::vector<std::vector<uint8_t>> store;
    const auto ts = q::expert_export_tensors(s, "mtp.", 0, res, store);
    q::write_safetensors(out + "/mtp-experts.safetensors", ts,
                         {{"format", "pt"}, {"xenithium_scheme", std::string("w4g128-sym-rtn") + (rotate ? "-hadamard-x128-h64" : "")}});
    std::printf("mtp-experts: %d experts RTN int4 g128%s (%.0f s)\n", s.experts, rotate ? ", rotated" : "", secs());
  }

  // ---- int2 draft head ----
  if (head) {
    const auto &tl = ck.at("lm_head.weight");
    if (tl.dtype != m::DType::bf16 || tl.shape.size() != 2) throw std::runtime_error("mtp: lm_head.weight is not BF16 [vocab, hidden]");
    const int V = static_cast<int>(tl.dim(0)), H = static_cast<int>(tl.dim(1)), G = H / 128;
    std::vector<uint16_t> w(static_cast<size_t>(tl.bytes / 2));
    read_tensor_range(ck, tl, 0, tl.bytes, w.data());
    std::vector<uint8_t> codes(static_cast<size_t>(V) * H), packed(static_cast<size_t>(V) * H / 4, 0);
    std::vector<float> scale(static_cast<size_t>(G) * V);
    const int blk = 1024;
    pfor((V + blk - 1) / blk, threads, [&](int b) { k::quantize_i2_groups(w.data(), V, H, codes.data(), scale.data(), b * blk, std::min(V, (b + 1) * blk)); });
    double e2 = 0, r2 = 0;
    for (int r = 0; r < V; ++r)
      for (int c = 0; c < H; ++c) {
        const size_t i = static_cast<size_t>(r) * H + c;
        packed[i / 4] |= static_cast<uint8_t>(codes[i] << (2 * (c % 4)));
        const double x = k::bf16_value(w[i]), y = (codes[i] - 1.5) * scale[static_cast<size_t>(c / 128) * V + r];
        e2 += (x - y) * (x - y);
        r2 += x * x;
      }
    q::write_safetensors(out + "/draft-head.safetensors",
                         {{"mtp.draft_head.codes", "U8", {V, H / 4}, packed.data(), packed.size()},
                          {"mtp.draft_head.scales", "F32", {G, V}, scale.data(), scale.size() * 4}},
                         {{"format", "pt"}, {"xenithium_scheme", "draft-head-int2-g128-f32scale"}});
    std::printf("draft head: int2 g128, %d x %d, %.1f MiB, weight relative error %.3f (%.0f s)\n", V, H,
                static_cast<double>(packed.size() + scale.size() * 4) / (1 << 20), std::sqrt(e2 / r2), secs());
  }

  // ---- config ----
  if (!experts_only) {
    std::ifstream cf(src + "/config.json");
    std::stringstream ss;
    ss << cf.rdbuf();
    std::string cfg = ss.str();
    if (cfg.find("\"quantization_config\"") != std::string::npos) throw std::runtime_error("mtp: the source config already has a quantization_config");
    size_t end = cfg.rfind('}');
    while (end > 0 && (cfg[end - 1] == '\n' || cfg[end - 1] == ' ' || cfg[end - 1] == '\r')) --end;
    cfg = cfg.substr(0, end) +
          ",\n  \"quantization_config\": {\"quant_method\": \"xenithium-mtp\", \"bits\": 4, \"group_size\": 128, \"sym\": true, " +
          std::string(rotate ? "\"xenithium_rotation\": \"hadamard-x128-h64\", " : "") +
          std::string(head ? "\"draft_head\": \"int2-g128\", " : "") + "\"xenithium_bake\": \"round-64\"}\n}\n";
    std::ofstream(out + "/config.json") << cfg;
  }
  std::printf("drafter -> %s (%.0f s)\n", out.c_str(), secs());
  return 0;
}

}  // namespace xenithium:bake
