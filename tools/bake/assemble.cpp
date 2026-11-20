// xenithium-bake assemble: the bake's pieces -> one model directory the engine loads as it is.
// xenithium-bake assemble --src BF16_DIR --out DIR --ple PLE_DIR [--trunk-per-file-gib 4]
// DIR already holds the tuned experts (experts-LNN.safetensors, native/experts-LNN.tp2; `tune`).
// Adds: trunk-NN.safetensors (every source tensor but the routed experts and the PLE table, BF16 as
// it is; the engine converts its plan matrices to int8 at load), the int8 PLE rows (hard links to
// PLE_DIR/ple-int8-*.safetensors, a copy across file systems), config.json with a quantization_config
// (int4 g128 sym, auto_gptq packing, act 8-bit), and the source's other files (tokenizer, template).
// Checks first that every layer's expert export and native blobs are there.
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include <atomic>
#include <chrono>
#include <thread>

#include "bake.hpp"
#include "xenithium/arch/qwen38/arch.hpp"
#include "xenithium/arch/qwen38/experts.hpp"
#include "xenithium/arch/qwen38/moe.hpp"
#include "xenithium/kernels/moe/moe.hpp"
#include "xenithium/model/bulk_read.hpp"
#include "xenithium/model/config.hpp"
#include "xenithium/quant/export.hpp"

namespace xenithium::bake {

namespace q = xenithium::quant;
namespace m = xenithium::model;
namespace fs = std::filesystem;

int cmd_assemble(const Args &a) {
  const std::string src = a.str("src", home() + "/models/Qwen3.8-Flash-Next-Uncensored"), out = a.str("out"), ple = a.str("ple");
  if (out.empty() || ple.empty()) throw std::runtime_error("--out and --ple are required");
  const auto ck = m::Checkpoint::open(src);
  int layers = 0;
  while (ck.find("model.language_model.layers." + std::to_string(layers) + ".mlp.experts.gate_up_proj")) ++layers;
  for (int l = 0; l < layers; ++l) {
    char e[64], n[64];
    std::snprintf(e, sizeof e, "/experts-L%02d.safetensors", l);
    std::snprintf(n, sizeof n, "/native/experts-L%02d.tp2", l);
    if (!fs::exists(out + e) || !fs::exists(out + n)) throw std::runtime_error("assemble: layer " + std::to_string(l) + " is not baked (" + out + e + ")");
  }
  // the PLE files
  std::vector<std::string> ples;
  for (const auto &f : fs::directory_iterator(ple))
    if (f.path().filename().string().rfind("ple-int8-", 0) == 0 && f.path().extension() == ".safetensors") ples.push_back(f.path().string());
  if (ples.empty()) throw std::runtime_error("assemble: no ple-int8-*.safetensors in " + ple);
  for (const auto &p : ples) {
    const fs::path dst = fs::path(out) / fs::path(p).filename();
    if (fs::exists(dst)) fs::remove(dst);
    std::error_code ec;
    fs::create_hard_link(p, dst, ec);
    if (ec) fs::copy_file(p, dst);
  }
  // the trunk
  const uint64_t cap = static_cast<uint64_t>(a.num("trunk-per-file-gib", 4)) << 30;
  auto is_ple_tab = [](const std::string &n) {
    return n.find("ple.ple_embedding.ngram_embedding.") != std::string::npos || n.find("ple.ple_embedding.layer_multipliers") != std::string::npos ||
           n.find("ple.ple_embedding.ngram_heads_") != std::string::npos;
  };
  std::vector<const m::TensorInfo *> trunk;
  uint64_t trunk_bytes = 0;
  for (const auto &t : ck.tensors())
    if (t.name.find(".mlp.experts.") == std::string::npos && !is_ple_tab(t.name)) trunk.push_back(&t), trunk_bytes += t.bytes;
  int file = 0;
  for (size_t i = 0; i < trunk.size();) {
    std::vector<std::vector<uint8_t>> store;
    std::vector<q::StTensor> ts;
    uint64_t sz = 0;
    for (; i < trunk.size() && (ts.empty() || sz + trunk[i]->bytes <= cap); ++i) {
      const auto &t = *trunk[i];
      store.emplace_back(t.bytes);
      read_tensor_range(ck, t, 0, t.bytes, store.back().data());
      ts.push_back({t.name, t.dtype_str, t.shape, store.back().data(), t.bytes});
      sz += t.bytes;
    }
    char n[48];
    std::snprintf(n, sizeof n, "/trunk-%02d.safetensors", file++);
    q::write_safetensors(out + n, ts, {{"format", "pt"}, {"xenithium_scheme", "trunk-bf16"}});
  }
  // config.json + quantization_config
  std::ifstream cf(src + "/config.json");
  std::stringstream ss;
  ss << cf.rdbuf();
  std::string cfg = ss.str();
  if (cfg.find("\"quantization_config\"") != std::string::npos) throw std::runtime_error("assemble: the source config already has a quantization_config");
  const size_t close = cfg.rfind('}');
  if (close == std::string::npos) throw std::runtime_error("assemble: config.json is not an object");
  const std::string qc = std::string() +
      ",\n  \"quantization_config\": {\"quant_method\": \"auto-round\", \"bits\": 4, \"group_size\": 128, \"sym\": true, \"data_type\": \"int\", "
      "\"packing_format\": \"auto_round:auto_gptq\", \"act_bits\": 8, \"act_group_size\": 128, \"act_dynamic\": true, "
      "\"block_name_to_quantize\": \"model.language_model.layers\", \"xenithium_bake\": \"round-64\", " +
      std::string(a.num("rotate", 0) != 0 ? "\"xenithium_rotation\": \"hadamard-x128-h64\", " : "") +
      "\"xenithium_note\": \"routed experts only (W4 g128 sym, AutoRound tuned for A8 activations); trunk BF16 (int8 at load); PLE int8 rows\"}\n";
  // the PLE rows are int8 now: text_config.ple_embedding_dtype (absent in a BF16 source)
  {
    const size_t tc = cfg.find("\"text_config\"");
    if (tc == std::string::npos) throw std::runtime_error("assemble: config.json has no text_config");
    const size_t pd = cfg.find("\"ple_embedding_dtype\"");
    if (pd != std::string::npos) {
      const size_t v0 = cfg.find('"', cfg.find(':', pd) + 1), v1 = cfg.find('"', v0 + 1);
      cfg.replace(v0, v1 - v0 + 1, "\"int8\"");
    } else {
      const size_t brace = cfg.find('{', tc);
      cfg.insert(brace + 1, "\n    \"ple_embedding_dtype\": \"int8\",");
    }
  }
  const size_t close2 = cfg.rfind('}');
  size_t body_end = close2;
  while (body_end > 0 && (cfg[body_end - 1] == '\n' || cfg[body_end - 1] == ' ' || cfg[body_end - 1] == '\r')) --body_end;
  cfg = cfg.substr(0, body_end) + qc + "}\n";
  std::ofstream(out + "/config.json") << cfg;
  // the other files
  int copied = 0;
  for (const auto &f : fs::directory_iterator(src)) {
    const std::string nm = f.path().filename().string();
    if (!f.is_regular_file() || f.path().extension() == ".safetensors" || nm == "model.safetensors.index.json" || nm == "config.json") continue;
    fs::copy_file(f.path(), fs::path(out) / nm, fs::copy_options::overwrite_existing);
    ++copied;
  }
  std::printf("assembled %s: %d layers of experts, %zu PLE files, %zu trunk tensors (%.2f GiB) in %d files, config + %d other files\n", out.c_str(),
              layers, ples.size(), trunk.size(), static_cast<double>(trunk_bytes) / (1ull << 30), file, copied);
  return 0;
}

// xenithium-bake pack-native --model GPTQ_DIR --out DIR [--threads N]
// An int4 GPTQ checkpoint's experts (qwen38-w4a16 layout) -> DIR/experts-LNN.tp2, the engine's
// per-card TP blobs, packed as the expert store packs them at load (bit for bit: the store's
// verify hashes). Put DIR at <model>/native and the load reads them as they are.
int cmd_pack_native(const Args &a) {
  namespace q38 = xenithium::arch::qwen38;
  namespace k = xenithium::kernels;
  const std::string dir = a.str("model", home() + "/models/qwen38-w4a16"), out = a.str("out");
  if (out.empty()) throw std::runtime_error("--out is required");
  fs::create_directories(out);
  const int threads = static_cast<int>(a.num("threads", std::thread::hardware_concurrency()));
  const auto ck = m::Checkpoint::open(dir);
  const q38::Config cfg = q38::Config::parse(m::ModelConfig::load(dir));
  const q38::ExpertLayout lay = q38::ExpertLayout::of(cfg);
  const k::MoeBlobLayout fl = q38::moe_blob_layout(lay);
  size_t st = lay.stride;
  const k::MoeBlobLayout tl = k::moe_tp_layout(cfg.hidden, cfg.moe_inter, 2, &st);
  const int E = cfg.experts;
  const auto t0 = std::chrono::steady_clock::now();
  std::vector<uint8_t> regions(st * static_cast<size_t>(E) * 2);
  for (int l = 0; l < cfg.layers; ++l) {
    std::vector<const m::TensorInfo *> ts;
    for (int e = 0; e < E; ++e)
      for (const auto *t : q38::expert_tensors(ck, l, e)) ts.push_back(t);
    const auto buf = m::bulk_read(ck, ts);
    std::fill(regions.begin(), regions.end(), 0);
    std::atomic<int> next{0};
    std::vector<std::thread> pool;
    for (int t = 0; t < threads; ++t)
      pool.emplace_back([&] {
        std::vector<uint8_t> full(lay.stride);
        for (int e; (e = next.fetch_add(1)) < E;) {
          std::fill(full.begin(), full.end(), 0);
          q38::pack_expert(lay, ck, buf, l, e, full.data());
          for (int c = 0; c < 2; ++c)
            k::moe_split_tp(fl, cfg.hidden, cfg.moe_inter, 2, c, full.data(), tl, regions.data() + (static_cast<size_t>(c) * E + e) * st);
        }
      });
    for (auto &t : pool) t.join();
    char n[48];
    std::snprintf(n, sizeof n, "/experts-L%02d.tp2", l);
    const std::string path = out + n, part = path + ".part";
    std::ofstream f(part, std::ios::binary);
    f.write(reinterpret_cast<const char *>(regions.data()), static_cast<std::streamsize>(regions.size()));
    f.close();
    if (!f) throw std::runtime_error("write failed: " + part);
    fs::rename(part, path);
    std::printf("layer %2d -> %s (%.0f s)\n", l, n + 1, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    std::fflush(stdout);
  }
  return 0;
}

}  // namespace xenithium:bake
