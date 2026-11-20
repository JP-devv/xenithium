// xenithium-bake ple: the PLE n-gram table, BF16 -> int8 rows (one fp32 scale per row).
// xenithium-bake ple --src BF16_DIR --out DIR [--shards A-B] [--per-file 16] [--threads 32]
// Every table shard...ngram_embedding.shard_S.weight` BF16 [rows][dim] becomes U8 [rows][dim + 4]
// (export.hpp: dim int8 codes, then the fp32 scale amax / 127), the same tensor name, in
// DIR/ple-int8-NN.safetensors (per-file shards each); the first file also carries the table's
// index tensors (layer_multipliers, ngram_heads_offsets, ngram_heads_vocab_sizes) unchanged.
// Prints the relative error of the table (Frobenius, every row).
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <stdexcept>
#include <thread>

#include "bake.hpp"
#include "xenithium/quant/export.hpp"

namespace xenithium::bake {

namespace q = xenithium::quant;
namespace m = xenithium::model;

int cmd_ple(const Args &a) {
  const auto ck = m::Checkpoint::open(a.str("src", home() + "/models/Qwen3.8-Flash-Next-Uncensored"));
  const std::string out = a.str("out");
  if (out.empty()) throw std::runtime_error("--out DIR is required");
  std::filesystem::create_directories(out);
  const int threads = static_cast<int>(a.num("threads", std::thread::hardware_concurrency()));
  const int per_file = static_cast<int>(a.num("per-file", 16));
  // find the table
  std::string base;
  for (const auto &t : ck.tensors()) {
    const size_t p = t.name.find("ple.ple_embedding.ngram_embedding.shard_");
    if (p != std::string::npos) {
      base = t.name.substr(0, p) + "ple.ple_embedding.";
      break;
    }
  }
  if (base.empty()) throw std::runtime_error("no PLE n-gram table in the source");
  int shards = 0;
  while (ck.find(base + "ngram_embedding.shard_" + std::to_string(shards) + ".weight")) ++shards;
  int s0 = 0, s1 = shards;
  if (a.has("shards")) {
    const std::string r = a.str("shards");
    const size_t d = r.find('-');
    s0 = std::stoi(r.substr(0, d));
    s1 = (d == std::string::npos ? s0 : std::stoi(r.substr(d + 1))) + 1;
    if (s0 < 0 || s1 > shards || s0 >= s1) throw std::runtime_error("--shards out of range");
  }
  std::printf("PLE table: %s* , %d shards, baking %d..%d\n", base.c_str(), shards, s0, s1 - 1);
  double e2 = 0, r2 = 0;
  int64_t rows_all = 0;
  const auto t0 = std::chrono::steady_clock::now();
  for (int f0 = s0; f0 < s1; f0 += per_file) {
    const int f1 = std::min(s1, f0 + per_file);
    std::vector<std::vector<uint8_t>> store;
    std::vector<q::StTensor> ts;
    store.reserve(static_cast<size_t>(f1 - f0) + 3);
    if (f0 == 0)
      for (const char *nm : {"layer_multipliers", "ngram_heads_offsets", "ngram_heads_vocab_sizes"}) {
        const auto &t = ck.at(base + nm);
        store.emplace_back(t.bytes);
        read_tensor_range(ck, t, 0, t.bytes, store.back().data());
        ts.push_back({t.name, t.dtype_str, t.shape, store.back().data(), t.bytes});
      }
    for (int s = f0; s < f1; ++s) {
      const auto &t = ck.at(base + "ngram_embedding.shard_" + std::to_string(s) + ".weight");
      if (t.dtype != m::DType::bf16 || t.shape.size() != 2) throw std::runtime_error(t.name + ": want BF16 [rows, dim]");
      const int64_t rows = t.dim(0);
      const int dim = static_cast<int>(t.dim(1));
      std::vector<uint16_t> src(static_cast<size_t>(t.bytes / 2));
      // parallel preads + conversion by row blocks
      store.emplace_back(static_cast<size_t>(rows) * (dim + 4));
      uint8_t *dst = store.back().data();
      std::vector<double> pe(static_cast<size_t>(threads), 0.0), pr(static_cast<size_t>(threads), 0.0);
      std::atomic<int64_t> next{0};
      const int64_t blk = 65536;
      std::vector<std::thread> pool;
      for (int ti = 0; ti < threads; ++ti)
        pool.emplace_back([&, ti] {
          for (int64_t b; (b = next.fetch_add(blk)) < rows;) {
            const int64_t n = std::min(blk, rows - b);
            read_tensor_range(ck, t, static_cast<uint64_t>(b) * dim * 2, static_cast<uint64_t>(n) * dim * 2, src.data() + b * dim);
            q::ple_int8_rows(src.data() + b * dim, n, dim, dst + static_cast<size_t>(b) * (dim + 4), &pe[static_cast<size_t>(ti)],
                             &pr[static_cast<size_t>(ti)]);
          }
        });
      for (auto &p : pool) p.join();
      for (int ti = 0; ti < threads; ++ti) e2 += pe[static_cast<size_t>(ti)], r2 += pr[static_cast<size_t>(ti)];
      rows_all += rows;
      ts.push_back({t.name, "U8", {rows, dim + 4}, dst, store.back().size()});
    }
    char name[64];
    std::snprintf(name, sizeof name, "/ple-int8-%02d.safetensors", f0 / per_file);
    q::write_safetensors(out + name, ts,
                         {{"format", "pt"}, {"xenithium_scheme", "ple-int8-row-f32scale"}, {"row_layout", "dim x int8 q | f32 LE s; s = amax / 127; q = rne(x / s)"}});
    const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("shards %3d..%3d -> %s  (%.0f s, rel error so far %.5f)\n", f0, f1 - 1, name + 1, sec, std::sqrt(e2 / r2));
    std::fflush(stdout);
  }
  std::printf("PLE int8: %lld rows, relative error %.5f (Frobenius)\n", static_cast<long long>(rows_all), std::sqrt(e2 / r2));
  return 0;
}

}  // namespace xenithium:bake
