// xenithium-inspect: open a model directory through its architecture module, validate the
// checkpoint against the arch's expectations, and print the per-card byte budget of a
// deployment (VRAM fit, hot experts, decode floor). Host-only: no card needed.
// xenithium-inspect MODEL_DIR [--cards N] [--rows R] [--context N] [--seqs N]
// [--dense-bytes 1|2] [--kv-bytes 1|2] [--drafter] [--unique-experts U]
// [--card b70|b65|b60] [--bw GBS] [--vram-gib G] [--reserve-gib G] [--gpu-embed] [--kv-shard heads|tokens]
// Exit status: 0 = the checkpoint is what the arch serves, 1 = validation problems,
// 2 = usage / I/O error.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

#include "xenithium/model/arch.hpp"
#include "xenithium/model/checkpoint.hpp"
#include "xenithium/model/config.hpp"
#include "xenithium/util/cards.hpp"

namespace m = xenithium::model;

namespace {

void usage() {
  std::fprintf(stderr,
               "usage: xenithium-inspect MODEL_DIR [--cards N] [--rows R] [--context N] [--seqs N]\n"
               "         [--dense-bytes 1|2] [--kv-bytes 1|2] [--drafter] [--unique-experts U]\n"
               "         [--card b70|b65|b60] [--bw GBS] [--vram-gib G] [--reserve-gib G] [--gpu-embed] [--kv-shard heads|tokens]\n");
}

double gib(double b) { return b / (1ull << 30); }

}  // namespace

int main(int argc, char **argv) {
  if (argc < 2 || !std::strcmp(argv[1], "-h") || !std::strcmp(argv[1], "--help")) {
    usage();
    return 2;
  }
  const std::string dir = argv[1];
  m::DeployParams p;
  for (int i = 2; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() -> const char * {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "%s needs a value\n", a.c_str());
        std::exit(2);
      }
      return argv[++i];
    };
    if (a == "--cards") p.cards = std::atoi(next());
    else if (a == "--rows") p.rows = std::atoi(next());
    else if (a == "--context") p.context = std::atoll(next());
    else if (a == "--seqs") p.seqs = std::atoi(next());
    else if (a == "--dense-bytes") p.dense_bytes = std::atoi(next());
    else if (a == "--kv-bytes") p.kv_bytes = std::atoi(next());
    else if (a == "--drafter") p.drafter = true;
    else if (a == "--unique-experts") p.unique_experts = std::atof(next());
    else if (a == "--card") {  // a card of util/cards.hpp: its read floor and free memory (B65 / B60: arithmetic)
      const auto *c = xenithium::util::card_by_key(next());
      if (!c) {
        std::fprintf(stderr, "--card takes b70, b65 or b60\n");
        return 2;
      }
      p.bandwidth_gbs = c->floor_gbs, p.vram_gib = c->free_gib;
    } else if (a == "--bw") p.bandwidth_gbs = std::atof(next());
    else if (a == "--vram-gib") p.vram_gib = std::atof(next());
    else if (a == "--reserve-gib") p.reserve_gib = std::atof(next());
    else if (a == "--gpu-embed") p.embed_on_host = false;
    else if (a == "--kv-shard") {
      const std::string v = next();
      if (v == "heads") p.kv_shard = m::DeployParams::KvShard::heads;
      else if (v == "tokens") p.kv_shard = m::DeployParams::KvShard::tokens;
      else {
        std::fprintf(stderr, "--kv-shard heads|tokens\n");
        return 2;
      }
    }
    else {
      std::fprintf(stderr, "unknown option %s\n", a.c_str());
      usage();
      return 2;
    }
  }

  try {
    const auto t0 = std::chrono::steady_clock::now();
    const m::ModelConfig cfg = m::ModelConfig::load(dir);
    const auto arch = m::open_arch(cfg);
    const m::Checkpoint ck = m::Checkpoint::open(dir);
    const double open_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

    std::printf("model   %s\n", ck.dir().c_str());
    std::printf("arch    %s\n", arch->describe().c_str());
    std::printf("files   %zu shards, %zu tensors, %.2f GiB payload (headers read in %.0f ms)\n\n", ck.shards().size(), ck.size(),
                gib(static_cast<double>(ck.payload_bytes())), open_ms);

    // Component inventory as stored.
    struct Agg {
      size_t n = 0;
      uint64_t bytes = 0;
    };
    std::vector<Agg> agg(static_cast<size_t>(m::Component::count));
    for (const auto &t : ck.tensors()) {
      auto &a = agg[static_cast<size_t>(arch->classify(t.name).comp)];
      ++a.n;
      a.bytes += t.bytes;
    }
    std::printf("%-18s %9s %10s\n", "component", "tensors", "GiB");
    for (size_t c = 0; c < agg.size(); ++c)
      if (agg[c].n) std::printf("%-18s %9zu %10.3f\n", std::string(m::component_name(static_cast<m::Component>(c))).c_str(), agg[c].n, gib(agg[c].bytes));

    std::vector<std::string> warnings;
    const auto problems = arch->validate(ck, &warnings);
    std::printf("\nvalidation: %s (%zu problems)\n", problems.empty() ? "OK" : "FAILED", problems.size());
    for (size_t i = 0; i < problems.size() && i < 40; ++i) std::printf("  ! %s\n", problems[i].c_str());
    if (problems.size() > 40) std::printf("  ! ... %zu more\n", problems.size() - 40);
    for (const auto &w : warnings) std::printf("  - %s\n", w.c_str());

    const m::Budget b = arch->budget(ck, p);
    std::printf("\nbudget per card: %d card(s), %d rows/step, context %lld x %d slot(s), dense %s, KV %s, %.0f GB/s floor, "
                "%.2f GiB VRAM, reserve %.2f GiB\n",
                p.cards, p.rows, static_cast<long long>(p.context), p.seqs, p.dense_bytes == 1 ? "8-bit" : "bf16",
                p.kv_bytes == 1 ? "8-bit" : "bf16", p.bandwidth_gbs, p.vram_gib, p.reserve_gib);
    std::printf("%-34s %13s %13s %10s  %s\n", "component", "resident GiB", "step MiB", "floor us", "note");
    for (const auto &r : b.rows) {
      std::printf("%-34s %13.3f %13.2f %10.1f  %s\n", r.name.c_str(), gib(r.resident_bytes), r.step_bytes / (1 << 20),
                  r.step_bytes / (p.bandwidth_gbs * 1e9) * 1e6, r.note.c_str());
    }
    const bool moe = b.experts_per_card > 0;  // dense archs report no expert rows
    if (moe)
      std::printf("%-34s %13.3f %13.2f %10.1f  %.1f unique per layer per step, half on each card\n", "routed experts, activated",
                  0.0, b.expert_step_bytes / (1 << 20), b.expert_step_bytes / (p.bandwidth_gbs * 1e9) * 1e6, b.unique_experts);
    std::printf("\nKV per token per slot   %.0f B per card\n", b.kv_bytes_per_token);
    if (moe)
      std::printf("one routed expert       %.0f B (%.3f MiB) -> 1 expert = %.0f tokens of one slot's KV\n", b.expert_bytes,
                  b.expert_bytes / (1 << 20), b.expert_bytes / b.kv_bytes_per_token);
    std::printf("%s resident     %.3f GiB per card (incl. KV at this context)\n", moe ? "non-expert" : "all-weight ", gib(b.resident_non_expert));
    if (moe)
      std::printf("hot experts that fit    %lld of %lld per card (%.1f %%)\n", static_cast<long long>(b.hot_experts_fit),
                  static_cast<long long>(b.experts_per_card), 100.0 * b.hot_experts_fit / b.experts_per_card);
    std::printf("decode step floor       %.2f ms (%sbytes / bandwidth, before any kernel overhead)\n", b.floor_ms, moe ? "all hot; " : "");
    for (const auto &n : b.notes) std::printf("  - %s\n", n.c_str());
    return problems.empty() ? 0 : 1;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "xenithium-inspect: %s\n", e.what());
    return 2;
  }
}
