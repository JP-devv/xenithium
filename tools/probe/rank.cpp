// xenithium-probe rank: a hot-expert ranking (XRC1, model/route_counts.hpp) for a MoE checkpoint, from the decode routes
// of real chat turns. The server reads it with --hot-rank (XEN_HOT_RANK); scripts/make-hot-rank.sh runs this.
// xenithium-probe rank --model DIR --prompts F.jsonl --out F.xrc [--gen N] [--context N]
// [--hot N] [--hot-rank F.xrc] [--trace-out F.xrt] [--moe-cpu MODE --moe-cpu-pct P --moe-cpu-threads N]
// F.jsonl: one OpenAI request per line ({"id", "messages", "max_tokens"}), rendered by the checkpoint's chat template
// (the server's defaults: generation prompt, thinking as the template decides), prefilled, then greedy decode until an
// EOS id or min(--gen N (default 192), max_tokens) tokens. Every decode step's routing (top-k ids per MoE layer) adds 1
// to its (layer, expert) count; prompt rows do not count (decode is what reads cold experts one step at a time). The
// arch comes from the checkpoint (qwen38).
// Outputs do not depend on the placement, so the run uses the default (auto) hot set.
// --hot N caps the hot experts per card (the served preset's count) and
// --hot-rank F places them by an XRC1 ranking, so the per-prompt ms per step and cold experts per step are those of that
// placement (the tokens are not: they never depend on it). --trace-out F writes every decode step's routes as an XRT1
// trace ([steps][layers][topk] uint16 ids per prompt).
// A prompt the chat template cannot render is skipped with a line (real session dumps carry odd messages).
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "xenithium/arch/qwen38/model.hpp"
#include "xenithium/kernels/rows.hpp"
#include "xenithium/model/arch.hpp"
#include "xenithium/model/config.hpp"
#include "xenithium/model/route_counts.hpp"
#include "xenithium/rt/fabric.hpp"
#include "xenithium/rt/l0.hpp"
#include "xenithium/text/chat.hpp"
#include "xenithium/text/tokenizer.hpp"
#include "xenithium/util/json.hpp"

#include "probe.hpp"

namespace q38 = xenithium::arch::qwen38;
namespace js = xenithium::json;

namespace {

using clk = std::chrono::steady_clock;

std::string read_all(const std::string &path) {
  std::FILE *f = std::fopen(path.c_str(), "rb");
  if (!f) return {};
  std::string t;
  char b[65536];
  for (size_t r; (r = std::fread(b, 1, sizeof b, f)) > 0;) t.append(b, r);
  std::fclose(f);
  return t;
}

std::vector<int32_t> eos_ids(const std::string &dir) {
  std::vector<int32_t> eos;
  js::Value v;
  if (!js::parse(read_all(dir + "/generation_config.json"), v)) return eos;
  if (const js::Value *e = v.get("eos_token_id")) {
    if (e->is_number()) eos.push_back(static_cast<int32_t>(e->num));
    else if (e->is_array())
      for (const auto &x : e->arr)
        if (x.is_number()) eos.push_back(static_cast<int32_t>(x.num));
  }
  return eos;
}

// The arch id the registry gives this checkpoint ("" when none).
std::string arch_of(const std::string &dir) {
  const auto mc = xenithium::model::ModelConfig::load(dir);
  for (const auto &e : xenithium::model::arch_registry())
    if (e.matches(mc)) return std::string(e.id);
  return "";
}

struct RankRun {
  int64_t hot = -1;  // --hot: hot experts per card (-1 = what VRAM holds)
  std::string hot_rank;  // --hot-rank: the placement's ranking
  std::string trace_out;  // --trace-out: XRT1 routes
  std::string moe_cpu = "off";  // --moe-cpu: the decode MoE's CPU path for cold experts
  int moe_cpu_pct = 70, moe_cpu_threads = 8, moe_cpu_min_cold = 2;
};

int rank_qwen38(const std::string &dir, const std::string &prompts, const std::string &out, int gen, int64_t context, const RankRun &rr) {
  xenithium::text::Tokenizer tok;
  std::string err;
  if (!tok.load(dir, err)) throw std::runtime_error(err);
  std::string note;
  const auto chat = xenithium::text::open_chat_format("qwen38", dir, "auto", &note);
  if (!chat) throw std::runtime_error("rank: no chat format for qwen38");
  const auto eos = eos_ids(dir);
  const std::string all = read_all(prompts);
  if (all.empty()) throw std::runtime_error("rank: cannot read " + prompts);

  q38::ModelOptions mo;
  mo.kv_split = g_kv_tokens ? xenithium::kernels::QsaKvSplit::tokens : xenithium::kernels::QsaKvSplit::heads;
  mo.context = context;
  mo.slots = 1;
  mo.kv_pool_tokens = 0;
  mo.hot = rr.hot;
  mo.moe_cpu.mode = rr.moe_cpu, mo.moe_cpu.pct = rr.moe_cpu_pct, mo.moe_cpu.threads = rr.moe_cpu_threads;
  mo.moe_cpu.min_cold = rr.moe_cpu_min_cold;
  if (!rr.hot_rank.empty()) {
    const auto c = q38::Config::parse(xenithium::model::ModelConfig::load(dir));
    mo.hot_rank = xenithium::model::read_route_counts(rr.hot_rank, c.layers, c.experts);
  }
  const auto loader = xenithium::l0::Loader::open();
  xenithium::Fabric fab(loader, {0, 1});
  const auto t_load = clk::now();
  q38::Model m(loader, fab, dir, mo);
  const auto &cfg = m.config();
  std::printf("rank: qwen38 loaded in %.1f s (%d layers x %d experts, top-%d); chat %s; up to %d tokens per prompt; MoE CPU path %s\n",
              std::chrono::duration<double>(clk::now() - t_load).count(), cfg.layers, cfg.experts, cfg.topk, note.c_str(), gen, m.load_stats().moe_cpu.c_str());
  std::fflush(stdout);

  const int L = cfg.layers, E = cfg.experts, K = cfg.topk;
  std::vector<double> counts(static_cast<size_t>(L) * E, 0.0);
  std::vector<int32_t> ids;
  int prompts_run = 0;
  int64_t steps_all = 0, cold_all = 0;
  double secs_all = 0, pre_all = 0, pre_tok = 0;
  struct TraceP {
    std::string name;
    int prompt = 0, steps = 0;
    std::vector<uint16_t> ids;
  };
  std::vector<TraceP> trace;
  std::vector<uint8_t> hot(static_cast<size_t>(L) * E);
  for (int l = 0; l < L; ++l)
    for (int e = 0; e < E; ++e) hot[static_cast<size_t>(l) * E + e] = m.expert_hot(l, e) ? 1 : 0;
  std::stringstream ls(all);
  for (std::string line; std::getline(ls, line);) {
    if (line.find_first_not_of(" \t\r") == std::string::npos) continue;
    js::Value v;
    if (!js::parse(line, v)) throw std::runtime_error("rank: a --prompts line is not JSON");
    const js::Value *msgs = v.get("messages");
    if (!msgs || !msgs->is_array()) throw std::runtime_error("rank: a --prompts line without messages");
    const std::string *id = v.get_string("id");
    int maxt = gen;
    if (const js::Value *mt = v.get("max_tokens"))
      if (mt->is_number()) maxt = std::min(gen, static_cast<int>(mt->num));
    std::string text;
    xenithium::text::ChatOptions co;
    if (!chat->render(*msgs, v.get("tools"), co, text, err)) {
      std::printf("  %-24s skipped: the chat template cannot render it (%s)\n", id ? id->c_str() : "?", err.c_str());
      continue;
    }
    const auto pt = tok.encode(text);
    const int n = static_cast<int>(pt.size());
    if (n + 2 >= m.capacity()) {
      std::printf("  %-24s skipped: %d prompt tokens do not fit --context %lld\n", id ? id->c_str() : "?", n, static_cast<long long>(m.capacity()));
      continue;
    }
    m.begin_sequence();
    const auto tp0 = clk::now();
    m.prefill(pt.data(), n);
    const int last = (n - 1) % xenithium::kernels::kDecodeRows;
    int32_t cur = m.tokens(last + 1)[static_cast<size_t>(last)];
    const double pre_s = std::chrono::duration<double>(clk::now() - tp0).count();  // The prefill (to its first token)
    pre_all += pre_s, pre_tok += n;
    const auto t0 = clk::now();
    int steps = 0;
    int64_t cold = 0;
    TraceP tp;
    tp.name = id ? *id : "?";
    tp.prompt = n;
    double busy = 0;  // the steps' wall, without the route reads (a blocking copy per step)
    for (; steps < maxt && m.position() + 1 < m.capacity(); ++steps) {
      if (std::find(eos.begin(), eos.end(), cur) != eos.end()) break;
      const auto ts = clk::now();
      m.step(&cur, 1);
      cur = m.tokens(1)[0];
      busy += std::chrono::duration<double>(clk::now() - ts).count();
      m.route_ids(1, ids);
      for (int l = 0; l < L; ++l)
        for (int j = 0; j < K; ++j) {
          const size_t at = static_cast<size_t>(l) * E + static_cast<size_t>(ids[static_cast<size_t>(l) * K + j]);
          counts[at] += 1.0;
          cold += hot[at] ? 0 : 1;  // 1 row: the step's ids are unique per layer
        }
      if (!rr.trace_out.empty())
        for (int x = 0; x < L * K; ++x) tp.ids.push_back(static_cast<uint16_t>(ids[static_cast<size_t>(x)]));
    }
    tp.steps = steps;
    if (!rr.trace_out.empty()) trace.push_back(std::move(tp));
    (void)t0;
    const double s = busy;
    cold_all += cold;
    ++prompts_run;
    steps_all += steps;
    secs_all += s;
    std::printf("  %-24s prompt %5d tokens, %4d decode steps, %.2f ms per step, %.2f cold experts per step, prefill %.0f ms\n", id ? id->c_str() : "?", n, steps,
                steps ? s * 1e3 / steps : 0.0, steps ? static_cast<double>(cold) / steps : 0.0, pre_s * 1e3);
    std::fflush(stdout);
  }
  std::printf("rank: prefill %.0f tokens/s over %d prompts (%.2f s)\n", pre_all > 0 ? pre_tok / pre_all : 0.0, prompts_run, pre_all);
  if (steps_all == 0) throw std::runtime_error("rank: no decode steps ran (empty prompt file?)");
  const std::string tmp = out + ".tmp";
  const std::filesystem::path op(out);
  if (op.has_parent_path()) std::filesystem::create_directories(op.parent_path());
  xenithium::model::write_route_counts(tmp, L, E, counts);
  std::filesystem::rename(tmp, op);
  std::printf("rank: %d prompts, %lld decode steps (%.2f ms per step, %.3f cold experts per step, hot %lld per card); ranking %s written (%d layers x %d experts)\n",
              prompts_run, static_cast<long long>(steps_all), secs_all * 1e3 / static_cast<double>(steps_all),
              static_cast<double>(cold_all) / static_cast<double>(steps_all), static_cast<long long>(m.load_stats().hot[0]), out.c_str(), L, E);
  if (m.moe_cpu_on()) {
    const auto st = m.moe_cpu_stats();
    std::printf("rank: MoE CPU path: %llu requests, %llu experts (%.2f per decode step), %.1f us per request%s\n", static_cast<unsigned long long>(st.requests),
                static_cast<unsigned long long>(st.experts), static_cast<double>(st.experts) / 2.0 / static_cast<double>(steps_all),
                st.requests ? st.busy_us / static_cast<double>(st.requests) : 0.0, st.timed_out ? "; TIMED OUT (switched off)" : "");
  }
  if (!rr.trace_out.empty()) {  // XRT1: header, layer indices, then per prompt name, prompt tokens, steps, ids
    std::FILE *f = std::fopen(rr.trace_out.c_str(), "wb");
    if (!f) throw std::runtime_error("rank: cannot write " + rr.trace_out);
    const uint32_t magic = 0x31545258u;
    const int32_t h[4] = {L, E, K, static_cast<int32_t>(trace.size())};
    std::fwrite(&magic, 4, 1, f);
    std::fwrite(h, 4, 4, f);
    for (int32_t l = 0; l < L; ++l) std::fwrite(&l, 4, 1, f);
    for (const auto &t : trace) {
      const int32_t nl = static_cast<int32_t>(t.name.size()), ps[2] = {t.prompt, t.steps};
      std::fwrite(&nl, 4, 1, f);
      std::fwrite(t.name.data(), 1, t.name.size(), f);
      std::fwrite(ps, 4, 2, f);
      std::fwrite(t.ids.data(), 2, t.ids.size(), f);
    }
    if (std::fclose(f) != 0) throw std::runtime_error("rank: write error " + rr.trace_out);
    std::printf("rank: trace %s written (%zu prompts)\n", rr.trace_out.c_str(), trace.size());
  }
  return 0;
}

}  // namespace

int cmd_rank(int argc, char **argv) {
  std::string dir, prompts, out;
  int gen = 192;
  int64_t context = 8192;
  RankRun rr;
  for (int i = 0; i + 1 < argc; i += 2) {
    const std::string a = argv[i], v = argv[i + 1];
    if (a == "--model") dir = v;
    else if (a == "--prompts") prompts = v;
    else if (a == "--out") out = v;
    else if (a == "--gen") gen = std::atoi(v.c_str());
    else if (a == "--context") context = std::atoll(v.c_str());
    else if (a == "--hot") rr.hot = std::atoll(v.c_str());
    else if (a == "--hot-rank") rr.hot_rank = v;
    else if (a == "--trace-out") rr.trace_out = v;
    else if (a == "--moe-cpu") rr.moe_cpu = v;
    else if (a == "--moe-cpu-pct") rr.moe_cpu_pct = std::atoi(v.c_str());
    else if (a == "--moe-cpu-threads") rr.moe_cpu_threads = std::atoi(v.c_str());
    else if (a == "--moe-cpu-min-cold") rr.moe_cpu_min_cold = std::atoi(v.c_str());
    else {
      std::fprintf(stderr, "rank: unknown option %s\n", a.c_str());
      return 2;
    }
  }
  if (dir.empty() || prompts.empty() || out.empty() || gen < 1) {
    std::fprintf(stderr, "usage: xenithium-probe rank --model DIR --prompts F.jsonl --out F.xrc [--gen N] [--context N] [--hot N] [--hot-rank F.xrc] [--trace-out F.xrt] [--moe-cpu off|auto|avx512|avx2 --moe-cpu-pct P --moe-cpu-threads N]\n");
    return 2;
  }
  const std::string arch = arch_of(dir);
  if (arch == "qwen38") return rank_qwen38(dir, prompts, out, gen, context, rr);
  std::fprintf(stderr, "rank: %s is %s, not a MoE arch with a hot-expert placement (qwen38)\n", dir.c_str(), arch.empty() ? "no known arch" : arch.c_str());
  return 2;
}
