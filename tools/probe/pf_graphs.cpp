// xenithium-probe model --pf-graph-gate N (qwen38): prompt-call graphs (ModelOptions::prefill_graphs) against the
// launches, in one load (the model loaded with prefill_graphs; Model::set_prefill_graphs switches them per run).
//
// 1. Bits. A script of N prompt calls per slot on two slots, the slots' calls alternating: sizes under
//    kPfGraphMaxRows on both sides of kPfSplitMinRows (the split sums and the mailbox sums), positions past the first
//    QSA bound bucket (2,048: the indexer's score and select run) and through the PLE layer, a 1-row decode step
//    after every ninth call (the HC state and the GDN parity move), six calls in parts (prefill_begin /
//    prefill_advance, the other slot's decode step between the parts). The script runs once launch by launch and once
//    with graphs, each slot's sequence from position 0 again. After every call the last slice's logits must have the
//    same bits; then 8 greedy decode tokens per slot must agree. The graph run must have recorded graphs and replayed
//    them.
// 2. Timing. Per size of --pf-graph-sizes: calls of that many rows from an empty sequence, blocks of replayed graphs
//    and of launches in turn (the graphs recorded first), medians of each. N < 0: the timing alone (one slot).
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

#include "xenithium/arch/qwen38/model.hpp"
#include "probe.hpp"

namespace q38 = xenithium::arch::qwen38;

namespace {

// FNV-1a over the bits of a call's logits: a mismatch in any value changes it
uint64_t bits_hash(const std::vector<float> &v) {
  uint64_t h = 1469598103934665603ull;
  for (float f : v) {
    uint32_t u = 0;
    std::memcpy(&u, &f, 4);
    for (int i = 0; i < 4; ++i) h = (h ^ ((u >> (8 * i)) & 0xFF)) * 1099511628211ull;
  }
  return h;
}

// The script's call sizes: on both sides of kPfSplitMinRows (mailbox and split sums), repeats among them.
constexpr int kSizes[] = {64, 64, 64, 64, 40, 24, 64, 64, 17, 47, 8, 56, 8, 56, 8, 56, 100, 28, 64, 64, 127, 1};
constexpr int kStepEvery = 9;      // a 1-row decode step after every ninth call of a slot
constexpr int kPartsCalls = 6;     // the slot's last calls run in parts, kPartsRows rows each (their keys repeat)
constexpr int kPartsRows = 48;
constexpr int kPartLayers = 16;    // layers per part
constexpr int kTailTokens = 8;     // greedy decode tokens per slot after the script

struct ScriptResult {
  std::vector<uint64_t> hashes;      // per call, slots interleaved
  std::vector<int32_t> tail;         // the greedy tokens, slot 0 then slot 1
  int64_t positions[2] = {};
};

ScriptResult run_script(q38::Model &m, const std::vector<int32_t> &tok, int calls) {
  ScriptResult r;
  std::vector<float> lg;
  constexpr int kSlots = 2;
  for (int s = 0; s < kSlots; ++s) {
    m.use_slot(s);
    m.begin_sequence();
  }
  int next[kSlots] = {};
  for (int i = 0; i < calls; ++i)
    for (int s = 0; s < kSlots; ++s) {
      m.use_slot(s);
      const bool parts = i >= calls - kPartsCalls;
      const int n = parts ? kPartsRows : kSizes[(i + s) % std::size(kSizes)];
      const int64_t p = m.position();
      if (p + n + kTailTokens + 1 >= static_cast<int64_t>(tok.size())) throw std::runtime_error("pf-graph-gate: the token file is too short for the script");
      const int32_t *t = tok.data() + p;
      if (parts) {  // in parts, a decode step of the other slot between them (as the server)
        m.prefill_begin(t, n);
        while (!m.prefill_advance(kPartLayers)) {
          m.use_slot(1 - s);
          m.step(&next[1 - s], 1);
          next[1 - s] = m.tokens(1)[0];
          m.use_slot(s);
        }
      } else {
        m.prefill(t, n);
      }
      const int rows = (n - 1) % 4 + 1;
      m.logits(rows, lg);
      r.hashes.push_back(bits_hash(lg));
      next[s] = m.tokens(rows)[static_cast<size_t>(rows - 1)];
      if (i % kStepEvery == kStepEvery - 1) {
        m.step(&next[s], 1);
        next[s] = m.tokens(1)[0];
      }
    }
  for (int s = 0; s < kSlots; ++s) {
    m.use_slot(s);
    for (int i = 0; i < kTailTokens; ++i) {
      r.tail.push_back(next[s]);
      m.step(&next[s], 1);
      next[s] = m.tokens(1)[0];
    }
    r.positions[s] = m.position();
  }
  return r;
}

double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v.empty() ? 0 : v[v.size() / 2];
}

int pf_graph_bits(q38::Model &m, const std::vector<int32_t> &tok, int calls) {
  int rc = 0;
  auto check = [&](bool ok, const std::string &what) {
    std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
    rc |= !ok;
  };
  if (m.slots() < 2) throw std::runtime_error("pf-graph-gate: needs two slots");
  m.set_prefill_graphs(false);
  const auto t0 = std::chrono::steady_clock::now();
  const ScriptResult launches = run_script(m, tok, calls);
  const auto t1 = std::chrono::steady_clock::now();
  m.set_prefill_graphs(true);
  const int rec0 = m.pf_graphs_recorded();
  const double ms0 = m.pf_graph_record_ms();
  const int64_t rep0 = m.pf_graphs_replayed();
  const ScriptResult graphs = run_script(m, tok, calls);
  const auto t2 = std::chrono::steady_clock::now();
  const int rec = m.pf_graphs_recorded() - rec0;
  const int64_t rep = m.pf_graphs_replayed() - rep0;
  size_t same = 0;
  for (size_t i = 0; i < launches.hashes.size(); ++i) same += launches.hashes[i] == graphs.hashes[i];
  std::printf("pf-graph-gate: %zu calls on 2 slots to positions %lld / %lld: launches %.1f s, graphs %.1f s (%d recorded in %.0f ms, %lld replayed)\n",
              launches.hashes.size(), static_cast<long long>(graphs.positions[0]), static_cast<long long>(graphs.positions[1]),
              std::chrono::duration<double>(t1 - t0).count(), std::chrono::duration<double>(t2 - t1).count(), rec, m.pf_graph_record_ms() - ms0, static_cast<long long>(rep));
  check(same == launches.hashes.size(),
        "prompt calls replayed from graphs == launched, the last slice's logits bit for bit (" + std::to_string(same) + " / " + std::to_string(launches.hashes.size()) + ")");
  check(launches.tail == graphs.tail, "greedy decode after the script: the same " + std::to_string(launches.tail.size()) + " tokens");
  check(rec > 0 && rep > rec, "the graph run recorded graphs and replayed them more often than it recorded");
  check(m.wire_errors() == 0, "no mailbox timeout");
  return rc;
}

void pf_graph_timing(q38::Model &m, const std::vector<int32_t> &tok, const std::vector<int> &sizes) {
  // blocks of kBlock calls with graphs, then kBlock launched, kBlocks times (consecutive calls alternate their wire
  // parities: two keys per size, both recorded by the warm-up)
  constexpr int kWarm = 4, kBlock = 6, kBlocks = 4;
  m.use_slot(0);
  for (int n : sizes) {
    if (n < 1 || n >= q38::kPfGraphMaxRows || n > static_cast<int>(tok.size())) continue;
    m.set_prefill_graphs(true);
    for (int i = 0; i < kWarm; ++i) {
      m.begin_sequence();
      m.prefill(tok.data(), n);
    }
    std::vector<double> on, off;
    const int64_t rep_before = m.pf_graphs_replayed();
    for (int i = 0; i < 2 * kBlocks * kBlock; ++i) {
      const bool g = (i / kBlock) % 2 == 0;
      m.set_prefill_graphs(g);
      m.begin_sequence();
      const auto ts = std::chrono::steady_clock::now();
      m.prefill(tok.data(), n);
      (g ? on : off).push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ts).count());
    }
    std::printf("pf-graph-gate: %d rows from position 0: graph %.2f ms, launches %.2f ms (median of %d each; %lld replays)\n", n, median(on), median(off),
                kBlocks * kBlock, static_cast<long long>(m.pf_graphs_replayed() - rep_before));
  }
  m.set_prefill_graphs(false);
}

}  // namespace

// calls > 0: the bits, then the timing; calls < 0: the timing alone
int pf_graph_gate(q38::Model &m, const std::vector<int32_t> &tok, int calls, const std::vector<int> &sizes) {
  int rc = 0;
  if (calls > 0) rc |= pf_graph_bits(m, tok, calls);
  pf_graph_timing(m, tok, sizes);
  return rc;
}
