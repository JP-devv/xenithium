// xenithium-probe select35 (audit C1): page-selected decode attention on the dense Qwen3.8-27B vs
// dense attention, one load for every case.
// xenithium-probe select35 --model DIR [--ppl F [--pf-lens P,P]] [--doc F,F] [--sel-budgets B,B] [--generate T] [--step-reps N]
// [--dec-attn exact|int8-fast] [--queue-prof 0|1] [--sel-kv 1] [--sel-pf 1 [--pf-a8 1] [--pf-chunk R]]
// Prompts, each prefilled into slot 0 (dense prefill: the selection is a decode-step lever), snapshotted, slot 1 forked
// from it:
// needle per length P of --pf-lens (default 131072,261120; needs --ppl): a user turn of the text of F's chunk 0
// (repeated past its end) with a needle sentence ("The secret passphrase is...") at 35 % depth and the
// question last. Its answer is a few tokens, so the KL steps after it run on (a high-entropy tail): the
// needle is the gate here, the KL is reported.
// doc per file of --doc: the file's text as the user turn (a document and a question whose answer is longer than
// T tokens): KL on a natural reply.
// raw with --sel-raw 1, per length P of --pf-lens: the first P tokens of F's chunk 0 as they are (no chat turn).
// The first token scored is the text's last one, so the first KL is that of the next-token distribution after
// the text: a copy test when the text repeats an earlier span there.
// The needle and doc prompts end with the assistant turn, thinking off. Per budget B (pages per sequence and kv head):
// KL T teacher-forced steps, slot 0 dense / slot 1 selected, the same token (slot 0's greedy pick) into both:
// KL(dense || selected) of every step's logits (mean, max) and top-1 agreement (reported: the budget is the
// user's trade);
// reply slot 1 again from the snapshot, T free greedy steps with the selection: its text beside the dense one
// (needle prompts: PASS = the selected run names the passphrase whenever the dense run does);
// speed steps at the position reached, 1 row and 4 rows (a verify step), dense and selected alternating per round,
// --step-reps rounds of 32: host wall median per step (native xenithium, two B70s), and one profiled step each
// (card 0's attention launch sum).
// --sel-kv 1 (documents only; instead of the three above): what a page-selected PREFILL would cost in quality, as a bound
// before any build. Per budget B the prompt's tokens past B pages are fed to slot 1 as 4-row decode steps with the
// selection, so their KV and GDN state are built on selected attention (a segment's rows share one page set, as a
// prefill's row block would); the reply, its steps selected too (what a server with the selection would then serve), is
// compared with the dense prompt's dense reply over T teacher-forced steps. The default run's KL at the same budget is
// the reply steps' share (dense KV), so the difference is the KV's.
// A control feeds the same tail by dense decode steps: decode-built against prefill-built KV, the floor of the numbers.
// --sel-pf 1; instead of the default three): the page-selected PREFILL itself
// (ModelOptions:prefill_select). Per budget B the prompt is prefilled again with B pages per attention row block and
// kv head; reported: the prefill's wall beside the dense one, the KL of the prompt's last row (the first reply token),
// and the reply's KL against the dense prompt's dense reply over T teacher-forced steps - once with dense steps (the
// selected KV's share alone) and once with the steps selected too (what a server with both settings serves). Needle
// prompts: PASS = the free reply on the selected prefill names the passphrase whenever the dense one does. --pf-a8 1
// and --pf-chunk R give the served prefill numerics and chunk; --dec-attn int8-fast also sets the prefill attention to
// int8-fast (what XEN_PREFILL_ATTN=auto serves on the int8 KV).
// --dec-attn: the decode attention numerics of every step, dense and selected alike (int8-fast: what `auto` serves).
// --queue-prof 0 runs the queues without event profiling, as the server does: the walls are then the served ones (a
// profiling queue makes every submit slower and the step host-bound) and the profiled column is left out. Take the walls
// from a --queue-prof 0 load and the per-launch times from a default one.
// No drafter; int8 KV; two slots on a shared pool.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <map>
#include <numeric>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include "xenithium/arch/qwen35/model.hpp"
#include "xenithium/rt/fabric.hpp"
#include "xenithium/text/tokenizer.hpp"

#include "probe.hpp"  // the run-wide flags

namespace {
namespace q35 = xenithium::arch::qwen35;
using clk = std::chrono::steady_clock;
using Tokens = std::vector<int32_t>;

constexpr std::string_view kPassphrase = "MAGENTA-7134";
constexpr std::string_view kUserOpen = "<|im_start|>user\n";
constexpr std::string_view kAssistantOpen = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";  // thinking off
constexpr int kTimedSteps = 32;  // steps per timing round and case
constexpr int kAuditStride = 9;  // audit_bounds reads every 9th page back (~65 KB per page, layer and kv head)
constexpr int kTailRows = 4;  // --sel-kv: rows per step of the prompt's tail (the widest step without a drafter)
constexpr int kTimedRows[] = {1, 4};  // the timed step shapes: a plain step and a 4-row verify (the widest without a drafter)

struct Prompt {
  std::string label;
  Tokens tokens;
  bool needle = false;  // the reply should name kPassphrase
  bool raw = false;  // plain text: the steps start on its last token (the prefill stops one short)
};

bool read_chunk0(const std::string &path, Tokens &out) {
  std::ifstream f(path, std::ios::binary);
  int32_t n = 0, len = 0;
  if (!f || !f.read(reinterpret_cast<char *>(&n), 4) || n <= 0 || !f.read(reinterpret_cast<char *>(&len), 4) || len <= 1) return false;
  out.resize(static_cast<size_t>(len));
  return static_cast<bool>(f.read(reinterpret_cast<char *>(out.data()), static_cast<std::streamsize>(len) * 4));
}

bool read_text(const std::string &path, std::string &out) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return false;
  out.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
  return !out.empty();
}

double kl(const std::vector<float> &p, const std::vector<float> &q) {  // KL(softmax p || softmax q), double
  double mp = -1e300, mq = -1e300;
  for (size_t i = 0; i < p.size(); ++i) mp = std::max(mp, static_cast<double>(p[i])), mq = std::max(mq, static_cast<double>(q[i]));
  double sp = 0, sq = 0;
  for (size_t i = 0; i < p.size(); ++i) sp += std::exp(p[i] - mp), sq += std::exp(q[i] - mq);
  const double lp = mp + std::log(sp), lq = mq + std::log(sq);
  double k = 0;
  for (size_t i = 0; i < p.size(); ++i) {
    const double a = p[i] - lp;
    k += std::exp(a) * (a - (q[i] - lq));
  }
  return k;
}

std::vector<std::string> split(const std::string &s) {
  std::vector<std::string> v;
  std::stringstream ss(s);
  std::string t;
  while (std::getline(ss, t, ','))
    if (!t.empty()) v.push_back(t);
  return v;
}

std::vector<int> ints(const std::string &s) {
  std::vector<int> v;
  for (const auto &t : split(s)) v.push_back(std::stoi(t));
  return v;
}

double median(std::vector<double> v) {
  if (v.empty()) return 0;
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

void append(Tokens &to, const Tokens &from) { to.insert(to.end(), from.begin(), from.end()); }

// `text` (repeated to fill) with the needle sentence at 35 % depth and the question last: `length` tokens in all
Prompt needle_prompt(const xenithium::text::Tokenizer &tok, const Tokens &text, int length) {
  const std::string pass(kPassphrase);
  const Tokens needle = tok.encode("\n\nThe secret passphrase is " + pass + ". Remember this passphrase, it will be asked for later.\n\n");
  const Tokens head = tok.encode(std::string(kUserOpen)), sep = tok.encode("\n\n");
  const Tokens question =
      tok.encode("\n\nWhat is the secret passphrase mentioned in the text above? Answer with the passphrase only." + std::string(kAssistantOpen));
  const int64_t body_len = length - static_cast<int64_t>(head.size() + needle.size() + question.size());
  Tokens body;
  while (static_cast<int64_t>(body.size()) < body_len) {
    if (!body.empty()) append(body, sep);
    append(body, text);
  }
  body.resize(static_cast<size_t>(body_len));
  const auto cut = body.begin() + static_cast<long>(body_len * 35 / 100);
  Prompt p{"needle at 35 %", head, true};
  p.tokens.insert(p.tokens.end(), body.begin(), cut);
  append(p.tokens, needle);
  p.tokens.insert(p.tokens.end(), cut, body.end());
  append(p.tokens, question);
  return p;
}

// the file's text as one user turn
Prompt doc_prompt(const xenithium::text::Tokenizer &tok, const std::string &path, const std::string &text) {
  return Prompt{"document " + path, tok.encode(std::string(kUserOpen) + text + std::string(kAssistantOpen)), false};
}

// The host's version of what a sequence measures for itself (ModelOptions:select_self; --sel-self checks
// Model:select_self_weights against it): per (attention layer, kv head) and step the pages' scores - the largest of
// q. mid + 0.5 |q|. rad over the head's q heads, mid / rad from the stored keys - through select_curve_add, and the
// summed curves through SelectShare:weights.
struct HostShare {
  static constexpr int kFractions = xenithium::arch::qwen35::SelectShare::kFractions;
  static constexpr int kSink = xenithium::kernels::kGqaSelSink, kLocal = xenithium::kernels::kGqaSelLocal;
  std::map<std::pair<int, int>, std::vector<double>> lost;  // (layer, kv head) -> the curves summed over the steps

  static float dot(const float *a, const float *b) {
    float s[8] = {};
    for (int d = 0; d < 256; d += 8)
      for (int j = 0; j < 8; ++j) s[j] += a[d + j] * b[d + j];
    return ((s[0] + s[1]) + (s[2] + s[3])) + ((s[4] + s[5]) + (s[6] + s[7]));
  }
  // One (layer, kv head): keys [tokens][256] (dequantised), the steps' (position, q [6][256]).
  void add(std::pair<int, int> unit, const std::vector<float> &keys, const std::vector<std::pair<int64_t, std::vector<float>>> &steps, int threads) {
    constexpr int G = 6, D = 256, PG = 64;
    const int pages_all = static_cast<int>(keys.size() / (static_cast<size_t>(PG) * D));
    std::vector<float> mid(static_cast<size_t>(pages_all) * D), rad(mid.size());
    for (int p = 0; p < pages_all; ++p)
      for (int d = 0; d < D; ++d) {
        float lo = INFINITY, hi = -INFINITY;
        for (int t = 0; t < PG; ++t) {
          const float v = keys[(static_cast<size_t>(p) * PG + t) * D + d];
          lo = std::min(lo, v), hi = std::max(hi, v);
        }
        mid[static_cast<size_t>(p) * D + d] = 0.5f * (hi + lo), rad[static_cast<size_t>(p) * D + d] = 0.5f * (hi - lo);
      }
    std::vector<std::vector<double>> part(static_cast<size_t>(threads), std::vector<double>(kFractions, 0.0));
    auto work = [&](int w) {
      std::vector<float> key, absq(D);
      for (size_t s = static_cast<size_t>(w); s < steps.size(); s += static_cast<size_t>(threads)) {
        const int tokens = static_cast<int>(steps[s].first) + 1, pages = (tokens + PG - 1) / PG;
        key.assign(static_cast<size_t>(pages), -INFINITY);
        for (int g = 0; g < G; ++g) {
          const float *q = steps[s].second.data() + static_cast<size_t>(g) * D;
          for (int d = 0; d < D; ++d) absq[static_cast<size_t>(d)] = std::abs(q[d]);
          for (int p = 0; p < pages; ++p)
            key[static_cast<size_t>(p)] = std::max(key[static_cast<size_t>(p)], dot(&mid[static_cast<size_t>(p) * D], q) + 0.5f * dot(&rad[static_cast<size_t>(p) * D], absq.data()));
        }
        xenithium::arch::qwen35::select_curve_add(key, tokens, PG, kSink, kLocal, 1.f / 16, part[static_cast<size_t>(w)]);
      }
    };
    std::vector<std::thread> pool;
    for (int w = 0; w < threads; ++w) pool.emplace_back(work, w);
    for (auto &t : pool) t.join();
    auto &sum = lost[unit];
    sum.resize(kFractions, 0.0);
    for (const auto &p : part)
      for (int f = 0; f < kFractions; ++f) sum[static_cast<size_t>(f)] += p[static_cast<size_t>(f)];
  }
  std::vector<float> weights() const {  // [unit], the units in the map's order (layer, kv head)
    std::vector<double> curves;
    for (const auto &u : lost) curves.insert(curves.end(), u.second.begin(), u.second.end());
    return xenithium::arch::qwen35::SelectShare::weights(curves, static_cast<int>(lost.size()));
  }
};

// The measurements of one loaded model: every prompt, every budget.
struct Probe {
  q35::Model &m;
  const xenithium::text::Tokenizer &tok;
  const Tokens &filler;  // tokens the timed steps feed
  int steps;  // T: teacher-forced / free steps
  int rounds;  // timing rounds
  bool profiled;  // the queues record launch times (walls are then host-bound)
  int failures = 0;

  void check(bool ok, const std::string &what) {
    std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
    failures += !ok;
  }
  void select(int pages) {  // 0 = dense
    q35::StepOptions o = m.step_options();
    o.attn_select = pages;
    m.set_step_options(o);
  }
  // one greedy step of the current slot on `token`; returns its pick
  int32_t step(int32_t token, std::vector<float> *logits = nullptr) {
    m.step(&token, 1);
    if (logits) m.logits(1, *logits);
    const int32_t pick = m.picks(0, 1)[0];
    m.accept(1);
    return pick;
  }

  struct Kl {
    double mean = 0, max = 0;
    double first = 0;  // the first step's
    int agree = 0;
    Tokens dense;  // the tokens fed (slot 0's greedy text)
  };
  // T teacher-forced steps from the snapshot: slot 0 dense, slot 1 with `pages`
  Kl teacher_forced(int snap, int32_t first, int pages) {
    m.snap_restore(0, snap);
    m.snap_fork(1, 0, snap);
    Kl r;
    std::vector<float> dense_logits, sel_logits;
    int32_t token = first;
    for (int i = 0; i < steps; ++i) {
      r.dense.push_back(token);
      m.use_slot(0);
      select(0);
      const int32_t dense_pick = step(token, &dense_logits);
      m.use_slot(1);
      select(pages);
      const int32_t sel_pick = step(token, &sel_logits);
      const double d = kl(dense_logits, sel_logits);
      if (i == 0) r.first = d;
      r.mean += d / steps, r.max = std::max(r.max, d), r.agree += dense_pick == sel_pick;
      token = dense_pick;
    }
    return r;
  }
  // T free greedy steps of slot 1 from the snapshot with `pages`
  Tokens free_run(int snap, int32_t first, int pages) {
    m.snap_fork(1, 0, snap);  // slot 0's pages below the snapshot are unchanged (its steps wrote past it)
    m.use_slot(1);
    select(pages);
    Tokens out;
    int32_t token = first;
    for (int i = 0; i < steps; ++i) {
      out.push_back(token);
      token = step(token);
    }
    return out;
  }
  // step walls at the position reached: slot 0 dense, slot 1 selected (positions within T of each other)
  void speed(int pages, int rows) {
    std::vector<double> dense_ms, sel_ms;
    Tokens batch(static_cast<size_t>(rows));
    auto run = [&](int slot, int budget, std::vector<double> &walls, int n) {
      m.use_slot(slot);
      select(budget);
      for (int i = 0; i < n; ++i) {
        for (size_t j = 0; j < batch.size(); ++j) batch[j] = filler[(static_cast<size_t>(i) * batch.size() + j) % filler.size()];
        const auto t0 = clk::now();
        m.verify_batch({q35::Model::SeqRows{slot, batch.data(), rows}});
        walls.push_back(std::chrono::duration<double, std::milli>(clk::now() - t0).count());
        m.accept_batch({1});
      }
    };
    for (int r = 0; r < rounds; ++r) {
      run(0, 0, dense_ms, kTimedSteps);
      run(1, pages, sel_ms, kTimedSteps);
    }
    std::printf("    %d row(s) at %lld: dense %.2f ms per step, selected %.2f (%+.1f %%)", rows, static_cast<long long>(m.position(1)), median(dense_ms),
                median(sel_ms), 100.0 * (median(sel_ms) / median(dense_ms) - 1));
    if (profiled) {
      m.set_profile(true);
      run(0, 0, dense_ms, 1);
      const double dense_attn_us = m.profile().attn;
      run(1, pages, sel_ms, 1);
      const double sel_attn_us = m.profile().attn;
      m.set_profile(false);
      std::printf("; card 0 attention (profiled) %.2f vs %.2f ms", dense_attn_us / 1e3, sel_attn_us / 1e3);
    }
    std::printf("\n");
  }

  // The key bounds of the slot's whole pages from `first_page` on (every kAuditStride-th and the last): they are what
  // the selection scores, and a decode step's pages get them only when the sequence leaves the page.
  void audit_bounds(int slot, int first_page, const std::string &what) {
    const int whole = static_cast<int>(m.position(slot) / 64);
    std::vector<int> pages;
    for (int p = first_page; p < whole; p += kAuditStride) pages.push_back(p);
    if (whole > first_page && pages.back() != whole - 1) pages.push_back(whole - 1);
    if (pages.empty()) return;
    const int64_t wrong = m.page_bound_errors(slot, pages);
    check(wrong == 0, "key bounds == the min / max of the stored keys on " + std::to_string(pages.size()) + " of " + what + " (" + std::to_string(wrong) +
                          " page x layer x kv head units differ)");
  }
  // --sel-kv, one reply: T teacher-forced steps of the current slot on `tokens` against the dense prompt's logits
  struct ReplyKl {
    double mean = 0, max = 0;
    int agree = 0;
  };
  ReplyKl reply_kl(const Tokens &tokens, const std::vector<std::vector<float>> &dense_logits, int pages) {
    select(pages);
    ReplyKl r;
    std::vector<float> logits;
    for (int i = 0; i < steps; ++i) {
      const int32_t pick = step(tokens[static_cast<size_t>(i)], &logits);
      const double d = kl(dense_logits[static_cast<size_t>(i)], logits);
      r.mean += d / steps, r.max = std::max(r.max, d);
      r.agree += i + 1 < steps ? pick == tokens[static_cast<size_t>(i) + 1] : 1;  // the last pick has no dense token to meet
    }
    return r;
  }
  // --sel-kv, one case: slot 1 = the prompt's first `head` tokens prefilled densely, the rest by decode steps under
  // `pages` (0 = dense steps, the control); then the reply, its steps under `pages` too
  void kv_case(const Prompt &p, const Tokens &reply, const std::vector<std::vector<float>> &dense_logits, int head, int pages) {
    const int total = static_cast<int>(p.tokens.size());
    m.use_slot(1);
    m.reset();
    select(0);
    m.prefill(p.tokens.data(), head);
    select(pages);
    const auto t0 = clk::now();
    for (int at = head; at < total; at += kTailRows) {
      const int rows = std::min(kTailRows, total - at);
      m.step(p.tokens.data() + at, rows);
      m.accept(rows);
    }
    const double tail_s = std::chrono::duration<double>(clk::now() - t0).count();
    const ReplyKl r = reply_kl(reply, dense_logits, pages);
    audit_bounds(1, head / 64, "the pages the decode steps wrote and left");
    std::printf("  %s: %d tokens past %d by %d-row decode steps in %.1f s\n", pages ? (std::to_string(pages) + " pages").c_str() : "control (dense steps)",
                total - head, head, kTailRows, tail_s);
    std::printf("    reply (%s steps) against the dense prompt's: KL mean %.5f, max %.5f over %d teacher-forced steps; top-1 %d / %d\n", pages ? "selected" : "dense",
                r.mean, r.max, steps, r.agree, steps);
  }
  // --sel-pf: the prompt prefilled into the current slot with `pages` per attention row block and kv head (0 = dense);
  // returns the wall in seconds
  double prefill_with(const Prompt &p, int pages) {
    q35::ModelOptions o = m.options();
    o.prefill_select = pages;
    m.reset();
    m.set_prefill_options(o);
    const auto t0 = clk::now();
    m.prefill(p.tokens.data(), static_cast<int>(p.tokens.size()));
    return std::chrono::duration<double>(clk::now() - t0).count();
  }
  // the logits of the prompt's last row after a prefill (the first reply token's distribution)
  std::vector<float> last_row_logits(const Prompt &p) {
    const int slice_rows = static_cast<int>((p.tokens.size() - 1) % 4) + 1;  // the prefill's last 4-row slice
    std::vector<float> all;
    m.logits(slice_rows, all);
    const size_t vocab = all.size() / static_cast<size_t>(slice_rows);
    return std::vector<float>(all.end() - static_cast<long>(vocab), all.end());
  }
  // --sel-pf on one prompt
  bool run_pf(const Prompt &p, const std::vector<int> &budgets) {
    for (int s = 0; s < 2; ++s) {
      m.use_slot(s);
      m.reset();
    }
    m.use_slot(0);
    select(0);
    prefill_with(p, 0);  // the first prefill of a load carries first-use costs: timed on the second
    const double dense_s = prefill_with(p, 0);
    const std::vector<float> dense_last = last_row_logits(p);
    Tokens reply{m.picks(0, 1)[0]};
    std::vector<std::vector<float>> dense_logits(static_cast<size_t>(steps));
    for (int i = 0; i < steps; ++i) reply.push_back(step(reply.back(), &dense_logits[static_cast<size_t>(i)]));
    const std::string dense_text = tok.decode(reply, true);
    const int total = static_cast<int>(p.tokens.size()), pages = (total + 63) / 64;
    std::printf("\n%d-token prompt (%s, %d pages), dense prefill %.2f s (%.0f tok/s); the dense reply: \"%s\"\n", total, p.label.c_str(), pages, dense_s,
                total / dense_s, dense_text.c_str());
    const std::string pass(kPassphrase);
    m.use_slot(1);
    for (const int budget : budgets) {
      if (pages <= budget) {
        std::printf("  %d pages: the prompt is within the budget (the dense prefill)\n", budget);
        continue;
      }
      const double sel_s = prefill_with(p, budget);
      const double first_kl = kl(dense_last, last_row_logits(p));
      const int32_t first_pick = m.picks(0, 1)[0];
      const int snap = m.snap_save(1);
      m.snap_sync();
      const ReplyKl kv_only = reply_kl(reply, dense_logits, 0);
      m.snap_restore(1, snap);
      const ReplyKl served = reply_kl(reply, dense_logits, budget);
      m.snap_restore(1, snap);
      select(budget);
      Tokens free_reply{first_pick};
      for (int i = 0; i < steps; ++i) free_reply.push_back(step(free_reply.back()));
      m.snap_free(snap);
      const std::string sel_text = tok.decode(free_reply, true);
      std::printf("  %d pages (%.1f %% of the context): selected prefill %.2f s (%.0f tok/s, %+.1f %% time); first reply token KL %.5f, %s\n", budget,
                  100.0 * budget / pages, sel_s, total / sel_s, 100.0 * (sel_s / dense_s - 1), first_kl, first_pick == reply[0] ? "the same pick" : "another pick");
      std::printf("    reply against the dense prompt's dense reply, %d teacher-forced steps: dense steps KL mean %.5f, max %.5f, top-1 %d / %d; selected steps KL mean "
                  "%.5f, max %.5f, top-1 %d / %d\n", steps, kv_only.mean, kv_only.max, kv_only.agree, steps, served.mean, served.max, served.agree, steps);
      std::printf("    free reply (selected steps): \"%s\"\n", sel_text.c_str());
      if (p.needle) {
        const bool dense_hit = dense_text.find(pass) != std::string::npos, sel_hit = sel_text.find(pass) != std::string::npos;
        check(!dense_hit || sel_hit, "P " + std::to_string(total) + ", " + std::to_string(budget) + " pages, selected prefill: the needle (" + pass + ") " +
                                         (dense_hit ? "answered on the dense prefill and on the selected one" : "not answered on the dense prefill (no gate)"));
      }
    }
    m.use_slot(1);
    prefill_with(p, 0);  // leaves the prefill dense for the next prompt
    return true;
  }
  // --sel-kv on one document prompt
  bool run_kv(const Prompt &p, const std::vector<int> &budgets) {
    for (int s = 0; s < 2; ++s) {
      m.use_slot(s);
      m.reset();
    }
    m.use_slot(0);
    select(0);
    m.prefill(p.tokens.data(), static_cast<int>(p.tokens.size()));
    Tokens reply{m.picks(0, 1)[0]};
    std::vector<std::vector<float>> dense_logits(static_cast<size_t>(steps));
    for (int i = 0; i < steps; ++i) reply.push_back(step(reply.back(), &dense_logits[static_cast<size_t>(i)]));
    const int total = static_cast<int>(p.tokens.size());
    std::printf("\n%d-token prompt (%s, %d pages); the dense reply: \"%s\"\n", total, p.label.c_str(), (total + 63) / 64, tok.decode(reply, true).c_str());
    int widest_tail_head = total;
    for (const int budget : budgets) {
      const int head = std::min(total, budget * 64) / kTailRows * kTailRows;
      if (head >= total) {
        std::printf("  %d pages: the prompt is within the budget\n", budget);
        continue;
      }
      widest_tail_head = std::min(widest_tail_head, head);
      kv_case(p, reply, dense_logits, head, budget);
    }
    if (widest_tail_head < total) kv_case(p, reply, dense_logits, widest_tail_head, 0);
    return true;
  }

  // --sel-self: the prompt, then g_sel_self selected 1-row steps at `budget` pages - the steps the model measures
  // itself on - into the host's version of that measurement
  bool host_share(const Prompt &p, HostShare &cal, int threads, int budget) {
    const int steps = g_sel_self;
    m.use_slot(0);
    m.reset();
    select(budget);
    m.prefill(p.tokens.data(), static_cast<int>(p.tokens.size()) - (p.raw ? 1 : 0));
    int32_t token = p.raw ? p.tokens.back() : m.picks(0, 1)[0];
    std::map<std::pair<int, int>, std::vector<std::pair<int64_t, std::vector<float>>>> qs;  // (layer, card) -> per step (position, q [hq][256])
    m.on_attn_q([&](int layer, int card, const sycl::ext::oneapi::bfloat16 *q, int hq, int64_t pos) {
      std::vector<float> host(static_cast<size_t>(hq) * 256);
      for (size_t i = 0; i < host.size(); ++i) host[i] = static_cast<float>(q[i]);
      qs[{layer, card}].emplace_back(pos, std::move(host));
    });
    for (int i = 0; i < steps; ++i) token = step(token);
    m.on_attn_q({});
    const int pages = (static_cast<int>(m.position()) + 63) / 64;
    std::vector<int8_t> codes;
    std::vector<float> scales, keys;
    const auto t0 = clk::now();
    for (const auto &[at, recs] : qs) {
      const int hkv = m.read_keys(0, at.first, at.second, pages, codes, scales);
      for (int kvh = 0; kvh < hkv; ++kvh) {
        keys.resize(static_cast<size_t>(pages) * 64 * 256);
        for (int pg = 0; pg < pages; ++pg)
          for (int t = 0; t < 64; ++t) {
            const size_t src = ((static_cast<size_t>(pg) * hkv + kvh) * 64 + t), dst = (static_cast<size_t>(pg) * 64 + t) * 256;
            for (int d = 0; d < 256; ++d) keys[dst + d] = static_cast<float>(codes[src * 256 + d]) * scales[src];
          }
        std::vector<std::pair<int64_t, std::vector<float>>> head_q;
        for (const auto &r : recs) head_q.emplace_back(r.first, std::vector<float>(r.second.begin() + kvh * 6 * 256, r.second.begin() + (kvh + 1) * 6 * 256));
        cal.add({at.first, at.second * hkv + kvh}, keys, head_q, threads);
      }
    }
    std::printf("%zu-token prompt (%s): %d selected steps, %d pages; the page scores of %zu (layer, kv head) units on the host in %.0f s\n", p.tokens.size(), p.label.c_str(), steps,
                pages, cal.lost.size(), std::chrono::duration<double>(clk::now() - t0).count());
    return true;
  }

  // --sel-self K: the weights the model measured for the sequence on its first K selected steps against the host's
  // from the same steps' q and the stored keys. The device scores pages from bf16 bounds with bf16 operands, the host
  // from the dequantised keys in fp32, so single units may land one allocation step apart.
  bool self_check(const Prompt &p, int budget, int threads) {
    HostShare host;
    if (!host_share(p, host, threads, budget)) return false;
    const std::vector<float> want = host.weights(), &got = m.select_self_weights(0);
    if (got.size() != want.size()) {
      check(false, "the model measured its weights after " + std::to_string(g_sel_self) + " selected steps (" + std::to_string(got.size()) + " of " + std::to_string(want.size()) + " units)");
      return true;
    }
    double sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0, worst = 0;
    for (size_t u = 0; u < want.size(); ++u) {
      const double x = want[u], y = got[u];
      sx += x, sy += y, sxx += x * x, syy += y * y, sxy += x * y, worst = std::max(worst, std::abs(x - y));
    }
    const double n = static_cast<double>(want.size()), corr = (n * sxy - sx * sy) / std::sqrt((n * sxx - sx * sx) * (n * syy - sy * sy));
    char text[200];
    std::snprintf(text, sizeof text, "the model's own weights at %d pages (%.2f .. %.2f, mean %.3f) against the host's from the same %d steps: correlation %.4f, largest difference %.3f",
                  budget, *std::min_element(got.begin(), got.end()), *std::max_element(got.begin(), got.end()), sy / n, g_sel_self, corr, worst);
    check(corr >= 0.98 && std::abs(sy / n - 1.0) < 1e-3, text);
    return true;
  }

  // --sel-dump DIR the prompt prefilled densely, then T dense greedy 1-row steps whose q of
  // every attention layer and card go to DIR/r<index>-c<card>-q.bin (records: int32 step, layer, position, slot, q
  // heads, kv heads; then the q heads' [256] bf16), and at the end the slot's keys to DIR/r<index>-c<card>-L<layer>-kv.bin
  // (int32 pages, kv heads, tokens, layer; the int8 codes [pages][kv heads][64][256]; their scales [pages][kv heads][64]).
  // What the host scripts read: the true attention of each step (the page-count and page-summary studies).
  bool dump(const Prompt &p, int index, const std::string &dir) {
    m.use_slot(0);
    m.reset();
    select(0);
    m.prefill(p.tokens.data(), static_cast<int>(p.tokens.size()) - (p.raw ? 1 : 0));
    int32_t token = p.raw ? p.tokens.back() : m.picks(0, 1)[0];
    const int cards = m.cards();
    std::vector<std::FILE *> files(static_cast<size_t>(cards), nullptr);
    for (int c = 0; c < cards; ++c) {
      const std::string path = dir + "/r" + std::to_string(index) + "-c" + std::to_string(c) + "-q.bin";
      if (!(files[static_cast<size_t>(c)] = std::fopen(path.c_str(), "wb"))) {
        std::fprintf(stderr, "select35: cannot write %s\n", path.c_str());
        return false;
      }
    }
    int step_index = 0, kv_heads = 0;
    m.on_attn_q([&](int layer, int card, const sycl::ext::oneapi::bfloat16 *q, int hq, int64_t pos) {
      const int32_t head[6] = {step_index, layer, static_cast<int32_t>(pos), 0, hq, kv_heads};
      std::FILE *f = files[static_cast<size_t>(card)];
      std::fwrite(head, sizeof(int32_t), 6, f);
      std::fwrite(q, sizeof(*q), static_cast<size_t>(hq) * 256, f);
    });
    {
      std::vector<int8_t> k;
      std::vector<float> ks;
      for (int l = 0; l < m.config().layers && kv_heads == 0; ++l)
        if (m.config().full_attention[static_cast<size_t>(l)]) kv_heads = m.read_keys(0, l, 0, 1, k, ks);
    }
    Tokens fed;
    for (; step_index < steps; ++step_index) {
      fed.push_back(token);
      token = step(token);
    }
    m.on_attn_q({});
    for (std::FILE *f : files) std::fclose(f);
    const int tokens = static_cast<int>(m.position()), pages = (tokens + 63) / 64;
    size_t bytes = 0;
    for (int c = 0; c < cards; ++c) {
      for (int l = 0; l < m.config().layers; ++l) {
        if (!m.config().full_attention[static_cast<size_t>(l)]) continue;
        std::vector<int8_t> k;
        std::vector<float> ks;
        const int hkv = m.read_keys(0, l, c, pages, k, ks);
        char name[64];
        std::snprintf(name, sizeof name, "/r%d-c%d-L%02d-kv.bin", index, c, l);
        std::FILE *f = std::fopen((dir + name).c_str(), "wb");
        if (!f) return false;
        const int32_t head[4] = {pages, hkv, tokens, l};
        std::fwrite(head, sizeof(int32_t), 4, f);
        std::fwrite(k.data(), 1, k.size(), f);
        std::fwrite(ks.data(), sizeof(float), ks.size(), f);
        std::fclose(f);
        bytes += k.size() + ks.size() * sizeof(float);
      }
    }
    std::printf("\n%zu-token prompt (%s): run %d dumped to %s: %d dense steps from position %zu, %d pages of keys per attention layer and card (%.2f GiB); the steps' text: \"%s\"\n",
                p.tokens.size(), p.label.c_str(), index, dir.c_str(), steps, p.tokens.size() - (p.raw ? 1 : 0), pages, static_cast<double>(bytes) / (1ull << 30),
                tok.decode(fed, true).c_str());
    return true;
  }

  bool run(const Prompt &p, const std::vector<int> &budgets) {
    for (int s = 0; s < 2; ++s) {
      m.use_slot(s);
      m.reset();
    }
    m.use_slot(0);
    select(0);
    const auto t0 = clk::now();
    m.prefill(p.tokens.data(), static_cast<int>(p.tokens.size()) - (p.raw ? 1 : 0));
    const double prefill_s = std::chrono::duration<double>(clk::now() - t0).count();
    const int32_t first = p.raw ? p.tokens.back() : m.picks(0, 1)[0];
    const int snap = m.snap_save(0);
    m.snap_sync();
    const int pages = static_cast<int>((p.tokens.size() + 63) / 64);
    std::printf("\n%zu-token prompt (%s, %d pages) prefilled in %.1f s; snapshot %d\n", p.tokens.size(), p.label.c_str(), pages, prefill_s, snap);
    if (!m.kv_reserve(1, static_cast<int64_t>(p.tokens.size()) + 64)) {
      std::fprintf(stderr, "select35: no pages for slot 1\n");
      return false;
    }
    const std::string pass(kPassphrase);
    for (const int budget : budgets) {
      const Kl k = teacher_forced(snap, first, budget);
      const Tokens sel = free_run(snap, first, budget);
      const std::string dense_text = tok.decode(k.dense, true), sel_text = tok.decode(sel, true);
      std::printf("  %d pages (%.1f %% of the context): KL(dense || selected) mean %.5f, max %.5f over %d teacher-forced steps; top-1 %d / %d; free run %s dense\n",
                  budget, 100.0 * budget / pages, k.mean, k.max, steps, k.agree, steps, sel == k.dense ? "==" : "!=");
      if (p.raw) std::printf("    the first step (the text's last token): KL %.5f\n", k.first);
      std::printf("    dense:    \"%s\"\n    selected: \"%s\"\n", dense_text.c_str(), sel_text.c_str());
      if (p.needle) {
        const bool dense_hit = dense_text.find(pass) != std::string::npos, sel_hit = sel_text.find(pass) != std::string::npos;
        check(!dense_hit || sel_hit, "P " + std::to_string(p.tokens.size()) + ", " + std::to_string(budget) + " pages: the needle (" + pass + ") " +
                                         (dense_hit ? "answered by the dense run and by the selected run" : "not answered by the dense run (no gate)"));
      }
      for (const int rows : kTimedRows) speed(budget, rows);
      audit_bounds(1, static_cast<int>(p.tokens.size() / 64), "the pages slot 1's steps wrote and left");
    }
    m.snap_free(snap);
    return true;
  }
};

}  // namespace

int cmd_select35(const std::string &model, const std::string &ppl, const std::string &lens_arg, const std::string &docs_arg,
                 const std::string &budgets_arg, int generate, int reps, int threads, int pf_chunk) {
  xenithium::text::Tokenizer tok;
  std::string terr;
  if (!tok.load(model, terr)) {
    std::fprintf(stderr, "select35: tokenizer: %s\n", terr.c_str());
    return 2;
  }
  Tokens text;
  if (!ppl.empty() && !read_chunk0(ppl, text)) {
    std::fprintf(stderr, "select35: --ppl F (a token file whose chunk 0 holds the text)\n");
    return 2;
  }
  std::vector<Prompt> prompts;
  if (!text.empty())
    for (const int length : ints(lens_arg.empty() ? "131072,261120" : lens_arg)) {
      if (!g_sel_raw) {
        prompts.push_back(needle_prompt(tok, text, length));
      } else if (length >= 2 && length <= static_cast<int>(text.size())) {
        prompts.push_back(Prompt{"raw text", Tokens(text.begin(), text.begin() + length), false, true});
      } else {
        std::fprintf(stderr, "select35: --sel-raw: --pf-lens %d is not within the file's chunk 0 (%zu tokens)\n", length, text.size());
        return 2;
      }
    }
  for (const auto &path : split(docs_arg)) {
    std::string body;
    if (!read_text(path, body)) {
      std::fprintf(stderr, "select35: --doc %s: not readable or empty\n", path.c_str());
      return 2;
    }
    prompts.push_back(doc_prompt(tok, path, body));
  }
  if (prompts.empty()) {
    std::fprintf(stderr, "select35: --ppl F (needle prompts) and / or --doc F,F (documents with their question)\n");
    return 2;
  }
  if (text.empty()) text = prompts[0].tokens;  // the timed steps' filler
  const std::vector<int> budgets = ints(budgets_arg.empty() ? "256,512,1024" : budgets_arg);
  const int T = generate > 0 ? generate : 32;
  const int R = reps > 0 ? reps : 3;
  size_t longest = 0;
  for (const auto &p : prompts) longest = std::max(longest, p.tokens.size());
  q35::ModelOptions mo;
  // the prompt, the free / KL steps, the timed steps (two row counts, dense + selected, R rounds + the profiled step)
  mo.context = (static_cast<int64_t>(longest) + T + 4 * kTimedSteps * R * 2 + 64 + 63) / 64 * 64;  // room for the timed steps of every shape
  mo.threads = threads;
  mo.slots = 2;
  mo.kv_pool_tokens = 2 * mo.context;
  mo.snap_gib = 1;
  mo.attn_select = *std::max_element(budgets.begin(), budgets.end());
  mo.select_self = g_sel_self;  // --sel-self K
  const bool int8_fast = g_dec_attn == "int8-fast";
  if (g_sel_pf) {  // the page-selected prefill, in the served prefill numerics when asked for
    mo.prefill_select = mo.attn_select;
    mo.prefill_w4_a8 = g_pf_a8 != 0;
    if (pf_chunk > 0) mo.prefill_chunk = pf_chunk;
    if (int8_fast) mo.prefill_attn_qk = 1, mo.prefill_attn_p_lo = false;
  }
  const auto loader = xenithium::l0::Loader::open();
  xenithium::Fabric fab(loader, {0, 1}, g_queue_prof);
  mo.loader = &loader;
  q35::Model m(fab, model, mo);
  std::printf("select35: %s loaded in %.1f s, context %lld per slot, 2 slots, int8 KV, page selection up to %d pages; queues %s\n", model.c_str(),
              m.load_seconds(), static_cast<long long>(m.capacity()), mo.attn_select,
              g_queue_prof ? "with event profiling (walls host-bound)" : "without event profiling (walls as served)");
  if (int8_fast) {
    q35::StepOptions o = m.step_options();
    o.attn_qk = 1, o.attn_p_lo = false;
    m.set_step_options(o);
  } else if (g_dec_attn != "exact") {
    std::fprintf(stderr, "select35: --dec-attn takes exact or int8-fast (the numerics page selection runs)\n");
    return 2;
  }
  std::printf("select35: decode attention numerics %s, dense and selected\n", g_dec_attn.c_str());
  Probe probe{m, tok, text, T, R, g_queue_prof};
  for (const auto &p : prompts) {
    if (!g_sel_dump.empty()) {
      if (!probe.dump(p, static_cast<int>(&p - prompts.data()), g_sel_dump)) return 2;
      continue;
    }
    if (g_sel_self > 0 && !g_sel_pf && !g_sel_kv) {  // the largest count the prompt's context passes
      int budget = 0;
      for (const int b : budgets)
        if (b > budget && static_cast<int64_t>(b) * 64 < static_cast<int64_t>(p.tokens.size())) budget = b;
      if (budget > 128 && !probe.self_check(p, budget, std::max(1, std::min(16, threads)))) return 2;
    }
    if (!(g_sel_pf ? probe.run_pf(p, budgets) : g_sel_kv ? probe.run_kv(p, budgets) : probe.run(p, budgets))) return 2;
  }
  probe.check(m.wire_errors() == 0, "no mailbox timeout");
  std::printf("select35: %s\n", probe.failures ? "FAIL" : "PASS");
  return probe.failures ? 1 : 0;
}
