// xenithium-probe prefix35: the prefix cache on the dense Qwen3.8-27B, one load for every part.
// xenithium-probe prefix35 --model DIR --ppl F [--prompt-len P] [--generate G] [--context C] [--dflash DIR] [--seqs TURNS]
// Text: chunk 0 of the token file F (>= P + 12K tokens). Its first P tokens are the "prompt"; prefill calls run as the
// serving engine sizes them alone (calls end at multiples of prefill_chunk; a cut ends a call early). MTP drafter on
// (or DFlash with --dflash), int8 KV, a pool shared by 3 slots, pages handed out shuffled. Diagnostics: XEN_P35_NODRAFT=1
// (no drafter), XEN_P35_SESSION_ONLY=1 (parts 0 and 2).
// 0 the call boundary: a 5,124-token prompt's last k rows in a call of their own vs not: which cuts keep the bits (the
// engine cuts only at multiples of 64 with 64+ rows on both sides: BatchEngine:pc_cut).
// 1 model level. A cold prefill of the prompt (slot 0) gives the reference last-row logits, the first drafts and G greedy
// tokens. Then, each against that reference or a cold run with the same call boundaries, bit for bit:
// restore, aligned: slot 1 prefills [0, R) (R a multiple of the chunk), snap_save, prefills 3,000 other tokens over
// it, snap_restore, prefills [R, P): logits, drafts and tokens == the cold run;
// fork: slot 2 gets slot 1's pages below R and the snapshot (snap_fork), prefills [R, P): the same;
// restore, unaligned: R' inside a chunk: == a cold run whose calls end at R' too; vs the reference (other call
// boundaries: other many-row numerics) KL, top-1 and the greedy tokens are reported;
// live: slot 0 after its G tokens continues with 1,000 new tokens from its live state, vs a cold prefill
// of the whole sequence (decode rows vs prefill rows: reported, KL).
// And the timings: snap_save / snap_restore (both cards waited), snap_fork at R = 8K / 16K / 32K positions.
// 2 engine level. core:BatchEngine over the adapter, an agent-like session of TURNS turns: turn 1 = the prompt + a
// 4-token "generation prompt" (its start = the request's prefix mark, as the server sets it); turn k + 1 = turn k, its
// G-token reply, 1,500 tokens of new text and the generation prompt again. Run with the prefix cache off and on (the
// same load): TTFT per turn, prompt tokens served from the cache, and the replies (snapshots at multiples of 64 keep
// a cold prefill's numerics: the same replies).
#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include "xenithium/arch/qwen35/model.hpp"
#include "xenithium/kernels/attention/gqa.hpp"
#include "xenithium/arch/qwen35/step_model.hpp"
#include "xenithium/core/batch_engine.hpp"
#include "xenithium/rt/fabric.hpp"
#include "xenithium/util/env.hpp"

#include "probe.hpp"  // the run-wide flags

namespace {
namespace q35 = xenithium::arch::qwen35;
namespace core = xenithium::core;
using clk = std::chrono::steady_clock;

bool read_chunk0(const std::string &path, std::vector<int32_t> &out) {
  std::ifstream f(path, std::ios::binary);
  int32_t n = 0, len = 0;
  if (!f || !f.read(reinterpret_cast<char *>(&n), 4) || n <= 0 || !f.read(reinterpret_cast<char *>(&len), 4) || len <= 1) return false;
  out.resize(static_cast<size_t>(len));
  return static_cast<bool>(f.read(reinterpret_cast<char *>(out.data()), static_cast<std::streamsize>(len) * 4));
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
size_t argmax(const std::vector<float> &v) { return static_cast<size_t>(std::max_element(v.begin(), v.end()) - v.begin()); }

struct Sink final : core::EventSink {
  std::mutex mu;
  std::condition_variable cv;
  std::map<uint64_t, std::vector<int32_t>> toks;
  std::map<uint64_t, clk::time_point> first;
  std::map<uint64_t, bool> fin;
  void on_events(const core::Event *e, size_t n) override {
    std::lock_guard<std::mutex> lk(mu);
    const auto now = clk::now();
    for (size_t i = 0; i < n; ++i) {
      auto &t = toks[e[i].id];
      if (e[i].n && !first.count(e[i].id)) first[e[i].id] = now;
      t.insert(t.end(), e[i].tokens, e[i].tokens + e[i].n);
      if (e[i].finish != core::Finish::none) fin[e[i].id] = true;
    }
    cv.notify_all();
  }
  bool wait(uint64_t id, double s) {
    std::unique_lock<std::mutex> lk(mu);
    return cv.wait_for(lk, std::chrono::duration<double>(s), [&] { return fin.count(id) > 0; });
  }
};

}  // namespace

int cmd_prefix35(const std::string &model, const std::string &ppl, int64_t prompt_len, int generate, int64_t context, int threads, int turns) {
  int failures = 0;
  auto check = [&](bool ok, const std::string &what) {
    std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
    failures += !ok;
  };
  std::vector<int32_t> text;
  if (ppl.empty() || !read_chunk0(ppl, text)) {
    std::fprintf(stderr, "prefix35: --ppl F (a token file whose chunk 0 holds the text)\n");
    return 2;
  }
  const int64_t P = prompt_len > 0 ? prompt_len : 16384;
  const int G = generate > 0 ? generate : 32;
  // the bit gates below need a row's bits independent of the call's size: every fa context split breaks that 's
  // last-wave tail the short-call split), so this probe runs
  // them with no splits (XEN_ATTN_TAIL 0) unless XEN_P35_TAIL says otherwise (1 / 2: the gates then report what they move)
  {
    const char *e = std::getenv("XEN_P35_TAIL");
    xenithium::kernels::gqa_attn_fa_set_tail(e ? std::atoi(e) : 0);
    std::printf("prefix35: prefill attention context splits mode %d (XEN_P35_TAIL; the server's default is 2)\n", xenithium::kernels::gqa_attn_fa_tail());
  }
  if (static_cast<int64_t>(text.size()) < P + 12000) {
    std::fprintf(stderr, "prefix35: the text has %zu tokens, needs --prompt-len + 12,000\n", text.size());
    return 2;
  }
  q35::ModelOptions mo;
  const int64_t need = P + 6000 + static_cast<int64_t>(turns) * 2200;
  mo.context = std::max<int64_t>(context > 0 ? context : 0, (need + 63) / 64 * 64);
  mo.threads = threads;
  mo.slots = 3;
  mo.kv_pool_tokens = 3 * mo.context;
  mo.kv_pool_scramble = true;
  mo.snap_gib = 4;
  if (!g_dflash.empty()) {
    mo.dflash_dir = g_dflash;
    mo.dflash_dense_bytes = g_dflash_bytes;
  } else {
    mo.mtp = !xenithium::util::env_flag("XEN_P35_NODRAFT");  // diagnostics: no drafter
  }
  const bool session_only = xenithium::util::env_flag("XEN_P35_SESSION_ONLY");  // diagnostics: part 2 only
  mo.mtp_vocab = 98304;
  const auto loader = xenithium::l0::Loader::open();
  if (g_tp != 1 && g_tp != 2) throw std::runtime_error("prefix35: --tp 1 or 2");
  xenithium::Fabric fab(loader, tp_devices("prefix35", false));  // --tp 1: device 0 (ZE_AFFINITY_MASK)
  mo.loader = &loader;
  q35::Model m(fab, model, mo);
  const int C = static_cast<int>(fab.size());
  auto snap_all = [&] {  // one snapshot over every card
    size_t b = 0;
    for (int c = 0; c < C; ++c) b += m.snap_bytes(c);
    return b;
  };
  q35::StepModelAdapter ad(m);
  const int chunk = m.prefill_chunk();
  const int depth = m.dflash() ? m.draft_depth_max() : mo.mtp ? 3 : 0;
  std::printf("prefix35: %s, TP%d, context %lld, %d slots, pool %lld tokens (shuffled pages), chunk %d; drafter %s; snapshot %.1f MiB (%.1f on card 0), store %d\n",
              model.c_str(), C, static_cast<long long>(m.capacity()), m.slots(), static_cast<long long>(m.kv_pool_tokens()), chunk, m.dflash() ? "DFlash" : mo.mtp ? "MTP" : "none",
              snap_all() / 1048576.0, m.snap_bytes(0) / 1048576.0, m.snap_capacity());
  auto wait_all = [&] {
    for (size_t c = 0; c < fab.size(); ++c) fab.compute(c).wait();
  };
  auto ms_since = [](clk::time_point t0) { return std::chrono::duration<double, std::milli>(clk::now() - t0).count(); };
  if (g_snap_prepin) {  // The server's idle work pins every id before the first save
    const auto tp = clk::now();
    int ids = 0;
    for (int n; (n = ad.idle_work(64)) > 0;) ids += n;
    std::printf("prefix35: %d snapshot ids pinned ahead of use (the adapter's idle work) in %.1f ms\n", ids, ms_since(tp));
  }
  // prefill calls over [from, to) of t as the engine sizes them: chunk rows from `from, a call also ends at `cut`; each
  // call is given the token after it when t has one (as the engine does inside a prompt)
  auto calls = [&](int slot, const std::vector<int32_t> &t, int64_t from, int64_t to, int64_t cut = -1, int64_t arm = -1) {
    for (int64_t p = from; p < to;) {
      int64_t e = std::min(to, (p / chunk + 1) * chunk);
      if (cut > p && cut < e) e = cut;
      if (arm > p && arm < e) m.snap_arm(slot, arm);  // the call spanning `arm` captures a snapshot there
      ad.prefill(slot, t.data() + p, static_cast<int>(e - p), e < static_cast<int64_t>(t.size()) ? t[static_cast<size_t>(e)] : -1);  // the engine: the next token when the prompt has one
      p = e;
    }
  };
  struct Out {
    std::vector<float> logits;
    std::vector<int32_t> drafts, toks;
  };
  auto finish = [&](int slot) {  // after a prompt's last call: the last row's logits, the first drafts, G greedy tokens
    Out o;
    ad.prefill_logits(o.logits);
    int32_t t = ad.prefill_pick();
    if (depth > 0) o.drafts = ad.first_drafts(slot, t, depth, 1);
    o.toks.push_back(t);
    for (int g = 1; g < G; ++g) {
      ad.step({slot}, &t);
      ad.picks(1, &t);
      o.toks.push_back(t);
    }
    return o;
  };
  std::vector<int32_t> prompt(text.begin(), text.begin() + P);
  std::vector<int32_t> junk(text.begin() + P + 4000, text.begin() + P + 7000);  // other text written over a snapshot's tail
  auto same = [](const Out &a, const Out &b) { return a.logits == b.logits && a.drafts == b.drafts && a.toks == b.toks; };
  auto agree = [](const std::vector<int32_t> &a, const std::vector<int32_t> &b) {
    size_t n = 0;
    while (n < a.size() && n < b.size() && a[n] == b[n]) ++n;
    return n;
  };

  // ---- 0. the call boundary: a prompt's last k rows in a call of their own vs inside the previous call. The rows'
  // numerics depend on the call's row count below some size (other kernel paths), so a cut is only free above it
  {
    const int64_t Pt = 4096 + 1024 + 4;  // a 1,028-row last call; cuts at multiples of 64 and not
    std::vector<int32_t> pt(text.begin(), text.begin() + Pt);
    ad.begin(0);
    calls(0, pt, 0, Pt);
    std::vector<float> l0;
    ad.prefill_logits(l0);
    std::string line;
    for (int k : {4, 36, 68, 132, 260, 516, 772, 964, 1028, 5, 100}) {
      ad.begin(0);
      calls(0, pt, 0, Pt, Pt - k);
      std::vector<float> l1;
      ad.prefill_logits(l1);
      const bool eq = l0 == l1;
      char b[96];
      std::snprintf(b, sizeof b, " %d: %s", k, eq ? "same" : "differs");
      line += b;
      // the engine's rule (BatchEngine:pc_cut): a cut at a multiple of 64 with 64+ rows on both sides keeps the bits
      const int64_t cut = Pt - k, head = cut - 4096;
      if (cut % 64 == 0 && k >= 64 && head >= 64) check(eq, "a cut at " + std::to_string(cut) + " (64+ rows each side, a multiple of 64) keeps the bits");
    }
    std::printf("call boundary: the last k rows of a %lld-token prompt (calls at 0, 2,048, 4,096) in a call of their own, last-row logits vs without the cut:%s\n",
                static_cast<long long>(Pt), line.c_str());
  }
  // ---- 1. model level ----
  if (!session_only) {
  ad.begin(0);
  auto t0 = clk::now();
  calls(0, prompt, 0, P);
  wait_all();
  const double cold_ms = ms_since(t0);
  const Out ref = finish(0);
  std::printf("cold: %lld-token prompt prefilled in %.0f ms (%.0f tok/s); greedy: %d tokens\n", static_cast<long long>(P), cold_ms, P / cold_ms * 1e3, G);

  const int64_t R = (P / 2) / chunk * chunk;  // aligned
  ad.begin(1);
  calls(1, prompt, 0, R);
  wait_all();
  t0 = clk::now();
  const int id1 = m.snap_save(1);
  wait_all();
  double save_ms = ms_since(t0);
  m.snap_sync();
  const double save_host_ms0 = ms_since(t0);  // the first save of an id allocates its pinned host buffers
  double save_ms0 = save_ms, save_host_ms = 0;
  {
    const int idx = m.snap_save(1);  // a second id (allocates), freed: the next save reuses its buffers
    m.snap_sync();
    m.snap_free(idx);
    wait_all();
    t0 = clk::now();
    const int idy = m.snap_save(1);
    wait_all();
    save_ms = ms_since(t0);
    m.snap_sync();
    save_host_ms = ms_since(t0);
    m.snap_free(idy);
  }
  check(id1 >= 0 && m.snap_pos(id1) == R, "snap_save at " + std::to_string(R) + ": id " + std::to_string(id1));
  calls(1, junk, 0, static_cast<int64_t>(junk.size()));  // the slot's state and pages past R overwritten
  m.kv_trim(1, R);
  t0 = clk::now();
  m.snap_restore(1, id1);
  wait_all();
  const double restore_ms = ms_since(t0);
  t0 = clk::now();
  calls(1, prompt, R, P);
  wait_all();
  const double tail_ms = ms_since(t0);
  const Out a1 = finish(1);
  check(same(a1, ref), "restore at " + std::to_string(R) + " (aligned) after 3,000 other tokens over it: logits, " + std::to_string(a1.drafts.size()) +
                           " first drafts and " + std::to_string(G) + " greedy tokens == the cold prefill, bit for bit");
  std::printf("  snap_save %.2f ms on the compute queue (%.2f ms with its host copy beside it; the id's first save %.2f / %.2f with the pinned allocation), snap_restore %.2f ms (%.1f MiB each way); the tail [%lld, %lld) prefilled in %.0f ms vs %.0f cold\n",
              save_ms, save_host_ms, save_ms0, save_host_ms0, restore_ms,
              snap_all() / 1048576.0, static_cast<long long>(R), static_cast<long long>(P), tail_ms, cold_ms);
  // fork: slot 1 still holds [0, R) under the snapshot
  ad.begin(2);
  m.kv_reserve(2, P + 64);
  t0 = clk::now();
  m.snap_fork(2, 1, id1);
  const double fork_ms = ms_since(t0);
  // The pages below R are shared with slot 1, not copied (ModelOptions:kv_share). Slot 1 lets go
  // of them and writes other tokens into fresh pages first (the shuffled pool hands out any free page): slot 2's must stay
  const int shared = m.kv_shared_pages(2);
  ad.begin(1);
  calls(1, junk, 0, std::min<int64_t>(R, static_cast<int64_t>(junk.size())));
  const uint64_t cow0 = m.kv_cow_count();
  calls(2, prompt, R, P);
  const Out a2 = finish(2);
  check(same(a2, ref), "fork of slot 1's snapshot at " + std::to_string(R) + " into slot 2 (" + (shared ? std::to_string(shared) + " KV pages shared" : std::string("KV pages copied")) +
                           "; slot 1 reset and rewritten before slot 2's tail): == the cold prefill, bit for bit");
  std::printf("  snap_fork %.2f ms (%lld positions of KV: target + drafter pages, the state; %d pages shared, %llu copy-on-write events in slot 2's tail)\n", fork_ms,
              static_cast<long long>(R), shared, static_cast<unsigned long long>(m.kv_cow_count() - cow0));

  // unaligned: R2 inside a chunk; the reference with the same call boundaries on slot 2, then the snapshot run on slot 1.
  // Two offsets into the 64-row page 5 too: a resume at pos % 64 < 32 reads the first rows of the ring
  // window's second page, which a misplaced scale copy once overwrote at TP2; at 37 only TP1 read them)
  for (const int64_t off : {int64_t{37}, int64_t{5}}) {
    const int64_t R2 = R + chunk / 2 + off;
    ad.begin(2);
    calls(2, prompt, 0, R2);
    calls(2, prompt, R2, P);
    const Out c2 = finish(2);
    ad.begin(1);
    calls(1, prompt, 0, R2);
    const int id2 = m.snap_save(1);
    calls(1, junk, 0, 2000);
    m.kv_trim(1, R2);
    m.snap_restore(1, id2);
    calls(1, prompt, R2, P);
    const Out a3 = finish(1);
    check(same(a3, c2), "restore at " + std::to_string(R2) + " (inside a chunk): == a cold run with the same call boundaries, bit for bit");
    {
      const double k = kl(ref.logits, a3.logits);
      std::printf("  vs the reference's boundaries: last-row KL %.3g, top-1 %s, greedy tokens equal for %zu of %d, first drafts %s\n", k,
                  argmax(ref.logits) == argmax(a3.logits) ? "same" : "differs", agree(ref.toks, a3.toks), G, ref.drafts == a3.drafts ? "same" : "differ");
      check(std::isfinite(k) && k < 0.01 && argmax(ref.logits) == argmax(a3.logits), "unaligned resume vs the reference: KL < 0.01 and the same top-1");
    }
    m.snap_free(id2);
  }
  // a resume point at a multiple of 64 inside a chunk: the many-row kernels' row arithmetic does not depend on the call
  // (the GDN chunked scan runs on absolute 64-row blocks), so this must equal the reference with its 2,048-row calls
  {
    const int64_t R3 = R + chunk / 2 + 64;
    ad.begin(1);
    calls(1, prompt, 0, R3);
    const int id3 = m.snap_save(1);
    calls(1, junk, 0, 1000);
    m.kv_trim(1, R3);
    m.snap_restore(1, id3);
    calls(1, prompt, R3, P);
    const Out a4 = finish(1);
    check(same(a4, ref), "restore at " + std::to_string(R3) + " (a multiple of 64 inside a chunk), the calls after it from there: == the reference (calls at "
                             "multiples of " + std::to_string(chunk) + "), bit for bit");
    m.snap_free(id3);
  }
  // the engine's order: the snapshot after a call that ends at R (next token known), the prompt's last call, decode
  // steps on the slot; then restore at R and a longer prompt from there. vs the cold prefill of that prompt
  {
    const int64_t Rm = P / 2 / 64 * 64 + 64;  // a mark-like cut inside a chunk
    std::vector<int32_t> p2(prompt.begin(), prompt.begin() + Rm + 4);
    std::vector<int32_t> p3(prompt.begin(), prompt.begin() + Rm);
    p3.insert(p3.end(), text.begin() + P + 9000, text.begin() + P + 10500);
    ad.begin(2);
    calls(2, p3, 0, static_cast<int64_t>(p3.size()), Rm);
    const Out c3 = finish(2);
    ad.begin(1);
    calls(1, p2, 0, Rm);  // the call ending at Rm is given p2[Rm]
    const int id4 = m.snap_save(1);
    calls(1, p2, Rm, static_cast<int64_t>(p2.size()));
    (void)finish(1);  // the first token and G - 1 decode steps on the slot
    m.kv_trim(1, Rm);
    m.snap_restore(1, id4);
    calls(1, p3, Rm, static_cast<int64_t>(p3.size()));
    const Out a5 = finish(1);
    std::printf("  restore at %lld after the slot's own last call and %d decode steps: last-row KL %.3g vs cold, greedy tokens equal for %zu of %d\n",
                static_cast<long long>(Rm), G - 1, kl(c3.logits, a5.logits), agree(c3.toks, a5.toks), G);
    check(same(a5, c3), "restore at " + std::to_string(Rm) + " after the slot decoded past it (the engine's order): == the cold prefill, bit for bit");
    m.snap_free(id4);
  }
  // (audit C4): a snapshot from inside a prefill call (snap_arm / snap_take) vs a call cut there
  {
    const int64_t Ri = R + chunk / 2 + 128;  // a multiple of 64 inside a chunk
    ad.begin(1);
    calls(1, prompt, 0, Ri);
    const int idc = m.snap_save(1);
    ad.begin(2);
    calls(2, prompt, 0, P, -1, Ri);
    const int idi = m.snap_take();
    m.snap_sync();
    check(idc >= 0 && idi >= 0 && m.snap_pos(idi) == Ri && m.snap_take() == -2, "inside-call snapshot at " + std::to_string(Ri) + ": taken once");
    bool eqb = idc >= 0 && idi >= 0;
    size_t diff = 0;
    for (int c = 0; c < C && eqb; ++c) {
      const auto *a = static_cast<const unsigned char *>(m.snap_host(idc, c)), *b = static_cast<const unsigned char *>(m.snap_host(idi, c));
      for (size_t i = 0; i < m.snap_bytes(c); ++i) diff += a[i] != b[i];
    }
    eqb = eqb && diff == 0;
    check(eqb, "inside-call snapshot == the snapshot after a call cut at " + std::to_string(Ri) + ", byte for byte (" +
                   std::to_string(snap_all() >> 20) + " MiB; bytes differing: " + std::to_string(diff) + ")");
    const Out a6 = finish(2);
    check(same(a6, ref), "the call that captured it: last-row logits, first drafts, greedy tokens == the cold prefill, bit for bit");
    m.kv_trim(2, Ri);
    m.snap_restore(2, idi);
    calls(2, prompt, Ri, P);
    const Out a7 = finish(2);
    check(same(a7, ref), "restore of the inside-call snapshot, the rest from there: == the cold prefill, bit for bit");
    m.snap_free(idc);
    m.snap_free(idi);
    // cost: the prompt cold / with the call cut at Ri (the cut) / with the capture inside, interleaved, median of 3
    std::vector<double> tc, tk, ti;
    for (int rep = 0; rep < 3; ++rep) {
      for (int v = 0; v < 3; ++v) {
        ad.begin(1);
        wait_all();
        t0 = clk::now();
        int id = -1;
        if (v == 1) {  // the engine's cut: a call ends at Ri, its snapshot, the rest
          calls(1, prompt, 0, Ri);
          id = m.snap_save(1);
          calls(1, prompt, Ri, P);
        } else {
          calls(1, prompt, 0, P, -1, v == 2 ? Ri : -1);
          if (v == 2) id = m.snap_take();
        }
        wait_all();
        (v == 0 ? tc : v == 1 ? tk : ti).push_back(ms_since(t0));
        m.snap_sync();
        if (id >= 0) m.snap_free(id);
      }
    }
    auto med = [](std::vector<double> v) {
      std::sort(v.begin(), v.end());
      return v[v.size() / 2];
    };
    std::printf("  inside-call snapshot: the %lld-token prompt cold %.1f ms, cut at %lld + snapshot %.1f ms (+%.1f), captured inside %.1f ms (+%.1f)\n",
                static_cast<long long>(P), med(tc), static_cast<long long>(Ri), med(tk), med(tk) - med(tc), med(ti), med(ti) - med(tc));
  }
  // live: slot 0 continues after its G tokens with 1,000 new tokens; vs a cold prefill of the whole sequence
  {
    std::vector<int32_t> more(text.begin() + P + 8000, text.begin() + P + 9000);
    std::vector<int32_t> cont{ref.toks.back()};
    cont.insert(cont.end(), more.begin(), more.end());
    if (m.dflash()) {  // the DFlash rings hold the decode rows' features in another order: no live resume (snapshots only)
      check(!m.live_ok(0), "DFlash: a live state is not offered for resume");
      goto after_live;
    }
    check(m.live_ok(0), "slot 0's live state after its greedy steps can take a prefill call");
    t0 = clk::now();
    calls(0, cont, 0, static_cast<int64_t>(cont.size()));
    wait_all();
    const double live_ms = ms_since(t0);
    std::vector<float> l0;
    ad.prefill_logits(l0);
    std::vector<int32_t> whole = prompt;
    whole.insert(whole.end(), ref.toks.begin(), ref.toks.end());
    whole.insert(whole.end(), more.begin(), more.end());
    ad.begin(1);
    t0 = clk::now();
    calls(1, whole, 0, static_cast<int64_t>(whole.size()));
    wait_all();
    const double whole_ms = ms_since(t0);
    std::vector<float> l1;
    ad.prefill_logits(l1);
    const double k = kl(l1, l0);
    std::printf("  live resume at %lld + 1,001 tokens: %.0f ms vs %.0f ms for the whole %zu-token sequence cold; last-row KL %.3g, top-1 %s\n",
                static_cast<long long>(P + G - 1), live_ms, whole_ms, whole.size(), k, argmax(l0) == argmax(l1) ? "same" : "differs");
    check(std::isfinite(k) && k < 0.01 && argmax(l0) == argmax(l1), "live resume vs the whole sequence cold: KL < 0.01 and the same top-1");
  }
after_live:
  // fork cost by length
  for (int64_t r : {int64_t{4096}, int64_t{8192}, int64_t{16384}, int64_t{24576}}) {
    if (r + 64 > m.capacity()) continue;
    ad.begin(1);
    calls(1, text, 0, r);
    const int id = m.snap_save(1);
    ad.begin(2);
    m.kv_reserve(2, r + 64);
    wait_all();
    t0 = clk::now();
    m.snap_fork(2, 1, id);
    std::printf("  snap_fork at %lld positions: %.2f ms\n", static_cast<long long>(r), ms_since(t0));
    m.snap_free(id);
  }
  m.snap_free(id1);

  }  // part 1
  // ---- 2. engine level: an agent-like session, cache off then on ----
  for (int s = 0; s < m.slots(); ++s) ad.begin(s);  // nothing of part 1 left on the slots
  if (turns > 0) {
    const std::vector<int32_t> gen{text[0], text[1], text[2], text[3]};  // the "generation prompt" ending every turn's prompt
    struct TurnRes {
      double ttft_ms = 0;
      size_t prompt = 0;
      std::vector<int32_t> out;
    };
    auto session = [&](bool cache, bool live, std::vector<TurnRes> &res, core::Stats &st, const std::vector<std::vector<int32_t>> *forced,
                       bool inside = true) {
      core::BatchConfig bc;
      bc.prefix_live = live;
      bc.prefix_inside = inside;  // (audit C4): false = the prefix mark's snapshot from a call cut there
      bc.eos_ids = {};
      bc.log_interval_s = 0;
      bc.prefix_cache = cache;
      bc.prefill_part_ms = 25;
      core::BatchEngine e(ad, bc);
      Sink sink;
      e.start(&sink);
      std::vector<int32_t> pr = prompt;
      pr.insert(pr.end(), gen.begin(), gen.end());
      for (int k = 0; k < turns; ++k) {
        core::SamplingParams p;
        p.temperature = 0;
        p.ignore_eos = true;
        p.max_tokens = G;
        p.prefix_mark = static_cast<int32_t>(pr.size() - gen.size());
        const uint64_t id = static_cast<uint64_t>(k + 1);
        const auto ts = clk::now();
        e.submit(id, pr, p);
        sink.wait(id, 600);
        TurnRes r;
        {
          std::lock_guard<std::mutex> lk(sink.mu);
          r.ttft_ms = std::chrono::duration<double, std::milli>(sink.first[id] - ts).count();
          r.out = sink.toks[id];
        }
        r.prompt = pr.size();
        res.push_back(r);
        // the next turn, as a chat template renders it: this prompt with its generation prompt, the reply (the cold run's,
        // so both runs see the same prompts), new text, the generation prompt again
        const auto &reply = forced ? (*forced)[static_cast<size_t>(k)] : r.out;
        pr.insert(pr.end(), reply.begin(), reply.end());
        const size_t at = static_cast<size_t>(P) + 1000 + static_cast<size_t>(k) * 1500;
        pr.insert(pr.end(), text.begin() + static_cast<std::ptrdiff_t>(at), text.begin() + static_cast<std::ptrdiff_t>(at + 1500));
        pr.insert(pr.end(), gen.begin(), gen.end());
      }
      e.stop();
      st = e.stats();
    };
    std::vector<TurnRes> off, on, lv;
    core::Stats st_off, st_on, st_lv;
    session(false, false, off, st_off, nullptr);
    std::vector<std::vector<int32_t>> replies;
    for (const auto &r : off) replies.push_back(r.out);
    session(true, false, on, st_on, &replies);
    session(true, true, lv, st_lv, &replies);
    std::vector<TurnRes> off2;  // the engine's own repeatability without the cache: the floor for "the same reply"
    core::Stats st_off2;
    session(false, false, off2, st_off2, &replies);
    // (audit C4): the prefix marks' snapshots from a call cut there vs inside the call, A / B / A'
    std::vector<TurnRes> cut, on2;
    core::Stats st_cut, st_on2;
    session(true, false, cut, st_cut, &replies, false);
    session(true, false, on2, st_on2, &replies, true);
    {
      std::string eq;
      for (size_t k = 0; k < off.size() && k < off2.size(); ++k) eq += " " + std::to_string(agree(off[k].out, off2[k].out));
      std::printf("  without the cache, a second run: reply tokens equal per turn:%s (of %d)\n", eq.c_str(), G);
    }
    std::printf("agent session, %d turns (greedy, %d tokens per reply; turn k + 1 = turn k's prompt (ends in a 4-token generation prompt, marked) + reply + 1,500 new tokens + the generation prompt)\n",
                turns, G);
    std::printf("  turn   prompt tokens   TTFT off ms   TTFT on ms   speed-up   reply tokens equal   on + live resumes: TTFT ms   reply tokens equal\n");
    double sum_off = 0, sum_on = 0, sum_lv = 0;
    bool first_same = true;
    for (int k = 0; k < turns; ++k) {
      const auto &a = off[static_cast<size_t>(k)], &b = on[static_cast<size_t>(k)], &c = lv[static_cast<size_t>(k)];
      std::printf("  %4d   %13zu   %11.0f   %10.0f   %7.1fx   %8zu of %zu   %26.0f   %8zu of %zu\n", k + 1, a.prompt, a.ttft_ms, b.ttft_ms, a.ttft_ms / b.ttft_ms,
                  agree(a.out, b.out), a.out.size(), c.ttft_ms, agree(a.out, c.out), a.out.size());
      if (k > 0) sum_off += a.ttft_ms, sum_on += b.ttft_ms, sum_lv += c.ttft_ms;
      first_same &= !a.out.empty() && !b.out.empty() && a.out[0] == b.out[0];
    }
    auto pcs = [](const core::Stats &st) {
      char b[200];
      std::snprintf(b, sizeof b, "%llu of %llu prompt tokens hit (%llu live, %llu restores, %llu forks)", static_cast<unsigned long long>(st.prefix_hits),
                    static_cast<unsigned long long>(st.prefix_queries), static_cast<unsigned long long>(st.prefix_live),
                    static_cast<unsigned long long>(st.prefix_restores), static_cast<unsigned long long>(st.prefix_forks));
      return std::string(b);
    };
    std::printf("  turns 2..%d: TTFT %.0f ms off, %.0f ms on (%s), %.0f ms on + live (%s)\n", turns, sum_off, sum_on, pcs(st_on).c_str(), sum_lv, pcs(st_lv).c_str());
    check(off.size() == on.size() && !on.empty() && on[0].out == off[0].out, "turn 1 (nothing cached): the same reply with the cache on");
    check(first_same, "every turn's first reply token is the same with the cache on");
    bool all_same = off.size() == on.size();
    for (size_t k = 0; all_same && k < off.size(); ++k) all_same = off[k].out == on[k].out;
    check(all_same, "every turn's whole reply is the same with the cache on (resumes at multiples of 64)");
    check(st_on.prefix_hits > 0 && st_off.prefix_hits == 0 && sum_on < sum_off, "turns 2.. continue a cached prefix and reach their first token sooner");
    {  // (audit C4)
      double s_cut = 0, s_on2 = 0, all_cut = 0, all_on = 0, all_on2 = 0;
      for (int k = 0; k < turns; ++k) {
        if (k > 0) s_cut += cut[static_cast<size_t>(k)].ttft_ms, s_on2 += on2[static_cast<size_t>(k)].ttft_ms;
        all_cut += cut[static_cast<size_t>(k)].ttft_ms, all_on += on[static_cast<size_t>(k)].ttft_ms, all_on2 += on2[static_cast<size_t>(k)].ttft_ms;
      }
      std::printf("  prefix marks' snapshots: inside the call %.0f / %.0f ms (turns 2..%d; all turns %.0f / %.0f), a call cut there (round 22) %.0f ms (all turns %.0f)\n"
                  "    inside: %s / %s; cut: %s\n",
                  sum_on, s_on2, turns, all_on, all_on2, s_cut, all_cut, pcs(st_on).c_str(), pcs(st_on2).c_str(), pcs(st_cut).c_str());
      bool same = cut.size() == on.size() && on2.size() == on.size();
      for (size_t k = 0; same && k < on.size(); ++k) same = cut[k].out == on[k].out && on2[k].out == on[k].out;
      check(same, "replies with the snapshots inside the call == with a call cut there, every turn");
    }
  }
  check(m.wire_errors() == 0, "no mailbox timeout");
  std::printf("prefix35: %s\n", failures ? "FAIL" : "PASS");
  return failures ? 1 : 0;
}
