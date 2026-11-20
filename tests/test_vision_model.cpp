// Xenithium · image input on the dense 27B (arch qwen35), model level. Needs both cards and the checkpoint
// (~/models/swift15-w4a16-autoround or argv[1]; skipped when absent).
// The interleaved M-RoPE has two paths in the model: a prefill call composes each row's rope row on the host (image rows
// mix three positions per frequency; text rows after an image sit at position + delta) and a decode step reads the 1-D
// table at position + delta from the segment record (kSegRope). Gate: a greedy reply decoded after a two-image prompt
// (chunks of 64 rows: the images straddle calls) against the same reply prefilled in one go on another slot - every row's
// pick equal and the logits as close as prefill-vs-decode is on a text-only prompt (the control); the delta after the
// prompt == the reference's (max(gh, gw) - tokens per image); a snapshot restored inside an image (by the prefix
// cache's API) gives the logits of the uninterrupted call; text rows without images use the rope by position (no rope
// rows).
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "xenithium/arch/common/vision.hpp"
#include "xenithium/arch/qwen35/model.hpp"
#include "xenithium/model/checkpoint.hpp"
#include "xenithium/model/config.hpp"
#include "xenithium/model/vision.hpp"
#include "xenithium/rt/fabric.hpp"
#include "xenithium/rt/l0.hpp"
#include "xenithium/text/tokenizer.hpp"
#include "xenithium/util/image.hpp"

namespace {

int g_checks = 0, g_failures = 0;
#define CHECK(cond, ...)                         \
  do {                                           \
    ++g_checks;                                  \
    if (!(cond)) {                               \
      ++g_failures;                              \
      std::printf("  [FAIL] " __VA_ARGS__);      \
      std::printf("\n");                         \
    }                                            \
  } while (0)

namespace m = xenithium::model;
namespace q35 = xenithium::arch::qwen35;

std::string read_all(const std::string &p) {
  std::ifstream f(p, std::ios::binary);
  std::stringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

int argmax(const float *row, int64_t n) { return static_cast<int>(std::max_element(row, row + n) - row); }

double kl(const float *p_logits, const float *q_logits, int64_t n) {  // KL(softmax p || softmax q)
  double mp = -1e300, mq = -1e300;
  for (int64_t i = 0; i < n; ++i) mp = std::max(mp, static_cast<double>(p_logits[i])), mq = std::max(mq, static_cast<double>(q_logits[i]));
  double zp = 0, zq = 0;
  for (int64_t i = 0; i < n; ++i) zp += std::exp(p_logits[i] - mp), zq += std::exp(q_logits[i] - mq);
  double k = 0;
  for (int64_t i = 0; i < n; ++i) {
    const double lp = p_logits[i] - mp - std::log(zp), lq = q_logits[i] - mq - std::log(zq);
    k += std::exp(lp) * (lp - lq);
  }
  return k;
}

struct Run {
  std::vector<int32_t> reply;  // greedy tokens after the prompt
  std::vector<std::vector<float>> lg;  // logits of the row predicting reply[i] (i >= 1: after reply[i - 1])
};

// greedy reply of `steps` tokens: the prompt's last row, then 1-row decode steps
Run decode(q35::Model &md, int slot, const std::vector<int32_t> &prompt, int steps) {
  Run r;
  md.use_slot(slot);
  md.reset();
  md.prefill(prompt.data(), static_cast<int>(prompt.size()));
  const int last = (static_cast<int>(prompt.size()) - 1) % 4 + 1;  // the head ran on the prompt's last slice
  std::vector<float> lg;
  md.logits(last, lg);
  const int64_t V = md.config().vocab;
  r.lg.emplace_back(lg.begin() + (last - 1) * V, lg.begin() + last * V);
  r.reply.push_back(argmax(r.lg.back().data(), V));
  for (int i = 1; i < steps; ++i) {
    const int32_t t = r.reply.back();
    md.step(&t, 1);
    md.accept(1);
    md.logits(1, lg);
    r.lg.emplace_back(lg.begin(), lg.begin() + V);
    r.reply.push_back(argmax(lg.data(), V));
  }
  return r;
}

// the same rows prefilled: prompt + reply[0.. steps - 2], the logits of rows prompt-1.. end
std::vector<std::vector<float>> prefill_rows(q35::Model &md, int slot, const std::vector<int32_t> &prompt, const std::vector<int32_t> &reply) {
  md.use_slot(slot);
  md.reset();
  std::vector<int32_t> ids = prompt;
  ids.insert(ids.end(), reply.begin(), reply.end() - 1);
  const int64_t V = md.config().vocab, first = static_cast<int64_t>(prompt.size()) - 1;
  std::vector<std::vector<float>> out(reply.size());
  std::vector<float> lg;
  md.prefill(ids.data(), static_cast<int>(ids.size()), [&](int64_t p0, int rows) {
    if (p0 + rows <= first) return;
    md.logits(rows, lg);
    for (int r = 0; r < rows; ++r)
      if (p0 + r >= first) out[static_cast<size_t>(p0 + r - first)].assign(lg.begin() + r * V, lg.begin() + (r + 1) * V);
  });
  return out;
}

struct Cmp {
  int picks_off = 0;
  double kl_max = 0, kl_mean = 0, dl_max = 0;  // dl: the largest |logit difference| over the rows' top-64 (decode's)
};
Cmp compare(const Run &d, const std::vector<std::vector<float>> &pf, int64_t V) {
  Cmp c;
  for (size_t i = 0; i < d.lg.size(); ++i) {
    c.picks_off += argmax(pf[i].data(), V) != d.reply[i];
    const double k = kl(d.lg[i].data(), pf[i].data(), V);
    c.kl_max = std::max(c.kl_max, k);
    c.kl_mean += k / static_cast<double>(d.lg.size());
    std::vector<int> idx(static_cast<size_t>(V));
    for (int64_t j = 0; j < V; ++j) idx[static_cast<size_t>(j)] = static_cast<int>(j);
    std::partial_sort(idx.begin(), idx.begin() + 64, idx.end(), [&](int a, int b) { return d.lg[i][a] > d.lg[i][b]; });
    for (int j = 0; j < 64; ++j) c.dl_max = std::max(c.dl_max, static_cast<double>(std::fabs(d.lg[i][idx[j]] - pf[i][idx[j]])));
  }
  return c;
}

}  // namespace

int main(int argc, char **argv) {
  const char *home = std::getenv("HOME");
  const std::string dir = argc > 1 ? argv[1] : std::string(home ? home : "") + "/models/swift15-w4a16-autoround";
  if (!std::filesystem::exists(dir + "/config.json")) {
    std::printf("[SKIP] vision-model: %s not present\n", dir.c_str());
    return 0;
  }
  const auto cfg = m::ModelConfig::load(dir);
  const m::VisionConfig vc = m::VisionConfig::parse(cfg);
  if (!vc.present) {
    std::printf("[SKIP] vision-model: %s has no vision tower\n", dir.c_str());
    return 0;
  }
  xenithium::text::Tokenizer tok;
  std::string err;
  if (!tok.load(dir, err)) {
    std::printf("[FAIL] tokenizer: %s\n", err.c_str());
    return 1;
  }
  const xenithium::l0::Loader loader = xenithium::l0::Loader::open();
  xenithium::Fabric fab(loader, std::vector<size_t>{0, 1}, false);
  q35::ModelOptions mo;
  mo.context = 4096;
  mo.slots = 2;
  mo.prefill_chunk = 64;  // the images straddle calls
  mo.snap_gib = 2;
  mo.loader = &loader;
  q35::Model md(fab, dir, mo);
  const auto ck = m::Checkpoint::open(dir);
  auto tower = std::make_unique<xenithium::arch::VisionTower>(fab, ck, vc, 4096);
  xenithium::arch::VisionStore store(fab, std::move(tower), vc, md.config().vocab, 4096);
  const int64_t V = md.config().vocab;

  // control: a text-only prompt, vision off (no rope rows), then on (the same bits: delta 0 = the rope by position)
  const std::vector<int32_t> text_prompt = tok.encode(
      "<|im_start|>user\nDescribe a red circle beside a blue square, then a planet seen from orbit.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
  const int steps = 24;
  const Run t_off = decode(md, 0, text_prompt, steps);
  md.set_vision(&store);
  const Run t_on = decode(md, 0, text_prompt, steps);
  bool same = t_on.reply == t_off.reply;
  for (size_t i = 0; i < t_on.lg.size() && same; ++i) same = t_on.lg[i] == t_off.lg[i];
  CHECK(same, "text-only: vision on changed the bits of a text prompt");
  std::printf("  text-only prompt: vision on == off bit for bit: %s\n", same ? "yes" : "NO");
  const Cmp ct = compare(t_on, prefill_rows(md, 1, text_prompt, t_on.reply), V);

  // two images: c_base420.jpg (14 x 22 patches -> 7 x 11 tokens), c_rgb.png
  std::vector<int32_t> prompt = tok.encode("<|im_start|>user\n<|vision_start|>");
  std::vector<std::shared_ptr<xenithium::arch::VisionStore::Lease>> leases;
  int64_t want_delta = 0;
  for (const char *name : {"c_base420.jpg", "synthetic 1024 x 512"}) {
    xenithium::util::Image im;
    m::VisionInput in;
    if (name[0] == 'c') {
      const std::string bytes = read_all(std::string(XENITHIUM_SOURCE_DIR) + "/tests/data/vision/" + name);
      CHECK(xenithium::util::decode_image(reinterpret_cast<const uint8_t *>(bytes.data()), bytes.size(), im, err), "%s: %s", name, err.c_str());
    } else {  // a wide image of 32 x 16 tokens (delta -480): bands and a disc, so the reply has something to describe
      im.w = 1024, im.h = 512;
      im.rgb.resize(static_cast<size_t>(im.w) * im.h * 3);
      for (int y = 0; y < im.h; ++y)
        for (int x = 0; x < im.w; ++x) {
          uint8_t *p = &im.rgb[(static_cast<size_t>(y) * im.w + x) * 3];
          const bool disc = (x - 700) * (x - 700) + (y - 256) * (y - 256) < 120 * 120;
          p[0] = disc ? 230 : static_cast<uint8_t>(x / 4), p[1] = disc ? 200 : static_cast<uint8_t>(y / 2), p[2] = disc ? 20 : 160;
        }
    }
    CHECK(m::preprocess(vc, im, 1048576, in, err), "%s: %s", name, err.c_str());
    const int gh = in.gh / vc.merge, gw = in.gw / vc.merge;
    auto lease = store.add(std::move(in), err);
    CHECK(lease != nullptr, "store.add: %s", err.c_str());
    if (!lease) return 1;
    for (int k = 0; k < lease->tokens; ++k) prompt.push_back(static_cast<int32_t>(lease->base + k));
    want_delta += std::max(gh, gw) - lease->tokens;
    std::printf("  image %s: %d x %d tokens at ids %lld..\n", name, gh, gw, static_cast<long long>(lease->base));
    leases.push_back(std::move(lease));
    const auto mid = tok.encode(leases.size() == 1 ? "<|vision_end|>and <|vision_start|>" : "<|vision_end|>Describe both images.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
    prompt.insert(prompt.end(), mid.begin(), mid.end());
  }
  const Run d = decode(md, 0, prompt, steps);
  CHECK(md.rope_delta(0) == want_delta, "delta after the prompt %lld, the reference %lld", static_cast<long long>(md.rope_delta(0)), static_cast<long long>(want_delta));
  const Cmp ci = compare(d, prefill_rows(md, 1, prompt, d.reply), V);
  std::printf("  reply: %s\n", tok.decode(d.reply, true).c_str());
  std::printf("  prefill vs decode, %d rows after the prompt: text-only picks off %d, KL max %.5f mean %.6f, top-64 |dlogit| %.3f | two images (delta %lld) "
              "picks off %d, KL max %.5f mean %.6f, |dlogit| %.3f\n",
              steps, ct.picks_off, ct.kl_max, ct.kl_mean, ct.dl_max, static_cast<long long>(md.rope_delta(0)), ci.picks_off, ci.kl_max, ci.kl_mean, ci.dl_max);
  CHECK(ci.picks_off == 0, "image prompt: %d picks of the prefilled reply differ from the decoded ones", ci.picks_off);
  // bounds measured: correct 1.2x the control's KL max, equal |dlogit|; the decode records' delta zeroed
  CHECK(ci.kl_max <= 3 * ct.kl_max, "image prompt: prefill vs decode KL %.5f past 3x the text control's %.5f", ci.kl_max, ct.kl_max);
  CHECK(ci.dl_max <= 3 * ct.dl_max, "image prompt: prefill vs decode |dlogit| %.3f past 3x the text control's %.3f", ci.dl_max, ct.dl_max);

  // a snapshot inside the first image's rows (position 64: the first call's end), restored on another slot, then the
  // rest: the last row's logits == the uninterrupted call's (prefill_rows above ran the same 64-row calls)
  md.use_slot(0);
  md.reset();
  md.prefill(prompt.data(), 64);
  const int snap = md.snap_save(0);
  CHECK(snap >= 0, "snapshot at 64");
  md.use_slot(1);
  md.reset();
  md.kv_reserve(1, static_cast<int64_t>(prompt.size()));
  md.snap_restore(1, snap);
  CHECK(md.rope_delta(1) == md.rope_delta(0), "the snapshot's delta");
  md.prefill(prompt.data() + 64, static_cast<int>(prompt.size()) - 64);
  std::vector<float> lg;
  const int last = (static_cast<int>(prompt.size()) - 64 - 1) % 4 + 1;
  md.logits(last, lg);
  const std::vector<float> row(lg.begin() + (last - 1) * V, lg.begin() + last * V);
  CHECK(row == d.lg[0], "restored inside an image: the prompt's last row differs from the uninterrupted call");
  std::printf("  snapshot at 64 (inside image 1, delta %lld) restored on slot 1: last row %s\n", static_cast<long long>(md.rope_delta(0)),
              row == d.lg[0] ? "== the uninterrupted call" : "DIFFERS");
  std::printf("%s: %d checks, %d failures\n", g_failures ? "FAIL" : "PASS", g_checks, g_failures);
  return g_failures ? 1 : 0;
}
