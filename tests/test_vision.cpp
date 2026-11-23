// Gate for the vision tower on the cards: the encoder's own kernels bit-exact against their host
// references (kernels/encoder/vit.hpp: position-grid sampling, 2-D rope), then the whole tower on a checkpoint's real
// weights against the reference's output (HF Qwen3VLVisionModel in bf16 on the CPU: per
// image the cosine of every token's row and the relative rms error, the tower split over two cards == on one card bit
// for bit, and the encode time; then the image store on every card of a fabric of up to four cards (each card's embedding
// gather reads the same image rows).
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <sycl/sycl.hpp>

#include "xenithium/arch/common/vision.hpp"
#include "xenithium/kernels/encoder/vit.hpp"
#include "xenithium/kernels/head/head.hpp"
#include "xenithium/kernels/pack.hpp"
#include "xenithium/model/checkpoint.hpp"
#include "xenithium/model/config.hpp"
#include "xenithium/model/vision.hpp"
#include "xenithium/rt/fabric.hpp"
#include "xenithium/rt/l0.hpp"
#include "xenithium/util/json.hpp"

namespace {
namespace k = xenithium::kernels;
namespace m = xenithium::model;
namespace json = xenithium::json;
using bf16 = k::vit_bf16;

int g_checks = 0, g_failures = 0;
#define CHECK(cond, ...)                                  \
  do {                                                    \
    ++g_checks;                                           \
    if (!(cond)) {                                        \
      ++g_failures;                                       \
      std::printf("[FAIL] %s:%d ", __FILE__, __LINE__);   \
      std::printf(__VA_ARGS__);                           \
      std::printf("\n");                                  \
    }                                                     \
  } while (0)

std::string read_all(const std::string &p) {
  std::ifstream f(p, std::ios::binary);
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

void kernel_cases(sycl::queue &q) {
  std::mt19937 rng(167);
  std::normal_distribution<float> nd(0.f, 1.f);
  const int n = 300, H = 1152, side = 48, heads = 16, hd = 72;
  // vit_pos_add
  std::vector<uint16_t> table(static_cast<size_t>(side) * side * H), x(static_cast<size_t>(n) * H), w(static_cast<size_t>(n) * 4);
  std::vector<int32_t> idx(static_cast<size_t>(n) * 4);
  for (auto &v : table) v = k::bf16_bits(nd(rng) * 0.05f);
  for (auto &v : x) v = k::bf16_bits(nd(rng));
  for (int p = 0; p < n; ++p)
    for (int j = 0; j < 4; ++j) {
      idx[static_cast<size_t>(p) * 4 + j] = static_cast<int32_t>(rng() % (side * side));
      w[static_cast<size_t>(p) * 4 + j] = k::bf16_bits(std::fabs(nd(rng)) * 0.3f);
    }
  auto *dt = sycl::malloc_device<uint16_t>(table.size(), q);
  auto *dx = sycl::malloc_device<uint16_t>(x.size(), q);
  auto *dw = sycl::malloc_device<uint16_t>(w.size(), q);
  auto *di = sycl::malloc_device<int32_t>(idx.size(), q);
  q.memcpy(dt, table.data(), table.size() * 2);
  q.memcpy(dx, x.data(), x.size() * 2);
  q.memcpy(dw, w.data(), w.size() * 2);
  q.memcpy(di, idx.data(), idx.size() * 4).wait();
  k::vit_pos_add(q, n, H, reinterpret_cast<bf16 *>(dt), di, reinterpret_cast<bf16 *>(dw), reinterpret_cast<bf16 *>(dx)).wait();
  std::vector<uint16_t> got(x.size());
  q.memcpy(got.data(), dx, got.size() * 2).wait();
  k::vit_pos_add_ref(n, H, table.data(), idx.data(), w.data(), x.data());
  size_t bad = 0;
  for (size_t i = 0; i < x.size(); ++i) bad += got[i] != x[i];
  CHECK(bad == 0, "vit_pos_add: %zu of %zu words differ from the host reference", bad, x.size());
  // vit_rope
  std::vector<uint16_t> qkv(static_cast<size_t>(n) * 3 * heads * hd);
  std::vector<float> cs(static_cast<size_t>(n) * hd / 2), sn(cs.size());
  for (auto &v : qkv) v = k::bf16_bits(nd(rng) * 2.f);
  for (size_t i = 0; i < cs.size(); ++i) {
    const float a = nd(rng) * 10.f;
    cs[i] = std::cos(a), sn[i] = std::sin(a);
  }
  auto *dq = sycl::malloc_device<uint16_t>(qkv.size(), q);
  auto *dc = sycl::malloc_device<float>(cs.size(), q);
  auto *ds = sycl::malloc_device<float>(sn.size(), q);
  q.memcpy(dq, qkv.data(), qkv.size() * 2);
  q.memcpy(dc, cs.data(), cs.size() * 4);
  q.memcpy(ds, sn.data(), sn.size() * 4).wait();
  k::vit_rope(q, n, heads, hd, dc, ds, reinterpret_cast<bf16 *>(dq)).wait();
  std::vector<uint16_t> gq(qkv.size());
  q.memcpy(gq.data(), dq, gq.size() * 2).wait();
  k::vit_rope_ref(n, heads, hd, cs.data(), sn.data(), qkv.data());
  bad = 0;
  for (size_t i = 0; i < qkv.size(); ++i) bad += gq[i] != qkv[i];
  CHECK(bad == 0, "vit_rope: %zu of %zu words differ from the host reference", bad, qkv.size());
  // vit_layer_norm (rows with a large offset and an outlier column: the two-pass statistics)
  {
    const int rows = 257;
    std::vector<uint16_t> xl(static_cast<size_t>(rows) * H), yl(xl.size()), yg(xl.size());
    std::vector<float> g(H), b(H);
    for (int r = 0; r < rows; ++r)
      for (int c = 0; c < H; ++c) xl[static_cast<size_t>(r) * H + c] = k::bf16_bits(nd(rng) + (r % 3 == 0 ? 300.f : 0.f) + (c == 5 && r % 2 ? 4000.f : 0.f));
    for (int c = 0; c < H; ++c) g[c] = nd(rng), b[c] = nd(rng) * 0.1f;
    auto *dxl = sycl::malloc_device<uint16_t>(xl.size(), q);
    auto *dyl = sycl::malloc_device<uint16_t>(xl.size(), q);
    auto *dg = sycl::malloc_device<float>(H, q);
    auto *db = sycl::malloc_device<float>(H, q);
    q.memcpy(dxl, xl.data(), xl.size() * 2);
    q.memcpy(dg, g.data(), H * 4);
    q.memcpy(db, b.data(), H * 4).wait();
    k::vit_layer_norm(q, rows, H, reinterpret_cast<bf16 *>(dxl), dg, db, 1e-6f, reinterpret_cast<bf16 *>(dyl)).wait();
    q.memcpy(yg.data(), dyl, yg.size() * 2).wait();
    k::vit_layer_norm_ref(rows, H, xl.data(), g.data(), b.data(), 1e-6f, yl.data());
    size_t badl = 0;
    for (size_t i = 0; i < yl.size(); ++i) badl += yl[i] != yg[i];
    CHECK(badl == 0, "vit_layer_norm: %zu of %zu words differ from the host reference", badl, yl.size());
    std::printf("  kernels: vit_layer_norm %d x %d: %s\n", rows, H, badl ? "DIFFERS" : "bit-exact");
    for (void *p : {static_cast<void *>(dxl), static_cast<void *>(dyl), static_cast<void *>(dg), static_cast<void *>(db)}) sycl::free(p, q);
  }
  for (void *p : {static_cast<void *>(dt), static_cast<void *>(dx), static_cast<void *>(dw), static_cast<void *>(di), static_cast<void *>(dq), static_cast<void *>(dc),
                  static_cast<void *>(ds)})
    sycl::free(p, q);
  std::printf("  kernels: vit_pos_add %d x %d, vit_rope %d x %d heads x %d: bit-exact\n", n, H, n, heads, hd);
}

// vit_attention (oneDNN SDPA) against a double-precision host attention on the same bf16 q, k, v: a sample of query
// rows, every head; the bound is bf16's (P and o rounded).
void attention_cases(sycl::queue &q) {
  std::mt19937 rng(7);
  std::normal_distribution<float> nd(0.f, 1.f);
  const int heads = 16, hd = 72;
  for (const int seq : {308, 936, 1012, 4048}) {
    std::vector<uint16_t> qkv(static_cast<size_t>(seq) * 3 * heads * hd);
    for (auto &v : qkv) v = k::bf16_bits(nd(rng) * 1.5f);
    auto *dq = sycl::malloc_device<uint16_t>(qkv.size(), q);
    auto *dout = sycl::malloc_device<uint16_t>(static_cast<size_t>(seq) * heads * hd, q);
    q.memcpy(dq, qkv.data(), qkv.size() * 2).wait();
    k::vit_attention(q, seq, heads, hd, reinterpret_cast<bf16 *>(dq), reinterpret_cast<bf16 *>(dout)).wait();
    std::vector<uint16_t> o(static_cast<size_t>(seq) * heads * hd);
    q.memcpy(o.data(), dout, o.size() * 2).wait();
    double worst = 0, num = 0, den = 0;
    int worst_row = -1;
    const size_t row = static_cast<size_t>(3) * heads * hd;
    for (int r = 0; r < seq; r += std::max(1, seq / 97)) {
      for (int h = 0; h < heads; ++h) {
        std::vector<double> s(static_cast<size_t>(seq));
        double mx = -1e300;
        for (int j = 0; j < seq; ++j) {
          double d = 0;
          for (int i = 0; i < hd; ++i) d += static_cast<double>(k::bf16_value(qkv[r * row + h * hd + i])) * k::bf16_value(qkv[j * row + heads * hd + h * hd + i]);
          s[static_cast<size_t>(j)] = d / std::sqrt(static_cast<double>(hd));
          mx = std::max(mx, s[static_cast<size_t>(j)]);
        }
        double z = 0;
        for (auto &v : s) z += (v = std::exp(v - mx));
        double rn = 0, rd = 0;
        for (int i = 0; i < hd; ++i) {
          double a = 0;
          for (int j = 0; j < seq; ++j) a += s[static_cast<size_t>(j)] / z * k::bf16_value(qkv[j * row + 2 * heads * hd + h * hd + i]);
          const double g = k::bf16_value(o[(static_cast<size_t>(r) * heads + h) * hd + i]);
          rn += (g - a) * (g - a), rd += a * a;
        }
        num += rn, den += rd;
        if (std::sqrt(rn / rd) > worst) worst = std::sqrt(rn / rd), worst_row = r;
      }
    }
    k::vit_attention(q, seq, heads, hd, reinterpret_cast<bf16 *>(dq), reinterpret_cast<bf16 *>(dout)).wait();
    std::vector<uint16_t> o2(o.size());
    q.memcpy(o2.data(), dout, o2.size() * 2).wait();
    CHECK(o2 == o, "vit_attention seq %d: not deterministic run to run", seq);
    std::printf("  vit_attention seq %4d: rel rms %.5f, worst (row, all heads) %.5f at row %d%s\n", seq, std::sqrt(num / den), worst, worst_row,
                o2 == o ? "" : "; NOT deterministic");
    CHECK(std::sqrt(num / den) < 0.01 && worst < 0.03, "vit_attention seq %d off a double attention (rel rms %.5f, worst %.5f)", seq, std::sqrt(num / den), worst);
    sycl::free(dq, q), sycl::free(dout, q);
  }
}

// vit_linear run to run on the tower's shapes at a few patch counts
void linear_cases(sycl::queue &q) {
  std::mt19937 rng(11);
  std::normal_distribution<float> nd(0.f, 1.f);
  struct S { int n, k; k::VitAct act; bool res; };
  const S shapes[] = {{3456, 1152, k::VitAct::none, false}, {1152, 1152, k::VitAct::none, true}, {4304, 1152, k::VitAct::gelu_tanh, false},
                      {1152, 4304, k::VitAct::none, true}, {4608, 4608, k::VitAct::gelu_erf, false}, {2560, 4608, k::VitAct::none, false}, {1152, 1536, k::VitAct::none, false}};
  for (const int m0 : {308, 936, 4048}) {
    for (const S &sh : shapes) {
      const int m = sh.k == 4608 ? m0 / 4 : m0;
      std::vector<uint16_t> a(static_cast<size_t>(m) * sh.k), w(static_cast<size_t>(sh.n) * sh.k), b(sh.n), r(static_cast<size_t>(m) * sh.n);
      for (auto &v : a) v = k::bf16_bits(nd(rng));
      for (auto &v : w) v = k::bf16_bits(nd(rng) * 0.03f);
      for (auto &v : b) v = k::bf16_bits(nd(rng) * 0.1f);
      for (auto &v : r) v = k::bf16_bits(nd(rng));
      auto *da = sycl::malloc_device<uint16_t>(a.size(), q);
      auto *dw = sycl::malloc_device<uint16_t>(w.size(), q);
      auto *db = sycl::malloc_device<uint16_t>(b.size(), q);
      auto *dr = sycl::malloc_device<uint16_t>(r.size(), q);
      auto *dout = sycl::malloc_device<uint16_t>(r.size(), q);
      q.memcpy(da, a.data(), a.size() * 2), q.memcpy(dw, w.data(), w.size() * 2), q.memcpy(db, b.data(), b.size() * 2), q.memcpy(dr, r.data(), r.size() * 2).wait();
      std::vector<uint16_t> o1(r.size()), o2(r.size());
      for (auto *o : {&o1, &o2}) {
        k::vit_linear(q, m, sh.n, sh.k, reinterpret_cast<bf16 *>(da), sh.k, reinterpret_cast<bf16 *>(dw), reinterpret_cast<bf16 *>(db), sh.act,
                      sh.res ? reinterpret_cast<bf16 *>(dr) : nullptr, reinterpret_cast<bf16 *>(dout), sh.n).wait();
        q.memcpy(o->data(), dout, o->size() * 2).wait();
      }
      // one row against a double reference (bias, activation, residual), loose: the rounding is bf16's
      double num = 0, den = 0;
      for (int j = 0; j < sh.n; ++j) {
        double acc = k::bf16_value(b[j]);
        for (int t = 0; t < sh.k; ++t) acc += static_cast<double>(k::bf16_value(a[t])) * k::bf16_value(w[static_cast<size_t>(j) * sh.k + t]);
        if (sh.act == k::VitAct::gelu_tanh) acc = 0.5 * acc * (1 + std::tanh(0.7978845608028654 * (acc + 0.044715 * acc * acc * acc)));
        if (sh.act == k::VitAct::gelu_erf) acc = 0.5 * acc * (1 + std::erf(acc / std::sqrt(2.0)));
        if (sh.res) acc += k::bf16_value(r[j]);
        const double g = k::bf16_value(o1[j]);
        num += (g - acc) * (g - acc), den += acc * acc;
      }
      const bool det = o1 == o2;
      CHECK(det, "vit_linear %d x %d x %d: not deterministic run to run", m, sh.n, sh.k);
      CHECK(std::sqrt(num / den) < 0.01, "vit_linear %d x %d x %d: row 0 rel rms %.5f vs double", m, sh.n, sh.k, std::sqrt(num / den));
      if (!det) std::printf("  vit_linear %d x %d x %d: NOT deterministic\n", m, sh.n, sh.k);
      for (void *p : {static_cast<void *>(da), static_cast<void *>(dw), static_cast<void *>(db), static_cast<void *>(dr), static_cast<void *>(dout)}) sycl::free(p, q);
    }
  }
  std::printf("  vit_linear: the tower's 7 shapes at 308 / 936 / 4048 patches checked (row 0 vs double, run to run)\n");
}

void tower_cases(const std::string &gold, const std::string &model_dir, const std::string &tag) {
  const std::string di = read_all(gold + "/dec/index.json"), pi = read_all(gold + "/pp/u8.json"), ti = read_all(gold + "/tower/index.json");
  if (di.empty() || pi.empty() || ti.empty()) {
    std::printf("[SKIP] no tower goldens in %s\n", gold.c_str());
    return;
  }
  json::Value D, P, T;
  json::parse(di, D), json::parse(pi, P), json::parse(ti, T);
  const auto cfg = m::ModelConfig::load(model_dir);
  const m::VisionConfig vc = m::VisionConfig::parse(cfg);
  CHECK(vc.present, "%s has a vision tower", model_dir.c_str());
  if (!vc.present) return;
  const auto ck = m::Checkpoint::open(model_dir);
  const xenithium::l0::Loader loader = xenithium::l0::Loader::open();
  const int max_patches = 4096;
  struct Out {
    std::vector<uint16_t> rows;
    double ms = 0;
  };
  for (const int ncards : {2, 1}) {
    xenithium::Fabric fab(loader, ncards == 2 ? std::vector<size_t>{0, 1} : std::vector<size_t>{0}, false);
    const auto tl = std::chrono::steady_clock::now();
    xenithium::arch::VisionTower tower(fab, ck, vc, max_patches);
    std::printf("  %s tower on %d card(s): loaded in %.2f s; %.3f / %.3f GiB on card 0 / last\n", tag.c_str(), ncards,
                std::chrono::duration<double>(std::chrono::steady_clock::now() - tl).count(), tower.bytes(0) / 1073741824.0, tower.bytes(ncards - 1) / 1073741824.0);
    for (const auto &kv : T.obj) {
      if (kv.first.rfind(tag + ".", 0) != 0) continue;
      const std::string key = kv.first.substr(tag.size() + 1);  // <image>.<cap>
      const std::string name = key.substr(0, key.rfind('.'));
      const int64_t cap = std::stoll(key.substr(key.rfind('.') + 1));
      const json::Value *d = D.get(name);
      const std::string rgb = read_all(gold + "/dec/" + name + ".rgb"), want = read_all(gold + "/tower/" + kv.first + ".f32");
      if (!d || rgb.empty() || want.empty()) continue;
      xenithium::util::Image im;
      im.w = static_cast<int>(d->get("w")->num), im.h = static_cast<int>(d->get("h")->num);
      im.rgb.assign(rgb.begin(), rgb.end());
      m::VisionInput in;
      std::string err;
      CHECK(m::preprocess(vc, im, cap, in, err), "%s", err.c_str());
      const int toks = in.tokens(vc.merge);
      std::vector<void *> dst;
      for (size_t c = 0; c < fab.size(); ++c) dst.push_back(fab.alloc(c, static_cast<size_t>(toks) * vc.out_hidden * 2 + 4096));
      tower.encode(in, dst);  // first: compiles the shapes
      std::vector<uint16_t> first(static_cast<size_t>(toks) * vc.out_hidden);
      fab.compute(0).memcpy(first.data(), dst[0], first.size() * 2).wait();
      double best = 1e9;
      for (int r = 0; r < 3; ++r) {
        tower.encode(in, dst);
        best = std::min(best, tower.last_ms());
      }
      std::vector<std::vector<uint16_t>> got(fab.size(), std::vector<uint16_t>(static_cast<size_t>(toks) * vc.out_hidden));
      for (size_t c = 0; c < fab.size(); ++c) fab.compute(c).memcpy(got[c].data(), dst[c], got[c].size() * 2).wait();
      CHECK(first == got[0], "%s: the encode is not deterministic run to run", key.c_str());
      if (fab.size() > 1) CHECK(got[0] == got[1], "%s: the cards' rows differ", key.c_str());
      static std::vector<std::pair<std::string, std::vector<uint16_t>>> two_card;
      if (ncards == 2) two_card.push_back({key, got[0]});
      else
        for (const auto &tc : two_card)
          if (tc.first == key) CHECK(tc.second == got[0], "%s: one card != two cards", key.c_str());
      // against the fp32 tower (the truth both bf16 implementations approximate), next to the reference's own bf16 run
      const std::string want32 = read_all(gold + "/tower/" + kv.first + ".fp32");
      const float *ref = reinterpret_cast<const float *>(want.data());
      const float *r32 = want32.empty() ? nullptr : reinterpret_cast<const float *>(want32.data());
      CHECK(want.size() == got[0].size() * 4, "%s: golden rows %zu vs %d tokens", key.c_str(), want.size() / 4 / vc.out_hidden, toks);
      if (want.size() != got[0].size() * 4 || !r32) continue;
      auto score = [&](auto val, double &cmin, double &cmean) {
        double num = 0, den = 0;
        cmin = 1, cmean = 0;
        for (int t = 0; t < toks; ++t) {
          double dot = 0, a2 = 0, b2 = 0;
          for (int j = 0; j < vc.out_hidden; ++j) {
            const size_t i = static_cast<size_t>(t) * vc.out_hidden + j;
            const double a = val(i), b = r32[i];
            dot += a * b, a2 += a * a, b2 += b * b, num += (a - b) * (a - b), den += b * b;
          }
          const double c = dot / std::sqrt(a2 * b2 + 1e-30);
          cmin = std::min(cmin, c), cmean += c / toks;
        }
        return std::sqrt(num / den);
      };
      double oc, om, hc, hm;
      const double ro = score([&](size_t i) { return static_cast<double>(k::bf16_value(got[0][i])); }, oc, om);
      const double rh = score([&](size_t i) { return static_cast<double>(ref[i]); }, hc, hm);
      std::printf("    %-26s grid %2dx%-3d %4d patches -> %4d tokens: encode %6.1f ms | vs fp32: rel rms %.4f, cos mean %.5f min %.4f "
                  "(the reference's bf16: %.4f, %.5f, %.4f)\n",
                  key.c_str(), in.gh, in.gw, in.patches_n(), toks, best, ro, om, oc, rh, hm, hc);
      CHECK(ro <= 1.25 * rh + 0.005 && om >= hm - 0.002, "%s: further from the fp32 tower than the reference's bf16 run", key.c_str());
      for (size_t c = 0; c < fab.size(); ++c) fab.release(dst[c]);
    }
  }
}

// The tower on videos: one encode per temporal patch (two frames; the reference attends within each temporal
// patch), the rows concatenated == the reference's run of the whole video, against its fp32 tower beside its own bf16 run.
void video_cases(const std::string &gold, const std::string &model_dir, const std::string &tag) {
  const std::string data = XENITHIUM_SOURCE_DIR "/tests/data/vision/video";
  const std::string ii = read_all(data + "/index.json"), ti = read_all(gold + "/tower.json");
  if (ii.empty() || ti.empty()) {
    std::printf("[SKIP] no video tower goldens in %s\n", gold.c_str());
    return;
  }
  json::Value I, T;
  json::parse(ii, I), json::parse(ti, T);
  const auto cfg = m::ModelConfig::load(model_dir);
  const m::VisionConfig vc = m::VisionConfig::parse(cfg);
  const auto ck = m::Checkpoint::open(model_dir);
  const xenithium::l0::Loader loader = xenithium::l0::Loader::open();
  xenithium::Fabric fab(loader, std::vector<size_t>{0, 1}, false);
  xenithium::arch::VisionTower tower(fab, ck, vc, 4096);
  for (const auto &kv : T.obj) {
    if (kv.first.rfind(tag + ".", 0) != 0) continue;
    const std::string name = kv.first.substr(tag.size() + 1);
    const json::Value *e = I.get(name);
    const std::string fr = read_all(data + "/" + name + ".frames.u8"), want32 = read_all(gold + "/" + kv.first + ".fp32"), want16 = read_all(gold + "/" + kv.first + ".f32");
    if (!e || fr.empty() || want32.empty() || want16.empty()) continue;
    const int n = static_cast<int>(e->get("n")->num), H = static_cast<int>(e->get("h")->num), W = static_cast<int>(e->get("w")->num);
    int rh = 0, rw = 0;
    std::string err;
    CHECK(m::video_smart_resize(n, H, W, vc.temporal, vc.patch * vc.merge, vc.video_min_pixels, vc.video_max_pixels, rh, rw, err), "%s", err.c_str());
    std::vector<xenithium::util::Image> rs(static_cast<size_t>(n));
    for (int f = 0; f < n; ++f) {
      xenithium::util::Image im;
      im.w = W, im.h = H;
      im.rgb.assign(fr.begin() + static_cast<std::ptrdiff_t>(static_cast<size_t>(f) * H * W * 3), fr.begin() + static_cast<std::ptrdiff_t>(static_cast<size_t>(f + 1) * H * W * 3));
      m::resize_bicubic_u8(im, rw, rh, rs[static_cast<size_t>(f)]);
    }
    std::vector<uint16_t> got;
    double ms = 0;
    for (int t = 0; t < (n + 1) / 2; ++t) {
      m::VisionInput in;
      m::preprocess_frames(vc, rs[static_cast<size_t>(2 * t)], rs[static_cast<size_t>(std::min(2 * t + 1, n - 1))], in);
      const int toks = in.tokens(vc.merge);
      std::vector<void *> dst;
      for (size_t c = 0; c < fab.size(); ++c) dst.push_back(fab.alloc(c, static_cast<size_t>(toks) * vc.out_hidden * 2 + 4096));
      tower.encode(in, dst);
      ms += tower.last_ms();
      std::vector<uint16_t> rows(static_cast<size_t>(toks) * vc.out_hidden);
      fab.compute(0).memcpy(rows.data(), dst[0], rows.size() * 2).wait();
      got.insert(got.end(), rows.begin(), rows.end());
      for (size_t c = 0; c < fab.size(); ++c) fab.release(dst[c]);
    }
    const float *r32 = reinterpret_cast<const float *>(want32.data()), *r16 = reinterpret_cast<const float *>(want16.data());
    CHECK(want32.size() == got.size() * 4, "%s: %zu rows vs the reference's %zu", name.c_str(), got.size() / vc.out_hidden, want32.size() / 4 / vc.out_hidden);
    if (want32.size() != got.size() * 4) continue;
    const int toks = static_cast<int>(got.size() / vc.out_hidden);
    auto score = [&](auto val, double &cmin, double &cmean) {
      double num = 0, den = 0;
      cmin = 1, cmean = 0;
      for (int t = 0; t < toks; ++t) {
        double dot = 0, a2 = 0, b2 = 0;
        for (int j = 0; j < vc.out_hidden; ++j) {
          const size_t i = static_cast<size_t>(t) * vc.out_hidden + j;
          const double a = val(i), b = r32[i];
          dot += a * b, a2 += a * a, b2 += b * b, num += (a - b) * (a - b), den += b * b;
        }
        const double c = dot / std::sqrt(a2 * b2 + 1e-30);
        cmin = std::min(cmin, c), cmean += c / toks;
      }
      return std::sqrt(num / den);
    };
    double oc, om, hc, hm;
    const double ro = score([&](size_t i) { return static_cast<double>(k::bf16_value(got[i])); }, oc, om);
    const double rh2 = score([&](size_t i) { return static_cast<double>(r16[i]); }, hc, hm);
    std::printf("    video %-14s %d frames -> %d tokens: encode %6.1f ms | vs fp32: rel rms %.4f, cos mean %.5f min %.4f (the reference's bf16: %.4f, %.5f, %.4f)\n",
                name.c_str(), n, toks, ms, ro, om, oc, rh2, hm, hc);
    CHECK(ro <= 1.25 * rh2 + 0.005 && om >= hm - 0.002, "%s: further from the fp32 tower than the reference's bf16 run", name.c_str());
  }
}

// The image store on every card of the fabric (up to four, as many as the host has): each card's embedding gather of
// an image's ids reads the rows the tower wrote, the same rows on every card, and a card past the store is refused
// (a pointer read past the store's list would fault on the card).
void store_cases(const std::string &model_dir) {
  const auto cfg = m::ModelConfig::load(model_dir);
  const m::VisionConfig vc = m::VisionConfig::parse(cfg);
  if (!vc.present) return;
  const auto ck = m::Checkpoint::open(model_dir);
  const xenithium::l0::Loader loader = xenithium::l0::Loader::open();
  constexpr size_t kMostCards = 4;  // the widest load an arch makes (qwen38 TP4)
  std::vector<size_t> ids;
  for (size_t c = 0; c < std::min(loader.size(), kMostCards); ++c) ids.push_back(c);
  xenithium::Fabric fab(loader, ids, false);
  constexpr int kMaxPatches = 1024;
  constexpr int64_t kRows = 512;
  const int32_t vocab = static_cast<int32_t>(m::ModelConfig::need_int(cfg.text(), "vocab_size"));
  xenithium::arch::VisionStore store(fab, std::make_unique<xenithium::arch::VisionTower>(fab, ck, vc, kMaxPatches), vc, vocab, kRows);
  CHECK(store.cards() == static_cast<int>(fab.size()), "the store holds rows on %d of %zu cards", store.cards(), fab.size());
  bool refused = false;
  try {
    (void)store.pool(static_cast<int>(fab.size()));
  } catch (const std::out_of_range &) {
    refused = true;
  }
  CHECK(refused, "a card past the store is refused");

  xenithium::util::Image im;  // a gradient, so the rows are not all equal
  im.w = 160, im.h = 120;
  im.rgb.resize(static_cast<size_t>(im.w) * im.h * 3);
  for (size_t i = 0; i < im.rgb.size(); ++i) im.rgb[i] = static_cast<uint8_t>((i * 7) % 251);
  m::VisionInput in;
  std::string err;
  CHECK(m::preprocess(vc, im, static_cast<int64_t>(kMaxPatches) * vc.patch * vc.patch, in, err), "%s", err.c_str());
  const auto lease = store.add(std::move(in), err);
  CHECK(lease != nullptr, "store add: %s", err.c_str());
  if (!lease) return;
  const int n = lease->tokens;
  std::vector<int32_t> toks(static_cast<size_t>(n)), mapped(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) toks[static_cast<size_t>(i)] = static_cast<int32_t>(lease->base + i);
  store.prepare(toks.data(), n, mapped.data());

  using hb = xenithium::kernels::head_bf16;
  const int H = vc.out_hidden;
  std::vector<std::vector<uint16_t>> rows(fab.size(), std::vector<uint16_t>(static_cast<size_t>(n) * H));
  for (size_t c = 0; c < fab.size(); ++c) {
    sycl::queue &q = fab.compute(c);
    auto *dev_ids = static_cast<int32_t *>(fab.alloc(c, mapped.size() * 4 + 4096));
    auto *out = static_cast<hb *>(fab.alloc(c, rows[c].size() * 2 + 4096));
    auto *table = static_cast<hb *>(fab.alloc(c, static_cast<size_t>(H) * 2 + 4096));  // no id < vocab: never read
    q.memcpy(dev_ids, mapped.data(), mapped.size() * 4).wait();
    xenithium::kernels::embed_gather_ext(q, table, dev_ids, n, H, 1, out, static_cast<const hb *>(store.pool(static_cast<int>(c))), vocab).wait();
    q.memcpy(rows[c].data(), out, rows[c].size() * 2).wait();
    for (void *p : {static_cast<void *>(dev_ids), static_cast<void *>(out), static_cast<void *>(table)}) fab.release(p);
  }
  size_t nonzero = 0;
  for (const uint16_t v : rows[0]) nonzero += v != 0;
  CHECK(nonzero > rows[0].size() / 2, "card 0 gathered the image's rows (%zu of %zu values nonzero)", nonzero, rows[0].size());
  for (size_t c = 1; c < fab.size(); ++c) CHECK(rows[c] == rows[0], "card %zu gathers the same image rows as card 0", c);
  std::printf("  store on %zu cards: %d image tokens gathered on every card\n", fab.size(), n);
}

}  // namespace

int main(int argc, char **argv) {
  const std::string gold = argc > 1 ? argv[1] : XENITHIUM_SOURCE_DIR "/.spike/r167/golden";
  const std::string model_dir = argc > 2 ? argv[2] : std::string(std::getenv("HOME")) + "/models/qwen38-w4a16";
  const std::string tag = argc > 3 ? argv[3] : "qwen38";
  {
    sycl::queue q(sycl::gpu_selector_v);
    kernel_cases(q);
    attention_cases(q);
    linear_cases(q);
  }
  tower_cases(gold, model_dir, tag);
  store_cases(model_dir);
  video_cases(argc > 4 ? argv[4] : XENITHIUM_SOURCE_DIR "/.spike/r168/golden/out", model_dir, tag);
  std::printf("%s: %d checks, %d failures\n", g_failures ? "FAIL" : "PASS", g_checks, g_failures);
  return g_failures ? 1 : 0;
}
