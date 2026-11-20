// Gate for the vision input's host side (xenithium/model/vision.hpp): smart_resize, the uint8 antialiased bicubic
// resize against the reference image processor's output pixels (tests/data/vision/pp.json + <image>.<cap>.u8: its
// pixel_values mapped back to 8-bit, exact), the normalisation + patch layout, the position / rotary tables. An
// optional argument names a directory with more goldens.
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "xenithium/kernels/pack.hpp"
#include "xenithium/model/vision.hpp"
#include "xenithium/util/json.hpp"

namespace {
namespace m = xenithium::model;
namespace u = xenithium::util;
namespace json = xenithium::json;

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

m::VisionConfig qwen_vc() {
  m::VisionConfig vc;
  vc.present = true;
  vc.depth = 27, vc.hidden = 1152, vc.heads = 16, vc.inter = 4304, vc.out_hidden = 2560, vc.num_pos = 2304;
  vc.act = "gelu_pytorch_tanh", vc.image_token = 248056;
  return vc;
}

void resize_cases() {
  std::string err;
  int h, w;
  CHECK(m::smart_resize(61, 97, 32, 65536, 1048576, h, w, err) && h == 224 && w == 352, "small image scaled up: %dx%d", h, w);
  CHECK(m::smart_resize(928, 1920, 32, 65536, 1048576, h, w, err) && h == 704 && w == 1472, "1920x928 at 1M: %dx%d", h, w);
  CHECK(m::smart_resize(928, 1920, 32, 65536, 262144, h, w, err) && h == 352 && w == 736, "1920x928 at 256K: %dx%d", h, w);
  CHECK(m::smart_resize(480, 640, 32, 65536, 16777216, h, w, err) && h == 480 && w == 640, "640x480 kept: %dx%d", h, w);
  CHECK(!m::smart_resize(10, 2500, 32, 65536, 1048576, h, w, err), "aspect ratio 250 refused");
  m::VisionConfig vc = qwen_vc();
  CHECK(m::vision_tokens(vc, 928, 1920, 1048576, err) == 1012, "tokens of 1920x928 at 1M");
}

// dec: dir with index.json + <name>.rgb (decoded images); pp: dir with pp.json / u8.json + <key>.u8
void golden_cases(const std::string &dec_dir, const std::string &pp_dir, const std::string &pp_index) {
  const std::string pi = read_all(pp_dir + "/" + pp_index), di = read_all(dec_dir + "/index.json");
  if (pi.empty() || di.empty()) {
    std::printf("[SKIP] no preprocessing goldens in %s\n", pp_dir.c_str());
    return;
  }
  json::Value P, D;
  CHECK(json::parse(pi, P) && json::parse(di, D), "golden index json");
  const m::VisionConfig vc = qwen_vc();
  for (const auto &kv : P.obj) {
    const std::string &key = kv.first;
    const size_t dot = key.rfind('.');
    const std::string name = key.substr(0, dot);
    const int64_t cap = std::stoll(key.substr(dot + 1));
    const json::Value *d = D.get(name);
    const std::string rgb = read_all(dec_dir + "/" + name + ".rgb"), want = read_all(pp_dir + "/" + key + ".u8");
    if (!d || rgb.empty() || want.empty()) {
      std::printf("[SKIP] %s\n", key.c_str());
      continue;
    }
    u::Image im;
    im.w = static_cast<int>(d->get("w")->num), im.h = static_cast<int>(d->get("h")->num);
    im.rgb.assign(rgb.begin(), rgb.end());
    m::VisionInput vi;
    std::string err;
    CHECK(m::preprocess(vc, im, cap, vi, err), "%s: %s", key.c_str(), err.c_str());
    const json::Value &g = *kv.second.get("grid");
    CHECK(vi.gh == static_cast<int>(g.arr[1].num) && vi.gw == static_cast<int>(g.arr[2].num), "%s: grid %dx%d", key.c_str(), vi.gh, vi.gw);
    // the resized pixels, rebuilt from the patches (frame 0) and compared as 8-bit values
    const int W = vi.width, H = vi.height;
    if (want.size() != static_cast<size_t>(W) * H * 3) {
      CHECK(false, "%s: golden size", key.c_str());
      continue;
    }
    size_t bad = 0, bad_bits = 0, bad_dup = 0;
    int maxd = 0;
    const int pd = vc.patch_dim();
    int p = 0;
    for (int bh = 0; bh < vi.gh / 2; ++bh)
      for (int bw = 0; bw < vi.gw / 2; ++bw)
        for (int i = 0; i < 2; ++i)
          for (int j = 0; j < 2; ++j, ++p)
            for (int c = 0; c < 3; ++c)
              for (int y = 0; y < 16; ++y)
                for (int x = 0; x < 16; ++x) {
                  const int r = (bh * 2 + i) * 16 + y, col = (bw * 2 + j) * 16 + x;
                  const int g8 = static_cast<uint8_t>(want[(static_cast<size_t>(r) * W + col) * 3 + c]);
                  const uint16_t got = vi.patches[static_cast<size_t>(p) * pd + ((c * 2 + 0) * 16 + y) * 16 + x];
                  const uint16_t dup = vi.patches[static_cast<size_t>(p) * pd + ((c * 2 + 1) * 16 + y) * 16 + x];
                  const uint16_t exp = xenithium::kernels::bf16_bits((static_cast<float>(g8) - 127.5f) / 127.5f);
                  if (got != exp) {
                    ++bad_bits;
                    const int v8 = static_cast<int>(std::lround(xenithium::kernels::bf16_value(got) * 127.5f + 127.5f));
                    if (v8 != g8) ++bad, maxd = std::max(maxd, std::abs(v8 - g8));
                  }
                  if (dup != got) ++bad_dup;
                }
    std::printf("  %-26s grid %3dx%-3d pixels differing %zu (max %d), bf16 words %zu\n", key.c_str(), vi.gh, vi.gw, bad, maxd, bad_bits);
    CHECK(bad == 0 && bad_bits == 0, "%s: %zu resized samples differ from the reference", key.c_str(), bad);
    CHECK(bad_dup == 0, "%s: temporal frames differ", key.c_str());
    // tables: bilinear weights sum to ~1, indices in range, rope angles of the row / column
    double wmax = 0;
    for (int q = 0; q < vi.patches_n(); ++q) {
      double s = 0;
      for (int k = 0; k < 4; ++k) {
        s += xenithium::kernels::bf16_value(vi.pos_w[static_cast<size_t>(q) * 4 + k]);
        CHECK(vi.pos_idx[static_cast<size_t>(q) * 4 + k] >= 0 && vi.pos_idx[static_cast<size_t>(q) * 4 + k] < vc.num_pos, "pos index");
      }
      wmax = std::max(wmax, std::fabs(s - 1.0));
    }
    CHECK(wmax < 0.02, "%s: bilinear weights sum off by %g", key.c_str(), wmax);
  }
}

// Mrope:row over the reference's get_rope_index golden (text runs and images, separate
// images) and Mrope:pick against apply_interleaved_mrope's selection of every (row, frequency).
void mrope_cases(const std::string &dir) {
  std::string idx_text = read_all(dir + "/index.json"), err;
  if (idx_text.empty()) {
    std::printf("  mrope: no golden in %s (skipped)\n", dir.c_str());
    return;
  }
  json::Value idx;
  CHECK(json::parse(idx_text, idx, &err), "mrope index.json: %s", err.c_str());
  const std::string pos_b = read_all(dir + "/pos.i32"), sel_b = read_all(dir + "/sel.i32");
  const int n = static_cast<int>(idx.get("n")->num), merge = static_cast<int>(idx.get("merge")->num);
  xenithium::model::Mrope mr;
  for (int i = 0; i < 3; ++i) mr.sec[i] = static_cast<int>(idx.get("mrope_section")->arr[static_cast<size_t>(i)].num);
  const int half = mr.sec[0] + mr.sec[1] + mr.sec[2];
  CHECK(pos_b.size() == static_cast<size_t>(3) * n * 4 && sel_b.size() == static_cast<size_t>(n) * half * 4, "mrope golden sizes");
  if (g_failures) return;
  const int32_t *gp = reinterpret_cast<const int32_t *>(pos_b.data()), *gs = reinterpret_cast<const int32_t *>(sel_b.data());
  int64_t delta = 0, P = 0, img_base = 1000000;
  size_t bad = 0, bad_sel = 0, images = 0;
  for (const json::Value &seg : idx.get("layout")->arr) {
    const bool text = seg.arr[0].str == "text";
    xenithium::model::MropeImage im;
    int rows = static_cast<int>(seg.arr[1].num);
    if (!text) {
      im.gh = rows / merge, im.gw = static_cast<int>(seg.arr[2].num) / merge, im.tokens = im.gh * im.gw, im.base = img_base;
      img_base += im.tokens;
      ++images;
      rows = im.tokens;
    }
    for (int k = 0; k < rows; ++k, ++P) {
      int64_t thw[3];
      xenithium::model::Mrope::row(P, delta, text ? nullptr : &im, k, thw);
      for (int a = 0; a < 3; ++a) bad += thw[a] != gp[static_cast<size_t>(a) * n + P];
      for (int i = 0; i < half; ++i) bad_sel += thw[mr.pick(i)] != gs[static_cast<size_t>(P) * half + i];
    }
  }
  CHECK(P == n, "mrope: %lld rows rebuilt of %d", static_cast<long long>(P), n);
  CHECK(bad == 0 && bad_sel == 0, "mrope: %zu positions, %zu frequency picks differ from the reference", bad, bad_sel);
  CHECK(delta == static_cast<int64_t>(idx.get("delta")->num), "mrope: delta %lld vs the reference's %lld", static_cast<long long>(delta),
        static_cast<long long>(idx.get("delta")->num));
  // compose: each frequency's cos / sin from its picked position's 1-D row
  std::vector<uint16_t> table(static_cast<size_t>(200) * 2 * half), row(2 * half);
  for (size_t i = 0; i < table.size(); ++i) table[i] = static_cast<uint16_t>(i * 2654435761u >> 16);
  const int64_t thw[3] = {7, 150, 31};
  mr.compose(table.data(), 2 * half, thw, row.data());
  size_t bad_c = 0;
  for (int i = 0; i < half; ++i)
    bad_c += row[i] != table[static_cast<size_t>(thw[mr.pick(i)]) * 2 * half + i] || row[half + i] != table[static_cast<size_t>(thw[mr.pick(i)]) * 2 * half + half + i];
  CHECK(bad_c == 0, "mrope compose: %zu frequencies off", bad_c);
  std::printf("  mrope: %d rows (%zu images) == the reference's positions and frequency picks, delta %lld\n", n,
              images, static_cast<long long>(delta));
}

}  // namespace

int main(int argc, char **argv) {
  resize_cases();
  const std::string data = XENITHIUM_SOURCE_DIR "/tests/data/vision";
  golden_cases(data, data, "pp.json");
  mrope_cases(data + "/mrope");
  if (argc > 1) golden_cases(std::string(argv[1]) + "/dec", std::string(argv[1]) + "/pp", "u8.json");
  std::printf("%s: %d checks, %d failures\n", g_failures ? "FAIL" : "PASS", g_checks, g_failures);
  return g_failures ? 1 : 0;
}
