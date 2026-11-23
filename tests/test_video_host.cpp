// Gate for video input on the host: the reference's frame sampling, sizing, timestamps and pixels
// (tests/data/vision/video/: goldens from the reference stack, and the ffmpeg frame
// source against the reference decoder's frames (when an ffmpeg is installed).
#include <unistd.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "xenithium/model/vision.hpp"
#include "xenithium/text/tokenizer.hpp"
#include "xenithium/util/json.hpp"
#include "xenithium/util/video.hpp"

namespace {
namespace m = xenithium::model;
namespace u = xenithium::util;
namespace json = xenithium::json;

int g_checks = 0, g_failures = 0;
#define CHECK(cond, ...)                                \
  do {                                                  \
    ++g_checks;                                         \
    if (!(cond)) {                                      \
      ++g_failures;                                     \
      std::printf("[FAIL] %s:%d ", __FILE__, __LINE__); \
      std::printf(__VA_ARGS__);                         \
      std::printf("\n");                                \
    }                                                   \
  } while (0)

std::string read_all(const std::string &p) {
  std::ifstream f(p, std::ios::binary);
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

m::VisionConfig config() {  // both checkpoints' processors (config.json / video_preprocessor_config.json)
  m::VisionConfig vc;
  vc.present = true;
  vc.depth = 27, vc.hidden = 1152, vc.heads = 16, vc.inter = 4304, vc.out_hidden = 2560, vc.num_pos = 2304;
  return vc;
}

void unit_cases() {
  // the reference loader: 75 frames at 25 fps, 2 fps -> 6 frames
  auto i1 = m::video_frame_indices(75, 25, 2, 4, 768);
  CHECK((i1 == std::vector<int64_t>{0, 15, 30, 44, 59, 74}), "indices 75 @ 25");
  auto i2 = m::video_frame_indices(10, 10, 2, 4, 768);  // 1 s: min 4 frames
  CHECK((i2 == std::vector<int64_t>{0, 3, 6, 9}), "indices min frames");
  auto i3 = m::video_frame_indices(3, 30, 2, 4, 768);  // fewer frames than the minimum: all of them
  CHECK((i3 == std::vector<int64_t>{0, 1, 2}), "indices short");
  auto i4 = m::video_frame_indices(30 * 3600, 30, 2, 4, 768);  // one hour: max 768
  CHECK(i4.size() == 768 && i4.front() == 0 && i4.back() == 30 * 3600 - 1, "indices max frames");
  auto i5 = m::video_frame_indices(240, 30, 60, 4, 768);  // the loader caps the rate at 30
  CHECK(i5.size() == 240, "indices fps cap 30 (%zu)", i5.size());
  auto ts = m::video_timestamps({0, 15, 30, 44, 59}, 24, 2);
  CHECK(ts.size() == 3 && m::video_timestamp_text(ts[0]) == "<0.3 seconds>" && m::video_timestamp_text(ts[2]) == "<2.5 seconds>", "timestamps odd");
  int h = 0, w = 0;
  std::string err;
  CHECK(m::video_smart_resize(6, 120, 160, 2, 32, 4096, 25165824, h, w, err) && h == 128 && w == 160, "resize small %dx%d", w, h);
  CHECK(!m::video_smart_resize(1, 120, 160, 2, 32, 4096, 25165824, h, w, err), "one frame refused");
  CHECK(m::video_smart_resize(16, 720, 1280, 2, 32, 4096, 25165824, h, w, err) && h == 704 && w == 1280, "resize 720p %dx%d", w, h);
}

// The reference's sampled frames -> our sizing + resize + patchify == its pixel_values (inverted to uint8), and the
// timestamps / token layout from its index.
void golden_cases(const std::string &dir, const std::string &tokdir) {
  const std::string idx_text = read_all(dir + "/index.json");
  if (idx_text.empty()) {
    std::printf("  video: no goldens in %s (skipped)\n", dir.c_str());
    return;
  }
  json::Value idx;
  std::string err;
  CHECK(json::parse(idx_text, idx, &err), "index.json: %s", err.c_str());
  xenithium::text::Tokenizer tok;
  const bool have_tok = !tokdir.empty() && tok.load(tokdir, err);
  const m::VisionConfig vc = config();
  const int factor = vc.patch * vc.merge;
  u::VideoTools tools;
  std::string why;
  const bool ff = u::find_video_tools("ffmpeg", tools, why);
  if (!ff) std::printf("  video: %s (decoder cases skipped)\n", why.c_str());
  for (const auto &kv : idx.obj) {
    const std::string &name = kv.first;
    const json::Value &e = kv.second;
    const int n = static_cast<int>(e.get("n")->num), H = static_cast<int>(e.get("h")->num), W = static_cast<int>(e.get("w")->num);
    const double fps = e.get("fps")->num;
    const int64_t total = static_cast<int64_t>(e.get("total")->num);
    std::vector<int64_t> want;
    for (const auto &x : e.get("indices")->arr) want.push_back(static_cast<int64_t>(x.num));
    CHECK(m::video_frame_indices(total, fps, vc.video_fps, vc.video_min_frames, vc.video_max_frames) == want, "%s: frame indices", name.c_str());
    // sizing + pixels from the reference's own frames
    int rh = 0, rw = 0;
    CHECK(m::video_smart_resize(n, H, W, vc.temporal, factor, vc.video_min_pixels, vc.video_max_pixels, rh, rw, err), "%s: resize %s", name.c_str(), err.c_str());
    const auto &grid = e.get("grid")->arr;
    const int gt = static_cast<int>(grid[0].num), gh = static_cast<int>(grid[1].num), gw = static_cast<int>(grid[2].num);
    CHECK(rh / vc.patch == gh && rw / vc.patch == gw && (n + 1) / 2 == gt, "%s: grid %dx%d vs %dx%d", name.c_str(), rh / vc.patch, rw / vc.patch, gh, gw);
    const std::string fr = read_all(dir + "/" + name + ".frames.u8");
    CHECK(fr.size() == static_cast<size_t>(n) * H * W * 3, "%s: frames file", name.c_str());
    std::vector<u::Image> rs(static_cast<size_t>(n));
    for (int f = 0; f < n; ++f) {
      u::Image im;
      im.w = W, im.h = H;
      im.rgb.assign(fr.begin() + static_cast<std::ptrdiff_t>(static_cast<size_t>(f) * H * W * 3), fr.begin() + static_cast<std::ptrdiff_t>(static_cast<size_t>(f + 1) * H * W * 3));
      m::resize_bicubic_u8(im, rw, rh, rs[static_cast<size_t>(f)]);
    }
    const std::string pv = read_all(dir + "/" + name + ".pv.u8");
    const int pd = vc.patch_dim();
    size_t bad = 0, rows = 0;
    std::vector<uint16_t> lut(256);
    for (int v = 0; v < 256; ++v) {  // bf16 of (v - 127.5) / 127.5, as the reference's values rounded to the tower's dtype
      float x = (static_cast<float>(v) - 127.5f) / 127.5f;
      uint32_t b;
      std::memcpy(&b, &x, 4);
      lut[static_cast<size_t>(v)] = static_cast<uint16_t>((b + 0x7fff + ((b >> 16) & 1)) >> 16);
    }
    for (int t = 0; t < gt; ++t) {
      const u::Image &f0 = rs[static_cast<size_t>(2 * t)], &f1 = rs[static_cast<size_t>(std::min(2 * t + 1, n - 1))];
      m::VisionInput in;
      m::preprocess_frames(vc, f0, f1, in);
      for (int p = 0; p < in.patches_n(); ++p, ++rows)
        for (int c = 0; c < pd; ++c)
          if (in.patches[static_cast<size_t>(p) * pd + c] != lut[static_cast<uint8_t>(pv[rows * pd + c])]) ++bad;
    }
    CHECK(rows * pd == pv.size() && bad == 0, "%s: %zu of %zu pixel values differ from the reference's", name.c_str(), bad, pv.size());
    // timestamps and the token layout (the processor's ids between the outer vision markers)
    const auto ts = m::video_timestamps(want, fps, vc.temporal);
    const auto &wts = e.get("timestamps")->arr;
    bool ts_ok = ts.size() == wts.size();
    for (size_t i = 0; ts_ok && i < ts.size(); ++i) ts_ok = m::video_timestamp_text(ts[i]) == wts[i].str;
    CHECK(ts_ok, "%s: timestamps", name.c_str());
    if (have_tok) {
      std::vector<int32_t> ids = {248053};
      const int per = gh * gw / (vc.merge * vc.merge);
      for (size_t i = 0; i < ts.size(); ++i) {
        for (int32_t t : tok.encode(m::video_timestamp_text(ts[i]))) ids.push_back(t);
        ids.push_back(248053);
        for (int k = 0; k < per; ++k) ids.push_back(248057);
        ids.push_back(248054);
      }
      ids.push_back(248054);
      std::vector<int32_t> ref;
      for (const auto &x : e.get("ids")->arr) ref.push_back(static_cast<int32_t>(x.num));
      CHECK(ids == ref, "%s: token layout (%zu vs %zu ids)", name.c_str(), ids.size(), ref.size());
    }
    // our frame source vs the reference decoder's frames
    if (ff) {
      const std::string bytes = read_all(dir + "/../vid/" + name);
      if (bytes.empty()) continue;
      u::VideoInfo vi;
      CHECK(u::probe_video(bytes, tools, 30, vi, err), "%s: probe %s", name.c_str(), err.c_str());
      CHECK(vi.total_frames == total && std::fabs(vi.fps - fps) < 1e-9, "%s: probe %lld frames @ %.3f vs %lld @ %.3f", name.c_str(),
            static_cast<long long>(vi.total_frames), vi.fps, static_cast<long long>(total), fps);
      size_t got = 0;
      double sum = 0;
      int maxd = 0, dims_bad = 0;
      size_t cnt = 0;
      CHECK(u::decode_video_frames(bytes, tools, want, 60,
                                   [&](size_t k, u::Image &&im) {
                                     if (im.w != W || im.h != H) {
                                       ++dims_bad;
                                       return true;
                                     }
                                     const uint8_t *ref = reinterpret_cast<const uint8_t *>(fr.data()) + k * static_cast<size_t>(H) * W * 3;
                                     for (size_t i = 0; i < im.rgb.size(); ++i) {
                                       const int d = std::abs(static_cast<int>(im.rgb[i]) - ref[i]);
                                       maxd = std::max(maxd, d);
                                       sum += d;
                                     }
                                     cnt += im.rgb.size();
                                     return true;
                                   },
                                   got, err),
            "%s: decode %s", name.c_str(), err.c_str());
      std::printf("  %s: %zu / %d frames decoded (%dx%d), vs the reference decoder: max |d| %d, mean |d| %.4f\n", name.c_str(), got, n, W, H, maxd,
                  cnt ? sum / static_cast<double>(cnt) : 0.0);
      CHECK(got == static_cast<size_t>(n) && dims_bad == 0, "%s: %zu frames, %d of another size", name.c_str(), got, dims_bad);
    }
  }
}

// A long clip's frame set in one decode: the sampler takes up to 768 frames per video, more than ffmpeg parses as one
// flat select sum (~100 terms), so the decoder must deliver all of them. The clip is made here by the found ffmpeg.
void long_video_cases() {
  u::VideoTools tools;
  std::string why;
  if (!u::find_video_tools("ffmpeg", tools, why)) {
    std::printf("  long video: %s (skipped)\n", why.c_str());
    return;
  }
  constexpr int kFrames = 800, kMostSampled = 768;  // kMostSampled: video_max_frames
  const std::string clip = (std::filesystem::temp_directory_path() / ("xen-long-video-" + std::to_string(::getpid()) + ".mp4")).string();
  const std::string cmd = "'" + tools.ffmpeg + "' -nostdin -loglevel error -y -f lavfi -i testsrc2=size=32x32:rate=25:duration=" +
                          std::to_string(kFrames / 25) + " -c:v mpeg4 -q:v 5 '" + clip + "'";
  const int rc = std::system(cmd.c_str());
  const std::string bytes = read_all(clip);
  std::filesystem::remove(clip);
  CHECK(rc == 0 && !bytes.empty(), "long video: ffmpeg made the %d-frame clip (exit %d)", kFrames, rc);
  if (bytes.empty()) return;
  std::vector<int64_t> want;  // evenly spread over the clip, as the sampler spreads them
  for (int i = 0; i < kMostSampled; ++i) want.push_back(static_cast<int64_t>(i) * (kFrames - 1) / (kMostSampled - 1));
  size_t got = 0;
  int dims_bad = 0;
  std::string err;
  const bool ok = u::decode_video_frames(bytes, tools, want, 60,
                                         [&](size_t, u::Image &&im) {
                                           dims_bad += im.w != 32 || im.h != 32;
                                           return true;
                                         },
                                         got, err);
  CHECK(ok && got == want.size() && dims_bad == 0, "long video: %zu of %zu frames decoded in one pass (%s)", got, want.size(), err.c_str());
}

// The 27B's M-RoPE over text + an image + a video (the processor's layout: outer markers, per frame a timestamp and its
// markers): each frame of a video takes the positions of an image of its grid (the reference splits a video into frames),
// Mrope:row over the store's per-frame entries == HF get_rope_index (types 1 image, 2 video).
void mrope_cases(const std::string &dir) {
  const std::string it = read_all(dir + "/index.json");
  if (it.empty()) return;
  json::Value idx;
  std::string err;
  CHECK(json::parse(it, idx, &err), "mrope index: %s", err.c_str());
  const int n = static_cast<int>(idx.get("n")->num), merge = static_cast<int>(idx.get("merge")->num);
  m::Mrope mr;
  for (int i = 0; i < 3; ++i) mr.sec[i] = static_cast<int>(idx.get("mrope_section")->arr[static_cast<size_t>(i)].num);
  const int half = mr.sec[0] + mr.sec[1] + mr.sec[2];
  const std::string pb = read_all(dir + "/pos.i32"), sb = read_all(dir + "/sel.i32"), tb = read_all(dir + "/types.i32");
  CHECK(pb.size() == static_cast<size_t>(3) * n * 4 && sb.size() == static_cast<size_t>(n) * half * 4 && tb.size() == static_cast<size_t>(n) * 4, "mrope sizes");
  if (g_failures) return;
  const int32_t *gp = reinterpret_cast<const int32_t *>(pb.data()), *gs = reinterpret_cast<const int32_t *>(sb.data()),
                *ty = reinterpret_cast<const int32_t *>(tb.data());
  const auto &ig = idx.get("image_grid")->arr, &vg = idx.get("video_grid")->arr;
  int64_t delta = 0, base = 1000000;
  size_t bad = 0, bad_sel = 0, frames = 0;
  m::MropeImage cur;
  int k = 0;
  for (int P = 0; P < n; ++P) {
    const m::MropeImage *im = nullptr;
    if (ty[P] != 0) {
      if (P == 0 || ty[P - 1] != ty[P]) {  // a new image / frame entry
        const auto &g = ty[P] == 1 ? ig : vg;
        cur.gh = static_cast<int>(g[1].num) / merge, cur.gw = static_cast<int>(g[2].num) / merge, cur.tokens = cur.gh * cur.gw, cur.base = base;
        base += cur.tokens;
        k = 0;
        frames += ty[P] == 2;
      }
      im = &cur;
    }
    int64_t thw[3];
    m::Mrope::row(P, delta, im, im ? k++ : 0, thw);
    for (int a = 0; a < 3; ++a) bad += thw[a] != gp[static_cast<size_t>(a) * n + P];
    for (int i = 0; i < half; ++i) bad_sel += thw[mr.pick(i)] != gs[static_cast<size_t>(P) * half + i];
  }
  CHECK(bad == 0 && bad_sel == 0 && delta == static_cast<int64_t>(idx.get("delta")->num),
        "mrope video: %zu positions, %zu picks differ; delta %lld vs %lld", bad, bad_sel, static_cast<long long>(delta), static_cast<long long>(idx.get("delta")->num));
  std::printf("  mrope: %d rows (an image + a video of %zu frames) == the reference's positions, delta %lld\n", n, frames, static_cast<long long>(delta));
}

}  // namespace

int main(int argc, char **argv) {
  const std::string dir = argc > 2 ? argv[2] : std::string(XENITHIUM_SOURCE_DIR) + "/tests/data/vision/video";  // argv[2]: another golden dir
  std::string tokdir = argc > 1 ? argv[1] : "";
  if (tokdir.empty()) {
    const char *home = std::getenv("HOME");
    tokdir = home ? std::string(home) + "/models/qwen38-w4a16" : "";
  }
  unit_cases();
  golden_cases(dir, tokdir);
  mrope_cases(dir + "/mrope");
  long_video_cases();
  std::printf("%d checks, %d failures\n", g_checks, g_failures);
  return g_failures == 0 ? 0 : 1;
}
