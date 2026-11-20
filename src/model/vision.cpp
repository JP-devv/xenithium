// Xenithium · the vision input on the host (xenithium/model/vision.hpp): the tower's configuration, the model's
// image processor (smart_resize, the uint8 antialiased bicubic resize, normalisation, patchify) and the per-patch
// position / rotary tables.
#include "xenithium/model/vision.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "xenithium/kernels/pack.hpp"

// the resize weights and the tables keep the image processor's operation order: no fused multiply-adds
#pragma clang fp contract(off)

namespace xenithium::model {

namespace {

using kernels::bf16_bits;

// ---- the uint8 resize (antialiased bicubic, a = -0.5) ----------------------------------------------------------------

double cubic_aa(double x) {
  constexpr double a = -0.5;
  x = std::fabs(x);
  if (x < 1.0) return ((a + 2) * x - (a + 3)) * x * x + 1;
  if (x < 2.0) return ((a * x - 5 * a) * x + 8 * a) * x - 4 * a;
  return 0.0;
}

struct Taps {
  std::vector<int> xmin, xsize;
  std::vector<int16_t> w;  // [out][ksize]
  int ksize = 0;
  int prec = 0;
};

// Double weights per output sample, normalised; then int16 at the precision the largest weight allows (< 2^15 at one
// more bit).
Taps taps(int in, int out) {
  Taps t;
  const double scale = static_cast<double>(in) / out;
  const double support = scale >= 1.0 ? 2.0 * scale : 2.0;
  const int max_size = static_cast<int>(std::ceil(support)) * 2 + 1;
  const double invscale = scale >= 1.0 ? 1.0 / scale : 1.0;
  t.ksize = max_size;
  t.xmin.resize(static_cast<size_t>(out));
  t.xsize.resize(static_cast<size_t>(out));
  std::vector<double> wd(static_cast<size_t>(out) * max_size, 0.0);
  double wt_max = 0.0;
  for (int i = 0; i < out; ++i) {
    const double center = scale * (i + 0.5);
    const int64_t xmin = std::max<int64_t>(static_cast<int64_t>(center - support + 0.5), 0);
    int64_t xsize = std::min<int64_t>(static_cast<int64_t>(center + support + 0.5), in) - xmin;
    xsize = std::clamp<int64_t>(xsize, 0, max_size);
    double *w = wd.data() + static_cast<size_t>(i) * max_size;
    double total = 0.0;
    for (int64_t j = 0; j < xsize; ++j) {
      w[j] = cubic_aa((static_cast<double>(j + xmin) - center + 0.5) * invscale);
      total += w[j];
    }
    double mx = 0.0;
    if (total != 0.0)
      for (int64_t j = 0; j < xsize; ++j) {
        w[j] /= total;
        mx = std::max(mx, w[j]);
      }
    wt_max = std::max(wt_max, mx);
    t.xmin[static_cast<size_t>(i)] = static_cast<int>(xmin);
    t.xsize[static_cast<size_t>(i)] = static_cast<int>(xsize);
  }
  int prec = 0;
  for (prec = 0; prec < 22; ++prec) {
    const int next = static_cast<int>(0.5 + wt_max * (1 << (prec + 1)));
    if (next >= (1 << 15)) break;
  }
  t.prec = prec;
  t.w.resize(wd.size());
  for (size_t k = 0; k < wd.size(); ++k) {
    const double v = wd[k] * (1 << prec);
    t.w[k] = static_cast<int16_t>(v < 0 ? static_cast<int>(-0.5 + v) : static_cast<int>(0.5 + v));
  }
  return t;
}

inline uint8_t clip_shift(int32_t s, int prec) {
  const int32_t v = s >> prec;
  return static_cast<uint8_t>(v < 0 ? 0 : v > 255 ? 255 : v);
}

// ---- small helpers ---------------------------------------------------------------------------------------------------

std::string read_file(const std::string &p) {
  std::ifstream f(p, std::ios::binary);
  if (!f) return {};
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

// torch.linspace(0, end, steps) in fp32: the first half start + step x i, the second half end - step x (steps - 1 - i)
float linspace_at(float end, int steps, int i) {
  if (steps == 1) return 0.0f;
  const float step = end / static_cast<float>(steps - 1);
  return i < steps / 2 ? 0.0f + step * static_cast<float>(i) : end - step * static_cast<float>(steps - i - 1);
}

}  // namespace

VisionConfig VisionConfig::parse(const ModelConfig &cfg) {
  VisionConfig vc;
  const json::Value &root = cfg.root();
  const json::Value *v = root.get("vision_config");
  if (!v || !v->is_object()) return vc;
  vc.present = true;
  vc.depth = static_cast<int>(ModelConfig::need_int(*v, "depth"));
  vc.hidden = static_cast<int>(ModelConfig::need_int(*v, "hidden_size"));
  vc.heads = static_cast<int>(ModelConfig::need_int(*v, "num_heads"));
  vc.inter = static_cast<int>(ModelConfig::need_int(*v, "intermediate_size"));
  vc.out_hidden = static_cast<int>(ModelConfig::need_int(*v, "out_hidden_size"));
  vc.patch = static_cast<int>(ModelConfig::int_or(*v, "patch_size", 16));
  vc.merge = static_cast<int>(ModelConfig::int_or(*v, "spatial_merge_size", 2));
  vc.temporal = static_cast<int>(ModelConfig::int_or(*v, "temporal_patch_size", 2));
  vc.in_ch = static_cast<int>(ModelConfig::int_or(*v, "in_channels", 3));
  vc.num_pos = static_cast<int>(ModelConfig::need_int(*v, "num_position_embeddings"));
  vc.act = ModelConfig::str_or(*v, "hidden_act", "gelu_pytorch_tanh");
  if (const json::Value *d = v->get("deepstack_visual_indexes"); d && d->is_array())
    for (const auto &x : d->arr) vc.deepstack.push_back(static_cast<int>(x.num));
  vc.image_token = static_cast<int>(ModelConfig::int_or(root, "image_token_id", -1));
  vc.vision_start = static_cast<int>(ModelConfig::int_or(root, "vision_start_token_id", -1));
  vc.vision_end = static_cast<int>(ModelConfig::int_or(root, "vision_end_token_id", -1));
  vc.video_token = static_cast<int>(ModelConfig::int_or(root, "video_token_id", -1));
  auto bad = [](const std::string &m) { throw std::runtime_error("config.json vision_config: " + m); };
  if (!vc.deepstack.empty()) bad("deepstack_visual_indexes is not empty (deepstack features are not served)");
  if (vc.act != "gelu_pytorch_tanh") bad("hidden_act " + vc.act + " (gelu_pytorch_tanh is served)");
  if (vc.heads <= 0 || vc.hidden % vc.heads || vc.head_dim() % 4) bad("hidden_size / num_heads is not a multiple of 4");
  if (vc.in_ch != 3) bad("in_channels must be 3");
  const int side = static_cast<int>(std::lround(std::sqrt(static_cast<double>(vc.num_pos))));
  if (side * side != vc.num_pos) bad("num_position_embeddings is not a square");
  if (vc.image_token < 0) bad("image_token_id missing");
  // the image processor's settings
  const std::string pp = read_file(cfg.dir() + "/preprocessor_config.json");
  if (!pp.empty()) {
    json::Value p;
    std::string err;
    if (!json::parse(pp, p, &err)) throw std::runtime_error("preprocessor_config.json: " + err);
    if (const json::Value *m = p.get("image_mean"); m && m->is_array() && m->arr.size() == 3)
      for (int c = 0; c < 3; ++c) vc.mean[c] = static_cast<float>(m->arr[static_cast<size_t>(c)].num);
    if (const json::Value *s = p.get("image_std"); s && s->is_array() && s->arr.size() == 3)
      for (int c = 0; c < 3; ++c) vc.std[c] = static_cast<float>(s->arr[static_cast<size_t>(c)].num);
    vc.rescale = ModelConfig::num_or(p, "rescale_factor", vc.rescale);
    if (const json::Value *sz = p.get("size"); sz && sz->is_object()) {
      vc.min_pixels = ModelConfig::int_or(*sz, "shortest_edge", vc.min_pixels);
      vc.max_pixels = ModelConfig::int_or(*sz, "longest_edge", vc.max_pixels);
    }
    vc.min_pixels = ModelConfig::int_or(p, "min_pixels", vc.min_pixels);
    vc.max_pixels = ModelConfig::int_or(p, "max_pixels", vc.max_pixels);
    if (ModelConfig::int_or(p, "patch_size", vc.patch) != vc.patch || ModelConfig::int_or(p, "merge_size", vc.merge) != vc.merge ||
        ModelConfig::int_or(p, "temporal_patch_size", vc.temporal) != vc.temporal)
      throw std::runtime_error("preprocessor_config.json: patch / merge / temporal sizes differ from config.json's vision_config");
  }
  // the video processor's settings
  const std::string vp = read_file(cfg.dir() + "/video_preprocessor_config.json");
  if (!vp.empty()) {
    json::Value p;
    std::string err;
    if (!json::parse(vp, p, &err)) throw std::runtime_error("video_preprocessor_config.json: " + err);
    if (const json::Value *sz = p.get("size"); sz && sz->is_object()) {
      vc.video_min_pixels = ModelConfig::int_or(*sz, "shortest_edge", vc.video_min_pixels);
      vc.video_max_pixels = ModelConfig::int_or(*sz, "longest_edge", vc.video_max_pixels);
    }
    vc.video_fps = ModelConfig::num_or(p, "fps", vc.video_fps);
    vc.video_min_frames = static_cast<int>(ModelConfig::int_or(p, "min_frames", vc.video_min_frames));
    vc.video_max_frames = static_cast<int>(ModelConfig::int_or(p, "max_frames", vc.video_max_frames));
  }
  return vc;
}

bool smart_resize(int height, int width, int factor, int64_t min_pixels, int64_t max_pixels, int &oh, int &ow, std::string &err) {
  if (height <= 0 || width <= 0) return err = "empty image", false;
  if (static_cast<double>(std::max(height, width)) / std::min(height, width) > 200) {
    err = "image aspect ratio must be smaller than 200, got " + std::to_string(static_cast<double>(std::max(height, width)) / std::min(height, width));
    return false;
  }
  // Python round: half to even
  auto pyround = [](double x) { return std::nearbyint(x); };
  int64_t h = static_cast<int64_t>(pyround(static_cast<double>(height) / factor)) * factor;
  int64_t w = static_cast<int64_t>(pyround(static_cast<double>(width) / factor)) * factor;
  if (h * w > max_pixels) {
    const double beta = std::sqrt(static_cast<double>(height) * width / static_cast<double>(max_pixels));
    h = std::max<int64_t>(factor, static_cast<int64_t>(std::floor(height / beta / factor)) * factor);
    w = std::max<int64_t>(factor, static_cast<int64_t>(std::floor(width / beta / factor)) * factor);
  } else if (h * w < min_pixels) {
    const double beta = std::sqrt(static_cast<double>(min_pixels) / (static_cast<double>(height) * width));
    h = static_cast<int64_t>(std::ceil(height * beta / factor)) * factor;
    w = static_cast<int64_t>(std::ceil(width * beta / factor)) * factor;
  }
  oh = static_cast<int>(h), ow = static_cast<int>(w);
  return true;
}

void resize_bicubic_u8(const util::Image &in, int ow, int oh, util::Image &out) {
  const int iw = in.w, ih = in.h;
  util::Image tmp;
  const util::Image *src = &in;
  if (ow != iw) {  // horizontal pass
    const Taps t = taps(iw, ow);
    tmp.w = ow, tmp.h = ih;
    tmp.rgb.resize(static_cast<size_t>(ow) * ih * 3);
    for (int y = 0; y < ih; ++y) {
      const uint8_t *row = in.rgb.data() + static_cast<size_t>(y) * iw * 3;
      uint8_t *o = tmp.rgb.data() + static_cast<size_t>(y) * ow * 3;
      for (int x = 0; x < ow; ++x) {
        const int16_t *k = t.w.data() + static_cast<size_t>(x) * t.ksize;
        const int x0 = t.xmin[static_cast<size_t>(x)], n = t.xsize[static_cast<size_t>(x)];
        for (int c = 0; c < 3; ++c) {
          int32_t s = 1 << (t.prec - 1);
          for (int j = 0; j < n; ++j) s += static_cast<int32_t>(row[(x0 + j) * 3 + c]) * k[j];
          o[x * 3 + c] = clip_shift(s, t.prec);
        }
      }
    }
    src = &tmp;
  }
  if (oh != ih) {  // vertical pass
    const Taps t = taps(ih, oh);
    out.w = ow, out.h = oh;
    out.rgb.resize(static_cast<size_t>(ow) * oh * 3);
    const int rw = ow * 3;
    for (int y = 0; y < oh; ++y) {
      const int16_t *k = t.w.data() + static_cast<size_t>(y) * t.ksize;
      const int y0 = t.xmin[static_cast<size_t>(y)], n = t.xsize[static_cast<size_t>(y)];
      uint8_t *o = out.rgb.data() + static_cast<size_t>(y) * rw;
      for (int i = 0; i < rw; ++i) {
        int32_t s = 1 << (t.prec - 1);
        for (int j = 0; j < n; ++j) s += static_cast<int32_t>(src->rgb[static_cast<size_t>(y0 + j) * rw + i]) * k[j];
        o[i] = clip_shift(s, t.prec);
      }
    }
  } else if (src == &tmp) {
    out = std::move(tmp);
  } else {
    out = in;
  }
}

int vision_tokens(const VisionConfig &vc, int height, int width, int64_t max_pixels, std::string &err) {
  int h, w;
  if (!smart_resize(height, width, vc.patch * vc.merge, vc.min_pixels, max_pixels > 0 ? max_pixels : vc.max_pixels, h, w, err)) return -1;
  return (h / vc.patch) * (w / vc.patch) / (vc.merge * vc.merge);
}

namespace {

// Steps 3-4 and the tables for frames[T] of the same H x W (an image: the one resized frame twice).
void fill_input(const VisionConfig &vc, const util::Image *const *frames, VisionInput &out) {
  const int H = frames[0]->h, W = frames[0]->w;
  const int P = vc.patch, M = vc.merge, T = vc.temporal, C = vc.in_ch;
  const int gh = H / P, gw = W / P;
  out.gh = gh, out.gw = gw, out.height = H, out.width = W;
  // rescale + normalise folded into one step: (v - mean / rescale) / (std / rescale), fp32
  float m[3], s[3];
  for (int c = 0; c < 3; ++c) {
    m[c] = static_cast<float>(vc.mean[c] * (1.0 / vc.rescale));
    s[c] = static_cast<float>(vc.std[c] * (1.0 / vc.rescale));
  }
  uint16_t lut[3][256];
  for (int c = 0; c < 3; ++c)
    for (int v = 0; v < 256; ++v) lut[c][v] = bf16_bits((static_cast<float>(v) - m[c]) / s[c]);
  const int pd = vc.patch_dim();
  const int n = gh * gw;
  out.patches.assign(static_cast<size_t>(n) * pd, 0);
  out.pos_idx.assign(static_cast<size_t>(n) * 4, 0);
  out.pos_w.assign(static_cast<size_t>(n) * 4, 0);
  const int hd = vc.head_dim(), nf = hd / 4;  // frequencies per axis
  out.rope_cos.assign(static_cast<size_t>(n) * (hd / 2), 0.0f);
  out.rope_sin.assign(static_cast<size_t>(n) * (hd / 2), 0.0f);
  // the rotary frequencies: 1 / theta^(arange(0, hd / 2, 2) / (hd / 2)), fp32
  std::vector<float> inv(static_cast<size_t>(nf));
  for (int i = 0; i < nf; ++i) inv[static_cast<size_t>(i)] = 1.0f / std::pow(10000.0f, static_cast<float>(2 * i) / static_cast<float>(hd / 2));
  const int side = static_cast<int>(std::lround(std::sqrt(static_cast<double>(vc.num_pos))));
  const float end = static_cast<float>(side - 1);
  int p = 0;
  for (int bh = 0; bh < gh / M; ++bh)
    for (int bw = 0; bw < gw / M; ++bw)
      for (int i = 0; i < M; ++i)
        for (int j = 0; j < M; ++j, ++p) {
          const int r = bh * M + i, cidx = bw * M + j;
          uint16_t *dst = out.patches.data() + static_cast<size_t>(p) * pd;
          for (int c = 0; c < C; ++c)
            for (int t = 0; t < T; ++t) {
              const uint8_t *px = frames[t]->rgb.data();
              for (int y = 0; y < P; ++y)
                for (int x = 0; x < P; ++x) dst[((c * T + t) * P + y) * P + x] = lut[c][px[(static_cast<size_t>(r * P + y) * W + (cidx * P + x)) * 3 + c]];
            }
          // the learned position grid, bilinear (the weights in bf16)
          const float hi = linspace_at(end, gh, r), wi = linspace_at(end, gw, cidx);
          const int hf = static_cast<int>(hi), wf = static_cast<int>(wi);
          const int hc = std::min(hf + 1, side - 1), wc = std::min(wf + 1, side - 1);
          const float dh = hi - static_cast<float>(hf), dw = wi - static_cast<float>(wf);
          const float w11 = dh * dw, w10 = dh - w11, w01 = dw - w11, w00 = 1.0f - dh - w01;
          int32_t *ix = out.pos_idx.data() + static_cast<size_t>(p) * 4;
          uint16_t *wt = out.pos_w.data() + static_cast<size_t>(p) * 4;
          ix[0] = hf * side + wf, ix[1] = hf * side + wc, ix[2] = hc * side + wf, ix[3] = hc * side + wc;
          wt[0] = bf16_bits(w00), wt[1] = bf16_bits(w01), wt[2] = bf16_bits(w10), wt[3] = bf16_bits(w11);
          float *rc = out.rope_cos.data() + static_cast<size_t>(p) * (hd / 2), *rsn = out.rope_sin.data() + static_cast<size_t>(p) * (hd / 2);
          for (int f = 0; f < nf; ++f) {
            const float ar = static_cast<float>(r) * inv[static_cast<size_t>(f)], ac = static_cast<float>(cidx) * inv[static_cast<size_t>(f)];
            rc[f] = std::cos(ar), rsn[f] = std::sin(ar);
            rc[nf + f] = std::cos(ac), rsn[nf + f] = std::sin(ac);
          }
        }
}

}  // namespace

bool preprocess(const VisionConfig &vc, const util::Image &img, int64_t max_pixels, VisionInput &out, std::string &err) {
  int H, W;
  if (!smart_resize(img.h, img.w, vc.patch * vc.merge, vc.min_pixels, max_pixels > 0 ? max_pixels : vc.max_pixels, H, W, err)) return false;
  util::Image rs;
  resize_bicubic_u8(img, W, H, rs);
  std::vector<const util::Image *> frames(static_cast<size_t>(vc.temporal), &rs);  // the image repeated in time
  fill_input(vc, frames.data(), out);
  out.video = false;
  return true;
}

void preprocess_frames(const VisionConfig &vc, const util::Image &f0, const util::Image &f1, VisionInput &out) {
  std::vector<const util::Image *> frames(static_cast<size_t>(vc.temporal), &f1);
  frames[0] = &f0;
  fill_input(vc, frames.data(), out);
  out.video = true;
}

std::vector<int64_t> video_frame_indices(int64_t total, double fps, double target_fps, int min_frames, int max_frames) {
  const double f = std::min(target_fps, 30.0);  // the fps ceiling of the video sampling
  int64_t n = static_cast<int64_t>(static_cast<double>(total) / fps * f);  // Python int: toward zero
  n = std::min<int64_t>({std::max<int64_t>(n, min_frames), max_frames, total});
  std::vector<int64_t> idx(static_cast<size_t>(std::max<int64_t>(n, 0)));
  if (n == 1) {
    idx[0] = 0;
  } else if (n > 1) {  // numpy linspace: start + i x step, the last one exactly stop; then round half to even
    const double stop = static_cast<double>(total - 1), step = stop / static_cast<double>(n - 1);
    for (int64_t i = 0; i < n; ++i) idx[static_cast<size_t>(i)] = static_cast<int64_t>(std::nearbyint(i == n - 1 ? stop : static_cast<double>(i) * step));
  }
  return idx;
}

bool video_smart_resize(int frames, int height, int width, int temporal, int factor, int64_t min_pixels, int64_t max_pixels, int &oh, int &ow,
                        std::string &err) {
  if (frames < temporal) return err = "the video has " + std::to_string(frames) + " frame(s); at least " + std::to_string(temporal) + " are needed", false;
  if (height <= 0 || width <= 0) return err = "empty video frame", false;
  double h = height, w = width;
  if (height < factor || width < factor) {  // int(height x scale): a tiny frame is scaled up first
    const double scale = std::max(static_cast<double>(factor) / height, static_cast<double>(factor) / width);
    h = static_cast<double>(static_cast<int64_t>(height * scale));
    w = static_cast<double>(static_cast<int64_t>(width * scale));
  }
  if (std::max(h, w) / std::min(h, w) > 200) return err = "video aspect ratio must be smaller than 200", false;
  int64_t hb = static_cast<int64_t>(std::nearbyint(h / factor)) * factor;
  int64_t wb = static_cast<int64_t>(std::nearbyint(w / factor)) * factor;
  const int64_t tb = static_cast<int64_t>(std::nearbyint(static_cast<double>(frames) / temporal)) * temporal;
  if (tb * hb * wb > max_pixels) {
    const double beta = std::sqrt(static_cast<double>(frames) * h * w / static_cast<double>(max_pixels));
    hb = std::max<int64_t>(factor, static_cast<int64_t>(std::floor(h / beta / factor)) * factor);
    wb = std::max<int64_t>(factor, static_cast<int64_t>(std::floor(w / beta / factor)) * factor);
  } else if (tb * hb * wb < min_pixels) {
    const double beta = std::sqrt(static_cast<double>(min_pixels) / (static_cast<double>(frames) * h * w));
    hb = static_cast<int64_t>(std::ceil(h * beta / factor)) * factor;
    wb = static_cast<int64_t>(std::ceil(w * beta / factor)) * factor;
  }
  oh = static_cast<int>(hb), ow = static_cast<int>(wb);
  return true;
}

std::vector<double> video_timestamps(const std::vector<int64_t> &indices, double fps, int temporal) {
  std::vector<int64_t> idx = indices;
  while (!idx.empty() && idx.size() % static_cast<size_t>(temporal)) idx.push_back(idx.back());
  std::vector<double> out;
  for (size_t i = 0; i < idx.size(); i += static_cast<size_t>(temporal))
    out.push_back((static_cast<double>(idx[i]) / fps + static_cast<double>(idx[i + temporal - 1]) / fps) / 2);
  return out;
}

std::string video_timestamp_text(double seconds) {
  char b[64];
  std::snprintf(b, sizeof b, "<%.1f seconds>", seconds);
  return b;
}

void Mrope::row(int64_t P, int64_t &delta, const MropeImage *img, int k, int64_t thw[3]) {
  if (!img) {
    thw[0] = thw[1] = thw[2] = P + delta;
    return;
  }
  const int64_t s = P - k + delta;
  thw[0] = s;
  thw[1] = s + k / img->gw;
  thw[2] = s + k % img->gw;
  if (k == img->tokens - 1) delta += std::max(img->gh, img->gw) - img->tokens;
}

void Mrope::compose(const uint16_t *table, int rot, const int64_t thw[3], uint16_t *out) const {
  const int half = rot / 2;
  for (int i = 0; i < half; ++i) {
    const uint16_t *src = table + static_cast<size_t>(thw[pick(i)]) * rot;
    out[i] = src[i];
    out[half + i] = src[half + i];
  }
}

}  // namespace xenithium:model
