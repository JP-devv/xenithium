// Xenithium · the Qwen3-VL-family vision input on the host: the tower's configuration (config.json `vision_config,
// preprocessor_config.json) and the image processor that runs before the tower:
// 1. smart_resize: both sides to multiples of patch x merge (32), the pixel count clamped to [min_pixels, max_pixels]
// by one scale factor (aspect ratio kept);
// 2. the resize itself: antialiased bicubic (a = -0.5) on 8-bit samples, horizontal pass then vertical, int16
// fixed-point weights (precision from the largest weight), a uint8 result per pass
// (tests/test_vision_host.cpp);
// 3. rescale + normalise: (v - 127.5) / 127.5 in fp32 (mean 0.5, std 0.5, 1 / 255 folded in),
// rounded to bf16 (the tower's input dtype);
// 4. patchify: temporal_patch_size identical frames, patches of [C][T][P][P] values, in merge order (each 2 x 2 group
// of patches consecutive: rows (gh / 2, gw / 2, 2, 2)).
// Also the per-patch tables the tower takes from the host: the learned position grid (num_position_embeddings =
// side x side) sampled bilinearly at linspace(0, side - 1, gh) x linspace(0, side - 1, gw) (4 indices + 4 bf16 weights
// per patch) and the 2-D rotary angles (row and column, theta 1e4, head_dim / 4 frequencies each).
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "xenithium/model/config.hpp"
#include "xenithium/util/image.hpp"

namespace xenithium::model {

struct VisionConfig {
  bool present = false;
  int depth = 0, hidden = 0, heads = 0, inter = 0, out_hidden = 0;
  int patch = 16, merge = 2, temporal = 2, in_ch = 3, num_pos = 0;
  std::string act;  // hidden_act of the blocks' MLP (gelu_pytorch_tanh)
  std::vector<int> deepstack;  // deepstack_visual_indexes (must be empty: not served)
  int image_token = -1, vision_start = -1, vision_end = -1;
  int video_token = -1;
  // the video processor (video_preprocessor_config.json): pixels over all sampled frames, frames per second
  // sampled, the frame count's bounds
  int64_t video_min_pixels = 4096, video_max_pixels = 25165824;
  double video_fps = 2;
  int video_min_frames = 4, video_max_frames = 768;
  float mean[3] = {0.5f, 0.5f, 0.5f}, std[3] = {0.5f, 0.5f, 0.5f};
  double rescale = 1.0 / 255.0;
  int64_t min_pixels = 65536, max_pixels = 16777216;

  int head_dim() const { return hidden / heads; }
  int patch_dim() const { return in_ch * temporal * patch * patch; }
  int merged_dim() const { return hidden * merge * merge; }
  // config.json (+ preprocessor_config.json beside it); present = false without a vision_config. Throws on a tower this
  // engine does not run (deepstack levels, another activation, head_dim not a multiple of 4).
  static VisionConfig parse(const ModelConfig &cfg);
};

// One preprocessed image.
struct VisionInput {
  int gh = 0, gw = 0;  // patches per column / row (grid_thw = (1, gh, gw))
  int height = 0, width = 0;  // the resized image
  std::vector<uint16_t> patches;  // bf16 [gh x gw][patch_dim], merge order
  std::vector<int32_t> pos_idx;  // [gh x gw][4] rows of the position table (merge order)
  std::vector<uint16_t> pos_w;  // bf16 [gh x gw][4] their weights
  std::vector<float> rope_cos, rope_sin;  // [gh x gw][head_dim / 2] of the angles: head_dim / 4 row frequencies, then
                                          // the column ones (fp32 cos / sin of fp32 angle = position x inv_freq)
  bool video = false;  // a video's temporal patch (two frames), named by the video token
  int patches_n() const { return gh * gw; }
  int tokens(int merge) const { return gh * gw / (merge * merge); }
};

// smart_resize of the image processor (factor = patch x merge). false + err for an aspect ratio over 200.
bool smart_resize(int height, int width, int factor, int64_t min_pixels, int64_t max_pixels, int &out_h, int &out_w, std::string &err);
// Antialiased bicubic resize of 8-bit RGB.
void resize_bicubic_u8(const util::Image &in, int out_w, int out_h, util::Image &out);
// Steps 1-4 plus the tables. max_pixels > 0 overrides the config's (the server's cap). false + err when the image cannot
// be processed (aspect ratio).
bool preprocess(const VisionConfig &vc, const util::Image &img, int64_t max_pixels, VisionInput &out, std::string &err);
// One temporal patch of a video: two frames already resized to the same H x W (multiples of patch x merge),
// patchified with the frames as the two temporal slices; the tables as an image of that size. video = true.
void preprocess_frames(const VisionConfig &vc, const util::Image &f0, const util::Image &f1, VisionInput &out);

// Video sampling and sizing (the Qwen3-VL video loader and video processor):
// frames: n = int(total / fps x min(target_fps, 30)), clamped to [min_frames, min(max_frames, total)], at
// round(linspace(0, total - 1, n)) (half to even, numpy's linspace);
// size: the video smart_resize over all n frames (t rounded to the temporal patch, the pixel budget over t x h x w);
// timestamps: per temporal patch, the mean of its two frames' index / fps (the last frame repeated for an odd n), shown
// to the model as "<%.1f seconds>" before each frame's tokens.
std::vector<int64_t> video_frame_indices(int64_t total, double fps, double target_fps, int min_frames, int max_frames);
bool video_smart_resize(int frames, int height, int width, int temporal, int factor, int64_t min_pixels, int64_t max_pixels, int &out_h,
                        int &out_w, std::string &err);
std::vector<double> video_timestamps(const std::vector<int64_t> &indices, double fps, int temporal);
std::string video_timestamp_text(double seconds);

// The tokens an image of this size becomes (no pixels touched); -1 + err as preprocess.
int vision_tokens(const VisionConfig &vc, int height, int width, int64_t max_pixels, std::string &err);

// qwen35: the text model's interleaved M-RoPE.
// A sequence's rows get 3-D positions (t, h, w): a text row at sequence position P is (P + d) on all three, d the
// sequence's delta (0 until its first image); the k-th token of an image of gh x gw merged tokens is at
// s = P - k + d: (s, s + k / gw, s + k % gw), and after its last token d += max(gh, gw) - tokens (the next text
// continues at s + max(gh, gw)). d only ever falls, so a text row's position P + d is a row of the 1-D table.
struct MropeImage {
  int64_t base = 0;  // the image's first token id
  int tokens = 0, gh = 0, gw = 0;  // merged grid (tokens = gh x gw)
};
struct Mrope {
  int sec[3] = {0, 0, 0};  // mrope_section (t, h, w frequencies; half the rotary dims in all)
  // row at sequence position P (img: the image the row's id names, k = id - base; null: a text row); moves delta past an
  // image's last token
  static void row(int64_t P, int64_t &delta, const MropeImage *img, int k, int64_t thw[3]);
  // the position frequency i takes: h where i % 3 == 1 and i < 3 sec[1], w where i % 3 == 2 and i < 3 sec[2], else t
  int pick(int i) const { return i % 3 == 1 && i < 3 * sec[1] ? 1 : (i % 3 == 2 && i < 3 * sec[2] ? 2 : 0); }
  // the rope row (cos | sin halves of rot / 2 each, bf16) of thw, from the 1-D table's rows [positions][rot]
  void compose(const uint16_t *table, int rot, const int64_t thw[3], uint16_t *out) const;
};

}  // namespace xenithium:model
