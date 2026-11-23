// Xenithium · video input: frames from a video file through an ffmpeg binary run as a subprocess (the engine
// links no video decoder). The file's bytes go to the child through a memory file (memfd, seekable: MP4 files with the
// index at the end work), ffprobe reads the stream's header facts, ffmpeg decodes only the frames asked for (a select
// filter) and writes them as binary PPM (8-bit RGB) to a pipe. Rotation metadata is applied by ffmpeg.
// Everything runs under one deadline; the child is killed when it passes.
#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

#include "xenithium/util/image.hpp"

namespace xenithium::util {

struct VideoTools {
  std::string ffmpeg, ffprobe;  // resolved executable paths
};
// The name the server's video decoder setting defaults to.
inline constexpr std::string_view kDefaultFfmpeg = "ffmpeg";
// Where ffmpeg is usually installed besides PATH: the system prefixes, then static builds unpacked under ~/bin,
// ~/.local/bin or ~/.hermes/tools/<name>/bin. Paths only; whether one exists is the caller's check.
std::vector<std::string> ffmpeg_install_paths();
// The ffmpeg executable (a path, or a name looked up on PATH; the default name also in ffmpeg_install_paths(), so a
// static build off PATH is found) and its ffprobe (beside it, else on PATH): true + the resolved paths, or false + why
// (the server then refuses video parts with that reason).
bool find_video_tools(const std::string &ffmpeg, VideoTools &out, std::string &why);

struct VideoInfo {
  int width = 0, height = 0;  // coded size (before rotation)
  int rotation = 0;  // display rotation in degrees (0, 90, 180, 270)
  int64_t total_frames = 0;  // the header's frame count, else round(duration x fps)
  double fps = 0, duration = 0;
  std::string codec;
};
// ffprobe on the bytes. false + err (a message a 400 can carry) when there is no video stream or it cannot be read.
bool probe_video(const std::string &bytes, const VideoTools &tools, double timeout_s, VideoInfo &out, std::string &err);

// Decodes the frames at `indices` (ascending, 0-based decode order) as RGB. on_frame(k, image) gets the k-th of them in
// order (return false to stop). delivered = how many arrived (fewer when the stream ends early). false + err on a
// decoder failure or the deadline.
bool decode_video_frames(const std::string &bytes, const VideoTools &tools, const std::vector<int64_t> &indices, double timeout_s,
                         const std::function<bool(size_t, Image &&)> &on_frame, size_t &delivered, std::string &err);

// The media type of a data: URL ("video/mp4" from "data:video/mp4;base64..."); empty when not a data: URL.
std::string data_url_type(const std::string &url);

}  // namespace xenithium:util
