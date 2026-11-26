// Xenithium · video frames through an ffmpeg subprocess. Written for this engine.
#include "xenithium/util/video.hpp"

#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <spawn.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <cstring>

#include "xenithium/util/json.hpp"

extern char **environ;

namespace xenithium::util {

namespace {

using Clock = std::chrono::steady_clock;

struct Fd {
  int fd = -1;
  Fd() = default;
  explicit Fd(int f) : fd(f) {}
  Fd(const Fd &) = delete;
  Fd &operator=(const Fd &) = delete;
  ~Fd() { reset(); }
  void reset() {
    if (fd >= 0) ::close(fd);
    fd = -1;
  }
};

bool executable(const std::string &p) { return ::access(p.c_str(), X_OK) == 0; }

std::string on_path(const std::string &name) {
  if (name.find('/') != std::string::npos) return executable(name) ? name : std::string();
  const char *path = std::getenv("PATH");
  if (!path) return {};
  std::string s = path;
  size_t a = 0;
  while (a <= s.size()) {
    size_t b = s.find(':', a);
    if (b == std::string::npos) b = s.size();
    const std::string dir = s.substr(a, b - a);
    if (!dir.empty() && executable(dir + "/" + name)) return dir + "/" + name;
    a = b + 1;
  }
  return {};
}

// The video's bytes in a memory file, at a descriptor >= 10 (so dup2 onto 3 in the child is a real copy).
bool memfile(const std::string &bytes, Fd &out, std::string &err) {
  const int m = ::memfd_create("xenithium-video", MFD_CLOEXEC);
  if (m < 0) return err = std::string("memfd_create: ") + std::strerror(errno), false;
  Fd tmp(m);
  size_t off = 0;
  while (off < bytes.size()) {
    const ssize_t w = ::write(m, bytes.data() + off, bytes.size() - off);
    if (w < 0 && errno == EINTR) continue;
    if (w <= 0) return err = std::string("memfd write: ") + std::strerror(errno), false;
    off += static_cast<size_t>(w);
  }
  out.fd = ::fcntl(m, F_DUPFD_CLOEXEC, 10);
  if (out.fd < 0) return err = std::string("fcntl: ") + std::strerror(errno), false;
  return true;
}

// Runs argv with the memory file as fd 3, stdout to on_out, stderr collected (its last 4 KiB). exit_ok: exited 0.
bool run(const std::vector<std::string> &argv, int memfd, Clock::time_point deadline, const std::function<bool(const char *, size_t)> &on_out,
         std::string &stderr_tail, bool &exit_ok, std::string &err) {
  int outp[2], errp[2];
  if (::pipe2(outp, O_CLOEXEC) != 0) return err = std::string("pipe: ") + std::strerror(errno), false;
  if (::pipe2(errp, O_CLOEXEC) != 0) {
    ::close(outp[0]), ::close(outp[1]);
    return err = std::string("pipe: ") + std::strerror(errno), false;
  }
  Fd out_r(outp[0]), out_w(outp[1]), err_r(errp[0]), err_w(errp[1]);
  posix_spawn_file_actions_t fa;
  posix_spawn_file_actions_init(&fa);
  posix_spawn_file_actions_addopen(&fa, 0, "/dev/null", O_RDONLY, 0);
  posix_spawn_file_actions_adddup2(&fa, out_w.fd, 1);
  posix_spawn_file_actions_adddup2(&fa, err_w.fd, 2);
  posix_spawn_file_actions_adddup2(&fa, memfd, 3);
  std::vector<char *> av;
  for (const auto &a : argv) av.push_back(const_cast<char *>(a.c_str()));
  av.push_back(nullptr);
  pid_t pid = -1;
  const int rc = ::posix_spawn(&pid, argv[0].c_str(), &fa, nullptr, av.data(), environ);
  posix_spawn_file_actions_destroy(&fa);
  out_w.reset();
  err_w.reset();
  if (rc != 0) return err = "cannot run " + argv[0] + ": " + std::strerror(rc), false;
  ::fcntl(out_r.fd, F_SETFL, O_NONBLOCK);
  ::fcntl(err_r.fd, F_SETFL, O_NONBLOCK);
  bool out_open = true, err_open = true, stopped = false, timed_out = false;
  std::vector<char> buf(1 << 20);
  while (out_open || err_open) {
    const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - Clock::now()).count();
    if (left <= 0) {
      timed_out = true;
      break;
    }
    pollfd p[2];
    int n = 0;
    if (out_open) p[n++] = {out_r.fd, POLLIN, 0};
    if (err_open) p[n++] = {err_r.fd, POLLIN, 0};
    const int r = ::poll(p, static_cast<nfds_t>(n), static_cast<int>(std::min<long long>(left, INT_MAX)));
    if (r < 0 && errno != EINTR) break;
    for (int i = 0; i < n; ++i) {
      if (!(p[i].revents & (POLLIN | POLLHUP | POLLERR))) continue;
      const bool is_out = p[i].fd == out_r.fd;
      const ssize_t got = ::read(p[i].fd, buf.data(), buf.size());
      if (got > 0) {
        if (is_out) {
          if (!stopped && !on_out(buf.data(), static_cast<size_t>(got))) {
            stopped = true;  // the consumer has what it needs: stop the child
            ::kill(pid, SIGKILL);
          }
        } else {
          stderr_tail.append(buf.data(), static_cast<size_t>(got));
          if (stderr_tail.size() > 4096) stderr_tail.erase(0, stderr_tail.size() - 4096);
        }
      } else if (got == 0 || (errno != EAGAIN && errno != EINTR)) {
        (is_out ? out_open : err_open) = false;
      }
    }
  }
  if (timed_out) ::kill(pid, SIGKILL);
  int status = 0;
  while (::waitpid(pid, &status, 0) < 0 && errno == EINTR) {
  }
  exit_ok = stopped || (WIFEXITED(status) && WEXITSTATUS(status) == 0);
  if (timed_out) return err = "video decoding took longer than " + std::string("the deadline"), false;
  return true;
}

std::string last_line(std::string s) {
  while (!s.empty() && (s.back() == '\n' || s.back() == '\r' || s.back() == ' ')) s.pop_back();
  const size_t nl = s.rfind('\n');
  return nl == std::string::npos ? s : s.substr(nl + 1);
}

double rational(const std::string *s) {
  if (!s) return 0;
  const size_t sl = s->find('/');
  const double a = std::atof(s->c_str());
  if (sl == std::string::npos) return a;
  const double b = std::atof(s->c_str() + sl + 1);
  return b > 0 ? a / b : 0;
}

double number_of(const json::Value *v) {
  if (!v) return 0;
  if (v->is_number()) return v->num;
  if (v->is_string()) return std::atof(v->str.c_str());
  return 0;
}

}  // namespace

bool find_video_tools(const std::string &ffmpeg, VideoTools &out, std::string &why) {
  out.ffmpeg = on_path(ffmpeg.empty() ? std::string("ffmpeg") : ffmpeg);
  if (out.ffmpeg.empty()) return why = "no ffmpeg executable (" + (ffmpeg.empty() ? std::string("ffmpeg") : ffmpeg) + ") found: video input is off", false;
  const size_t sl = out.ffmpeg.rfind('/');
  const std::string beside = out.ffmpeg.substr(0, sl + 1) + "ffprobe";
  out.ffprobe = executable(beside) ? beside : on_path("ffprobe");
  if (out.ffprobe.empty()) return why = "no ffprobe beside " + out.ffmpeg + " or on PATH: video input is off", false;
  return true;
}

bool probe_video(const std::string &bytes, const VideoTools &tools, double timeout_s, VideoInfo &out, std::string &err) {
  Fd mf;
  if (!memfile(bytes, mf, err)) return false;
  const auto deadline = Clock::now() + std::chrono::milliseconds(static_cast<long long>(timeout_s * 1000));
  std::string text, tail;
  bool ok = false;
  const std::vector<std::string> argv = {tools.ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
                                         "stream=codec_name,width,height,nb_frames,avg_frame_rate,r_frame_rate,duration:"
                                         "stream_side_data=rotation:format=duration",
                                         "-of", "json", "/proc/self/fd/3"};
  if (!run(argv, mf.fd, deadline, [&](const char *p, size_t n) { text.append(p, n); return text.size() < (1u << 20); }, tail, ok, err)) return false;
  json::Value v;
  if (!ok || !json::parse(text, v)) return err = "not a readable video" + (tail.empty() ? std::string() : ": " + last_line(tail)), false;
  const json::Value *streams = v.get_array("streams");
  if (!streams || streams->arr.empty()) return err = "no video stream in the file", false;
  const json::Value &s = streams->arr[0];
  out.width = static_cast<int>(number_of(s.get("width")));
  out.height = static_cast<int>(number_of(s.get("height")));
  out.codec = s.get_string("codec_name") ? *s.get_string("codec_name") : "";
  // the frame rate: r_frame_rate, unless it is implausibly high next to a sane average
  const double r = rational(s.get_string("r_frame_rate")), avg = rational(s.get_string("avg_frame_rate"));
  out.fps = (avg > 0 && avg < 70 && r > 210) || r <= 0 ? avg : r;
  double dur = 0;
  if (const json::Value *fmt = v.get_object("format")) dur = number_of(fmt->get("duration"));
  if (dur <= 0) dur = number_of(s.get("duration"));
  out.duration = dur;
  out.total_frames = static_cast<int64_t>(number_of(s.get("nb_frames")));
  if (out.total_frames <= 0) out.total_frames = static_cast<int64_t>(std::floor(dur * out.fps + 0.5));
  if (const json::Value *sd = s.get_array("side_data_list"))
    for (const auto &e : sd->arr)
      if (const json::Value *rot = e.get("rotation")) out.rotation = ((static_cast<int>(number_of(rot)) % 360) + 360) % 360;
  if (out.width <= 0 || out.height <= 0 || out.fps <= 0 || out.total_frames <= 0)
    return err = "the video's size, frame rate or frame count cannot be read", false;
  return true;
}

bool decode_video_frames(const std::string &bytes, const VideoTools &tools, const std::vector<int64_t> &indices, double timeout_s,
                         const std::function<bool(size_t, Image &&)> &on_frame, size_t &delivered, std::string &err) {
  delivered = 0;
  if (indices.empty()) return true;
  Fd mf;
  if (!memfile(bytes, mf, err)) return false;
  std::string sel = "select=";
  for (size_t i = 0; i < indices.size(); ++i) sel += (i ? "+eq(n\\," : "eq(n\\,") + std::to_string(indices[i]) + ")";
  const std::vector<std::string> argv = {tools.ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-i", "/proc/self/fd/3",
                                         "-map", "0:v:0", "-an", "-sn", "-dn", "-vf", sel, "-fps_mode", "passthrough",
                                         "-sws_flags", "bicubic", "-pix_fmt", "rgb24", "-c:v", "ppm", "-f", "image2pipe", "-"};
  const auto deadline = Clock::now() + std::chrono::milliseconds(static_cast<long long>(timeout_s * 1000));
  // incremental binary PPM reader: "P6" ws W ws H ws 255 one-ws, then W x H x 3 bytes
  std::string buf;
  bool bad = false, stop = false;
  auto parse = [&]() {
    for (;;) {
      size_t p = 0;
      int fields[3] = {0, 0, 0};
      if (buf.size() < 2) return;
      if (buf[0] != 'P' || buf[1] != '6') return void(bad = true);
      p = 2;
      for (int f = 0; f < 3; ++f) {
        while (p < buf.size() && std::isspace(static_cast<unsigned char>(buf[p]))) ++p;
        if (p >= buf.size()) return;
        int v = 0;
        size_t q = p;
        while (q < buf.size() && buf[q] >= '0' && buf[q] <= '9') v = v * 10 + (buf[q++] - '0');
        if (q >= buf.size()) return;
        if (q == p) return void(bad = true);
        fields[f] = v;
        p = q;
      }
      ++p;  // the single whitespace after maxval
      if (fields[2] != 255 || fields[0] <= 0 || fields[1] <= 0) return void(bad = true);
      const size_t need = static_cast<size_t>(fields[0]) * fields[1] * 3;
      if (buf.size() < p + need) return;
      Image im;
      im.w = fields[0], im.h = fields[1];
      im.rgb.assign(buf.begin() + static_cast<std::ptrdiff_t>(p), buf.begin() + static_cast<std::ptrdiff_t>(p + need));
      buf.erase(0, p + need);
      if (!on_frame(delivered++, std::move(im)) || delivered >= indices.size()) return void(stop = true);
    }
  };
  std::string tail;
  bool ok = false;
  if (!run(argv, mf.fd, deadline,
           [&](const char *p, size_t n) {
             buf.append(p, n);
             parse();
             return !bad && !stop;
           },
           tail, ok, err)) {
    if (err.find("deadline") != std::string::npos) err = "decoding the video took longer than " + std::to_string(static_cast<int>(timeout_s)) + " s";
    return false;
  }
  if (bad) return err = "the video decoder wrote an unexpected frame format", false;
  if (!ok && delivered < indices.size()) return err = "the video cannot be decoded" + (tail.empty() ? std::string() : ": " + last_line(tail)), false;
  return true;
}

std::string data_url_type(const std::string &url) {
  if (url.rfind("data:", 0) != 0) return {};
  const size_t end = url.find_first_of(";,", 5);
  std::string t = url.substr(5, end == std::string::npos ? std::string::npos : end - 5);
  for (char &c : t) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return t;
}

}  // namespace xenithium:util
