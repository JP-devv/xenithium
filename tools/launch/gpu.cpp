#include "gpu.hpp"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>

namespace xenithium::launch {

namespace fs = std::filesystem;

namespace {

bool read_line(const std::string &path, std::string &out) {
  std::ifstream f(path);
  if (!f) return false;
  std::getline(f, out);
  while (!out.empty() && (out.back() == '\n' || out.back() == '\r' || out.back() == ' ')) out.pop_back();
  return true;
}

bool read_num(const std::string &path, double &v) {
  std::string s;
  if (!read_line(path, s) || s.empty()) return false;
  char *end = nullptr;
  v = std::strtod(s.c_str(), &end);
  return end && end != s.c_str();
}

}  // namespace

std::vector<GpuInfo> read_gpus(const std::string &drm_root) {
  std::vector<GpuInfo> out;
  std::error_code ec;
  for (const auto &e : fs::directory_iterator(drm_root, ec)) {
    const std::string name = e.path().filename().string();
    if (name.size() < 5 || name.compare(0, 4, "card") != 0 || name.find_first_not_of("0123456789", 4) != std::string::npos) continue;
    for (const auto &h : fs::directory_iterator(e.path() / "device" / "hwmon", ec)) {
      std::string driver;
      read_line((h.path() / "name").string(), driver);
      if (driver != "xe") continue;
      const std::string hw = h.path().string();
      GpuInfo g;
      g.card = std::atoi(name.c_str() + 4);
      const fs::path dev = fs::canonical(e.path() / "device", ec);
      g.bdf = ec ? "" : dev.filename().string();
      double v = 0;
      if (read_num(hw + "/power1_cap", v)) g.cap_w = v / 1e6;
      if (read_num(hw + "/fan1_input", v)) g.fan_rpm = static_cast<int>(v);
      // temp<N>_label names the sensor ("pkg", "vram", and many per-unit ones); millidegrees C.
      for (const auto &f : fs::directory_iterator(h.path(), ec)) {
        const std::string fn = f.path().filename().string();
        if (fn.compare(0, 4, "temp") != 0 || fn.size() < 7 || fn.compare(fn.size() - 6, 6, "_label") != 0) continue;
        std::string label;
        read_line(f.path().string(), label);
        double t = 0;
        if ((label == "pkg" || label == "vram") && read_num(hw + "/" + fn.substr(0, fn.size() - 6) + "_input", t))
          (label == "pkg" ? g.temp_pkg : g.temp_vram) = t / 1000.0;
      }
      // energy<N>_label "card" is the whole board; "pkg" the GPU die only.
      for (int i = 1; i <= 4; ++i) {
        std::string label;
        if (!read_line(hw + "/energy" + std::to_string(i) + "_label", label) || label != "card") continue;
        if (read_num(hw + "/energy" + std::to_string(i) + "_input", v) && v >= 0) {
          g.have_energy = true;
          g.energy_uj = static_cast<uint64_t>(v);
        }
        break;
      }
      out.push_back(std::move(g));
      break;
    }
  }
  std::sort(out.begin(), out.end(), [](const GpuInfo &a, const GpuInfo &b) { return a.card < b.card; });
  return out;
}

double power_between(uint64_t e0, uint64_t e1, double seconds) {
  if (seconds <= 0 || e1 < e0) return -1;
  return static_cast<double>(e1 - e0) / 1e6 / seconds;
}

}  // namespace xenithium:launch
