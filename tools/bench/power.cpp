#include "power.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <optional>

namespace xenithium::bench {
namespace {

namespace fs = std::filesystem;

// Readings per second: the clock moves within a decode step's time, so a few readings per second give its mean over
// a scenario; the energy counter needs none of them (it is cumulative) but they draw the power-over-time trace.
constexpr double kSamplesPerSecond = 4;
constexpr double kMicro = 1e-6;

// Where the drivers publish the GPU's actual clock, relative to /sys/class/drm/cardN: xe, then i915.
constexpr const char *kClockFiles[] = {"device/tile0/gt0/freq0/act_freq", "gt/gt0/rps_act_freq_mhz", "gt_act_freq_mhz"};

std::optional<uint64_t> read_number(const std::string &path) {
  std::ifstream in(path);
  uint64_t value = 0;
  return in >> value ? std::optional(value) : std::nullopt;
}

// cardN, not its connectors (card0-DP-1).
bool is_card_dir(const std::string &name) {
  return name.rfind("card", 0) == 0 && name.size() > 4 && std::all_of(name.begin() + 4, name.end(), [](char c) { return c >= '0' && c <= '9'; });
}

// The card's hwmon directory that has a board energy counter; empty when it has none.
std::string hwmon_with_energy(const fs::path &card) {
  std::error_code ec;
  for (const auto &entry : fs::directory_iterator(card / "device" / "hwmon", ec))
    if (fs::exists(entry.path() / "energy1_input", ec)) return entry.path().string();
  return {};
}

uint64_t delta(uint64_t from, uint64_t to) { return to >= from ? to - from : 0; }  // a counter that restarted counts as nothing

}  // namespace

std::unique_ptr<PowerMeter> PowerMeter::open(const std::string &drm_root) {
  std::vector<std::string> names;
  std::error_code ec;
  for (const auto &entry : fs::directory_iterator(drm_root, ec))
    if (is_card_dir(entry.path().filename().string())) names.push_back(entry.path().filename().string());
  std::ranges::sort(names);

  std::vector<PowerCard> cards;
  std::vector<CardFiles> files;
  for (const std::string &name : names) {
    const fs::path card = fs::path(drm_root) / name;
    const std::string hwmon = hwmon_with_energy(card);
    if (hwmon.empty()) continue;
    CardFiles f{hwmon + "/energy1_input", {}};
    for (const char *clock : kClockFiles)
      if (f.clock_path.empty() && fs::exists(card / clock, ec)) f.clock_path = (card / clock).string();
    cards.push_back({name, static_cast<double>(read_number(hwmon + "/power1_cap").value_or(0)) * kMicro});
    files.push_back(std::move(f));
  }
  if (cards.empty()) return nullptr;
  return std::unique_ptr<PowerMeter>(new PowerMeter(std::move(cards), std::move(files)));
}

PowerMeter::PowerMeter(std::vector<PowerCard> cards, std::vector<CardFiles> files)
    : cards_(std::move(cards)), files_(std::move(files)), started_(std::chrono::steady_clock::now()) {
  samples_.push_back(read_now());
  sampler_ = std::jthread([this](std::stop_token stop) { sample_until_stopped(stop); });
}

PowerMeter::~PowerMeter() = default;

PowerSample PowerMeter::read_now() const {
  PowerSample s;
  s.at_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - started_).count();
  for (const CardFiles &f : files_) {
    s.energy_uj.push_back(read_number(f.energy_path).value_or(0));
    s.mhz.push_back(f.clock_path.empty() ? 0.0 : static_cast<double>(read_number(f.clock_path).value_or(0)));
  }
  return s;
}

void PowerMeter::sample_until_stopped(std::stop_token stop) {
  const auto period = std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::duration<double>(1.0 / kSamplesPerSecond));
  while (!stop.stop_requested()) {
    std::this_thread::sleep_for(period);
    PowerSample s = read_now();
    const std::scoped_lock lock(mutex_);
    samples_.push_back(std::move(s));
  }
}

double PowerMeter::mark() {
  PowerSample s = read_now();
  const double at_s = s.at_s;
  const std::scoped_lock lock(mutex_);
  // the sampler may have appended a later reading between read_now and the lock: keep the samples ascending
  samples_.insert(std::ranges::upper_bound(samples_, at_s, {}, &PowerSample::at_s), std::move(s));
  return at_s;
}

PowerUse PowerMeter::use(double from_s, double to_s) const {
  const std::scoped_lock lock(mutex_);
  PowerUse use;
  const auto first = std::ranges::lower_bound(samples_, from_s, {}, &PowerSample::at_s);
  auto last = std::ranges::upper_bound(samples_, to_s, {}, &PowerSample::at_s);
  if (first == samples_.end() || last == samples_.begin() || --last <= first) return use;
  use.seconds = last->at_s - first->at_s;
  for (size_t card = 0; card < cards_.size(); ++card) {
    const double joules = static_cast<double>(delta(first->energy_uj[card], last->energy_uj[card])) * kMicro;
    use.joules += joules;
    use.watts_per_card.push_back(joules / use.seconds);
  }
  double mhz_sum = 0;
  int mhz_count = 0;
  for (auto it = first; it <= last; ++it)
    for (const double mhz : it->mhz) {
      if (mhz <= 0) continue;
      mhz_sum += mhz, ++mhz_count;
      use.min_mhz = use.min_mhz == 0 ? mhz : std::min(use.min_mhz, mhz);
    }
  if (mhz_count > 0) use.mean_mhz = mhz_sum / mhz_count;
  return use;
}

std::vector<PowerPoint> PowerMeter::trace(double from_s, size_t max_points) const {
  const std::scoped_lock lock(mutex_);
  std::vector<PowerPoint> points;
  const size_t begin = static_cast<size_t>(std::ranges::lower_bound(samples_, from_s, {}, &PowerSample::at_s) - samples_.begin());
  if (samples_.size() < begin + 2 || max_points == 0) return points;
  const size_t stride = std::max<size_t>(1, (samples_.size() - begin - 1 + max_points - 1) / max_points);
  for (size_t at = begin + stride; at < samples_.size(); at += stride) {
    const PowerSample &from = samples_[at - stride], &to = samples_[at];
    const double seconds = to.at_s - from.at_s;
    if (seconds <= 0) continue;
    PowerPoint point{to.at_s - from_s, {}, 0};
    double mhz_sum = 0;
    for (size_t card = 0; card < cards_.size(); ++card) {
      point.watts.push_back(static_cast<double>(delta(from.energy_uj[card], to.energy_uj[card])) * kMicro / seconds);
      mhz_sum += to.mhz[card];
    }
    point.mhz = mhz_sum / static_cast<double>(cards_.size());
    points.push_back(std::move(point));
  }
  return points;
}

}  // namespace xenithium:bench
