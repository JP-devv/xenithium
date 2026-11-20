// Xenithium · the hot-expert ranking recorded while serving: see model/route_record.hpp.
#include "xenithium/model/route_record.hpp"

#include <cmath>
#include <filesystem>
#include <stdexcept>

#include "xenithium/model/route_counts.hpp"

namespace xenithium::model {

RouteRecord::RouteRecord(int layers, int experts, std::string path, double every_s, double half_life_periods)
    : L_(layers), E_(experts), path_(std::move(path)), every_s_(every_s), decay_(std::pow(0.5, 1.0 / half_life_periods)),
      score_(static_cast<size_t>(layers) * experts, 0.0), period_(score_.size(), 0.0), seen_(static_cast<size_t>(experts), 0u),
      t_(std::chrono::steady_clock::now()) {
  if (path_.empty() || !std::filesystem::exists(path_)) {
    status_ = path_.empty() ? "kept in memory" : "no ranking file yet: the server creates " + path_;
    return;
  }
  try {
    score_ = read_route_counts(path_, layers, experts);
    loaded_ = true;
    status_ = "ranking " + path_;
  } catch (const std::exception &e) {
    ok_ = false;  // another model's file (or a broken one): never overwrite it
    status_ = e.what();
  }
}

void RouteRecord::add_step(int layer, const int32_t *ids, int n) {
  if (++stamp_ == 0) std::fill(seen_.begin(), seen_.end(), 0u), stamp_ = 1;
  double *p = period_.data() + static_cast<size_t>(layer) * E_;
  for (int j = 0; j < n; ++j) {
    const int32_t e = ids[j];
    if (e < 0 || e >= E_ || seen_[static_cast<size_t>(e)] == stamp_) continue;
    seen_[static_cast<size_t>(e)] = stamp_;
    p[e] += 1.0;
  }
}

void RouteRecord::add_counts(const std::vector<double> &delta) {
  if (delta.size() != period_.size()) throw std::invalid_argument("route record: counts of another shape");
  for (size_t i = 0; i < delta.size(); ++i) period_[i] += delta[i];
}

bool RouteRecord::due() const { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t_).count() >= every_s_; }

double RouteRecord::pending() const {
  double s = 0;
  for (double v : period_) s += v;
  return s;
}

bool RouteRecord::save() {
  t_ = std::chrono::steady_clock::now();
  if (pending() <= 0) return ok_;  // no decode steps this period: no decay, no write
  for (size_t i = 0; i < score_.size(); ++i) score_[i] = score_[i] * decay_ + period_[i];
  std::fill(period_.begin(), period_.end(), 0.0);
  ++periods_;
  if (path_.empty() || !ok_) return ok_;
  try {
    const std::filesystem::path p(path_);
    if (p.has_parent_path()) std::filesystem::create_directories(p.parent_path());
    const std::string tmp = path_ + ".tmp";
    write_route_counts(tmp, L_, E_, score_);
    std::filesystem::rename(tmp, p);
    status_ = "ranking " + path_;
    return true;
  } catch (const std::exception &e) {
    status_ = e.what();
    return false;
  }
}

}  // namespace xenithium:model
