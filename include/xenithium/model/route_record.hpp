// Xenithium · the hot-expert ranking recorded while serving.
// A MoE server (qwen38) counts which experts its decode steps read, keeps a long-term score per (layer,
// expert) and writes it to its ranking file (XRC1, model/route_counts.hpp) every `every_s` seconds of serving and at
// exit; the next boot places the hot set from that file. Nothing has to exist before the first start: without a file
// the scores start at zero (the boot's default placement) and the first save creates it. A file from an earlier run, or
// one made by scripts/make-hot-rank.sh, seeds the scores and keeps being updated.
// Scores: + 1 per expert per decode step it was read in (the union over the step's rows: one read per expert), folded
// in per period: score = score x decay + the period's reads. The decay runs per ACTIVE period only (a period without
// decode steps neither decays nor writes), so idle time does not age the ranking. The default half-life is 60 active
// periods (an hour of decode traffic at the 60 s period). Host only; the engine feeds it.
#pragma once

#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

namespace xenithium::model {

class RouteRecord {
 public:
  static constexpr double kHalfLifePeriods = 60;

  // path: the ranking file (read at construction when it exists; written by save). Empty path = scores kept in memory
  // only (tests). A file of another shape is not read and never overwritten (status says why; ok false).
  RouteRecord(int layers, int experts, std::string path, double every_s = 60, double half_life_periods = kHalfLifePeriods);

  bool ok() const { return ok_; }
  bool loaded() const { return loaded_; }  // the scores came from the file
  const std::string &status() const { return status_; }
  const std::string &path() const { return path_; }
  int layers() const { return L_; }
  int experts() const { return E_; }

  // The scores [layer][expert] (dense layers zero), what the next boot ranks by.
  const std::vector<double> &scores() const { return score_; }
  // One step's reads of a MoE layer: n expert ids, counted once per unique id.
  void add_step(int layer, const int32_t *ids, int n);
  // Reads counted elsewhere (a device histogram): delta [layer][expert] added to this period.
  void add_counts(const std::vector<double> &delta);
  // True when a period has passed since the last fold (the caller then brings its counts in and calls save).
  bool due() const;
  // Folds the period's reads into the scores (score x decay + reads; nothing when there were none) and writes the file
  // (a temporary file renamed over the old one: a crash never leaves a torn file). Returns false on a write error
  // (status has it; the scores are kept and the next save tries again).
  bool save();
  int64_t periods() const { return periods_; }  // folds with reads since construction
  double pending() const;  // reads not folded yet

 private:
  int L_, E_;
  std::string path_;
  double every_s_, decay_;
  bool ok_ = true, loaded_ = false;
  std::string status_;
  std::vector<double> score_, period_;
  std::vector<uint32_t> seen_;
  uint32_t stamp_ = 0;
  int64_t periods_ = 0;
  std::chrono::steady_clock::time_point t_;
};

}  // namespace xenithium:model
