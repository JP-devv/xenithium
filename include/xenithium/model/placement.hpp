// Xenithium · hot-expert placement that follows the traffic: decayed scores per (MoE layer, expert) from the decode
// reads, and once a round the exchanges that move the best-scoring cold experts into VRAM in place of the weakest hot
// ones of the same layer (a layer's hot count never changes). Host only: the engine (qwen38::Model, ModelOptions::
// exchange) feeds it the cards' read counts and carries the exchanges out; the route simulator replays recorded traces
// through the same class.
//
// Scores: + 1 per expert per decode step it was read in (the union over the step's rows), x decay once a round. A cold
// expert replaces a hot one when its score > the hot one's x hysteresis + margin; a round takes per layer at most
// `budget` pairs and over all layers at most `max_swaps`, the largest score gaps first.
#pragma once

#include <cstdint>
#include <vector>

namespace xenithium::model {

struct PlacementOptions {
  int budget = 1;            // exchanges per layer per round
  int max_swaps = 32;        // exchanges per round over all layers (the engine's spare blobs per card)
  double decay = 0.90;       // score decay per round
  double hysteresis = 1.25;  // a cold expert replaces a hot one when its score > hot x hysteresis + margin
  double margin = 1.0;
};

class Placement {
 public:
  struct Swap {
    int layer, hot, cold;  // `hot` leaves VRAM, `cold` enters it
    double gain;           // score(cold) - score(hot) when chosen
  };
  // hot: [layer][expert] flags, the starting placement; prior (optional): [layer][expert] counts (the ranking file),
  // scaled per layer to `prior_reads` reads so a fresh server starts from the ranking but the traffic soon decides.
  Placement(int layers, int experts, std::vector<uint8_t> hot, const std::vector<double> *prior, const PlacementOptions &o,
            double prior_reads = 16);

  // One decode step's routes of a layer: n ids (rows x topk), counted once per unique id.
  void add(int layer, const int32_t *ids, int n);
  // Reads counted elsewhere (the cards' histogram): delta [layer][expert].
  void add_counts(const std::vector<double> &delta);
  // A round: the exchanges (the hot flags already show them; undo() one the caller could not make), then the decay.
  // no_enter (optional, [layer][expert]): cold experts that may not enter this round (already in VRAM elsewhere).
  std::vector<Swap> round(const std::vector<uint8_t> *no_enter = nullptr);
  void undo(const Swap &s);

  bool hot(int layer, int expert) const { return hot_[idx(layer, expert)] != 0; }
  double score(int layer, int expert) const { return score_[idx(layer, expert)]; }
  int64_t rounds() const { return rounds_; }
  int64_t swaps() const { return swaps_; }
  const PlacementOptions &options() const { return o_; }

 private:
  size_t idx(int l, int e) const { return static_cast<size_t>(l) * static_cast<size_t>(E_) + static_cast<size_t>(e); }
  PlacementOptions o_;
  int L_, E_;
  std::vector<uint8_t> hot_;
  std::vector<double> score_;
  std::vector<uint32_t> seen_;  // [experts] the add() stamp an expert was last counted at
  uint32_t stamp_ = 0;
  int64_t rounds_ = 0, swaps_ = 0;
};

}  // namespace xenithium::model
