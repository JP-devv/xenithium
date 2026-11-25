// Host oracle for the Swift trained QSA sidecar; independent of its device projection precision.
#pragma once
#include <array>
#include <cstdint>
#include <span>
#include <vector>

#include "xenithium/arch/qwen35/arch.hpp"
#include "xenithium/model/bulk_read.hpp"

namespace xenithium::arch::qwen35::trained_qsa {
inline constexpr int dim = 128, heads = 4;
using Key = std::array<float, dim>;
using WideKey = std::array<double, dim>;
using Query = std::array<WideKey, heads>;

enum class NormStorage { delta, multiplier };
struct Projection { Query q{}; WideKey k{}; };
struct Normalized { Query q{}; Key k{}; };  // first canonical FP32 cache boundary

class Layer {
 public:
  // This reads only the five tensors of one full-attention layer. Norm storage
  // is the caller's checked sidecar metadata (missing = delta in K8V4).
  static Layer load(const model::Checkpoint &bank, const Config &target, int layer,
                    NormStorage storage, const model::BulkOptions &io = {});
  Projection project(std::span<const uint16_t> bf16_hidden) const;
  Normalized normalize(const Projection &row, int64_t text_position) const;
  Key pool(std::span<const Key, 4> keys, int64_t first_text_position) const;
  // Native dense GEMM tiles containing the three BF16 expansion planes of
  // the F32 bank in [Q0..Q3|K] order. Device sums accumulate in F32.
  std::array<std::vector<uint16_t>,3> device_tiles() const;
  std::array<Key,3> multipliers() const { return {qnorm_,knorm_,bnorm_}; }
  int hidden() const { return hidden_; }
  double theta() const { return theta_; }
  float epsilon() const { return static_cast<float>(eps_); }
 private:
  int hidden_ = 0;
  double theta_ = 0, eps_ = 0;
  std::vector<float> qw_, kw_;
  Key qnorm_{}, knorm_{}, bnorm_{};
};

double score(const Query &q, const Key &block);
// Takes one score per complete logical block; ties go to the lower id. The
// existing page planner adds the causal own block, not this scorer.
std::vector<int32_t> select(std::span<const double> complete_scores);
}  // namespace xenithium::arch::qwen35::trained_qsa
