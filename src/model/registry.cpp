// The architecture registry: one line per supported model family.
#include <algorithm>
#include <stdexcept>

#include "xenithium/arch/flint/arch.hpp"
#include "xenithium/arch/llama/arch.hpp"
#include "xenithium/arch/qwen35/arch.hpp"
#include "xenithium/arch/qwen38/arch.hpp"
#include "xenithium/model/arch.hpp"

namespace xenithium::model {

std::vector<std::pair<int64_t, int64_t>> slice_row_ranges(int64_t d0, const std::vector<std::pair<int64_t, int64_t>> &ranges, int card, int cards) {
  int64_t total = 0;
  for (const auto &r : ranges) {
    if (r.first < 0 || r.second > d0 || r.second <= r.first) throw std::runtime_error("slice_row_ranges: a row range outside the tensor");
    total += r.second - r.first;
  }
  if (cards < 1 || card < 0 || card >= cards || total % cards) throw std::runtime_error("slice_row_ranges: the row ranges do not split over the cards");
  const int64_t lo = total / cards * card, hi = total / cards * (card + 1);
  std::vector<std::pair<int64_t, int64_t>> out;
  int64_t at = 0;
  for (const auto &r : ranges) {  // virtual rows [at, at + len) are range r
    const int64_t len = r.second - r.first, a = std::max(lo, at), b = std::min(hi, at + len);
    if (b > a) out.push_back({r.first + (a - at), r.first + (b - at)});
    at += len;
  }
  return out;
}

std::string_view component_name(Component c) {
  switch (c) {
    case Component::embedding: return "embedding";
    case Component::lm_head: return "lm_head";
    case Component::attention: return "attention";
    case Component::linear_attention: return "linear_attention";
    case Component::routed_experts: return "routed_experts";
    case Component::shared_expert: return "shared_expert";
    case Component::router: return "router";
    case Component::dense_mlp: return "dense_mlp";
    case Component::hyper_connection: return "hyper_connection";
    case Component::norm: return "norm";
    case Component::ple: return "ple";
    case Component::ple_table: return "ple_table";
    case Component::mtp: return "mtp";
    case Component::vision: return "vision";
    case Component::other: return "other";
    default: return "?";
  }
}

const std::vector<ArchEntry> &arch_registry() {
  static const std::vector<ArchEntry> entries = {
      arch::qwen38::entry(),
      arch::qwen35::entry(),
      arch::llama::entry(),
      arch::flint::entry(),
  };
  return entries;
}

std::unique_ptr<Arch> open_arch(const ModelConfig &cfg) {
  for (const auto &e : arch_registry())
    if (e.matches(cfg)) return e.create(cfg);
  std::string known;
  for (const auto &e : arch_registry()) known += (known.empty() ? "" : ", ") + std::string(e.id);
  std::string archs;
  for (const auto &a : cfg.architectures()) archs += (archs.empty() ? "" : ", ") + a;
  throw std::runtime_error(cfg.dir() + ": no architecture module serves model_type '" + cfg.model_type() +
                           "' (architectures: " + archs + "); known: " + known);
}

}  // namespace xenithium:model
