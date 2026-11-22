// K8V4 trained-indexer handoff: ordered logical four-token blocks -> paged GQA.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

#include "xenithium/kernels/attention/qsa_ref.hpp"

namespace xenithium::kernels {

inline constexpr int kTrainedQsaTopBlocks = 512;
inline constexpr int kTrainedQsaPages = kTrainedQsaTopBlocks + 1;
static_assert(kQsaPage == 64 && kQsaCompress == 4 && kQsaPageBlocks == 16 && kQsaMaxTokens / kQsaPage <= 32768);

// Each entry has a logical page (high bits) and the 16 visible 4-token
// groups within it (low bits). Physical pages always come from the slot's
// existing page table; no physical page number is taken from the selector.
constexpr int trained_qsa_page(int32_t entry) { return entry >> 16; }
constexpr uint32_t trained_qsa_mask(int32_t entry) { return uint32_t(entry) & 0xffffu; }
constexpr size_t trained_qsa_page_base(const int32_t *slot_pages, int page, int hkv, int kvh) {
  return (size_t(slot_pages[page]) * hkv + kvh) * kQsaPage;
}
constexpr bool trained_qsa_keeps(uint32_t mask, int lane) { return ((mask >> (lane / kQsaCompress)) & 1u) != 0; }

// Shared CPU/device compactor for ascending native qsa_select IDs. -1 is
// malformed input; 0 is a fully visible row. Destination has 513 entries.
inline int trained_qsa_pack_row(int pos, const int32_t *ids, int n, int32_t *dst) {
  if (pos < 0 || pos >= kQsaMaxTokens) return -1;
  const int complete = (pos + 1) / kQsaCompress, own = pos / kQsaCompress;
  if (n != (complete < kTrainedQsaTopBlocks ? complete : kTrainedQsaTopBlocks)) return -1;
  int prev = -1;
  for (int i = 0; i < n; ++i) {
    if (ids[i] <= prev || ids[i] >= complete) return -1;
    prev = ids[i];
  }
  const int add_own = prev != own;
  if (n + add_own == own + 1) return 0;
  int count = 0;
  for (int i = 0; i < n + add_own; ++i) {
    const int b = i < n ? ids[i] : own;
    const int page = b / kQsaPageBlocks, bit = 1 << (b % kQsaPageBlocks);
    if (count == 0 || trained_qsa_page(dst[count - 1]) != page) dst[count++] = page << 16;
    dst[count - 1] |= bit;
  }
  return count;
}

struct TrainedQsaRow {
  int position;  // zero-based query position, matching the corresponding segment record
  // Unpadded result of the trained selector BEFORE the mandatory own-block
  // union. Exactly min(512, (position+1)/4) unique complete logical blocks.
  std::span<const int32_t> top_complete;
};
struct TrainedQsaPlan {
  // [rows][kTrainedQsaPages], ascending logical pages; unused entries are zero
  std::vector<int32_t> entries;
  // [rows][local_kv_heads], reuses the stock selected-page merge convention:
  // count zero means that row is dense. All heads of a row share the same list.
  std::vector<int32_t> counts;
  bool sparse() const { return !entries.empty(); }
};

// Host handoff for the first eager path. An all-visible step returns empty
// buffers so the caller leaves GqaStep::trained_qsa unset (stock dispatch).
inline TrainedQsaPlan trained_qsa_plan(std::span<const TrainedQsaRow> rows, int kv_heads) {
  if (rows.empty() || rows.size() > 64 || kv_heads < 1 || kv_heads > 4)
    throw std::invalid_argument("trained QSA: expected 1..64 rows and 1..4 local KV heads");
  TrainedQsaPlan out;
  for (size_t r = 0; r < rows.size(); ++r) {
    const auto [p, picked] = rows[r];
    if (p < 0 || p >= kQsaMaxTokens) throw std::invalid_argument("trained QSA: position outside KV capacity");
    if (picked.size() > size_t(kTrainedQsaTopBlocks)) throw std::invalid_argument("trained QSA: too many complete blocks");
    std::vector<int32_t> blocks(picked.begin(), picked.end());
    std::sort(blocks.begin(), blocks.end());
    int32_t packed[kTrainedQsaPages];
    const int count = trained_qsa_pack_row(p, blocks.data(), static_cast<int>(blocks.size()), packed);
    if (count < 0) throw std::invalid_argument("trained QSA: incomplete, future, negative or duplicate block selection");
    if (!count) continue;
    if (!out.sparse()) {
      out.entries.resize(rows.size() * kTrainedQsaPages);
      out.counts.resize(rows.size() * kv_heads);
    }
    std::copy_n(packed, count, out.entries.data() + r * kTrainedQsaPages);
    std::fill_n(out.counts.data() + r * kv_heads, kv_heads, count);
  }
  return out;
}

}  // namespace xenithium::kernels
