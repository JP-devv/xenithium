// Xenithium · the architecture interface: what every supported model family provides.
// The engine is multi-model. Everything above this interface (inspect tool, loader,
// placement, server) is written against `Arch`; everything model-specific (tensor
// naming, shapes, how components spread over the cards, the byte budget) lives in one
// arch module under src/arch/<id>/. Adding a model = one new module + one line in
// src/model/registry.cpp.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "xenithium/model/checkpoint.hpp"
#include "xenithium/model/config.hpp"
#include "xenithium/util/cards.hpp"

namespace xenithium::model {

// Coarse tensor classes shared by all architectures (the budget and the loader group by
// these). Anything an arch does not recognise is `other` and is reported, never folded
// silently into a known class.
enum class Component : uint8_t {
  embedding,
  lm_head,
  attention,  // full / sparse softmax attention (incl. its indexer)
  linear_attention,  // recurrent / linear attention (GDN...)
  routed_experts,
  shared_expert,
  router,
  dense_mlp,
  hyper_connection,
  norm,
  ple,  // per-layer embedding block weights
  ple_table,  // the large n-gram embedding table (host / NVMe resident)
  mtp,  // multi-token-prediction drafter layers
  vision,  // multimodal tower (not served by a text engine)
  other,
  count
};
std::string_view component_name(Component c);

struct TensorRole {
  Component comp = Component::other;
  int layer = -1;  // decoder layer, -1 = not per layer
  int expert = -1;  // routed expert id, -1 = not an expert
};

// How a deployment spends the cards: the inputs of the byte budget.
struct DeployParams {
  int cards = 2;
  int rows = 4;  // rows per decode step (1 token + drafts)
  int64_t context = 8192;  // tokens per sequence slot
  int seqs = 1;  // sequence slots
  int dense_bytes = 1;  // dense weights on the card: 1 = 8-bit (converted at load), 2 = bf16
  int kv_bytes = 2;  // 2 = bf16 KV, 1 = 8-bit KV (+ one fp32 scale per token and head)
  bool kv_v4 = false;  // with kv_bytes 1: V as 4-bit codes (qwen38's int8-v4 KV)
  bool drafter = false;  // an MTP drafter layer is loaded (adds its KV and weights)
  int64_t kv_tokens = 0;  // KV tokens held in all (a shared pool); 0 = context x seqs
  double unique_experts = 0;  // routed experts touched per layer per step; 0 = independent-routing estimate
  double bandwidth_gbs = util::plan_card().floor_gbs;  // VRAM read floor (GB/s, 1e9); util/cards.hpp
  // Per card. Sysman's free device memory on an idle card of the host (util:plan_card: the emulated card, else the
  // first Battlemage card in sysfs, else the B70: 31.85 GiB free of 31.89); zeDeviceGetMemoryProperties reports 30.30,
  // which undercounts by ~1.5 GiB. Loaders pass the live sysman value.
  double vram_gib = util::plan_card().free_gib;
  double reserve_gib = 0.75;  // scratch, graphs, runtime
  bool embed_on_host = true;  // the embedding table in pinned host memory (a step gathers rows)
  // How attention state is split over the cards: by KV head (each card all tokens of its
  // heads) or by token (each card half of every sequence's tokens, all heads).
  enum class KvShard : uint8_t { heads, tokens } kv_shard = KvShard::heads;
};

// One component's cost on ONE card.
struct BudgetRow {
  std::string name;
  double resident_bytes = 0;  // VRAM held
  double step_bytes = 0;  // bytes read per decode step
  std::string note;
};

struct Budget {
  std::vector<BudgetRow> rows;  // non-expert components + KV; experts reported apart
  double kv_bytes_per_token = 0;  // per card, per sequence slot
  double expert_bytes = 0;  // one routed expert as the card stores it
  int64_t experts_per_card = 0;  // routed experts one card owns (all layers)
  double expert_step_bytes = 0;  // activated experts read per step on one card
  double unique_experts = 0;  // per layer per step (the value used)
  double resident_non_expert = 0;  // bytes, incl. KV at the asked context
  int64_t hot_experts_fit = 0;  // experts that fit after everything else (capped)
  double floor_ms = 0;  // whole decode step at the bandwidth floor
  std::vector<std::string> notes;
};

// How one non-expert tensor is placed: the arch module's declaration, the loader's input.
// A tensor's leading dimension may be made of `sections` (e.g. GDN's q | k | v rows); a rows
// split gives every card its contiguous 1/C of EACH section, concatenated in section order.
struct DenseSpec {
  enum class Split : uint8_t { replicate, rows, cols, host };  // cols = split the last dim (k)
  // raw = bytes as stored (vectors, tables; an f16 source is converted to bf16 at load); w4 = a pack-quantized Linear
  // kept int4 (group codes + fp16 group scales; `name` is its `weight_packed, the module's `weight_scale` sits beside it).
  // w4r: a BF16 Linear [n][k] stored as w4 by round-to-nearest at load (per (row, 128-k group): signed
  // fp16 scale = -extreme / 8, the group's largest |w| on code -8 - the AutoRound / head4-calib RTN convention); the store
  // then holds it exactly like a w4 tensor (DeviceTensor:store == w4).
  // w4d: a GPTQ int4 Linear (`name` is its `qweight` [k/8][n], `scales` [k/128][n] and the symmetric `qzeros` beside it)
  // dequantised at load (value = scale x (code - 8)) to the dense precision `dense_bytes, stored as that matrix.
  enum class Store : uint8_t { i8, bf16, raw, w4, w4r, w4d };
  // w4 orientation: ct = compressed-tensors `weight_packed` [n][k/8] + `weight_scale` [n][k/128];
  // gptq = AutoRound / GPTQ `qweight` [k/8][n] + `scales` [k/128][n] + `qzeros` [k/128][n/8] (must be the
  // symmetric 0x77777777 words). Both land in the same W4 block layout on the card.
  // ct_zp = compressed-tensors asymmetric (arch llama: Nanbeige4.3): `weight_packed` [n][k/8] codes q in 0..15 (no
  // offset), `weight_scale` [n][k/group] (bf16 or f16), `weight_zero_point` I32 [n/8][k/group] (8 four-bit zero points per
  // word along n): w = (q - z) x scale, groups of `w4_group.
  // mlx = affine (arch flint): `<module>.weight` U8 [n][k/2] (the ct word order: element 2j in the low nibble of
  // byte j, q in 0..15).scales` and.biases` BF16 [n][k/group]: w = q x scale + bias, groups of `w4_group` (64).
  enum class W4Layout : uint8_t { ct, gptq, ct_zp, mlx };
  std::string name;
  Split split = Split::replicate;
  Store store = Store::raw;
  W4Layout w4_layout = W4Layout::ct;
  int w4_group = 128;  // k per scale (ct_zp: the checkpoint's group size)
  std::vector<int64_t> sections;  // empty = one section (the whole leading dim)
  // Tensors whose rows follow this one's in the same device matrix (same columns and dtype),
  // stored under `name`; replicated or row-split matrices (under a rows split each appended
  // tensor contributes its own contiguous 1/cards of rows). Rows are padded to 16 after the
  // last one.
  std::vector<std::string> append;
  // qwen38 TP4: under a rows split, the appended tensors are `append_heads` heads of rows (the
  // QSA k / v: 2 kv heads); with fewer heads than cards, card c takes head c x append_heads / cards whole (TP4: cards
  // 2h and 2h + 1 hold kv head h), else its own 1/cards. 0 = the plain split.
  int append_heads = 0;
  Component comp = Component::other;
  int layer = -1;
  // (the MTP drafter's vocabulary subset): Split:rows only, no sections / append: the source rows
  // taken, in order, as one matrix of their total count, split over the cards in contiguous 1/cards (the total must
  // divide by the cards). Empty = every row.
  std::vector<std::pair<int64_t, int64_t>> row_ranges;
  std::string key;  // the store's name for it (empty = name): one tensor stored twice under two names
  bool w4_mse = false;  // w4r: the scale by an MSE clip search (alpha 0.70..1.0 of the RTN scale), not plain RTN
  int dense_bytes = 2;  // w4d only: the dense precision 1 = int8 per row, 2 = bf16
};

// DenseSpec:row_ranges under a rows split: the source row ranges card `card` of `cards` holds (its contiguous 1/cards
// of the ranges taken in order as one matrix). Throws when a range is empty or outside [0, d0) or the total does not
// divide by the cards.
std::vector<std::pair<int64_t, int64_t>> slice_row_ranges(int64_t d0, const std::vector<std::pair<int64_t, int64_t>> &ranges, int card, int cards);

class Arch {
 public:
  virtual ~Arch() = default;
  virtual std::string_view id() const = 0;
  // Human-readable summary of the typed configuration (multi-line).
  virtual std::string describe() const = 0;
  virtual TensorRole classify(std::string_view tensor_name) const = 0;
  // Every expected tensor present with the expected dtype and shape. Returns the problems
  // (empty = the checkpoint is what this arch serves). `warnings` gets non-fatal findings.
  virtual std::vector<std::string> validate(const Checkpoint &ck, std::vector<std::string> *warnings) const = 0;
  virtual Budget budget(const Checkpoint &ck, const DeployParams &p) const = 0;
  // Every non-expert tensor the engine loads, with its placement. Routed experts, vision and
  // the PLE table are not part of it (they have their own stores).
  virtual std::vector<DenseSpec> dense_plan(const Checkpoint &ck, const DeployParams &p) const = 0;
};

struct ArchEntry {
  std::string_view id;
  std::string_view description;
  std::string_view label;  // short name the launcher's model list shows ("dense Qwen3.8-27B")
  std::string_view model_type;  // config.json model_type of the checkpoints it serves (others match by architectures)
  bool (*matches)(const ModelConfig &);
  std::unique_ptr<Arch> (*create)(const ModelConfig &);
};

const std::vector<ArchEntry> &arch_registry();
// The arch whose `matches` accepts this config; throws listing the known archs otherwise.
std::unique_ptr<Arch> open_arch(const ModelConfig &cfg);

}  // namespace xenithium:model
