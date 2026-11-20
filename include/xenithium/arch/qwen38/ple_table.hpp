// Xenithium · qwen38 PLE n-gram table on the host: the layout derived from config.json (checked
// against the buffers the checkpoint stores), the token ids -> table row ids, and the rows read
// in place from the shards (model::RowReader) and dequantised to bf16 (FP8 code table, int8 rows or BF16).
//
// The table is 320,001,536 rows x 160 FP8 (47.68 GiB) in 128 shard tensors over 10 files; it
// does not fit next to the rest of the model in a 64 GB host, so a step reads its rows from the
// drive: heads (16) rows per token.
#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "xenithium/arch/qwen38/arch.hpp"
#include "xenithium/kernels/embed/ple_ref.hpp"
#include "xenithium/model/checkpoint.hpp"
#include "xenithium/model/row_reader.hpp"

namespace xenithium::arch::qwen38 {

class PleTable {
 public:
  // Derives the layout, compares it with the checkpoint's layer_multipliers /
  // ngram_heads_vocab_sizes / ngram_heads_offsets (throws on any difference: every row would be
  // wrong), checks the shard tensors cover exactly the padded rows, opens the reader.
  PleTable(const model::Checkpoint &ck, const Config &cfg, int threads);

  const kernels::PleLayout &layout() const { return L_; }
  int heads() const { return L_.heads; }
  int head_dim() const { return head_dim_; }
  int dim() const { return L_.heads * head_dim_; }
  uint16_t scale_bits() const { return scale_; }
  const std::array<uint16_t, 256> &lut() const { return lut_; }
  model::RowReader &reader() { return *reader_; }

  // ids [n][heads] of n consecutive tokens of one sequence; ctx is advanced past them.
  void ids(const int64_t *tokens, int n, kernels::PleContext &ctx, int64_t *ids) const;
  // The table's row format: FP8 E4M3 + one global scale (code table, exact), int8 rows with an fp32
  // scale each (the bake's,: dequant bf16(q * s)), or BF16 rows (a BF16 source, copied).
  enum class Format : uint8_t { fp8, int8_rows, bf16 };
  Format format() const { return fmt_; }
  size_t row_bytes() const { return row_bytes_; }
  // Raw rows of ids [n][heads] -> codes [n][heads][row_bytes] (the step's read).
  void read(const int64_t *ids, int n, uint8_t *codes);
  // codes -> bf16 bits [n][dim].
  void dequant(const uint8_t *codes, int n, uint16_t *emb) const;

 private:
  kernels::PleLayout L_;
  int head_dim_ = 0;
  Format fmt_ = Format::fp8;
  size_t row_bytes_ = 0;
  uint16_t scale_ = 0;
  std::array<uint16_t, 256> lut_{};
  std::unique_ptr<model::RowReader> reader_;
};

// The table's tensor names: "<layer>.ple.ple_embedding.ngram_embedding.shard_<i>.weight".
std::string ple_prefix(int layer);

}  // namespace xenithium::arch::qwen38
