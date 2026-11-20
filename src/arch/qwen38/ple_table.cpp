#include "xenithium/arch/qwen38/ple_table.hpp"

#include <fcntl.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <stdexcept>

namespace xenithium::arch::qwen38 {

namespace {

std::vector<uint8_t> read_tensor(const model::Checkpoint &ck, const std::string &name) {
  const model::TensorInfo &t = ck.at(name);
  std::vector<uint8_t> b(t.bytes);
  const int fd = ::open(ck.shards().at(t.shard).path.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) throw std::runtime_error("ple: open " + ck.shards().at(t.shard).path + ": " + std::strerror(errno));
  size_t got = 0;
  while (got < b.size()) {
    const ssize_t r = ::pread(fd, b.data() + got, b.size() - got, static_cast<off_t>(t.offset + got));
    if (r <= 0) {
      ::close(fd);
      throw std::runtime_error("ple: short read of " + name);
    }
    got += static_cast<size_t>(r);
  }
  ::close(fd);
  return b;
}

std::vector<int64_t> read_i64(const model::Checkpoint &ck, const std::string &name) {
  const auto b = read_tensor(ck, name);
  if (ck.at(name).dtype != model::DType::i64) throw std::runtime_error("ple: " + name + " is not I64");
  std::vector<int64_t> v(b.size() / 8);
  std::memcpy(v.data(), b.data(), v.size() * 8);
  return v;
}

}  // namespace

std::string ple_prefix(int layer) { return std::string(kPrefix) + "layers." + std::to_string(layer) + ".ple.ple_embedding."; }

PleTable::PleTable(const model::Checkpoint &ck, const Config &cfg, int threads) {
  if (cfg.ple_layer < 0) throw std::runtime_error("ple: the model has no PLE layer");
  if (cfg.eos.empty()) throw std::runtime_error("ple: config has no eos_token_id");
  const std::string P = ple_prefix(cfg.ple_layer);
  // layer_index: the block's index among the PLE layers (one here).
  L_ = kernels::PleLayout::derive(cfg.vocab, cfg.ngram, cfg.heads_per_ngram, cfg.ngram_vocab_base, cfg.ple_seed, 0, cfg.ple_divisor, cfg.eos[0]);
  head_dim_ = cfg.ple_head_dim();
  const auto mult = read_i64(ck, P + "layer_multipliers"), sizes = read_i64(ck, P + "ngram_heads_vocab_sizes"),
             offs = read_i64(ck, P + "ngram_heads_offsets");
  if (mult != L_.mult || sizes != L_.size || offs != L_.offset)
    throw std::runtime_error("ple: the derived n-gram layout differs from the checkpoint's buffers (multipliers / sizes / offsets)");

  const std::string E = P + "ngram_embedding.";
  // Row format from the first shard: F8_E4M3 [rows, hd] + one bf16 weight_scale (the RadixArk table),
  // U8 [rows, hd + 4] int8 rows with an fp32 scale each (the bake), or BF16 [rows, hd] (a
  // BF16 source checkpoint).
  const model::TensorInfo &t0 = ck.at(E + "shard_0.weight");
  if (t0.dtype == model::DType::f8_e4m3) {
    fmt_ = Format::fp8;
    row_bytes_ = static_cast<size_t>(head_dim_);
    const auto sc = read_tensor(ck, E + "weight_scale");
    if (ck.at(E + "weight_scale").dtype != model::DType::bf16 || sc.size() != 2) throw std::runtime_error("ple: weight_scale is not one bf16");
    std::memcpy(&scale_, sc.data(), 2);
    lut_ = kernels::ple_fp8_lut(scale_);
  } else if (t0.dtype == model::DType::u8) {
    fmt_ = Format::int8_rows;
    row_bytes_ = static_cast<size_t>(head_dim_) + 4;
  } else if (t0.dtype == model::DType::bf16) {
    fmt_ = Format::bf16;
    row_bytes_ = static_cast<size_t>(head_dim_) * 2;
  } else {
    throw std::runtime_error("ple: " + t0.name + ": " + t0.dtype_str + " rows (want F8_E4M3, U8 int8 rows or BF16)");
  }

  std::vector<model::RowReader::Segment> segs;
  int64_t rows = 0;
  for (int s = 0; s < cfg.ple_split_parts; ++s) {
    const model::TensorInfo &t = ck.at(E + "shard_" + std::to_string(s) + ".weight");
    if (t.dtype != t0.dtype || t.shape.size() != 2 || static_cast<size_t>(t.shape[1]) * (t.dtype == model::DType::bf16 ? 2 : 1) != row_bytes_)
      throw std::runtime_error("ple: " + t.name + " is not " + t0.dtype_str + " [rows, " + std::to_string(t0.shape[1]) + "] like shard 0");
    segs.push_back({ck.shards().at(t.shard).path, t.offset, t.shape[0]});
    rows += t.shape[0];
  }
  if (rows != L_.padded_rows)
    throw std::runtime_error("ple: the shards hold " + std::to_string(rows) + " rows, the layout wants " + std::to_string(L_.padded_rows));
  reader_ = std::make_unique<model::RowReader>(std::move(segs), row_bytes_, threads);
}

void PleTable::ids(const int64_t *tokens, int n, kernels::PleContext &ctx, int64_t *ids) const {
  for (int i = 0; i < n; ++i) {
    kernels::ple_ngram_ids(L_, tokens[i], ctx.prev.data(), ids + static_cast<size_t>(i) * L_.heads);
    ctx.push(L_, tokens[i]);
  }
}

void PleTable::read(const int64_t *ids, int n, uint8_t *codes) { reader_->gather(ids, static_cast<size_t>(n) * L_.heads, codes); }

void PleTable::dequant(const uint8_t *codes, int n, uint16_t *emb) const {
  const size_t rows = static_cast<size_t>(n) * L_.heads, hd = static_cast<size_t>(head_dim_);
  if (fmt_ == Format::fp8) {
    for (size_t i = 0; i < rows * hd; ++i) emb[i] = lut_[codes[i]];
  } else if (fmt_ == Format::bf16) {
    std::memcpy(emb, codes, rows * hd * 2);
  } else {
    // int8 rows: bf16(q * s), round to nearest even (the bake's reference rounding)
    for (size_t r = 0; r < rows; ++r) {
      const uint8_t *row = codes + r * row_bytes_;
      float s;
      std::memcpy(&s, row + hd, 4);
      for (size_t i = 0; i < hd; ++i) {
        const float v = static_cast<float>(static_cast<int8_t>(row[i])) * s;
        uint32_t u;
        std::memcpy(&u, &v, 4);
        emb[r * hd + i] = static_cast<uint16_t>((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
      }
    }
  }
}

}  // namespace xenithium::arch::qwen38
