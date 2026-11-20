// Xenithium · a checkpoint directory: every tensor of every safetensors shard, by name.
// The catalog is built from the shard headers themselves (no Python, no index file
// needed at run time). Each header is preceded by its own little-endian uint64 length;
// tensor offsets are relative to the end of the header, so the absolute file offset is
// 8 + header_len + begin. The length is the one read from the file, never a re-serialised
// header (re-serialising changes whitespace and shifts every offset into plausible garbage).
// Opening validates what a later read would otherwise turn into wrong weights:
// every [begin, end) lies inside the file, ranges do not overlap, byte counts equal
// numel x element size, and names are unique across shards.
#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "xenithium/model/dtype.hpp"

namespace xenithium::model {

struct TensorInfo {
  std::string name;
  uint32_t shard = 0;  // index into Checkpoint:shards
  DType dtype = DType::unknown;
  std::string dtype_str;  // as written (kept for unknown dtypes)
  std::vector<int64_t> shape;
  uint64_t offset = 0;  // absolute file offset of the first byte
  uint64_t bytes = 0;

  int64_t numel() const {
    int64_t n = 1;
    for (const int64_t d : shape) n *= d;
    return n;
  }
  int64_t dim(size_t i) const { return i < shape.size() ? shape[i] : 1; }
};

struct ShardInfo {
  std::string path;  // absolute
  std::string file;  // name inside the directory
  uint64_t file_bytes = 0;
  uint64_t header_bytes = 0;  // the JSON header alone (without the 8-byte length)
  uint64_t data_offset = 0;  // 8 + header_bytes
  size_t tensors = 0;
};

// One shard's header, parsed and validated against the file size. Throws std:runtime_error
// with the file name and the offending tensor on any inconsistency.
std::vector<TensorInfo> read_safetensors_header(const std::string &path, ShardInfo *shard_out);

class Checkpoint {
 public:
  // Every *.safetensors file in `dir` (sorted by name). When `cross_check_index` and a
  // model.safetensors.index.json exists, every weight_map entry must name a tensor that is
  // in the shard it says (tensors absent from the index, such as side tables, are allowed).
  static Checkpoint open(const std::string &dir, bool cross_check_index = true);

  const std::string &dir() const { return dir_; }
  const std::vector<ShardInfo> &shards() const { return shards_; }
  const std::vector<TensorInfo> &tensors() const { return tensors_; }
  size_t size() const { return tensors_.size(); }

  const TensorInfo *find(std::string_view name) const;
  const TensorInfo &at(std::string_view name) const;  // throws when absent
  uint64_t payload_bytes() const;  // sum of tensor bytes

 private:
  std::string dir_;
  std::vector<ShardInfo> shards_;
  std::vector<TensorInfo> tensors_;
  struct Hash {
    using is_transparent = void;
    size_t operator()(std::string_view s) const { return std::hash<std::string_view>{}(s); }
  };
  std::unordered_map<std::string, uint32_t, Hash, std::equal_to<>> by_name_;
};

}  // namespace xenithium:model
