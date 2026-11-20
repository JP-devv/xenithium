// Xenithium · bake outputs: a safetensors writer and the tensor layouts the bake writes.
// experts (AutoRound export, the names and layout of qwen38-w4a16 / auto_round:auto_gptq):
// <prefix>layers.L.mlp.experts.E.{gate,up,down}_proj.{qweight,qzeros,scales}
// qweight I32 [k/8][n] (nibble i of word r = code of k 8r + i, + 8), qzeros I32 [k/128][n/8] =
// 0x77777777 (zero 8 stored as 7, GPTQ v1), scales F16 [k/128][n]
// PLE int8 rows: <prefix>layers.P.ple.ple_embedding.ngram_embedding.shard_S.weight U8 [rows][dim + 4]:
// per row dim int8 codes (q = rne(x / s) clamped to +-127) then the fp32 scale s = amax / 127
// (little endian), one read per gathered row.
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "xenithium/quant/autoround.hpp"

namespace xenithium::quant {

struct StTensor {
  std::string name;
  std::string dtype;  // "I32", "F16", "U8", "BF16", "F32", "I64"...
  std::vector<int64_t> shape;
  const void *data = nullptr;
  uint64_t bytes = 0;
};

// One safetensors file (header JSON padded to 8 bytes, tensors in the given order). Throws on I/O
// errors; writes to path + ".part" and renames at the end, so a crash leaves no half file.
void write_safetensors(const std::string &path, const std::vector<StTensor> &tensors,
                       const std::map<std::string, std::string> &metadata = {});

// The 9 export tensors of every expert of one layer, from the tuner's results ([experts]).
// Buffers live in `store` (kept alive by the caller until written).
std::vector<StTensor> expert_export_tensors(const ArShape &s, const std::string &prefix, int layer,
                                            const std::vector<ExpertQuant> &q, std::vector<std::vector<uint8_t>> &store);

// codes [n][k] (in [-8, 7]) -> GPTQ qweight [k/8][n] (nibble = code + 8).
void gptq_pack(const int8_t *codes, int n, int k, uint32_t *qweight);
// Inverse of the export, for checks: one projection's codes [n][k] and fp16 scales [n][k/128]
// from qweight [k/8][n] + scales [k/128][n].
void gptq_unpack(const uint32_t *qweight, const uint16_t *scales, int n, int k, int8_t *codes, uint16_t *scale_nk);

// PLE int8 rows: bf16 [rows][dim] -> [rows][dim + 4]; returns the sum of squared errors and of
// squared values (for the report).
void ple_int8_rows(const uint16_t *src, int64_t rows, int dim, uint8_t *dst, double *err2, double *ref2);

}  // namespace xenithium:quant
