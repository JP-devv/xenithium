// xenithium-probe: host-side helpers of the qwen35 subcommands (qwen35.cpp), shared with qwen35_gdn.cpp.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "xenithium/model/arch.hpp"
#include "xenithium/model/bulk_read.hpp"
#include "xenithium/model/checkpoint.hpp"

// A card's slice of one w4 spec rebuilt on the host from the raw tensors, independently of DenseStore, in the GPTQ
// orientation w4a16_ref takes: qweight [k/8][n] words, scales fp32 [k/128][n] (n padded as the store pads).
struct HostW4 {
  std::vector<uint32_t> gptq;
  std::vector<float> scales;
  int64_t rows = 0;
};
HostW4 host_w4_slice(const xenithium::model::Checkpoint &ck, const xenithium::model::DenseSpec &sp, const xenithium::model::BulkBuffer &buf,
                     const std::vector<const xenithium::model::TensorInfo *> &srcs, int c, int C, int n, int kk);
// The (packed, scale) tensor pairs of a w4 spec in either layout, appends after, as bulk_read takes them.
std::vector<const xenithium::model::TensorInfo *> w4_sources(const xenithium::model::Checkpoint &ck, const xenithium::model::DenseSpec &sp);
// Dequantised [n][k] floats of a host slice: (q - 8) x scale.
std::vector<float> deq_w4(const HostW4 &h, int n, int kk);
float f16_to_float(uint16_t h);
