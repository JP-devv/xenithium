// Xenithium · host-side placement of a DFlash 2 / DSpine drafter's tensors (no SYCL).
// dflash_dense_plan is shared by the loader and the host tests: a bf16 `.weight` Linear keeps the
// existing path (int8 / bf16, or RTN int4 for q / o / gate | up / down at dense_bytes 4); a GPTQ
// int4 Linear (`.qweight` / `.scales` / `.qzeros`, sym, group 128, no act order: DFlashConfig::load checks the
// checkpoint's quantization_config) reuses the 27B body's layout (kept int4 where the drafter would RTN, otherwise
// dequantised to the dense precision).
#pragma once

#include <vector>

#include "xenithium/model/arch.hpp"
#include "xenithium/model/checkpoint.hpp"

namespace xenithium::arch {

// layers = DFlashConfig::layers; dspine = Kind::dspine (no convolution / selector).
// dense_bytes 1 = int8 per row, 2 = bf16, 4 = int4 by RTN (q, o, gate | up, down; the rest int8).
std::vector<model::DenseSpec> dflash_dense_plan(const model::Checkpoint &ck, int layers, int dense_bytes, bool dspine = false);

}  // namespace xenithium::arch
