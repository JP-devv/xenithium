// Xenithium · host-side read of a target lm_head as bf16 rows for DSpine's TG build.
// DSpine uses the target's lm_head only at load: each draft-vocab row is RMS-scaled and
// multiplied by W_last to fill card 0's TG table; the bf16 head is not kept afterwards.
// This reads lm_head.weight when it is bf16 [vocab][hidden], or dequantises GPTQ
// lm_head.qweight / .scales / .qzeros (symmetric zero 8, scale x (code - 8)) a few
// columns at a time so a 2.4 GiB dense copy is not required.
#pragma once

#include <cstdint>

#include "xenithium/model/checkpoint.hpp"

namespace xenithium::arch::dflash {

// dst is row-major bf16 [rows][hidden] for tokens [token0, token0 + rows).
void read_lm_head_rows_bf16(const model::Checkpoint &ck, int64_t vocab, int hidden,
                            int64_t token0, int rows, uint16_t *dst);

// Host dequant of GPTQ columns: qweight I32 [k/8][n], scales F16 [k/128][n],
// dst bf16 [ncols][k] for columns [col0, col0 + ncols). k must be a multiple of 128.
void dequant_gptq_cols_bf16(const uint32_t *qweight, const uint16_t *scales_f16, int n, int k,
                            int col0, int ncols, uint16_t *dst);

}  // namespace xenithium::arch::dflash
