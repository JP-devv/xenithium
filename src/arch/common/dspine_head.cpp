#include "xenithium/arch/common/dspine_head.hpp"

#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

#include "xenithium/model/dtype.hpp"

namespace xenithium::arch::dflash {
namespace {

float f16_to_f32(uint16_t h) {
  const unsigned s = (h >> 15) & 1u, e = (h >> 10) & 0x1fu, m = h & 0x3ffu;
  if (e == 0) {
    if (m == 0) {
      uint32_t u = s << 31;
      float f;
      std::memcpy(&f, &u, 4);
      return f;
    }
    const float x = std::ldexp(static_cast<float>(m), -24);
    return s ? -x : x;
  }
  if (e == 31) {
    uint32_t u = (s << 31) | 0x7f800000u | (m << 13);
    float f;
    std::memcpy(&f, &u, 4);
    return f;
  }
  uint32_t u = (s << 31) | ((e + 112u) << 23) | (m << 13);
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

uint16_t f32_to_bf16(float v) {
  uint32_t u;
  std::memcpy(&u, &v, 4);
  const uint32_t lsb = (u >> 16) & 1u;
  return static_cast<uint16_t>((u + 0x7fffu + lsb) >> 16);
}

void pread_exact(const model::Checkpoint &ck, const model::TensorInfo &ti, uint64_t elem_off, uint64_t nelem, size_t es,
                 void *dst) {
  const uint64_t want = nelem * es;
  if (ti.bytes < (elem_off + nelem) * es) throw std::runtime_error("dflash (DSpine): short tensor " + ti.name);
  const int fd = ::open(ck.shards()[ti.shard].path.c_str(), O_RDONLY);
  if (fd < 0) throw std::runtime_error("dflash (DSpine): cannot open " + ck.shards()[ti.shard].path);
  size_t got = 0;
  while (got < want) {
    const ssize_t r = ::pread(fd, static_cast<char *>(dst) + got, want - got,
                              static_cast<off_t>(ti.offset + elem_off * es + got));
    if (r <= 0) break;
    got += static_cast<size_t>(r);
  }
  ::close(fd);
  if (got != want) throw std::runtime_error("dflash (DSpine): short read of " + ti.name);
}

}  // namespace

void dequant_gptq_cols_bf16(const uint32_t *qweight, const uint16_t *scales_f16, int n, int k, int col0, int ncols,
                            uint16_t *dst) {
  if (n < 1 || k < 1 || k % 128 || col0 < 0 || ncols < 1 || col0 + ncols > n)
    throw std::invalid_argument("dflash (DSpine): GPTQ lm_head dequant bounds");
  const int groups = k / 128;
  for (int c = 0; c < ncols; ++c) {
    const int col = col0 + c;
    for (int g = 0; g < groups; ++g) {
      const float s = f16_to_f32(scales_f16[static_cast<size_t>(g) * n + col]);
      for (int i = 0; i < 128; ++i) {
        const int kk = g * 128 + i;
        const uint32_t w = qweight[static_cast<size_t>(kk / 8) * n + col];
        const int q = static_cast<int>((w >> (4 * (kk % 8))) & 15u);
        dst[static_cast<size_t>(c) * k + kk] = f32_to_bf16(s * static_cast<float>(q - 8));
      }
    }
  }
}

void read_lm_head_rows_bf16(const model::Checkpoint &ck, int64_t vocab, int hidden, int64_t token0, int rows,
                            uint16_t *dst) {
  if (vocab < 1 || hidden < 1 || rows < 1 || token0 < 0 || token0 + rows > vocab)
    throw std::invalid_argument("dflash (DSpine): lm_head row range");
  const auto *w = ck.find("lm_head.weight");
  if (w && w->dtype == model::DType::bf16 && w->shape == std::vector<int64_t>{vocab, hidden}) {
    pread_exact(ck, *w, static_cast<uint64_t>(token0) * static_cast<uint64_t>(hidden),
                static_cast<uint64_t>(rows) * static_cast<uint64_t>(hidden), 2, dst);
    return;
  }
  const auto *qw = ck.find("lm_head.qweight");
  const auto *sc = ck.find("lm_head.scales");
  const auto *qz = ck.find("lm_head.qzeros");
  if (!qw || !sc)
    throw std::runtime_error(
        "dflash (DSpine): the target needs lm_head.weight bf16 [vocab][hidden] or GPTQ "
        "lm_head.qweight / .scales (int4, dequantised at load for the TG tables)");
  if (qw->dtype != model::DType::i32 || sc->dtype != model::DType::f16)
    throw std::runtime_error("dflash (DSpine): lm_head.qweight must be I32 and lm_head.scales F16");
  if (qw->shape != std::vector<int64_t>{hidden / 8, vocab} || hidden % 128)
    throw std::runtime_error("dflash (DSpine): lm_head.qweight must be I32 [hidden/8][vocab] with hidden a multiple of 128");
  if (sc->shape != std::vector<int64_t>{hidden / 128, vocab})
    throw std::runtime_error("dflash (DSpine): lm_head.scales must be F16 [hidden/128][vocab]");
  if (qz) {
    if (qz->dtype != model::DType::i32 || qz->shape != std::vector<int64_t>{hidden / 128, (vocab + 7) / 8})
      throw std::runtime_error("dflash (DSpine): lm_head.qzeros must be I32 [hidden/128][vocab/8]");
    const int64_t w0 = token0 / 8, w1 = (token0 + rows + 7) / 8;
    const int groups = hidden / 128;
    std::vector<uint32_t> words(static_cast<size_t>(w1 - w0));
    for (int g = 0; g < groups; ++g) {
      pread_exact(ck, *qz, static_cast<uint64_t>(g) * static_cast<uint64_t>((vocab + 7) / 8) + static_cast<uint64_t>(w0),
                  static_cast<uint64_t>(w1 - w0), 4, words.data());
      for (uint32_t x : words)
        if (x != 0x77777777u)
          throw std::runtime_error("dflash (DSpine): lm_head.qzeros is not the symmetric 0x77777777");
    }
  }
  const int n = static_cast<int>(vocab), k = hidden;
  const int k8 = k / 8, groups = k / 128;
  std::vector<uint32_t> qcol(static_cast<size_t>(k8) * rows);
  std::vector<uint16_t> scol(static_cast<size_t>(groups) * rows);
  for (int r = 0; r < k8; ++r)
    pread_exact(ck, *qw, static_cast<uint64_t>(r) * static_cast<uint64_t>(n) + static_cast<uint64_t>(token0),
                static_cast<uint64_t>(rows), 4, qcol.data() + static_cast<size_t>(r) * rows);
  for (int g = 0; g < groups; ++g)
    pread_exact(ck, *sc, static_cast<uint64_t>(g) * static_cast<uint64_t>(n) + static_cast<uint64_t>(token0),
                static_cast<uint64_t>(rows), 2, scol.data() + static_cast<size_t>(g) * rows);
  dequant_gptq_cols_bf16(qcol.data(), scol.data(), rows, k, 0, rows, dst);
}

}  // namespace xenithium::arch::dflash
