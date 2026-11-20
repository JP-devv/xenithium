#include "xenithium/kernels/embed/embed_aff.hpp"

#include <cmath>
#include <cstring>
#include <stdexcept>

namespace xenithium::kernels {
namespace {

inline float bits_f(uint16_t b) {
  const uint32_t u = static_cast<uint32_t>(b) << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

// fp32 -> bf16 bits, round to nearest even (no NaN in this data)
inline uint16_t f_bits(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  u += 0x7fffu + ((u >> 16) & 1u);
  return static_cast<uint16_t>(u >> 16);
}

class EmbedAff8;

}  // namespace

sycl::event embed_aff8(sycl::queue &q, int rows, const int32_t *ids, const uint8_t *table, const uint16_t *scales, const uint16_t *biases,
                       int64_t vocab, int hidden, sycl::ext::oneapi::bfloat16 *out, int ldo) {
  if (rows < 1 || hidden % kEmbedAffGroup || ldo < hidden) throw std::invalid_argument("embed_aff8: shape (hidden % 64, ldo >= hidden)");
  const int groups = hidden / kEmbedAffGroup;
  auto *o = reinterpret_cast<uint16_t *>(out);
  return q.parallel_for<EmbedAff8>(sycl::range<2>(static_cast<size_t>(rows), static_cast<size_t>(hidden)), [=](sycl::id<2> ix) {
    const int r = static_cast<int>(ix[0]), c = static_cast<int>(ix[1]);
    const int64_t id = ids[r];
    uint16_t v = 0;
    if (id >= 0 && id < vocab) {
      const size_t gi = static_cast<size_t>(id) * groups + c / kEmbedAffGroup;
      uint32_t su = static_cast<uint32_t>(scales[gi]) << 16, bu = static_cast<uint32_t>(biases[gi]) << 16;
      const float s = sycl::bit_cast<float>(su), b = sycl::bit_cast<float>(bu);
      const float x = sycl::fma(static_cast<float>(table[static_cast<size_t>(id) * hidden + c]), s, b);
      uint32_t u = sycl::bit_cast<uint32_t>(x);
      u += 0x7fffu + ((u >> 16) & 1u);
      v = static_cast<uint16_t>(u >> 16);
    }
    o[static_cast<size_t>(r) * ldo + c] = v;
  });
}

void embed_aff8_ref(int32_t id, const uint8_t *table, const uint16_t *scales, const uint16_t *biases, int64_t vocab, int hidden, uint16_t *out) {
  const int groups = hidden / kEmbedAffGroup;
  for (int c = 0; c < hidden; ++c) {
    if (id < 0 || id >= vocab) {
      out[c] = 0;
      continue;
    }
    const size_t gi = static_cast<size_t>(id) * groups + c / kEmbedAffGroup;
    out[c] = f_bits(std::fma(static_cast<float>(table[static_cast<size_t>(id) * hidden + c]), bits_f(scales[gi]), bits_f(biases[gi])));
  }
}

}  // namespace xenithium::kernels
