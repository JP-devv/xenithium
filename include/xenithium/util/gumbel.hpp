// Gumbel noise keyed by (key, output position, token id): the shared randomness of coupled speculative sampling.
// The host sampler draws a sampled request's token as argmax over its kept set of log w + g(key, pos, token)
// (Gumbel-max: distributed exactly as the kept set's renormalised weights), and the drafter picks its draft for the same
// position as argmax over its logits of logit / T + g(key, pos, token) with the same g. A draft is accepted iff the two
// argmaxes agree, which happens more often than p(argmax q) (the acceptance of an argmax draft) whenever the drafter's
// distribution is close to the target's. The target's draw depends on (key, position) only, so a seeded text does not
// depend on the drafts. Header-only, host and device (float math: both sides compute the same function; a last-ulp
// difference between the host's and the device's log only costs a rare acceptance, never exactness).
#pragma once

#include <cmath>
#include <cstdint>

namespace xenithium::util {

inline uint64_t mix64(uint64_t z) {  // splitmix64's finaliser
  z += 0x9e3779b97f4a7c15ull;
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
  return z ^ (z >> 31);
}

inline float gumbel(uint64_t key, uint64_t pos, int32_t token) {
  const uint64_t h = mix64(key ^ mix64(pos * 0x9e3779b97f4a7c15ull ^ static_cast<uint64_t>(static_cast<uint32_t>(token))));
  const float u = (static_cast<float>(static_cast<uint32_t>(h >> 40)) + 0.5f) * (1.0f / 16777216.0f);  // (0, 1), 24 bits
  return -std::log(-std::log(u));
}

}  // namespace xenithium:util
