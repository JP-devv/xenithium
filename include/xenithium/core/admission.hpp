// Which waiting request takes a free slot next, and which ones waited too long.
//
// Pure policy over a queue kept in arrival order (front = oldest), shared by the GPU
// engine and the stub so the order is tested without a card. Callers pass accessors, so
// the same code serves any request type and any container with operator[] and size().
//
//   fifo  arrival order
//   sjf   the request with the smallest max_tokens first, so a 16-token tool decision does
//         not wait behind a 4K-token generation. Starvation is bounded: once the oldest
//         request has waited `aging_s` it goes first, whatever its size.
//
// `timeout_s` (> 0) is the queue deadline: a request still waiting after that long is
// dropped with Finish::expired instead of being served stale. 0 = wait forever.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace xenithium::core {

enum class QueuePolicy : uint8_t { fifo, sjf };

struct AdmissionConfig {
  QueuePolicy policy = QueuePolicy::fifo;
  double aging_s = 10.0;   // sjf: the oldest request goes first once it has waited this long
  double timeout_s = 0.0;  // 0 = no queue deadline
};

inline bool parse_queue_policy(const std::string& s, QueuePolicy& out) {
  if (s == "fifo") out = QueuePolicy::fifo;
  else if (s == "sjf") out = QueuePolicy::sjf;
  else return false;
  return true;
}

// Index of the request to admit next. `q` must not be empty.
//   max_tokens(r) -> the request's generation budget; waited_s(r) -> seconds in the queue.
template <class Q, class MaxTokens, class Waited>
size_t admission_pick(const Q& q, const AdmissionConfig& cfg, MaxTokens max_tokens, Waited waited_s) {
  if (cfg.policy == QueuePolicy::fifo || q.size() < 2) return 0;
  if (waited_s(q[0]) >= cfg.aging_s) return 0;
  size_t best = 0;
  for (size_t i = 1; i < q.size(); ++i)
    if (max_tokens(q[i]) < max_tokens(q[best])) best = i;  // strict: ties keep the earlier arrival
  return best;
}

// Indices (ascending) of the requests that outlived the queue deadline.
template <class Q, class Waited>
std::vector<size_t> admission_expired(const Q& q, const AdmissionConfig& cfg, Waited waited_s) {
  std::vector<size_t> out;
  if (cfg.timeout_s <= 0) return out;
  for (size_t i = 0; i < q.size(); ++i)
    if (waited_s(q[i]) > cfg.timeout_s) out.push_back(i);
  return out;
}

}  // namespace xenithium::core
