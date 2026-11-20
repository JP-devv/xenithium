// The launcher's chat with the served model: the conversation state, the worker that streams an
// answer over the OpenAI endpoint, and cancel. The tab itself (drawing, keys) lives in app.cpp.
// Semantics: a send appends the user turn and an empty assistant turn, then a worker POSTs
// /v1/chat/completions with stream: true and folds the content/reasoning deltas into the assistant
// turn. History = every finished turn, re-sent on each request. A cancelled or failed answer is
// dropped from the history (it is truncated, not an answer); the user turn stays, so the next
// send continues the conversation without the garbage.
#pragma once

#include <atomic>
#include <chrono>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace xenithium::launch {

class ChatSession {
 public:
  using Clock = std::chrono::steady_clock;

  struct Msg {
    std::string role;  // "user" or "assistant"
    std::string reasoning;  // assistant only: the thinking deltas
    std::string text;  // the content
    bool done = false;  // assistant only: the answer finished
  };
  enum class Status : uint8_t { idle, running, done, error, cancelled };

  struct State {
    std::vector<Msg> msgs;
    Status status = Status::idle;
    std::string error;  // status == error: what went wrong
    Clock::time_point started{};  // the running / last request (elapsed = now - started while running)
    double elapsed = 0;  // seconds of the last finished request
    long long tokens = -1;  // completion tokens of the last answer (usage chunk); -1 unknown
  };

  ChatSession();
  ~ChatSession();
  ChatSession(const ChatSession &) = delete;
  ChatSession &operator=(const ChatSession &) = delete;

  State get() const;
  bool busy() const;  // a request is streaming
  bool send(const std::string &text, int port);  // false when one is still running
  void cancel();  // aborts the stream (the socket closes; the server sees it)
  void clear();  // drops the conversation (refused while busy)

 private:
  void run(std::string body, int port);  // the worker: connect, POST, parse the SSE

  mutable std::mutex mu_;
  std::vector<Msg> msgs_;
  Status status_ = Status::idle;
  std::string error_;
  Clock::time_point started_{};
  double elapsed_ = 0;
  long long tokens_ = -1;
  std::thread th_;
  std::atomic<bool> cancel_{false};
  std::atomic<int> fd_{-1};  // the request socket, so cancel can unblock the recv
};

}  // namespace xenithium:launch
