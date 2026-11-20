// OpenAI-compatible HTTP server over the core's Engine interface.
// One thread runs an epoll loop over the listening socket, every connection and an
// eventfd the engine signals once per step; generation requests are parsed, rendered and
// tokenized on prepare threads (Config:prepare_threads) and come back through the same eventfd. Engine events are handed over under a
// short lock and drained in the loop, so the engine thread never touches a socket
// and the server never waits on a step. Every connection that received tokens in a
// step is written once (one HTTP chunk per step), with TCP_NODELAY so a streamed
// step leaves the host as soon as it is decoded.
// Endpoints: POST /v1/completions, POST /v1/chat/completions (both with SSE streaming,
// `n` choices, logprobs/echo), GET /v1/models and /v1/models/{id}, /health, /version,
// /metrics (Prometheus, including the speculative-decoding counters), CORS preflight.
// Request surface: sampling parameters (temperature, top_p,
// top_k, min_p, seed, penalties, logit_bias, stop/stop_token_ids), max_tokens /
// max_completion_tokens, tool_choice (auto / none / required / a named function).
// Structured output (response_format json_object / json_schema, guided_json /
// guided_choice, structured_outputs json / choice) is enforced token by token
// (text/json_constraint.hpp); guided_regex / guided_grammar answer 400. No API key: `Authorization` is accepted and ignored. A client that
// disconnects mid-generation cancels its request in the engine immediately.
#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "xenithium/core/engine.hpp"
#include "xenithium/server/http.hpp"
#include "xenithium/text/chat.hpp"
#include "xenithium/text/tokenizer.hpp"
#include "xenithium/util/fetch.hpp"
#include "xenithium/util/image.hpp"

namespace xenithium::server {

// Image input. The model side of an image content part (data: URLs only): the server decodes the image
// (util:decode_image) on a prepare thread and hands it here; the model registers it and answers with the token ids
// that name it in the prompt. The chat template prints one placeholder token (image_token) per image; the server
// replaces the k-th one with the k-th image's ids. The lease keeps the image's rows while the request lives.
class MediaHook {
 public:
  virtual ~MediaHook() = default;
  struct Tokens {
    std::shared_ptr<void> lease;
    int64_t base = 0;  // ids base.. base + count - 1
    int count = 0;
  };
  // false + err: status 400 (an image the model cannot take) or 503 (no room now: retry).
  virtual bool add_image(const util::Image &img, Tokens &out, int &status, std::string &err) = 0;
  virtual int32_t image_token() const = 0;
  virtual int64_t max_pixels() const = 0;  // the cap a larger image is scaled down to
  // Video input. A video file's bytes -> one Tokens per temporal patch (two frames) and the text shown before
  // each ("<1.5 seconds>"); the template prints one video placeholder token (video_token) per video, which the server
  // replaces by, per frame, the timestamp's tokens, <|vision_start|>, the frame's ids, <|vision_end|>.
  // false + err as add_image; a hook without video support answers 400 with its reason.
  virtual bool add_video(const std::string &bytes, std::vector<Tokens> &frames, std::vector<std::string> &stamps, int &status,
                         std::string &err) {
    (void)bytes, (void)frames, (void)stamps;
    status = 400;
    err = "video input is not supported by this server";
    return false;
  }
  virtual int32_t video_token() const { return -1; }
  virtual int32_t vision_start() const { return -1; }
  virtual int32_t vision_end() const { return -1; }
  // The pad token shown at a virtual id (image or video): prompt logprobs, penalties.
  virtual int32_t pad_token(int64_t id) const { (void)id; return image_token(); }
};

struct Config {
  std::string host = "0.0.0.0";  // every interface: reachable from the LAN
  int port = 8000;  // 0 = pick a free port (tests)
  int backlog = 1024;
  http::Limits limits;
  size_t max_output_buffer = 64ull * 1024 * 1024;  // per connection; a slower reader is dropped
  int idle_timeout_s = 300;  // close a connection with no traffic and no request; 0 = never
  std::string cors_origin = "*";  // Access-Control-Allow-Origin; empty = no CORS headers
  // Requests allowed to wait for a slot (beyond the ones running); one more gets 429 with
  // Retry-After instead of queueing. 0 = unlimited. Needs an engine that reports its slots.
  int max_queue = 0;
  // Structured output: the logits row length the token masks cover (0 = the tokenizer's
  // ids) and the ids that may end a constrained document.
  int64_t vocab = 0;
  std::vector<int32_t> eos_ids;
  // Threads that parse, render and tokenize generation requests, so a long prompt or a
  // first grammar compile does not stall the loop's streams. 0 = on the loop thread.
  int prepare_threads = 2;
  // Image input (nullptr = text only: image parts answer 400).
  MediaHook* media = nullptr;
  // How media URLs are loaded (data: always; http(s) fetched unless fetch.enabled is false).
  util::FetchOptions fetch;
};

class Server {
 public:
  // chat: the model's chat format; nullptr = /v1/chat/completions answers 400.
  Server(Config cfg, core::Engine& engine, const text::Tokenizer& tok, const text::ChatFormat* chat);
  ~Server();
  Server(const Server&) = delete;
  Server& operator=(const Server&) = delete;

  bool listen(std::string& err);
  int port() const;
  core::EventSink* sink();  // pass to Engine:start
  void run();  // serves until stop
  void stop();  // any thread; async-signal-safe
  size_t live_generations() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace xenithium:server
