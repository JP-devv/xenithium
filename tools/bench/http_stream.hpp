// xenithium-bench: an HTTP/1.1 response and a server-sent-event stream, decoded as the bytes arrive.
// Both are plain state machines over the bytes fed so far (no sockets), so a request can time-stamp each event when
// its bytes came in, and the tests feed them split at every byte.
#pragma once

#include <cstddef>
#include <functional>
#include <string>
#include <string_view>

namespace xenithium::bench {

// The body of one HTTP/1.1 response: chunked, Content-Length, or until the peer closes.
class HttpResponseDecoder {
 public:
  // Feeds received bytes; the body bytes they complete are appended to `body. False = not HTTP (error says why).
  bool feed(std::string_view bytes, std::string &body);
  bool head_done() const { return state_ != State::head; }
  int status() const { return status_; }
  // The body ended by its own framing (a chunked or Content-Length body); a close-delimited body ends with the socket.
  bool done() const { return state_ == State::done; }
  bool until_close() const { return state_ == State::until_close; }
  const std::string &error() const { return error_; }

 private:
  enum class State { head, chunk_size, chunk_data, chunk_end, length, until_close, done };
  bool parse_head();
  bool fail(std::string why);

  State state_ = State::head;
  std::string pending_;  // bytes not yet consumed (a split head, chunk-size line or chunk end)
  size_t remaining_ = 0;  // bytes left of the current chunk / of a Content-Length body
  int status_ = 0;
  std::string error_;
};

// Splits a text/event-stream body into events and hands each event's data (its `data:` lines joined by '\n') to the
// callback. Comment lines and other fields are skipped; both "\n\n" and "\r\n\r\n" end an event.
class SseSplitter {
 public:
  void feed(std::string_view body, const std::function<void(std::string_view data)> &on_event);

 private:
  void flush_line(std::string_view line, const std::function<void(std::string_view data)> &on_event);

  std::string line_;  // the current, unfinished line
  std::string data_;  // the current event's data so far
  bool has_data_ = false;
};

}  // namespace xenithium:bench
