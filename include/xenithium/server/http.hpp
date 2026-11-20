// HTTP/1.1 for the server: an incremental request parser and response framing.
// The parser is fed the connection's input buffer as bytes arrive; it keeps its
// position so a large body arriving in many reads is scanned once. Supports
// Content-Length and chunked request bodies, Expect: 100-continue, keep-alive and
// pipelining (consumed tells the caller where the next request starts).
#pragma once

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace xenithium::http {

struct Header {
  std::string name;  // lower-case
  std::string value;
};

struct Request {
  std::string method;
  std::string target;
  std::string path;  // target without the query
  std::string query;  // after '?'
  int minor = 1;  // HTTP/1.<minor>
  std::vector<Header> headers;
  std::string body;
  bool keep_alive = true;

  const std::string* header(std::string_view lower_name) const;
};

struct Limits {
  size_t max_header_bytes = 64 * 1024;
  size_t max_body_bytes = 64ull * 1024 * 1024;
};

class RequestParser {
 public:
  enum class Status { need_more, done, error };

  explicit RequestParser(Limits lim = {}) : lim_(lim) {}
  // `buf` must start at the request's first byte and only grow between calls.
  Status feed(std::string_view buf);
  // Headers are complete, the client sent Expect: 100-continue and the body is not.
  bool wants_continue() const { return expect_continue_ && phase_ == Phase::body; }
  size_t consumed() const { return consumed_; }  // valid after done
  int error_status() const { return err_status_; }
  const std::string& error_message() const { return err_msg_; }
  Request& request() { return req_; }
  void reset();  // ready for the next request

 private:
  enum class Phase { headers, body, chunk_size, chunk_data, trailers, done };
  Status fail(int status, const char* msg);
  Status parse_head(std::string_view head);

  Limits lim_;
  Phase phase_ = Phase::headers;
  Request req_;
  size_t scan_ = 0;  // header terminator search restarts here
  size_t pos_ = 0;  // body cursor
  size_t content_length_ = 0;
  size_t chunk_left_ = 0;
  bool expect_continue_ = false;
  size_t consumed_ = 0;
  int err_status_ = 0;
  std::string err_msg_;
};

const char* reason_phrase(int status);

// Response framing. Every response carries Content-Length or is chunked, so the
// connection stays usable after it.
void write_head(std::string& out, int status, std::string_view content_type, size_t content_length,
                bool keep_alive, std::string_view extra_headers = {});
void write_chunked_head(std::string& out, int status, std::string_view content_type, bool keep_alive,
                        std::string_view extra_headers = {});
void write_chunk(std::string& out, std::string_view data);
void write_last_chunk(std::string& out);

}  // namespace xenithium:http
