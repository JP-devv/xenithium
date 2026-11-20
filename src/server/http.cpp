#include "xenithium/server/http.hpp"

#include <charconv>
#include <cstdio>

namespace xenithium::http {

namespace {

char lower(char c) { return (c >= 'A' && c <= 'Z') ? static_cast<char>(c + 32) : c; }

bool ieq(std::string_view a, std::string_view b) {
  if (a.size() != b.size()) return false;
  for (size_t i = 0; i < a.size(); ++i)
    if (lower(a[i]) != lower(b[i])) return false;
  return true;
}

bool tchar(char c) {
  if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9')) return true;
  switch (c) {
    case '!': case '#': case '$': case '%': case '&': case '\'': case '*': case '+':
    case '-': case '.': case '^': case '_': case '`': case '|': case '~': return true;
    default: return false;
  }
}

std::string_view trim_ows(std::string_view s) {
  while (!s.empty() && (s.front() == ' ' || s.front() == '\t')) s.remove_prefix(1);
  while (!s.empty() && (s.back() == ' ' || s.back() == '\t')) s.remove_suffix(1);
  return s;
}

// Calls f(token) for each comma-separated token (trimmed, possibly empty).
template <class F>
void each_token(std::string_view v, F&& f) {
  while (true) {
    const size_t c = v.find(',');
    f(trim_ows(v.substr(0, c)));
    if (c == std::string_view::npos) return;
    v.remove_prefix(c + 1);
  }
}

}  // namespace

const std::string* Request::header(std::string_view lower_name) const {
  for (const auto& h : headers)
    if (h.name == lower_name) return &h.value;
  return nullptr;
}

void RequestParser::reset() {
  phase_ = Phase::headers;
  req_ = Request{};
  scan_ = pos_ = content_length_ = chunk_left_ = consumed_ = 0;
  expect_continue_ = false;
  err_status_ = 0;
  err_msg_.clear();
}

RequestParser::Status RequestParser::fail(int status, const char* msg) {
  err_status_ = status;
  err_msg_ = msg;
  return Status::error;
}

RequestParser::Status RequestParser::parse_head(std::string_view head) {
  // Request line.
  size_t eol = head.find("\r\n");
  std::string_view line = head.substr(0, eol);
  const size_t sp1 = line.find(' ');
  const size_t sp2 = sp1 == std::string_view::npos ? sp1 : line.find(' ', sp1 + 1);
  if (sp1 == std::string_view::npos || sp2 == std::string_view::npos || sp1 == 0 || sp2 == sp1 + 1)
    return fail(400, "malformed request line");
  const std::string_view method = line.substr(0, sp1);
  for (char c : method)
    if (!tchar(c)) return fail(400, "malformed method");
  const std::string_view target = line.substr(sp1 + 1, sp2 - sp1 - 1);
  const std::string_view version = line.substr(sp2 + 1);
  if (version.size() != 8 || version.compare(0, 7, "HTTP/1.") != 0 || version[7] < '0' || version[7] > '9')
    return fail(505, "unsupported HTTP version");
  req_.method.assign(method);
  req_.target.assign(target);
  const size_t q = target.find('?');
  req_.path.assign(target.substr(0, q));
  if (q != std::string_view::npos) req_.query.assign(target.substr(q + 1));
  req_.minor = version[7] - '0';

  // Header fields.
  bool have_cl = false, chunked = false, conn_close = false, conn_keep = false;
  size_t cl = 0;
  std::string_view rest = eol == std::string_view::npos ? std::string_view{} : head.substr(eol + 2);
  while (!rest.empty()) {
    eol = rest.find("\r\n");
    line = rest.substr(0, eol);
    rest = eol == std::string_view::npos ? std::string_view{} : rest.substr(eol + 2);
    if (line.empty()) continue;
    if (line.front() == ' ' || line.front() == '\t') return fail(400, "obsolete header folding");
    const size_t colon = line.find(':');
    if (colon == std::string_view::npos || colon == 0) return fail(400, "malformed header");
    const std::string_view name = line.substr(0, colon);
    for (char c : name)
      if (!tchar(c)) return fail(400, "malformed header name");
    const std::string_view value = trim_ows(line.substr(colon + 1));
    Header h;
    h.name.reserve(name.size());
    for (char c : name) h.name.push_back(lower(c));
    h.value.assign(value);

    if (h.name == "content-length") {
      size_t v = 0;
      auto r = std::from_chars(value.data(), value.data() + value.size(), v);
      if (r.ec != std::errc() || r.ptr != value.data() + value.size() || value.empty())
        return fail(400, "bad Content-Length");
      if (have_cl && v != cl) return fail(400, "conflicting Content-Length");
      have_cl = true;
      cl = v;
    } else if (h.name == "transfer-encoding") {
      std::string_view last;
      each_token(value, [&](std::string_view t) {
        if (!t.empty()) last = t;
      });
      if (!ieq(last, "chunked")) return fail(501, "unsupported Transfer-Encoding");
      chunked = true;
    } else if (h.name == "connection") {
      each_token(value, [&](std::string_view t) {
        if (ieq(t, "close")) conn_close = true;
        if (ieq(t, "keep-alive")) conn_keep = true;
      });
    } else if (h.name == "expect") {
      if (!ieq(value, "100-continue")) return fail(417, "unsupported Expect");
      expect_continue_ = req_.minor >= 1;
    }
    req_.headers.push_back(std::move(h));
  }
  req_.keep_alive = req_.minor >= 1 ? !conn_close : (conn_keep && !conn_close);

  if (chunked) {
    phase_ = Phase::chunk_size;
  } else {
    if (cl > lim_.max_body_bytes) return fail(413, "request body too large");
    content_length_ = cl;
    phase_ = Phase::body;
  }
  return Status::need_more;
}

RequestParser::Status RequestParser::feed(std::string_view buf) {
  if (phase_ == Phase::done) return Status::done;
  if (phase_ == Phase::headers) {
    const size_t from = scan_ >= 3 ? scan_ - 3 : 0;
    const size_t end = buf.find("\r\n\r\n", from);
    if (end == std::string_view::npos) {
      scan_ = buf.size();
      if (buf.size() > lim_.max_header_bytes) return fail(431, "request header too large");
      return Status::need_more;
    }
    if (end + 4 > lim_.max_header_bytes) return fail(431, "request header too large");
    if (parse_head(buf.substr(0, end)) == Status::error) return Status::error;
    pos_ = end + 4;
  }

  if (phase_ == Phase::body) {
    if (buf.size() - pos_ < content_length_) return Status::need_more;
    req_.body.assign(buf.substr(pos_, content_length_));
    consumed_ = pos_ + content_length_;
    phase_ = Phase::done;
    expect_continue_ = false;
    return Status::done;
  }

  while (true) {
    if (phase_ == Phase::chunk_size) {
      const size_t eol = buf.find("\r\n", pos_);
      if (eol == std::string_view::npos) {
        if (buf.size() - pos_ > 1024) return fail(400, "chunk size line too long");
        return Status::need_more;
      }
      std::string_view line = buf.substr(pos_, eol - pos_);
      line = line.substr(0, line.find(';'));  // drop extensions
      line = trim_ows(line);
      size_t n = 0;
      auto r = std::from_chars(line.data(), line.data() + line.size(), n, 16);
      if (line.empty() || r.ec != std::errc() || r.ptr != line.data() + line.size())
        return fail(400, "bad chunk size");
      pos_ = eol + 2;
      if (n == 0) {
        phase_ = Phase::trailers;
      } else {
        if (req_.body.size() + n > lim_.max_body_bytes) return fail(413, "request body too large");
        chunk_left_ = n;
        phase_ = Phase::chunk_data;
      }
    }
    if (phase_ == Phase::chunk_data) {
      if (buf.size() - pos_ < chunk_left_ + 2) return Status::need_more;
      req_.body.append(buf.substr(pos_, chunk_left_));
      if (buf.compare(pos_ + chunk_left_, 2, "\r\n") != 0) return fail(400, "bad chunk terminator");
      pos_ += chunk_left_ + 2;
      phase_ = Phase::chunk_size;
      continue;
    }
    if (phase_ == Phase::trailers) {
      // Trailer fields are ignored; an empty line ends the message.
      while (true) {
        const size_t eol = buf.find("\r\n", pos_);
        if (eol == std::string_view::npos) {
          if (buf.size() - pos_ > lim_.max_header_bytes) return fail(431, "trailers too large");
          return Status::need_more;
        }
        const bool empty = eol == pos_;
        pos_ = eol + 2;
        if (empty) {
          consumed_ = pos_;
          phase_ = Phase::done;
          expect_continue_ = false;
          return Status::done;
        }
      }
    }
  }
}

const char* reason_phrase(int status) {
  switch (status) {
    case 100: return "Continue";
    case 200: return "OK";
    case 204: return "No Content";
    case 400: return "Bad Request";
    case 404: return "Not Found";
    case 405: return "Method Not Allowed";
    case 408: return "Request Timeout";
    case 411: return "Length Required";
    case 413: return "Content Too Large";
    case 417: return "Expectation Failed";
    case 422: return "Unprocessable Content";
    case 429: return "Too Many Requests";
    case 431: return "Request Header Fields Too Large";
    case 500: return "Internal Server Error";
    case 501: return "Not Implemented";
    case 503: return "Service Unavailable";
    case 505: return "HTTP Version Not Supported";
    default: return "Unknown";
  }
}

namespace {

void status_line(std::string& out, int status) {
  char b[64];
  const int n = std::snprintf(b, sizeof(b), "HTTP/1.1 %d %s\r\n", status, reason_phrase(status));
  out.append(b, static_cast<size_t>(n));
}

void common(std::string& out, std::string_view content_type, bool keep_alive, std::string_view extra) {
  if (!content_type.empty()) {
    out += "Content-Type: ";
    out += content_type;
    out += "\r\n";
  }
  out += keep_alive ? "Connection: keep-alive\r\n" : "Connection: close\r\n";
  out += "Server: xenithium\r\n";
  out += extra;
}

}  // namespace

void write_head(std::string& out, int status, std::string_view content_type, size_t content_length,
                bool keep_alive, std::string_view extra_headers) {
  status_line(out, status);
  common(out, content_type, keep_alive, extra_headers);
  char b[48];
  const int n = std::snprintf(b, sizeof(b), "Content-Length: %zu\r\n\r\n", content_length);
  out.append(b, static_cast<size_t>(n));
}

void write_chunked_head(std::string& out, int status, std::string_view content_type, bool keep_alive,
                        std::string_view extra_headers) {
  status_line(out, status);
  common(out, content_type, keep_alive, extra_headers);
  out += "Transfer-Encoding: chunked\r\n\r\n";
}

void write_chunk(std::string& out, std::string_view data) {
  if (data.empty()) return;  // a zero-size chunk would end the body
  char b[24];
  const int n = std::snprintf(b, sizeof(b), "%zx\r\n", data.size());
  out.append(b, static_cast<size_t>(n));
  out += data;
  out += "\r\n";
}

void write_last_chunk(std::string& out) { out += "0\r\n\r\n"; }

}  // namespace xenithium:http
