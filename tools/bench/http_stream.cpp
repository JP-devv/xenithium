#include "http_stream.hpp"

#include <algorithm>
#include <cctype>
#include <charconv>

namespace xenithium::bench {
namespace {

constexpr std::string_view kHeadEnd = "\r\n\r\n";
constexpr std::string_view kLineEnd = "\r\n";

std::string lower(std::string_view s) {
  std::string out(s);
  std::ranges::transform(out, out.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return out;
}

std::string_view trimmed(std::string_view s) {
  while (!s.empty() && (s.front() == ' ' || s.front() == '\t')) s.remove_prefix(1);
  while (!s.empty() && (s.back() == ' ' || s.back() == '\t' || s.back() == '\r')) s.remove_suffix(1);
  return s;
}

}  // namespace

bool HttpResponseDecoder::fail(std::string why) {
  error_ = std::move(why);
  return false;
}

// pending_ holds the whole head: the status line, then "name: value" lines.
bool HttpResponseDecoder::parse_head() {
  const size_t head_len = pending_.find(kHeadEnd);
  const std::string_view head(pending_.data(), head_len);
  if (head.size() < 12 || head.substr(0, 5) != "HTTP/") return fail("not an HTTP response: " + std::string(head.substr(0, 40)));
  const std::string_view code = head.substr(head.find(' ') + 1, 3);
  if (std::from_chars(code.data(), code.data() + code.size(), status_).ec != std::errc{}) return fail("no HTTP status code");
  state_ = State::until_close;
  for (size_t at = head.find(kLineEnd); at != std::string_view::npos;) {
    const size_t next = head.find(kLineEnd, at + kLineEnd.size());
    const std::string_view line = head.substr(at + kLineEnd.size(), next == std::string_view::npos ? std::string_view::npos : next - at - kLineEnd.size());
    at = next;
    const size_t colon = line.find(':');
    if (colon == std::string_view::npos) continue;
    const std::string name = lower(trimmed(line.substr(0, colon)));
    const std::string_view value = trimmed(line.substr(colon + 1));
    if (name == "transfer-encoding" && lower(value).find("chunked") != std::string::npos) state_ = State::chunk_size;
    if (name == "content-length" && state_ != State::chunk_size) {
      if (std::from_chars(value.data(), value.data() + value.size(), remaining_).ec != std::errc{}) return fail("bad Content-Length");
      state_ = remaining_ == 0 ? State::done : State::length;
    }
  }
  pending_.erase(0, head_len + kHeadEnd.size());
  return true;
}

bool HttpResponseDecoder::feed(std::string_view bytes, std::string &body) {
  if (!error_.empty()) return false;
  pending_.append(bytes);
  if (state_ == State::head) {
    if (pending_.find(kHeadEnd) == std::string::npos) return true;
    if (!parse_head()) return false;
  }
  while (!pending_.empty() && state_ != State::done) {
    switch (state_) {
      case State::chunk_size: {
        const size_t eol = pending_.find(kLineEnd);
        if (eol == std::string::npos) return true;
        // the size is hex, optionally followed by ";extension"
        if (std::from_chars(pending_.data(), pending_.data() + eol, remaining_, 16).ec != std::errc{}) return fail("bad chunk size");
        pending_.erase(0, eol + kLineEnd.size());
        state_ = remaining_ == 0 ? State::done : State::chunk_data;  // the last chunk; trailers are not read
        break;
      }
      case State::chunk_data:
      case State::length: {
        const size_t take = std::min(remaining_, pending_.size());
        body.append(pending_, 0, take);
        pending_.erase(0, take);
        remaining_ -= take;
        if (remaining_ == 0) state_ = state_ == State::length ? State::done : State::chunk_end;
        break;
      }
      case State::chunk_end:
        if (pending_.size() < kLineEnd.size()) return true;
        pending_.erase(0, kLineEnd.size());
        state_ = State::chunk_size;
        break;
      case State::until_close:
        body.append(pending_);
        pending_.clear();
        break;
      case State::head:
      case State::done:
        break;
    }
  }
  return true;
}

void SseSplitter::flush_line(std::string_view line, const std::function<void(std::string_view data)> &on_event) {
  if (!line.empty() && line.back() == '\r') line.remove_suffix(1);
  if (line.empty()) {  // a blank line ends the event
    if (has_data_) on_event(data_);
    data_.clear();
    has_data_ = false;
    return;
  }
  constexpr std::string_view kData = "data:";
  if (line.substr(0, kData.size()) != kData) return;  // a comment ("...") or another field
  line.remove_prefix(kData.size());
  if (!line.empty() && line.front() == ' ') line.remove_prefix(1);
  if (has_data_) data_ += '\n';
  data_.append(line);
  has_data_ = true;
}

void SseSplitter::feed(std::string_view body, const std::function<void(std::string_view data)> &on_event) {
  for (size_t at = 0; at < body.size();) {
    const size_t eol = body.find('\n', at);
    if (eol == std::string_view::npos) {
      line_.append(body.substr(at));
      return;
    }
    line_.append(body.substr(at, eol - at));
    flush_line(line_, on_event);
    line_.clear();
    at = eol + 1;
  }
}

}  // namespace xenithium:bench
