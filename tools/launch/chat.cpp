// The chat worker: a raw HTTP/1.1 POST to the local server with an incremental
// reader of the chunked SSE answer. Raw sockets, not http:request, because the answer must be
// shown while it streams and Esc must abort it mid-flight (http:request reads to the end).
#include "chat.hpp"

#include <fcntl.h>
#include <netdb.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#include "xenithium/util/json.hpp"

namespace xenithium::launch {

namespace {

using Clock = std::chrono::steady_clock;

struct Fd {
  int fd = -1;
  ~Fd() {
    if (fd >= 0) ::close(fd);
  }
};

// Waits for the socket to be ready. True = ready; false = the wait ended (timeout); throws when
// cancel is set, or on a poll error. The read loop calls it again after a false, so a long
// silence before the first token is not an error - only cancel ends a stalled stream.
bool wait(int fd, short ev, int ms, const std::atomic<bool> &cancel) {
  const auto deadline = Clock::now() + std::chrono::milliseconds(ms);
  for (;;) {
    if (cancel.load()) throw std::runtime_error("cancelled");
    const long long left =
        std::chrono::duration_cast<std::chrono::milliseconds>(deadline - Clock::now()).count();
    if (left <= 0) return false;
    pollfd p{fd, ev, 0};
    const int r = ::poll(&p, 1, static_cast<int>(std::min<long long>(left, 1000)));
    if (r > 0) return true;
    if (r < 0 && errno != EINTR) throw std::runtime_error("poll: " + std::string(std::strerror(errno)));
  }
}

// Connect to 127.0.0.1:port; -1 when every address failed (no route), throws on timeout / cancel.
int connect_to(int port, const std::atomic<bool> &cancel) {
  addrinfo hints{}, *res = nullptr;
  hints.ai_socktype = SOCK_STREAM;
  char ps[16];
  std::snprintf(ps, sizeof ps, "%d", port);
  if (::getaddrinfo("127.0.0.1", ps, &hints, &res) != 0 || !res) return -1;
  int fd = -1;
  for (addrinfo *a = res; a && fd < 0; a = a->ai_next) {
    fd = ::socket(a->ai_family, a->ai_socktype, a->ai_protocol);
    if (fd < 0) continue;
    const int fl = ::fcntl(fd, F_GETFL, 0);
    ::fcntl(fd, F_SETFL, fl | O_NONBLOCK);
    if (::connect(fd, a->ai_addr, a->ai_addrlen) != 0 && errno != EINPROGRESS) {
      ::close(fd);
      fd = -1;
      continue;
    }
    try {
      if (!wait(fd, POLLOUT, 3000, cancel)) throw std::runtime_error("connect timed out");
      int soerr = 0;
      socklen_t sl = sizeof soerr;
      if (::getsockopt(fd, SOL_SOCKET, SO_ERROR, &soerr, &sl) != 0 || soerr != 0) {
        ::close(fd);
        fd = -1;
        continue;
      }
      ::fcntl(fd, F_SETFL, fl);  // blocking again; cancel unblocks the recv with shutdown
    } catch (...) {
      ::close(fd);
      ::freeaddrinfo(res);
      throw;
    }
  }
  ::freeaddrinfo(res);
  return fd;
}

// Incremental reader of an HTTP response with an SSE body (the server answers a stream with
// transfer-encoding: chunked; content-length and close-delimited bodies are read too).
struct SseReader {
  bool headers_done = false;
  int status = 0;
  bool chunked = false;
  long long content_left = -1;  // -1 = read to the close
  bool finished = false;
  std::string buf;  // unparsed bytes (header then chunk framing)
  std::string evbuf;  // complete body bytes not yet split into events

  bool feed(const std::string &bytes, std::string &err) {
    buf += bytes;
    if (!headers_done) {
      const size_t he = buf.find("\r\n\r\n");
      if (he == std::string::npos) {
        if (buf.size() > (1 << 16)) {
          err = "no HTTP response header";
          return false;
        }
        return true;
      }
      const std::string head = buf.substr(0, he);
      if (head.rfind("HTTP/1.", 0) != 0 || head.find(' ') == std::string::npos) {
        err = "no HTTP response header";
        return false;
      }
      status = std::atoi(head.c_str() + head.find(' ') + 1);
      std::string low = head;
      std::transform(low.begin(), low.end(), low.begin(),
                     [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
      chunked = low.find("transfer-encoding: chunked") != std::string::npos;
      content_left = -1;
      if (!chunked) {
        const size_t cl = low.find("content-length:");
        if (cl != std::string::npos) content_left = std::strtoll(low.c_str() + cl + 15, nullptr, 10);
      }
      buf.erase(0, he + 4);
      headers_done = true;
    }
    for (;;) {
      if (chunked) {
        const size_t le = buf.find("\r\n");
        if (le == std::string::npos) break;
        std::string size_line = buf.substr(0, le);
        const size_t semi = size_line.find(';');
        if (semi != std::string::npos) size_line.resize(semi);
        char *end = nullptr;
        const long long n = std::strtoll(size_line.c_str(), &end, 16);
        if (end == size_line.c_str()) {
          err = "bad chunk size";
          return false;
        }
        if (n == 0) {  // the last chunk: trailers may follow, they carry nothing for us
          finished = true;
          return true;
        }
        if (n < 0 || static_cast<size_t>(n) + 2 > buf.size() - le - 2) break;  // wait for the whole chunk
        evbuf.append(buf, le + 2, static_cast<size_t>(n));
        buf.erase(0, le + 2 + static_cast<size_t>(n) + 2);  // the chunk plus its trailing \r\n
      } else if (content_left >= 0) {
        const size_t take = static_cast<size_t>(std::min<long long>(content_left, static_cast<long long>(buf.size())));
        evbuf.append(buf, 0, take);
        buf.erase(0, take);
        content_left -= static_cast<long long>(take);
        if (content_left == 0) finished = true;
        return true;
      } else {
        evbuf += buf;  // close-delimited: everything is body
        buf.clear();
        return true;
      }
    }
    return true;
  }

  // One complete SSE event (the concatenated data: payloads). False when none is complete yet.
  bool next_event(std::string &payload) {
    for (;;) {
      const size_t e = evbuf.find("\n\n");
      if (e == std::string::npos) return false;
      const std::string event = evbuf.substr(0, e);
      evbuf.erase(0, e + 2);
      payload.clear();
      bool any = false;
      size_t p = 0;
      for (;;) {
        const size_t le = event.find('\n', p);
        std::string_view line(event.data() + p, (le == std::string::npos ? event.size() : le) - p);
        if (!line.empty() && line.back() == '\r') line.remove_suffix(1);
        if (line.rfind("data:", 0) == 0) {
          std::string_view d = line.substr(5);
          if (!d.empty() && d.front() == ' ') d.remove_prefix(1);
          if (!payload.empty()) payload += '\n';
          payload.append(d);
          any = true;
        }
        if (le == std::string::npos) break;
        p = le + 1;
      }
      if (any) return true;
    }
  }
};

}  // namespace

ChatSession::ChatSession() = default;

ChatSession::~ChatSession() {
  cancel_ = true;
  const int fd = fd_.load();
  if (fd > 0) ::shutdown(fd, SHUT_RDWR);
  if (th_.joinable()) th_.join();
}

ChatSession::State ChatSession::get() const {
  std::lock_guard<std::mutex> g(mu_);
  return State{msgs_, status_, error_, started_, elapsed_, tokens_};
}

bool ChatSession::busy() const {
  std::lock_guard<std::mutex> g(mu_);
  return status_ == Status::running;
}

bool ChatSession::send(const std::string &text, int port) {
  if (th_.joinable()) th_.join();
  {
    std::lock_guard<std::mutex> g(mu_);
    if (status_ == Status::running) return false;
    msgs_.push_back({"user", "", text, true});
    msgs_.push_back({"assistant", "", "", false});
    status_ = Status::running;
    error_.clear();
    elapsed_ = 0;
    tokens_ = -1;
    started_ = Clock::now();
  }
  // The request body: every finished turn of the conversation.
  json::Value req = json::Value::object();
  json::Value arr = json::Value::array();
  {
    std::lock_guard<std::mutex> g(mu_);
    for (const Msg &m : msgs_) {
      if (!m.done) continue;  // the fresh assistant placeholder
      json::Value o = json::Value::object();
      o.set("role", json::Value::string(m.role));
      o.set("content", json::Value::string(m.text));
      arr.push(std::move(o));
    }
  }
  req.set("messages", std::move(arr));
  req.set("stream", json::Value::boolean(true));
  json::Value so = json::Value::object();
  so.set("include_usage", json::Value::boolean(true));
  req.set("stream_options", std::move(so));
  cancel_ = false;
  th_ = std::thread([this, body = json::dump(req), port] { run(std::move(body), port); });
  return true;
}

void ChatSession::cancel() {
  cancel_ = true;
  const int fd = fd_.load();
  if (fd > 0) ::shutdown(fd, SHUT_RDWR);
}

void ChatSession::clear() {
  if (th_.joinable()) th_.join();
  std::lock_guard<std::mutex> g(mu_);
  if (status_ == Status::running) return;
  msgs_.clear();
  status_ = Status::idle;
  error_.clear();
  elapsed_ = 0;
  tokens_ = -1;
}

void ChatSession::run(std::string body, int port) {
  const auto t0 = Clock::now();
  auto finish = [&](Status st, std::string err) {
    const double el = std::chrono::duration<double>(Clock::now() - t0).count();
    std::lock_guard<std::mutex> g(mu_);
    elapsed_ = el;
    if (st == Status::done) {
      if (!msgs_.empty() && !msgs_.back().done) msgs_.back().done = true;
    } else {
      // A truncated answer is not part of the conversation (the user turn stays).
      while (!msgs_.empty() && msgs_.back().role == "assistant" && !msgs_.back().done) msgs_.pop_back();
    }
    status_ = st;
    error_ = std::move(err);
  };

  Fd s;
  const std::string req = "POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n"
                          "Content-Type: application/json\r\nContent-Length: " +
                          std::to_string(body.size()) + "\r\nAccept: text/event-stream\r\n\r\n" + body;
  try {
    s.fd = connect_to(port, cancel_);  // throws on cancel / timeout
    if (s.fd < 0) {
      finish(Status::error, "cannot connect to the server on 127.0.0.1:" + std::to_string(port));
      return;
    }
    fd_ = s.fd;
    for (size_t off = 0; off < req.size();) {
      if (!wait(s.fd, POLLOUT, 3000, cancel_)) throw std::runtime_error("send timed out");
      const ssize_t n = ::send(s.fd, req.data() + off, req.size() - off, MSG_NOSIGNAL);
      if (n < 0 && errno != EINTR) throw std::runtime_error("send: " + std::string(std::strerror(errno)));
      if (n > 0) off += static_cast<size_t>(n);
    }
    SseReader rd;
    char buf[1 << 16];
    bool done = false, http_bad = false;
    std::string err;
    for (;;) {
      if (cancel_.load()) {
        finish(Status::cancelled, "");
        return;
      }
      if (!wait(s.fd, POLLIN, 1000, cancel_)) continue;  // no bytes for a while: not an error
      const ssize_t n = ::recv(s.fd, buf, sizeof buf, 0);
      if (n == 0) break;  // the server closed: the body (if any) is complete
      if (n < 0) {
        if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK) continue;
        throw std::runtime_error("recv: " + std::string(std::strerror(errno)));
      }
      if (!rd.feed(std::string(buf, static_cast<size_t>(n)), err)) throw std::runtime_error(err);
      if (rd.headers_done && rd.status != 200) {
        http_bad = true;  // drain the error body (bounded), then report it
        if (rd.finished || rd.evbuf.size() > (1 << 16)) break;
        continue;
      }
      std::string payload;
      while (rd.next_event(payload)) {
        if (payload == "[DONE]") {
          done = true;
          break;
        }
        json::Value v;
        std::string perr;
        if (!json::parse(payload, v, &perr)) throw std::runtime_error("bad stream event: " + perr);
        if (const json::Value *e = v.get_object("error")) {
          const std::string *m = e->get_string("message");
          throw std::runtime_error(m && !m->empty() ? *m : "the server answered with an error");
        }
        std::lock_guard<std::mutex> g(mu_);
        if (!msgs_.empty()) {
          Msg &m = msgs_.back();
          if (const json::Value *ch = v.get_array("choices"); ch && !ch->arr.empty() && ch->arr[0].is_object()) {
            if (const json::Value *delta = ch->arr[0].get_object("delta")) {
              if (const std::string *c = delta->get_string("content")) m.text += *c;
              if (const std::string *r = delta->get_string("reasoning")) m.reasoning += *r;
            }
          }
          if (const json::Value *u = v.get_object("usage")) {
            const json::Value *ct = u->get("completion_tokens");
            if (ct && ct->is_int()) tokens_ = static_cast<long long>(ct->num);
          }
        }
      }
      if (done || rd.finished) break;
    }
    if (http_bad) {
      std::string msg = "the server refused (HTTP " + std::to_string(rd.status) + ")";
      json::Value v;
      std::string perr;
      if (json::parse(rd.evbuf, v, &perr))
        if (const json::Value *e = v.get_object("error"))
          if (const std::string *m = e->get_string("message"); m && !m->empty()) msg = *m;
      throw std::runtime_error(msg);
    }
    finish(done ? Status::done : Status::error, done ? "" : "the stream ended before the answer finished");
  } catch (const std::exception &e) {
    if (cancel_.load()) finish(Status::cancelled, "");
    else finish(Status::error, e.what());
  }
  fd_ = -1;
}

}  // namespace xenithium:launch
