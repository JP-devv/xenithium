#include "xenithium/util/http.hpp"

#include <netdb.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace xenithium::http {

namespace {

struct Fd {
  int fd = -1;
  ~Fd() {
    if (fd >= 0) ::close(fd);
  }
};

[[noreturn]] void fail(const std::string &m) { throw std::runtime_error("http: " + m); }

// Waits for the socket to be ready; throws when the deadline passes.
void wait_ready(int fd, short ev, std::chrono::steady_clock::time_point deadline) {
  for (;;) {
    const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - std::chrono::steady_clock::now()).count();
    if (left <= 0) fail("timeout");
    pollfd p{fd, ev, 0};
    const int r = ::poll(&p, 1, static_cast<int>(std::min<long long>(left, 1 << 30)));
    if (r > 0) return;
    if (r < 0 && errno != EINTR) fail(std::string("poll: ") + std::strerror(errno));
  }
}

std::string dechunk(const std::string &b) {
  std::string out;
  size_t p = 0;
  for (;;) {
    const size_t e = b.find("\r\n", p);
    if (e == std::string::npos) fail("bad chunked body");
    const size_t n = std::stoul(b.substr(p, e - p), nullptr, 16);
    p = e + 2;
    if (n == 0) return out;
    if (p + n > b.size()) fail("truncated chunk");
    out.append(b, p, n);
    p += n + 2;
  }
}

}  // namespace

Response request(const std::string &method, const std::string &url, const std::string &body, double timeout_s) {
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(static_cast<long long>(timeout_s * 1000));
  if (url.rfind("http://", 0) != 0) fail("only http:// URLs: " + url);
  const size_t hs = 7, ps = url.find('/', hs);
  const std::string hostport = url.substr(hs, ps == std::string::npos ? std::string::npos : ps - hs);
  const std::string path = ps == std::string::npos ? "/" : url.substr(ps);
  const size_t colon = hostport.rfind(':');
  const std::string host = colon == std::string::npos ? hostport : hostport.substr(0, colon);
  const std::string port = colon == std::string::npos ? "80" : hostport.substr(colon + 1);

  addrinfo hints{}, *res = nullptr;
  hints.ai_socktype = SOCK_STREAM;
  if (const int r = ::getaddrinfo(host.c_str(), port.c_str(), &hints, &res); r != 0) fail("resolve " + host + ": " + gai_strerror(r));
  Fd s;
  std::string last = "no address";
  for (addrinfo *a = res; a; a = a->ai_next) {
    s.fd = ::socket(a->ai_family, a->ai_socktype, a->ai_protocol);
    if (s.fd < 0) continue;
    if (::connect(s.fd, a->ai_addr, a->ai_addrlen) == 0) break;
    last = std::strerror(errno);
    ::close(s.fd);
    s.fd = -1;
  }
  ::freeaddrinfo(res);
  if (s.fd < 0) fail("connect " + hostport + ": " + last);

  std::string req = method + " " + path + " HTTP/1.1\r\nHost: " + hostport + "\r\nConnection: close\r\nAccept: application/json\r\n";
  if (!body.empty()) req += "Content-Type: application/json\r\nContent-Length: " + std::to_string(body.size()) + "\r\n";
  req += "\r\n" + body;
  for (size_t off = 0; off < req.size();) {
    wait_ready(s.fd, POLLOUT, deadline);
    const ssize_t n = ::send(s.fd, req.data() + off, req.size() - off, MSG_NOSIGNAL);
    if (n < 0 && errno != EINTR) fail(std::string("send: ") + std::strerror(errno));
    if (n > 0) off += static_cast<size_t>(n);
  }
  std::string raw;
  char buf[1 << 16];
  for (;;) {
    wait_ready(s.fd, POLLIN, deadline);
    const ssize_t n = ::recv(s.fd, buf, sizeof buf, 0);
    if (n == 0) break;
    if (n < 0) {
      if (errno == EINTR) continue;
      fail(std::string("recv: ") + std::strerror(errno));
    }
    raw.append(buf, static_cast<size_t>(n));
  }
  const size_t he = raw.find("\r\n\r\n");
  if (he == std::string::npos || raw.rfind("HTTP/1.", 0) != 0) fail("no HTTP response header");
  Response r;
  r.status = std::atoi(raw.c_str() + raw.find(' ') + 1);
  std::string head = raw.substr(0, he);
  for (auto &c : head) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  r.body = raw.substr(he + 4);
  if (head.find("transfer-encoding: chunked") != std::string::npos) r.body = dechunk(r.body);
  else if (const size_t cl = head.find("content-length:"); cl != std::string::npos) {
    const size_t n = std::stoul(head.substr(cl + 15));
    if (r.body.size() < n) fail("truncated body");
    r.body.resize(n);
  }
  return r;
}

}  // namespace xenithium:http
