// Gate for media URLs (xenithium/util/fetch.hpp): URL splitting, a local http and https server (the test's
// certificate in tests/data/tls/, self-signed for localhost / 127.0.0.1) serving Content-Length, chunked and to-EOF
// bodies, redirects (absolute, relative, a loop), errors, the size cap, the timeout and the domain allow-list.
#include <arpa/inet.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <thread>

#include "xenithium/util/fetch.hpp"
#include "xenithium/util/tls.hpp"

namespace {
namespace u = xenithium::util;

int g_checks = 0, g_failures = 0;
#define CHECK(cond, ...)                                \
  do {                                                  \
    ++g_checks;                                         \
    if (!(cond)) {                                      \
      ++g_failures;                                     \
      std::printf("[FAIL] %s:%d ", __FILE__, __LINE__); \
      std::printf(__VA_ARGS__);                         \
      std::printf("\n");                                \
    }                                                   \
  } while (0)

std::string payload(size_t n, unsigned seed) {
  std::string s(n, '\0');
  unsigned x = seed;
  for (auto &c : s) {
    x = x * 1103515245u + 12345u;
    c = static_cast<char>(x >> 16);
  }
  return s;
}

// A one-thread local server (plain or TLS) answering by path.
class TestServer {
 public:
  TestServer(bool tls, std::string cert, std::string key) : tls_(tls), cert_(std::move(cert)), key_(std::move(key)) {
    fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
    const int one = 1;
    setsockopt(fd_, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    ::bind(fd_, reinterpret_cast<sockaddr *>(&a), sizeof a);
    ::listen(fd_, 16);
    socklen_t len = sizeof a;
    getsockname(fd_, reinterpret_cast<sockaddr *>(&a), &len);
    port_ = ntohs(a.sin_port);
    th_ = std::thread([this] { loop(); });
  }
  ~TestServer() {
    stop_ = true;
    th_.join();
    ::close(fd_);
  }
  int port() const { return port_; }
  const std::string big = payload(300000, 7);

 private:
  bool tls_;
  std::string cert_, key_;
  int fd_ = -1, port_ = 0;
  std::atomic<bool> stop_{false};
  std::thread th_;

  void loop() {
    while (!stop_) {
      pollfd p{fd_, POLLIN, 0};
      if (::poll(&p, 1, 50) <= 0) continue;
      const int c = ::accept4(fd_, nullptr, nullptr, SOCK_NONBLOCK);
      if (c < 0) continue;
      serve(c);
      ::close(c);
    }
  }
  void serve(int c) {
    const auto dl = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    std::unique_ptr<u::tls::Conn> t;
    std::string err;
    if (tls_) {
      t = std::make_unique<u::tls::Conn>();
      if (!t->accept(c, cert_, key_, dl, err)) return;  // a client that refused our certificate
    }
    auto rd = [&](char *b, size_t n) -> long {
      if (t) return t->read(b, n, dl, err);
      for (;;) {
        const ssize_t r = ::recv(c, b, n, 0);
        if (r >= 0) return r;
        if (errno != EAGAIN) return -1;
        pollfd p{c, POLLIN, 0};
        if (::poll(&p, 1, 2000) <= 0) return -1;
      }
    };
    auto wr = [&](const std::string &s) {
      if (t) return void(t->write_all(s.data(), s.size(), dl, err));
      size_t off = 0;
      while (off < s.size()) {
        const ssize_t n = ::send(c, s.data() + off, s.size() - off, MSG_NOSIGNAL);
        if (n > 0) off += static_cast<size_t>(n);
        else if (n < 0 && errno == EAGAIN) {
          pollfd p{c, POLLOUT, 0};
          ::poll(&p, 1, 2000);
        } else return;
      }
    };
    std::string req;
    char buf[4096];
    while (req.find("\r\n\r\n") == std::string::npos) {
      const long n = rd(buf, sizeof buf);
      if (n <= 0) return;
      req.append(buf, static_cast<size_t>(n));
    }
    const size_t sp = req.find(' ');
    const std::string path = req.substr(sp + 1, req.find(' ', sp + 1) - sp - 1);
    const std::string me = std::string(tls_ ? "https" : "http") + "://127.0.0.1:" + std::to_string(port_);
    if (path == "/img") {
      wr("HTTP/1.1 200 OK\r\nContent-Type: image/png\r\nContent-Length: " + std::to_string(big.size()) + "\r\n\r\n" + big);
    } else if (path == "/chunked") {
      std::string s = "HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n";
      size_t off = 0, k = 1;
      while (off < big.size()) {
        const size_t n = std::min(big.size() - off, k * 997);
        char h[32];
        std::snprintf(h, sizeof h, "%zx%s\r\n", n, k % 3 == 0 ? ";ext=1" : "");
        s += h + big.substr(off, n) + "\r\n";
        off += n;
        ++k;
      }
      s += "0\r\nX-Trailer: 1\r\n\r\n";
      wr(s);
    } else if (path == "/eof") {
      wr("HTTP/1.1 200 OK\r\n\r\n" + big.substr(0, 1000));
      if (t) t->shutdown();
    } else if (path == "/r1") {
      wr("HTTP/1.1 302 Found\r\nLocation: /img\r\nContent-Length: 0\r\n\r\n");
    } else if (path == "/r2") {
      wr("HTTP/1.1 301 Moved\r\nLocation: " + me + "/chunked\r\nContent-Length: 0\r\n\r\n");
    } else if (path == "/rel/a") {
      wr("HTTP/1.1 307 Temporary\r\nlocation: b?x=1\r\nContent-Length: 0\r\n\r\n");
    } else if (path == "/rel/b?x=1") {
      wr("HTTP/1.1 200 OK\r\nContent-Length: 9\r\n\r\nrel-b x=1");
    } else if (path == "/loop") {
      wr("HTTP/1.1 302 Found\r\nLocation: /loop\r\nContent-Length: 0\r\n\r\n");
    } else if (path == "/big") {
      wr("HTTP/1.1 200 OK\r\nContent-Length: 10000000\r\n\r\n" + big);
    } else if (path == "/bigeof") {
      wr("HTTP/1.1 200 OK\r\n\r\n" + big);
    } else if (path == "/slow") {
      std::this_thread::sleep_for(std::chrono::milliseconds(800));
    } else {
      wr("HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n");
    }
  }
};

void split_cases() {
  u::UrlParts p;
  CHECK(u::split_url("http://Example.COM/a/b.png?x=1#frag", p) && p.scheme == "http" && p.host == "example.com" && p.port == "80" &&
            p.path == "/a/b.png?x=1",
        "split basic");
  CHECK(u::split_url("HTTPS://user:pw@host:8443", p) && p.scheme == "https" && p.host == "host" && p.port == "8443" && p.path == "/",
        "split userinfo / port / empty path");
  CHECK(u::split_url("http://[::1]:8080/x", p) && p.host == "::1" && p.port == "8080" && p.path == "/x", "split IPv6");
  CHECK(u::split_url("http://h?q=1", p) && p.path == "/?q=1", "split query without path");
  CHECK(!u::split_url("ftp://h/x", p), "ftp refused");
  CHECK(!u::split_url("http://h:99x/", p), "bad port refused");
  CHECK(!u::split_url("http:///x", p), "no host refused");
  CHECK(!u::split_url("http://h/a b", p), "space in path refused");
  const auto d = u::split_domains(" A.com, b.org  c.net,");
  CHECK(d.size() == 3 && d[0] == "a.com" && d[1] == "b.org" && d[2] == "c.net", "split_domains");
}

void server_cases(bool tls, const std::string &dir) {
  TestServer srv(tls, dir + "/cert.pem", dir + "/key.pem");
  const char *tag = tls ? "https" : "http";
  const std::string base = std::string(tag) + "://127.0.0.1:" + std::to_string(srv.port());
  const std::string named = std::string(tag) + "://localhost:" + std::to_string(srv.port());
  u::FetchOptions o;
  o.timeout_s = 5;
  o.ca_file = dir + "/cert.pem";
  std::string body, err;
  CHECK(u::fetch_url(base + "/img", o, body, err) && body == srv.big, "%s content-length: %s", tag, err.c_str());
  CHECK(u::fetch_url(named + "/img", o, body, err) && body == srv.big, "%s by host name: %s", tag, err.c_str());
  CHECK(u::fetch_url(base + "/chunked", o, body, err) && body == srv.big, "%s chunked: %s", tag, err.c_str());
  CHECK(u::fetch_url(base + "/eof", o, body, err) && body == srv.big.substr(0, 1000), "%s to EOF: %s", tag, err.c_str());
  CHECK(u::fetch_url(base + "/r1", o, body, err) && body == srv.big, "%s 302 relative: %s", tag, err.c_str());
  CHECK(u::fetch_url(base + "/r2", o, body, err) && body == srv.big, "%s 301 absolute: %s", tag, err.c_str());
  CHECK(u::fetch_url(base + "/rel/a#f", o, body, err) && body == "rel-b x=1", "%s 307 relative path: %s", tag, err.c_str());
  CHECK(!u::fetch_url(base + "/loop", o, body, err) && err.find("too many redirects") != std::string::npos, "%s loop: %s", tag, err.c_str());
  CHECK(!u::fetch_url(base + "/nope", o, body, err) && err.find("HTTP 404") != std::string::npos, "%s 404: %s", tag, err.c_str());
  u::FetchOptions small = o;
  small.max_bytes = 100000;
  CHECK(!u::fetch_url(base + "/big", small, body, err) && err.find("larger than") != std::string::npos, "%s cap by length: %s", tag, err.c_str());
  CHECK(!u::fetch_url(base + "/bigeof", small, body, err) && err.find("larger than") != std::string::npos, "%s cap to EOF: %s", tag, err.c_str());
  CHECK(!u::fetch_url(base + "/chunked", small, body, err) && err.find("larger than") != std::string::npos, "%s cap chunked: %s", tag, err.c_str());
  u::FetchOptions quick = o;
  quick.timeout_s = 0.3;
  const auto t0 = std::chrono::steady_clock::now();
  CHECK(!u::fetch_url(base + "/slow", quick, body, err) && err.find("timed out") != std::string::npos, "%s timeout: %s", tag, err.c_str());
  const double took = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  CHECK(took < 0.7, "%s timeout honoured (%.2f s)", tag, took);
  std::this_thread::sleep_for(std::chrono::milliseconds(600));  // let the server finish /slow
  u::FetchOptions only = o;
  only.allowed_domains = {"example.com"};
  CHECK(!u::fetch_url(base + "/img", only, body, err) && err.find("allowed domains") != std::string::npos, "%s allow-list refuses", tag);
  only.allowed_domains = {"localhost"};
  CHECK(u::fetch_url(named + "/img", only, body, err) && body == srv.big, "%s allow-list admits: %s", tag, err.c_str());
  CHECK(!u::fetch_url(named + "/r2", only, body, err) && err.find("allowed domains") != std::string::npos,
        "%s allow-list checked on the redirect hop", tag);
  if (tls) {
    u::FetchOptions sys = o;
    sys.ca_file.clear();  // the system store does not trust the test's certificate
    CHECK(!u::fetch_url(base + "/img", sys, body, err) && err.find("verif") != std::string::npos, "https untrusted refused: %s", err.c_str());
  }
  CHECK(u::load_media_url(base + "/img", o, body, err) && body == srv.big, "%s load_media_url", tag);
}

}  // namespace

int main() {
  const std::string dir = std::string(XENITHIUM_SOURCE_DIR) + "/tests/data/tls";
  split_cases();
  std::string body, err;
  u::FetchOptions o;
  CHECK(u::load_media_url("data:image/png;base64,aGk=", o, body, err) && body == "hi", "data url");
  CHECK(!u::load_media_url("file:///etc/passwd", o, body, err) && err.find("file:") != std::string::npos, "file refused");
  CHECK(!u::load_media_url("ftp://h/x.png", o, body, err), "ftp refused");
  o.enabled = false;
  CHECK(!u::load_media_url("http://127.0.0.1:1/x.png", o, body, err) && err.find("XEN_MEDIA_URLS") != std::string::npos, "off");
  CHECK(u::load_media_url("data:image/png;base64,aGk=", o, body, err), "data url with fetching off");
  server_cases(false, dir);
  std::string why;
  if (u::tls::available(&why)) server_cases(true, dir);
  else std::printf("[SKIP] https: %s\n", why.c_str());
  std::printf("%d checks, %d failures\n", g_checks, g_failures);
  return g_failures == 0 ? 0 : 1;
}
