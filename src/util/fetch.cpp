// Xenithium · media URLs: data: in place, http(s) GET with redirects, size cap and timeout. Written for
// this engine (POSIX sockets; TLS by util/tls).
#include "xenithium/util/fetch.hpp"

#include <fcntl.h>
#include <netdb.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <climits>
#include <cstring>
#include <memory>

#include "xenithium/util/image.hpp"
#include "xenithium/util/tls.hpp"

namespace xenithium::util {

namespace {

using Clock = std::chrono::steady_clock;

std::string lower(std::string s) {
  for (char &c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return s;
}

struct Fd {
  int fd = -1;
  ~Fd() {
    if (fd >= 0) ::close(fd);
  }
};

bool wait_fd(int fd, short ev, Clock::time_point deadline) {
  for (;;) {
    const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - Clock::now()).count();
    if (left <= 0) return false;
    pollfd p{fd, ev, 0};
    const int r = ::poll(&p, 1, static_cast<int>(std::min<long long>(left, INT_MAX)));
    if (r > 0) return true;
    if (r < 0 && errno != EINTR) return false;
  }
}

// A non-blocking connected socket to host:port within the deadline.
bool dial(const std::string &host, const std::string &port, Clock::time_point deadline, Fd &s, std::string &err) {
  addrinfo hints{}, *res = nullptr;
  hints.ai_socktype = SOCK_STREAM;
  if (const int r = ::getaddrinfo(host.c_str(), port.c_str(), &hints, &res); r != 0)
    return err = "cannot resolve " + host + ": " + gai_strerror(r), false;
  std::string last = "no address";
  for (addrinfo *a = res; a; a = a->ai_next) {
    s.fd = ::socket(a->ai_family, a->ai_socktype | SOCK_NONBLOCK | SOCK_CLOEXEC, a->ai_protocol);
    if (s.fd < 0) continue;
    int rc = ::connect(s.fd, a->ai_addr, a->ai_addrlen);
    if (rc != 0 && errno == EINPROGRESS) {
      if (wait_fd(s.fd, POLLOUT, deadline)) {
        int soerr = 0;
        socklen_t len = sizeof soerr;
        ::getsockopt(s.fd, SOL_SOCKET, SO_ERROR, &soerr, &len);
        rc = soerr == 0 ? 0 : -1;
        errno = soerr;
      } else {
        errno = ETIMEDOUT;
      }
    }
    if (rc == 0) break;
    last = std::strerror(errno);
    ::close(s.fd);
    s.fd = -1;
  }
  ::freeaddrinfo(res);
  if (s.fd < 0) return err = "cannot connect to " + host + ":" + port + ": " + last, false;
  return true;
}

// Plain or TLS reads / writes over one connection.
struct Stream {
  int fd = -1;
  std::unique_ptr<tls::Conn> t;
  bool write_all(const std::string &s, Clock::time_point deadline, std::string &err) {
    if (t) return t->write_all(s.data(), s.size(), deadline, err);
    for (size_t off = 0; off < s.size();) {
      const ssize_t n = ::send(fd, s.data() + off, s.size() - off, MSG_NOSIGNAL);
      if (n > 0) {
        off += static_cast<size_t>(n);
        continue;
      }
      if (n < 0 && errno == EINTR) continue;
      if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
        if (!wait_fd(fd, POLLOUT, deadline)) return err = "timed out", false;
        continue;
      }
      return err = std::string("send: ") + std::strerror(errno), false;
    }
    return true;
  }
  long read(char *buf, size_t n, Clock::time_point deadline, std::string &err) {
    if (t) return t->read(buf, n, deadline, err);
    for (;;) {
      const ssize_t r = ::recv(fd, buf, n, 0);
      if (r >= 0) return r;
      if (errno == EINTR) continue;
      if (errno == EAGAIN || errno == EWOULDBLOCK) {
        if (!wait_fd(fd, POLLIN, deadline)) return err = "timed out", -1;
        continue;
      }
      err = std::string("recv: ") + std::strerror(errno);
      return -1;
    }
  }
};

// Incremental chunked-body decoder: feed raw bytes, done once the last chunk and its trailers passed.
struct Dechunk {
  enum St { size, size_ext, size_lf, data, data_cr, data_lf, trailer, trailer_lf, end } st = size;
  size_t left = 0;
  bool got_digit = false, line_empty = true;
  bool bad = false;
  bool feed(const char *p, size_t n, std::string &out, size_t cap) {
    for (size_t i = 0; i < n && st != end && !bad; ++i) {
      const char c = p[i];
      switch (st) {
        case size:
          if (std::isxdigit(static_cast<unsigned char>(c))) {
            if (left > (SIZE_MAX >> 4)) return bad = true, false;
            left = left * 16 + static_cast<size_t>(std::isdigit(static_cast<unsigned char>(c)) ? c - '0' : (std::tolower(c) - 'a' + 10));
            got_digit = true;
          } else if (c == ';' || c == ' ' || c == '\t') {
            st = size_ext;
          } else if (c == '\r') {
            st = size_lf;
          } else {
            bad = true;
          }
          break;
        case size_ext:
          if (c == '\r') st = size_lf;
          break;
        case size_lf:
          if (c != '\n' || !got_digit) return bad = true, false;
          got_digit = false;
          if (left == 0) {
            st = trailer;
            line_empty = true;
          } else {
            st = data;
          }
          break;
        case data: {
          const size_t take = std::min(left, n - i);
          if (out.size() + take > cap) return false;
          out.append(p + i, take);
          left -= take;
          i += take - 1;
          if (left == 0) st = data_cr;
          break;
        }
        case data_cr:
          if (c != '\r') return bad = true, false;
          st = data_lf;
          break;
        case data_lf:
          if (c != '\n') return bad = true, false;
          st = size;
          break;
        case trailer:
          if (c == '\r') st = trailer_lf;
          else line_empty = false;
          break;
        case trailer_lf:
          if (c != '\n') return bad = true, false;
          if (line_empty) st = end;
          else {
            st = trailer;
            line_empty = true;
          }
          break;
        case end: break;
      }
    }
    return !bad;
  }
};

std::string resolve_location(const UrlParts &base, const std::string &loc) {
  if (loc.find("://") != std::string::npos) return loc;
  const bool v6 = base.host.find(':') != std::string::npos;
  const std::string host = v6 ? "[" + base.host + "]" : base.host;
  const bool def = (base.scheme == "http" && base.port == "80") || (base.scheme == "https" && base.port == "443");
  const std::string origin = base.scheme + "://" + host + (def ? "" : ":" + base.port);
  if (loc.rfind("//", 0) == 0) return base.scheme + ":" + loc;
  if (!loc.empty() && loc[0] == '/') return origin + loc;
  std::string dir = base.path.substr(0, base.path.find('?'));
  dir = dir.substr(0, dir.rfind('/') + 1);
  if (dir.empty()) dir = "/";
  return origin + dir + loc;
}

}  // namespace

bool split_url(const std::string &url, UrlParts &out) {
  const size_t se = url.find("://");
  if (se == std::string::npos) return false;
  out.scheme = lower(url.substr(0, se));
  if (out.scheme != "http" && out.scheme != "https") return false;
  const size_t hs = se + 3;
  size_t he = url.find_first_of("/?#", hs);
  if (he == std::string::npos) he = url.size();
  std::string auth = url.substr(hs, he - hs);
  if (const size_t at = auth.rfind('@'); at != std::string::npos) auth = auth.substr(at + 1);
  out.port = out.scheme == "https" ? "443" : "80";
  if (!auth.empty() && auth[0] == '[') {
    const size_t rb = auth.find(']');
    if (rb == std::string::npos) return false;
    out.host = auth.substr(1, rb - 1);
    if (rb + 1 < auth.size()) {
      if (auth[rb + 1] != ':') return false;
      out.port = auth.substr(rb + 2);
    }
  } else {
    const size_t colon = auth.rfind(':');
    out.host = auth.substr(0, colon);
    if (colon != std::string::npos) out.port = auth.substr(colon + 1);
  }
  out.host = lower(out.host);
  if (out.host.empty() || out.port.empty() || out.port.size() > 5 ||
      !std::all_of(out.port.begin(), out.port.end(), [](char c) { return c >= '0' && c <= '9'; }))
    return false;
  std::string rest = url.substr(he);
  if (const size_t h = rest.find('#'); h != std::string::npos) rest.resize(h);
  if (rest.empty() || rest[0] != '/') rest = "/" + rest;
  for (char c : rest)
    if (static_cast<unsigned char>(c) <= 0x20 || c == 0x7f) return false;  // a request line cannot carry it
  out.path = rest;
  return true;
}

std::vector<std::string> split_domains(const std::string &list) {
  std::vector<std::string> out;
  std::string cur;
  auto push = [&] {
    size_t a = 0, b = cur.size();
    while (a < b && std::isspace(static_cast<unsigned char>(cur[a]))) ++a;
    while (b > a && std::isspace(static_cast<unsigned char>(cur[b - 1]))) --b;
    if (b > a) out.push_back(lower(cur.substr(a, b - a)));
    cur.clear();
  };
  for (char c : list) {
    if (c == ',' || c == ' ') push();
    else cur += c;
  }
  push();
  return out;
}

bool fetch_url(const std::string &url0, const FetchOptions &opt, std::string &body, std::string &err) {
  if (!opt.enabled) return err = "http(s) URLs are off on this server (XEN_MEDIA_URLS=0): send the media as a base64 data: URL", false;
  const auto deadline = Clock::now() + std::chrono::milliseconds(static_cast<long long>(opt.timeout_s * 1000));
  std::string url = url0;
  for (int hop = 0;; ++hop) {
    UrlParts u;
    if (!split_url(url, u)) return err = "unsupported URL (http://, https:// or data: only): " + url.substr(0, 200), false;
    if (!opt.allowed_domains.empty() &&
        std::find(opt.allowed_domains.begin(), opt.allowed_domains.end(), u.host) == opt.allowed_domains.end())
      return err = "the URL must be from one of the allowed domains (XEN_MEDIA_DOMAINS); its host is " + u.host, false;
    const bool https = u.scheme == "https";
    std::string why;
    if (https && !tls::available(&why)) return err = why, false;
    Fd s;
    if (!dial(u.host, u.port, deadline, s, err)) return false;
    Stream st;
    st.fd = s.fd;
    if (https) {
      st.t = std::make_unique<tls::Conn>();
      std::string terr;
      if (!st.t->connect(s.fd, u.host, opt.ca_file, deadline, terr)) return err = u.host + ": " + terr, false;
    }
    const bool v6 = u.host.find(':') != std::string::npos;
    const bool def = (!https && u.port == "80") || (https && u.port == "443");
    std::string req = "GET " + u.path + " HTTP/1.1\r\nHost: " + (v6 ? "[" + u.host + "]" : u.host) + (def ? "" : ":" + u.port) +
                      "\r\nUser-Agent: xenithium\r\nAccept: */*\r\nAccept-Encoding: identity\r\nConnection: close\r\n\r\n";
    std::string werr;
    if (!st.write_all(req, deadline, werr)) return err = "fetching " + u.host + ": " + werr, false;

    // the head
    std::string raw;
    char buf[1 << 16];
    size_t he = std::string::npos;
    while ((he = raw.find("\r\n\r\n")) == std::string::npos) {
      if (raw.size() > (64u << 10)) return err = "fetching " + u.host + ": response header too large", false;
      std::string rerr;
      const long n = st.read(buf, sizeof buf, deadline, rerr);
      if (n < 0) return err = "fetching " + u.host + ": " + rerr, false;
      if (n == 0) return err = "fetching " + u.host + ": connection closed before the response header", false;
      raw.append(buf, static_cast<size_t>(n));
    }
    if (raw.rfind("HTTP/1.", 0) != 0) return err = "fetching " + u.host + ": not an HTTP response", false;
    const int status = std::atoi(raw.c_str() + raw.find(' ') + 1);
    const std::string head = lower(raw.substr(0, he));
    std::string rest = raw.substr(he + 4);
    auto header = [&](const char *name) -> std::string {
      const std::string key = std::string("\r\n") + name + ":";
      const size_t p = head.find(key);
      if (p == std::string::npos) return {};
      size_t a = p + key.size(), b = head.find("\r\n", a);
      if (b == std::string::npos) b = head.size();
      while (a < b && (head[a] == ' ' || head[a] == '\t')) ++a;
      while (b > a && (head[b - 1] == ' ' || head[b - 1] == '\t')) --b;
      return raw.substr(a, b - a);  // original case (Location)
    };
    if (status == 301 || status == 302 || status == 303 || status == 307 || status == 308) {
      const std::string loc = header("location");
      if (loc.empty()) return err = "fetching " + u.host + ": HTTP " + std::to_string(status) + " without a Location", false;
      if (hop >= opt.max_redirects) return err = "fetching " + url0.substr(0, 200) + ": too many redirects", false;
      url = resolve_location(u, loc);
      continue;
    }
    if (status < 200 || status >= 300)
      return err = "fetching " + url.substr(0, 200) + ": HTTP " + std::to_string(status), false;

    // the body
    body.clear();
    const std::string te = lower(header("transfer-encoding")), cl = header("content-length");
    const std::string too_big = "the media at " + url.substr(0, 200) + " is larger than " + std::to_string(opt.max_bytes >> 20) + " MiB";
    if (te.find("chunked") != std::string::npos) {
      Dechunk dc;
      if (!dc.feed(rest.data(), rest.size(), body, opt.max_bytes)) return err = dc.bad ? "fetching " + u.host + ": bad chunked body" : too_big, false;
      while (dc.st != Dechunk::end) {
        std::string rerr;
        const long n = st.read(buf, sizeof buf, deadline, rerr);
        if (n < 0) return err = "fetching " + u.host + ": " + rerr, false;
        if (n == 0) return err = "fetching " + u.host + ": truncated chunked body", false;
        if (!dc.feed(buf, static_cast<size_t>(n), body, opt.max_bytes)) return err = dc.bad ? "fetching " + u.host + ": bad chunked body" : too_big, false;
      }
    } else if (!cl.empty()) {
      char *endp = nullptr;
      const unsigned long long want = std::strtoull(cl.c_str(), &endp, 10);
      if (endp == cl.c_str()) return err = "fetching " + u.host + ": bad Content-Length", false;
      if (want > opt.max_bytes) return err = too_big, false;
      body = std::move(rest);
      while (body.size() < want) {
        std::string rerr;
        const long n = st.read(buf, sizeof buf, deadline, rerr);
        if (n < 0) return err = "fetching " + u.host + ": " + rerr, false;
        if (n == 0) return err = "fetching " + u.host + ": truncated body", false;
        body.append(buf, static_cast<size_t>(n));
      }
      body.resize(want);
    } else {  // to the end of the connection
      body = std::move(rest);
      for (;;) {
        if (body.size() > opt.max_bytes) return err = too_big, false;
        std::string rerr;
        const long n = st.read(buf, sizeof buf, deadline, rerr);
        if (n < 0) return err = "fetching " + u.host + ": " + rerr, false;
        if (n == 0) break;
        body.append(buf, static_cast<size_t>(n));
      }
    }
    if (st.t) st.t->shutdown();
    return true;
  }
}

bool load_media_url(const std::string &url, const FetchOptions &opt, std::string &bytes, std::string &err) {
  if (url.rfind("data:", 0) == 0) return parse_data_url(url, bytes, err);
  const std::string scheme = lower(url.substr(0, url.find(':')));
  if (scheme == "http" || scheme == "https") return fetch_url(url, opt, bytes, err);
  if (scheme == "file") return err = "file: URLs are not accepted: send the media as a data: or http(s) URL", false;
  return err = "unsupported media URL (data:, http:// or https://): " + url.substr(0, 64), false;
}

}  // namespace xenithium:util
