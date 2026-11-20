// Xenithium · TLS over the system's OpenSSL loaded with dlopen. Written for this engine; only the public
// OpenSSL 3 C API is used, declared here by hand (no headers at build time).
#include "xenithium/util/tls.hpp"

#include <arpa/inet.h>
#include <dlfcn.h>
#include <poll.h>

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cstdint>
#include <cstring>
#include <mutex>

namespace xenithium::util::tls {

namespace {

// The OpenSSL 3 ABI constants used below (ssl.h / tls1.h / x509_vfy.h).
constexpr int kVerifyPeer = 1;  // SSL_VERIFY_PEER
constexpr int kFiletypePem = 1;  // SSL_FILETYPE_PEM
constexpr int kCtrlSetTlsextHostname = 55;  // SSL_CTRL_SET_TLSEXT_HOSTNAME
constexpr int kCtrlSetMinProto = 123;  // SSL_CTRL_SET_MIN_PROTO_VERSION
constexpr long kTls12 = 0x0303;  // TLS1_2_VERSION
constexpr uint64_t kOpIgnoreUnexpectedEof = 1ull << 7;  // SSL_OP_IGNORE_UNEXPECTED_EOF: a close without close_notify = EOF
constexpr int kErrWantRead = 2, kErrWantWrite = 3, kErrSyscall = 5, kErrZeroReturn = 6;

struct Api {
  bool ok = false;
  std::string why;
  const void *(*TLS_client_method)() = nullptr;
  const void *(*TLS_server_method)() = nullptr;
  void *(*SSL_CTX_new)(const void *) = nullptr;
  void (*SSL_CTX_free)(void *) = nullptr;
  int (*SSL_CTX_set_default_verify_paths)(void *) = nullptr;
  int (*SSL_CTX_load_verify_locations)(void *, const char *, const char *) = nullptr;
  void (*SSL_CTX_set_verify)(void *, int, void *) = nullptr;
  int (*SSL_CTX_use_certificate_chain_file)(void *, const char *) = nullptr;
  int (*SSL_CTX_use_PrivateKey_file)(void *, const char *, int) = nullptr;
  long (*SSL_CTX_ctrl)(void *, int, long, void *) = nullptr;
  uint64_t (*SSL_CTX_set_options)(void *, uint64_t) = nullptr;
  void *(*SSL_new)(void *) = nullptr;
  void (*SSL_free)(void *) = nullptr;
  int (*SSL_set_fd)(void *, int) = nullptr;
  long (*SSL_ctrl)(void *, int, long, void *) = nullptr;
  int (*SSL_set1_host)(void *, const char *) = nullptr;
  void *(*SSL_get0_param)(void *) = nullptr;
  int (*SSL_connect)(void *) = nullptr;
  int (*SSL_accept)(void *) = nullptr;
  int (*SSL_read)(void *, void *, int) = nullptr;
  int (*SSL_write)(void *, const void *, int) = nullptr;
  int (*SSL_get_error)(const void *, int) = nullptr;
  long (*SSL_get_verify_result)(const void *) = nullptr;
  int (*SSL_shutdown)(void *) = nullptr;
  unsigned long (*ERR_get_error)() = nullptr;
  void (*ERR_clear_error)() = nullptr;
  void (*ERR_error_string_n)(unsigned long, char *, size_t) = nullptr;
  const char *(*X509_verify_cert_error_string)(long) = nullptr;
  int (*X509_VERIFY_PARAM_set1_ip_asc)(void *, const char *) = nullptr;
};

const Api &api() {
  static Api a;
  static std::once_flag once;
  std::call_once(once, [] {
    void *crypto = dlopen("libcrypto.so.3", RTLD_NOW | RTLD_LOCAL);
    void *ssl = crypto ? dlopen("libssl.so.3", RTLD_NOW | RTLD_LOCAL) : nullptr;
    if (!crypto || !ssl) {
      const char *e = dlerror();
      a.why = std::string("https is off: cannot load the system's OpenSSL 3 (libssl.so.3 / libcrypto.so.3)") + (e ? std::string(": ") + e : "");
      return;
    }
    bool all = true;
    auto get = [&](void *lib, const char *name, auto &fn) {
      fn = reinterpret_cast<std::remove_reference_t<decltype(fn)>>(dlsym(lib, name));
      if (!fn && all) {
        all = false;
        a.why = std::string("https is off: OpenSSL has no ") + name;
      }
    };
    get(ssl, "TLS_client_method", a.TLS_client_method);
    get(ssl, "TLS_server_method", a.TLS_server_method);
    get(ssl, "SSL_CTX_new", a.SSL_CTX_new);
    get(ssl, "SSL_CTX_free", a.SSL_CTX_free);
    get(ssl, "SSL_CTX_set_default_verify_paths", a.SSL_CTX_set_default_verify_paths);
    get(ssl, "SSL_CTX_load_verify_locations", a.SSL_CTX_load_verify_locations);
    get(ssl, "SSL_CTX_set_verify", a.SSL_CTX_set_verify);
    get(ssl, "SSL_CTX_use_certificate_chain_file", a.SSL_CTX_use_certificate_chain_file);
    get(ssl, "SSL_CTX_use_PrivateKey_file", a.SSL_CTX_use_PrivateKey_file);
    get(ssl, "SSL_CTX_ctrl", a.SSL_CTX_ctrl);
    get(ssl, "SSL_CTX_set_options", a.SSL_CTX_set_options);
    get(ssl, "SSL_new", a.SSL_new);
    get(ssl, "SSL_free", a.SSL_free);
    get(ssl, "SSL_set_fd", a.SSL_set_fd);
    get(ssl, "SSL_ctrl", a.SSL_ctrl);
    get(ssl, "SSL_set1_host", a.SSL_set1_host);
    get(ssl, "SSL_get0_param", a.SSL_get0_param);
    get(ssl, "SSL_connect", a.SSL_connect);
    get(ssl, "SSL_accept", a.SSL_accept);
    get(ssl, "SSL_read", a.SSL_read);
    get(ssl, "SSL_write", a.SSL_write);
    get(ssl, "SSL_get_error", a.SSL_get_error);
    get(ssl, "SSL_get_verify_result", a.SSL_get_verify_result);
    get(ssl, "SSL_shutdown", a.SSL_shutdown);
    get(crypto, "ERR_get_error", a.ERR_get_error);
    get(crypto, "ERR_clear_error", a.ERR_clear_error);
    get(crypto, "ERR_error_string_n", a.ERR_error_string_n);
    get(crypto, "X509_verify_cert_error_string", a.X509_verify_cert_error_string);
    get(crypto, "X509_VERIFY_PARAM_set1_ip_asc", a.X509_VERIFY_PARAM_set1_ip_asc);
    a.ok = all;
  });
  return a;
}

std::string ssl_error(const Api &a, const char *what) {
  std::string s = what;
  if (const unsigned long e = a.ERR_get_error(); e != 0) {
    char buf[256];
    a.ERR_error_string_n(e, buf, sizeof buf);
    s += std::string(": ") + buf;
  }
  return s;
}

// Waits for the socket; false = the deadline passed.
bool wait_fd(int fd, short ev, Deadline deadline) {
  for (;;) {
    const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - std::chrono::steady_clock::now()).count();
    if (left <= 0) return false;
    pollfd p{fd, ev, 0};
    const int r = ::poll(&p, 1, static_cast<int>(std::min<long long>(left, INT_MAX)));
    if (r > 0) return true;
    if (r < 0 && errno != EINTR) return false;
  }
}

}  // namespace

bool available(std::string *why) {
  const Api &a = api();
  if (!a.ok && why) *why = a.why;
  return a.ok;
}

Conn::~Conn() {
  const Api &a = api();
  if (!a.ok) return;
  if (ssl_) a.SSL_free(ssl_);
  if (ctx_) a.SSL_CTX_free(ctx_);
}

bool Conn::handshake(bool client, Deadline deadline, std::string &err) {
  const Api &a = api();
  for (;;) {
    a.ERR_clear_error();
    const int r = client ? a.SSL_connect(ssl_) : a.SSL_accept(ssl_);
    if (r == 1) return true;
    const int e = a.SSL_get_error(ssl_, r);
    if (e == kErrWantRead || e == kErrWantWrite) {
      if (!wait_fd(fd_, e == kErrWantRead ? POLLIN : POLLOUT, deadline)) return err = "TLS handshake timed out", false;
      continue;
    }
    if (client) {
      const long v = a.SSL_get_verify_result(ssl_);
      if (v != 0) return err = std::string("TLS certificate verification failed: ") + a.X509_verify_cert_error_string(v), false;
    }
    return err = ssl_error(a, "TLS handshake failed"), false;
  }
}

bool Conn::connect(int fd, const std::string &host, const std::string &ca_file, Deadline deadline, std::string &err) {
  if (!available(&err)) return false;
  const Api &a = api();
  fd_ = fd;
  ctx_ = a.SSL_CTX_new(a.TLS_client_method());
  if (!ctx_) return err = ssl_error(a, "TLS context"), false;
  a.SSL_CTX_ctrl(ctx_, kCtrlSetMinProto, kTls12, nullptr);
  a.SSL_CTX_set_options(ctx_, kOpIgnoreUnexpectedEof);
  const bool ca = ca_file.empty() ? a.SSL_CTX_set_default_verify_paths(ctx_) == 1
                                  : a.SSL_CTX_load_verify_locations(ctx_, ca_file.c_str(), nullptr) == 1;
  if (!ca) return err = ssl_error(a, "TLS trust store"), false;
  a.SSL_CTX_set_verify(ctx_, kVerifyPeer, nullptr);
  ssl_ = a.SSL_new(ctx_);
  if (!ssl_ || a.SSL_set_fd(ssl_, fd) != 1) return err = ssl_error(a, "TLS session"), false;
  unsigned char ip[16];
  const bool literal = inet_pton(AF_INET, host.c_str(), ip) == 1 || inet_pton(AF_INET6, host.c_str(), ip) == 1;
  if (literal) {  // an address: no SNI, the certificate must name the IP
    if (a.X509_VERIFY_PARAM_set1_ip_asc(a.SSL_get0_param(ssl_), host.c_str()) != 1) return err = "TLS: bad IP " + host, false;
  } else {
    a.SSL_ctrl(ssl_, kCtrlSetTlsextHostname, 0, const_cast<char *>(host.c_str()));
    if (a.SSL_set1_host(ssl_, host.c_str()) != 1) return err = "TLS: bad host name " + host, false;
  }
  return handshake(true, deadline, err);
}

bool Conn::accept(int fd, const std::string &cert_file, const std::string &key_file, Deadline deadline, std::string &err) {
  if (!available(&err)) return false;
  const Api &a = api();
  fd_ = fd;
  ctx_ = a.SSL_CTX_new(a.TLS_server_method());
  if (!ctx_) return err = ssl_error(a, "TLS context"), false;
  if (a.SSL_CTX_use_certificate_chain_file(ctx_, cert_file.c_str()) != 1) return err = ssl_error(a, "TLS certificate"), false;
  if (a.SSL_CTX_use_PrivateKey_file(ctx_, key_file.c_str(), kFiletypePem) != 1) return err = ssl_error(a, "TLS key"), false;
  ssl_ = a.SSL_new(ctx_);
  if (!ssl_ || a.SSL_set_fd(ssl_, fd) != 1) return err = ssl_error(a, "TLS session"), false;
  return handshake(false, deadline, err);
}

bool Conn::write_all(const void *data, size_t n, Deadline deadline, std::string &err) {
  const Api &a = api();
  const char *p = static_cast<const char *>(data);
  while (n > 0) {
    a.ERR_clear_error();
    const int r = a.SSL_write(ssl_, p, static_cast<int>(std::min<size_t>(n, 1 << 20)));
    if (r > 0) {
      p += r;
      n -= static_cast<size_t>(r);
      continue;
    }
    const int e = a.SSL_get_error(ssl_, r);
    if (e == kErrWantRead || e == kErrWantWrite) {
      if (!wait_fd(fd_, e == kErrWantRead ? POLLIN : POLLOUT, deadline)) return err = "timed out", false;
      continue;
    }
    return err = ssl_error(a, "TLS write"), false;
  }
  return true;
}

long Conn::read(void *data, size_t n, Deadline deadline, std::string &err) {
  const Api &a = api();
  for (;;) {
    a.ERR_clear_error();
    const int r = a.SSL_read(ssl_, data, static_cast<int>(std::min<size_t>(n, INT_MAX)));
    if (r > 0) return r;
    const int e = a.SSL_get_error(ssl_, r);
    if (e == kErrZeroReturn) return 0;
    if (e == kErrWantRead || e == kErrWantWrite) {
      if (!wait_fd(fd_, e == kErrWantRead ? POLLIN : POLLOUT, deadline)) return err = "timed out", -1;
      continue;
    }
    if (e == kErrSyscall && a.ERR_get_error() == 0) return 0;  // EOF without close_notify: the body's length decides
    err = ssl_error(a, "TLS read");
    return -1;
  }
}

void Conn::shutdown() {
  const Api &a = api();
  if (a.ok && ssl_) a.SSL_shutdown(ssl_);
}

}  // namespace xenithium:util:tls
