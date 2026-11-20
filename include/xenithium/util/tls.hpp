// Xenithium · TLS over the system's OpenSSL, loaded at run time (for https media URLs). The engine has no build
// dependency on OpenSSL: libssl.so.3 / libcrypto.so.3 are opened with dlopen on first use and the few functions a
// client (and the test's server) needs are looked up by name. Without the libraries https is off (available false
// with the reason) and everything else works. Sockets are non-blocking; every call waits with poll up to a deadline.
#pragma once

#include <chrono>
#include <cstddef>
#include <string>

namespace xenithium::util::tls {

using Deadline = std::chrono::steady_clock::time_point;

// true once libssl / libcrypto are loaded and every needed function was found; else false + why.
bool available(std::string *why = nullptr);

// One TLS session on a connected, non-blocking socket (the socket stays owned by the caller).
class Conn {
 public:
  Conn() = default;
  ~Conn();
  Conn(const Conn &) = delete;
  Conn &operator=(const Conn &) = delete;

  // Client handshake: SNI = host, the peer's certificate verified against ca_file (PEM; empty = the system's default
  // store, which also honours SSL_CERT_FILE / SSL_CERT_DIR) and its name against host. false + err on failure.
  bool connect(int fd, const std::string &host, const std::string &ca_file, Deadline deadline, std::string &err);
  // Server handshake with a PEM certificate chain and key (the tests' local https server).
  bool accept(int fd, const std::string &cert_file, const std::string &key_file, Deadline deadline, std::string &err);

  // All of n bytes; false + err on error / timeout.
  bool write_all(const void *data, size_t n, Deadline deadline, std::string &err);
  // Up to n bytes: > 0 read, 0 = the peer closed (close_notify or EOF), -1 = error / timeout (err set).
  long read(void *data, size_t n, Deadline deadline, std::string &err);
  void shutdown();

 private:
  void *ctx_ = nullptr;
  void *ssl_ = nullptr;
  int fd_ = -1;
  bool handshake(bool client, Deadline deadline, std::string &err);
};

}  // namespace xenithium:util:tls
