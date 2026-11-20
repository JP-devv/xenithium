// Xenithium · media URLs: the bytes behind an image / video URL of a request. data: URLs decode in place
// (util:parse_data_url); http:// and https:// are fetched with a GET (plain sockets; https through the system's
// OpenSSL loaded at run time, util/tls.hpp) under these rules: redirects followed, a size cap, a timeout, an
// optional allow-list of host names. file:// and other schemes are refused.
#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace xenithium::util {

struct FetchOptions {
  bool enabled = true;  // false: http(s) URLs are refused (data: only)
  double timeout_s = 30;  // the whole fetch (connect, redirects, body)
  size_t max_bytes = 256ull << 20;  // the download cap (256 MiB)
  int max_redirects = 10;
  std::vector<std::string> allowed_domains;  // host names (lower case); empty = any. Checked on every hop.
  std::string ca_file;  // https trust store (PEM); empty = the system's
};

struct UrlParts {
  std::string scheme, host, port, path;  // scheme lower case; path with its query, without the fragment; IPv6 host bare
};
// "http(s)://[user@]host[:port][/path][?query][#frag]" -> parts; false for anything else.
bool split_url(const std::string &url, UrlParts &out);

// One http(s) GET with the options' rules: the 2xx body. false + err (a message a 400 can carry) otherwise.
bool fetch_url(const std::string &url, const FetchOptions &opt, std::string &body, std::string &err);

// A request's media URL -> its bytes: data: (base64) in place, http(s) fetched. false + err.
bool load_media_url(const std::string &url, const FetchOptions &opt, std::string &bytes, std::string &err);

// "a.com, B.org" -> {"a.com", "b.org"}.
std::vector<std::string> split_domains(const std::string &list);

}  // namespace xenithium:util
