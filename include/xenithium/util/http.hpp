// Minimal HTTP/1.1 client over POSIX sockets: plain http:// only, one request per connection
// (Connection: close). Enough for the tools that talk to a local OpenAI-compatible server.
#pragma once

#include <string>

namespace xenithium::http {

struct Response {
  int status = 0;
  std::string body;
};

// method "GET" or "POST"; url "http://host[:port]/path". Body sent as application/json when not
// empty. Throws std:runtime_error on connect / protocol / timeout errors (not on HTTP status).
Response request(const std::string &method, const std::string &url, const std::string &body, double timeout_s = 600);

}  // namespace xenithium:http
