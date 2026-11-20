// Host-only gate for the chat-template Jinja interpreter (text/jinja.hpp) against tests/data/jinja_golden.json: small
// templates over the features chat templates use, each with its golden render. Every case renders byte-equal, or
// fails where the golden records an error (raise_exception and undefined-value errors with the golden's exact
// message).
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>

#include "xenithium/text/jinja.hpp"
#include "xenithium/util/json.hpp"

namespace {
namespace json = xenithium::json;
namespace jinja = xenithium::text::jinja;

int g_failures = 0, g_checks = 0;
bool check(bool ok, const std::string &what) {
  ++g_checks;
  if (!ok) {
    ++g_failures;
    std::printf("[FAIL] %s\n", what.c_str());
  }
  return ok;
}
std::string show(const std::string &s) {
  std::string o;
  json::write_string(s, o);
  return o;
}

}  // namespace

int main(int argc, char **argv) {
  std::string path = argc > 1 ? argv[1] : std::string(XENITHIUM_SOURCE_DIR) + "/tests/data/jinja_golden.json";
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    std::printf("jinja: cannot open %s\n", path.c_str());
    return 2;
  }
  std::stringstream ss;
  ss << in.rdbuf();
  json::Value g;
  std::string err;
  if (!json::parse(ss.str(), g, &err)) {
    std::printf("jinja: bad golden: %s\n", err.c_str());
    return 2;
  }
  size_t n = 0, errors = 0;
  for (const auto &c : g.get_array("cases")->arr) {
    ++n;
    const std::string name = *c.get_string("name");
    std::string perr;
    auto t = jinja::Template::parse(*c.get_string("template"), &perr);
    std::string out, rerr;
    const bool ok = t && t->render(*c.get("context"), out, rerr);
    if (const std::string *want = c.get_string("output")) {
      if (!check(ok && out == *want, name)) {
        if (!t) std::printf("    parse error: %s\n", perr.c_str());
        else if (!ok) std::printf("    render error: %s\n", rerr.c_str());
        else std::printf("    want: %s\n    got:  %s\n", show(*want).c_str(), show(out).c_str());
      }
      continue;
    }
    ++errors;
    const std::string &type = *c.get_string("error"), &msg = *c.get_string("message");
    if (!check(!ok, name + " fails (" + type + ")")) {
      std::printf("    got: %s\n", show(out).c_str());
      continue;
    }
    if (type == "TemplateError" || type == "UndefinedError")
      if (!check(rerr == msg, name + " message")) std::printf("    want: %s\n    got:  %s\n", msg.c_str(), (t ? rerr : perr).c_str());
  }
  // constructs outside the subset are parse errors (the caller falls back to the built-in renderer)
  for (const char *bad : {"{% include 'x' %}", "{% extends 'x' %}", "{% call f() %}{% endcall %}", "{% if x %}", "{{ 1 + }}", "{% for %}"}) {
    std::string e;
    check(jinja::Template::parse(bad, &e) == nullptr && !e.empty(), std::string("rejects ") + bad);
  }
  std::printf("jinja: %zu golden cases (%zu errors), %d checks, %d failed\n", n, errors, g_checks, g_failures);
  return g_failures ? 1 : 0;
}
