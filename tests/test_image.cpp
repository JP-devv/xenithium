// Gate for the image input (xenithium/util/image.hpp): base64 / data URLs, and the PNG / JPEG decoders against the
// golden decoded pixels (tests/data/vision/: each file's RGB after the EXIF / transparency rules). Every case must be
// pixel-exact.
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "xenithium/util/image.hpp"
#include "xenithium/util/json.hpp"

namespace {
namespace u = xenithium::util;
namespace json = xenithium::json;

int g_checks = 0, g_failures = 0;
#define CHECK(cond, ...)                                  \
  do {                                                    \
    ++g_checks;                                           \
    if (!(cond)) {                                        \
      ++g_failures;                                       \
      std::printf("[FAIL] %s:%d ", __FILE__, __LINE__);   \
      std::printf(__VA_ARGS__);                           \
      std::printf("\n");                                  \
    }                                                     \
  } while (0)

std::string read_all(const std::string &p) {
  std::ifstream f(p, std::ios::binary);
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

void base64_cases() {
  std::string out, err;
  CHECK(u::base64_decode("aGVsbG8=", out) && out == "hello", "base64 hello");
  CHECK(u::base64_decode("aGVsbG8h", out) && out == "hello!", "base64 hello!");
  CHECK(u::base64_decode("aGk=", out) && out == "hi", "base64 hi");
  CHECK(u::base64_decode("", out) && out.empty(), "base64 empty");
  CHECK(!u::base64_decode("aGk", out), "base64 short quantum");
  CHECK(!u::base64_decode("aG=k", out), "base64 data after pad");
  CHECK(!u::base64_decode("aGk=\n", out), "base64 whitespace is invalid (strict)");
  for (int n = 0; n < 40; ++n) {
    std::string s;
    for (int i = 0; i < n; ++i) s += static_cast<char>((i * 37 + n) & 255);
    CHECK(u::base64_decode(u::base64_encode(s), out) && out == s, "base64 round trip %d", n);
  }
  CHECK(u::parse_data_url("data:image/png;base64,aGk=", out, err) && out == "hi", "data url");
  CHECK(!u::parse_data_url("https://x/y.png", out, err) && err.find("data:") != std::string::npos, "http url refused");
  CHECK(!u::parse_data_url("data:image/png,abc", out, err), "non-base64 data url refused");
  u::Image im;
  CHECK(!u::decode_image(reinterpret_cast<const uint8_t *>("GIF89a...."), 10, im, err), "gif refused");
}

void golden_cases(const std::string &dir) {
  const std::string idx_text = read_all(dir + "/index.json");
  if (idx_text.empty()) {
    std::printf("[SKIP] no goldens in %s\n", dir.c_str());
    return;
  }
  json::Value idx;
  std::string err;
  CHECK(json::parse(idx_text, idx, &err), "index.json: %s", err.c_str());
  for (const auto &kv : idx.obj) {
    const std::string &name = kv.first;
    const std::string file = read_all(dir + "/" +  *kv.second.get_string("path"));
    const std::string want = read_all(dir + "/" + name + ".rgb");
    if (file.empty() || want.empty()) {
      std::printf("[SKIP] %s: missing files\n", name.c_str());
      continue;
    }
    u::Image im;
    std::string e;
    const bool ok = u::decode_image(reinterpret_cast<const uint8_t *>(file.data()), file.size(), im, e);
    CHECK(ok, "%s: decode failed: %s", name.c_str(), e.c_str());
    if (!ok) continue;
    const int w = static_cast<int>(kv.second.get("w")->num), h = static_cast<int>(kv.second.get("h")->num);
    CHECK(im.w == w && im.h == h, "%s: size %dx%d, want %dx%d", name.c_str(), im.w, im.h, w, h);
    if (im.w != w || im.h != h || want.size() != im.rgb.size()) continue;
    size_t diff = 0;
    int maxd = 0;
    for (size_t i = 0; i < want.size(); ++i) {
      const int d = std::abs(static_cast<int>(static_cast<uint8_t>(want[i])) - im.rgb[i]);
      if (d) ++diff, maxd = std::max(maxd, d);
    }
    std::printf("  %-20s %5dx%-5d differing samples %zu of %zu (max %d)\n", name.c_str(), w, h, diff, want.size(), maxd);
    CHECK(diff == 0, "%s: %zu samples differ from the reference (max %d)", name.c_str(), diff, maxd);
  }
}

}  // namespace

int main(int argc, char **argv) {
  base64_cases();
  golden_cases(argc > 1 ? argv[1] : XENITHIUM_SOURCE_DIR "/tests/data/vision");
  std::printf("%s: %d checks, %d failures\n", g_failures ? "FAIL" : "PASS", g_checks, g_failures);
  return g_failures ? 1 : 0;
}
