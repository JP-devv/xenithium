// Gate for the byte-level BPE tokenizer: ids equal to the reference tokenizer's
// AutoTokenizer) on the golden corpus tests/data/tokenizer_golden.json, decodes
// in both skip_special modes, utf8_at, the streaming Detokenizer and StopMatcher, plus load time
// and encode throughput. Needs the checkpoint's tokenizer.json (skips the golden part without it).
#include <sys/stat.h>
#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "xenithium/text/tokenizer.hpp"
#include "xenithium/util/json.hpp"


namespace {
namespace tx = xenithium::text;
namespace json = xenithium::json;

int g_checks = 0, g_failures = 0;
#define CHECK(cond, ...)                         \
  do {                                           \
    ++g_checks;                                  \
    if (!(cond)) {                               \
      ++g_failures;                              \
      std::printf("[FAIL] %s:%d ", __FILE__, __LINE__); \
      std::printf(__VA_ARGS__);                  \
      std::printf("\n");                         \
    }                                            \
  } while (0)

double now_s() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

bool exists(const std::string &p) {
  struct stat st;
  return ::stat(p.c_str(), &st) == 0;
}

std::string read_all(const std::string &p) {
  std::ifstream f(p, std::ios::binary);
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

std::string hex(std::string_view s) {
  std::string o;
  char b[4];
  for (unsigned char c : s) {
    std::snprintf(b, sizeof b, "%02X", c);
    o += b;
  }
  return o;
}

std::string ids_str(const std::vector<int32_t> &v, size_t max = 24) {
  std::string o = "[";
  for (size_t i = 0; i < v.size() && i < max; ++i) o += (i ? "," : "") + std::to_string(v[i]);
  return o + (v.size() > max ? ",...]" : "]");
}

std::vector<int32_t> to_ids(const json::Value &a) {
  std::vector<int32_t> v;
  for (const json::Value &x : a.arr) v.push_back(static_cast<int32_t>(std::stol(x.str)));
  return v;
}

// ---- utf8_at -----------------------------------------------------------------------------------
void test_utf8() {
  size_t len = 0;
  struct Case {
    std::string s;
    tx::Utf8 want;
    size_t len;
  };
  const Case cases[] = {
      {"A", tx::Utf8::complete, 1},
      {"\xC3\xA9", tx::Utf8::complete, 2},
      {"\xE2\x82\xAC", tx::Utf8::complete, 3},
      {"\xF0\x9F\x98\x80", tx::Utf8::complete, 4},
      {"\xF4\x8F\xBF\xBF", tx::Utf8::complete, 4},
      {"\xC3", tx::Utf8::incomplete, 1},
      {"\xE2\x82", tx::Utf8::incomplete, 2},
      {"\xF0\x9F\x98", tx::Utf8::incomplete, 3},
      {"\x80", tx::Utf8::invalid, 1},
      {"\xBF", tx::Utf8::invalid, 1},
      {"\xC0\xAF", tx::Utf8::invalid, 1},  // overlong
      {"\xC1\xBF", tx::Utf8::invalid, 1},
      {"\xE0\x80\x80", tx::Utf8::invalid, 1},  // overlong
      {"\xED\xA0\x80", tx::Utf8::invalid, 1},  // surrogate
      {"\xF0\x80\x80\x80", tx::Utf8::invalid, 1},
      {"\xF4\x90\x80\x80", tx::Utf8::invalid, 1},  // > U+10FFFF
      {"\xF5\x80\x80\x80", tx::Utf8::invalid, 1},
      {"\xFF", tx::Utf8::invalid, 1},
      {"\xE2\x82" "A", tx::Utf8::invalid, 1},  // cut short by a non-continuation byte
      {"\xC3" "A", tx::Utf8::invalid, 1},
      {"\xED\x9F\xBF", tx::Utf8::complete, 3},
      {"\xEE\x80\x80", tx::Utf8::complete, 3},
  };
  for (const Case &c : cases) {
    tx::Utf8 got = tx::utf8_at(c.s, 0, len);
    CHECK(got == c.want && len == c.len, "utf8_at(%s) = %d/%zu, want %d/%zu", hex(c.s).c_str(), int(got), len,
          int(c.want), c.len);
  }
  std::string s = "a\xC3\xA9\xE2\x82";
  CHECK(tx::utf8_at(s, 1, len) == tx::Utf8::complete && len == 2, "utf8_at offset");
  CHECK(tx::utf8_at(s, 3, len) == tx::Utf8::incomplete && len == 2, "utf8_at offset tail");
}

// ---- StopMatcher (no tokenizer needed) ------------------------------------------------------------
std::string run_stops(std::vector<std::string> stops, const std::vector<std::string> &pieces, bool *stopped,
                      std::vector<std::string> *steps = nullptr) {
  tx::StopMatcher m(std::move(stops));
  std::string out;
  bool st = false;
  for (const std::string &p : pieces) {
    size_t before = out.size();
    st = m.feed(p, out);
    if (steps) steps->push_back(out.substr(before));
  }
  if (!st) m.flush(out);
  if (stopped) *stopped = st;
  return out;
}

void test_stop_matcher() {
  bool st;
  std::vector<std::string> steps;
  // a match spanning pieces; text before it released, the match and everything after dropped
  CHECK(run_stops({"STOP"}, {"hello S", "TO", "P world"}, &st, &steps) == "hello " && st, "spanning match");
  CHECK(steps.size() == 3 && steps[0] == "hello " && steps[1].empty() && steps[2].empty(), "holdback steps");
  // partial prefix held back, then released when it does not continue
  steps.clear();
  CHECK(run_stops({"STOP"}, {"ab ST", "Xcd"}, &st, &steps) == "ab STXcd" && !st, "release after false prefix");
  CHECK(steps.size() == 2 && steps[0] == "ab " && steps[1] == "STXcd", "release step");
  // flush releases a held tail when no match came
  CHECK(run_stops({"STOP"}, {"abc", "ST"}, &st) == "abcST" && !st, "flush tail");
  // multiple stops: earliest match in the text wins
  CHECK(run_stops({"world", "lo"}, {"hello world"}, &st) == "hel" && st, "earliest of several");
  CHECK(run_stops({"\n\n", "END"}, {"line\n", "\nmore"}, &st) == "line" && st, "newline stop across pieces");
  // a stop that is a prefix of another; the longer one never needs more than the shorter
  CHECK(run_stops({"ab", "abc"}, {"xa", "bc"}, &st) == "x" && st, "prefix stops");
  // empty stop strings ignored
  CHECK(run_stops({"", ""}, {"abc"}, &st) == "abc" && !st, "empty stops ignored");
  CHECK(run_stops({}, {"a", "b"}, &st) == "ab" && !st, "no stops");
  // match at the very start, and nothing more after stopping
  {
    tx::StopMatcher m({"<|x|>"});
    std::string out;
    CHECK(m.feed("<|", out) == false && out.empty(), "start held");
    CHECK(m.feed("x|>tail", out) && out.empty() && m.stopped(), "match at start");
    CHECK(m.feed("more", out) && out.empty(), "stays stopped");
    m.flush(out);
    CHECK(out.empty(), "flush after stop");
  }
  // UTF-8 stop strings held at character boundaries
  CHECK(run_stops({"\xE3\x80\x82\xE3\x80\x82"}, {"\xE4\xB8\xAD\xE3\x80\x82", "\xE6\x96\x87"}, &st) ==
            "\xE4\xB8\xAD\xE3\x80\x82\xE6\x96\x87" && !st,
        "utf8 holdback released");
  // held text longer than one piece, match completes several pieces later
  CHECK(run_stops({"abcdef"}, {"x", "a", "b", "c", "d", "e", "f", "g"}, &st) == "x" && st, "long spanning");
  // overlapping prefix: "aab" in "aaab"
  CHECK(run_stops({"aab"}, {"a", "a", "a", "b"}, &st) == "a" && st, "overlapping prefix");
}

// GPT-2 byte-level character of a byte, as UTF-8 (to name single-byte tokens in tests).
std::string byte_token(int b) {
  int n = 0, cp = -1;
  for (int x = 0; x < 256; ++x) {
    bool printable = (x >= '!' && x <= '~') || (x >= 0xA1 && x <= 0xAC) || (x >= 0xAE && x <= 0xFF);
    int c = printable ? x : 256 + n++;
    if (x == b) cp = c;
  }
  std::string s;
  if (cp < 0x80) s.push_back(static_cast<char>(cp));
  else {
    s.push_back(static_cast<char>(0xC0 | (cp >> 6)));
    s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  }
  return s;
}

std::string stream(const tx::Tokenizer &tok, const std::vector<int32_t> &ids, bool skip,
                   std::vector<std::string> *steps = nullptr) {
  tx::Detokenizer d(tok, skip);
  std::string out;
  for (int32_t id : ids) {
    size_t before = out.size();
    d.push(id);
    d.take(out);
    if (steps) steps->push_back(out.substr(before));
  }
  d.flush(out);
  return out;
}

// flint: the Flint checkpoint (end of turn <|end_of_text|> 150001, roles <|user|> 150003), else the Qwen family.
void test_detokenizer(const tx::Tokenizer &tok, bool flint) {
  const char *end_name = flint ? "<|end_of_text|>" : "<|im_end|>";
  int32_t e2 = tok.token_id(byte_token(0xE2)), x82 = tok.token_id(byte_token(0x82)),
          xac = tok.token_id(byte_token(0xAC)), xff = tok.token_id(byte_token(0xFF)), A = tok.token_id("A"),
          im_end = tok.token_id(end_name), think = tok.token_id("<think>");
  CHECK(e2 >= 0 && x82 >= 0 && xac >= 0 && xff >= 0 && A >= 0, "byte tokens found");
  if (flint) CHECK(im_end == 150001 && think == 150006 && tok.token_id("<|user|>") == 150003 && !tok.is_special(think), "added token ids");
  else CHECK(im_end == 248046 && think >= 0 && tok.token_id("<|im_start|>") == 248045, "added token ids");
  CHECK(tok.token_id("Ġthe") >= 0 && tok.token_id("no-such-token-xyz") == -1, "token_id vocab form");
  std::vector<std::string> steps;
  // a multi-byte character split across tokens is released only when complete
  CHECK(stream(tok, {A, e2, x82, xac, A}, false, &steps) == "A\xE2\x82\xAC" "A", "split euro");
  CHECK(steps.size() == 5 && steps[0] == "A" && steps[1].empty() && steps[2].empty() && steps[3] == "\xE2\x82\xAC",
        "split euro steps");
  // an invalid byte becomes U+FFFD at once
  steps.clear();
  CHECK(stream(tok, {xff, A}, false, &steps) == "\xEF\xBF\xBD" "A" && steps[0] == "\xEF\xBF\xBD", "invalid byte");
  // a truncated sequence followed by a non-continuation: one U+FFFD (maximal subpart) when known
  steps.clear();
  CHECK(stream(tok, {e2, x82, A}, false, &steps) == "\xEF\xBF\xBD" "A" && steps[1].empty() &&
            steps[2] == "\xEF\xBF\xBD" "A",
        "truncated then ascii");
  // flush turns an incomplete tail into U+FFFD
  CHECK(stream(tok, {A, e2, x82}, false) == "A\xEF\xBF\xBD", "flush incomplete tail");
  CHECK(stream(tok, {x82, x82}, false) == "\xEF\xBF\xBD\xEF\xBF\xBD", "lone continuations");
  // skip_special
  CHECK(stream(tok, {A, im_end, A}, true) == "AA", "skip special");
  CHECK(stream(tok, {A, im_end, A}, false) == std::string("A") + end_name + "A", "keep special");
  CHECK(stream(tok, {think}, true) == tok.decode({think}, true), "think as decode");
  CHECK(tok.is_special(im_end) && !tok.is_special(A) && !tok.is_special(-1) && !tok.is_special(1 << 30), "is_special");
  std::string s;
  tok.decode_piece(-5, s, false);
  tok.decode_piece(tok.vocab_size(), s, false);
  CHECK(s.empty(), "unknown ids decode to nothing");
}

std::string byte_token_sp(int b) {
  char name[8];
  std::snprintf(name, sizeof name, "<0x%02X>", b);
  return name;
}

// SentencePiece-style decoding rules beyond the golden's random cases (byte runs, the strip, streaming steps).
void test_detokenizer_sp(const tx::Tokenizer &tok) {
  auto bt = [&](int b) { return tok.token_id(byte_token_sp(b)); };
  const int32_t e2 = bt(0xE2), x82 = bt(0x82), xac = bt(0xAC), xff = bt(0xFF), hello = tok.token_id("\xE2\x96\x81Hello"),
                im_end = tok.token_id("<|im_end|>"), a = tok.token_id("a");
  CHECK(tok.byte_fallback() && e2 >= 0 && x82 >= 0 && xac >= 0 && xff >= 0 && hello >= 0 && im_end >= 0 && a >= 0, "sp tokens found");
  CHECK(tok.byte_value(e2) == 0xE2 && tok.byte_value(hello) == -1 && tok.byte_value(-1) == -1, "byte_value");
  std::vector<std::string> steps;
  CHECK(stream(tok, {hello, e2, x82, xac, a}, false, &steps) == "Hello\xE2\x82\xAC" "a", "sp euro");
  // the run is held until a text token ends it
  CHECK(steps.size() == 5 && steps[0] == "Hello" && steps[1].empty() && steps[3].empty() && steps[4] == "\xE2\x82\xAC" "a", "sp euro steps");
  // a run that is not valid UTF-8 as a whole: one U+FFFD per byte, the valid prefix included
  CHECK(tok.decode({e2, x82, xac, xff, a}, false) == "\xEF\xBF\xBD\xEF\xBF\xBD\xEF\xBF\xBD\xEF\xBF\xBD" "a", "sp invalid run");
  // a skipped special token does not end a run
  CHECK(tok.decode({e2, im_end, x82, xac}, true) == "\xE2\x82\xAC", "sp run across a skipped special");
  CHECK(tok.decode({e2, im_end, x82, xac}, false) == "\xEF\xBF\xBD<|im_end|>\xEF\xBF\xBD\xEF\xBF\xBD", "sp run ended by a kept special");
  // one leading space of the whole text
  CHECK(tok.decode({hello, hello}, false) == "Hello Hello" && tok.decode({im_end, hello}, true) == "Hello", "sp strip");
}

int run_model(const std::string &model, const std::string &golden_path, bool sp, bool flint = false) {
  if (!exists(model + "/tokenizer.json")) {
    std::printf("SKIP golden: %s/tokenizer.json not found (set XENITHIUM_TOKENIZER_MODEL%s)\n", model.c_str(), sp ? "_SP" : "");
    return 0;
  }
  tx::Tokenizer tok;
  std::string err;
  {
    tx::Tokenizer bad;
    CHECK(!bad.load("/nonexistent-dir", err) && !err.empty(), "load refuses a missing file");
  }
  double t0 = now_s();
  bool ok = tok.load(model, err);
  double load_s = now_s() - t0;
  CHECK(ok, "load: %s", err.c_str());
  if (!ok) return 1;
  std::printf("%s: load %.3f s, vocab_size %d\n", model.c_str(), load_s, tok.vocab_size());

  json::Value g;
  std::string gtext = read_all(golden_path);
  CHECK(json::parse(gtext, g, &err), "golden parse (%s): %s", golden_path.c_str(), err.c_str());
  const json::Value *cases = g.get_array("cases"), *dcases = g.get_array("decode_cases"), *meta = g.get_object("meta");
  if (!cases || !dcases || !meta) {
    std::printf("golden missing: %s\n", golden_path.c_str());
    return 1;
  }
  CHECK(std::to_string(tok.vocab_size()) == meta->get("vocab_size")->str, "vocab_size %d", tok.vocab_size());
  for (const json::Value &sid : meta->get_array("special_ids")->arr)
    CHECK(tok.is_special(std::stoi(sid.str)), "special id %s", sid.str.c_str());

  size_t n_cases = 0, n_tokens = 0, enc_bad = 0, dec_bad = 0;
  for (const json::Value &c : cases->arr) {
    const std::string &name = *c.get_string("name"), &text = *c.get_string("text");
    std::vector<int32_t> want = to_ids(*c.get("ids")), got = tok.encode(text);
    ++n_cases;
    n_tokens += want.size();
    bool e_ok = got == want;
    if (!e_ok) {
      ++enc_bad;
      size_t k = 0;
      while (k < got.size() && k < want.size() && got[k] == want[k]) ++k;
      std::vector<int32_t> gt(got.begin() + std::min(k, got.size()), got.end()),
          wt(want.begin() + std::min(k, want.size()), want.end());
      std::printf("  encode %s: first diff at %zu: got %s want %s\n", name.c_str(), k, ids_str(gt, 8).c_str(),
                  ids_str(wt, 8).c_str());
    }
    CHECK(e_ok, "encode %s", name.c_str());
    for (bool skip : {false, true}) {
      std::string d = tok.decode(want, skip);
      const std::string &wd = *c.get_string(skip ? "dec_skip" : "dec");
      bool d_ok = d == wd;
      dec_bad += !d_ok;
      CHECK(d_ok, "decode %s skip=%d: got %s want %s", name.c_str(), skip, hex(d.substr(0, 40)).c_str(),
            hex(wd.substr(0, 40)).c_str());
      CHECK(stream(tok, want, skip) == wd, "stream %s skip=%d", name.c_str(), skip);
    }
  }
  size_t n_dec = 0;
  for (const json::Value &c : dcases->arr) {
    std::vector<int32_t> ids = to_ids(*c.get("ids"));
    for (bool skip : {false, true}) {
      const std::string &wd = *c.get_string(skip ? "dec_skip" : "dec");
      std::string d = tok.decode(ids, skip);
      ++n_dec;
      dec_bad += d != wd;
      CHECK(d == wd, "decode_case %s skip=%d: got %s want %s", ids_str(ids).c_str(), skip, hex(d).c_str(),
            hex(wd).c_str());
      CHECK(stream(tok, ids, skip) == wd, "stream decode_case %s skip=%d", ids_str(ids).c_str(), skip);
    }
  }
  std::printf("golden: %zu cases, %zu tokens, %zu encode mismatches; %zu decode-only checks, %zu decode mismatches\n",
              n_cases, n_tokens, enc_bad, n_dec, dec_bad);

  if (sp) test_detokenizer_sp(tok);
  else test_detokenizer(tok, flint);
  CHECK(!tok.encode("\xFF\xE2\x82" "abc").empty(), "invalid UTF-8 input encodes (as U+FFFD)");

  // load refuses a split pattern it does not implement (the classic pattern with \p{N} dropped).
  if (!sp) {
    std::string js = read_all(model + "/tokenizer.json");
    const std::string mark = flint ? R"(\\p{N}{1,3})" : R"(\\p{N}| ?)";
    size_t at = js.find(mark);
    CHECK(at != std::string::npos, "pattern located in tokenizer.json");
    if (at != std::string::npos) {
      js.replace(at, mark.size(), flint ? R"(\\p{N}{1,4})" : " ?");
      const char *tmp = std::getenv("TMPDIR");
      std::string tmpl = std::string(tmp ? tmp : "/tmp") + "/xenithium-tok-XXXXXX";
      std::vector<char> buf(tmpl.begin(), tmpl.end());
      buf.push_back('\0');
      if (::mkdtemp(buf.data())) {
        std::string d = buf.data();
        std::ofstream(d + "/tokenizer.json", std::ios::binary) << js;
        tx::Tokenizer other;
        std::string e;
        CHECK(!other.load(d, e) && e.find("split pattern") != std::string::npos, "refuses an unknown split pattern: %s", e.c_str());
        std::remove((d + "/tokenizer.json").c_str());
        ::rmdir(d.c_str());
      }
    }
  }

  // Thread safety: concurrent encodes (shared word cache) agree with the golden.
  {
    std::vector<int> bad(4, 0);
    std::vector<std::thread> th;
    for (int t = 0; t < 4; ++t)
      th.emplace_back([&, t] {
        for (size_t i = t; i < cases->arr.size(); i += 2)
          if (tok.encode(*cases->arr[i].get_string("text")) != to_ids(*cases->arr[i].get("ids"))) ++bad[t];
      });
    for (auto &x : th) x.join();
    CHECK(bad[0] + bad[1] + bad[2] + bad[3] == 0, "concurrent encode");
  }

  // Throughput on a long text: every long golden text, repeated; cold (fresh tokenizer cache
  // contents do not matter much: texts repeat words) and warm.
  {
    std::string big;
    for (const json::Value &c : cases->arr)
      if (c.get_string("name")->rfind("long-", 0) == 0) big += *c.get_string("text");
    std::string text;
    while (text.size() < (4u << 20)) text += big;
    tx::Tokenizer fresh;  // empty word cache
    fresh.load(model, err);
    double a = now_s();
    size_t n1 = fresh.encode(big).size();
    double cold = now_s() - a;
    a = now_s();
    size_t n = tok.encode(text).size();
    double warm = now_s() - a;
    std::printf("encode: cold cache %zu KB, %zu tokens: %.2f M tok/s; %.1f MB, %zu tokens: %.2f M tok/s (%.1f MB/s)\n",
                big.size() / 1024, n1, n1 / cold / 1e6, text.size() / 1e6, n, n / warm / 1e6, text.size() / warm / 1e6);
  }
  return 0;
}

// The golden sits next to this source (CMake compiles it by absolute path), so ctest finds it from the build dir.
std::string golden(const char *env, const char *file) {
  if (std::getenv(env)) return std::getenv(env);
  std::string here = __FILE__;
  std::string p = here.substr(0, here.find_last_of('/') + 1) + "data/" + file;
  return exists(p) ? p : std::string("tests/data/") + file;
}

}  // namespace

int main() {
  test_utf8();
  test_stop_matcher();
  // byte-level (Qwen family), SentencePiece-style (Nanbeige4.3) and Flint; XENITHIUM_TOKENIZER_MODEL[_SP] /
  // XENITHIUM_TOKENIZER_GOLDEN[_SP] override the checkpoint / golden.
  const char *env = std::getenv("XENITHIUM_TOKENIZER_MODEL"), *env_sp = std::getenv("XENITHIUM_TOKENIZER_MODEL_SP");
  const char *home = std::getenv("HOME");
  const std::string models = std::string(home ? home : "") + "/models/";  // the checkpoints' default folder
  int rc = run_model(env ? env : models + "qwen38-w4a16", golden("XENITHIUM_TOKENIZER_GOLDEN", "tokenizer_golden.json"), false);
  rc |= run_model(env_sp ? env_sp : models + "nanbeige43-33b-int4", golden("XENITHIUM_TOKENIZER_GOLDEN_SP", "tokenizer_golden_sp.json"), true);
  // byte-level with the Flint pre-tokenizer (digit groups, CJK runs, cased words); XENITHIUM_TOKENIZER_MODEL_FLINT overrides
  const char *env_fl = std::getenv("XENITHIUM_TOKENIZER_MODEL_FLINT");
  rc |= run_model(env_fl ? env_fl : models + "flint-int4", golden("XENITHIUM_TOKENIZER_GOLDEN_FLINT", "tokenizer_golden_flint.json"), false, true);
  std::printf("tokenizer: %d checks, %d failed\n", g_checks, g_failures);
  return (g_failures || rc) ? 1 : 0;
}
