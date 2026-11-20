// Gates for the bake tool's host pieces (src/quant): snappy, the Python-literal converter, fp16
// rounding, the quantiser, RTN == a zero-step tune, the CPU AutoRound reference beating RTN on
// held-out rows, the calibration file round trip, and the parquet reader on a real dataset file
// when ~/ai/datasets is present.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

#include "xenithium/quant/autoround.hpp"
#include "xenithium/quant/calib.hpp"
#include "xenithium/quant/parquet.hpp"
#include "xenithium/util/json.hpp"
#include "quant_synth.hpp"

namespace {
namespace q = xenithium::quant;
using quant_test::Synth;
using quant_test::make_synth;

int g_failures = 0;
void expect(bool ok, const std::string &what) {
  std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
  if (!ok) ++g_failures;
}

void test_snappy() {
  // literal "abcd", copy-1 (len 8, offset 4) -> "abcdabcdabcd"
  const uint8_t s1[] = {12, 3 << 2, 'a', 'b', 'c', 'd', static_cast<uint8_t>(1 | ((8 - 4) << 2)), 4};
  expect(q::snappy_decompress(s1, sizeof s1) == "abcdabcdabcd", "snappy: literal + overlapping copy-1");
  // copy-2: "xy" then copy len 6 offset 2
  const uint8_t s2[] = {8, 1 << 2, 'x', 'y', static_cast<uint8_t>(2 | ((6 - 1) << 2)), 2, 0};
  expect(q::snappy_decompress(s2, sizeof s2) == "xyxyxyxy", "snappy: copy-2");
  const uint8_t bad[] = {5, 1 << 2, 'x', 'y', static_cast<uint8_t>(1 | (0 << 2)), 9};
  bool threw = false;
  try {
    q::snappy_decompress(bad, sizeof bad);
  } catch (const std::exception &) {
    threw = true;
  }
  expect(threw, "snappy: an offset past the output throws");
}

void test_pylit() {
  const std::string in = "[{'id': 'call_0', 'ok': True, 'n': None, 'x': -1.5, 'args': '{\"path\": \"a\\'b\"}', 'u': '\\u00e9\\n'}]";
  xenithium::json::Value v;
  const bool ok = xenithium::json::parse(q::python_literal_to_json(in), v);
  expect(ok && v.is_array() && v.arr.size() == 1, "python literal: parses to JSON");
  if (ok) {
    const auto &o = v.arr[0];
    expect(o.get_string("args") && *o.get_string("args") == "{\"path\": \"a'b\"}", "python literal: nested quotes kept");
    expect(o.get("ok") && o.get("ok")->is_bool() && o.get("n") && o.get("n")->is_null(), "python literal: True / None");
    expect(o.get_string("u") && *o.get_string("u") == "\xc3\xa9\n", "python literal: \\u escape and newline");
  }
}

void test_half() {
  // against the compiler's _Float16 conversion on a sweep of floats
  std::mt19937 rng(1);
  int bad = 0;
  for (int i = 0; i < 2000000; ++i) {
    uint32_t u = rng();
    u = (u & 0x807fffffu) | ((100u + (rng() % 60u)) << 23);  // exponents across the half range
    float f;
    std::memcpy(&f, &u, 4);
    const _Float16 h = static_cast<_Float16>(f);
    uint16_t hb;
    std::memcpy(&hb, &h, 2);
    if (q::ar_half_bits(f) != hb) ++bad;
    if (q::ar_half(hb) != static_cast<float>(h)) ++bad;
  }
  expect(bad == 0, "fp16: ar_half_bits / ar_half == _Float16 on 2M floats (" + std::to_string(bad) + " off)");
}

void test_quantiser() {
  // ties round to even, clamp at -8 / 7, all-zero group
  expect(q::ar_code(2.5f, 1.f, 0.f) == 2 && q::ar_code(3.5f, 1.f, 0.f) == 4, "code: ties to even");
  bool inr = true;
  expect(q::ar_code(-9.f, 1.f, 0.f, &inr) == -8 && !inr, "code: clamp below, out of range");
  expect(q::ar_code(7.6f, 1.f, 0.f, &inr) == 7 && !inr, "code: clamp above (7.5 -> 8 -> 7)");
  expect(q::ar_code(0.3f, 1.f, 0.3f) == 1, "code: v shifts the rounding");
  expect(q::ar_scale(0.f, 0.f, 1.f, 1.f) != 0.f, "scale: an all-zero group keeps a non-zero scale");
  const float s = q::ar_scale(-0.75f, 0.3f, 1.f, 1.f);
  expect(s == q::ar_round_half(0.75f / 8.f) && q::ar_code(-0.75f, s, 0.f) == -8, "scale: |min| dominates -> s = |min| / 8 > 0, min -> -8");
  const float s2 = q::ar_scale(-0.3f, 0.75f, 1.f, 1.f);
  expect(s2 == -q::ar_round_half(0.75f / 8.f) && q::ar_code(0.75f, s2, 0.f) == -8 && q::ar_code(-0.3f, s2, 0.f) == 3,
         "scale: max dominates -> s = -max / 8 < 0, max -> -8");
  float src[128], dst[128];
  for (int i = 0; i < 128; ++i) src[i] = (i - 64) * 0.5f;
  q::ar_fake_quant_rows(src, 1, 128, 128, dst);
  bool ok = true;
  for (int i = 0; i < 128; ++i) {
    const float amax = 32.f;
    const float want = std::nearbyint(src[i] * (127.f / amax)) * (amax / 127.f);
    ok = ok && dst[i] == want;
  }
  expect(ok, "A8 fake quant: amax / 127, rne");
}

void test_autoround_cpu() {
  Synth S = make_synth(40, 64, 3);
  // iters = 0 == RTN, bit for bit
  {
    std::vector<std::unique_ptr<q::ArWorker>> ws;
    ws.push_back(q::make_cpu_worker(8));
    q::ArConfig cfg;
    cfg.iters = 0;
    std::vector<q::ExpertQuant> out;
    q::tune_layer(S.s, cfg, S.d, S.src, ws, 8, out);
    bool same = true;
    for (int e = 0; e < S.s.experts; ++e) {
      q::ExpertQuant r;
      q::rtn_expert(S.s, S.src, e, r);
      same = same && r.gu_q == out[static_cast<size_t>(e)].gu_q && r.gu_s == out[static_cast<size_t>(e)].gu_s &&
             r.dn_q == out[static_cast<size_t>(e)].dn_q && r.dn_s == out[static_cast<size_t>(e)].dn_s;
    }
    expect(same, "autoround cpu: iters 0 == rtn_expert (codes and scales)");
  }
  {
    std::vector<std::unique_ptr<q::ArWorker>> ws;
    ws.push_back(q::make_cpu_worker(16));
    q::ArConfig cfg;
    cfg.iters = 100;
    cfg.log_every = 0;
    std::vector<q::ExpertQuant> out;
    const auto st = q::tune_layer(S.s, cfg, S.d, S.src, ws, 8, out);
    char b[160];
    std::snprintf(b, sizeof b, "autoround cpu: held-out loss tuned %.4e < RTN %.4e (%.1f%%)", st.tuned_loss, st.rtn_loss,
                  100.0 * (st.tuned_loss / st.rtn_loss - 1.0));
    expect(st.tuned_loss < st.rtn_loss, b);
    bool range = true;
    for (const auto &o : out)
      for (const int8_t c : o.gu_q) range = range && c >= -8 && c <= 7;
    expect(range, "autoround cpu: codes in [-8, 7]");
  }
}

void test_calib_io() {
  q::CalibSet c;
  c.seqlen = 5;
  c.sources = {"a", "bb"};
  c.source_of = {0, 1, 1};
  for (int i = 0; i < 15; ++i) c.tokens.push_back(i * 7);
  const std::string p = (std::filesystem::temp_directory_path() / "xen-test-calib.xcb").string();
  q::write_calib(p, c);
  const auto r = q::read_calib(p);
  std::filesystem::remove(p);
  expect(r.seqlen == 5 && r.sources == c.sources && r.source_of == c.source_of && r.tokens == c.tokens, "calib: XCB1 round trip");
}

void test_parquet_real() {
  const char *h = std::getenv("HOME");
  const std::string f = std::string(h ? h : ".") + "/ai/datasets/pile-10k/data";
  if (!std::filesystem::is_directory(f)) {
    std::printf("[SKIP] parquet: no %s\n", f.c_str());
    return;
  }
  for (const auto &e : std::filesystem::directory_iterator(f)) {
    if (e.path().extension() != ".parquet") continue;
    const auto pf = q::ParquetFile::open(e.path().string());
    const auto t = pf.strings("text");
    expect(pf.rows() == 10000 && t.size() == 10000 && t[0].rfind("It is done, and submitted.", 0) == 0,
           "parquet: pile-10k text column (10,000 rows, first row's text)");
    return;
  }
}

}  // namespace

int main() {
  test_snappy();
  test_pylit();
  test_half();
  test_quantiser();
  test_calib_io();
  test_parquet_real();
  test_autoround_cpu();
  std::printf("%s: %d failure(s)\n", g_failures ? "FAIL" : "PASS", g_failures);
  return g_failures ? 1 : 0;
}
