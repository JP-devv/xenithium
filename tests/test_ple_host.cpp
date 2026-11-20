// Host-only gate for PLE (precise floating point, like the code under test): the n-gram layout,
// the token -> row ids, the table rows read from the drive and their dequantisation, and the host
// reference of the block, all against goldens.
// The goldens are hashes for a 40-token sequence with eos inside it: the n-gram ids, the FP8 codes
// of those rows, and the dequantised bf16 rows. With the reference dump file present
// (XENITHIUM_PLE_REF) the host reference of the block is also run against the dump's layer
// output on real weights.
// Needs the checkpoint (XENITHIUM_MODEL, default ~/models/qwen38-w4a16); skipped without it.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

#include "xenithium/arch/qwen38/arch.hpp"
#include "xenithium/arch/qwen38/ple_table.hpp"
#include "xenithium/kernels/pack.hpp"
#include "xenithium/kernels/embed/ple_ref.hpp"
#include "xenithium/model/checkpoint.hpp"
#include "xenithium/model/config.hpp"

namespace {
namespace k = xenithium::kernels;
namespace m = xenithium::model;
namespace q38 = xenithium::arch::qwen38;

int g_failures = 0;
void expect(bool ok, const std::string &what) {
  std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
  if (!ok) ++g_failures;
}
uint64_t fnv(const void *p, size_t n, uint64_t h = 0xcbf29ce484222325ull) {
  const auto *b = static_cast<const uint8_t *>(p);
  for (size_t i = 0; i < n; ++i) h = (h ^ b[i]) * 0x100000001b3ull;
  return h;
}

// The golden sequence: eos at 7, 8 and 20; ids 0 and vocab - 1 after it.
const int64_t kToks[40] = {248045, 58708,  242161, 92009,  167557, 141786, 181896, 248044, 248044, 5311,   38448,  195949, 107093, 223548,
                           27057,  182899, 57802,  181403, 77420,  191245, 248044, 0,      248319, 172893, 130402, 185659, 74770,  200974,
                           20582,  20829,  180152, 67337,  216115, 214980, 222645, 224053, 245575, 177979, 100522, 245097};
const uint64_t kIdsHash = 0x15065f3dcad65db8ull, kCodesHash = 0x2c2bcffa2b934e9cull, kEmbHash = 0x3fc1d332ac2a509aull;
const int64_t kIds0[16] = {13981229,  30076798,  49239434,  61315716,  98540601,  111186732, 120378863, 146384625,
                           173637879, 184336749, 217808755, 222173721, 249579227, 277180543, 299757935, 312970430};

std::vector<uint8_t> read_all(const m::Checkpoint &ck, const std::string &name) {
  const auto &t = ck.at(name);
  std::vector<uint8_t> b(t.bytes);
  const int fd = ::open(ck.shards().at(t.shard).path.c_str(), O_RDONLY);
  size_t got = 0;
  while (fd >= 0 && got < b.size()) {
    const ssize_t r = ::pread(fd, b.data() + got, b.size() - got, static_cast<off_t>(t.offset + got));
    if (r <= 0) break;
    got += static_cast<size_t>(r);
  }
  if (fd >= 0) ::close(fd);
  if (got != b.size()) throw std::runtime_error("short read of " + name);
  return b;
}
std::vector<float> bf16_tensor(const m::Checkpoint &ck, const std::string &name) {
  const auto b = read_all(ck, name);
  std::vector<float> v(b.size() / 2);
  for (size_t i = 0; i < v.size(); ++i) {
    uint16_t x;
    std::memcpy(&x, b.data() + 2 * i, 2);
    v[i] = k::bf16_value(x);
  }
  return v;
}

void test_layout_and_reads(const m::Checkpoint &ck, const q38::Config &cfg, q38::PleTable &tab) {
  const auto &L = tab.layout();
  expect(true, "derived layout == the checkpoint's multipliers / sizes / offsets (" + std::to_string(L.heads) + " heads, " +
                   std::to_string(L.rows) + " rows, padded " + std::to_string(L.padded_rows) + ", shards cover them)");

  // ids of the golden sequence (context eos before position 0).
  std::vector<int64_t> ids(40 * L.heads);
  k::PleContext ctx;
  ctx.reset(L);
  tab.ids(kToks, 40, ctx, ids.data());
  bool row0 = std::equal(kIds0, kIds0 + 16, ids.begin());
  expect(fnv(ids.data(), ids.size() * 8) == kIdsHash && row0, "n-gram ids of 40 tokens (eos at 7, 8, 20) == the reference class (hash + row 0)");
  // The same ids one token at a time, through a context that is carried along (decode order).
  {
    std::vector<int64_t> one(40 * L.heads);
    k::PleContext c2;
    c2.reset(L);
    for (int t = 0; t < 40; ++t) tab.ids(kToks + t, 1, c2, one.data() + t * L.heads);
    expect(one == ids, "ids token by token == ids of the whole sequence");
  }
  // Rows from the drive, then the dequantisation.
  std::vector<uint8_t> codes(ids.size() * tab.head_dim());
  tab.read(ids.data(), 40, codes.data());
  expect(fnv(codes.data(), codes.size()) == kCodesHash, "FP8 rows read in place == the reference rows (640 rows of 160)");
  std::vector<uint16_t> emb(codes.size());
  tab.dequant(codes.data(), 40, emb.data());
  expect(fnv(emb.data(), emb.size() * 2) == kEmbHash, "dequantised rows (code table) == bf16(e4m3) * bf16(scale) of the reference");

  // The reader against direct reads of the shard tensors, at random ids and every shard edge.
  {
    const std::string E = q38::ple_prefix(cfg.ple_layer) + "ngram_embedding.";
    std::vector<int64_t> want;
    std::vector<std::pair<int, int64_t>> where;  // shard, row in shard
    int64_t first = 0;
    std::mt19937_64 rng(8);
    for (int s = 0; s < cfg.ple_split_parts; ++s) {
      const auto &t = ck.at(E + "shard_" + std::to_string(s) + ".weight");
      for (int64_t r : {int64_t{0}, t.shape[0] - 1, static_cast<int64_t>(rng() % t.shape[0])}) {
        want.push_back(first + r);
        where.push_back({s, r});
      }
      first += t.shape[0];
    }
    std::vector<uint8_t> got(want.size() * tab.head_dim());
    tab.reader().gather(want.data(), want.size(), got.data());
    bool ok = true;
    for (size_t i = 0; i < want.size(); ++i) {
      const auto &t = ck.at(E + "shard_" + std::to_string(where[i].first) + ".weight");
      std::vector<uint8_t> row(tab.head_dim());
      const int fd = ::open(ck.shards().at(t.shard).path.c_str(), O_RDONLY);
      ok &= ::pread(fd, row.data(), row.size(), static_cast<off_t>(t.offset + where[i].second * tab.head_dim())) == static_cast<ssize_t>(row.size());
      ::close(fd);
      ok &= std::memcmp(row.data(), got.data() + i * tab.head_dim(), row.size()) == 0;
    }
    expect(ok, "reader == direct reads of the shard tensors (" + std::to_string(want.size()) + " rows: first, last and one random row of every shard)");
    bool threw = false;
    try {
      const int64_t bad = L.padded_rows;
      tab.reader().gather(&bad, 1, got.data());
    } catch (const std::exception &) {
      threw = true;
    }
    expect(threw, "reader rejects a row id past the table");
  }
}

// The host reference of the block against the reference dump's layer output on real bf16 weights.
void test_block(const m::Checkpoint &ck, const q38::Config &cfg, const std::string &path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    std::printf("[SKIP] block reference vs the reference layer: no dump at %s\n", path.c_str());
    return;
  }
  int32_t hdr[8];
  f.read(reinterpret_cast<char *>(hdr), sizeof hdr);
  const int T = hdr[0], NH = hdr[1], H = hdr[2], S = hdr[3], E = hdr[4], C = S * H;
  auto skip = [&](size_t n) { f.seekg(static_cast<std::streamoff>(n), std::ios::cur); };
  auto bf = [&](size_t n) {
    std::vector<uint16_t> b(n);
    f.read(reinterpret_cast<char *>(b.data()), static_cast<std::streamsize>(n * 2));
    std::vector<float> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = k::bf16_value(b[i]);
    return v;
  };
  skip(static_cast<size_t>(T) * 8 + static_cast<size_t>(T) * NH * 8 + static_cast<size_t>(T) * NH * (E / NH));
  const auto emb = bf(static_cast<size_t>(T) * E), hidden = bf(static_cast<size_t>(T) * C), key = bf(static_cast<size_t>(T) * C),
             value = bf(static_cast<size_t>(T) * H), ple = bf(static_cast<size_t>(T) * C), out = bf(static_cast<size_t>(T) * C);
  if (!f) {
    expect(false, "read the reference dump " + path);
    return;
  }
  const std::string P = std::string(q38::kPrefix) + "layers." + std::to_string(cfg.ple_layer) + ".ple.";
  const auto wk = bf16_tensor(ck, P + "key_proj.weight"), wv = bf16_tensor(ck, P + "value_proj.weight"), nk = bf16_tensor(ck, P + "norm_key.weight"),
             nq = bf16_tensor(ck, P + "norm_query.weight"), ncw = bf16_tensor(ck, P + "norm_conv.weight"), cw = bf16_tensor(ck, P + "conv1d.weight");

  // Projections: ours (double sums, one rounding) vs the dump's (a bf16 matmul).
  size_t kv_off = 0, kv_n = 0;
  double kv_worst = 0;
  for (int t = 0; t < T; ++t) {
    double rms[2] = {0, 0};  // of the dump's key row and value row
    for (int n = 0; n < C; ++n) rms[0] += static_cast<double>(key[static_cast<size_t>(t) * C + n]) * key[static_cast<size_t>(t) * C + n];
    for (int n = 0; n < H; ++n) rms[1] += static_cast<double>(value[static_cast<size_t>(t) * H + n]) * value[static_cast<size_t>(t) * H + n];
    rms[0] = std::sqrt(rms[0] / C);
    rms[1] = std::sqrt(rms[1] / H);
    for (int n = 0; n < C + H; ++n) {
      const float *w = n < C ? &wk[static_cast<size_t>(n) * E] : &wv[static_cast<size_t>(n - C) * E];
      double acc = 0;
      for (int i = 0; i < E; ++i) acc += static_cast<double>(w[i]) * emb[static_cast<size_t>(t) * E + i];
      const float mine = k::bf16_value(k::bf16_bits(static_cast<float>(acc)));
      const float ref = n < C ? key[static_cast<size_t>(t) * C + n] : value[static_cast<size_t>(t) * H + n - C];
      ++kv_n;
      if (mine != ref) {
        ++kv_off;
        kv_worst = std::max(kv_worst, std::fabs(mine - ref) / std::max({std::fabs(static_cast<double>(ref)), rms[n < C ? 0 : 1], 1e-30}));
      }
    }
  }
  char b[256];
  std::snprintf(b, sizeof b, "kv projections (bf16 weights) vs the reference: %zu / %zu differ, worst %.2e of max(|ref|, row rms)", kv_off, kv_n,
                kv_worst);
  expect(kv_worst <= 1.0 / 128 && kv_off * 100 <= kv_n, b);

  // Gate + conv teacher-forced at the dump's key / value, token by token with a ring.
  std::vector<float> normed(static_cast<size_t>(T) * C), res(hidden), gated(C);
  size_t off = 0, n = 0, ple_off = 0;
  double worst = 0;
  for (int t = 0; t < T; ++t) {
    const size_t o = static_cast<size_t>(t) * C;
    k::ple_ref_gate(H, S, &key[o], &value[static_cast<size_t>(t) * H], &hidden[o], nk.data(), nq.data(), ncw.data(), static_cast<float>(cfg.rms_eps),
                    gated.data(), &normed[o]);
    const float *taps[k::kPleTaps];
    for (int j = 0; j < k::kPleTaps; ++j) {
      const int p = t - k::kPleDil * (k::kPleTaps - 1 - j);
      taps[j] = p >= 0 ? &normed[static_cast<size_t>(p) * C] : nullptr;
    }
    float *r = &res[o];
    std::vector<float> pv(C, 0.f);  // conv into zeros = the block's output alone
    k::ple_ref_conv(C, taps, cw.data(), gated.data(), pv.data());
    k::ple_ref_conv(C, taps, cw.data(), gated.data(), r);
    for (int c = 0; c < C; ++c) {
      ++n;
      ple_off += pv[c] != ple[o + c];
      if (r[c] != out[o + c]) {
        ++off;
        worst = std::max(worst, std::isfinite(out[o + c]) ? std::fabs(r[c] - out[o + c]) / std::max(std::fabs(static_cast<double>(out[o + c])), 1e-30) : HUGE_VAL);
      }
    }
  }
  std::snprintf(b, sizeof b,
                "block (teacher-forced at k / v, 40 tokens incl. the conv start) vs the reference layer: ple %zu, out %zu of %zu differ, out worst %.2e",
                ple_off, off, n, worst);
  expect(worst <= 1.0 / 64 && off * 200 <= n, b);
}

}  // namespace

int main() {
  try {
    const char *md = std::getenv("XENITHIUM_MODEL");
    const std::string dir = md ? md : std::string(std::getenv("HOME") ? std::getenv("HOME") : "") + "/models/qwen38-w4a16";
    if (::access((dir + "/config.json").c_str(), R_OK) != 0) {
      std::printf("[SKIP] no checkpoint at %s\nOK: 0 failure(s)\n", dir.c_str());
      return 0;
    }
    const auto ck = m::Checkpoint::open(dir);
    const auto cfg = q38::Config::parse(m::ModelConfig::load(dir));
    q38::PleTable tab(ck, cfg, 16);
    test_layout_and_reads(ck, cfg, tab);
    const char *rp = std::getenv("XENITHIUM_PLE_REF");
    test_block(ck, cfg, rp ? rp : ".spike/r08/ref/ple_ref.bin");
  } catch (const std::exception &e) {
    std::printf("[FAIL] exception: %s\n", e.what());
    ++g_failures;
  }
  if (g_failures) {
    std::printf("FAILED: %d failure(s)\n", g_failures);
    return 1;
  }
  std::printf("OK: 0 failure(s)\n");
  return 0;
}
