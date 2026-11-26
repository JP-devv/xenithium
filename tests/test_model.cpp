// Model-layer gate (host only): the safetensors catalog on synthetic shards (every
// corruption must throw), then the real qwen38-w4a16 checkpoint when present
// (XEN_TEST_MODEL overrides the path; absent = those checks are skipped, not passed).
#include <algorithm>
#include <stdexcept>
#include <thread>
#include <future>
#include <chrono>
#include <atomic>
#include <bit>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <string>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

#include "xenithium/arch/flint/arch.hpp"
#include "xenithium/arch/llama/arch.hpp"
#include "xenithium/arch/qwen35/arch.hpp"
#include "xenithium/arch/qwen38/experts.hpp"
#include "xenithium/arch/qwen38/mtp.hpp"
#include "xenithium/kernels/pack.hpp"
#include "xenithium/model/arch.hpp"
#include "xenithium/model/bulk_read.hpp"
#include "xenithium/model/checkpoint.hpp"
#include "xenithium/model/kv_lend.hpp"
#include "xenithium/model/kv_pages.hpp"

namespace fs = std::filesystem;
namespace m = xenithium::model;

namespace {

int g_fail = 0, g_pass = 0;
void expect(bool ok, const std::string &what) {
  std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
  ok ? ++g_pass : ++g_fail;
}
bool throws(const std::function<void()> &f, std::string *msg = nullptr) {
  try {
    f();
  } catch (const std::exception &e) {
    if (msg) *msg = e.what();
    return true;
  }
  return false;
}

// Writes a safetensors file: header (padded with `pad` spaces) + `data.
void write_shard(const fs::path &p, const std::string &header, const std::string &data, int pad = 0) {
  std::string h = header + std::string(pad, ' ');
  std::ofstream o(p, std::ios::binary);
  uint64_t n = h.size();
  for (int i = 0; i < 8; ++i) o.put(static_cast<char>((n >> (8 * i)) & 0xff));
  o << h << data;
}

std::string first_bytes_hex(const std::string &path, uint64_t off, int n) {
  const int fd = ::open(path.c_str(), O_RDONLY);
  unsigned char b[16] = {};
  const bool ok = fd >= 0 && ::pread(fd, b, n, static_cast<off_t>(off)) == n;
  if (fd >= 0) ::close(fd);
  if (!ok) return "<read error>";
  std::string s;
  char x[3];
  for (int i = 0; i < n; ++i) {
    std::snprintf(x, sizeof x, "%02x", b[i]);
    s += x;
  }
  return s;
}

void synthetic() {
  const char *tmp = std::getenv("TMPDIR");
  const fs::path dir = fs::path(tmp ? tmp : "/tmp") / ("xen-test-model-" + std::to_string(::getpid()));
  fs::remove_all(dir);
  fs::create_directories(dir);
  const std::string data(4 + 12 + 2, '\x11');
  // a: I32 [1] at 0..4, b: BF16 [2,3] at 4..16, c: F8_E4M3 [2] at 16..18
  const std::string ok_hdr =
      R"({"__metadata__":{"format":"pt"},"a":{"dtype":"I32","shape":[1],"data_offsets":[0,4]},)"
      R"("b":{"dtype":"BF16","shape":[2,3],"data_offsets":[4,16]},"c":{"dtype":"F8_E4M3","shape":[2],"data_offsets":[16,18]}})";
  write_shard(dir / "m-1.safetensors", ok_hdr, data, 7);
  {
    const auto ck = m::Checkpoint::open(dir.string());
    expect(ck.size() == 3 && ck.shards().size() == 1, "synthetic: 3 tensors in 1 shard");
    const uint64_t base = 8 + ok_hdr.size() + 7;
    expect(ck.at("b").offset == base + 4 && ck.at("b").bytes == 12, "synthetic: absolute offset = 8 + header length read + begin");
    expect(ck.at("c").dtype == m::DType::f8_e4m3 && ck.at("b").shape == std::vector<int64_t>{2, 3}, "synthetic: dtype and shape");
    expect(ck.find("missing") == nullptr, "synthetic: absent name -> null");
  }
  struct Bad {
    const char *what, *hdr;
    size_t data;
  };
  for (const Bad b : {
           Bad{"range past the end of the file", R"({"a":{"dtype":"I32","shape":[1],"data_offsets":[0,4]}})", 2},
           Bad{"overlapping tensors",
               R"({"a":{"dtype":"I32","shape":[1],"data_offsets":[0,4]},"b":{"dtype":"I32","shape":[1],"data_offsets":[2,6]}})", 8},
           Bad{"bytes != numel x element size", R"({"a":{"dtype":"BF16","shape":[3],"data_offsets":[0,4]}})", 4},
           Bad{"begin > end", R"({"a":{"dtype":"I32","shape":[1],"data_offsets":[4,0]}})", 4},
           Bad{"not JSON", R"({"a":)", 4},
           Bad{"negative shape", R"({"a":{"dtype":"I32","shape":[-1],"data_offsets":[0,4]}})", 4},
       }) {
    fs::remove_all(dir);
    fs::create_directories(dir);
    write_shard(dir / "x.safetensors", b.hdr, std::string(b.data, '\0'));
    std::string msg;
    const bool t = throws([&] { m::Checkpoint::open(dir.string()); }, &msg);
    expect(t, std::string("synthetic: ") + b.what + " throws (" + msg + ")");
  }
  {
    fs::remove_all(dir);
    fs::create_directories(dir);
    std::ofstream(dir / "short.safetensors", std::ios::binary) << "abc";
    expect(throws([&] { m::Checkpoint::open(dir.string()); }), "synthetic: file shorter than the length prefix throws");
  }
  {
    fs::remove_all(dir);
    fs::create_directories(dir);
    const std::string h = R"({"a":{"dtype":"I32","shape":[1],"data_offsets":[0,4]}})";
    write_shard(dir / "1.safetensors", h, std::string(4, '\0'));
    write_shard(dir / "2.safetensors", h, std::string(4, '\0'));
    expect(throws([&] { m::Checkpoint::open(dir.string()); }), "synthetic: duplicate name across shards throws");
  }
  {
    fs::remove_all(dir);
    fs::create_directories(dir);
    write_shard(dir / "1.safetensors", R"({"a":{"dtype":"I32","shape":[1],"data_offsets":[0,4]}})", std::string(4, '\0'));
    std::ofstream(dir / "model.safetensors.index.json") << R"({"weight_map":{"a":"2.safetensors"}})";
    expect(throws([&] { m::Checkpoint::open(dir.string()); }), "synthetic: index naming the wrong shard throws");
    expect(!throws([&] { m::Checkpoint::open(dir.string(), false); }), "synthetic: index cross-check can be disabled");
  }
  fs::remove_all(dir);
}

// Bulk read on synthetic shards: data equals the bytes written for tensors at unaligned
// offsets (incl. one ending at EOF, not on a 4 KiB boundary), merging across a gap, a tiny
// piece size (many preads), the buffered path, a tensor outside the read, and a shard
// truncated after the catalog was built.
void bulk_synthetic() {
  const char *tmp = std::getenv("TMPDIR");
  const fs::path dir = fs::path(tmp ? tmp : "/tmp") / ("xen-test-bulk-" + std::to_string(::getpid()));
  fs::remove_all(dir);
  fs::create_directories(dir);
  // a: 5000 B at 0, b: 3 B at 5000, c: 20000 B at 9000 (gap 997 B), file ends right after c.
  std::string data(29000, '\0');
  for (size_t i = 0; i < data.size(); ++i) data[i] = static_cast<char>((i * 131 + 7) & 0xff);
  const std::string h =
      R"({"a":{"dtype":"U8","shape":[5000],"data_offsets":[0,5000]},"b":{"dtype":"U8","shape":[3],"data_offsets":[5000,5003]},)"
      R"("c":{"dtype":"U8","shape":[20000],"data_offsets":[9000,29000]}})";
  write_shard(dir / "s.safetensors", h, data, 3);
  const auto ck = m::Checkpoint::open(dir.string());
  auto check = [&](const m::BulkOptions &o, const std::string &label) {
    const auto buf = m::bulk_read(ck, {&ck.at("a"), &ck.at("c"), &ck.at("b")}, o);
    bool ok = std::memcmp(buf.data(ck.at("a")), data.data(), 5000) == 0 && std::memcmp(buf.data(ck.at("b")), data.data() + 5000, 3) == 0 &&
              std::memcmp(buf.data(ck.at("c")), data.data() + 9000, 20000) == 0;
    expect(ok, "bulk: bytes equal the file (" + label + ", " + std::to_string(buf.stats().ranges) + " range(s), " +
                   std::to_string(buf.stats().pieces) + " piece(s))");
  };
  m::BulkOptions o;
  check(o, "O_DIRECT, merged");
  o.merge_gap = 100;
  o.piece = 4096;
  o.threads = 3;
  check(o, "split at the gap, 4 KiB pieces");
  o.direct = false;
  check(o, "buffered");
  {
    const auto buf = m::bulk_read(ck, {&ck.at("a")});
    expect(throws([&] { buf.data(ck.at("c")); }), "bulk: a tensor outside the read throws");
  }
  {
    m::BufferPool pool;
    const auto o2 = pool.options();
    const uint8_t *first = nullptr;
    {
      const auto buf = m::bulk_read(ck, {&ck.at("a"), &ck.at("c")}, o2);
      first = buf.data(ck.at("a"));
    }
    const auto buf = m::bulk_read(ck, {&ck.at("a"), &ck.at("c")}, o2);
    expect(buf.data(ck.at("a")) == first && std::memcmp(buf.data(ck.at("c")), data.data() + 9000, 20000) == 0,
           "bulk: a pooled buffer is reused by the next read and holds the new bytes");
  }
  fs::resize_file(dir / "s.safetensors", fs::file_size(dir / "s.safetensors") - 100);
  std::string msg;
  expect(throws([&] { m::bulk_read(ck, {&ck.at("c")}); }, &msg), "bulk: shard truncated after the catalog throws (" + msg + ")");
  fs::remove_all(dir);
}

void real_model() {
  const char *env = std::getenv("XEN_TEST_MODEL");
  const char *home = std::getenv("HOME");
  const std::string dir = env ? env : std::string(home ? home : "") + "/models/qwen38-w4a16";
  if (!fs::exists(dir + "/config.json")) {
    std::printf("[SKIP] real model: %s not present\n", dir.c_str());
    return;
  }
  const auto cfg = m::ModelConfig::load(dir);
  const auto arch = m::open_arch(cfg);
  expect(arch->id() == "qwen38", "real: config.json selects arch qwen38");
  const auto ck = m::Checkpoint::open(dir);
  expect(ck.size() == 222716 && ck.shards().size() == 25,
         "real: 222,716 tensors in 25 shards (got " + std::to_string(ck.size()) + " in " + std::to_string(ck.shards().size()) + ")");
  std::vector<std::string> warn;
  const auto problems = arch->validate(ck, &warn);
  expect(problems.empty(), "real: validate() finds no problems (" + std::to_string(problems.size()) + ")");
  for (size_t i = 0; i < problems.size() && i < 5; ++i) std::printf("    %s\n", problems[i].c_str());

  // Golden offsets, computed independently with Python (struct + json over the headers).
  struct Golden {
    const char *name, *file;
    uint64_t offset, bytes;
    const char *hex;
  };
  for (const Golden g : {
           Golden{"lm_head.weight", "model-00017-of-00017.safetensors", 528, 1271398400, "6a3bcdbb70bc26ba"},
           Golden{"model.language_model.layers.47.mlp.experts.511.down_proj.scales", "model-00015-of-00017.safetensors",
                  5046404336ull, 25600, "109d6d9c1f1e301d"},
           Golden{"model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_127.weight",
                  "model-plefp8-00002.safetensors", 2400013758ull, 400001920, "e8595ddb60ccd55a"},
       }) {
    const auto *t = ck.find(g.name);
    const bool ok = t && ck.shards()[t->shard].file == g.file && t->offset == g.offset && t->bytes == g.bytes &&
                    first_bytes_hex(ck.shards()[t->shard].path, t->offset, 8) == g.hex;
    expect(ok, std::string("real: golden offset + first bytes of ") + g.name);
  }

  // Budget arithmetic (bf16 / 8-bit KV, drafter).
  m::DeployParams p;
  p.drafter = true;
  p.kv_bytes = 2;
  auto b = arch->budget(ck, p);
  expect(b.kv_bytes_per_token == 14144, "real: KV bf16 + drafter = 14,144 B/token/slot/card (" + std::to_string(b.kv_bytes_per_token) + ")");
  p.kv_bytes = 1;
  b = arch->budget(ck, p);
  expect(b.kv_bytes_per_token == 7592, "real: KV 8-bit + drafter = 7,592 B/token/slot/card (" + std::to_string(b.kv_bytes_per_token) + ")");
  expect(b.expert_bytes == 2534400, "real: one expert (codes + scales) = 2,534,400 B");
  expect(b.experts_per_card == 12288, "real: 12,288 routed experts per card at EP2");

  // Expert blob: the packer's sections equal the plain packers on the raw tensors.
  namespace q38 = xenithium::arch::qwen38;
  namespace kk = xenithium::kernels;
  const auto qc = q38::Config::parse(cfg);
  const auto lay = q38::ExpertLayout::of(qc);
  expect(lay.bytes == 2534400 && lay.stride == 2535424, "real: expert blob 2,534,400 B, stride 2,535,424 B");
  for (const int layer : {0, 47}) {
    const int e = layer == 0 ? 0 : 511;
    const auto ts = q38::expert_tensors(ck, layer, e);
    const auto buf = m::bulk_read(ck, ts);
    std::vector<uint8_t> blob(lay.stride, 0xA5);
    q38::pack_expert(lay, ck, buf, layer, e, blob.data());
    const int H = qc.hidden, I = qc.moe_inter;
    auto q = [&](const char *proj) { return buf.as<uint32_t>(ck.at(q38::expert_name(layer, e, proj, "qweight"))); };
    auto sc = [&](const char *proj) { return buf.data(ck.at(q38::expert_name(layer, e, proj, "scales"))); };
    // gate|up via an explicit concatenation along N.
    std::vector<uint32_t> cat(static_cast<size_t>(H / 8) * 2 * I);
    for (int r = 0; r < H / 8; ++r) {
      std::memcpy(&cat[static_cast<size_t>(r) * 2 * I], q("gate_proj") + static_cast<size_t>(r) * I, I * 4);
      std::memcpy(&cat[static_cast<size_t>(r) * 2 * I + I], q("up_proj") + static_cast<size_t>(r) * I, I * 4);
    }
    std::vector<uint8_t> gu(static_cast<size_t>(H) * I), dn(static_cast<size_t>(I) * H / 2);
    kk::pack_w4_gptq(cat.data(), 2 * I, H, gu.data());
    kk::pack_w4_gptq(q("down_proj"), H, I, dn.data());
    bool ok = std::memcmp(blob.data() + lay.gu_codes, gu.data(), gu.size()) == 0 &&
              std::memcmp(blob.data() + lay.dn_codes, dn.data(), dn.size()) == 0 &&
              std::memcmp(blob.data() + lay.dn_scales, sc("down_proj"), static_cast<size_t>(I / 128) * H * 2) == 0;
    for (int g = 0; g < H / 128 && ok; ++g)
      ok = std::memcmp(blob.data() + lay.gu_scales + static_cast<size_t>(g) * 4 * I, sc("gate_proj") + static_cast<size_t>(g) * 2 * I, 2 * I) == 0 &&
           std::memcmp(blob.data() + lay.gu_scales + static_cast<size_t>(g) * 4 * I + 2 * I, sc("up_proj") + static_cast<size_t>(g) * 2 * I, 2 * I) == 0;
    for (size_t i = lay.bytes; i < lay.stride && ok; ++i) ok = blob[i] == 0;
    expect(ok, "real: layer " + std::to_string(layer) + " expert " + std::to_string(e) + " blob == plain packers on the raw tensors");
  }
}

}  // namespace

// The MTP drafter's experts: fp16 conversion, g32 -> g128 requant, and the packed blob
// decoded back == the requantised codes and scales (synthetic expert, hidden 256, inter 128).
void mtp_host() {
  namespace q38 = xenithium::arch::qwen38;
  bool ok = true;
  for (uint32_t h = 0; h < 65536 && ok; ++h) {
    if ((h & 0x7C00u) == 0x7C00u && (h & 0x3FFu)) continue;  // NaN
    ok = q38::fp16_bits(q38::fp16_value(static_cast<uint16_t>(h))) == h;
  }
  uint32_t seed = 12345;
  auto rnd = [&] { return seed = seed * 1664525u + 1013904223u; };
  for (int i = 0; i < 2000000 && ok; ++i) {
    float f;
    uint32_t x = rnd();
    x = (x & 0x80000000u) | (0x30000000u + x % 0x18000000u);  // ~2^-31.. 2^17: subnormal, normal, overflow
    std::memcpy(&f, &x, 4);
    const _Float16 r = static_cast<_Float16>(f);
    uint16_t want;
    std::memcpy(&want, &r, 2);
    ok = q38::fp16_bits(f) == want;
    if (!ok) std::printf("  fp16 of %a: %04x, want %04x\n", static_cast<double>(f), q38::fp16_bits(f), want);
  }
  expect(ok, "mtp: fp16_bits == the compiler's _Float16 (every fp16 round trip, 2M random floats incl. subnormals / overflow)");

  const int H = 256, I = 128;
  auto g32 = [&](int n, int kd) {
    std::vector<uint8_t> pk(static_cast<size_t>(n) * kd / 2);
    for (auto &b : pk) b = static_cast<uint8_t>(rnd() >> 24);
    std::vector<uint16_t> sc(static_cast<size_t>(n) * kd / 32);
    for (auto &s : sc) s = xenithium::kernels::bf16_bits(1e-3f * (1 + (rnd() >> 24) % 50));
    return q38::mtp_unpack_g32(pk.data(), sc.data(), n, kd);
  };
  const q38::W4Rows gate = q38::mtp_requant_g128(g32(I, H)), up = q38::mtp_requant_g128(g32(I, H)), down = q38::mtp_requant_g128(g32(H, I));
  bool range = true;
  for (const auto *m : {&gate, &up, &down})
    for (int8_t c : m->q) range &= c >= -8 && c <= 7;
  expect(range && gate.G == 128 && gate.s.size() == static_cast<size_t>(I) * H / 128, "mtp: g128 requant codes in [-8, 7], one scale per 128");
  q38::ExpertLayout lay;
  lay.hidden = H;
  lay.inter = I;
  lay.gu_codes = 0;
  lay.dn_codes = static_cast<size_t>(H) * 2 * I / 2;
  lay.gu_scales = lay.dn_codes + static_cast<size_t>(I) * H / 2;
  lay.dn_scales = lay.gu_scales + static_cast<size_t>(H) / 128 * 2 * I * 2;
  lay.bytes = lay.dn_scales + static_cast<size_t>(I) / 128 * H * 2;
  lay.stride = (lay.bytes + 4095) / 4096 * 4096;
  std::vector<uint8_t> blob(lay.stride, 0xAB);
  const int64_t inexact = q38::mtp_pack_expert(lay, gate, up, down, blob.data());
  // Decode: block (k group g, 16-column tile t) of 1024 bytes, byte (kg / 2) * 16 + lane = column
  // 16 t + lane at kg (low nibble) and kg + 1 (high), stored code + 8.
  auto code = [&](size_t sec, int ncols, int kk, int col) {
    const size_t blk = (static_cast<size_t>(kk / 128) * (ncols / 16) + col / 16) * 1024;
    const uint8_t b = blob[sec + blk + static_cast<size_t>((kk % 128) / 2) * 16 + col % 16];
    return static_cast<int>(((kk & 1) ? b >> 4 : b & 15)) - 8;
  };
  auto scale = [&](size_t off) {
    uint16_t h;
    std::memcpy(&h, blob.data() + off, 2);
    return q38::fp16_value(h);
  };
  bool same = true;
  for (int c = 0; c < 2 * I && same; ++c)
    for (int kk = 0; kk < H && same; ++kk) {
      const q38::W4Rows &m = c < I ? gate : up;
      const int col = c % I;
      same = code(lay.gu_codes, 2 * I, kk, c) == m.q[static_cast<size_t>(col) * H + kk] &&
             scale(lay.gu_scales + (static_cast<size_t>(kk / 128) * 2 * I + c) * 2) == m.s[static_cast<size_t>(col) * (H / 128) + kk / 128];
    }
  for (int c = 0; c < H && same; ++c)
    for (int kk = 0; kk < I && same; ++kk)
      same = code(lay.dn_codes, H, kk, c) == down.q[static_cast<size_t>(c) * I + kk] &&
             scale(lay.dn_scales + (static_cast<size_t>(kk / 128) * H + c) * 2) == down.s[static_cast<size_t>(c) * (I / 128) + kk / 128];
  for (size_t i = lay.bytes; i < lay.stride && same; ++i) same = blob[i] == 0;
  expect(same && inexact == 0, "mtp: packed expert blob decodes to the g128 codes and scales (gate|up, down, fp16 exact, padding zero)");
}

// A dense Llama-family checkpoint (arch llama: Nanbeige4.3-33B-INT4, compressed-tensors int4 asymmetric g32): config
// selects the module, the checkpoint validates, the zero points' layout, the budget arithmetic and the dense plan.
void real_model_flint() {
  const char *env = std::getenv("XEN_TEST_MODEL_FLINT");
  const char *home = std::getenv("HOME");
  const std::string dir = env ? env : std::string(home ? home : "") + "/models/flint-int4";
  if (!fs::exists(dir + "/config.json")) {
    std::printf("[SKIP] real flint model: %s not present\n", dir.c_str());
    return;
  }
  const auto cfg = m::ModelConfig::load(dir);
  const auto arch = m::open_arch(cfg);
  expect(arch->id() == "flint", "flint: config.json (model_type flint) selects arch flint");
  const auto c = xenithium::arch::flint::Config::parse(cfg);
  expect(c.layers == 49 && c.hidden == 2048 && c.q_heads == 32 && c.kv_heads == 4 && c.head_dim == 128 && c.vocab == 150144 && c.window == 513 &&
             c.full_layers() == 14 && c.moe_layers() == 47 && c.experts == 256 && c.topk == 6 && c.moe_inter == 512 && c.shared_inter == 1024 &&
             c.inter == 6144 && c.w_bits == 4 && c.w_group == 64 && c.embed_bits == 8 && c.sinks && c.rope_theta == 1e4 && !c.sliding[0] &&
             !c.sliding[48] && c.sliding[2] && c.dense[0] && c.dense[1] && !c.dense[2] && c.eos.size() == 4,
         "flint: config (49 x 2048, 32 / 4 heads x 128, 35 sliding 513 + 14 full, 2 dense + 47 MoE 256 x top-6, MLX int4 g64, int8 embedding)");
  expect(xenithium::arch::flint::ring_tokens(c, 1) == 640 && xenithium::arch::flint::ring_tokens(c, 8) == 640,
         "flint: sliding ring = 640 tokens per slot (window 513 + 8 rows, whole pages + 1)");
  const auto ck = m::Checkpoint::open(dir);
  expect(ck.size() == 1751 && ck.shards().size() == 5, "flint: 1,751 tensors in 5 shards (got " + std::to_string(ck.size()) + ")");
  std::vector<std::string> warn;
  const auto problems = arch->validate(ck, &warn);
  expect(problems.empty(), "flint: validate() finds no problems (" + std::to_string(problems.size()) + ")");
  for (size_t i = 0; i < problems.size() && i < 5; ++i) std::printf("    %s\n", problems[i].c_str());
  size_t other = 0, experts = 0, router = 0;
  for (const auto &t : ck.tensors()) {
    const auto r = arch->classify(t.name);
    other += r.comp == m::Component::other;
    experts += r.comp == m::Component::routed_experts;
    router += r.comp == m::Component::router;
  }
  expect(other == 0 && experts == 47 * 9 && router == 47, "flint: every tensor classified; 47 x 9 stacked expert tensors, 47 routers");
  m::DeployParams p;
  p.kv_bytes = 2;
  p.rows = 1;
  auto b = arch->budget(ck, p);
  expect(b.kv_bytes_per_token == 14 * 2 * 4 * 128 * 2 / 2, "flint: full-attention KV bf16 = 14,336 B/token/slot/card (" + std::to_string(b.kv_bytes_per_token) + ")");
  expect(b.expert_bytes == 3.0 * 512 * 2048 * (0.5 + 4.0 / 64) && b.hot_experts_fit == b.experts_per_card && b.experts_per_card == 47 * 128,
         "flint: one expert 1,769,472 B (codes + bf16 scale / bias); every expert fits per card");
  size_t w4 = 0, w4_bad = 0;
  for (const auto &sp : arch->dense_plan(ck, p)) {
    if (sp.store != m::DenseSpec::Store::w4) continue;
    ++w4;
    if (sp.w4_layout != m::DenseSpec::W4Layout::mlx || sp.w4_group != 64 || sp.name.size() < 7 || sp.name.compare(sp.name.size() - 7, 7, ".weight")) ++w4_bad;
  }
  // lm_head + q|k|v and o per layer + the two dense MLPs' gate|up and down (the shared experts live in the MoE store)
  expect(w4 == 1 + 49 * 2 + 2 * 2 && w4_bad == 0, "flint: dense plan: 103 int4 matrices, all MLX g64 (" + std::to_string(w4) + ")");
  // the runtime's configuration (arch:qwen35:Model serves flint)
  const auto rc = xenithium::arch::flint::runtime_config(cfg);
  int sliding = 0, moe = 0;
  for (bool v : rc.sliding) sliding += v;
  for (bool v : rc.moe) moe += v;
  expect(rc.flint && rc.w_mlx && !rc.norm_offset && rc.qk_norm && !rc.attn_output_gate && rc.head_dim == 128 && rc.window == 513 && sliding == 35 &&
             moe == 47 && rc.experts == 256 && rc.topk == 6 && rc.prefix == "model." && rc.packed_name("model.layers.0.mlp.gate_proj") == "model.layers.0.mlp.gate_proj.weight" &&
             rc.scale_name("x") == "x.scales" && std::count(rc.full_attention.begin(), rc.full_attention.end(), true) == 49 && !rc.llama(),
         "flint: runtime config (every layer GQA, 35 sliding, 47 MoE, MLX names, plain norms with q / k norms)");
}

void real_model_llama() {
  const char *env = std::getenv("XEN_TEST_MODEL_LLAMA");
  const char *home = std::getenv("HOME");
  const std::string dir = env ? env : std::string(home ? home : "") + "/models/nanbeige43-33b-int4";
  if (!fs::exists(dir + "/config.json")) {
    std::printf("[SKIP] real llama model: %s not present\n", dir.c_str());
    return;
  }
  const auto cfg = m::ModelConfig::load(dir);
  const auto arch = m::open_arch(cfg);
  expect(arch->id() == "llama", "llama: config.json (model_type llama) selects arch llama");
  const auto c = xenithium::arch::llama::parse_config(cfg);
  expect(c.layers == 52 && c.hidden == 7168 && c.q_heads == 56 && c.kv_heads == 8 && c.head_dim == 128 && c.inter == 22016 && c.vocab == 166912 &&
             c.w_group == 32 && c.w_zero && !c.qk_norm && !c.norm_offset && !c.attn_output_gate && c.partial_rotary == 1.0 && c.rope_theta == 1.2e8 &&
             c.prefix == "model.",
         "llama: the dense decoder's config (52 x 7168, 56 / 8 heads x 128, int4 g32 with zero points, plain norm, full rotary)");
  const auto ck = m::Checkpoint::open(dir);
  expect(ck.size() == 1563 && ck.shards().size() == 315,
         "llama: 1,563 tensors in 315 shards (got " + std::to_string(ck.size()) + " in " + std::to_string(ck.shards().size()) + ")");
  std::vector<std::string> warn;
  const auto problems = arch->validate(ck, &warn);
  expect(problems.empty(), "llama: validate() finds no problems (" + std::to_string(problems.size()) + ")");
  for (size_t i = 0; i < problems.size() && i < 5; ++i) std::printf("    %s\n", problems[i].c_str());
  size_t other = 0, attn = 0;
  for (const auto &t : ck.tensors()) {
    const auto r = arch->classify(t.name);
    if (r.comp == m::Component::other) ++other;
    if (r.comp == m::Component::attention) ++attn;
  }
  expect(other == 0 && attn == 52 * 16, "llama: every tensor classified; 52 x 16 attention tensors (" + std::to_string(attn) + ")");
  // weight_shape holds [n, k] (I32 here); the zero points are [n / 8, k / 32]: eight 4-bit zero points per word along n
  // (the word's nibble i = row 8 w + i).
  {
    const auto *s = ck.find("model.layers.0.self_attn.k_proj.weight_shape");
    const auto *z = ck.find("model.layers.0.self_attn.k_proj.weight_zero_point");
    expect(s && z && first_bytes_hex(ck.shards()[s->shard].path, s->offset, 8) == "00040000001c0000" && z->shape[0] == 128 && z->shape[1] == 224,
           "llama: k_proj weight_shape [1024, 7168], zero points [128, 224]");
  }
  m::DeployParams p;
  p.kv_bytes = 2;
  auto b = arch->budget(ck, p);
  expect(b.kv_bytes_per_token == 52 * 2 * 4 * 128 * 2, "llama: KV bf16 = 106,496 B/token/slot/card (" + std::to_string(b.kv_bytes_per_token) + ")");
  double mlp_step = 0;
  for (const auto &r : b.rows)
    if (r.name.rfind("dense MLP", 0) == 0) mlp_step = r.step_bytes;
  expect(mlp_step == 52.0 * 3 * 22016 * 7168 * (0.5 + 2.0 / 32 + 0.5 / 32) / 2, "llama: dense MLP step bytes per card (int4 + fp16 scale + 4-bit zero per 32)");
  const auto plan = arch->dense_plan(ck, p);
  size_t w4 = 0, w4_bad = 0;
  for (const auto &sp : plan) {
    if (sp.store != m::DenseSpec::Store::w4) continue;
    ++w4;
    if (sp.w4_layout != m::DenseSpec::W4Layout::ct_zp || sp.w4_group != 32 || sp.name.find(".weight_packed") == std::string::npos) ++w4_bad;
    for (const auto &a : sp.append)
      if (!ck.find(a)) ++w4_bad;
  }
  expect(w4 == 52 * 4 && w4_bad == 0, "llama: dense plan has 208 w4 matrices (q|k|v, o, gate|up, down per layer), all ct_zp g32");
}

// The dense Qwen3.8-27B (arch qwen35, compressed-tensors int4): config selects the module, the
// checkpoint validates, golden offsets, and the budget arithmetic.
void real_model_qwen35() {
  const char *env = std::getenv("XEN_TEST_MODEL_QWEN35");
  const char *home = std::getenv("HOME");
  const std::string dir = env ? env : std::string(home ? home : "") + "/models/Qwen3.8-27B-W4A16-AutoRound";
  if (!fs::exists(dir + "/config.json")) {
    std::printf("[SKIP] real qwen35 model: %s not present\n", dir.c_str());
    return;
  }
  const auto cfg = m::ModelConfig::load(dir);
  const auto arch = m::open_arch(cfg);
  expect(arch->id() == "qwen35", "qwen35: config.json (model_type qwen3_5) selects arch qwen35");
  const auto ck = m::Checkpoint::open(dir);
  expect(ck.size() == 1999 && ck.shards().size() == 8,
         "qwen35: 1,999 tensors in 8 shards (got " + std::to_string(ck.size()) + " in " + std::to_string(ck.shards().size()) + ")");
  std::vector<std::string> warn;
  const auto problems = arch->validate(ck, &warn);
  expect(problems.empty(), "qwen35: validate() finds no problems (" + std::to_string(problems.size()) + ")");
  for (size_t i = 0; i < problems.size() && i < 5; ++i) std::printf("    %s\n", problems[i].c_str());
  size_t other = 0, mlp = 0;
  for (const auto &t : ck.tensors()) {
    const auto r = arch->classify(t.name);
    if (r.comp == m::Component::other) ++other;
    if (r.comp == m::Component::dense_mlp) ++mlp;
  }
  expect(other == 0 && mlp == 64 * 9, "qwen35: every tensor classified; 64 x 9 dense-MLP tensors (" + std::to_string(mlp) + ")");

  // Golden offsets, computed independently with Python (struct + json over the headers).
  struct Golden {
    const char *name, *file;
    uint64_t offset, bytes;
    const char *hex;
  };
  for (const Golden g : {
           Golden{"lm_head.weight", "model-00007-of-00007.safetensors", 96, 2542796800ull, "153b0f3d3c3cc2bc"},
           Golden{"model.language_model.layers.63.mlp.down_proj.weight_packed", "model-00004-of-00007.safetensors", 2708010088ull, 44564480,
                  "ceee4821c8ecd866"},
           Golden{"model.language_model.layers.3.self_attn.q_proj.weight_scale", "model-00001-of-00007.safetensors", 3174221392ull, 983040,
                  "649d6d1d049c641d"},
           Golden{"mtp.fc.weight", "model_extra_tensors.safetensors", 1608, 104857600, "ea3b9cbc01bcb73c"},
       }) {
    const auto *t = ck.find(g.name);
    const bool ok = t && ck.shards()[t->shard].file == g.file && t->offset == g.offset && t->bytes == g.bytes &&
                    first_bytes_hex(ck.shards()[t->shard].path, t->offset, 8) == g.hex;
    expect(ok, std::string("qwen35: golden offset + first bytes of ") + g.name);
  }
  // weight_shape holds [n, k]: the packed tensor is [n, k / 8], the scales [n, k / 128].
  {
    const auto *s = ck.find("model.language_model.layers.0.linear_attn.in_proj_qkv.weight_shape");
    const auto *w = ck.find("model.language_model.layers.0.linear_attn.in_proj_qkv.weight_packed");
    expect(s && w && first_bytes_hex(ck.shards()[s->shard].path, s->offset, 8) == "0028000000000000" && w->shape[0] == 10240 && w->shape[1] == 640,
           "qwen35: weight_shape [10240, 5120] matches weight_packed [10240, 640] (n rows, k / 8 words)");
  }

  // Budget arithmetic (per card, TP2): KV per token = layers x 2 x (4 / 2 heads) x 256 x bytes.
  m::DeployParams p;
  p.kv_bytes = 2;
  auto b = arch->budget(ck, p);
  expect(b.kv_bytes_per_token == 16 * 2 * 2 * 256 * 2, "qwen35: KV bf16 = 32,768 B/token/slot/card (" + std::to_string(b.kv_bytes_per_token) + ")");
  p.kv_bytes = 1;
  p.drafter = true;
  b = arch->budget(ck, p);
  expect(b.kv_bytes_per_token == 17 * (2 * 2 * 256 + 2 * 2 * 4), "qwen35: KV 8-bit + drafter = 17,680 B/token/slot/card (" + std::to_string(b.kv_bytes_per_token) + ")");
  expect(b.experts_per_card == 0 && b.expert_bytes == 0, "qwen35: no routed experts in the budget");
  double mlp_step = 0;
  for (const auto &r : b.rows)
    if (r.name.rfind("dense MLP", 0) == 0) mlp_step = r.step_bytes;
  // 64 layers x (2 x 17408 x 5120 + 5120 x 17408) x (0.5 + 2 / 128) B / 2 cards
  expect(mlp_step == 64.0 * 3 * 17408 * 5120 * (0.5 + 2.0 / 128) / 2, "qwen35: dense MLP step bytes = 4,411,883,520 per card");
  // The dense plan names only tensors that exist and marks every pack-quantized Linear w4.
  const auto plan = arch->dense_plan(ck, p);
  size_t w4 = 0, w4_bad = 0;
  for (const auto &sp : plan) {
    if (sp.store != m::DenseSpec::Store::w4) continue;
    ++w4;
    const std::string mod = sp.name.substr(0, sp.name.size() - std::strlen(".weight_packed"));
    if (sp.name.find(".weight_packed") == std::string::npos || !ck.find(mod + ".weight_scale")) ++w4_bad;
    for (const auto &a : sp.append)
      if (a.find(".weight_packed") == std::string::npos || !ck.find(a.substr(0, a.size() - std::strlen(".weight_packed")) + ".weight_scale")) ++w4_bad;
  }
  expect(w4 == 256 && w4_bad == 0, "qwen35: dense plan has 256 w4 matrices, each with its weight_scale (" + std::to_string(w4) + ", bad " + std::to_string(w4_bad) + ")");

  // The format bridge on a real matrix (layer 0 in_proj_z, [6144 x 5120]): the GPTQ-ordered words
  // hold the same codes element for element, the scales are finite and positive, the words then
  // pack into the engine's W4 blocks, and the codes look like a symmetric min-max quantiser's
  // (every (row, group) stays within |q - 8| <= 8; most reach 7 or 8). Scales may be negative.
  {
    namespace kk = xenithium::kernels;
    const std::string mod = "model.language_model.layers.0.linear_attn.in_proj_z";
    const auto &tp = ck.at(mod + ".weight_packed"), &ts = ck.at(mod + ".weight_scale");
    const auto buf = m::bulk_read(ck, {&tp, &ts});
    const int n = static_cast<int>(tp.shape[0]), k = static_cast<int>(tp.shape[1]) * 8;
    const uint32_t *packed = buf.as<uint32_t>(tp);
    const uint16_t *scales = buf.as<uint16_t>(ts);
    std::vector<uint32_t> gptq(static_cast<size_t>(k / 8) * n);
    std::vector<uint16_t> gs(static_cast<size_t>(k / 128) * n);
    kk::ct_to_gptq_qweight(packed, n, k, gptq.data());
    kk::ct_to_gptq_scales(scales, n, k, gs.data());
    size_t mism = 0, at_edge = 0, groups = 0, bad_scale = 0, neg_scale = 0;
    for (int row = 0; row < n; row += 7) {
      for (int g = 0; g < k / 128; ++g) {
        int amax = 0;
        for (int kg = 0; kg < 128; ++kg) {
          const int kk_ = g * 128 + kg;
          const int q = kk::ct_code(packed, k, row, kk_);
          const int q2 = static_cast<int>((gptq[static_cast<size_t>(kk_ / 8) * n + row] >> (4 * (kk_ % 8))) & 15u);
          if (q != q2) ++mism;
          amax = std::max(amax, std::abs(q - kk::kW4Zero));
        }
        ++groups;
        if (amax >= 7) ++at_edge;
        const uint16_t s = gs[static_cast<size_t>(g) * n + row];
        const int exp = (s >> 10) & 31;
        // AutoRound's symmetric scale carries the sign of the group's larger extreme (so that it maps
        // to code 0 = -8 x |scale|): about half the scales are negative, and w = (q - 8) x scale holds.
        if (s != scales[static_cast<size_t>(row) * (k / 128) + g] || exp == 31 || (s & 0x7fff) == 0) ++bad_scale;
        if (s & 0x8000) ++neg_scale;
      }
    }
    expect(mism == 0, "qwen35: ct_to_gptq_qweight keeps every code (" + std::to_string(mism) + " mismatches over " + std::to_string(groups * 128) + ")");
    expect(bad_scale == 0, "qwen35: scales transposed exactly, all finite and non-zero (" + std::to_string(bad_scale) + " bad)");
    std::printf("    qwen35 in_proj_z scales: %.1f %% negative (AutoRound's signed symmetric scale; the sign rides into the GEMM)\n",
                100.0 * static_cast<double>(neg_scale) / static_cast<double>(groups));
    std::vector<uint8_t> blocks(static_cast<size_t>(n) * k / 2);
    kk::pack_w4_gptq(gptq.data(), n, k, blocks.data());
    // Block (g, t) byte (kg/2)*16 + lane holds column t*16+lane at kg (low) and kg+1 (high).
    size_t bmism = 0;
    for (int col = 0; col < n; col += 101)
      for (int kk_ = 0; kk_ < k; kk_ += 3) {
        const int g = kk_ / 128, kg = kk_ % 128, t = col / 16, lane = col % 16;
        const uint8_t byte = blocks[(static_cast<size_t>(g) * (n / 16) + t) * 1024 + (kg / 2) * 16 + lane];
        const int q = (kg & 1) ? (byte >> 4) : (byte & 15);
        if (q != kk::ct_code(packed, k, col, kk_)) ++bmism;
      }
    expect(bmism == 0, "qwen35: the bridged words pack into the engine's W4 blocks with the right codes (" + std::to_string(bmism) + " mismatches)");
    // The direct packer (what DenseStore runs, in row chunks) == the bridge + pack_w4_gptq, byte for byte.
    std::vector<uint8_t> direct(blocks.size(), 0xA5);
    for (int r0 = 0; r0 < n; r0 += 2048) kk::pack_w4_ct_rows(packed, n, k, r0, std::min(n, r0 + 2048), direct.data());
    expect(direct == blocks, "qwen35: pack_w4_ct_rows (chunked) == pack_w4_gptq of the bridged words, byte for byte");
    const double edge = static_cast<double>(at_edge) / static_cast<double>(groups);
    std::printf("    qwen35 in_proj_z codes: %.1f %% of (row, group) reach |q - 8| >= 7 (symmetric min-max expects most)\n", 100 * edge);
    expect(edge > 0.5, "qwen35: the code distribution is a symmetric group quantiser's");
  }
}

// The same architecture in AutoRound's GPTQ orientation (`auto_round:auto_gptq`: qweight [k/8][n], scales [k/128][n],
// qzeros [k/128][n/8]): the Swift-1.5 finetune's checkpoint (`~/models/swift15-w4a16-autoround`) when present.
// Config selects the module, validate passes, the plan marks the gptq layout, the budget is the
// ct checkpoint's, the zeros are the symmetric 0x77777777 words, and the chunked packer equals pack_w4_gptq.
void real_model_qwen35_gptq() {
  const char *env = std::getenv("XEN_TEST_MODEL_SWIFT15");
  const char *home = std::getenv("HOME");
  const std::string dir = env ? env : std::string(home ? home : "") + "/models/swift15-w4a16-autoround";
  if (!fs::exists(dir + "/config.json")) {
    std::printf("[SKIP] real qwen35 gptq model: %s not present\n", dir.c_str());
    return;
  }
  namespace kk = xenithium::kernels;
  const auto cfg = m::ModelConfig::load(dir);
  const auto arch = m::open_arch(cfg);
  expect(arch->id() == "qwen35", "qwen35 gptq: config.json (model_type qwen3_5, quant_method auto-round) selects arch qwen35");
  const auto ck = m::Checkpoint::open(dir);
  expect(ck.size() == 1999 && ck.shards().size() == 6,
         "qwen35 gptq: 1,999 tensors in 6 shards (got " + std::to_string(ck.size()) + " in " + std::to_string(ck.shards().size()) + ")");
  std::vector<std::string> warn;
  const auto problems = arch->validate(ck, &warn);
  expect(problems.empty(), "qwen35 gptq: validate() finds no problems (" + std::to_string(problems.size()) + ")");
  for (size_t i = 0; i < problems.size() && i < 5; ++i) std::printf("    %s\n", problems[i].c_str());
  size_t other = 0, mlp = 0;
  for (const auto &t : ck.tensors()) {
    const auto r = arch->classify(t.name);
    if (r.comp == m::Component::other) ++other;
    if (r.comp == m::Component::dense_mlp) ++mlp;
  }
  expect(other == 0 && mlp == 64 * 9, "qwen35 gptq: every tensor classified; 64 x 9 dense-MLP tensors (qweight, scales, qzeros x 3) (" + std::to_string(mlp) + ")");
  m::DeployParams p;
  const auto b = arch->budget(ck, p);
  double mlp_step = 0;
  for (const auto &r : b.rows)
    if (r.name.rfind("dense MLP", 0) == 0) mlp_step = r.step_bytes;
  expect(mlp_step == 64.0 * 3 * 17408 * 5120 * (0.5 + 2.0 / 128) / 2, "qwen35 gptq: dense MLP step bytes = 4,411,883,520 per card (as the ct checkpoint)");
  const auto plan = arch->dense_plan(ck, p);
  size_t w4 = 0, w4_bad = 0;
  for (const auto &sp : plan) {
    if (sp.store != m::DenseSpec::Store::w4) continue;
    ++w4;
    auto ok = [&](const std::string &nm) {
      if (nm.find(".qweight") == std::string::npos) return false;
      const std::string mod = nm.substr(0, nm.size() - std::strlen(".qweight"));
      return ck.find(mod + ".scales") && ck.find(mod + ".qzeros");
    };
    if (sp.w4_layout != m::DenseSpec::W4Layout::gptq || !ok(sp.name)) ++w4_bad;
    for (const auto &a : sp.append)
      if (!ok(a)) ++w4_bad;
  }
  expect(w4 == 256 && w4_bad == 0, "qwen35 gptq: dense plan has 256 w4 matrices in the gptq layout, each with scales and qzeros (" + std::to_string(w4) + ", bad " +
                                        std::to_string(w4_bad) + ")");
  // Layer 0 in_proj_z [6144 x 5120]: zeros symmetric, scales finite and non-zero (about half negative), the chunked
  // packer == pack_w4_gptq byte for byte, codes a symmetric min-max quantiser's.
  {
    const std::string mod = "model.language_model.layers.0.linear_attn.in_proj_z";
    const auto &tq = ck.at(mod + ".qweight"), &ts = ck.at(mod + ".scales"), &tz = ck.at(mod + ".qzeros");
    const auto buf = m::bulk_read(ck, {&tq, &ts, &tz});
    const int k = static_cast<int>(tq.shape[0]) * 8, n = static_cast<int>(tq.shape[1]);
    const uint32_t *qw = buf.as<uint32_t>(tq), *qz = buf.as<uint32_t>(tz);
    const uint16_t *sc = buf.as<uint16_t>(ts);
    size_t bad_zero = 0, bad_scale = 0, neg = 0, at_edge = 0, groups = 0;
    for (size_t i = 0; i < static_cast<size_t>(tz.numel()); ++i) bad_zero += qz[i] != 0x77777777u;
    for (int col = 0; col < n; col += 7)
      for (int g = 0; g < k / 128; ++g) {
        const uint16_t s = sc[static_cast<size_t>(g) * n + col];
        if (((s >> 10) & 31) == 31 || (s & 0x7fff) == 0) ++bad_scale;
        neg += (s & 0x8000) != 0;
        int amax = 0;
        for (int kg = 0; kg < 128; ++kg) {
          const int kk_ = g * 128 + kg;
          amax = std::max(amax, std::abs(static_cast<int>((qw[static_cast<size_t>(kk_ / 8) * n + col] >> (4 * (kk_ % 8))) & 15u) - kk::kW4Zero));
        }
        ++groups;
        at_edge += amax >= 7;
      }
    expect(bad_zero == 0, "qwen35 gptq: every qzeros word is the symmetric 0x77777777 (" + std::to_string(bad_zero) + " bad of " + std::to_string(tz.numel()) + ")");
    expect(bad_scale == 0, "qwen35 gptq: scales finite and non-zero (" + std::to_string(bad_scale) + " bad)");
    std::printf("    qwen35 gptq in_proj_z scales: %.1f %% negative; %.1f %% of (col, group) reach |q - 8| >= 7\n", 100.0 * neg / groups, 100.0 * at_edge / groups);
    expect(at_edge > groups / 2, "qwen35 gptq: the code distribution is a symmetric group quantiser's");
    std::vector<uint8_t> whole(static_cast<size_t>(n) * k / 2), chunked(whole.size(), 0xA5);
    kk::pack_w4_gptq(qw, n, k, whole.data());
    for (int r0 = 0; r0 < n; r0 += 2048) kk::pack_w4_gptq_rows(qw, n, k, r0, std::min(n, r0 + 2048), chunked.data());
    expect(chunked == whole, "qwen35 gptq: pack_w4_gptq_rows (chunked) == pack_w4_gptq, byte for byte");
  }
}

// the drafter's placement. slice_row_ranges (the draft vocabulary rows) and, on the real
// checkpoint, mtp_dense_plan: every mtp tensor placed once, the fc split along k, the head subset rows.
// the weights a sequence measures for itself (SelectShare, select_curve_add, select_key_score)
void select_share() {
  namespace q35 = xenithium::arch::qwen35;
  constexpr int F = q35::SelectShare::kFractions, kUnits = 8;
  auto curves = [&](auto lost_at) {  // lost_at(unit, fraction)
    std::vector<double> lost(static_cast<size_t>(kUnits) * F);
    for (int u = 0; u < kUnits; ++u)
      for (int f = 0; f < F; ++f) lost[static_cast<size_t>(u) * F + f] = lost_at(u, q35::SelectShare::fraction(f));
    return lost;
  };
  auto mean_of = [](const std::vector<float> &w) {
    double m = 0;
    for (const float v : w) m += v / static_cast<double>(w.size());
    return m;
  };
  const auto same = q35::SelectShare::weights(curves([](int, double f) { return 1.0 - f; }), kUnits);
  expect(std::all_of(same.begin(), same.end(), [&](float v) { return v > 0.25f && v < 2.5f; }) && std::abs(mean_of(same) - 1.0) < 1e-6,
         "page share: equal curves: weights of mean 1");
  // unit 0 is spread (its loss falls all the way), the others are sharp (nothing left to remove past 5 % of the pages)
  const auto mixed = q35::SelectShare::weights(curves([](int u, double f) { return u == 0 ? 1.0 - f : std::max(0.0, 0.05 - f); }), kUnits);
  const float sharp = *std::max_element(mixed.begin() + 1, mixed.end());
  expect(mixed[0] > 1.5f && sharp < 1.0f && std::abs(mean_of(mixed) - 1.0) < 1e-6, "page share: the spread unit gets more pages than the sharp ones (" + std::to_string(mixed[0]) +
                                                                                 " against at most " + std::to_string(sharp) + "), mean 1");
  expect(mixed == q35::SelectShare::weights(curves([](int u, double f) { return u == 0 ? 1.0 - f : std::max(0.0, 0.05 - f); }), kUnits), "page share: the same curves, the same weights");
  bool thrown = false;
  try {
    (void)q35::SelectShare::weights(std::vector<double>(7), kUnits);
  } catch (const std::invalid_argument &) {
    thrown = true;
  }
  expect(thrown, "page share: a curve of the wrong length is refused");
  // the counts the weights give at a common count: the pages past the floor are shared, the sum stays
  int sum = 0, fewest = 1 << 30;
  for (const float v : mixed) sum += q35::select_head_pages(v, 512), fewest = std::min(fewest, q35::select_head_pages(v, 512));
  expect(std::abs(sum - 512 * kUnits) <= kUnits && fewest >= q35::kSelectFloor && q35::select_head_pages(mixed[0], 128) == 128 && q35::select_head_pages(2.f, 96) == 96,
         "page share: at 512 pages the heads' counts sum to the common count x heads and none is under the floor; up to the floor every head keeps the common count");

  auto order_key = [](float v) {  // gqa_sel.cpp order_keys
    const uint32_t bits = std::bit_cast<uint32_t>(v);
    return (bits & 0x80000000u) != 0u ? ~bits : bits | 0x80000000u;
  };
  bool round_trip = true, ordered = true;
  const float values[] = {-1e30f, -37.5f, -1e-20f, 0.f, 1e-20f, 0.25f, 812.f, 1e30f};
  for (size_t i = 0; i < std::size(values); ++i) {
    round_trip = round_trip && q35::select_key_score(order_key(values[i])) == values[i];
    ordered = ordered && (i == 0 || order_key(values[i]) > order_key(values[i - 1]));
  }
  expect(round_trip && ordered, "page share: a score comes back from its order key, and the keys order as the scores");

  // 40 full pages of equal score: the mass is spread evenly, so keeping n pages loses 1 - n / 40
  constexpr int kPages = 40, kSink = 1, kLocal = 16;
  std::vector<double> lost(F, 0.0);
  q35::select_curve_add(std::vector<float>(kPages, 3.f), static_cast<int64_t>(kPages) * 64, 64, kSink, kLocal, 1.f / 16, lost);
  bool even = true;
  for (int f = 0; f < F; ++f)
    even = even && std::abs(lost[static_cast<size_t>(f)] - (1.0 - q35::SelectShare::count(kPages, q35::SelectShare::fraction(f), kSink + kLocal) / static_cast<double>(kPages))) < 1e-6;
  expect(even, "page curve: equal scores lose the dropped pages' share of the tokens at every fraction");
  // one page far above the rest holds nearly all the mass: it is the first page the score keeps
  std::vector<float> peaked(kPages, 0.f);
  peaked[7] = 400.f;
  std::vector<double> lp(F, 0.0);
  q35::select_curve_add(peaked, static_cast<int64_t>(kPages - 1) * 64 + 10, 64, kSink, kLocal, 1.f / 16, lp);
  expect(lp[0] < 1e-6 && std::is_sorted(lp.rbegin(), lp.rend()), "page curve: a page that holds the mass is kept first; the curve never rises");
}


void mtp35_plan() {
  using R = std::vector<std::pair<int64_t, int64_t>>;
  const R ranges{{0, 98304}, {248000, 248128}};
  const auto c0 = m::slice_row_ranges(248320, ranges, 0, 2), c1 = m::slice_row_ranges(248320, ranges, 1, 2);
  expect(c0 == R{{0, 49216}} && c1 == R{{49216, 98304}, {248000, 248128}}, "slice_row_ranges: [0, 98304) + 128 special rows -> 49,216 rows per card in order");
  expect(m::slice_row_ranges(10, {{2, 4}, {6, 10}}, 2, 3) == R{{8, 10}}, "slice_row_ranges: 3 cards, the last third of two ranges");
  bool threw = false;
  try {
    m::slice_row_ranges(10, {{0, 3}}, 0, 2);
  } catch (const std::exception &) {
    threw = true;
  }
  expect(threw, "slice_row_ranges: an odd total over 2 cards throws");

  const char *env = std::getenv("XEN_TEST_MODEL_SWIFT15");
  const char *home = std::getenv("HOME");
  const std::string dir = env ? env : std::string(home ? home : "") + "/models/swift15-w4a16-autoround";
  if (!fs::exists(dir + "/config.json")) {
    std::printf("[SKIP] qwen35 mtp plan: %s not present\n", dir.c_str());
    return;
  }
  const auto ck = m::Checkpoint::open(dir);
  size_t mtp = 0;
  for (const auto &t : ck.tensors()) mtp += t.name.rfind("mtp.", 0) == 0;
  for (int bytes : {1, 2}) {
    const auto plan = xenithium::arch::qwen35::mtp_dense_plan(ck, bytes, ranges);
    size_t placed = 0, fc_cols = 0, head = 0, bad_store = 0;
    for (const auto &sp : plan) {
      if (sp.key == xenithium::arch::qwen35::kMtpHeadKey) {
        head += sp.name == "lm_head.weight" && sp.split == m::DenseSpec::Split::rows && sp.row_ranges == ranges;
        continue;
      }
      placed += 1 + sp.append.size();
      if (sp.name.find("mtp.fc.") != std::string::npos) fc_cols += sp.split == m::DenseSpec::Split::cols;
      const bool mat = ck.at(sp.name).shape.size() == 2;
      if (mat != (sp.store != m::DenseSpec::Store::raw) || (mat && sp.store != (bytes == 1 ? m::DenseSpec::Store::i8 : m::DenseSpec::Store::bf16))) ++bad_store;
    }
    expect(placed == mtp, "mtp35 plan (" + std::to_string(bytes) + " B): every mtp.* tensor placed once (" + std::to_string(placed) + " of " + std::to_string(mtp) + ")");
    expect(fc_cols == 1 && head == 1, "mtp35 plan: fc split along k (embedding half | hidden half); the draft head = lm_head's subset rows");
    expect(bad_store == 0, "mtp35 plan: matrices int8 per row at 1 byte, bf16 at 2; norms raw (" + std::to_string(bad_store) + " off)");
  }
}

// the shared KV pool's host table (model/kv_pages.hpp)
void kv_pages() {
  {  // fixed layout: the identity table, every page owned from the start, release keeps them
    m::KvPages f(3, 4, 12, false);
    bool id = true;
    for (int i = 0; i < 12; ++i) id &= f.table()[static_cast<size_t>(i)] == i;
    expect(id && !f.shared() && f.held_pages(2) == 4 && f.reserve(1, 256) && !f.reserve(1, 257), "kv pages: fixed layout = identity table, reserve up to the capacity");
    f.release(1);
    expect(f.held_pages(1) == 4 && f.table()[4] == 4, "kv pages: fixed layout keeps its pages on release");
  }
  {  // shared pool: lazy pages, all or nothing, lowest first, reuse after release, dirty ranges
    m::KvPages p(4, 8, 10, true);
    expect(p.shared() && p.free_tokens() == 640 && p.used_tokens() == 0, "kv pages: pool of 10 pages = 640 tokens, none held");
    p.clear_dirty();
    expect(p.reserve(0, 1) && p.held_pages(0) == 1 && p.table()[0] == 0 && p.free_tokens() == 576, "kv pages: 1 token takes 1 page (page 0 first)");
    expect(p.dirty() && p.dirty_lo() == 0 && p.dirty_hi() == 1, "kv pages: the new entry is the dirty range");
    p.clear_dirty();
    expect(p.reserve(0, 64) && p.held_pages(0) == 1 && !p.dirty(), "kv pages: positions inside held pages take nothing");
    expect(p.reserve(1, 300) && p.held_pages(1) == 5 && p.table()[8] == 1 && p.table()[12] == 5, "kv pages: 300 tokens = 5 pages, handed out lowest first");
    expect(p.dirty_lo() == 8 && p.dirty_hi() == 13, "kv pages: dirty range = slot 1's new entries");
    const auto before = p.table();
    expect(!p.reserve(2, 64 * 5) && p.held_pages(2) == 0 && p.table() == before, "kv pages: not enough free pages -> false, nothing changed");
    expect(!p.reserve(3, 8 * 64 + 1), "kv pages: past the slot's capacity -> false");
    expect(p.reserve(2, 64 * 4) && p.free_tokens() == 0 && p.used_tokens() == 640, "kv pages: the last 4 pages");
    p.clear_dirty();
    p.release(1);
    expect(p.held_pages(1) == 0 && p.free_tokens() == 320 && p.table()[8] == 0 && p.dirty_lo() == 8 && p.dirty_hi() == 13,
           "kv pages: release returns the pages, zeroes the entries (marked dirty)");
    expect(p.reserve(3, 64) && p.table()[24] == 1, "kv pages: released pages are reused, lowest first");
    bool distinct = true;  // every held entry is a distinct page
    std::vector<int> seen(10, 0);
    for (int s = 0; s < 4; ++s)
      for (int i = 0; i < p.held_pages(s); ++i) distinct &= seen[static_cast<size_t>(p.table()[static_cast<size_t>(s * 8 + i)])]++ == 0;
    expect(distinct, "kv pages: held pages are distinct");
    bool threw = false;
    try {
      m::KvPages bad(4, 8, 7, true);
    } catch (const std::exception &) {
      threw = true;
    }
    expect(threw, "kv pages: a pool smaller than one slot's capacity throws");
  }
  {  // shared pages with reference counts, copy-on-write
    m::KvPages p(3, 8, 12, true);
    expect(p.reserve(0, 64 * 4) && p.reserve(1, 64 * 6), "kv share: slot 0 holds 4 pages, slot 1 6");
    const int32_t s0 = p.page(0, 0), s3 = p.page(0, 3);
    expect(p.free_tokens() == 64 * 2 && p.share(1, 0, 3) && p.page(1, 0) == s0 && p.page(1, 2) == p.page(0, 2) && p.held_pages(1) == 6,
           "kv share: slot 1's first 3 pages become slot 0's, its pages past 3 kept");
    expect(p.free_tokens() == 64 * 5 && p.used_tokens() == 64 * 7 && p.shared_pages(1) == 3 && p.shared_pages(0) == 3 && !p.page_shared(0, 3),
           "kv share: slot 1's own 3 pages back on the free list; 3 pages shared, slot 0's 4th private");
    expect(!p.share(1, 0, 5) && !p.share(0, 0, 1), "kv share: past the source's pages or onto itself -> false");
    p.release(0);
    expect(p.free_tokens() == 64 * 6 && p.shared_pages(1) == 0 && p.page(1, 0) == s0, "kv share: the source lets go: slot 1 keeps the pages (now its own), only slot 0's 4th freed");
    expect(p.reserve(2, 64 * 6), "kv share: the freed pages are handed out again");
    bool distinct = true;  // no page held twice (a lost reference count would hand slot 1's pages to slot 2)
    std::vector<int> seen(12, 0);
    for (int s = 0; s < 3; ++s)
      for (int i = 0; i < p.held_pages(s); ++i) distinct &= seen[static_cast<size_t>(p.page(s, i))]++ == 0;
    expect(distinct && s3 >= 0, "kv share: after release every held page is distinct");
    p.release(2);
    expect(p.reserve(0, 64 * 2) && p.share(0, 1, 2), "kv share: slot 0 shares slot 1's first 2 pages");
    const int32_t old1 = p.page(0, 1);
    const auto mv = p.unshare(0, 1);
    expect(mv.size() == 1 && mv[0].first == old1 && mv[0].second == p.page(0, 1) && p.page(1, 1) == old1 && !p.page_shared(0, 1) && p.page_shared(0, 0),
           "kv share: unshare from page 1 gives slot 0 a private copy of page 1 (the pair to copy), page 0 still shared");
    expect(p.unshare(0, 1).empty(), "kv share: nothing left to unshare there");
    p.trim(1, 64);
    expect(p.held_pages(1) == 1 && p.page_shared(0, 0), "kv share: trim lets go of slot 1's pages past 1 (page 0 still shared)");
    p.release(0);
    p.release(1);
    expect(p.free_tokens() == 64 * 12 && p.used_tokens() == 0, "kv share: every page back after both release");
  }
  {  // scramble: another order, the same accounting
    m::KvPages a(2, 16, 20, true), b(2, 16, 20, true, true);
    a.reserve(0, 16 * 64);
    b.reserve(0, 16 * 64);
    std::vector<int32_t> ra(a.table().begin(), a.table().begin() + 16), rb(b.table().begin(), b.table().begin() + 16);
    std::vector<int32_t> sa = ra, sb = rb;
    std::sort(sb.begin(), sb.end());
    expect(ra != rb && std::adjacent_find(sb.begin(), sb.end()) == sb.end() && sb.back() < 20 && a.free_tokens() == b.free_tokens(),
           "kv pages: scramble hands out other (distinct, in-pool) pages, same counts");
  }
  {  // the lend limit
    m::KvPages p(2, 16, 32, true), sc(2, 16, 32, true, true);
    expect(p.limit() == 32 && p.free_below_limit() == 32 && !sc.set_limit(8), "kv lend limit: none by default; a scrambled pool takes none");
    expect(p.set_limit(8) && p.reserve(0, 8 * 64) && p.free_below_limit() == 0 && !p.reserve(1, 1) && p.pages_short(1, 1) == 1,
           "kv lend limit: reserve hands out pages below the limit only");
    expect(p.set_limit(12) && p.reserve(1, 64) && p.page(1, 0) == 8 && p.high_water() == 9 && !p.set_limit(8),
           "kv lend limit: a raise frees pages, a lower stops at the high water");
    p.release(1);
    expect(p.high_water() == 8 && p.set_limit(8) && p.free_below_limit() == 0 && p.free_tokens() == 24 * 64, "kv lend limit: lowered to the high water");
    p.trim(0, 3 * 64);
    expect(p.high_water() == 3 && p.free_below_limit() == 5 && p.pages_short(0, 5 * 64) == 2 && p.reserve(0, 8 * 64), "kv lend limit: trim frees the top");
    p.share(1, 0, 4);
    expect(p.set_limit(8) && !p.reserve(1, 9 * 64), "kv lend limit: shared pages count once");
    bool threw = false;
    try {
      p.unshare(1, 0);
    } catch (const std::runtime_error &) {
      threw = true;
    }
    expect(threw, "kv lend limit: a copy-on-write takes pages below the limit only");
  }
}

// the lend slots (model/kv_lend.hpp) against a brute-force overlap check at every limit
void kv_lend() {
  const std::vector<size_t> pb = {1000, 64, 0, 4096};
  const int PP = 100;
  const m::KvLend kl(pb, PP, 3000, 256, 1000);
  bool sorted = true, inside = true, apart = true, prefix = true, aligned = true;
  for (size_t i = 0; i < kl.size(); ++i) {
    const auto &a = kl.slots()[i];
    if (i > 0) sorted &= kl.slots()[i - 1].min_page >= a.min_page;
    inside &= a.off + 3000 <= PP * pb[static_cast<size_t>(a.buf)] && a.min_page >= 1;
    aligned &= a.off % 256 == 0;
    for (size_t j = i + 1; j < kl.size(); ++j) {
      const auto &b = kl.slots()[j];
      if (a.buf == b.buf) apart &= a.off + 3000 <= b.off || b.off + 3000 <= a.off;
    }
  }
  for (int W = 1; W <= PP; ++W) {
    const int u = kl.usable(W);
    for (size_t i = 0; i < kl.size(); ++i) {
      const auto &a = kl.slots()[i];
      const bool free = static_cast<size_t>(W) * pb[static_cast<size_t>(a.buf)] <= a.off;  // KV pages [0, W) end before it
      prefix &= free == (static_cast<int>(i) < u);
    }
    if (u > 0) prefix &= kl.limit_for(u, PP) >= W && kl.usable(kl.limit_for(u, PP)) >= u;
  }
  int per[4] = {0, 0, 0, 0};
  for (const auto &a : kl.slots()) ++per[a.buf];
  expect(sorted && inside && apart && aligned && per[2] == 0 && per[0] == 32 && per[3] > 100,
         "kv lend: slots sorted by min page, inside their buffer, disjoint, aligned, none in an empty buffer (" + std::to_string(kl.size()) + ")");
  expect(prefix && kl.usable(PP) == 0 && kl.limit_for(0, PP) == PP, "kv lend: usable(W) = exactly the slots no KV page below W touches");
  const m::KvLend capped(pb, PP, 3000, 256, 10);
  expect(capped.size() == 10 && capped.slots()[9].min_page == kl.slots()[9].min_page, "kv lend: a cap keeps the slots with the highest min pages");
}

// regions of mixed sizes (KvLendRegions) against a brute-force check at every limit
void kv_lend_regions() {
  const std::vector<size_t> pb = {1000, 64, 0, 4096};
  const int PP = 100;
  std::vector<size_t> sizes;
  for (int i = 0; i < 40; ++i) sizes.push_back(i % 3 == 0 ? 44000 : i % 3 == 1 ? 21000 : 18300);
  sizes.push_back(500000);  // larger than any buffer: no region
  const m::KvLendRegions kr(pb, PP, sizes, 256);
  bool inside = true, apart = true, aligned = true, fits = true, exact = true;
  int placed = 0;
  for (size_t i = 0; i < kr.size(); ++i) {
    const auto &a = kr.regions()[i];
    if (a.buf < 0) continue;
    ++placed;
    fits &= a.bytes >= sizes[i] && a.bytes < sizes[i] + 256;
    inside &= a.off + a.bytes <= PP * pb[static_cast<size_t>(a.buf)] && a.min_page >= 1;
    aligned &= a.off % 256 == 0;
    for (size_t j = i + 1; j < kr.size(); ++j) {
      const auto &b = kr.regions()[j];
      if (a.buf == b.buf) apart &= a.off + a.bytes <= b.off || b.off + b.bytes <= a.off;
    }
  }
  for (int W = 1; W <= PP; ++W) {
    int n = 0, lo = PP;
    for (size_t i = 0; i < kr.size(); ++i) {
      const auto &a = kr.regions()[i];
      const bool free = a.buf >= 0 && static_cast<size_t>(W) * pb[static_cast<size_t>(a.buf)] <= a.off;  // KV pages [0, W) end before it
      exact &= free == kr.usable(i, W);
      if (free) ++n, lo = std::min(lo, a.min_page);
    }
    exact &= n == kr.lent(W) && lo == kr.limit_for(W, PP) && kr.lent(kr.limit_for(W, PP)) == n;
  }
  expect(inside && apart && aligned && fits && kr.regions().back().buf < 0 && kr.regions()[0].buf == 3 && placed > 8 && placed < 41,
         "kv lend regions: each inside its buffer, disjoint, aligned, the size asked; an item too large gets none (" + std::to_string(placed) + " of 41 placed)");
  expect(exact && kr.lent(PP) == 0 && kr.limit_for(PP, PP) == PP, "kv lend regions: usable(i, W) = exactly the regions no KV page below W touches; limit_for keeps them");
}

// the host logits copy's vectorised rounding == bf16_value(bf16_bits(v)) bit for bit (ties both
// ways, NaN payloads, infinities, subnormals, the largest finite values that round to inf).
void bf16_round() {
  std::vector<uint32_t> bits = {0x00000000u, 0x80000000u, 0x7f800000u, 0xff800000u, 0x7fc00000u, 0x7f800001u, 0xffbfffffu,
                                0x7f7fffffu, 0xff7fffffu, 0x7f7f8000u, 0x7f7f7fffu, 0x00008000u, 0x00018000u, 0x00007fffu,
                                0x3f808000u, 0x3f818000u, 0x3f807fffu, 0x3f80ffffu, 0x00000001u, 0x807fffffu};
  uint64_t st = 0x9e3779b97f4a7c15ull;
  for (int i = 0; i < 200000; ++i) {
    st ^= st << 13, st ^= st >> 7, st ^= st << 17;
    bits.push_back(static_cast<uint32_t>(st));
  }
  std::vector<float> src(bits.size()), dst(bits.size());
  std::memcpy(src.data(), bits.data(), bits.size() * 4);
  xenithium::kernels::bf16_round_floats(src.data(), dst.data(), src.size());
  size_t bad = 0;
  for (size_t i = 0; i < src.size(); ++i) {
    const float ref = xenithium::kernels::bf16_value(xenithium::kernels::bf16_bits(src[i]));
    bad += std::memcmp(&ref, &dst[i], 4) != 0;
  }
  expect(bad == 0, "bf16_round_floats == bf16_value(bf16_bits) on " + std::to_string(src.size()) + " values incl. ties, NaN, inf, subnormals (" +
                       std::to_string(bad) + " differ)");
}

// Engine: kernels:quantize_w4_row, the load-time int4 RTN of DenseSpec:Store:w4r. On random bf16 rows (plus
// an all-zero group and a group whose extreme is positive / negative): each code is clamp(rint(w / s), -8, 7) + 8 at the
// compressed-tensors position ct_code reads, s = fp16(-extreme / 8) (the extreme lands on code -8 exactly), a zero
// group gets scale 0 and code 8, and the MSE search never does worse than plain RTN on a group.
void w4_rtn() {
  namespace kk = xenithium::kernels;
  const int n = 37, k = 512, G = k / 128;
  std::vector<uint16_t> w(static_cast<size_t>(n) * k);
  uint64_t st = 0x2545f4914f6cdd1dull;
  for (auto &b : w) {
    st ^= st << 13, st ^= st >> 7, st ^= st << 17;
    const float v = (static_cast<float>(st % 20001) - 10000.f) * 1e-6f * ((st >> 20) % 7 == 0 ? 6.f : 1.f);
    b = kk::bf16_bits(v);
  }
  for (int i = 0; i < 128; ++i) w[static_cast<size_t>(3) * k + 128 + i] = 0;  // row 3 group 1: all zero
  std::vector<uint32_t> words(static_cast<size_t>(n) * k / 8), words2(words.size());
  std::vector<uint16_t> sc(static_cast<size_t>(n) * G), sc2(sc.size());
  for (int r = 0; r < n; ++r) {
    kk::quantize_w4_row(&w[static_cast<size_t>(r) * k], k, &words[static_cast<size_t>(r) * (k / 8)], &sc[static_cast<size_t>(r) * G], false);
    kk::quantize_w4_row(&w[static_cast<size_t>(r) * k], k, &words2[static_cast<size_t>(r) * (k / 8)], &sc2[static_cast<size_t>(r) * G], true);
  }
  size_t bad_code = 0, bad_scale = 0, bad_zero = 0, mse_worse = 0, edge_miss = 0, pos = 0, neg = 0;
  for (int r = 0; r < n; ++r)
    for (int g = 0; g < G; ++g) {
      float ext = 0;
      for (int i = 0; i < 128; ++i) {
        const float v = kk::bf16_value(w[static_cast<size_t>(r) * k + g * 128 + i]);
        if (std::fabs(v) > std::fabs(ext)) ext = v;
      }
      const uint16_t sb = sc[static_cast<size_t>(r) * G + g];
      _Float16 h;
      std::memcpy(&h, &sb, 2);
      const float s = static_cast<float>(h);
      if (ext == 0.f) {
        bad_zero += s != 0.f;
        for (int i = 0; i < 128; ++i) bad_zero += kk::ct_code(words.data(), k, r, g * 128 + i) != 8;
        continue;
      }
      const _Float16 want = static_cast<_Float16>(-ext / 8.f);
      uint16_t wb;
      std::memcpy(&wb, &want, 2);
      bad_scale += wb != sb;
      (ext > 0 ? pos : neg)++;
      double e1 = 0, e2 = 0;
      const uint16_t sb2 = sc2[static_cast<size_t>(r) * G + g];
      _Float16 h2;
      std::memcpy(&h2, &sb2, 2);
      const float s2 = static_cast<float>(h2);
      for (int i = 0; i < 128; ++i) {
        const float v = kk::bf16_value(w[static_cast<size_t>(r) * k + g * 128 + i]);
        const int q = kk::ct_code(words.data(), k, r, g * 128 + i) - 8, q2 = kk::ct_code(words2.data(), k, r, g * 128 + i) - 8;
        bad_code += q != static_cast<int>(std::clamp(std::nearbyint(v / s), -8.f, 7.f));
        if (v == ext) edge_miss += q != -8;
        e1 += (v - q * s) * (v - q * s), e2 += (v - q2 * s2) * (v - q2 * s2);
      }
      mse_worse += e2 > e1 * (1 + 1e-9);
    }
  expect(bad_code == 0 && bad_scale == 0 && edge_miss == 0 && pos > 0 && neg > 0,
         "quantize_w4_row: codes clamp(rint(w / s)) + 8 at ct_code's positions, s = fp16(-extreme / 8), extreme on code -8, both signs (" +
             std::to_string(bad_code) + " / " + std::to_string(bad_scale) + " / " + std::to_string(edge_miss) + " bad)");
  expect(bad_zero == 0, "quantize_w4_row: an all-zero group -> scale 0, codes 8");
  expect(mse_worse == 0, "quantize_w4_row: the MSE clip search's error <= plain RTN's on every group (" + std::to_string(mse_worse) + " groups worse)");
}

// model:SeqReader (the loaders' one reader thread): items in order, at most `ahead` past the last taken,
// the gate holds the first read, a read error reaches take, the destructor stops a gated reader.
void seq_reader() {
  using namespace std::chrono_literals;
  std::atomic<int> started{0}, max_lead{0};
  std::atomic<int> taken{0};
  {
    xenithium::model::SeqReader r(10, 3, [&](int i) {
      started = i + 1;
      max_lead = std::max(max_lead.load(), i - taken.load());
      return xenithium::model::BulkBuffer{};
    });
    std::this_thread::sleep_for(50ms);
    const int before = started.load();
    bool order = true;
    for (int i = 0; i < 10; ++i) {
      (void)r.take(i);
      taken = i + 1;
    }
    expect(before == 3 && order && started == 10 && max_lead <= 3, "seq reader: reads 3 ahead before any take, all 10 in order");
  }
  {
    std::promise<void> gate;
    std::atomic<int> n{0};
    bool done = false;
    {
      xenithium::model::SeqReader r(4, 4, [&](int) { ++n; return xenithium::model::BulkBuffer{}; }, [&] { done = true; }, gate.get_future().share());
      std::this_thread::sleep_for(30ms);
      const int held = n.load();
      gate.set_value();
      for (int i = 0; i < 4; ++i) (void)r.take(i);
      std::this_thread::sleep_for(10ms);
      expect(held == 0 && n == 4 && done, "seq reader: the gate holds the first read; done after the last");
    }
  }
  {
    xenithium::model::SeqReader r(3, 2, [&](int i) -> xenithium::model::BulkBuffer {
      if (i == 1) throw std::runtime_error("read 1 failed");
      return {};
    });
    bool first = true, threw = false;
    try {
      (void)r.take(0);
    } catch (...) {
      first = false;
    }
    try {
      (void)r.take(1);
    } catch (const std::runtime_error &e) {
      threw = std::string(e.what()) == "read 1 failed";
    }
    expect(first && threw, "seq reader: item 0 arrives, item 1's read error reaches take(1)");
  }
  {
    std::promise<void> gate;  // never opened: the destructor must still stop the reader
    const auto t0 = std::chrono::steady_clock::now();
    { xenithium::model::SeqReader r(2, 1, [&](int) { return xenithium::model::BulkBuffer{}; }, {}, gate.get_future().share()); }
    expect(std::chrono::steady_clock::now() - t0 < 1s, "seq reader: destroyed while gated, stops at once");
  }
}

int main() {
  try {
    bf16_round();
    w4_rtn();
    synthetic();
    bulk_synthetic();
    mtp_host();
    real_model();
    real_model_qwen35();
    real_model_llama();
    real_model_flint();
    real_model_qwen35_gptq();
    mtp35_plan();
    select_share();
    kv_pages();
    kv_lend();
    kv_lend_regions();
    seq_reader();
  } catch (const std::exception &e) {
    std::printf("[FAIL] exception: %s\n", e.what());
    return 1;
  }
  std::printf("%s: %d passed, %d failed\n", g_fail ? "FAILED" : "OK", g_pass, g_fail);
  return g_fail ? 1 : 0;
}
