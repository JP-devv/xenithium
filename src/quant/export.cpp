#include "xenithium/quant/export.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <memory>
#include <stdexcept>

#include "xenithium/util/json.hpp"

namespace xenithium::quant {

void write_safetensors(const std::string &path, const std::vector<StTensor> &tensors, const std::map<std::string, std::string> &metadata) {
  using json::Value;
  Value h = Value::object();
  if (!metadata.empty()) {
    Value m = Value::object();
    for (const auto &[k, v] : metadata) m.set(k, Value::string(v));
    h.set("__metadata__", std::move(m));
  }
  uint64_t off = 0;
  for (const auto &t : tensors) {
    uint64_t numel = 1;
    for (const int64_t d : t.shape) numel *= static_cast<uint64_t>(d);
    const uint64_t es = t.dtype == "I32" || t.dtype == "F32" ? 4 : t.dtype == "F16" || t.dtype == "BF16" ? 2 : t.dtype == "I64" ? 8 : 1;
    if (numel * es != t.bytes) throw std::runtime_error("safetensors: " + t.name + ": bytes != numel x element size");
    Value e = Value::object();
    e.set("dtype", Value::string(t.dtype));
    Value sh = Value::array();
    for (const int64_t d : t.shape) sh.push(Value::integer(d));
    e.set("shape", std::move(sh));
    Value o = Value::array();
    o.push(Value::integer(static_cast<int64_t>(off)));
    o.push(Value::integer(static_cast<int64_t>(off + t.bytes)));
    e.set("data_offsets", std::move(o));
    h.set(t.name, std::move(e));
    off += t.bytes;
  }
  std::string hs = json::dump(h);
  while (hs.size() % 8) hs += ' ';
  const std::string part = path + ".part";
  std::unique_ptr<FILE, int (*)(FILE *)> f(std::fopen(part.c_str(), "wb"), &std::fclose);
  if (!f) throw std::runtime_error("safetensors: cannot write " + part);
  const uint64_t hl = hs.size();
  bool ok = std::fwrite(&hl, 8, 1, f.get()) == 1 && std::fwrite(hs.data(), 1, hs.size(), f.get()) == hs.size();
  for (const auto &t : tensors) ok = ok && (t.bytes == 0 || std::fwrite(t.data, 1, t.bytes, f.get()) == t.bytes);
  ok = ok && std::fflush(f.get()) == 0;
  f.reset();
  if (!ok) throw std::runtime_error("safetensors: write failed: " + part);
  std::filesystem::rename(part, path);
}

void gptq_pack(const int8_t *q, int n, int k, uint32_t *qw) {
  for (int r = 0; r < k / 8; ++r)
    for (int c = 0; c < n; ++c) {
      uint32_t w = 0;
      for (int i = 0; i < 8; ++i) w |= static_cast<uint32_t>(q[static_cast<size_t>(c) * k + 8 * r + i] + 8) << (4 * i);
      qw[static_cast<size_t>(r) * n + c] = w;
    }
}

void gptq_unpack(const uint32_t *qweight, const uint16_t *scales, int n, int k, int8_t *codes, uint16_t *scale_nk) {
  for (int r = 0; r < k / 8; ++r)
    for (int c = 0; c < n; ++c) {
      const uint32_t w = qweight[static_cast<size_t>(r) * n + c];
      for (int i = 0; i < 8; ++i) codes[static_cast<size_t>(c) * k + 8 * r + i] = static_cast<int8_t>(static_cast<int>((w >> (4 * i)) & 15u) - 8);
    }
  for (int g = 0; g < k / 128; ++g)
    for (int c = 0; c < n; ++c) scale_nk[static_cast<size_t>(c) * (k / 128) + g] = scales[static_cast<size_t>(g) * n + c];
}

std::vector<StTensor> expert_export_tensors(const ArShape &s, const std::string &prefix, int layer, const std::vector<ExpertQuant> &q,
                                            std::vector<std::vector<uint8_t>> &store) {
  const int H = s.hidden, I = s.inter, G = s.wgroup;
  if (G != 128) throw std::runtime_error("export: group 128 only");
  std::vector<StTensor> out;
  store.reserve(store.size() + q.size() * 9);
  // one projection: codes rows [r0, r0 + n) of an [*][k] code matrix
  auto proj = [&](const std::string &base, const int8_t *codes, const uint16_t *sc, int n, int k) {
    store.emplace_back(static_cast<size_t>(k / 8) * n * 4);
    gptq_pack(codes, n, k, reinterpret_cast<uint32_t *>(store.back().data()));
    out.push_back({base + "qweight", "I32", {k / 8, n}, store.back().data(), store.back().size()});
    store.emplace_back(static_cast<size_t>(k / G) * (n / 8) * 4);
    auto *z = reinterpret_cast<uint32_t *>(store.back().data());
    for (size_t i = 0; i < store.back().size() / 4; ++i) z[i] = 0x77777777u;
    out.push_back({base + "qzeros", "I32", {k / G, n / 8}, store.back().data(), store.back().size()});
    store.emplace_back(static_cast<size_t>(k / G) * n * 2);
    auto *st = reinterpret_cast<uint16_t *>(store.back().data());
    for (int g = 0; g < k / G; ++g)
      for (int c = 0; c < n; ++c) st[static_cast<size_t>(g) * n + c] = sc[static_cast<size_t>(c) * (k / G) + g];
    out.push_back({base + "scales", "F16", {k / G, n}, store.back().data(), store.back().size()});
  };
  for (size_t e = 0; e < q.size(); ++e) {
    const ExpertQuant &x = q[e];
    if (x.gu_q.size() != static_cast<size_t>(2) * I * H || x.dn_q.size() != static_cast<size_t>(H) * I)
      throw std::runtime_error("export: expert " + std::to_string(e) + " has no result");
    const std::string b = prefix + "layers." + std::to_string(layer) + ".mlp.experts." + std::to_string(e) + ".";
    const size_t gs = static_cast<size_t>(I) * (H / G);
    proj(b + "gate_proj.", x.gu_q.data(), x.gu_s.data(), I, H);
    proj(b + "up_proj.", x.gu_q.data() + static_cast<size_t>(I) * H, x.gu_s.data() + gs, I, H);
    proj(b + "down_proj.", x.dn_q.data(), x.dn_s.data(), H, I);
  }
  return out;
}

void ple_int8_rows(const uint16_t *src, int64_t rows, int dim, uint8_t *dst, double *err2, double *ref2) {
  double e2 = 0, r2 = 0;
  const size_t rb = static_cast<size_t>(dim) + 4;
  for (int64_t r = 0; r < rows; ++r) {
    const uint16_t *v = src + static_cast<size_t>(r) * dim;
    uint8_t *o = dst + static_cast<size_t>(r) * rb;
    float amax = 0.f;
    for (int i = 0; i < dim; ++i) amax = std::max(amax, std::fabs(ar_bf16(v[i])));
    const float s = amax / 127.f;
    for (int i = 0; i < dim; ++i) {
      const float x = ar_bf16(v[i]);
      float t = amax > 0.f ? std::nearbyint(x / s) : 0.f;
      t = std::min(127.f, std::max(-127.f, t));
      o[i] = static_cast<uint8_t>(static_cast<int8_t>(t));
      const double d = static_cast<double>(t) * s - x;
      e2 += d * d;
      r2 += static_cast<double>(x) * x;
    }
    std::memcpy(o + dim, &s, 4);
  }
  if (err2) *err2 += e2;
  if (ref2) *ref2 += r2;
}

}  // namespace xenithium:quant
