// xenithium-bake dense-tune: AutoRound W4A8 of a dense model's MLPs (qwen35, the 27B), from its W4A16 GPTQ export
// The dense MLP is one "expert" with weight 1 (ArShape experts 1, top-1), so the
// routed experts' tuner (autoround.hpp) runs it as is: gate_up [2I][H] = gate rows then up rows, down [H][I], 128-k
// weight groups, the A8 fake quant per (row, 128-k group) of x and of h (the dense W4A8 prefill's quantiser).
// The source weights are the export's own dequantised values (q x s): RTN at the same format gives the export's codes
// back, so RTN = today's model, and tuning moves codes and clipping for int8 activations against the W4A16 path.
// Inputs: a capture directory (probe model35's --cap: per layer, per prefill call of seqlen rows, bf16
// [rows][H]): L<l>-a8-x.bin (the MLP input on the W4A8 path), L<l>-a8-res.bin (its residual before the MLP add),
// L<l>-a16-res.bin and L<l>-a16-mlp.bin (the W4A16 path's residual and MLP output). The block objective:
// || res_q + m(x_q) - (res_fp + m_fp) ||^2 -> T = res_fp + m_fp - res_q, G = 1.
// xenithium-bake dense-tune --model W4A16_DIR --cap DIR --layers A-B --seqlen 2048 [--samples N] [--eval 8]
// [--iters 200] [--out DIR] [--gpus N]
// --out: per layer DIR/mlp-L<l>.safetensors with gate_proj / up_proj / down_proj in the export's GPTQ layout.
#include <sys/stat.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "bake.hpp"
#include "xenithium/quant/autoround.hpp"
#include "xenithium/quant/export.hpp"

namespace m = xenithium::model;
namespace q = xenithium::quant;

namespace xenithium::bake {
namespace {

std::vector<uint16_t> read_bf16(const std::string &path, size_t elems) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("dense-tune: cannot read " + path);
  std::vector<uint16_t> v(elems);
  f.read(reinterpret_cast<char *>(v.data()), static_cast<std::streamsize>(elems * 2));
  if (static_cast<size_t>(f.gcount()) != elems * 2) throw std::runtime_error("dense-tune: " + path + " is short");
  return v;
}

size_t file_elems(const std::string &path) {
  struct stat st {};
  if (::stat(path.c_str(), &st) != 0) throw std::runtime_error("dense-tune: no " + path);
  return static_cast<size_t>(st.st_size) / 2;
}

// one GPTQ projection [n][k] of the export, dequantised to bf16 (q x s, the value the W4A16 kernels compute with)
void dequant(const m::Checkpoint &ck, const std::string &base, int n, int k, uint16_t *dst) {
  const auto &tq = ck.at(base + ".qweight"), &ts = ck.at(base + ".scales"), &tz = ck.at(base + ".qzeros");
  if (tq.dim(0) != k / 8 || tq.dim(1) != n || ts.dim(0) != k / 128 || ts.dim(1) != n) throw std::runtime_error("dense-tune: " + base + " shape");
  std::vector<uint32_t> qw(static_cast<size_t>(k / 8) * n), z(tz.bytes / 4);
  std::vector<uint16_t> sc(static_cast<size_t>(k / 128) * n);
  read_tensor_range(ck, tq, 0, tq.bytes, qw.data());
  read_tensor_range(ck, ts, 0, ts.bytes, sc.data());
  read_tensor_range(ck, tz, 0, tz.bytes, z.data());
  for (uint32_t w : z)
    if (w != 0x77777777u) throw std::runtime_error("dense-tune: " + base + " has zero points other than 8 (asymmetric)");
  std::vector<int8_t> codes(static_cast<size_t>(n) * k);
  std::vector<uint16_t> snk(static_cast<size_t>(n) * (k / 128));
  q::gptq_unpack(qw.data(), sc.data(), n, k, codes.data(), snk.data());
  for (int r = 0; r < n; ++r)
    for (int c = 0; c < k; ++c) {
      const float s = q::ar_half(snk[static_cast<size_t>(r) * (k / 128) + c / 128]);
      dst[static_cast<size_t>(r) * k + c] = q::ar_bf16_bits(s * static_cast<float>(codes[static_cast<size_t>(r) * k + c]));
    }
}

std::pair<int, int> range(const Args &a, int layers) {
  const std::string r = a.str("layers", "0-" + std::to_string(layers - 1));
  const size_t d = r.find('-');
  const int l0 = std::stoi(r.substr(0, d)), l1 = d == std::string::npos ? l0 : std::stoi(r.substr(d + 1));
  return {l0, l1 + 1};
}

}  // namespace

int cmd_dense_tune(const Args &a) {
  const std::string mdir = a.str("model", home() + "/models/swift15-w4a16-autoround"), cap = a.str("cap");
  if (cap.empty()) throw std::runtime_error("dense-tune: --cap DIR is required");
  const auto ck = m::Checkpoint::open(mdir);
  const std::string P = a.str("prefix", "model.language_model.layers.");
  // gate_proj qweight [H / 8][I], down_proj qweight [I / 8][H]
  const int H = static_cast<int>(ck.at(P + "0.mlp.gate_proj.qweight").dim(0)) * 8, I = static_cast<int>(ck.at(P + "0.mlp.gate_proj.qweight").dim(1));
  if (ck.at(P + "0.mlp.down_proj.qweight").dim(0) * 8 != I || ck.at(P + "0.mlp.down_proj.qweight").dim(1) != H) throw std::runtime_error("dense-tune: down_proj shape");
  const int seqlen = static_cast<int>(a.num("seqlen", 2048));
  q::ArShape s;
  s.hidden = H, s.inter = I, s.experts = 1, s.topk = 1, s.wgroup = 128, s.xgroup = 128, s.hgroup = 128;
  q::ArConfig cfg;
  cfg.iters = static_cast<int>(a.num("iters", 200));
  cfg.batch_samples = static_cast<int>(a.num("batch", 4));
  cfg.floor = a.num("floor", 1) != 0;
  const int eval = static_cast<int>(a.num("eval", 8));
  const bool local = a.num("local", 0) != 0;  // --local 1: capture files L<l>-x.bin / L<l>-mlp.bin, the local objective
  const std::string out = a.str("out");
  if (!out.empty()) std::filesystem::create_directories(out);
  auto workers = make_workers(a);
  if (workers.size() > 1) workers.resize(1);  // one expert: one worker holds it
  const auto [l0, l1] = range(a, 64);
  std::vector<uint16_t> gu(static_cast<size_t>(2) * I * H), dn(static_cast<size_t>(H) * I);
  for (int l = l0; l < l1; ++l) {
    const auto t0 = std::chrono::steady_clock::now();
    const std::string L = cap + "/L" + std::to_string(l) + "-";
    const size_t E = file_elems(L + (local ? "x.bin" : "a8-x.bin"));
    int samples = static_cast<int>(E / (static_cast<size_t>(seqlen) * H));
    if (a.has("samples")) samples = std::min(samples, static_cast<int>(a.num("samples", samples)));
    const size_t n = static_cast<size_t>(samples) * seqlen * H;
    std::vector<uint16_t> x, t;
    if (local) {  // the MLP's own A8 error: x on the A8 path with the MLP kept W4A16 (keep=mlp), T = that MLP's output
      x = read_bf16(L + "x.bin", n);
      t = read_bf16(L + "mlp.bin", n);
    } else {
      x = read_bf16(L + "a8-x.bin", n);
      const auto rq = read_bf16(L + "a8-res.bin", n), rf = read_bf16(L + "a16-res.bin", n), mf = read_bf16(L + "a16-mlp.bin", n);
      t.resize(n);
      for (size_t i = 0; i < n; ++i) t[i] = q::ar_bf16_bits(q::ar_bf16(rf[i]) + q::ar_bf16(mf[i]) - q::ar_bf16(rq[i]));
    }
    std::vector<int32_t> ids(static_cast<size_t>(samples) * seqlen, 0);
    std::vector<float> w(ids.size(), 1.f);
    q::LayerData d;
    d.samples = samples, d.seqlen = seqlen, d.x = x.data(), d.ids = ids.data(), d.w = w.data(), d.t = t.data();
    const std::string b = P + std::to_string(l) + ".mlp.";
    dequant(ck, b + "gate_proj", I, H, gu.data());
    dequant(ck, b + "up_proj", I, H, gu.data() + static_cast<size_t>(I) * H);
    dequant(ck, b + "down_proj", H, I, dn.data());
    const double tr = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::vector<q::ExpertQuant> res(1);
    const auto st = q::tune_layer(s, cfg, d, q::ExpertSource{gu.data(), dn.data()}, workers, eval, res);
    // RTN must give the export's codes back (the source is their dequantised values)
    q::ExpertQuant rtn;
    q::rtn_expert(s, q::ExpertSource{gu.data(), dn.data()}, 0, rtn);
    size_t moved = 0, total = res[0].gu_q.size() + res[0].dn_q.size();
    for (size_t i = 0; i < res[0].gu_q.size(); ++i) moved += res[0].gu_q[i] != rtn.gu_q[i];
    for (size_t i = 0; i < res[0].dn_q.size(); ++i) moved += res[0].dn_q[i] != rtn.dn_q[i];
    const double tq = st.rtn_loss - st.floor_loss, tt = st.tuned_loss - st.floor_loss;
    std::printf("layer %2d: %d samples (%d eval), read %.0f s, tune %.0f s; held-out loss floor %.6g, RTN %.6g, tuned %.6g; quant part RTN %.6g -> tuned %.6g (%+.1f %%); codes moved %.2f %%\n",
                l, samples, eval, tr, st.seconds, st.floor_loss, st.rtn_loss, st.tuned_loss, tq, tt, tq > 0 ? 100 * (tt / tq - 1) : 0.0, 100.0 * moved / total);
    std::fflush(stdout);
    if (!out.empty()) {
      std::vector<std::vector<uint8_t>> store;
      std::vector<q::StTensor> ts;
      auto proj = [&](const std::string &name, const int8_t *codes, const uint16_t *sc, int nn, int kk) {
        store.emplace_back(static_cast<size_t>(kk / 8) * nn * 4);
        q::gptq_pack(codes, nn, kk, reinterpret_cast<uint32_t *>(store.back().data()));
        ts.push_back({name + ".qweight", "I32", {kk / 8, nn}, store.back().data(), store.back().size()});
        store.emplace_back(static_cast<size_t>(kk / 128) * (nn / 8) * 4);
        std::fill(reinterpret_cast<uint32_t *>(store.back().data()), reinterpret_cast<uint32_t *>(store.back().data()) + store.back().size() / 4, 0x77777777u);
        ts.push_back({name + ".qzeros", "I32", {kk / 128, nn / 8}, store.back().data(), store.back().size()});
        store.emplace_back(static_cast<size_t>(kk / 128) * nn * 2);
        auto *o = reinterpret_cast<uint16_t *>(store.back().data());
        for (int g = 0; g < kk / 128; ++g)
          for (int r = 0; r < nn; ++r) o[static_cast<size_t>(g) * nn + r] = sc[static_cast<size_t>(r) * (kk / 128) + g];
        ts.push_back({name + ".scales", "F16", {kk / 128, nn}, store.back().data(), store.back().size()});
      };
      const auto &e = res[0];
      proj(b + "gate_proj", e.gu_q.data(), e.gu_s.data(), I, H);
      proj(b + "up_proj", e.gu_q.data() + static_cast<size_t>(I) * H, e.gu_s.data() + static_cast<size_t>(I) * (H / 128), I, H);
      proj(b + "down_proj", e.dn_q.data(), e.dn_s.data(), H, I);
      q::write_safetensors(out + "/mlp-L" + std::to_string(l) + ".safetensors", ts, {{"format", "pt"}, {"xenithium_scheme", "w4g128-sym-autoround-a8"}});
    }
  }
  return 0;
}

// xenithium-bake dense-assemble --model SRC --tuned DIR --out DST: SRC with every tensor of DIR's *.safetensors written over
// its namesake (same dtype and bytes, in place at the same offsets): a shard that holds one is copied, the rest of SRC is
// hard-linked. A DST that exists is replaced shard by shard.
int cmd_dense_assemble(const Args &a) {
  namespace fs = std::filesystem;
  const std::string src = a.str("model"), tdir = a.str("tuned"), out = a.str("out");
  if (src.empty() || tdir.empty() || out.empty()) throw std::runtime_error("dense-assemble: --model, --tuned and --out are required");
  const auto ck = m::Checkpoint::open(src);
  const auto tk = m::Checkpoint::open(tdir, false);
  fs::create_directories(out);
  std::map<uint32_t, std::vector<const m::TensorInfo *>> by_shard;  // source shard -> tuned tensors in it
  for (const auto &t : tk.tensors()) {
    const auto &s = ck.at(t.name);
    if (s.bytes != t.bytes || s.dtype_str != t.dtype_str) throw std::runtime_error("dense-assemble: " + t.name + " differs in dtype / size");
    by_shard[s.shard].push_back(&t);
  }
  for (const auto &e : fs::directory_iterator(src)) {
    if (!e.is_regular_file()) continue;
    const std::string name = e.path().filename().string(), dst = out + "/" + name;
    fs::remove(dst);
    uint32_t si = 0;
    bool tuned = false;
    for (; si < ck.shards().size(); ++si)
      if (ck.shards()[si].file == name) {
        tuned = by_shard.count(si) != 0;
        break;
      }
    if (!tuned) {
      std::error_code ec;
      fs::create_hard_link(e.path(), dst, ec);
      if (ec) fs::copy_file(e.path(), dst);
      continue;
    }
    fs::copy_file(e.path(), dst);
    FILE *f = std::fopen(dst.c_str(), "r+b");
    if (!f) throw std::runtime_error("dense-assemble: cannot open " + dst);
    std::vector<uint8_t> buf;
    for (const m::TensorInfo *t : by_shard[si]) {
      buf.resize(t->bytes);
      read_tensor_range(tk, *t, 0, t->bytes, buf.data());
      const auto &s = ck.at(t->name);
      if (std::fseek(f, static_cast<long>(s.offset), SEEK_SET) != 0 || std::fwrite(buf.data(), 1, buf.size(), f) != buf.size())
        throw std::runtime_error("dense-assemble: write failed in " + dst);
    }
    std::fclose(f);
    std::printf("dense-assemble: %s: %zu tensors replaced\n", name.c_str(), by_shard[si].size());
  }
  return 0;
}

}  // namespace xenithium:bake
