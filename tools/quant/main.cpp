// xenithium-quant: 8-bit formats for the dense weights, measured on the real checkpoint.
// xenithium-quant MODEL_DIR [--rows N] [--tensors NAME,NAME...]
// (--tensors replaces the default set; names without "lm_head" get the language-model prefix)
// For a set of dense tensors (each arch module's real matrices), quantises every output row
// three ways - int8 symmetric (amax), int8 with a per-row clip search, FP8 E4M3 (amax / 448) -
// and prints the relative Frobenius error of the weights and of y = x W^T for Gaussian x
// (a proxy: real activations have outlier channels). Host only.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "xenithium/kernels/pack.hpp"
#include "xenithium/model/bulk_read.hpp"
#include "xenithium/model/checkpoint.hpp"

namespace k = xenithium::kernels;
namespace m = xenithium::model;

int main(int argc, char **argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: xenithium-quant MODEL_DIR [--rows N]\n");
    return 2;
  }
  int max_rows = 4096;
  for (int i = 2; i + 1 < argc; i += 2)
    if (!std::strcmp(argv[i], "--rows")) max_rows = std::atoi(argv[i + 1]);
  const auto ck = m::Checkpoint::open(argv[1]);
  const std::string P = "model.language_model.";
  std::vector<std::string> names = {
      P + "layers.0.linear_attn.in_proj_qkv.weight",       P + "layers.0.linear_attn.in_proj_z.weight",
      P + "layers.0.linear_attn.out_proj.weight",          P + "layers.20.linear_attn.in_proj_qkv.weight",
      P + "layers.3.self_attn.q_proj.weight",              P + "layers.3.self_attn.k_proj.weight",
      P + "layers.3.self_attn.o_proj.weight",              P + "layers.3.self_attn.indexer.index_qk_proj.weight",
      P + "layers.0.attn_hyper_connection.input_mix_weight_down.weight",
      P + "layers.0.attn_hyper_connection.input_mix_weight_up.weight",
      P + "layers.0.mlp.shared_expert.gate_proj.weight",   P + "layers.0.mlp.shared_expert.down_proj.weight",
      P + "layers.0.mlp.gate.weight",                      P + "layers.1.ple.key_proj.weight",
      "lm_head.weight",
  };
  for (int i = 2; i + 1 < argc; ++i)
    if (!std::strcmp(argv[i], "--tensors")) {
      names.clear();
      std::string list = argv[i + 1];
      for (size_t a = 0; a < list.size();) {
        size_t b = list.find(',', a);
        if (b == std::string::npos) b = list.size();
        const std::string nm = list.substr(a, b - a);
        names.push_back(nm.rfind("lm_head", 0) == 0 || nm.rfind("model.", 0) == 0 ? nm : P + nm);
        a = b + 1;
      }
    }
  std::vector<const m::TensorInfo *> ts;
  for (const auto &n : names) ts.push_back(&ck.at(n));
  const auto buf = m::bulk_read(ck, ts);
  std::printf("rows per tensor: up to %d; x ~ N(0,1), 4 rows; errors are ||ref - q|| / ||ref|| (Frobenius)\n", max_rows);
  std::printf("%-62s %12s | %9s %9s %9s | %9s %9s %9s\n", "tensor", "[n x k]", "w i8", "w i8clip", "w e4m3", "y i8", "y i8clip",
              "y e4m3");
  double sum_w[3] = {0, 0, 0}, sum_y[3] = {0, 0, 0};
  for (const auto *t : ts) {
    const int n_all = static_cast<int>(t->dim(0)), kd = static_cast<int>(t->dim(1));
    const int n = std::min(n_all, max_rows);
    const auto *w = buf.as<uint16_t>(*t);
    std::vector<int8_t> q1(static_cast<size_t>(n) * kd), q2(q1.size());
    std::vector<uint8_t> q3(q1.size());
    std::vector<float> s1(n), s2(n), s3(n);
    k::quantize_i8_rows(w, n, kd, q1.data(), s1.data(), 1);
    k::quantize_i8_rows(w, n, kd, q2.data(), s2.data(), 16);
    k::quantize_e4m3_rows(w, n, kd, q3.data(), s3.data());
    double ref2 = 0, e[3] = {0, 0, 0};
    for (int r = 0; r < n; ++r)
      for (int i = 0; i < kd; ++i) {
        const size_t at = static_cast<size_t>(r) * kd + i;
        const double v = k::bf16_value(w[at]);
        ref2 += v * v;
        const double d1 = v - q1[at] * double(s1[r]), d2 = v - q2[at] * double(s2[r]), d3 = v - k::e4m3_value(q3[at]) * double(s3[r]);
        e[0] += d1 * d1;
        e[1] += d2 * d2;
        e[2] += d3 * d3;
      }
    // Output error: y = x W^T, 4 Gaussian rows.
    std::mt19937 rng(1);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> x(4 * static_cast<size_t>(kd));
    for (auto &v : x) v = nd(rng);
    std::vector<double> yr(4 * static_cast<size_t>(n)), y1(yr.size()), y2(yr.size()), y3(yr.size());
    k::dense_ref_bf16(x.data(), w, 4, n, kd, yr.data());
    k::dense_ref_i8(x.data(), q1.data(), s1.data(), 4, n, kd, y1.data());
    k::dense_ref_i8(x.data(), q2.data(), s2.data(), 4, n, kd, y2.data());
    for (int r = 0; r < 4; ++r)
      for (int c = 0; c < n; ++c) {
        double acc = 0;
        for (int i = 0; i < kd; ++i) acc += double(x[static_cast<size_t>(r) * kd + i]) * k::e4m3_value(q3[static_cast<size_t>(c) * kd + i]);
        y3[static_cast<size_t>(r) * n + c] = acc * s3[c];
      }
    double yy = 0, ey[3] = {0, 0, 0};
    for (size_t i = 0; i < yr.size(); ++i) {
      yy += yr[i] * yr[i];
      ey[0] += (yr[i] - y1[i]) * (yr[i] - y1[i]);
      ey[1] += (yr[i] - y2[i]) * (yr[i] - y2[i]);
      ey[2] += (yr[i] - y3[i]) * (yr[i] - y3[i]);
    }
    std::string nm = t->name.substr(t->name.rfind("layers.") != std::string::npos ? t->name.rfind("layers.") : 0);
    char dims[32];
    std::snprintf(dims, sizeof dims, "%dx%d%s", n_all, kd, n < n_all ? "*" : "");
    std::printf("%-62s %12s | %9.5f %9.5f %9.5f | %9.5f %9.5f %9.5f\n", nm.c_str(), dims, std::sqrt(e[0] / ref2), std::sqrt(e[1] / ref2),
                std::sqrt(e[2] / ref2), std::sqrt(ey[0] / yy), std::sqrt(ey[1] / yy), std::sqrt(ey[2] / yy));
    for (int f = 0; f < 3; ++f) {
      sum_w[f] += std::sqrt(e[f] / ref2);
      sum_y[f] += std::sqrt(ey[f] / yy);
    }
  }
  const double nt = static_cast<double>(ts.size());
  std::printf("%-62s %12s | %9.5f %9.5f %9.5f | %9.5f %9.5f %9.5f\n", "mean", "", sum_w[0] / nt, sum_w[1] / nt, sum_w[2] / nt,
              sum_y[0] / nt, sum_y[1] / nt, sum_y[2] / nt);
  std::printf("(* = first %d output rows only)\n", max_rows);
  return 0;
}
