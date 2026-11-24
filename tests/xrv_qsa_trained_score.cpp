// XRV adapter: qsa.trained_score (trained QSA decode index scores, the selector's score stage).
// Replays one score launch on fixed inputs without the model: the production reference (the FMA / sub-group shuffle
// kernel of trained_qsa_device.cpp index_scores_tile<Half=true, Tree=true>, copied verbatim below) and the candidate
// (kernels::qsa_trained_dec_xmx_score), against an independent fp64 CPU reference. Inputs: deterministic synthetic
// (--seed) or a fixture directory (--fixture DIR: query.f32le [rows][4][128], pool.f16le [pitch][128], meta.json-free:
// dims on the command line). Output: one JSON record on stdout (timing median / p90 / p99 + samples per kernel, max
// relative error vs fp64, exact top-512 identity vs fp64 with near-tie accounting, hardware / runtime identity).
// Build: cmake --build build --target xenithium-xrv-qsa-trained-score. Do not run it on cards that are serving. Run: xenithium-xrv-qsa-trained-score --rows 16 --seg-rows 8 --bound 22000 --repeat 50
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <vector>
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include "xenithium/kernels/attention/qsa.hpp"
#include "xenithium/kernels/rows.hpp"
namespace k = xenithium::kernels;

constexpr int heads = 4, dim = 128, workers = 32, kTop = 512;

// production reference: index_scores_tile<Half=true, Tree=true> (trained_qsa_device.cpp), History -> raw pointers
sycl::event ref_scores(sycl::queue &q, const float *query, const sycl::half *pool16, int cap, int rows, int bound, const int32_t *rec, int stride,
                       int layout, uint32_t *scores, int32_t *eligible) {
  constexpr int R = 4, B = 8;
  const int pitch = cap / 4, tiles = (bound + B - 1) / B;
  const int segments = k::seg_count(rows, layout), groups = (k::seg_size(rows, layout) + R - 1) / R;
  const auto regs = sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>};
  return q.parallel_for(sycl::nd_range<2>({size_t(segments * groups), size_t(tiles) * workers}, {1, workers}), regs,
                        [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(workers)]] {
    const int g = it.get_group(0), s = g / groups, j0 = (g % groups) * R, b0 = it.get_group(1) * B, lane = it.get_local_id(1);
    int rseg, nseg;
    k::seg_span(layout, rows, s, rseg, nseg);
    const int n = sycl::min(R, nseg - j0);
    if (n < 1) return;
    const int start = rec[s * stride + k::kSegPos], slot = rec[s * stride + k::kSegSlot];
    const int r0 = rseg + j0, nmax = start / 4;
    if (!b0 && !lane) for (int r = 0; r < n; ++r) eligible[r0 + r] = nmax;
    if (b0 >= nmax) return;
    const auto sg = it.get_sub_group();
    float qr[R][heads][dim / workers];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
      for (int head = 0; head < heads; ++head)
#pragma unroll
        for (int i = 0; i < dim / workers; ++i) qr[r][head][i] = r < n ? query[(size_t(r0 + r) * heads + head) * dim + lane + i * workers] : 0.f;
    for (int b = b0; b < sycl::min(b0 + B, nmax); ++b) {
      const size_t at = (size_t(slot) * pitch + b) * dim;
      float kr[dim / workers], score[R]{};
#pragma unroll
      for (int i = 0; i < dim / workers; ++i) kr[i] = static_cast<float>(pool16[at + lane + i * workers]);
#pragma unroll
      for (int head = 0; head < heads; ++head)
#pragma unroll
        for (int r = 0; r < R; ++r) {
          float dot = qr[r][head][0] * kr[0];
#pragma unroll
          for (int i = 1; i < dim / workers; ++i) dot = sycl::fma(qr[r][head][i], kr[i], dot);
          dot += sycl::permute_group_by_xor(sg, dot, 16);
          dot += sycl::permute_group_by_xor(sg, dot, 8);
          dot += sycl::permute_group_by_xor(sg, dot, 4);
          dot += sycl::permute_group_by_xor(sg, dot, 2);
          dot += sycl::permute_group_by_xor(sg, dot, 1);
          score[r] += sycl::fmax(0.f, dot);
        }
      if (!lane)
#pragma unroll
        for (int r = 0; r < R; ++r) if (r < n) scores[size_t(r0 + r) * pitch + b] = sycl::bit_cast<uint32_t>(score[r]);
    }
  });
}

struct Stat { double med, p90, p99; std::vector<double> s; };
Stat stats(std::vector<double> v) {
  Stat st{0, 0, 0, v};
  std::sort(v.begin(), v.end());
  auto at = [&](double p) { return v[std::min(v.size() - 1, size_t(p * double(v.size() - 1) + 0.5))]; };
  st.med = at(0.5), st.p90 = at(0.9), st.p99 = at(0.99);
  return st;
}
double ev_ms(const sycl::event &e) {
  return double(e.get_profiling_info<sycl::info::event_profiling::command_end>() - e.get_profiling_info<sycl::info::event_profiling::command_start>()) / 1e6;
}
template <class T> bool load(const std::string &path, std::vector<T> &v) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return false;
  f.read(reinterpret_cast<char *>(v.data()), std::streamsize(v.size() * sizeof(T)));
  return bool(f);
}

int main(int argc, char **argv) {
  int rows = 16, seg_rows = 8, bound = 22000, repeat = 50, warmup = 5, device = 0;
  unsigned seed = 1;
  std::string fixture;
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string a = argv[i], v = argv[i + 1];
    if (a == "--rows") rows = std::stoi(v);
    else if (a == "--seg-rows") seg_rows = std::stoi(v);
    else if (a == "--bound") bound = std::stoi(v);
    else if (a == "--repeat") repeat = std::stoi(v);
    else if (a == "--warmup") warmup = std::stoi(v);
    else if (a == "--device") device = std::stoi(v);
    else if (a == "--seed") seed = unsigned(std::stoul(v));
    else if (a == "--fixture") fixture = v;
    else { std::fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
  }
  if (seg_rows < 1 || seg_rows > rows || rows % seg_rows || seg_rows > 16 || bound <= kTop) { std::fprintf(stderr, "bad dims\n"); return 2; }
  auto gpus = sycl::device::get_devices(sycl::info::device_type::gpu);
  sycl::queue q{gpus.at(size_t(device)), sycl::property_list{sycl::property::queue::enable_profiling{}, sycl::property::queue::in_order{}}};
  const int segs = rows / seg_rows, pitch = (bound + 64) / 64 * 64, cap = pitch * 4, stride = k::kSegRec;
  // every segment: start = 4 * bound (all `bound` blocks committed), slot 0
  std::vector<int32_t> rec(size_t(segs) * stride, 0);
  for (int s = 0; s < segs; ++s) rec[size_t(s) * stride + k::kSegPos] = 4 * bound, rec[size_t(s) * stride + k::kSegSlot] = 0;
  std::vector<float> query(size_t(rows) * heads * dim);
  std::vector<sycl::half> pool(size_t(pitch) * dim);
  if (!fixture.empty()) {
    if (!load(fixture + "/query.f32le", query) || !load(fixture + "/pool.f16le", pool)) { std::fprintf(stderr, "fixture read failed\n"); return 2; }
  } else {  // keys: unit-ish normed rows with a per-block bias (a long-tailed score distribution, as the trained index)
    std::mt19937 g(seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (auto &x : query) x = nd(g) * 0.09f;
    for (int b = 0; b < pitch; ++b) {
      const float bias = 0.3f * nd(g);
      for (int d = 0; d < dim; ++d) pool[size_t(b) * dim + d] = sycl::half(nd(g) * 0.09f + (d < 8 ? bias : 0.f));
    }
  }
  float *dq = sycl::malloc_device<float>(query.size(), q);
  sycl::half *dp = sycl::malloc_device<sycl::half>(pool.size(), q);
  int32_t *drec = sycl::malloc_device<int32_t>(rec.size(), q), *del = sycl::malloc_device<int32_t>(rows, q);
  uint32_t *dsr = sycl::malloc_device<uint32_t>(size_t(rows) * pitch, q), *dsc = sycl::malloc_device<uint32_t>(size_t(rows) * pitch, q);
  q.memcpy(dq, query.data(), query.size() * 4);
  q.memcpy(dp, pool.data(), pool.size() * 2);
  q.memcpy(drec, rec.data(), rec.size() * 4);
  q.memset(dsr, 0, size_t(rows) * pitch * 4);
  q.memset(dsc, 0, size_t(rows) * pitch * 4).wait();
  auto run_ref = [&] { return ref_scores(q, dq, dp, cap, rows, bound, drec, stride, seg_rows, dsr, del); };
  auto run_cand = [&] { return k::qsa_trained_dec_xmx_score(q, dq, dp, drec, stride, seg_rows, rows, pitch, bound, dsc, del); };
  auto time = [&](auto fn) {
    for (int i = 0; i < warmup; ++i) fn();
    q.wait();
    std::vector<double> t;
    for (int i = 0; i < repeat; ++i) { auto e = fn(); e.wait(); t.push_back(ev_ms(e)); }
    return stats(t);
  };
  const Stat tr = time(run_ref), tc = time(run_cand);
  std::vector<uint32_t> sr(size_t(rows) * pitch), sc(size_t(rows) * pitch);
  q.memcpy(sr.data(), dsr, sr.size() * 4);
  q.memcpy(sc.data(), dsc, sc.size() * 4).wait();
  // fp64 reference over the same fp16 keys and fp32 queries; top-512 exact identity with near-tie accounting
  double maxrel_ref = 0, maxrel_cand = 0;
  long id_ref = 0, id_cand = 0, tie_ref = 0, tie_cand = 0, ab_same = 0, total = 0;
  for (int r = 0; r < rows; ++r) {
    std::vector<double> e(bound);
    for (int b = 0; b < bound; ++b) {
      double s = 0;
      for (int h = 0; h < heads; ++h) {
        double dot = 0;
        for (int d = 0; d < dim; ++d) dot += double(query[(size_t(r) * heads + h) * dim + d]) * double(float(pool[size_t(b) * dim + d]));
        s += std::max(0.0, dot);
      }
      e[b] = s;
    }
    auto topk = [&](auto val) {
      std::vector<int> idx(bound);
      for (int b = 0; b < bound; ++b) idx[b] = b;
      std::stable_sort(idx.begin(), idx.end(), [&](int x, int y) { return val(x) > val(y); });
      idx.resize(kTop);
      std::sort(idx.begin(), idx.end());
      return idx;
    };
    const auto fr = [&](int b) { return double(sycl::bit_cast<float>(sr[size_t(r) * pitch + b])); };
    const auto fc = [&](int b) { return double(sycl::bit_cast<float>(sc[size_t(r) * pitch + b])); };
    std::vector<double> sorted = e;
    std::nth_element(sorted.begin(), sorted.begin() + kTop - 1, sorted.end(), std::greater<>());
    const double thr = sorted[kTop - 1];
    const auto te = topk([&](int b) { return e[b]; }), trr = topk(fr), tcc = topk(fc);
    auto score_vs = [&](const std::vector<int> &t, long &same, long &tie) {
      std::vector<int> in;
      std::set_intersection(t.begin(), t.end(), te.begin(), te.end(), std::back_inserter(in));
      same += long(in.size());
      for (int b : t)  // a pick not in the fp64 set: a near tie when its exact score is within 1e-5 relative of the cut
        if (!std::binary_search(te.begin(), te.end(), b) && std::fabs(e[b] - thr) <= 1e-5 * std::max(1e-12, std::fabs(thr))) ++tie;
    };
    score_vs(trr, id_ref, tie_ref);
    score_vs(tcc, id_cand, tie_cand);
    {
      std::vector<int> in;
      std::set_intersection(trr.begin(), trr.end(), tcc.begin(), tcc.end(), std::back_inserter(in));
      ab_same += long(in.size());
    }
    total += kTop;
    for (int b = 0; b < bound; ++b) {
      const double den = std::max(1e-6, std::fabs(e[b]));
      maxrel_ref = std::max(maxrel_ref, std::fabs(fr(b) - e[b]) / den);
      maxrel_cand = std::max(maxrel_cand, std::fabs(fc(b) - e[b]) / den);
    }
  }
  const auto dev = q.get_device();
  auto pr = [](const char *name, const Stat &s) {
    std::printf("\"%s\":{\"median_ms\":%.4f,\"p90_ms\":%.4f,\"p99_ms\":%.4f,\"samples\":[", name, s.med, s.p90, s.p99);
    for (size_t i = 0; i < s.s.size(); ++i) std::printf("%s%.4f", i ? "," : "", s.s[i]);
    std::printf("]}");
  };
  std::printf("{\"adapter\":\"qsa.trained_score\",\"rows\":%d,\"seg_rows\":%d,\"bound\":%d,\"seed\":%u,\"fixture\":\"%s\",\"device\":\"%s\",\"driver\":\"%s\",",
              rows, seg_rows, bound, seed, fixture.c_str(), dev.get_info<sycl::info::device::name>().c_str(), dev.get_info<sycl::info::device::driver_version>().c_str());
  pr("reference", tr);
  std::printf(",");
  pr("candidate", tc);
  std::printf(",\"speedup\":%.3f,\"maxrel_vs_fp64\":{\"reference\":%.3g,\"candidate\":%.3g},", tr.med / tc.med, maxrel_ref, maxrel_cand);
  std::printf("\"top512_vs_fp64\":{\"reference\":[%ld,%ld],\"candidate\":[%ld,%ld],\"total\":%ld},\"top512_reference_vs_candidate\":%ld,", id_ref, tie_ref,
              id_cand, tie_cand, total, ab_same);
  const bool pass = id_cand + tie_cand == total;
  std::printf("\"pass\":%s,\"note\":\"top512 pairs = [identical picks, extra picks that are near ties of the fp64 cut]\"}\n", pass ? "true" : "false");
  sycl::free(dq, q); sycl::free(dp, q); sycl::free(drec, q); sycl::free(del, q); sycl::free(dsr, q); sycl::free(dsc, q);
  return pass ? 0 : 1;
}
