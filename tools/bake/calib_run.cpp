// xenithium-bake selftest-calib / tune: block-wise AutoRound of the routed experts on the cards.
// xenithium-bake selftest-calib --model W4A16_DIR --calib FILE.xcb
// Gates the calibration path against the engine before any bake relies on it:
// (1) blobs packed from the tuner's codes == the engine's pack_expert of the same int4 expert,
// bit for bit (full blob and both TP halves);
// (2) one sample through calib_layer layer by layer, the state carried through the host,
// == prefill of the same tokens: each layer's MoE input and residual, bit for bit.
// xenithium-bake tune --src BF16_DIR --calib FILE.xcb --ple INT8_PLE_DIR --out DIR --work DIR
// [--samples N] [--layers A-B] [--iters 200] [--eval 16] [--gpus N] [--resume 1]
// [--holdout-spread 1] (held-out samples spread over every source; 0 = the old tail)
// AutoRound's block-wise pipeline, layer by layer (sequential: each block's inputs come from
// the already-quantised previous blocks). Two engine models on the cards: the targets (BF16
// trunk, BF16 source PLE, BF16 experts through the tuner's exact forward) and the quantised
// path (int8 trunk as baked, int8 PLE rows, the tuned W4A8 experts on the engine's moe_rows).
// Per layer l:
// fp pass every sample: fp state(l-1) -> layer l -> fp state(l) (targets)
// q pass 1 every sample: q state(l-1) -> layer l up to its MoE: x, routing, the shared term,
// the residual and injection gates -> T, G, c (autoround.hpp) against fp state(l)
// tune 200 signSGD steps on the GPUs (the block-output MSE), best step kept
// write DIR/experts-LNN.safetensors (the export) + DIR/native/experts-LNN.tp2 (per-card blobs)
// q pass 2 every sample: q state(l-1) -> layer l with the tuned experts -> q state(l)
// States live in WORK/{fp,q}.state (one record per sample, updated in place: ~52 MB per sample
// per path at 2,048 tokens). --resume continues after the last layer WORK/progress records.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <functional>
#include <stdexcept>
#include <thread>
#include <unistd.h>

#include "bake.hpp"
#include "xenithium/arch/qwen38/experts.hpp"
#include "xenithium/arch/qwen38/model.hpp"
#include "xenithium/arch/qwen38/moe.hpp"
#include "xenithium/eval/logprobs.hpp"
#include "xenithium/kernels/moe/moe.hpp"
#include "xenithium/kernels/pack.hpp"
#include "xenithium/model/bulk_read.hpp"
#include "xenithium/model/config.hpp"
#include "xenithium/quant/calib.hpp"
#include "xenithium/quant/export.hpp"
#include "xenithium/rt/fabric.hpp"
#include "xenithium/rt/l0.hpp"

namespace xenithium::bake {

namespace q = xenithium::quant;
namespace m = xenithium::model;
namespace k = xenithium::kernels;
namespace q38 = xenithium::arch::qwen38;

namespace {

using clk = std::chrono::steady_clock;
double secs(clk::time_point t) { return std::chrono::duration<double>(clk::now() - t).count(); }

void pfor(int n, int threads, const std::function<void(int)> &fn) {
  std::atomic<int> next{0};
  std::vector<std::thread> pool;
  for (int t = 0; t < std::max(1, std::min(threads, n)); ++t)
    pool.emplace_back([&] {
      for (int i; (i = next.fetch_add(1)) < n;) fn(i);
    });
  for (auto &t : pool) t.join();
}

// ---- blobs from the tuner's codes ------------------------------------------------------------------
// The engine's full blob (experts.hpp layout) of one expert, as pack_expert builds it from the
// export tensors: GPTQ qweight -> W4 tiles, scales fp16 [k/128][n] (gate | up per group row).
void pack_full_blob(const q38::ExpertLayout &lay, const q::ExpertQuant &e, uint8_t *out) {
  const int H = lay.hidden, I = lay.inter;
  std::vector<uint32_t> g(static_cast<size_t>(H / 8) * I), u(g.size()), d(static_cast<size_t>(I / 8) * H);
  q::gptq_pack(e.gu_q.data(), I, H, g.data());
  q::gptq_pack(e.gu_q.data() + static_cast<size_t>(I) * H, I, H, u.data());
  q::gptq_pack(e.dn_q.data(), H, I, d.data());
  std::memset(out, 0, lay.stride);
  k::pack_w4_gptq_cat(g.data(), u.data(), I, 2 * I, H, out + lay.gu_codes);
  k::pack_w4_gptq(d.data(), H, I, out + lay.dn_codes);
  auto *gs = reinterpret_cast<uint16_t *>(out + lay.gu_scales);
  for (int gi = 0; gi < H / 128; ++gi)
    for (int n = 0; n < 2 * I; ++n) gs[static_cast<size_t>(gi) * 2 * I + n] = e.gu_s[static_cast<size_t>(n) * (H / 128) + gi];
  auto *ds = reinterpret_cast<uint16_t *>(out + lay.dn_scales);
  for (int gi = 0; gi < I / 128; ++gi)
    for (int n = 0; n < H; ++n) ds[static_cast<size_t>(gi) * H + n] = e.dn_s[static_cast<size_t>(n) * (I / 128) + gi];
}

// Every expert's per-card TP blobs: regions[c] = [experts][tp stride].
void pack_tp_regions(const q38::Config &cfg, const std::vector<q::ExpertQuant> &res, std::vector<std::vector<uint8_t>> &regions, int threads) {
  const q38::ExpertLayout lay = q38::ExpertLayout::of(cfg);
  const k::MoeBlobLayout fl = q38::moe_blob_layout(lay);
  size_t st = lay.stride;
  const k::MoeBlobLayout tl = k::moe_tp_layout(cfg.hidden, cfg.moe_inter, 2, &st);
  regions.assign(2, std::vector<uint8_t>(st * res.size(), 0));
  pfor(static_cast<int>(res.size()), threads, [&](int e) {
    thread_local std::vector<uint8_t> full;
    full.assign(lay.stride, 0);
    pack_full_blob(lay, res[static_cast<size_t>(e)], full.data());
    for (int c = 0; c < 2; ++c)
      k::moe_split_tp(fl, cfg.hidden, cfg.moe_inter, 2, c, full.data(), tl, regions[static_cast<size_t>(c)].data() + static_cast<size_t>(e) * st);
  });
}

q38::Config config_of(const std::string &dir) { return q38::Config::parse(m::ModelConfig::load(dir)); }

// ---- the per-row tuning data from a capture and the target state (autoround.hpp's derivation) ------
inline float sig(float x) { return 1.f / (1.f + std::exp(-x)); }

struct RowSink {
  int H = 0, S = 0, K = 0;
  uint16_t *x = nullptr, *t = nullptr;
  int32_t *ids = nullptr;
  float *w = nullptr, *g = nullptr, *c = nullptr;
};

// cap: the q path's MoE inputs; fp: the target state after the layer. Writes rows [0, m) of the sink.
void tuning_rows(const q38::Model::CalibCapture &cap, const q38::Model::CalibState &fp, int m, const RowSink &o, double *g_mean) {
  const int H = o.H, S = o.S, K = o.K;
  double gsum = 0;
  for (int r = 0; r < m; ++r) {
    float gq[8], gf[8];
    for (int s = 0; s < S; ++s) {
      gq[s] = 2.f * sig(q::ar_round_bf16(cap.inj[static_cast<size_t>(r) * S + s]) / static_cast<float>(S));
      gf[s] = 2.f * sig(q::ar_round_bf16(fp.inj[static_cast<size_t>(r) * S + s]) / static_cast<float>(S));
    }
    double G = 0;
    for (int s = 0; s < S; ++s) G += static_cast<double>(gq[s]) * gq[s];
    gsum += G;
    double cc = 0, tt = 0;
    const float sg = cap.sg[static_cast<size_t>(r)];
    for (int n = 0; n < H; ++n) {
      double acc = 0, dd = 0;
      const float ymf = q::ar_bf16(fp.ym[static_cast<size_t>(r) * H + n]);
      for (int s = 0; s < S; ++s) {
        const size_t at = (static_cast<size_t>(r) * S + s) * H + n;
        const float out_fp = q::ar_round_bf16(q::ar_bf16(fp.res[at]) + ymf * gf[s]);  // the target block output
        const double d = static_cast<double>(out_fp) - q::ar_bf16(cap.res[at]);
        acc += gq[s] * d;
        dd += d * d;
      }
      const double tn = acc / G;
      cc += dd;
      tt += tn * tn;
      const double T = tn - static_cast<double>(sg) * cap.sy[static_cast<size_t>(r) * H + n];
      o.t[static_cast<size_t>(r) * H + n] = q::ar_bf16_bits(static_cast<float>(T));
    }
    std::memcpy(o.x + static_cast<size_t>(r) * H, cap.xm.data() + static_cast<size_t>(r) * H, static_cast<size_t>(H) * 2);
    std::memcpy(o.ids + static_cast<size_t>(r) * K, cap.ids.data() + static_cast<size_t>(r) * K, static_cast<size_t>(K) * 4);
    std::memcpy(o.w + static_cast<size_t>(r) * K, cap.topw.data() + static_cast<size_t>(r) * K, static_cast<size_t>(K) * 4);
    o.g[r] = static_cast<float>(G);
    o.c[r] = static_cast<float>(cc - G * tt);
  }
  if (g_mean) *g_mean += gsum;
}

// ---- which calibration samples (cmd_tune's comment) --------------------------------------------------
std::vector<int> sample_picks(int F, int N, int eval, bool spread) {
  std::vector<int> pick(static_cast<size_t>(N));
  if (!spread) {
    for (int i = 0; i < N; ++i) pick[static_cast<size_t>(i)] = static_cast<int>(static_cast<int64_t>(i) * F / N);
    return pick;
  }
  std::vector<char> held(static_cast<size_t>(F), 0);
  std::vector<int> rest, ho;
  for (int j = 0; j < eval; ++j) held[static_cast<size_t>((2 * static_cast<int64_t>(j) + 1) * F / (2 * eval))] = 1;
  for (int i = 0; i < F; ++i) (held[static_cast<size_t>(i)] ? ho : rest).push_back(i);
  const int train = N - eval, R = static_cast<int>(rest.size());
  if (static_cast<int>(ho.size()) != eval || train > R) throw std::runtime_error("held-out spread: samples / eval do not fit the calibration file");
  for (int i = 0; i < train; ++i) pick[static_cast<size_t>(i)] = rest[static_cast<size_t>(static_cast<int64_t>(i) * R / train)];
  for (int j = 0; j < eval; ++j) pick[static_cast<size_t>(train + j)] = ho[static_cast<size_t>(j)];
  return pick;
}

// A WORK directory's picks (WORK/samples; none = the pre-round-65 stride with `eval_default` held out).
std::vector<int> work_picks(const std::string &work, int F, int N, int eval_default, int *eval_out) {
  std::ifstream pf(work + "/samples");
  std::string kind;
  int n = -1, e = -1;
  if (!(pf >> kind >> n >> e)) {
    *eval_out = eval_default;
    return sample_picks(F, N, eval_default, false);
  }
  if (n != N) throw std::runtime_error(work + "/samples says " + std::to_string(n) + " samples, the state files hold " + std::to_string(N));
  *eval_out = e;
  return sample_picks(F, N, e, kind == "spread");
}

// ---- state files: one record per sample, read / written in place --------------------------------
class StateFile {
 public:
  StateFile(const std::string &path, int m, int H, int S, int samples, bool create) : m_(m), H_(H), S_(S) {
    rec_ = static_cast<size_t>(m) * (static_cast<size_t>(S) * H * 2 + static_cast<size_t>(S) * 4 + static_cast<size_t>(H) * 2);
    fd_ = ::open(path.c_str(), O_RDWR | (create ? O_CREAT : 0), 0644);
    if (fd_ < 0) throw std::runtime_error("cannot open " + path);
    if (create && ::ftruncate(fd_, static_cast<off_t>(rec_ * samples)) != 0) throw std::runtime_error("cannot size " + path);
  }
  ~StateFile() {
    if (fd_ >= 0) ::close(fd_);
  }
  void write(int s, const q38::Model::CalibState &st) {
    io(true, s, const_cast<q38::Model::CalibState &>(st));
  }
  void read(int s, q38::Model::CalibState &st) {
    st.m = m_;
    st.res.resize(static_cast<size_t>(m_) * S_ * H_);
    st.inj.resize(static_cast<size_t>(m_) * S_);
    st.ym.resize(static_cast<size_t>(m_) * H_);
    io(false, s, st);
  }

 private:
  void io(bool wr, int s, q38::Model::CalibState &st) {
    off_t off = static_cast<off_t>(rec_ * s);
    auto one = [&](void *p, size_t n) {
      size_t done = 0;
      while (done < n) {
        const ssize_t r = wr ? ::pwrite(fd_, static_cast<char *>(p) + done, n - done, off + static_cast<off_t>(done))
                             : ::pread(fd_, static_cast<char *>(p) + done, n - done, off + static_cast<off_t>(done));
        if (r <= 0) throw std::runtime_error("state file: short I/O");
        done += static_cast<size_t>(r);
      }
      off += static_cast<off_t>(n);
    };
    one(st.res.data(), st.res.size() * 2);
    one(st.inj.data(), st.inj.size() * 4);
    one(st.ym.data(), st.ym.size() * 2);
  }
  int fd_ = -1, m_, H_, S_;
  size_t rec_ = 0;
};

// The BF16 routed sum of a capture's rows on the workers (exact forward), + the shared term -> bf16 ym.
struct ExactMoe {
  const q::ArShape *s = nullptr;
  std::vector<std::unique_ptr<q::ArWorker>> *workers = nullptr;
  std::vector<int> owner, local, n_local;
  void split(int experts) {
    const int W = static_cast<int>(workers->size());
    owner.assign(static_cast<size_t>(experts), 0);
    local.assign(static_cast<size_t>(experts), 0);
    n_local.assign(static_cast<size_t>(W), 0);
    for (int e = 0; e < experts; ++e) {
      owner[static_cast<size_t>(e)] = e * W / experts;
      local[static_cast<size_t>(e)] = n_local[static_cast<size_t>(owner[static_cast<size_t>(e)])]++;
    }
  }
  void setup(const q::ExpertSource &src) {
    std::vector<std::thread> th;
    for (size_t w = 0; w < workers->size(); ++w)
      th.emplace_back([&, w] {
        std::vector<int> mine;
        for (int e = 0; e < s->experts; ++e)
          if (owner[static_cast<size_t>(e)] == static_cast<int>(w)) mine.push_back(e);
        (*workers)[w]->setup(*s, mine, src);
        (*workers)[w]->exact(true);
      });
    for (auto &t : th) t.join();
  }
  void run(const q38::Model::CalibCapture &cap, std::vector<uint16_t> &ym) {
    const int H = s->hidden, K = s->topk, R = static_cast<int>(cap.sg.size());
    const size_t W = workers->size();
    std::vector<q::ArBatch> bs(W);
    for (size_t w = 0; w < W; ++w) {
      bs[w].rows = R;
      bs[w].x = cap.xm.data();
      bs[w].seg_start.assign(static_cast<size_t>(n_local[w]) + 1, 0);
      bs[w].row_asg.assign(static_cast<size_t>(R) * K, -1);
    }
    for (int r = 0; r < R; ++r)
      for (int kk = 0; kk < K; ++kk) {
        const int e = cap.ids[static_cast<size_t>(r) * K + kk];
        ++bs[static_cast<size_t>(owner[static_cast<size_t>(e)])].seg_start[static_cast<size_t>(local[static_cast<size_t>(e)]) + 1];
      }
    std::vector<std::vector<int32_t>> fill(W);
    for (size_t w = 0; w < W; ++w) {
      for (size_t i = 1; i < bs[w].seg_start.size(); ++i) bs[w].seg_start[i] += bs[w].seg_start[i - 1];
      bs[w].asg_row.assign(static_cast<size_t>(bs[w].seg_start.back()), 0);
      bs[w].asg_w.assign(bs[w].asg_row.size(), 0.f);
      fill[w].assign(bs[w].seg_start.begin(), bs[w].seg_start.end() - 1);
    }
    for (int r = 0; r < R; ++r)
      for (int kk = 0; kk < K; ++kk) {
        const int e = cap.ids[static_cast<size_t>(r) * K + kk];
        const size_t w = static_cast<size_t>(owner[static_cast<size_t>(e)]);
        const int a = fill[w][static_cast<size_t>(local[static_cast<size_t>(e)])]++;
        bs[w].asg_row[static_cast<size_t>(a)] = r;
        bs[w].asg_w[static_cast<size_t>(a)] = cap.topw[static_cast<size_t>(r) * K + kk];
        bs[w].row_asg[static_cast<size_t>(r) * K + kk] = a;
      }
    std::vector<std::vector<float>> parts(W, std::vector<float>(static_cast<size_t>(R) * H));
    std::vector<std::thread> th;
    for (size_t w = 0; w < W; ++w)
      th.emplace_back([&, w] {
        (*workers)[w]->begin(bs[w]);
        (*workers)[w]->forward(parts[w].data());
      });
    for (auto &t : th) t.join();
    ym.resize(static_cast<size_t>(R) * H);
    for (int r = 0; r < R; ++r) {
      const float sg = cap.sg[static_cast<size_t>(r)];
      for (int n = 0; n < H; ++n) {
        const size_t at = static_cast<size_t>(r) * H + n;
        float sum = 0.f;
        for (size_t w = 0; w < W; ++w) sum += parts[w][at];
        ym[at] = q::ar_bf16_bits(sum + sg * cap.sy[at]);
      }
    }
  }
};

}  // namespace

// ---------------------------------------------------------------------------------------------------
int cmd_selftest_calib(const Args &a) {
  const std::string dir = a.str("model", home() + "/models/qwen38-w4a16");
  const auto cs = q::read_calib(a.str("calib"));
  const int m = cs.seqlen;
  int fails = 0;
  auto expect = [&](bool ok, const std::string &w) {
    std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", w.c_str());
    fails += !ok;
  };
  // (1) blob packing
  {
    const auto ck = m::Checkpoint::open(dir);
    const q38::Config cfg = config_of(dir);
    const q38::ExpertLayout lay = q38::ExpertLayout::of(cfg);
    const int H = cfg.hidden, I = cfg.moe_inter;
    bool same = true;
    for (const auto [l, e] : {std::pair{0, 0}, {7, 311}, {47, 511}}) {
      std::vector<const m::TensorInfo *> ts = q38::expert_tensors(ck, l, e);
      const auto buf = m::bulk_read(ck, ts);
      std::vector<uint8_t> ref(lay.stride), mine(lay.stride);
      q38::pack_expert(lay, ck, buf, l, e, ref.data());
      q::ExpertQuant x;
      x.gu_q.resize(static_cast<size_t>(2) * I * H);
      x.gu_s.resize(static_cast<size_t>(2) * I * (H / 128));
      x.dn_q.resize(static_cast<size_t>(H) * I);
      x.dn_s.resize(static_cast<size_t>(H) * (I / 128));
      auto t = [&](const char *proj, const char *part) -> const m::TensorInfo & { return ck.at(q38::expert_name(l, e, proj, part)); };
      q::gptq_unpack(buf.as<uint32_t>(t("gate_proj", "qweight")), buf.as<uint16_t>(t("gate_proj", "scales")), I, H, x.gu_q.data(), x.gu_s.data());
      q::gptq_unpack(buf.as<uint32_t>(t("up_proj", "qweight")), buf.as<uint16_t>(t("up_proj", "scales")), I, H,
                     x.gu_q.data() + static_cast<size_t>(I) * H, x.gu_s.data() + static_cast<size_t>(I) * (H / 128));
      q::gptq_unpack(buf.as<uint32_t>(t("down_proj", "qweight")), buf.as<uint16_t>(t("down_proj", "scales")), H, I, x.dn_q.data(), x.dn_s.data());
      pack_full_blob(lay, x, mine.data());
      same = same && std::memcmp(ref.data(), mine.data(), lay.bytes) == 0;
    }
    expect(same, "blobs from codes == pack_expert (3 experts, full blobs)");
  }
  // (2) calib_layer chain == prefill
  const auto loader = l0::Loader::open();
  Fabric fab(loader, {0, 1});
  q38::ModelOptions mo;
  mo.context = (m + 63) / 64 * 64;
  mo.prefill_chunk = (m + 3) / 4 * 4;
  mo.graphs = false;
  mo.kv = k::QsaKv::bf16;
  mo.hot = a.has("hot") ? a.num("hot", 0) : -1;
  const std::string mtp = a.str("mtp", "");  // also gate calib_mtp against the drafter's prefill pass
  mo.mtp_dir = mtp;
  q38::Model model(loader, fab, dir, mo);
  std::printf("model loaded: %.1f s (hot %lld / card)\n", model.load_stats().total_s, static_cast<long long>(model.load_stats().hot[0]));
  const int32_t *tok = cs.sample(0);
  const int32_t next_after = mtp.empty() ? -1 : tok[0];  // any token: the drafter's last row
  q38::Model::PfTrace tr;
  tr.rows = 64;
  tr.points = {"xm", "mlp_mix.res"};
  if (!mtp.empty()) tr.points.insert(tr.points.end(), {"MTP.xm", "MTP.mlp_mix.res", "MTP.moe.ym"});
  model.trace_prefill(&tr);
  model.use_slot(0);
  model.begin_sequence();
  model.prefill(tok, m, {}, next_after);
  model.trace_prefill(nullptr);
  std::map<std::string, size_t> at;
  for (size_t i = 0; i < tr.names.size(); ++i) at[tr.names[i]] = i;
  const int L = model.config().layers, H = model.config().hidden, W = 4 * H;
  q38::Model::CalibState st, nx;
  int bad_layers = 0;
  for (int l = 0; l < L; ++l) {
    q38::Model::CalibCapture cap;
    model.calib_layer(tok, m, l, l ? &st : nullptr, &nx, &cap);
    const auto &tx = tr.data.at(at.at("L" + std::to_string(l) + ".xm")), &trs = tr.data.at(at.at("L" + std::to_string(l) + ".mlp_mix.res"));
    const bool ok = std::memcmp(tx.data(), cap.xm.data(), tx.size() * 2) == 0 && std::memcmp(trs.data(), cap.res.data(), trs.size() * 2) == 0;
    if (!ok) {
      size_t dx = 0, dr = 0;
      for (size_t i = 0; i < tx.size(); ++i) dx += tx[i] != cap.xm[i];
      for (size_t i = 0; i < trs.size(); ++i) dr += trs[i] != cap.res[i];
      if (bad_layers < 6) std::printf("  layer %d: xm %zu / %zu differ, res %zu / %zu differ\n", l, dx, tx.size(), dr, trs.size());
      ++bad_layers;
    }
    std::swap(st, nx);
    (void)W;
  }
  expect(bad_layers == 0, "calib_layer chain (host state) == prefill: xm and residual of every layer, 64 rows (" +
                              std::to_string(L - bad_layers) + " / " + std::to_string(L) + " layers bit-identical)");
  // (3): the drafter's layer on the chain's last state == the drafter's many-row pass in prefill
  if (!mtp.empty()) {
    std::vector<int32_t> next(static_cast<size_t>(m));
    for (int r = 0; r < m; ++r) next[static_cast<size_t>(r)] = r + 1 < m ? tok[r + 1] : next_after;
    q38::Model::CalibCapture cap;
    q38::Model::CalibState out;
    model.calib_mtp(tok, m, st, next.data(), &out, &cap);
    auto same = [&](const char *point, const std::vector<uint16_t> &mine) {
      const auto &ref = tr.data.at(at.at(point));
      size_t d = 0;
      for (size_t i = 0; i < ref.size(); ++i) d += ref[i] != mine[i];
      if (d) std::printf("  %s: %zu / %zu differ\n", point, d, ref.size());
      return d == 0;
    };
    const bool ok_x = same("MTP.xm", cap.xm), ok_r = same("MTP.mlp_mix.res", cap.res), ok_y = same("MTP.moe.ym", out.ym);
    expect(ok_x && ok_r && ok_y, "calib_mtp on the chain's state == the drafter's prefill pass: xm, residual, MoE output, 64 rows");
  }
  std::printf("%s: %d failure(s)\n", fails ? "FAIL" : "PASS", fails);
  return fails ? 1 : 0;
}

// ---------------------------------------------------------------------------------------------------
int cmd_tune(const Args &a) {
  const std::string src = a.str("src", home() + "/models/Qwen3.8-Flash-Next-Uncensored");
  const std::string out = a.str("out"), work = a.str("work"), ple = a.str("ple");
  if (out.empty() || work.empty() || ple.empty()) throw std::runtime_error("--out, --work and --ple are required");
  std::filesystem::create_directories(out + "/native");
  std::filesystem::create_directories(work);
  const auto cs = q::read_calib(a.str("calib"));
  const int N = std::min(cs.samples(), static_cast<int>(a.num("samples", cs.samples())));
  const int m = cs.seqlen, eval = static_cast<int>(a.num("eval", 16));
  if (eval >= N) throw std::runtime_error("--eval must be < samples");
  // Which calibration samples a run uses (pick[i] = the file's sample for state i; the last `eval` are
  // held out). By default the held-out samples are an even stride over the WHOLE file, so
  // every source is in them (the file is grouped by source), and they are the same for any --samples;
  // training takes an even stride over the rest. Before (and --holdout-spread 0): an even
  // stride over the file with the last `eval` picked held out, i.e. all from its last source. The
  // choice is recorded in WORK/samples ("spread|stride N eval"); a WORK without it is the old stride.
  std::vector<int> pick;
  {
    const std::string pick_file = work + "/samples";
    bool spread = a.num("holdout-spread", 1) != 0;
    if (a.has("resume")) {
      std::ifstream pf(pick_file);
      std::string kind;
      int n = -1, e = -1;
      const bool rec = static_cast<bool>(pf >> kind >> n >> e);
      const bool was = rec && kind == "spread";
      if (rec && (n != N || e != eval)) throw std::runtime_error("--resume: " + work + " used " + std::to_string(n) + " samples / " + std::to_string(e) + " held out");
      if (a.has("holdout-spread") && was != spread) throw std::runtime_error("--resume: " + work + " was baked with --holdout-spread " + std::to_string(was));
      spread = was;
    } else {
      std::ofstream(pick_file) << (spread ? "spread" : "stride") << ' ' << N << ' ' << eval << '\n';
    }
    pick = sample_picks(cs.samples(), N, eval, spread);
  }
  auto tok = [&](int i) { return cs.sample(pick[static_cast<size_t>(i)]); };
  const int threads = static_cast<int>(std::thread::hardware_concurrency());
  q::ArConfig cfg;
  cfg.iters = static_cast<int>(a.num("iters", 200));
  cfg.log_every = static_cast<int>(a.num("log-every", 50));
  cfg.floor = true;
  const bool rotate = a.num("rotate", 0) != 0;  // experts in a block-Hadamard basis (x, W_gu 128; h, W_dn 64)
  const auto ck = m::Checkpoint::open(src);
  const q38::Config mc = config_of(src);
  q::ArShape s;
  s.hidden = mc.hidden, s.inter = mc.moe_inter, s.experts = mc.experts, s.topk = mc.topk;
  const int H = s.hidden, S = 4, K = s.topk, L = mc.layers;

  // workers first (their VRAM before the models' chunk buffers)
  auto workers = make_workers(a);
  const auto loader = l0::Loader::open();
  Fabric fab(loader, {0, 1});
  q38::ModelOptions base;
  base.context = (m + 63) / 64 * 64;
  base.prefill_chunk = (m + 3) / 4 * 4;
  base.graphs = false;
  base.kv = k::QsaKv::bf16;
  base.calib = true;
  base.prefill_cold_stage = 0;
  q38::ModelOptions fo = base, qo = base;
  // the targets: every plan matrix in bf16 except the GDN qkv(z) projection (its kernel is int8-only;
  // it stays int8 on both paths, so the tuning does not see its error)
  fo.bf16_tensors = {"self_attn.", "linear_attn.out_proj", "shared_expert", "hyper_connection", "ple.", "lm_head", "embed", "mlp.gate"};
  fo.prefill_moe_a8 = false;
  qo.ple_dir = ple;  // the quantised path: int8 PLE rows, int8 trunk (as baked)
  qo.prefill_moe_a8 = true;
  qo.moe_rotate = rotate;
  auto t0 = clk::now();
  q38::Model fp(loader, fab, src, fo);
  q38::Model qm(loader, fab, src, qo);
  std::printf("models: %.1f s (targets: %d plan matrices bf16; quantised path int8)\n", secs(t0), fp.load_stats().bf16_overrides);
  fp.calib_zero_blobs();

  // resume
  int l0 = 0, l1 = L;
  if (a.has("layers")) {
    const std::string r = a.str("layers");
    const size_t d = r.find('-');
    l0 = std::stoi(r.substr(0, d));
    l1 = (d == std::string::npos ? l0 : std::stoi(r.substr(d + 1))) + 1;
  }
  if (rotate && a.has("compare-rotate")) throw std::runtime_error("--compare-rotate compares against the plain basis: not with --rotate");
  const std::string prog = work + "/progress", scheme_file = work + "/scheme";
  if (a.has("resume")) {
    std::ifstream f(prog);
    int done = -1;
    if (!(f >> done)) throw std::runtime_error("--resume: no " + prog);
    l0 = done + 1;
    std::printf("resuming after layer %d\n", done);
    {  // a stop inside the fp pass or q pass 2 left that state file half rewritten
      std::ifstream bf(work + "/busy");
      std::string what;
      int bl = -1;
      if (bf >> what >> bl)
        throw std::runtime_error("--resume: the run stopped inside the " + what + " pass of layer " + std::to_string(bl) + " (WORK/busy): " + what +
                                 ".state is mixed; restart the bake");
    }
    std::ifstream sf(scheme_file);
    int was = -1;
    if (!(sf >> was) || was != static_cast<int>(rotate))
      throw std::runtime_error("--resume: " + work + " was baked with rotate " + std::to_string(was) + ", this run has " + std::to_string(rotate));
  } else {
    std::ofstream(scheme_file) << static_cast<int>(rotate) << '\n';
  }
  StateFile fps(work + "/fp.state", m, H, S, N, l0 == 0), qs(work + "/q.state", m, H, S, N, l0 == 0);

  // the layer's tuning data (RAM: ~11 GB at 512 x 2,048)
  const int64_t R = static_cast<int64_t>(N) * m;
  std::vector<uint16_t> X(static_cast<size_t>(R) * H), T(static_cast<size_t>(R) * H);
  std::vector<int32_t> IDS(static_cast<size_t>(R) * K);
  std::vector<float> WT(IDS.size()), G(static_cast<size_t>(R)), C(static_cast<size_t>(R));
  std::ofstream log(out + "/bake-log.tsv", std::ios::app);
  log << "layer\tsamples\titers\trotate\trtn_loss\ttuned_loss\tgain\tfloor_loss\tfp_s\tq1_s\ttune_s\tq2_s\ttotal_s\n";

  // bf16 rows -> bf16(H_blk rows) in place
  auto rot_rows = [&](uint16_t *v, size_t rows, int cols, int blk) {
    const int chunk = 4096;
    const int nchunks = static_cast<int>((rows + chunk - 1) / chunk);
    pfor(nchunks, threads, [&](int ci) {
      std::vector<float> f(static_cast<size_t>(cols));
      for (size_t r = static_cast<size_t>(ci) * chunk; r < std::min(rows, static_cast<size_t>(ci + 1) * chunk); ++r) {
        uint16_t *row = v + r * cols;
        for (int i = 0; i < cols; ++i) f[static_cast<size_t>(i)] = q::ar_bf16(row[i]);
        q::ar_hadamard_blocks(f.data(), f.size(), blk);
        for (int i = 0; i < cols; ++i) row[i] = q::ar_bf16_bits(f[static_cast<size_t>(i)]);
      }
    });
  };
  q::ArShape st_shape = s;
  if (rotate) st_shape.rot_x = 128, st_shape.rot_h = 64;
  std::vector<uint16_t> gu_t, dn_t;  // the tuner's sources (rotated copies under --rotate)
  ExactMoe ex;
  ex.s = &s;
  ex.workers = &workers;
  ex.split(s.experts);
  std::vector<uint16_t> gu, dn;
  for (int l = l0; l < l1; ++l) {
    const auto tl = clk::now();
    read_layer_experts(ck, l, s, gu, dn);
    const q::ExpertSource es{gu.data(), dn.data()};
    // fp pass: the targets. WORK/fp-done records the layer fp.state holds after this pass
    // (the pass rewrites it in place), so a run stopped after it (during q pass 1 / the tuning) resumes
    // at this layer without redoing it (redoing it would read layer-l states as its input).
    auto t1 = clk::now();
    bool fp_ready = false;
    if (l == l0 && a.has("resume")) {
      std::ifstream ff(work + "/fp-done");
      int fl = -1;
      fp_ready = (ff >> fl) && fl == l;
    }
    if (fp_ready) {
      std::printf("layer %2d: fp.state already holds this layer's targets (WORK/fp-done): fp pass skipped\n", l);
    } else {
      std::ofstream(work + "/busy") << "fp " << l << '\n';  // a stop inside a pass leaves its state file mixed
      ex.setup(es);
      const q38::Model::CalibMoeHook hook = [&](int, const q38::Model::CalibCapture &cap, std::vector<uint16_t> &ym) { ex.run(cap, ym); };
      q38::Model::CalibState in, outst;
      for (int smp = 0; smp < N; ++smp) {
        if (l > 0) fps.read(smp, in);
        fp.calib_layer(tok(smp), m, l, l ? &in : nullptr, &outst, nullptr, &hook);
        fps.write(smp, outst);
      }
      ::sync();
      std::ofstream(work + "/fp-done") << l << '\n';
      std::filesystem::remove(work + "/busy");
    }
    const double fp_s = secs(t1);
    // q pass 1: the tuning rows
    t1 = clk::now();
    qm.calib_zero_blobs();
    double gmean = 0;
    {
      q38::Model::CalibState in, tgt;
      q38::Model::CalibCapture cap;
      for (int smp = 0; smp < N; ++smp) {
        if (l > 0) qs.read(smp, in);
        qm.calib_layer(tok(smp), m, l, l ? &in : nullptr, nullptr, &cap);
        fps.read(smp, tgt);
        const size_t r0 = static_cast<size_t>(smp) * m;
        RowSink o{H, S, K, X.data() + r0 * H, T.data() + r0 * H, IDS.data() + r0 * K, WT.data() + r0 * K, G.data() + r0, C.data() + r0};
        tuning_rows(cap, tgt, m, o, &gmean);
      }
    }
    const double q1_s = secs(t1);
    // tune
    t1 = clk::now();
    q::LayerData d;
    d.samples = N, d.seqlen = m, d.x = X.data(), d.ids = IDS.data(), d.w = WT.data(), d.t = T.data(), d.g = G.data(), d.c = C.data();
    for (auto &w : workers) w->exact(false);
    std::vector<q::ExpertQuant> res;
    std::fprintf(stderr, "layer %d: tuning (%d samples, %d held out, mean G %.3f%s)\n", l, N, eval, gmean / static_cast<double>(R), rotate ? ", rotated" : "");
    q::ExpertSource ts_src = es;
    if (rotate) {  // the routed experts' input and weights in the rotated basis
      rot_rows(X.data(), static_cast<size_t>(R), H, 128);
      gu_t = gu;
      dn_t = dn;
      rot_rows(gu_t.data(), static_cast<size_t>(s.experts) * 2 * s.inter, H, 128);
      rot_rows(dn_t.data(), static_cast<size_t>(s.experts) * H, s.inter, 64);
      ts_src = q::ExpertSource{gu_t.data(), dn_t.data()};
    }
    const auto st = q::tune_layer(st_shape, cfg, d, ts_src, workers, eval, res);
    const double tune_s = secs(t1);
    if (a.has("compare-rotate")) {  // the same layer in a block-Hadamard basis (x, W_gu: 128; h, W_dn: 64)
      const auto tr = clk::now();
      auto rot_bf16 = [&](const uint16_t *src, size_t rows, int cols, int blk, std::vector<uint16_t> &dst) {
        dst.resize(rows * static_cast<size_t>(cols));
        pfor(static_cast<int>(std::min<size_t>(rows, 1u << 30)), threads, [&](int r) {
          thread_local std::vector<float> v;
          v.resize(static_cast<size_t>(cols));
          for (int i = 0; i < cols; ++i) v[static_cast<size_t>(i)] = q::ar_bf16(src[static_cast<size_t>(r) * cols + i]);
          q::ar_hadamard_blocks(v.data(), v.size(), blk);
          for (int i = 0; i < cols; ++i) dst[static_cast<size_t>(r) * cols + i] = q::ar_bf16_bits(v[static_cast<size_t>(i)]);
        });
      };
      std::vector<uint16_t> Xr, gur, dnr;
      rot_bf16(X.data(), static_cast<size_t>(R), H, 128, Xr);
      rot_bf16(gu.data(), static_cast<size_t>(s.experts) * 2 * s.inter, H, 128, gur);
      rot_bf16(dn.data(), static_cast<size_t>(s.experts) * H, s.inter, 64, dnr);
      q::ArShape sr = s;
      sr.rot_x = 128, sr.rot_h = 64;
      q::LayerData dr = d;
      dr.x = Xr.data();
      std::vector<q::ExpertQuant> rr;
      const auto str = q::tune_layer(sr, cfg, dr, q::ExpertSource{gur.data(), dnr.data()}, workers, eval, rr);
      std::printf("layer %2d ROTATE: held-out block loss RTN %.4e tuned %.4e  |  plain RTN %.4e tuned %.4e  |  rotated vs plain: RTN %+.2f%%, tuned %+.2f%%  (%.0f s)\n", l,
                  str.rtn_loss, str.tuned_loss, st.rtn_loss, st.tuned_loss, 100.0 * (str.rtn_loss / st.rtn_loss - 1.0),
                  100.0 * (str.tuned_loss / st.tuned_loss - 1.0), secs(tr));
      std::fflush(stdout);
    }
    // write: the export + the per-card blobs, then load them into the quantised path
    std::vector<std::vector<uint8_t>> store;
    const auto ts = q::expert_export_tensors(s, "model.language_model.", l, res, store);
    char name[96];
    std::snprintf(name, sizeof name, "/experts-L%02d.safetensors", l);
    q::write_safetensors(out + name, ts,
                         {{"format", "pt"},
                          {"xenithium_scheme", std::string(cfg.iters ? "w4g128-sym-autoround-a8" : "w4g128-sym-rtn") + (rotate ? "-hadamard-x128-h64" : "")},
                          {"iters", std::to_string(cfg.iters)}});
    std::vector<std::vector<uint8_t>> regions;
    pack_tp_regions(mc, res, regions, threads);
    {
      std::snprintf(name, sizeof name, "/native/experts-L%02d.tp2", l);
      const std::string p = out + name + std::string(".part");
      std::ofstream f(p, std::ios::binary);
      for (const auto &rg : regions) f.write(reinterpret_cast<const char *>(rg.data()), static_cast<std::streamsize>(rg.size()));
      f.close();
      if (!f) throw std::runtime_error("write failed: " + p);
      std::filesystem::rename(p, out + name);
    }
    for (int c = 0; c < 2; ++c) qm.calib_set_blobs(c, regions[static_cast<size_t>(c)].data(), regions[static_cast<size_t>(c)].size());
    // q pass 2: the quantised path's state after the tuned layer
    t1 = clk::now();
    {
      std::ofstream(work + "/busy") << "q " << l << '\n';
      q38::Model::CalibState in, outst;
      for (int smp = 0; smp < N; ++smp) {
        if (l > 0) qs.read(smp, in);
        qm.calib_layer(tok(smp), m, l, l ? &in : nullptr, &outst, nullptr);
        qs.write(smp, outst);
      }
    }
    const double q2_s = secs(t1), tot = secs(tl);
    const double red = st.rtn_loss - st.floor_loss, redt = st.tuned_loss - st.floor_loss;
    std::printf("layer %2d: held-out block loss RTN %.4e tuned %.4e (%+.2f%%)  floor %.4e: quant part RTN %.4e tuned %.4e (%+.2f%%)  fp %.0f s  q1 %.0f s  tune %.0f s  q2 %.0f s  total %.0f s\n",
                l, st.rtn_loss, st.tuned_loss, 100.0 * (st.tuned_loss / st.rtn_loss - 1.0), st.floor_loss, red, redt, 100.0 * (redt / red - 1.0), fp_s, q1_s,
                tune_s, q2_s, tot);
    std::fflush(stdout);
    log << l << '\t' << N << '\t' << cfg.iters << '\t' << rotate << '\t' << st.rtn_loss << '\t' << st.tuned_loss << '\t' << (st.tuned_loss / st.rtn_loss - 1.0) << '\t'
        << st.floor_loss << '\t' << fp_s
        << '\t' << q1_s << '\t' << tune_s << '\t' << q2_s << '\t' << tot << '\n';
    log.flush();
    ::sync();
    std::ofstream(prog) << l << '\n';
    std::filesystem::remove(work + "/busy");
  }
  return 0;
}


// ---------------------------------------------------------------------------------------------------
// xenithium-bake mtp-tune: AutoRound of the MTP drafter's routed experts on the quantised
// target's hiddens.
// xenithium-bake mtp-tune --src BF16_DIR --mtp DRAFTER_DIR --work WORK --out DIR
// [--calib WORK/calib.xcb] [--samples N] [--iters 200] [--eval 16] [--gpus N]
// The WORK's sample picks and held-out set come from WORK/samples (cmd_tune); its held-out states are the
// drafter's held-out samples too.
// DRAFTER_DIR: `xenithium-bake mtp`'s output (RTN experts; the dense trunk, draft head, config).
// WORK: a finished `tune` run's directory: fp.state / q.state hold every sample's state after the
// target's last layer (the BF16 targets' and the quantised path's), the inputs here. Per sample (m - 1
// rows: row r drafts the token after r + 1, the calibration's own next token):
// targets the BF16 drafter (dense trunk bf16, exact BF16 experts on the tuner GPUs via the hook) on
// the BF16 target's state -> the state after the drafter's MoE
// inputs the drafter as served (W8 trunk, W4A8 experts) on the quantised target's state -> its MoE
// input, routing, shared term (the capture)
// then one tune_layer (the block-output MSE, as `tune`), and DIR = DRAFTER_DIR with the tuned experts
// (mtp-experts.safetensors; the other files hard-linked). The engine's own held-out block loss is
// measured before (the loaded RTN experts) and after (the tuned ones, uploaded into the drafter).
namespace {

// The engine's block output loss: mean over rows x streams x hidden of (out_q - out_fp)^2, out =
// bf16(res + ym * 2 sigmoid(inj / S)) per stream (tuning_rows' block output).
double block_loss(const q38::Model::CalibState &qs, const q38::Model::CalibState &fs, int m, int H, int S) {
  double e = 0;
  for (int r = 0; r < m; ++r)
    for (int s = 0; s < S; ++s) {
      const float gq = 2.f * sig(q::ar_round_bf16(qs.inj[static_cast<size_t>(r) * S + s]) / static_cast<float>(S));
      const float gf = 2.f * sig(q::ar_round_bf16(fs.inj[static_cast<size_t>(r) * S + s]) / static_cast<float>(S));
      for (int n = 0; n < H; ++n) {
        const size_t at = (static_cast<size_t>(r) * S + s) * H + n, y = static_cast<size_t>(r) * H + n;
        const double oq = q::ar_round_bf16(q::ar_bf16(qs.res[at]) + q::ar_bf16(qs.ym[y]) * gq);
        const double of = q::ar_round_bf16(q::ar_bf16(fs.res[at]) + q::ar_bf16(fs.ym[y]) * gf);
        e += (oq - of) * (oq - of);
      }
    }
  return e / (static_cast<double>(m) * S * H);
}

}  // namespace

int cmd_mtp_tune(const Args &a) {
  // WORK from `xenithium-bake mtp-states` (the served target's states after its last layer, fp.state
  // = q.state): the drafter as served on W4A16 experts (no W4A8 activations), the target's own PLE table, the BF16 experts
  // from --bf16-experts DIR (fused or per expert).
  const bool served = a.num("served", 0) != 0;
  const std::string src = a.str("src", home() + (served ? "/models/qwen38-w4a16" : "/models/Qwen3.8-Flash-Next-Uncensored"));
  const std::string mtp = a.str("mtp"), work = a.str("work"), out = a.str("out");
  if (mtp.empty() || work.empty() || out.empty()) throw std::runtime_error("--mtp, --work and --out are required");
  if ((std::filesystem::exists(out) && std::filesystem::equivalent(mtp, out)) || std::filesystem::exists(out + "/mtp-experts.safetensors"))
    throw std::runtime_error("--out must be a new directory (not --mtp, no mtp-experts.safetensors yet)");
  const auto cs = q::read_calib(a.str("calib", work + "/calib.xcb"));
  const int m = cs.seqlen, mm = m - 1;
  const q38::Config mc = config_of(src);
  q::ArShape s;
  s.hidden = mc.hidden, s.inter = mc.moe_inter, s.experts = mc.experts, s.topk = mc.topk;
  const int H = s.hidden, S = 4, K = s.topk, L = mc.layers;
  {
    std::ifstream f(work + "/progress");
    int done = -1;
    if (!(f >> done) || done != L - 1) throw std::runtime_error(work + ": the target's tune has not finished (progress " + std::to_string(done) + ", want " + std::to_string(L - 1) + ")");
  }
  // the bake's sample set: N states, picked from the calibration file by an even stride (cmd_tune)
  const size_t rec = static_cast<size_t>(m) * (static_cast<size_t>(S) * H * 2 + static_cast<size_t>(S) * 4 + static_cast<size_t>(H) * 2);
  const uintmax_t fsz = std::filesystem::file_size(work + "/q.state");
  if (fsz % rec || std::filesystem::file_size(work + "/fp.state") != fsz) throw std::runtime_error(work + ": state files do not match the calibration's seqlen");
  const int N = static_cast<int>(fsz / rec);
  int ev_w = 16;
  const std::vector<int> wpick = work_picks(work, cs.samples(), N, static_cast<int>(a.num("eval", 16)), &ev_w);
  const int use = std::min(N, static_cast<int>(a.num("samples", N))), eval = static_cast<int>(a.num("eval", ev_w));
  if (eval >= use || eval > N) throw std::runtime_error("--eval must be < samples");
  // state index per used sample: the last `eval` = the WORK's held-out states (its last ones), the rest an
  // even stride over its training states
  std::vector<int> sel(static_cast<size_t>(use));
  for (int i = 0; i < use - eval; ++i) sel[static_cast<size_t>(i)] = static_cast<int>(static_cast<int64_t>(i) * (N - eval) / (use - eval));
  for (int j = 0; j < eval; ++j) sel[static_cast<size_t>(use - eval + j)] = N - eval + j;
  auto tok = [&](int state) { return cs.sample(wpick[static_cast<size_t>(state)]); };
  bool rotate = false;
  {
    std::ifstream cf(mtp + "/config.json");
    std::stringstream ss;
    ss << cf.rdbuf();
    rotate = ss.str().find("\"xenithium_rotation\"") != std::string::npos;
  }
  const int threads = static_cast<int>(std::thread::hardware_concurrency());
  q::ArConfig cfg;
  cfg.iters = static_cast<int>(a.num("iters", 200));
  cfg.log_every = static_cast<int>(a.num("log-every", 50));
  cfg.floor = true;
  const auto ck = m::Checkpoint::open(src);

  auto workers = make_workers(a);
  const auto loader = l0::Loader::open();
  Fabric fab(loader, {0, 1});
  q38::ModelOptions base;
  base.context = (mm + 63) / 64 * 64;
  base.prefill_chunk = (mm + 3) / 4 * 4;
  base.graphs = false;
  base.kv = k::QsaKv::bf16;
  base.calib = true;
  base.prefill_cold_stage = 0;
  base.mtp_dir = mtp;
  q38::ModelOptions fo = base, qo = base;
  fo.bf16_tensors = {"self_attn.", "linear_attn.out_proj", "shared_expert", "hyper_connection", "ple.", "lm_head", "embed", "mlp.gate"};
  fo.mtp_bf16_tensors = {"mtp.fc_", "self_attn.", "shared_expert", "hyper_connection", "mlp.gate"};
  fo.prefill_moe_a8 = false;
  if (served) {  // the drafter as served on the W4A16 target (calibration mode: no target experts needed)
    qo.prefill_moe_a8 = false;
  } else {
    qo.ple_dir = work + "/ple";
    qo.prefill_moe_a8 = true;
  }
  auto t0 = clk::now();
  q38::Model fp(loader, fab, src, fo);
  std::fprintf(stderr, "mtp-tune: BF16-drafter model loaded (%.1f s)\n", secs(t0));
  auto qm = std::make_unique<q38::Model>(loader, fab, src, qo);
  std::fprintf(stderr, "mtp-tune: served model loaded (%.1f s)\n", secs(t0));
  std::printf("models: %.1f s (BF16 drafter: %d plan matrices bf16; drafter %s, experts %s)\n", secs(t0), fp.load_stats().mtp_bf16_overrides, mtp.c_str(),
              rotate ? "rotated" : "plain");

  std::vector<uint16_t> gu, dn;
  if (a.has("bf16-experts")) read_mtp_experts_any(m::Checkpoint::open(a.str("bf16-experts")), "mtp.layers.0.mlp.experts.", s, gu, dn);
  else read_fused_experts(ck, "mtp.layers.0.mlp.experts.", s, gu, dn);
  const q::ExpertSource es{gu.data(), dn.data()};
  ExactMoe ex;
  ex.s = &s;
  ex.workers = &workers;
  ex.split(s.experts);
  ex.setup(es);
  std::fprintf(stderr, "mtp-tune: exact BF16 experts on the tuner GPUs (%.1f s)\n", secs(t0));
  const q38::Model::CalibMoeHook hook = [&](int, const q38::Model::CalibCapture &cap, std::vector<uint16_t> &ym) { ex.run(cap, ym); };

  // the passes: targets + the tuning rows; the held-out samples' states kept for the engine's loss
  const int64_t R = static_cast<int64_t>(use) * mm;
  std::vector<uint16_t> X(static_cast<size_t>(R) * H), T(static_cast<size_t>(R) * H);
  std::vector<int32_t> IDS(static_cast<size_t>(R) * K);
  std::vector<float> WT(IDS.size()), G(static_cast<size_t>(R)), C(static_cast<size_t>(R));
  auto fps = std::make_unique<StateFile>(work + "/fp.state", m, H, S, N, false);
  auto qss = std::make_unique<StateFile>(work + "/q.state", m, H, S, N, false);
  std::vector<q38::Model::CalibState> ev_fp(static_cast<size_t>(eval));
  double gmean = 0, eng_rtn = 0;
  auto t1 = clk::now();
  {
    q38::Model::CalibState in, qout;
    q38::Model::CalibCapture cap;
    for (int i = 0; i < use; ++i) {
      const int st = sel[static_cast<size_t>(i)];
      const int32_t *t = tok(st);
      q38::Model::CalibState tgt;
      const bool held = i >= use - eval;
      fps->read(st, in);
      fp.calib_mtp(t, mm, in, t + 1, &tgt, nullptr, &hook);
      qss->read(st, in);
      qm->calib_mtp(t, mm, in, t + 1, held ? &qout : nullptr, &cap);
      const size_t r0 = static_cast<size_t>(i) * mm;
      RowSink o{H, S, K, X.data() + r0 * H, T.data() + r0 * H, IDS.data() + r0 * K, WT.data() + r0 * K, G.data() + r0, C.data() + r0};
      tuning_rows(cap, tgt, mm, o, &gmean);
      if (held) {
        eng_rtn += block_loss(qout, tgt, mm, H, S) / eval;
        ev_fp[static_cast<size_t>(i - (use - eval))] = std::move(tgt);
      }
    }
  }
  const double pass_s = secs(t1);

  // tune
  t1 = clk::now();
  q::LayerData d;
  d.samples = use, d.seqlen = mm, d.x = X.data(), d.ids = IDS.data(), d.w = WT.data(), d.t = T.data(), d.g = G.data(), d.c = C.data();
  for (auto &w : workers) w->exact(false);
  q::ArShape st_shape = s;
  std::vector<uint16_t> gu_t, dn_t;
  q::ExpertSource ts_src = es;
  if (rotate) {  // the routed experts' input and weights in the drafter's block-Hadamard basis
    auto rot_rows = [&](uint16_t *v, size_t rows, int cols, int blk) {
      const int chunk = 4096;
      pfor(static_cast<int>((rows + chunk - 1) / chunk), threads, [&](int ci) {
        std::vector<float> f(static_cast<size_t>(cols));
        for (size_t r = static_cast<size_t>(ci) * chunk; r < std::min(rows, static_cast<size_t>(ci + 1) * chunk); ++r) {
          uint16_t *row = v + r * cols;
          for (int i = 0; i < cols; ++i) f[static_cast<size_t>(i)] = q::ar_bf16(row[i]);
          q::ar_hadamard_blocks(f.data(), f.size(), blk);
          for (int i = 0; i < cols; ++i) row[i] = q::ar_bf16_bits(f[static_cast<size_t>(i)]);
        }
      });
    };
    st_shape.rot_x = 128, st_shape.rot_h = 64;
    rot_rows(X.data(), static_cast<size_t>(R), H, 128);
    gu_t = gu;
    dn_t = dn;
    rot_rows(gu_t.data(), static_cast<size_t>(s.experts) * 2 * s.inter, H, 128);
    rot_rows(dn_t.data(), static_cast<size_t>(s.experts) * H, s.inter, 64);
    ts_src = q::ExpertSource{gu_t.data(), dn_t.data()};
  }
  std::fprintf(stderr, "drafter: tuning (%d samples x %d rows, %d held out, mean G %.3f%s)\n", use, mm, eval, gmean / static_cast<double>(R), rotate ? ", rotated" : "");
  std::vector<q::ExpertQuant> res;
  const auto stt = q::tune_layer(st_shape, cfg, d, ts_src, workers, eval, res);
  const double tune_s = secs(t1);
  // the engine's held-out loss with the tuned experts
  std::vector<std::vector<uint8_t>> regions;
  pack_tp_regions(mc, res, regions, threads);

  for (int c = 0; c < 2; ++c) qm->calib_set_mtp_blobs(c, regions[static_cast<size_t>(c)].data(), regions[static_cast<size_t>(c)].size());
  double eng_tuned = 0;
  {
    q38::Model::CalibState in, qout;
    for (int i = use - eval; i < use; ++i) {
      const int st = sel[static_cast<size_t>(i)];
      qss->read(st, in);
      qm->calib_mtp(tok(st), mm, in, tok(st) + 1, &qout, nullptr);
      eng_tuned += block_loss(qout, ev_fp[static_cast<size_t>(i - (use - eval))], mm, H, S) / eval;
    }
  }
  // write: DIR = the drafter with the tuned experts
  std::filesystem::create_directories(out);
  for (const auto &e : std::filesystem::directory_iterator(mtp)) {
    const std::string name = e.path().filename().string();
    if (name == "mtp-experts.safetensors" || !e.is_regular_file()) continue;
    std::error_code ec;
    std::filesystem::create_hard_link(e.path(), out + "/" + name, ec);
    if (ec) std::filesystem::copy_file(e.path(), out + "/" + name, std::filesystem::copy_options::overwrite_existing);
  }
  {
    std::vector<std::vector<uint8_t>> store;
    const auto ts = q::expert_export_tensors(s, "mtp.", 0, res, store);
    q::write_safetensors(out + "/mtp-experts.safetensors", ts,
                         {{"format", "pt"},
                          {"xenithium_scheme", std::string(cfg.iters ? (served ? "w4g128-sym-autoround-a16" : "w4g128-sym-autoround-a8") : "w4g128-sym-rtn") + (rotate ? "-hadamard-x128-h64" : "")},
                          {"iters", std::to_string(cfg.iters)},
                          {"calibration", served ? "the served target's hiddens, BF16 drafter targets (round 113)" : "the quantised target's hiddens (round 65)"}});
  }
  const double red = stt.rtn_loss - stt.floor_loss, redt = stt.tuned_loss - stt.floor_loss;
  std::printf("drafter: tuner held-out block loss RTN %.4e tuned %.4e (%+.2f%%)  floor %.4e: quant part RTN %.4e tuned %.4e (%+.2f%%)\n", stt.rtn_loss,
              stt.tuned_loss, 100.0 * (stt.tuned_loss / stt.rtn_loss - 1.0), stt.floor_loss, red, redt, 100.0 * (redt / red - 1.0));
  std::printf("drafter: engine held-out block loss (%d samples) loaded RTN %.4e tuned %.4e (%+.2f%%)  passes %.0f s  tune %.0f s  total %.0f s -> %s\n", eval,
              eng_rtn, eng_tuned, 100.0 * (eng_tuned / eng_rtn - 1.0), pass_s, tune_s, secs(t0), out.c_str());
  return 0;
}


// ---------------------------------------------------------------------------------------------------
// xenithium-bake mtp-states: a WORK directory for `mtp-tune --served 1` from a served checkpoint (a W4A16
// export: no BF16 target to tune against): every picked sample through the served model as a prefill call runs it
// (normal mode, its own experts; Model:calib_final_state), the state after the last layer -> WORK/q.state, fp.state a
// hard link of it (the drafter's targets then differ from its inputs only by the drafter itself), progress / samples as
// `tune` writes them, calib.xcb linked.
// xenithium-bake mtp-states --src SERVED_DIR --work WORK --calib FILE [--samples 512] [--eval 16]
int cmd_mtp_states(const Args &a) {
  const std::string src = a.str("src", home() + "/models/qwen38-w4a16"), work = a.str("work"), calib = a.str("calib", home() + "/models/.bake-work/calib.xcb");
  if (work.empty()) throw std::runtime_error("--work is required");
  std::filesystem::create_directories(work);
  const auto cs = q::read_calib(calib);
  const int m = cs.seqlen;
  const q38::Config mc = config_of(src);
  const int H = mc.hidden, S = 4, L = mc.layers;
  const int N = std::min(cs.samples(), static_cast<int>(a.num("samples", 512))), eval = static_cast<int>(a.num("eval", 16));
  const std::vector<int> pick = sample_picks(cs.samples(), N, eval, true);
  const auto loader = l0::Loader::open();
  Fabric fab(loader, {0, 1});
  q38::ModelOptions o;
  o.context = (m + 63) / 64 * 64;
  o.prefill_chunk = (m + 3) / 4 * 4;
  o.graphs = false;
  o.kv = k::QsaKv::bf16;
  o.prefill_moe_a8 = false;
  auto t0 = clk::now();
  q38::Model qm(loader, fab, src, o);
  std::fprintf(stderr, "mtp-states: %s loaded (%.1f s); %d samples x %d rows (%d held out) -> %s\n", src.c_str(), secs(t0), N, m, eval, work.c_str());
  {
    StateFile qs(work + "/q.state", m, H, S, N, true);
    q38::Model::CalibState st;
    t0 = clk::now();
    for (int i = 0; i < N; ++i) {
      qm.calib_final_state(cs.sample(pick[static_cast<size_t>(i)]), m, st);
      qs.write(i, st);
      if ((i + 1) % 64 == 0 || i + 1 == N) std::fprintf(stderr, "mtp-states: %d / %d (%.0f s)\n", i + 1, N, secs(t0));
    }
  }
  std::filesystem::remove(work + "/fp.state");
  std::filesystem::create_hard_link(work + "/q.state", work + "/fp.state");
  if (!std::filesystem::exists(work + "/calib.xcb")) std::filesystem::create_symlink(std::filesystem::absolute(calib), work + "/calib.xcb");
  std::ofstream(work + "/samples") << "spread " << N << ' ' << eval << '\n';
  std::ofstream(work + "/progress") << (L - 1) << '\n';
  return 0;
}

// ---------------------------------------------------------------------------------------------------
// xenithium-bake head-capture: the draft head's inputs (the drafter's final mixer output, pass
// 1) on the quantised target's states, for the head's calibration (head-calib, host only).
// xenithium-bake head-capture --src BF16_DIR --mtp DRAFTER_DIR --work WORK --out FILE
// [--samples 256] [--eval-samples 2]
// The drafter as served (W8 trunk, its experts, mtp-tune's when DRAFTER_DIR is its output) on WORK's
// q.state after the last layer: --samples training states (an even stride over WORK's training states)
// and --eval-samples of WORK's held-out states, m - 1 rows each.
// FILE: uint32 'XHI1', int32 hidden, int64 train rows, int64 eval rows, then bf16 [rows][hidden] (train
// rows first).
int cmd_head_capture(const Args &a) {
  const std::string src = a.str("src", home() + "/models/Qwen3.8-Flash-Next-Uncensored");
  const std::string mtp = a.str("mtp"), work = a.str("work"), outf = a.str("out");
  if (mtp.empty() || work.empty() || outf.empty()) throw std::runtime_error("--mtp, --work and --out are required");
  const auto cs = q::read_calib(a.str("calib", work + "/calib.xcb"));
  const int m = cs.seqlen, mm = m - 1;
  const q38::Config mc = config_of(src);
  const int H = mc.hidden, S = 4, L = mc.layers;
  {
    std::ifstream f(work + "/progress");
    int done = -1;
    if (!(f >> done) || done != L - 1) throw std::runtime_error(work + ": the target's tune has not finished");
  }
  const size_t rec = static_cast<size_t>(m) * (static_cast<size_t>(S) * H * 2 + static_cast<size_t>(S) * 4 + static_cast<size_t>(H) * 2);
  const uintmax_t fsz = std::filesystem::file_size(work + "/q.state");
  if (fsz % rec) throw std::runtime_error(work + ": q.state does not match the calibration's seqlen");
  const int N = static_cast<int>(fsz / rec);
  int ev_w = 16;
  const std::vector<int> wpick = work_picks(work, cs.samples(), N, 16, &ev_w);
  const int ntr = static_cast<int>(a.num("samples", 256)), nev = static_cast<int>(a.num("eval-samples", 2));
  if (ntr < 1 || ntr > N - ev_w || nev < 0 || nev > ev_w) throw std::runtime_error("--samples / --eval-samples out of range");
  std::vector<int> sel;
  for (int i = 0; i < ntr; ++i) sel.push_back(static_cast<int>(static_cast<int64_t>(i) * (N - ev_w) / ntr));
  for (int j = 0; j < nev; ++j) sel.push_back(N - ev_w + j);

  const auto loader = l0::Loader::open();
  Fabric fab(loader, {0, 1});
  q38::ModelOptions qo;
  qo.context = (mm + 63) / 64 * 64;
  qo.prefill_chunk = (mm + 3) / 4 * 4;
  qo.graphs = false;
  qo.kv = k::QsaKv::bf16;
  qo.calib = true;
  qo.prefill_cold_stage = 0;
  qo.mtp_dir = mtp;
  qo.ple_dir = work + "/ple";
  qo.prefill_moe_a8 = true;
  const auto t0 = clk::now();
  q38::Model qm(loader, fab, src, qo);
  std::printf("model + drafter %s: %.1f s\n", mtp.c_str(), secs(t0));
  std::ofstream f(outf, std::ios::binary);
  const uint32_t magic = 0x31494858u;  // 'XHI1'
  const int32_t hid = H;
  const int64_t tr_rows = static_cast<int64_t>(ntr) * mm, ev_rows = static_cast<int64_t>(nev) * mm;
  f.write(reinterpret_cast<const char *>(&magic), 4);
  f.write(reinterpret_cast<const char *>(&hid), 4);
  f.write(reinterpret_cast<const char *>(&tr_rows), 8);
  f.write(reinterpret_cast<const char *>(&ev_rows), 8);
  StateFile qss(work + "/q.state", m, H, S, N, false);
  q38::Model::CalibState in;
  std::vector<uint16_t> rows;
  for (size_t i = 0; i < sel.size(); ++i) {
    const int st = sel[i];
    const int32_t *t = cs.sample(wpick[static_cast<size_t>(st)]);
    qss.read(st, in);
    qm.calib_mtp(t, mm, in, t + 1, nullptr, nullptr, nullptr, &rows);
    f.write(reinterpret_cast<const char *>(rows.data()), static_cast<std::streamsize>(rows.size() * 2));
  }
  f.close();
  if (!f) throw std::runtime_error("write failed: " + outf);
  std::printf("head-capture: %lld train + %lld eval rows x %d -> %s (%.0f s)\n", static_cast<long long>(tr_rows), static_cast<long long>(ev_rows), H,
              outf.c_str(), secs(t0));
  return 0;
}


// ---------------------------------------------------------------------------------------------------
// xenithium-bake kld: the bake's quality gate on its own held-out samples.
// xenithium-bake kld --src BF16_DIR --work WORK --out DIR [--k 64] [--samples 16] [--rtn plain,rot]
// [--baked-chain 1 --baked MODEL_DIR] [--a16 1] [--dense-a8 1]
// The finished bake's WORK holds every sample's state after the last layer on both paths: fp.state =
// the BF16 model (BF16 trunk + exact BF16 experts), q.state = the baked model's prefill path (int8
// trunk, int8 PLE, the tuned W4A8 experts). Their heads give, per held-out sample (WORK's held-out
// states, spread over every source), m - 1 rows of prompt log-probs (row j predicts token j + 1):
// DIR/ref.xlp the BF16 model (lm_head in bf16)
// DIR/baked.xlp the baked model (lm_head int8, as served)
// and with --rtn, the same samples through a q-path chain from layer 0 with RTN experts instead of the
// tuned ones (the round-trip-to-nearest baseline, AutoRound's iters 0, signed /8 scales):
// DIR/rtn-plain.xlp RTN in the checkpoint's basis (what quantising the BF16 checkpoint online gives)
// DIR/rtn-rot.xlp RTN in the rotated basis (separates the rotation's gain from AutoRound's)
// --baked-chain 1 --baked MODEL_DIR adds the same chain with the bake's tuned blobs
// (MODEL_DIR/native/experts-LNN.tp2) -> DIR/chain-baked.xlp: baked.xlp's model recomputed from layer 0
// (on the W4A8 path it must equal baked.xlp, which checks the chain). --a16 1 runs every chain with bf16 MoE activations
// (W4A16, the decode path's numerics; the prefill W4A8 path is the default): files rtn-*-a16.xlp,
// chain-baked-a16.xlp.
// Compare with `xenithium-kl compare DIR/ref.xlp DIR/<test>.xlp` (KL on ref's top k + a tail bucket).
int cmd_kld(const Args &a) {
  const std::string src = a.str("src", home() + "/models/Qwen3.8-Flash-Next-Uncensored");
  const std::string work = a.str("work"), out = a.str("out");
  if (work.empty() || out.empty()) throw std::runtime_error("--work and --out are required");
  std::filesystem::create_directories(out);
  const auto cs = q::read_calib(a.str("calib", work + "/calib.xcb"));
  const int m = cs.seqlen, rows = m - 1, K = static_cast<int>(a.num("k", 64));
  const q38::Config mc = config_of(src);
  q::ArShape s;
  s.hidden = mc.hidden, s.inter = mc.moe_inter, s.experts = mc.experts, s.topk = mc.topk;
  const int H = s.hidden, S = 4, L = mc.layers;
  {
    std::ifstream f(work + "/progress");
    int done = -1;
    if (!(f >> done) || done != L - 1) throw std::runtime_error(work + ": the target's tune has not finished");
  }
  bool rotate = false;
  {
    std::ifstream sf(work + "/scheme");
    int r = 0;
    if (sf >> r) rotate = r != 0;
  }
  const size_t rec = static_cast<size_t>(m) * (static_cast<size_t>(S) * H * 2 + static_cast<size_t>(S) * 4 + static_cast<size_t>(H) * 2);
  const uintmax_t fsz = std::filesystem::file_size(work + "/q.state");
  if (fsz % rec) throw std::runtime_error(work + ": q.state does not match the calibration's seqlen");
  const int N = static_cast<int>(fsz / rec);
  int ev_w = 16;
  const std::vector<int> wpick = work_picks(work, cs.samples(), N, 16, &ev_w);
  const int ns = std::min(ev_w, static_cast<int>(a.num("samples", ev_w)));
  std::vector<int> sel;  // WORK's held-out states
  for (int j = 0; j < ns; ++j) sel.push_back(N - ev_w + j);
  const int threads = static_cast<int>(std::thread::hardware_concurrency());
  const std::string rtn = a.str("rtn", "");

  const auto loader = l0::Loader::open();
  Fabric fab(loader, {0, 1});
  q38::ModelOptions base;
  base.context = (m + 63) / 64 * 64;
  base.prefill_chunk = (m + 3) / 4 * 4;
  base.graphs = false;
  base.kv = k::QsaKv::bf16;
  base.calib = true;
  base.prefill_cold_stage = 0;
  q38::ModelOptions fo = base, qo = base;
  fo.bf16_tensors = {"self_attn.", "linear_attn.out_proj", "shared_expert", "hyper_connection", "ple.", "lm_head", "embed", "mlp.gate"};
  fo.prefill_moe_a8 = false;
  qo.ple_dir = work + "/ple";
  qo.prefill_moe_a8 = true;

  // log-probs of one state's rows through a model's head -> one XLP1 chunk
  auto lp_chunk = [&](q38::Model &md, const int32_t *t, const q38::Model::CalibState &st, std::vector<eval::LpRow> &chunk) {
    chunk.assign(static_cast<size_t>(rows), eval::LpRow{});
    md.calib_logits(t, rows, st, [&](int r0, int gr, const uint16_t *lg) {
      const int64_t V = mc.vocab;
      pfor(gr, threads, [&](int i) {
        std::vector<float> f(static_cast<size_t>(V));
        const uint16_t *row = lg + static_cast<size_t>(i) * V;
        for (int64_t v = 0; v < V; ++v) f[static_cast<size_t>(v)] = q::ar_bf16(row[v]);
        chunk[static_cast<size_t>(r0 + i)] = eval::top_logprobs(f.data(), V, K, t[r0 + i + 1]);
      });
    });
  };
  auto write = [&](const std::string &name, const eval::LpSet &set) {
    std::string err;
    if (!eval::write_lp(out + "/" + name, set, &err)) throw std::runtime_error("kld: " + err);
    std::printf("  -> %s/%s (%zu chunks x %d rows, k %d)\n", out.c_str(), name.c_str(), set.chunks.size(), rows, K);
    std::fflush(stdout);
  };
  auto t0 = clk::now();
  {  // the BF16 reference and the baked model from WORK's states
    eval::LpSet ref, baked;
    ref.k = baked.k = K;
    q38::Model::CalibState st;
    {
      q38::Model fp(loader, fab, src, fo);
      StateFile fps(work + "/fp.state", m, H, S, N, false);
      for (int st_i : sel) {
        fps.read(st_i, st);
        ref.chunks.emplace_back();
        lp_chunk(fp, cs.sample(wpick[static_cast<size_t>(st_i)]), st, ref.chunks.back());
      }
    }
    write("ref.xlp", ref);
    qo.moe_rotate = rotate;
    q38::Model qm(loader, fab, src, qo);
    StateFile qss(work + "/q.state", m, H, S, N, false);
    for (int st_i : sel) {
      qss.read(st_i, st);
      baked.chunks.emplace_back();
      lp_chunk(qm, cs.sample(wpick[static_cast<size_t>(st_i)]), st, baked.chunks.back());
    }
    write("baked.xlp", baked);
    std::printf("kld: ref + baked, %d samples (%.0f s)\n", ns, secs(t0));
  }
  // Chains from layer 0: RTN experts (--rtn), or the bake's own tuned blobs (--baked-chain).
  // --a16 1 runs every chain with bf16 MoE activations (W4A16: the decode path's numerics) instead of
  // the W4A8 prefill path; its files get a "-a16" suffix. --dense-a8 1 runs the chains with the
  // dense trunk on W8A8 (int8 activations, ModelOptions:prefill_dense_a8): suffix "-d8".
  const auto ck = m::Checkpoint::open(src);
  const bool a16 = a.num("a16", 0) != 0;
  const bool dense_a8 = a.num("dense-a8", 0) != 0;
  const std::string sfx = std::string(a16 ? "-a16" : "") + (dense_a8 ? "-d8" : "");
  std::vector<std::string> chains;
  for (const char *v : {"plain", "rot"})
    if (rtn.find(v) != std::string::npos) chains.push_back(std::string("rtn-") + v);
  const std::string baked_dir = a.str("baked", "");
  if (a.num("baked-chain", 0) != 0) {
    if (baked_dir.empty()) throw std::runtime_error("kld: --baked-chain needs --baked BAKED_MODEL_DIR (its native/experts-LNN.tp2)");
    chains.push_back("chain-baked");
  }
  for (const std::string &chain : chains) {
    const auto tv = clk::now();
    const bool from_bake = chain == "chain-baked";
    const bool rot = from_bake ? rotate : chain == "rtn-rot";
    const std::string variant = chain + sfx;
    qo.moe_rotate = rot;
    qo.prefill_moe_a8 = !a16;
    qo.prefill_dense_a8 = dense_a8;
    q38::Model qm(loader, fab, src, qo);
    std::vector<q38::Model::CalibState> st(sel.size()), nx(1);
    std::vector<uint16_t> gu, dn;
    std::vector<uint8_t> blob;
    for (int l = 0; l < L; ++l) {
      if (from_bake) {
        char n[48];
        std::snprintf(n, sizeof n, "/native/experts-L%02d.tp2", l);
        const std::string path = baked_dir + n;
        const uintmax_t bsz = std::filesystem::file_size(path);
        if (bsz % 2) throw std::runtime_error("kld: odd size " + path);
        blob.resize(static_cast<size_t>(bsz));
        std::ifstream f(path, std::ios::binary);
        if (!f.read(reinterpret_cast<char *>(blob.data()), static_cast<std::streamsize>(bsz))) throw std::runtime_error("kld: cannot read " + path);
        for (int c = 0; c < 2; ++c) qm.calib_set_blobs(c, blob.data() + static_cast<size_t>(c) * (bsz / 2), static_cast<size_t>(bsz / 2));
        for (size_t i = 0; i < sel.size(); ++i) {
          qm.calib_layer(cs.sample(wpick[static_cast<size_t>(sel[i])]), m, l, l ? &st[i] : nullptr, &nx[0], nullptr);
          std::swap(st[i], nx[0]);
        }
        if (l % 8 == 7) {
          std::printf("  %s: layer %d (%.0f s)\n", variant.c_str(), l, secs(tv));
          std::fflush(stdout);
        }
        continue;
      }
      read_layer_experts(ck, l, s, gu, dn);
      if (rot) {
        auto rot_rows = [&](uint16_t *v, size_t nrows, int cols, int blk) {
          pfor(static_cast<int>((nrows + 4095) / 4096), threads, [&](int ci) {
            std::vector<float> f(static_cast<size_t>(cols));
            for (size_t r = static_cast<size_t>(ci) * 4096; r < std::min(nrows, static_cast<size_t>(ci + 1) * 4096); ++r) {
              uint16_t *row = v + r * cols;
              for (int i = 0; i < cols; ++i) f[static_cast<size_t>(i)] = q::ar_bf16(row[i]);
              q::ar_hadamard_blocks(f.data(), f.size(), blk);
              for (int i = 0; i < cols; ++i) row[i] = q::ar_bf16_bits(f[static_cast<size_t>(i)]);
            }
          });
        };
        rot_rows(gu.data(), static_cast<size_t>(s.experts) * 2 * s.inter, H, 128);
        rot_rows(dn.data(), static_cast<size_t>(s.experts) * H, s.inter, 64);
      }
      std::vector<q::ExpertQuant> res(static_cast<size_t>(s.experts));
      const q::ExpertSource es{gu.data(), dn.data()};
      pfor(s.experts, threads, [&](int e) { q::rtn_expert(s, es, e, res[static_cast<size_t>(e)]); });
      std::vector<std::vector<uint8_t>> regions;
      pack_tp_regions(mc, res, regions, threads);
      for (int c = 0; c < 2; ++c) qm.calib_set_blobs(c, regions[static_cast<size_t>(c)].data(), regions[static_cast<size_t>(c)].size());
      for (size_t i = 0; i < sel.size(); ++i) {
        qm.calib_layer(cs.sample(wpick[static_cast<size_t>(sel[i])]), m, l, l ? &st[i] : nullptr, &nx[0], nullptr);
        std::swap(st[i], nx[0]);
      }
      if (l % 8 == 7) {
        std::printf("  %s: layer %d (%.0f s)\n", variant.c_str(), l, secs(tv));
        std::fflush(stdout);
      }
    }
    eval::LpSet set;
    set.k = K;
    for (size_t i = 0; i < sel.size(); ++i) {
      set.chunks.emplace_back();
      lp_chunk(qm, cs.sample(wpick[static_cast<size_t>(sel[i])]), st[i], set.chunks.back());
    }
    write(variant + ".xlp", set);
    std::printf("kld: %s (%.0f s)\n", variant.c_str(), secs(tv));
  }
  return 0;
}

}  // namespace xenithium:bake
