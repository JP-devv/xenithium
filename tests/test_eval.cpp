// Host-only gate for the KL tooling (eval/logprobs + eval/reference): top-k log-probs, the XLP1 file,
// the coarse KL, and the reference dump path end to end against a fake server (the completions response
// shape, a toy model) over a real socket: HTTP client, JSON parse, row alignment, dump compare.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <thread>

#include "xenithium/eval/logprobs.hpp"
#include "xenithium/eval/reference.hpp"
#include "xenithium/util/json.hpp"

namespace {
namespace ev = xenithium::eval;
int g_failures = 0;
void expect(bool ok, const std::string &what) {
  std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
  if (!ok) ++g_failures;
}

std::vector<double> log_softmax(const std::vector<float> &l) {
  double mx = l[0];
  for (float v : l) mx = std::max(mx, static_cast<double>(v));
  double se = 0;
  for (float v : l) se += std::exp(static_cast<double>(v) - mx);
  std::vector<double> o;
  for (float v : l) o.push_back(static_cast<double>(v) - mx - std::log(se));
  return o;
}

double kl_full(const std::vector<double> &a, const std::vector<double> &b) {
  double s = 0;
  for (size_t i = 0; i < a.size(); ++i) s += std::exp(a[i]) * (a[i] - b[i]);
  return s;
}

// ---- toy model + fake server -------------------------------------------------------------------
constexpr int kToyV = 300;

// Log-probs after `prefix` (distribution of the next token); `shift` perturbs the logits.
std::vector<double> toy_logprobs(const std::vector<int32_t> &prefix, double shift) {
  uint64_t h = 1469598103934665603ull;
  for (int32_t t : prefix) h = (h ^ static_cast<uint32_t>(t)) * 1099511628211ull;
  std::mt19937_64 base(h), noise(prefix.size() * 7919 + 1);
  std::normal_distribution<double> g(0, 1);
  std::vector<double> lg(kToyV);
  for (auto &x : lg) x = 3 * g(base);
  for (auto &x : lg) x += shift * g(noise);
  double mx = lg[0];
  for (double x : lg) mx = std::max(mx, x);
  double se = 0;
  for (double x : lg) se += std::exp(x - mx);
  for (auto &x : lg) x -= mx + std::log(se);
  return lg;
}

std::vector<int32_t> toy_ranked(const std::vector<double> &lp) {
  std::vector<int32_t> o(kToyV);
  for (int i = 0; i < kToyV; ++i) o[static_cast<size_t>(i)] = i;
  std::sort(o.begin(), o.end(), [&](int32_t a, int32_t b) { return lp[a] > lp[b] || (lp[a] == lp[b] && a < b); });
  return o;
}

namespace json = xenithium::json;

std::string toy_completion(const std::string &req_body, double shift) {
  json::Value req;
  if (!json::parse(req_body, req)) return "{}";
  std::vector<int32_t> ids;
  for (const auto &v : req.get_array("prompt")->arr) ids.push_back(static_cast<int32_t>(v.num));
  const int k = static_cast<int>(req.get("prompt_logprobs")->num);
  auto plp = json::Value::array();
  plp.push(json::Value());
  for (size_t j = 1; j < ids.size(); ++j) {
    const auto lp = toy_logprobs(std::vector<int32_t>(ids.begin(), ids.begin() + static_cast<long>(j)), shift);
    const auto order = toy_ranked(lp);
    auto d = json::Value::object();
    auto entry = [&](int32_t id) {
      const int rank = static_cast<int>(std::find(order.begin(), order.end(), id) - order.begin()) + 1;
      auto e = json::Value::object();
      e.set("logprob", json::Value::real(lp[id]));
      e.set("rank", json::Value::integer(rank));
      e.set("decoded_token", json::Value::string("x"));
      d.set(std::to_string(id), std::move(e));
    };
    for (int r = 0; r < k; ++r) entry(order[static_cast<size_t>(r)]);
    entry(ids[j]);  // the prompt token, with its own rank (the server adds it when outside the top k)
    plp.push(std::move(d));
  }
  const auto lp = toy_logprobs(ids, shift);
  const auto order = toy_ranked(lp);
  auto top = json::Value::object();
  for (int r = 0; r < k; ++r) top.set("token_id:" + std::to_string(order[static_cast<size_t>(r)]), json::Value::real(lp[order[static_cast<size_t>(r)]]));
  auto lpb = json::Value::object();
  auto toks = json::Value::array(), tlps = json::Value::array(), tops = json::Value::array();
  toks.push(json::Value::string("token_id:" + std::to_string(order[0])));
  tlps.push(json::Value::real(lp[order[0]]));
  tops.push(std::move(top));
  lpb.set("tokens", std::move(toks));
  lpb.set("token_logprobs", std::move(tlps));
  lpb.set("top_logprobs", std::move(tops));
  auto ch = json::Value::object();
  ch.set("text", json::Value::string("x"));
  ch.set("prompt_logprobs", std::move(plp));
  ch.set("logprobs", std::move(lpb));
  auto chs = json::Value::array();
  chs.push(std::move(ch));
  auto out = json::Value::object();
  out.set("choices", std::move(chs));
  return json::dump(out, json::Style::python);
}

// One thread, one connection at a time. /v1/models answers with Content-Length, completions with
// a chunked body (both framings of the client get used).
struct FakeServer {
  int fd = -1, port = 0;
  double shift = 0;
  std::atomic<bool> stop{false};
  std::thread th;
  explicit FakeServer(double s) : shift(s) {
    fd = ::socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    ::bind(fd, reinterpret_cast<sockaddr *>(&a), sizeof a);
    ::listen(fd, 4);
    socklen_t len = sizeof a;
    ::getsockname(fd, reinterpret_cast<sockaddr *>(&a), &len);
    port = ntohs(a.sin_port);
    th = std::thread([this] { run(); });
  }
  ~FakeServer() {
    stop = true;
    ::shutdown(fd, SHUT_RDWR);
    ::close(fd);
    th.join();
  }
  std::string url() const { return "http://127.0.0.1:" + std::to_string(port); }
  void run() {
    while (!stop) {
      const int c = ::accept(fd, nullptr, nullptr);
      if (c < 0) return;
      std::string in;
      char buf[65536];
      size_t need = std::string::npos;
      for (;;) {
        const ssize_t n = ::recv(c, buf, sizeof buf, 0);
        if (n <= 0) break;
        in.append(buf, static_cast<size_t>(n));
        const size_t he = in.find("\r\n\r\n");
        if (he != std::string::npos && need == std::string::npos) {
          const size_t cl = in.find("Content-Length: ");
          need = he + 4 + (cl != std::string::npos && cl < he ? std::stoul(in.substr(cl + 16)) : 0);
        }
        if (need != std::string::npos && in.size() >= need) break;
      }
      std::string out;
      if (in.rfind("GET /v1/models", 0) == 0) {
        const std::string b = R"({"object":"list","data":[{"id":"toy","object":"model"}]})";
        out = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: " + std::to_string(b.size()) + "\r\n\r\n" + b;
      } else if (in.rfind("POST /v1/completions", 0) == 0) {
        const std::string b = toy_completion(in.substr(in.find("\r\n\r\n") + 4), shift);
        out = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nTransfer-Encoding: chunked\r\n\r\n";
        for (size_t p = 0; p < b.size(); p += 1000) {
          const size_t m = std::min<size_t>(1000, b.size() - p);
          char hx[16];
          std::snprintf(hx, sizeof hx, "%zx\r\n", m);
          out += hx + b.substr(p, m) + "\r\n";
        }
        out += "0\r\n\r\n";
      } else {
        out = "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n";
      }
      for (size_t o = 0; o < out.size();) {
        const ssize_t n = ::send(c, out.data() + o, out.size() - o, MSG_NOSIGNAL);
        if (n <= 0) break;
        o += static_cast<size_t>(n);
      }
      ::close(c);
    }
  }
};
}  // namespace

int main() {
  std::mt19937 rng(11);
  std::normal_distribution<float> nd(0.f, 2.5f);
  const int V = 3000;
  {
    bool ok = true;
    for (int rep = 0; rep < 50 && ok; ++rep) {
      std::vector<float> l(V);
      for (auto &v : l) v = nd(rng);
      l[17] = l[2900] = 40.f;  // a tie at the top: lower id first
      const auto ls = log_softmax(l);
      const int32_t target = static_cast<int32_t>(rng() % V);
      const auto r = ev::top_logprobs(l.data(), V, 32, target);
      ok &= r.ids.size() == 32 && r.ids[0] == 17 && r.ids[1] == 2900;
      ok &= r.target_lp == static_cast<float>(ls[static_cast<size_t>(target)]);
      for (size_t i = 0; i < r.ids.size(); ++i) {
        ok &= r.lps[i] == static_cast<float>(ls[static_cast<size_t>(r.ids[i])]);
        if (i) ok &= r.lps[i] <= r.lps[i - 1];
      }
      int above = 0;  // nothing outside the top 32 beats its last entry
      for (int i = 0; i < V; ++i) above += ls[static_cast<size_t>(i)] > ls[static_cast<size_t>(r.ids.back())];
      ok &= above == 31;
    }
    expect(ok, "top_logprobs: top 32 sorted, ties lower id first, log-probs == double log-softmax, nothing missed");
  }
  {
    ev::LpSet s;
    s.k = 8;
    s.chunks.resize(3);
    for (int c = 0; c < 3; ++c)
      for (int j = 0; j < 5 + c; ++j) {
        ev::LpRow r;
        r.target = c * 100 + j;
        r.target_lp = -0.5f * static_cast<float>(j);
        const int m = (j == 2) ? 3 : 8;  // a short row is padded
        for (int i = 0; i < m; ++i) r.ids.push_back(1000 * c + 10 * j + i), r.lps.push_back(-static_cast<float>(i));
        s.chunks[static_cast<size_t>(c)].push_back(r);
      }
    const char *tmp = std::getenv("TMPDIR");
    const std::string path = std::string(tmp ? tmp : "/tmp") + "/xen-test-eval.xlp";
    ev::LpSet t;
    std::string err;
    bool ok = ev::write_lp(path, s, &err) && ev::read_lp(path, t, &err) && t.k == 8 && t.chunks.size() == 3;
    for (size_t c = 0; ok && c < 3; ++c) {
      ok &= t.chunks[c].size() == s.chunks[c].size();
      for (size_t j = 0; ok && j < s.chunks[c].size(); ++j) {
        const auto &a = s.chunks[c][j], &b = t.chunks[c][j];
        ok &= a.target == b.target && a.target_lp == b.target_lp && a.ids == b.ids && a.lps == b.lps;
      }
    }
    std::remove(path.c_str());
    expect(ok, "XLP1 write -> read round trip (3 chunks, padded short rows) " + err);
  }
  {
    bool ok_full = true, ok_bound = true, ok_same = true;
    double worst = 0;
    for (int rep = 0; rep < 50; ++rep) {
      std::vector<float> a(V), b(V);
      for (int i = 0; i < V; ++i) a[static_cast<size_t>(i)] = nd(rng) * 2, b[static_cast<size_t>(i)] = a[static_cast<size_t>(i)] + 0.3f * nd(rng);
      const auto la = log_softmax(a), lb = log_softmax(b);
      const double exact = kl_full(la, lb);
      auto test_lp = [&](int32_t id, bool &ex) {
        ex = true;
        return lb[static_cast<size_t>(id)];
      };
      int32_t tb = 0;
      for (int i = 1; i < V; ++i)
        if (b[static_cast<size_t>(i)] > b[static_cast<size_t>(tb)]) tb = i;
      const auto full = ev::kl_coarse(ev::top_logprobs(a.data(), V, V, 0), test_lp, tb);
      // the full row stores fp32 log-probs: compare at fp32 precision
      ok_full &= std::fabs(full.kl - exact) <= 1e-5 * std::max(1.0, exact);
      worst = std::max(worst, std::isfinite(full.kl - exact) ? std::fabs(full.kl - exact) : HUGE_VAL);
      for (int k : {1, 5, 20, 64}) {
        const auto c = ev::kl_coarse(ev::top_logprobs(a.data(), V, k, 0), test_lp, tb);
        ok_bound &= c.kl >= 0 && c.kl <= exact + 1e-6 && c.missing == 0;
      }
      auto self_lp = [&](int32_t id, bool &ex) {
        ex = true;
        return static_cast<double>(static_cast<float>(la[static_cast<size_t>(id)]));
      };
      const auto same = ev::kl_coarse(ev::top_logprobs(a.data(), V, 20, 0), self_lp, ev::top_logprobs(a.data(), V, 1, 0).ids[0]);
      ok_same &= same.kl < 1e-12 && same.top1_same;
    }
    char w[64];
    std::snprintf(w, sizeof w, "%.2e", worst);
    expect(ok_full, std::string("kl_coarse with k = vocab == the exact KL (worst |diff| ") + w + ")");
    expect(ok_bound, "kl_coarse with k = 1 / 5 / 20 / 64: 0 <= coarse <= exact (a lower bound)");
    expect(ok_same, "kl_coarse of a distribution against itself: 0, top-1 same");
  }
  {
    ev::KlStats st({128, 512});
    ev::KlRow r;
    r.kl = 0.01;
    r.top1_same = true;
    for (int64_t p : {0LL, 127LL, 128LL, 600LL, 9000LL}) st.add(p, r, -1.0, -2.0);
    const bool ok = st.buckets[0].rows == 2 && st.buckets[1].rows == 1 && st.buckets[2].rows == 2 && st.all().rows == 5 &&
                    std::fabs(std::exp(st.all().nll_ref / 5) - std::exp(1.0)) < 1e-12 && st.table().find("512+") != std::string::npos;
    expect(ok, "KlStats buckets [0,128) [128,512) [512,inf) and the total");
  }
  {
    // The reference dump path end to end: fake server -> HTTP -> JSON -> rows -> XLP1 -> compare.
    std::mt19937 trng(3);
    std::vector<std::vector<int32_t>> chunks;
    for (int n : {40, 57}) {
      chunks.emplace_back();
      for (int i = 0; i < n; ++i) chunks.back().push_back(static_cast<int32_t>(trng() % kToyV));
    }
    ev::LpSet ref, shifted;
    ref.k = shifted.k = 20;
    std::string model, err;
    try {
      FakeServer a(0.0), b(0.4);
      model = ev::served_model(a.url());
      for (const auto &c : chunks) {
        ref.chunks.push_back(ev::dump_reference_chunk(a.url(), model, c, 20, 30));
        shifted.chunks.push_back(ev::dump_reference_chunk(b.url(), model, c, 20, 30));
      }
    } catch (const std::exception &e) {
      err = e.what();
    }
    expect(err.empty() && model == "toy", "fake reference server: /v1/models (Content-Length) and two chunks of /v1/completions (chunked) " + err);
    bool ok = ref.chunks.size() == 2 && ref.chunks[0].size() == 40 && ref.chunks[1].size() == 57;
    double kl_exact = 0, kl_bound = 0;
    bool bound_ok = true;
    int64_t n = 0, outside = 0;
    for (size_t c = 0; ok && c < chunks.size(); ++c)
      for (size_t j = 0; j < chunks[c].size(); ++j) {
        const std::vector<int32_t> prefix(chunks[c].begin(), chunks[c].begin() + static_cast<long>(j) + 1);
        const auto lp = toy_logprobs(prefix, 0), lq = toy_logprobs(prefix, 0.4);
        const auto order = toy_ranked(lp);
        const int32_t tgt = j + 1 < chunks[c].size() ? chunks[c][j + 1] : order[0];
        const auto &r = ref.chunks[c][j];
        ok &= r.target == tgt && r.target_lp == static_cast<float>(lp[tgt]) && r.ids.size() == 20;
        for (size_t i = 0; ok && i < r.ids.size(); ++i) ok &= r.ids[i] == order[i] && r.lps[i] == static_cast<float>(lp[order[i]]);
        outside += std::find(r.ids.begin(), r.ids.end(), tgt) == r.ids.end();
        const double ex = kl_full(lp, lq);
        auto exact_lp = [&](int32_t id, bool &e) {
          e = true;
          return lq[id];
        };
        const double cb = ev::kl_coarse(r, exact_lp, -1).kl;
        bound_ok &= cb <= ex + 1e-6;
        kl_exact += ex;
        kl_bound += cb;
        ++n;
      }
    expect(ok && outside > 0, "reference rows line up with the token file (n rows per n tokens); top 20, targets and log-probs == the toy model; " +
                                  std::to_string(outside) + " prompt tokens outside the top 20 kept out of it by rank");
    expect(bound_ok, "coarse KL <= the exact full-vocab KL on every row (shifted toy model)");
    try {
      const auto self = ev::compare_dumps(ref, ref, {128});
      const auto d = ev::compare_dumps(ref, shifted, {128});
      const auto &x = d.stats.all();
      const double mean = x.kl / static_cast<double>(x.rows);
      char line[200];
      std::snprintf(line, sizeof line, "compare_dumps: self KL 0, top-1 100 %%; shifted: mean %.6f (exact-on-test %.6f, full %.6f), %lld bounded ids",
                    mean, kl_bound / static_cast<double>(n), kl_exact / static_cast<double>(n), static_cast<long long>(x.missing));
      expect(self.stats.all().kl == 0 && self.stats.all().top1 == self.stats.all().rows && self.stats.all().rows == 97 && mean > 0 &&
                 mean <= kl_exact / static_cast<double>(n) + 1e-6,
             line);
    } catch (const std::exception &e) {
      expect(false, std::string("compare_dumps: ") + e.what());
    }
    bool threw = false;
    try {
      auto bad = ref;
      bad.chunks[1][5].target += 1;
      ev::compare_dumps(ref, bad, {});
    } catch (const std::exception &) {
      threw = true;
    }
    json::Value resp;
    json::parse(toy_completion(ev::completion_request("toy", chunks[0], 20), 0), resp);
    bool threw2 = false;
    try {
      ev::rows_from_completion(resp, chunks[1], 20);  // 40 prompt log-probs for a 57-token chunk
    } catch (const std::exception &) {
      threw2 = true;
    }
    expect(threw && threw2, "mismatched targets and a response for another chunk are refused");
  }
  std::printf("%s: %d failure(s)\n", g_failures ? "FAIL" : "PASS", g_failures);
  return g_failures ? 1 : 0;
}
