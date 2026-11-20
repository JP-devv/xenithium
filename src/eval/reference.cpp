#include "xenithium/eval/reference.hpp"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <utility>

#include "xenithium/util/http.hpp"

namespace xenithium::eval {

namespace {

[[noreturn]] void fail(const std::string &m) { throw std::runtime_error(m); }

double number(const json::Value *v, const char *what) {
  if (!v || !v->is_number()) fail(std::string("completion: missing number ") + what);
  return v->num;
}

int32_t parse_id(const std::string &s) {
  const std::string d = s.rfind("token_id:", 0) == 0 ? s.substr(9) : s;
  char *end = nullptr;
  const long v = std::strtol(d.c_str(), &end, 10);
  if (d.empty() || *end != '\0' || v < 0) fail("completion: bad token key '" + s + "'");
  return static_cast<int32_t>(v);
}

void top_k(std::vector<std::pair<int32_t, double>> e, int k, LpRow &r) {
  std::sort(e.begin(), e.end(), [](const auto &a, const auto &b) { return a.second > b.second || (a.second == b.second && a.first < b.first); });
  if (static_cast<int>(e.size()) > k) e.resize(static_cast<size_t>(k));
  for (const auto &[id, lp] : e) {
    r.ids.push_back(id);
    r.lps.push_back(static_cast<float>(lp));
  }
}

}  // namespace

std::vector<std::vector<int32_t>> read_tokens(const std::string &path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) fail("cannot open " + path);
  int32_t n = 0;
  if (!f.read(reinterpret_cast<char *>(&n), 4) || n <= 0 || n > 100000) fail(path + ": bad token file header");
  std::vector<std::vector<int32_t>> chunks(static_cast<size_t>(n));
  for (auto &c : chunks) {
    int32_t len = 0;
    if (!f.read(reinterpret_cast<char *>(&len), 4) || len <= 1 || len > (1 << 24)) fail(path + ": bad chunk length");
    c.resize(static_cast<size_t>(len));
    if (!f.read(reinterpret_cast<char *>(c.data()), 4LL * len)) fail(path + ": truncated");
  }
  if (f.peek() != std::char_traits<char>::eof()) fail(path + ": trailing bytes");
  return chunks;
}

std::string completion_request(const std::string &model, const std::vector<int32_t> &ids, int k) {
  auto v = json::Value::object();
  v.set("model", json::Value::string(model));
  auto p = json::Value::array();
  for (int32_t t : ids) p.push(json::Value::integer(t));
  v.set("prompt", std::move(p));
  v.set("max_tokens", json::Value::integer(1));
  v.set("temperature", json::Value::integer(0));
  v.set("prompt_logprobs", json::Value::integer(k));
  v.set("logprobs", json::Value::integer(k));
  v.set("return_tokens_as_token_ids", json::Value::boolean(true));
  return json::dump(v);
}

std::vector<LpRow> rows_from_completion(const json::Value &resp, const std::vector<int32_t> &ids, int k) {
  const json::Value *choices = resp.get_array("choices");
  if (!choices || choices->arr.empty()) fail("completion: no choices");
  const json::Value &ch = choices->arr[0];
  const json::Value *plp = ch.get_array("prompt_logprobs");
  if (!plp || plp->arr.size() != ids.size())
    fail("completion: prompt_logprobs has " + std::to_string(plp ? plp->arr.size() : 0) + " entries for " + std::to_string(ids.size()) + " tokens");
  std::vector<LpRow> rows;
  rows.reserve(ids.size());
  for (size_t j = 0; j + 1 < ids.size(); ++j) {
    const json::Value &d = plp->arr[j + 1];
    if (!d.is_object()) fail("completion: prompt_logprobs[" + std::to_string(j + 1) + "] is not an object");
    LpRow r;
    r.target = ids[j + 1];
    bool have_target = false;
    std::vector<std::pair<int32_t, double>> e;
    for (const auto &m : d.obj) {
      const int32_t id = parse_id(m.first);
      const double lp = number(m.second.get("logprob"), "logprob");
      if (id == r.target) r.target_lp = static_cast<float>(lp), have_target = true;
      const json::Value *rank = m.second.get("rank");
      if (rank && rank->is_number() && rank->num > k) continue;  // the prompt token outside the top k
      e.emplace_back(id, lp);
    }
    if (!have_target) fail("completion: row " + std::to_string(j) + ": the prompt token " + std::to_string(r.target) + " is missing");
    top_k(std::move(e), k, r);
    rows.push_back(std::move(r));
  }
  const json::Value *lp = ch.get_object("logprobs");
  if (!lp) fail("completion: no logprobs");
  const json::Value *toks = lp->get_array("tokens"), *tlps = lp->get_array("token_logprobs"), *tops = lp->get_array("top_logprobs");
  if (!toks || !tlps || !tops || toks->arr.empty() || tops->arr.empty() || !toks->arr.back().is_string()) fail("completion: bad logprobs block");
  LpRow r;
  r.target = parse_id(toks->arr.back().str);
  r.target_lp = static_cast<float>(number(&tlps->arr.back(), "token_logprobs"));
  std::vector<std::pair<int32_t, double>> e;
  for (const auto &m : tops->arr.back().obj) e.emplace_back(parse_id(m.first), number(&m.second, "top_logprobs"));
  top_k(std::move(e), k, r);
  rows.push_back(std::move(r));
  return rows;
}

std::string served_model(const std::string &base_url) {
  const auto r = http::request("GET", base_url + "/v1/models", "", 60);
  json::Value v;
  std::string err;
  if (r.status != 200 || !json::parse(r.body, v, &err)) fail("/v1/models: HTTP " + std::to_string(r.status) + " " + err);
  const json::Value *d = v.get_array("data");
  if (!d || d->arr.empty() || !d->arr[0].get_string("id")) fail("/v1/models: no model");
  return *d->arr[0].get_string("id");
}

std::vector<LpRow> dump_reference_chunk(const std::string &base_url, const std::string &model, const std::vector<int32_t> &ids, int k,
                                        double timeout_s) {
  const auto r = http::request("POST", base_url + "/v1/completions", completion_request(model, ids, k), timeout_s);
  if (r.status != 200) fail("/v1/completions: HTTP " + std::to_string(r.status) + ": " + r.body.substr(0, 500));
  json::Value v;
  std::string err;
  if (!json::parse(r.body, v, &err)) fail("/v1/completions: bad JSON: " + err);
  return rows_from_completion(v, ids, k);
}

DumpCompare compare_dumps(const LpSet &ref, const LpSet &test, const std::vector<int64_t> &edges) {
  DumpCompare out{KlStats(edges), 0};
  if (ref.chunks.size() != test.chunks.size())
    fail("chunks differ: " + std::to_string(ref.chunks.size()) + " vs " + std::to_string(test.chunks.size()));
  for (size_t c = 0; c < ref.chunks.size(); ++c) {
    const auto &rc = ref.chunks[c], &tc = test.chunks[c];
    if (rc.size() != tc.size()) fail("chunk " + std::to_string(c) + ": rows differ: " + std::to_string(rc.size()) + " vs " + std::to_string(tc.size()));
    for (size_t j = 0; j < rc.size(); ++j) {
      const auto &r = rc[j], &t = tc[j];
      if (j + 1 < rc.size() && r.target != t.target)
        fail("chunk " + std::to_string(c) + " row " + std::to_string(j) + ": targets differ (" + std::to_string(r.target) + " vs " +
             std::to_string(t.target) + "): not the same token file");
      if (j + 1 == rc.size()) out.last_rows_differ += r.target != t.target;
      auto lp = [&](int32_t id, bool &exact) {
        for (size_t i = 0; i < t.ids.size(); ++i)
          if (t.ids[i] == id) return exact = true, static_cast<double>(t.lps[i]);
        exact = false;
        return t.lps.empty() ? -1e30 : static_cast<double>(t.lps.back());
      };
      out.stats.add(static_cast<int64_t>(j), kl_coarse(r, lp, t.ids.empty() ? -1 : t.ids[0]), r.target_lp, t.target_lp);
    }
  }
  return out;
}

}  // namespace xenithium::eval
