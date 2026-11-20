// xenithium-evalrun: a small downstream gate against an OpenAI-compatible server.
// xenithium-evalrun --url http://127.0.0.1:8000 --out DIR [--concurrency 4] [--tasks gsm8k,humaneval,needle]
// [--datasets ~/ai/datasets] [--gsm8k-n 400] [--shots 8] [--python /usr/bin/python3]
// [--model DIR (tokenizer, for the needle haystack)] [--needle-ctx 16384,32768]
// [--needle-depths 0.1,0.3,0.5,0.7,0.9] [--needle-reps 2]
// xenithium-evalrun --compare DIR_A DIR_B (accuracy of both and the paired flips per task)
// Every request is a chat turn with thinking off, greedy (temperature 0), model "xenithium".
// gsm8k: the first N of gsm8k/test.jsonl; prompt = the first `shots` train.jsonl solutions (calculator marks
// <<...>> removed) as Question / Answer pairs, then the question (~1,000 prompt tokens at 8 shots: the
// prefill numerics shape the prompt's state, the rest is decode). Graded on the number after the last
// "####" (else the last number of the reply) against the test answer's.
// humaneval: humaneval/HumanEval.jsonl (164); the reply's first fenced code block (else the reply), after the
// prompt's lines before `def <entry_point>` (imports, helpers), + the test + check(<entry_point>), run by
// `timeout 20 PYTHON -I` in DIR/he: exit 0 = pass. (The test harness is the dataset's Python; the tool
// itself is C++.)
// needle: per (context, depth, rep) one prompt: ultrachat_200k test_sft assistant replies as the haystack
// (tokenizer-sized to the context minus 256), one sentence "The secret code for project NAME is NNNNNNN."
// inserted at the depth (a paragraph boundary), then the question; graded on the number in the reply.
// Per-item results go to DIR/<task>.tsv (id, ok, detail), the summary to stdout; --compare reads two DIRs.
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <mutex>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "xenithium/quant/parquet.hpp"
#include "xenithium/text/tokenizer.hpp"
#include "xenithium/util/http.hpp"
#include "xenithium/util/json.hpp"

namespace {

namespace fs = std::filesystem;
using xenithium::json::Value;

struct Opts {
  std::string url = "http://127.0.0.1:8000", out, datasets, python = "/usr/bin/python3", model;
  std::string tasks = "gsm8k,humaneval,needle";
  int conc = 4, gsm_n = 400, shots = 8, needle_reps = 2;
  std::vector<int> needle_ctx{16384, 32768};
  std::vector<double> needle_depths{0.1, 0.3, 0.5, 0.7, 0.9};
};

std::vector<std::string> split(const std::string &s, char d) {
  std::vector<std::string> v;
  std::stringstream ss(s);
  std::string t;
  while (std::getline(ss, t, d))
    if (!t.empty()) v.push_back(t);
  return v;
}

std::vector<Value> read_jsonl(const std::string &path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot read " + path);
  std::vector<Value> rows;
  std::string line, err;
  while (std::getline(f, line)) {
    if (line.empty()) continue;
    Value v;
    if (!xenithium::json::parse(line, v, &err)) throw std::runtime_error(path + ": " + err);
    rows.push_back(std::move(v));
  }
  return rows;
}

std::string field(const Value &v, const char *k) {
  const std::string *s = v.get_string(k);
  return s ? *s : std::string();
}

// One chat turn, thinking off, greedy. Returns the reply's content ("" + err on failure).
std::string chat(const Opts &o, const std::string &user, int max_tokens, std::string &err) {
  Value body = Value::object(), msgs = Value::array(), m = Value::object(), kw = Value::object();
  body.set("model", Value::string("xenithium"));
  body.set("temperature", Value::real(0.0));
  body.set("max_tokens", Value::integer(max_tokens));
  m.set("role", Value::string("user"));
  m.set("content", Value::string(user));
  msgs.push(std::move(m));
  body.set("messages", std::move(msgs));
  kw.set("enable_thinking", Value::boolean(false));
  body.set("chat_template_kwargs", std::move(kw));
  try {
    const auto r = xenithium::http::request("POST", o.url + "/v1/chat/completions", xenithium::json::dump(body), 1800);
    if (r.status != 200) {
      err = "HTTP " + std::to_string(r.status) + ": " + r.body.substr(0, 200);
      return {};
    }
    Value v;
    if (!xenithium::json::parse(r.body, v, &err)) return {};
    const Value *ch = v.get_array("choices");
    if (!ch || ch->arr.empty()) {
      err = "no choices";
      return {};
    }
    const Value *msg = ch->arr[0].get_object("message");
    const std::string *c = msg ? msg->get_string("content") : nullptr;
    return c ? *c : std::string();
  } catch (const std::exception &e) {
    err = e.what();
    return {};
  }
}

// Runs f(i) for i in [0, n) on `conc` threads.
void parallel(int n, int conc, const std::function<void(int)> &f) {
  std::atomic<int> next{0};
  std::vector<std::thread> th;
  for (int t = 0; t < std::max(1, conc); ++t)
    th.emplace_back([&] {
      for (int i; (i = next++) < n;) f(i);
    });
  for (auto &t : th) t.join();
}

std::string one_line(std::string s) {
  for (char &c : s)
    if (c == '\n' || c == '\t' || c == '\r') c = ' ';
  return s.substr(0, 160);
}

struct Item {
  bool ok = false;
  std::string detail;
};

void write_tsv(const std::string &path, const std::vector<Item> &items) {
  std::ofstream f(path);
  for (size_t i = 0; i < items.size(); ++i) f << i << '\t' << (items[i].ok ? 1 : 0) << '\t' << one_line(items[i].detail) << '\n';
}

// ---- gsm8k ---------------------------------------------------------------------------------------------------------
std::string strip_calc(const std::string &a) {
  std::string out;
  for (size_t i = 0; i < a.size();) {
    if (a.compare(i, 2, "<<") == 0) {
      const size_t e = a.find(">>", i);
      if (e != std::string::npos) {
        i = e + 2;
        continue;
      }
    }
    out += a[i++];
  }
  return out;
}

// The number after the last "####", else the last number in the text; commas and '$' dropped. NaN when none.
double answer_number(const std::string &text) {
  std::string t;
  const size_t h = text.rfind("####");
  t = h != std::string::npos ? text.substr(h + 4) : text;
  std::string num, last;
  bool in_num = false;
  auto flush = [&] {
    if (!num.empty() && num != "-" && num != ".") last = num;
    num.clear();
    in_num = false;
  };
  for (size_t i = 0; i < t.size(); ++i) {
    const char c = t[i];
    if (std::isdigit(static_cast<unsigned char>(c)) || (c == '.' && in_num) || (c == '-' && !in_num && i + 1 < t.size() && std::isdigit(static_cast<unsigned char>(t[i + 1])))) {
      num += c;
      in_num = true;
    } else if (c == ',' && in_num && i + 1 < t.size() && std::isdigit(static_cast<unsigned char>(t[i + 1]))) {
      continue;
    } else {
      flush();
      if (h != std::string::npos && !last.empty()) break;  // after ####: the first number
    }
  }
  flush();
  while (!last.empty() && last.back() == '.') last.pop_back();
  return last.empty() ? std::nan("") : std::strtod(last.c_str(), nullptr);
}

std::vector<Item> run_gsm8k(const Opts &o) {
  const auto test = read_jsonl(o.datasets + "/gsm8k/test.jsonl"), train = read_jsonl(o.datasets + "/gsm8k/train.jsonl");
  std::string shots = "Solve the math word problem. Show brief reasoning, then give the final answer on the last line as \"#### <number>\".\n\n";
  for (int s = 0; s < o.shots && s < static_cast<int>(train.size()); ++s)
    shots += "Question: " + field(train[static_cast<size_t>(s)], "question") + "\nAnswer: " + strip_calc(field(train[static_cast<size_t>(s)], "answer")) + "\n\n";
  const int n = std::min<int>(o.gsm_n, static_cast<int>(test.size()));
  std::vector<Item> items(static_cast<size_t>(n));
  parallel(n, o.conc, [&](int i) {
    const Value &row = test[static_cast<size_t>(i)];
    const double gold = answer_number(field(row, "answer"));
    std::string err;
    const std::string reply = chat(o, shots + "Question: " + field(row, "question") + "\nAnswer:", 512, err);
    const double got = answer_number(reply);
    Item &it = items[static_cast<size_t>(i)];
    it.ok = err.empty() && !std::isnan(got) && std::fabs(got - gold) < 1e-6;
    char d[96];
    std::snprintf(d, sizeof d, "gold %g got %g ", gold, got);
    it.detail = err.empty() ? d + reply.substr(reply.size() > 80 ? reply.size() - 80 : 0) : "error " + err;
  });
  return items;
}

// ---- humaneval -----------------------------------------------------------------------------------------------------
std::string code_block(const std::string &reply) {
  size_t s = reply.find("```");
  if (s == std::string::npos) return reply;
  s = reply.find('\n', s);
  if (s == std::string::npos) return reply;
  const size_t e = reply.find("```", s + 1);
  return reply.substr(s + 1, e == std::string::npos ? std::string::npos : e - s - 1);
}

std::vector<Item> run_humaneval(const Opts &o) {
  const auto rows = read_jsonl(o.datasets + "/humaneval/HumanEval.jsonl");
  const std::string dir = fs::absolute(o.out + "/he").string();  // the run cds into it: absolute file paths
  fs::create_directories(dir);
  std::vector<Item> items(rows.size());
  parallel(static_cast<int>(rows.size()), o.conc, [&](int i) {
    const Value &row = rows[static_cast<size_t>(i)];
    const std::string prompt = field(row, "prompt"), entry = field(row, "entry_point");
    std::string err;
    const std::string reply = chat(o,
                                   "Complete this Python function. Reply with the complete function (its signature included) in one ```python code "
                                   "block, no explanation.\n\n```python\n" + prompt + "```",
                                   768, err);
    Item &it = items[static_cast<size_t>(i)];
    if (!err.empty()) {
      it.detail = "error " + err;
      return;
    }
    const size_t d = prompt.find("def " + entry);
    const std::string prog = (d == std::string::npos ? std::string() : prompt.substr(0, d)) + "\n" + code_block(reply) + "\n\n" + field(row, "test") +
                             "\ncheck(" + entry + ")\n";
    const std::string file = dir + "/" + std::to_string(i) + ".py";
    std::ofstream(file) << prog;
    const std::string cmd = "cd '" + dir + "' && timeout 20 '" + o.python + "' -I '" + file + "' > /dev/null 2>&1";
    it.ok = std::system(cmd.c_str()) == 0;
    it.detail = entry;
  });
  return items;
}

// ---- needle --------------------------------------------------------------------------------------------------------
std::vector<Item> run_needle(const Opts &o) {
  if (o.model.empty()) throw std::runtime_error("needle: --model DIR (the tokenizer) is required");
  xenithium::text::Tokenizer tok;
  std::string err;
  if (!tok.load(o.model, err)) throw std::runtime_error("tokenizer: " + err);
  std::vector<std::string> files;
  for (const auto &e : fs::directory_iterator(o.datasets + "/ultrachat_200k/data"))
    if (e.path().filename().string().rfind("test_sft-", 0) == 0) files.push_back(e.path().string());
  std::sort(files.begin(), files.end());
  if (files.empty()) throw std::runtime_error("needle: no ultrachat_200k test_sft parquet");
  std::vector<std::string> paras;  // assistant replies, one paragraph each
  {
    const auto pf = xenithium::quant::ParquetFile::open(files[0]);
    const auto content = pf.string_lists("messages.list.item.content"), role = pf.string_lists("messages.list.item.role");
    for (size_t r = 0; r < content.size() && paras.size() < 20000; ++r)
      for (size_t k = 0; k < content[r].size() && k < role[r].size(); ++k)
        if (role[r][k] == "assistant" && content[r][k].size() > 200) paras.push_back(content[r][k]);
  }
  std::vector<size_t> ptok(paras.size(), 0);
  struct Case {
    int ctx;
    double depth;
    int rep;
  };
  std::vector<Case> cases;
  for (int ctx : o.needle_ctx)
    for (double d : o.needle_depths)
      for (int r = 0; r < o.needle_reps; ++r) cases.push_back({ctx, d, r});
  static const char *names[] = {"Aurora", "Basalt", "Cobalt", "Dune", "Ember", "Fjord", "Garnet", "Harbor", "Indigo", "Juniper"};
  std::vector<Item> items(cases.size());
  std::vector<std::string> prompts(cases.size()), codes(cases.size());
  size_t cursor = 0;
  for (size_t ci = 0; ci < cases.size(); ++ci) {
    const Case &c = cases[ci];
    std::mt19937 rng(static_cast<unsigned>(1000 + ci));
    const std::string name = names[ci % 10];
    const std::string code = std::to_string(1000000 + rng() % 9000000);
    const std::string needle = "The secret code for project " + name + " is " + code + ".";
    const std::string q = "\n</document>\n\nWhat is the secret code for project " + name + "? Reply with the number only.";
    const size_t budget = static_cast<size_t>(c.ctx) - 256;
    std::vector<size_t> pick;
    size_t total = tok.encode(needle + q).size() + 32;
    while (total < budget) {
      const size_t p = cursor++ % paras.size();
      if (!ptok[p]) ptok[p] = tok.encode(paras[p] + "\n\n").size();
      if (total + ptok[p] > budget) break;
      total += ptok[p];
      pick.push_back(p);
    }
    const size_t at = static_cast<size_t>(std::lround(c.depth * static_cast<double>(pick.size())));
    std::string doc = "Read the document below and answer the question after it.\n\n<document>\n";
    for (size_t i = 0; i <= pick.size(); ++i) {
      if (i == at) doc += needle + "\n\n";
      if (i < pick.size()) doc += paras[pick[i]] + "\n\n";
    }
    prompts[ci] = doc + q;
    codes[ci] = code;
  }
  parallel(static_cast<int>(cases.size()), std::min(o.conc, 2), [&](int i) {
    std::string e;
    const std::string reply = chat(o, prompts[static_cast<size_t>(i)], 32, e);
    Item &it = items[static_cast<size_t>(i)];
    it.ok = e.empty() && reply.find(codes[static_cast<size_t>(i)]) != std::string::npos;
    char d[64];
    std::snprintf(d, sizeof d, "ctx %d depth %.2f rep %d: ", cases[static_cast<size_t>(i)].ctx, cases[static_cast<size_t>(i)].depth, cases[static_cast<size_t>(i)].rep);
    it.detail = d + (e.empty() ? "want " + codes[static_cast<size_t>(i)] + " got " + reply : "error " + e);
  });
  return items;
}

std::vector<int> read_ok(const std::string &path) {
  std::ifstream f(path);
  std::vector<int> v;
  std::string line;
  while (std::getline(f, line)) {
    const auto t = split(line, '\t');
    if (t.size() >= 2) v.push_back(std::stoi(t[1]));
  }
  return v;
}

int compare(const std::string &a, const std::string &b) {
  std::printf("%-10s %8s %8s %8s %8s %8s\n", "task", "items", "A ok", "B ok", "A only", "B only");
  for (const char *t : {"gsm8k", "humaneval", "needle"}) {
    const auto x = read_ok(a + "/" + t + ".tsv"), y = read_ok(b + "/" + t + ".tsv");
    if (x.empty() || x.size() != y.size()) continue;
    int ox = 0, oy = 0, onlyx = 0, onlyy = 0;
    for (size_t i = 0; i < x.size(); ++i) ox += x[i], oy += y[i], onlyx += x[i] && !y[i], onlyy += y[i] && !x[i];
    std::printf("%-10s %8zu %8d %8d %8d %8d\n", t, x.size(), ox, oy, onlyx, onlyy);
  }
  return 0;
}

}  // namespace

int main(int argc, char **argv) {
  Opts o;
  const char *home = std::getenv("HOME");
  o.datasets = std::string(home ? home : ".") + "/ai/datasets";
  try {
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      auto val = [&]() -> std::string {
        if (i + 1 >= argc) throw std::runtime_error(a + " needs a value");
        return argv[++i];
      };
      if (a == "--compare") {
        const std::string x = val(), y = val();
        return compare(x, y);
      } else if (a == "--url") o.url = val();
      else if (a == "--out") o.out = val();
      else if (a == "--tasks") o.tasks = val();
      else if (a == "--concurrency") o.conc = std::stoi(val());
      else if (a == "--datasets") o.datasets = val();
      else if (a == "--gsm8k-n") o.gsm_n = std::stoi(val());
      else if (a == "--shots") o.shots = std::stoi(val());
      else if (a == "--python") o.python = val();
      else if (a == "--model") o.model = val();
      else if (a == "--needle-reps") o.needle_reps = std::stoi(val());
      else if (a == "--needle-ctx") {
        o.needle_ctx.clear();
        for (const auto &t : split(val(), ',')) o.needle_ctx.push_back(std::stoi(t));
      } else if (a == "--needle-depths") {
        o.needle_depths.clear();
        for (const auto &t : split(val(), ',')) o.needle_depths.push_back(std::stod(t));
      } else {
        std::fprintf(stderr, "xenithium-evalrun: unknown option %s\n", a.c_str());
        return 2;
      }
    }
    if (o.out.empty()) throw std::runtime_error("--out DIR is required");
    fs::create_directories(o.out);
    for (const auto &t : split(o.tasks, ',')) {
      const auto t0 = std::chrono::steady_clock::now();
      std::vector<Item> items;
      if (t == "gsm8k") items = run_gsm8k(o);
      else if (t == "humaneval") items = run_humaneval(o);
      else if (t == "needle") items = run_needle(o);
      else throw std::runtime_error("unknown task " + t);
      write_tsv(o.out + "/" + t + ".tsv", items);
      int ok = 0, errs = 0;
      for (const auto &it : items) ok += it.ok, errs += it.detail.rfind("error", 0) == 0;
      const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      std::printf("%-10s %4d / %4zu = %5.1f %%  (%d request errors, %.0f s)\n", t.c_str(), ok, items.size(), 100.0 * ok / std::max<size_t>(1, items.size()), errs, s);
      std::fflush(stdout);
    }
  } catch (const std::exception &e) {
    std::fprintf(stderr, "xenithium-evalrun: %s\n", e.what());
    return 1;
  }
  return 0;
}
