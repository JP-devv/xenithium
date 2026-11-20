#include "xenithium/quant/calib.hpp"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <thread>

#include "xenithium/quant/parquet.hpp"
#include "xenithium/text/chat.hpp"
#include "xenithium/text/tokenizer.hpp"
#include "xenithium/util/json.hpp"

namespace xenithium::quant {

namespace fs = std::filesystem;
using json::Value;

namespace {

[[noreturn]] void fail(const std::string &w) { throw std::runtime_error("calib: " + w); }

// splitmix64: a portable seeded stream (std distributions differ between libraries).
struct Rng {
  uint64_t s;
  uint64_t next() {
    uint64_t z = (s += 0x9e3779b97f4a7c15ull);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
  }
  uint64_t below(uint64_t n) { return n ? next() % n : 0; }
  template <class T>
  void shuffle(std::vector<T> &v) {
    for (size_t i = v.size(); i > 1; --i) std::swap(v[i - 1], v[below(i)]);
  }
};

std::vector<std::string> glob_dir(const std::string &dir, const std::string &suffix) {
  std::vector<std::string> out;
  if (!fs::is_directory(dir)) fail("no directory " + dir);
  for (const auto &e : fs::directory_iterator(dir))
    if (e.is_regular_file() && e.path().string().size() >= suffix.size() &&
        e.path().string().compare(e.path().string().size() - suffix.size(), suffix.size(), suffix) == 0)
      out.push_back(e.path().string());
  std::sort(out.begin(), out.end());
  if (out.empty()) fail("no *" + suffix + " in " + dir);
  return out;
}

std::string slurp(const std::string &path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) fail("cannot read " + path);
  std::stringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

// ---- Python literal -> JSON ------------------------------------------------------------------
struct PyLit {
  const std::string &s;
  size_t i = 0;
  std::string out;
  void ws() {
    while (i < s.size() && (s[i] == ' ' || s[i] == '\n' || s[i] == '\t' || s[i] == '\r')) ++i;
  }
  [[noreturn]] void bad(const char *w) { fail(std::string("python literal: ") + w + " at " + std::to_string(i)); }
  void utf8(uint32_t cp) {
    if (cp < 0x80) {
      out += static_cast<char>(cp);
    } else if (cp < 0x800) {
      out += static_cast<char>(0xc0 | (cp >> 6));
      out += static_cast<char>(0x80 | (cp & 0x3f));
    } else if (cp < 0x10000) {
      out += static_cast<char>(0xe0 | (cp >> 12));
      out += static_cast<char>(0x80 | ((cp >> 6) & 0x3f));
      out += static_cast<char>(0x80 | (cp & 0x3f));
    } else {
      out += static_cast<char>(0xf0 | (cp >> 18));
      out += static_cast<char>(0x80 | ((cp >> 12) & 0x3f));
      out += static_cast<char>(0x80 | ((cp >> 6) & 0x3f));
      out += static_cast<char>(0x80 | (cp & 0x3f));
    }
  }
  // A Python string literal -> a JSON string (re-escaped).
  void str() {
    const char q = s[i++];
    std::string raw;
    auto hex = [&](int n) {
      if (i + n > s.size()) bad("short escape");
      const uint32_t v = static_cast<uint32_t>(std::stoul(s.substr(i, n), nullptr, 16));
      i += n;
      return v;
    };
    std::string tmp;
    std::swap(tmp, out);  // decode into out, then escape
    for (;;) {
      if (i >= s.size()) bad("unterminated string");
      const char c = s[i++];
      if (c == q) break;
      if (c != '\\') {
        out += c;
        continue;
      }
      if (i >= s.size()) bad("dangling backslash");
      const char e = s[i++];
      switch (e) {
        case 'n': out += '\n'; break;
        case 't': out += '\t'; break;
        case 'r': out += '\r'; break;
        case '0': out += '\0'; break;
        case 'a': out += '\a'; break;
        case 'b': out += '\b'; break;
        case 'f': out += '\f'; break;
        case 'v': out += '\v'; break;
        case 'x': utf8(hex(2)); break;
        case 'u': utf8(hex(4)); break;
        case 'U': utf8(hex(8)); break;
        case '\n': break;
        default: out += e; break;  // \\ \' \"
      }
    }
    raw.swap(out);
    out.swap(tmp);
    out += json::dump(Value::string(raw));
  }
  void value(int depth) {
    if (depth > 200) bad("too deep");
    ws();
    if (i >= s.size()) bad("unexpected end");
    const char c = s[i];
    if (c == '\'' || c == '"') return str();
    if (c == '{' || c == '[') {
      const char close = c == '{' ? '}' : ']';
      out += c;
      ++i;
      bool first = true;
      for (;;) {
        ws();
        if (i < s.size() && s[i] == close) {
          ++i;
          break;
        }
        if (!first) {
          if (s[i] != ',') bad("expected ,");
          ++i;
          ws();
          if (i < s.size() && s[i] == close) {  // trailing comma
            ++i;
            break;
          }
          out += ',';
        }
        first = false;
        value(depth + 1);
        if (close == '}') {
          ws();
          if (i >= s.size() || s[i] != ':') bad("expected :");
          ++i;
          out += ':';
          value(depth + 1);
        }
      }
      out += close;
      return;
    }
    if (s.compare(i, 4, "True") == 0) { out += "true"; i += 4; return; }
    if (s.compare(i, 5, "False") == 0) { out += "false"; i += 5; return; }
    if (s.compare(i, 4, "None") == 0) { out += "null"; i += 4; return; }
    size_t j = i;
    while (j < s.size() && (std::isdigit(static_cast<unsigned char>(s[j])) || s[j] == '-' || s[j] == '+' || s[j] == '.' || s[j] == 'e' || s[j] == 'E')) ++j;
    if (j == i) bad("unexpected character");
    out.append(s, i, j - i);
    i = j;
  }
};

// One document = text ready to tokenize (chats already rendered).
struct Doc {
  std::string text;
};

Value msg(const std::string &role, const std::string &content) {
  Value m = Value::object();
  m.set("role", Value::string(role));
  // Reasoning traces stored inline (<think>...</think> answer): the template wants them as the
  // message's reasoning, or it adds an empty think block in front of the inline one.
  if (role == "assistant" && content.rfind("<think>", 0) == 0) {
    const size_t e = content.find("</think>");
    if (e != std::string::npos) {
      size_t a = 7, b = e;
      while (a < b && (content[a] == '\n' || content[a] == ' ')) ++a;
      while (b > a && (content[b - 1] == '\n' || content[b - 1] == ' ')) --b;
      size_t c = e + 8;
      while (c < content.size() && (content[c] == '\n' || content[c] == ' ')) ++c;
      m.set("reasoning_content", Value::string(content.substr(a, b - a)));
      m.set("content", Value::string(content.substr(c)));
      return m;
    }
  }
  m.set("content", Value::string(content));
  return m;
}

bool render(const text::ChatFormat &fmt, const Value &messages, const Value *tools, std::string &out) {
  text::ChatOptions o;
  o.add_generation_prompt = false;
  std::string err;
  return fmt.render(messages, tools, o, out, err);
}

}  // namespace

std::string python_literal_to_json(const std::string &src) {
  PyLit p{src, 0, {}};
  p.value(0);
  p.ws();
  if (p.i != src.size()) p.bad("trailing characters");
  return p.out;
}

std::vector<CalibSource> default_mix(const std::string &root, int samples) {
  if (samples < 5) fail("need at least 5 samples for the mix");
  const int base[5] = {160, 96, 112, 48, 96};  // per 512
  int n[5], sum = 0;
  for (int i = 0; i < 5; ++i) sum += n[i] = std::max(1, base[i] * samples / 512);
  n[0] += samples - sum;  // the remainder goes to the plain-text source
  std::vector<CalibSource> v;
  v.push_back({"pile-10k", "parquet-text", glob_dir(root + "/pile-10k/data", ".parquet"), "text", n[0]});
  v.push_back({"ultrachat_200k", "parquet-chat", glob_dir(root + "/ultrachat_200k/data", ".parquet"), "messages.list.item", n[1]});
  v.push_back({"Mixture-of-Thoughts", "parquet-chat", glob_dir(root + "/Mixture-of-Thoughts/all", ".parquet"), "messages.list.element", n[2]});
  std::vector<std::string> hf;
  for (const char *f : {"func-calling.json", "glaive-function-calling-5k.json", "json-mode-agentic.json"}) hf.push_back(root + "/hermes-function-calling-v1/" + f);
  v.push_back({"hermes-function-calling", "sharegpt-json", hf, "", n[3]});
  v.push_back({"fable-agentic", "jsonl-chat", {root + "/Fable-5.1-Max-Reasoning-Filtered-10000x/10k/train.jsonl"}, "", n[4]});
  // ultrachat: only the SFT splits (the gen splits repeat the prompts)
  auto &u = v[1].files;
  u.erase(std::remove_if(u.begin(), u.end(), [](const std::string &f) { return f.find("train_sft") == std::string::npos; }), u.end());
  for (const auto &s : v)
    for (const auto &f : s.files)
      if (!fs::exists(f)) fail("missing " + f);
  return v;
}

namespace {

// Documents of one file, in file order.
std::vector<Doc> load_docs(const CalibSource &src, const std::string &file, const text::ChatFormat &fmt, int &failed) {
  std::vector<Doc> docs;
  if (src.kind == "parquet-text") {
    const auto pf = ParquetFile::open(file);
    for (auto &s : pf.strings(src.column))
      if (!s.empty()) docs.push_back({std::move(s)});
    return docs;
  }
  if (src.kind == "parquet-chat") {
    const auto pf = ParquetFile::open(file);
    const auto content = pf.string_lists(src.column + ".content"), role = pf.string_lists(src.column + ".role");
    for (size_t r = 0; r < content.size(); ++r) {
      if (content[r].size() != role[r].size()) fail(file + ": content / role lists differ in row " + std::to_string(r));
      Value ms = Value::array();
      for (size_t k = 0; k < content[r].size(); ++k) ms.push(msg(role[r][k], content[r][k]));
      Doc d;
      if (ms.arr.empty() || !render(fmt, ms, nullptr, d.text)) {
        ++failed;
        continue;
      }
      docs.push_back(std::move(d));
    }
    return docs;
  }
  if (src.kind == "sharegpt-json") {
    Value root;
    std::string err;
    if (!json::parse(slurp(file), root, &err) || !root.is_array()) fail(file + ": " + err);
    for (const Value &rec : root.arr) {
      const Value *conv = rec.get_array("conversations");
      if (!conv) {
        ++failed;
        continue;
      }
      Value ms = Value::array();
      for (const Value &t : conv->arr) {
        const std::string *from = t.get_string("from"), *val = t.get_string("value");
        if (!from || !val) continue;
        const std::string role = *from == "human" ? "user" : *from == "gpt" ? "assistant" : *from == "tool" ? "tool" : *from;
        Value m = msg(role, *val);
        if (role == "tool") m.set("tool_call_id", Value::string("call_0"));
        ms.push(std::move(m));
      }
      Doc d;
      if (ms.arr.empty() || !render(fmt, ms, nullptr, d.text)) {
        ++failed;
        continue;
      }
      docs.push_back(std::move(d));
    }
    return docs;
  }
  if (src.kind == "jsonl-chat") {
    std::ifstream f(file);
    if (!f) fail("cannot read " + file);
    std::string line;
    while (std::getline(f, line)) {
      Value rec;
      if (line.empty() || !json::parse(line, rec) || !rec.get_array("messages")) {
        ++failed;
        continue;
      }
      Value ms = Value::array();
      bool ok = true;
      for (const Value &m0 : rec.get_array("messages")->arr) {
        Value m = m0;
        if (const std::string *tc = m.get_string("tool_calls")) {  // a Python repr: convert
          Value parsed;
          try {
            if (!json::parse(python_literal_to_json(*tc), parsed)) ok = false;
          } catch (const std::exception &) {
            ok = false;
          }
          if (!ok) break;
          m.set("tool_calls", std::move(parsed));
        }
        if (const Value *tc = m.get("tool_calls"); tc && tc->is_null()) m.erase("tool_calls");
        ms.push(std::move(m));
      }
      Doc d;
      if (!ok || ms.arr.empty() || !render(fmt, ms, rec.get_array("tools"), d.text)) {
        ++failed;
        continue;
      }
      docs.push_back(std::move(d));
    }
    return docs;
  }
  fail("unknown source kind " + src.kind);
}

}  // namespace

CalibSet build_calib(const std::vector<CalibSource> &mix, const std::string &model_dir, int seqlen, uint64_t seed, int threads,
                     CalibBuildStats *stats) {
  if (seqlen < 16) fail("seqlen too small");
  text::Tokenizer tok;
  std::string err;
  if (!tok.load(model_dir, err)) fail("tokenizer: " + err);
  const auto fmt = text::open_chat_format("qwen38");
  if (!fmt) fail("no chat format for qwen38");
  const int32_t eos = tok.token_id("<|endoftext|>");
  if (eos < 0) fail("tokenizer has no <|endoftext|>");
  CalibSet c;
  c.seqlen = seqlen;
  if (stats) stats->docs_used.assign(mix.size(), 0), stats->docs_failed.assign(mix.size(), 0);
  threads = std::max(1, threads);
  for (size_t si = 0; si < mix.size(); ++si) {
    const CalibSource &src = mix[si];
    c.sources.push_back(src.name);
    Rng rng{seed ^ (0x5157ull * (si + 1))};
    std::vector<std::string> files = src.files;
    rng.shuffle(files);
    int made = 0, failed = 0, used = 0;
    std::vector<int32_t> cur;  // the sample being packed
    for (size_t fi = 0; fi < files.size() && made < src.samples; ++fi) {
      std::vector<Doc> docs = load_docs(src, files[fi], *fmt, failed);
      std::vector<size_t> order(docs.size());
      for (size_t i = 0; i < order.size(); ++i) order[i] = i;
      rng.shuffle(order);
      // tokenize in batches on threads, consume in order
      size_t at = 0;
      while (at < order.size() && made < src.samples) {
        const size_t batch = std::min(order.size() - at, static_cast<size_t>(threads) * 4);
        std::vector<std::vector<int32_t>> ids(batch);
        std::atomic<size_t> next{0};
        std::vector<std::thread> pool;
        for (int t = 0; t < threads; ++t)
          pool.emplace_back([&] {
            for (size_t j; (j = next.fetch_add(1)) < batch;) ids[j] = tok.encode(docs[order[at + j]].text);
          });
        for (auto &t : pool) t.join();
        for (size_t j = 0; j < batch && made < src.samples; ++j) {
          auto &d = ids[j];
          if (d.empty()) continue;
          ++used;
          if (static_cast<int>(d.size()) >= seqlen && cur.empty()) {
            // a long document alone: a seeded window (the head of agentic traces is the same
            // system prompt + tool list in every record)
            const size_t off = src.kind == "jsonl-chat" ? rng.below(d.size() - seqlen + 1) : 0;
            c.tokens.insert(c.tokens.end(), d.begin() + static_cast<long>(off), d.begin() + static_cast<long>(off) + seqlen);
            c.source_of.push_back(static_cast<int32_t>(si));
            ++made;
            continue;
          }
          if (!cur.empty() && src.kind == "parquet-text") cur.push_back(eos);
          cur.insert(cur.end(), d.begin(), d.end());
          if (static_cast<int>(cur.size()) >= seqlen) {
            c.tokens.insert(c.tokens.end(), cur.begin(), cur.begin() + seqlen);
            c.source_of.push_back(static_cast<int32_t>(si));
            ++made;
            cur.clear();
          }
        }
        at += batch;
      }
    }
    if (made < src.samples) fail(src.name + ": only " + std::to_string(made) + " of " + std::to_string(src.samples) + " samples");
    if (stats) stats->docs_used[si] = used, stats->docs_failed[si] = failed;
  }
  return c;
}

void write_calib(const std::string &path, const CalibSet &c) {
  std::unique_ptr<FILE, int (*)(FILE *)> f(std::fopen(path.c_str(), "wb"), &std::fclose);
  if (!f) fail("cannot write " + path);
  const int32_t h[4] = {0x31424358, c.samples(), c.seqlen, static_cast<int32_t>(c.sources.size())};  // 'XCB1'
  std::fwrite(h, 4, 4, f.get());
  for (const auto &s : c.sources) {
    char name[64] = {};
    std::strncpy(name, s.c_str(), 63);
    std::fwrite(name, 1, 64, f.get());
  }
  for (int s = 0; s < c.samples(); ++s) {
    std::fwrite(&c.source_of[static_cast<size_t>(s)], 4, 1, f.get());
    std::fwrite(c.sample(s), 4, static_cast<size_t>(c.seqlen), f.get());
  }
  if (std::fflush(f.get()) != 0) fail("write failed: " + path);
}

CalibSet read_calib(const std::string &path) {
  std::unique_ptr<FILE, int (*)(FILE *)> f(std::fopen(path.c_str(), "rb"), &std::fclose);
  if (!f) fail("cannot read " + path);
  int32_t h[4];
  if (std::fread(h, 4, 4, f.get()) != 4 || h[0] != 0x31424358) fail(path + ": not an XCB1 file");
  CalibSet c;
  c.seqlen = h[2];
  for (int i = 0; i < h[3]; ++i) {
    char name[65] = {};
    if (std::fread(name, 1, 64, f.get()) != 64) fail(path + ": truncated");
    c.sources.emplace_back(name);
  }
  c.source_of.resize(static_cast<size_t>(h[1]));
  c.tokens.resize(static_cast<size_t>(h[1]) * c.seqlen);
  for (int s = 0; s < h[1]; ++s)
    if (std::fread(&c.source_of[static_cast<size_t>(s)], 4, 1, f.get()) != 1 ||
        std::fread(c.tokens.data() + static_cast<size_t>(s) * c.seqlen, 4, static_cast<size_t>(c.seqlen), f.get()) != static_cast<size_t>(c.seqlen))
      fail(path + ": truncated");
  return c;
}

}  // namespace xenithium:quant
