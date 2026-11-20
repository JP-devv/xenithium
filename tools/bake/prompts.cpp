// xenithium-bake prompts: chat prompts for a generation-time capture, as a probe token file.
// xenithium-bake prompts --model MODEL_DIR --out FILE [--datasets DIR] [--split test_sft] [--count 256]
// [--offset 0] [--max-tokens 1024] [--concat TOKENS]
// Rows of ultrachat_200k's `split` parquet files, `count` of them evenly spaced over the split starting at row
// `offset` of each stride (two disjoint sets = two offsets, e.g. 0 for calibration and stride / 2 for held-out).
// Each prompt = the conversation's first user message, rendered by the model's chat template with the generation
// prompt (the server's default: thinking on), tokenised by the model's tokenizer. Prompts longer than `max-tokens`
// are skipped (the next row is taken). File: int32 n, then per prompt int32 length + ids (the probe's --ppl format).
// --concat TOKENS: one chunk of exactly TOKENS ids instead: whole conversations (every turn, rendered by the chat
// template) of the rows offset, offset + stride... appended until the chunk is full - text of an even
// difficulty over the whole length (a long-context KL by position whose buckets read the same kind of text).
#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>

#include "bake.hpp"
#include "xenithium/quant/parquet.hpp"
#include "xenithium/text/chat.hpp"
#include "xenithium/text/tokenizer.hpp"
#include "xenithium/util/json.hpp"

namespace xenithium::bake {

namespace fs = std::filesystem;

int cmd_prompts(const Args &a) {
  const std::string root = a.str("datasets", home() + "/ai/datasets"), model = a.str("model"), out = a.str("out");
  const std::string split = a.str("split", "test_sft");
  if (model.empty() || out.empty()) throw std::runtime_error("--model and --out are required");
  const int count = static_cast<int>(a.num("count", 256)), max_tok = static_cast<int>(a.num("max-tokens", 1024));
  const long offset = a.num("offset", 0);
  std::vector<std::string> files;
  for (const auto &e : fs::directory_iterator(root + "/ultrachat_200k/data"))
    if (e.path().filename().string().rfind(split + "-", 0) == 0) files.push_back(e.path().string());
  std::sort(files.begin(), files.end());
  if (files.empty()) throw std::runtime_error("no ultrachat_200k " + split + " parquet files under " + root);
  std::vector<std::string> users;  // the first user message of every row, in file order
  std::vector<std::vector<std::string>> convs, roles;  // --concat: every turn of every row
  const long concat = a.num("concat", 0);
  for (const auto &f : files) {
    const auto pf = quant::ParquetFile::open(f);
    const auto content = pf.string_lists("messages.list.item.content"), role = pf.string_lists("messages.list.item.role");
    if (concat > 0) {
      convs.insert(convs.end(), content.begin(), content.end());
      roles.insert(roles.end(), role.begin(), role.end());
    }
    for (size_t r = 0; r < content.size(); ++r) {
      std::string u;
      for (size_t k = 0; k < content[r].size() && k < role[r].size(); ++k)
        if (role[r][k] == "user") {
          u = content[r][k];
          break;
        }
      users.push_back(std::move(u));
    }
  }
  const auto fmt = text::open_chat_format("qwen38");
  if (!fmt) throw std::runtime_error("no chat format");
  text::Tokenizer tok;
  std::string err;
  if (!tok.load(model, err)) throw std::runtime_error("tokenizer: " + err);
  if (concat > 0) {
    const size_t stride = std::max<size_t>(1, convs.size() / static_cast<size_t>(count));
    std::vector<int32_t> ids;
    size_t used = 0;
    for (size_t r = static_cast<size_t>(offset); r < convs.size() && static_cast<long>(ids.size()) < concat; r += stride) {
      json::Value ms = json::Value::array();
      for (size_t k = 0; k < convs[r].size() && k < roles[r].size(); ++k) {
        json::Value m = json::Value::object();
        m.set("role", json::Value::string(roles[r][k]));
        m.set("content", json::Value::string(convs[r][k]));
        ms.push(std::move(m));
      }
      text::ChatOptions o;
      o.add_generation_prompt = false;
      std::string txt, e;
      if (!fmt->render(ms, nullptr, o, txt, e)) continue;
      const auto t = tok.encode(txt);
      ids.insert(ids.end(), t.begin(), t.end());
      ++used;
    }
    if (static_cast<long>(ids.size()) < concat) throw std::runtime_error("--concat: the split ran out of rows");
    ids.resize(static_cast<size_t>(concat));
    std::ofstream f(out, std::ios::binary);
    const int32_t n = 1, len = static_cast<int32_t>(concat);
    f.write(reinterpret_cast<const char *>(&n), 4);
    f.write(reinterpret_cast<const char *>(&len), 4);
    f.write(reinterpret_cast<const char *>(ids.data()), static_cast<std::streamsize>(len) * 4);
    if (!f) throw std::runtime_error("cannot write " + out);
    std::printf("prompts --concat: %d tokens from %zu conversations of %s (stride %zu, offset %ld) -> %s\n", len, used, split.c_str(), stride, offset,
                out.c_str());
    return 0;
  }
  const size_t stride = users.size() / static_cast<size_t>(count);
  if (stride < 1 || static_cast<size_t>(offset) >= stride) throw std::runtime_error("--count too large or --offset past the stride");
  std::vector<std::vector<int32_t>> prompts;
  size_t skipped = 0, longest = 0, total = 0;
  for (int i = 0; i < count; ++i) {
    for (size_t r = static_cast<size_t>(i) * stride + static_cast<size_t>(offset); r < static_cast<size_t>(i + 1) * stride; ++r) {
      if (users[r].empty()) continue;
      json::Value ms = json::Value::array(), m = json::Value::object();
      m.set("role", json::Value::string("user"));
      m.set("content", json::Value::string(users[r]));
      ms.push(std::move(m));
      text::ChatOptions o;
      o.add_generation_prompt = true;
      std::string txt, e;
      if (!fmt->render(ms, nullptr, o, txt, e)) continue;
      auto ids = tok.encode(txt);
      if (static_cast<int>(ids.size()) > max_tok || ids.size() < 2) {
        ++skipped;
        continue;
      }
      longest = std::max(longest, ids.size());
      total += ids.size();
      prompts.push_back(std::move(ids));
      break;
    }
  }
  std::ofstream f(out, std::ios::binary);
  const int32_t n = static_cast<int32_t>(prompts.size());
  f.write(reinterpret_cast<const char *>(&n), 4);
  for (const auto &p : prompts) {
    const int32_t len = static_cast<int32_t>(p.size());
    f.write(reinterpret_cast<const char *>(&len), 4);
    f.write(reinterpret_cast<const char *>(p.data()), static_cast<std::streamsize>(len) * 4);
  }
  if (!f) throw std::runtime_error("cannot write " + out);
  std::printf("prompts: %d of %s (%zu rows, stride %zu, offset %ld), %zu tokens, longest %zu, %zu skipped as longer than %d -> %s\n", n, split.c_str(), users.size(),
              stride, offset, total, longest, skipped, max_tok, out.c_str());
  return n == count ? 0 : 1;
}

}  // namespace xenithium:bake
