#include "xenithium/model/config.hpp"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <stdexcept>

namespace xenithium::model {

std::string read_text_file(const std::string &path) {
  std::FILE *fp = std::fopen(path.c_str(), "rb");
  if (!fp) throw std::runtime_error(path + ": cannot open");
  std::string text;
  char buf[1 << 16];
  size_t n;
  while ((n = std::fread(buf, 1, sizeof buf, fp)) > 0) text.append(buf, n);
  const bool bad = std::ferror(fp) != 0;
  std::fclose(fp);
  if (bad) throw std::runtime_error(path + ": read error");
  return text;
}

ModelConfig ModelConfig::load(const std::string &model_dir) {
  ModelConfig c;
  c.dir_ = std::filesystem::absolute(model_dir).lexically_normal().string();
  const std::string path = c.dir_ + "/config.json";
  std::string err;
  if (!json::parse(read_text_file(path), c.root_, &err)) throw std::runtime_error(path + ": " + err);
  if (!c.root_.is_object()) throw std::runtime_error(path + ": not a JSON object");
  return c;
}

const json::Value &ModelConfig::text() const {
  const json::Value *t = root_.get_object("text_config");
  return t ? *t : root_;
}

std::string ModelConfig::model_type() const { return str_or(root_, "model_type", ""); }

std::vector<std::string> ModelConfig::architectures() const {
  std::vector<std::string> out;
  if (const json::Value *a = root_.get_array("architectures"))
    for (const auto &v : a->arr)
      if (v.is_string()) out.push_back(v.str);
  return out;
}

namespace {
[[noreturn]] void bad(std::string_view key, const char *what) {
  throw std::runtime_error("config.json: " + std::string(key) + " " + what);
}
int64_t to_int(const json::Value &v, std::string_view key) {
  if (!v.is_number() || !v.is_int()) bad(key, "is not an integer");
  errno = 0;
  char *end = nullptr;
  const long long x = std::strtoll(v.str.c_str(), &end, 10);
  if (errno != 0 || *end != '\0') bad(key, "is out of range");
  return x;
}
}  // namespace

const json::Value &ModelConfig::need(const json::Value &obj, std::string_view key) {
  const json::Value *v = obj.get(key);
  if (!v) bad(key, "is missing");
  return *v;
}
int64_t ModelConfig::need_int(const json::Value &obj, std::string_view key) { return to_int(need(obj, key), key); }
double ModelConfig::need_num(const json::Value &obj, std::string_view key) {
  const json::Value &v = need(obj, key);
  if (!v.is_number()) bad(key, "is not a number");
  return v.num;
}
std::string ModelConfig::need_str(const json::Value &obj, std::string_view key) {
  const json::Value &v = need(obj, key);
  if (!v.is_string()) bad(key, "is not a string");
  return v.str;
}
bool ModelConfig::need_bool(const json::Value &obj, std::string_view key) {
  const json::Value &v = need(obj, key);
  if (!v.is_bool()) bad(key, "is not a boolean");
  return v.b;
}
std::vector<int64_t> ModelConfig::need_ints(const json::Value &obj, std::string_view key) {
  const json::Value &v = need(obj, key);
  if (!v.is_array()) bad(key, "is not an array");
  std::vector<int64_t> out;
  for (const auto &e : v.arr) out.push_back(to_int(e, key));
  return out;
}
std::vector<std::string> ModelConfig::need_strs(const json::Value &obj, std::string_view key) {
  const json::Value &v = need(obj, key);
  if (!v.is_array()) bad(key, "is not an array");
  std::vector<std::string> out;
  for (const auto &e : v.arr) {
    if (!e.is_string()) bad(key, "has a non-string element");
    out.push_back(e.str);
  }
  return out;
}
int64_t ModelConfig::int_or(const json::Value &obj, std::string_view key, int64_t fallback) {
  const json::Value *v = obj.get(key);
  return (v && !v->is_null()) ? to_int(*v, key) : fallback;
}
double ModelConfig::num_or(const json::Value &obj, std::string_view key, double fallback) {
  const json::Value *v = obj.get(key);
  if (!v || v->is_null()) return fallback;
  if (!v->is_number()) bad(key, "is not a number");
  return v->num;
}
std::string ModelConfig::str_or(const json::Value &obj, std::string_view key, std::string fallback) {
  const json::Value *v = obj.get(key);
  if (!v || v->is_null()) return fallback;
  if (!v->is_string()) bad(key, "is not a string");
  return v->str;
}

}  // namespace xenithium:model
