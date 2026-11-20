#include "xenithium/model/checkpoint.hpp"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <stdexcept>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include "xenithium/util/json.hpp"

namespace xenithium::model {

DType dtype_from_string(std::string_view s) {
  if (s == "F32") return DType::f32;
  if (s == "F16") return DType::f16;
  if (s == "BF16") return DType::bf16;
  if (s == "F8_E4M3") return DType::f8_e4m3;
  if (s == "F8_E5M2") return DType::f8_e5m2;
  if (s == "I64") return DType::i64;
  if (s == "I32") return DType::i32;
  if (s == "I16") return DType::i16;
  if (s == "I8") return DType::i8;
  if (s == "U8") return DType::u8;
  if (s == "BOOL") return DType::boolean;
  return DType::unknown;
}

std::string_view dtype_name(DType t) {
  switch (t) {
    case DType::f32: return "F32";
    case DType::f16: return "F16";
    case DType::bf16: return "BF16";
    case DType::f8_e4m3: return "F8_E4M3";
    case DType::f8_e5m2: return "F8_E5M2";
    case DType::i64: return "I64";
    case DType::i32: return "I32";
    case DType::i16: return "I16";
    case DType::i8: return "I8";
    case DType::u8: return "U8";
    case DType::boolean: return "BOOL";
    default: return "?";
  }
}

namespace {

[[noreturn]] void fail(const std::string &file, const std::string &what) {
  throw std::runtime_error(file + ": " + what);
}

// Exact unsigned integer from a JSON number (its lexeme), never through a double.
uint64_t as_u64(const json::Value &v, const std::string &file, const std::string &what) {
  if (!v.is_number() || !v.is_int() || v.str.empty() || v.str[0] == '-') fail(file, what + ": not a non-negative integer");
  errno = 0;
  char *end = nullptr;
  const unsigned long long x = std::strtoull(v.str.c_str(), &end, 10);
  if (errno != 0 || end == v.str.c_str() || *end != '\0') fail(file, what + ": integer out of range");
  return x;
}

bool read_full(int fd, void *buf, size_t n, uint64_t off) {
  auto *p = static_cast<char *>(buf);
  while (n > 0) {
    const ssize_t got = ::pread(fd, p, n, static_cast<off_t>(off));
    if (got < 0 && errno == EINTR) continue;
    if (got <= 0) return false;
    p += got;
    n -= static_cast<size_t>(got);
    off += static_cast<uint64_t>(got);
  }
  return true;
}

}  // namespace

std::vector<TensorInfo> read_safetensors_header(const std::string &path, ShardInfo *shard_out) {
  const std::string file = std::filesystem::path(path).filename().string();
  const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) fail(path, std::string("open: ") + std::strerror(errno));
  struct stat st{};
  if (::fstat(fd, &st) != 0) {
    ::close(fd);
    fail(path, "fstat failed");
  }
  const uint64_t file_bytes = static_cast<uint64_t>(st.st_size);
  uint64_t header_len = 0;
  unsigned char len_bytes[8];
  if (file_bytes < 8 || !read_full(fd, len_bytes, 8, 0)) {
    ::close(fd);
    fail(file, "too short for a safetensors header length");
  }
  for (int i = 7; i >= 0; --i) header_len = (header_len << 8) | len_bytes[i];
  // A header beyond 256 MiB or the file is corruption, not a model.
  if (header_len == 0 || header_len > (256ull << 20) || 8 + header_len > file_bytes) {
    ::close(fd);
    fail(file, "header length " + std::to_string(header_len) + " does not fit the file (" +
                   std::to_string(file_bytes) + " bytes)");
  }
  std::string text(header_len, '\0');
  const bool ok = read_full(fd, text.data(), header_len, 8);
  ::close(fd);
  if (!ok) fail(file, "short read of the header");
  // Trailing spaces are legal padding in the format.
  while (!text.empty() && text.back() == ' ') text.pop_back();

  json::Value root;
  std::string err;
  if (!json::parse(text, root, &err)) fail(file, "header is not valid JSON: " + err);
  if (!root.is_object()) fail(file, "header is not a JSON object");

  const uint64_t data_offset = 8 + header_len;
  const uint64_t data_bytes = file_bytes - data_offset;
  std::vector<TensorInfo> out;
  out.reserve(root.obj.size());
  for (const auto &m : root.obj) {
    if (m.first == "__metadata__") continue;
    const json::Value &t = m.second;
    const std::string where = "tensor " + m.first;
    if (!t.is_object()) fail(file, where + ": entry is not an object");
    const std::string *dt = t.get_string("dtype");
    const json::Value *shape = t.get_array("shape");
    const json::Value *offs = t.get_array("data_offsets");
    if (!dt || !shape || !offs || offs->arr.size() != 2) fail(file, where + ": needs dtype, shape, data_offsets[2]");
    TensorInfo ti;
    ti.name = m.first;
    ti.dtype_str = *dt;
    ti.dtype = dtype_from_string(*dt);
    for (const auto &d : shape->arr) ti.shape.push_back(static_cast<int64_t>(as_u64(d, file, where + " shape")));
    const uint64_t b = as_u64(offs->arr[0], file, where + " begin");
    const uint64_t e = as_u64(offs->arr[1], file, where + " end");
    if (e < b || e > data_bytes)
      fail(file, where + ": data_offsets [" + std::to_string(b) + ", " + std::to_string(e) + ") outside the " +
                     std::to_string(data_bytes) + "-byte data section");
    ti.offset = data_offset + b;
    ti.bytes = e - b;
    if (ti.dtype != DType::unknown) {
      const uint64_t want = static_cast<uint64_t>(ti.numel()) * dtype_size(ti.dtype);
      if (want != ti.bytes)
        fail(file, where + ": " + std::to_string(ti.bytes) + " bytes for " + std::to_string(ti.numel()) + " x " +
                       *dt + " (want " + std::to_string(want) + ")");
    }
    out.push_back(std::move(ti));
  }
  // Ranges must not overlap (identical empty tensors excepted).
  std::vector<const TensorInfo *> order;
  order.reserve(out.size());
  for (const auto &t : out) order.push_back(&t);
  std::sort(order.begin(), order.end(), [](const TensorInfo *a, const TensorInfo *b) { return a->offset < b->offset; });
  for (size_t i = 1; i < order.size(); ++i)
    if (order[i - 1]->bytes > 0 && order[i - 1]->offset + order[i - 1]->bytes > order[i]->offset)
      fail(file, "tensors " + order[i - 1]->name + " and " + order[i]->name + " overlap");

  if (shard_out) {
    shard_out->path = path;
    shard_out->file = file;
    shard_out->file_bytes = file_bytes;
    shard_out->header_bytes = header_len;
    shard_out->data_offset = data_offset;
    shard_out->tensors = out.size();
  }
  return out;
}

Checkpoint Checkpoint::open(const std::string &dir, bool cross_check_index) {
  namespace fs = std::filesystem;
  Checkpoint ck;
  ck.dir_ = fs::absolute(dir).lexically_normal().string();
  if (!fs::is_directory(ck.dir_)) throw std::runtime_error(ck.dir_ + ": not a directory");
  std::vector<std::string> files;
  for (const auto &e : fs::directory_iterator(ck.dir_))
    if (e.is_regular_file() && e.path().extension() == ".safetensors") files.push_back(e.path().string());
  std::sort(files.begin(), files.end());
  if (files.empty()) throw std::runtime_error(ck.dir_ + ": no .safetensors files");

  for (const auto &f : files) {
    ShardInfo si;
    auto ts = read_safetensors_header(f, &si);
    const auto shard = static_cast<uint32_t>(ck.shards_.size());
    ck.shards_.push_back(std::move(si));
    for (auto &t : ts) {
      t.shard = shard;
      ck.tensors_.push_back(std::move(t));
    }
  }
  ck.by_name_.reserve(ck.tensors_.size());
  for (uint32_t i = 0; i < ck.tensors_.size(); ++i) {
    const auto [it, fresh] = ck.by_name_.emplace(ck.tensors_[i].name, i);
    if (!fresh)
      throw std::runtime_error(ck.dir_ + ": tensor " + ck.tensors_[i].name + " appears in " +
                               ck.shards_[ck.tensors_[it->second].shard].file + " and " +
                               ck.shards_[ck.tensors_[i].shard].file);
  }

  const fs::path index = fs::path(ck.dir_) / "model.safetensors.index.json";
  if (cross_check_index && fs::exists(index)) {
    std::FILE *fp = std::fopen(index.c_str(), "rb");
    if (!fp) throw std::runtime_error(index.string() + ": cannot open");
    std::string text;
    char buf[1 << 16];
    size_t n;
    while ((n = std::fread(buf, 1, sizeof buf, fp)) > 0) text.append(buf, n);
    std::fclose(fp);
    json::Value root;
    std::string err;
    if (!json::parse(text, root, &err)) throw std::runtime_error(index.string() + ": " + err);
    const json::Value *wm = root.get_object("weight_map");
    if (!wm) throw std::runtime_error(index.string() + ": no weight_map");
    for (const auto &m : wm->obj) {
      const TensorInfo *t = ck.find(m.first);
      if (!t) throw std::runtime_error(index.string() + ": " + m.first + " is listed but in no shard");
      if (!m.second.is_string() || ck.shards_[t->shard].file != m.second.str)
        throw std::runtime_error(index.string() + ": " + m.first + " is listed in " + m.second.str + " but found in " +
                                 ck.shards_[t->shard].file);
    }
  }
  return ck;
}

const TensorInfo *Checkpoint::find(std::string_view name) const {
  const auto it = by_name_.find(name);
  return it == by_name_.end() ? nullptr : &tensors_[it->second];
}

const TensorInfo &Checkpoint::at(std::string_view name) const {
  const TensorInfo *t = find(name);
  if (!t) throw std::runtime_error(dir_ + ": no tensor " + std::string(name));
  return *t;
}

uint64_t Checkpoint::payload_bytes() const {
  uint64_t s = 0;
  for (const auto &t : tensors_) s += t.bytes;
  return s;
}

}  // namespace xenithium:model
