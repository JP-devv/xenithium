// Xenithium · a model directory's config.json, model-agnostic.
// Architectures read their typed configuration from here; this layer only loads the
// file and offers checked lookups that name the missing key in the error.
#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "xenithium/util/json.hpp"

namespace xenithium::model {

class ModelConfig {
 public:
  static ModelConfig load(const std::string &model_dir);

  const std::string &dir() const { return dir_; }
  const json::Value &root() const { return root_; }
  // The language-model section: `text_config` when present (multimodal checkpoints),
  // else the root.
  const json::Value &text() const;

  std::string model_type() const;  // root model_type ("" when absent)
  std::vector<std::string> architectures() const;  // root architectures[]

  // Checked lookups in an object; throw std:runtime_error("config.json: <key>...").
  static const json::Value &need(const json::Value &obj, std::string_view key);
  static int64_t need_int(const json::Value &obj, std::string_view key);
  static double need_num(const json::Value &obj, std::string_view key);
  static std::string need_str(const json::Value &obj, std::string_view key);
  static bool need_bool(const json::Value &obj, std::string_view key);
  static std::vector<int64_t> need_ints(const json::Value &obj, std::string_view key);
  static std::vector<std::string> need_strs(const json::Value &obj, std::string_view key);
  static int64_t int_or(const json::Value &obj, std::string_view key, int64_t fallback);
  static double num_or(const json::Value &obj, std::string_view key, double fallback);
  static std::string str_or(const json::Value &obj, std::string_view key, std::string fallback);

 private:
  std::string dir_;
  json::Value root_;
};

// Reads a whole file; throws with the path on failure.
std::string read_text_file(const std::string &path);

}  // namespace xenithium:model
