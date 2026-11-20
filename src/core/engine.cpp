#include "xenithium/core/engine.hpp"

#include <fstream>
#include <sstream>

#include "xenithium/util/json.hpp"

namespace xenithium::core {

bool load_generation_defaults(const std::string& model_dir, SamplingParams& sp, std::string& err) {
  std::ifstream f(model_dir + "/generation_config.json", std::ios::binary);
  if (!f) return true;
  std::stringstream ss;
  ss << f.rdbuf();
  json::Value g;
  if (!json::parse(ss.str(), g, &err) || !g.is_object()) {
    err = "generation_config.json: " + err;
    return false;
  }
  auto num = [&](const char* k, auto& out) {
    const json::Value* v = g.get(k);
    if (v && v->is_number()) out = static_cast<std::remove_reference_t<decltype(out)>>(v->num);
  };
  num("temperature", sp.temperature);
  num("top_p", sp.top_p);
  num("top_k", sp.top_k);
  num("min_p", sp.min_p);
  num("repetition_penalty", sp.repetition_penalty);
  if (sp.top_k < 0) sp.top_k = 0;
  return true;
}

const char* finish_name(Finish f) {
  switch (f) {
    case Finish::stop: return "stop";
    case Finish::length: return "length";
    case Finish::cancelled: return "abort";
    case Finish::error: return "error";
    case Finish::expired: return "expired";
    case Finish::none: break;
  }
  return "";
}

}  // namespace xenithium::core
