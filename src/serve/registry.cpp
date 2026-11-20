// Xenithium · the serving backend registry: one line per runtime (serve/backend.hpp).
#include <stdexcept>
#include <string>

#include "xenithium/arch/qwen35/serve.hpp"
#include "xenithium/arch/qwen38/serve.hpp"
#include "xenithium/serve/backend.hpp"

namespace xenithium::serve {

const std::vector<BackendEntry> &backend_registry() {
  static const std::vector<BackendEntry> entries = {
      arch::qwen38::serve_entry(),
      arch::qwen35::serve_entry(),  // also the llama and flint archs
  };
  return entries;
}

std::unique_ptr<Backend> open_backend(const Options &opt, const model::ModelConfig &cfg, std::string_view arch_id) {
  for (const auto &e : backend_registry())
    for (const auto a : e.archs)
      if (a == arch_id) return e.open(opt, cfg, arch_id);
  throw OptionError("no serving adapter for arch " + std::string(arch_id));
}

std::string plan_backend(const Options &opt, const model::ModelConfig &cfg, std::string_view arch_id) {
  for (const auto &e : backend_registry())
    for (const auto a : e.archs)
      if (a == arch_id) {
        if (!e.plan) throw OptionError("--plan: the " + std::string(e.name) + " runtime has no memory plan (its weights are all in VRAM)");
        return e.plan(opt, cfg, arch_id);
      }
  throw OptionError("no serving adapter for arch " + std::string(arch_id));
}

}  // namespace xenithium:serve
