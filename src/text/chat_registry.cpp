// Chat formats by arch id: one line per arch that has one.
#include "xenithium/text/chat_formats.hpp"
#include "xenithium/text/chat.hpp"

#include <fstream>
#include <sstream>

#include "xenithium/text/jinja.hpp"

namespace xenithium::text {

Flag flag_of(const json::Value *v) {
  if (!v || v->is_null()) return Flag::unset;
  if (v->is_bool()) return v->b ? Flag::yes : Flag::no;
  return Flag::other;
}

std::unique_ptr<ChatFormat> open_chat_format(std::string_view arch_id) {
  if (arch_id == "qwen38") return make_chat_format();
  // the dense Qwen3.8-27B ships the same chat_template.jinja byte for byte and the same special tokens:
  // the same format
  if (arch_id == "qwen35") return make_chat_format();
  // Llama-family checkpoints: the one served (Nanbeige4.3-33B) ships ChatML with <think> and XML tool calls
  if (arch_id == "llama") return make_chat_format(ChatTemplate::nanbeige43);
  // Recursive Flint: its own role-marker template and the GLM-4.7-style tool-call parser
  if (arch_id == "flint") return make_chat_format(ChatTemplate::flint);
  return nullptr;
}

std::unique_ptr<ChatFormat> open_chat_format(std::string_view arch_id, const std::string &model_dir, std::string_view source, std::string *note) {
  auto base = open_chat_format(arch_id);
  auto say = [&](const std::string &s) {
    if (note) *note = s;
  };
  if (!base) {
    say("no chat format for arch " + std::string(arch_id));
    return nullptr;
  }
  if (source == "builtin") {
    say("built-in renderer (arch " + std::string(arch_id) + ")");
    return base;
  }
  json::Value globals;
  std::string tpl = jinja::load_chat_template(model_dir, &globals), from = "the checkpoint's chat template";
  if (source != "auto" && source != "jinja") {
    std::ifstream f{std::string(source), std::ios::binary};
    if (!f) {
      say("cannot read the chat template " + std::string(source));
      return nullptr;
    }
    std::ostringstream ss;
    ss << f.rdbuf();
    tpl = ss.str();
    from = std::string(source);
  }
  if (tpl.empty()) {
    if (source == "jinja") {
      say("the checkpoint ships no chat template");
      return nullptr;
    }
    say("built-in renderer (the checkpoint ships no chat template)");
    return base;
  }
  std::string err;
  auto t = jinja::Template::parse(tpl, &err);
  if (!t) {
    if (source != "auto") {
      say(from + " does not parse: " + err);
      return nullptr;
    }
    say("built-in renderer (" + from + " does not parse: " + err + ")");
    return base;
  }
  // a developer role the template knows is kept, else it is renamed to system
  const bool developer = tpl.find("'developer'") != std::string::npos || tpl.find("\"developer\"") != std::string::npos;
  say("Jinja renderer over " + from + (t->iterates_content() ? " (content as parts)" : " (content as one string)"));
  return make_jinja_chat_format(std::move(base), std::move(t), std::move(globals), developer);
}

}  // namespace xenithium:text
