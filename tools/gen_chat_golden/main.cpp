// xenithium-gen-chat-golden: a chat-format golden from the engine's own Jinja interpreter (text/jinja.hpp) over a
// checkpoint's chat template: every render case of a cases file rendered to its prompt (or the error the template /
// the request normalisation raises), parser cases copied through (parsing is the arch's output parser, not the
// template's). Host only. With --check, the cases file's own prompts are compared instead (exit 1 on a difference):
// that is how a golden made by the reference stack gates the interpreter.
// xenithium-gen-chat-golden --model DIR --cases FILE [--out FILE] [--arch ID] [--template auto|jinja|PATH] [--check]
// The cases file is a golden's layout: {"render": [{"name", "request": {"messages"[, "tools"]}, "options"}], "parser":
// [...]} (prompts ignored unless --check). --arch defaults to the checkpoint's (config.json through the arch registry).
// Exit status: 0 = written / all equal, 1 = differences (--check) or a case the template could not render where the
// cases file had a prompt, 2 = usage / I/O error.
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>

#include "xenithium/model/arch.hpp"
#include "xenithium/model/config.hpp"
#include "xenithium/text/chat.hpp"
#include "xenithium/util/json.hpp"

namespace json = xenithium::json;
namespace text = xenithium::text;

namespace {

void usage() {
  std::fprintf(stderr,
               "usage: xenithium-gen-chat-golden --model DIR --cases FILE [--out FILE] [--arch ID] [--template auto|jinja|PATH] [--check]\n");
}

bool read_file(const std::string &p, std::string &s) {
  std::ifstream f(p, std::ios::binary);
  if (!f) return false;
  std::ostringstream ss;
  ss << f.rdbuf();
  s = ss.str();
  return true;
}

std::string show(const std::string &s) {
  std::string o;
  json::write_string(s, o);
  return o;
}

}  // namespace

int main(int argc, char **argv) {
  std::string model, cases, out, arch, source = "jinja";
  bool check = false;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto val = [&]() -> std::string {
      if (i + 1 >= argc) {
        usage();
        std::exit(2);
      }
      return argv[++i];
    };
    if (a == "--model") model = val();
    else if (a == "--cases") cases = val();
    else if (a == "--out") out = val();
    else if (a == "--arch") arch = val();
    else if (a == "--template") source = val();
    else if (a == "--check") check = true;
    else if (a == "-h" || a == "--help") return usage(), 0;
    else return usage(), 2;
  }
  if (model.empty() || cases.empty() || (out.empty() && !check)) return usage(), 2;
  try {
    if (arch.empty()) arch = xenithium::model::open_arch(xenithium::model::ModelConfig::load(model))->id();
  } catch (const std::exception &e) {
    std::fprintf(stderr, "gen-chat-golden: %s (pass --arch)\n", e.what());
    return 2;
  }
  std::string note;
  auto fmt = text::open_chat_format(arch, model, source, &note);
  if (!fmt) {
    std::fprintf(stderr, "gen-chat-golden: %s\n", note.c_str());
    return 2;
  }
  std::fprintf(stderr, "gen-chat-golden: arch %s, %s\n", arch.c_str(), note.c_str());
  std::string src, err;
  json::Value in;
  if (!read_file(cases, src) || !json::parse(src, in, &err) || !in.get_array("render")) {
    std::fprintf(stderr, "gen-chat-golden: cannot read the cases %s %s\n", cases.c_str(), err.c_str());
    return 2;
  }
  json::Value golden = json::Value::object();
  golden.set("about", json::Value::string("Chat-format golden rendered by xenithium's Jinja interpreter (" + note + ", arch " + arch +
                                           ") from " + model + "; parser cases copied from " + cases));
  json::Value render = json::Value::array();
  int diffs = 0, failed = 0, n = 0;
  for (const auto &c : in.get_array("render")->arr) {
    ++n;
    const std::string *name = c.get_string("name");
    const json::Value *req = c.get("request");
    if (!name || !req || !req->get("messages")) {
      std::fprintf(stderr, "gen-chat-golden: case %d has no name / request.messages\n", n);
      return 2;
    }
    static const json::Value kNone = json::Value::object();
    const json::Value *opts = c.get("options") ? c.get("options") : &kNone;
    text::ChatOptions opt;
    if (const json::Value *v = opts->get("add_generation_prompt")) opt.add_generation_prompt = v->is_bool() && v->b;
    opt.enable_thinking = text::flag_of(opts->get("enable_thinking"));
    opt.preserve_thinking = text::flag_of(opts->get("preserve_thinking"));
    opt.reasoning_effort = opts->get("reasoning_effort");
    json::Value kw = json::Value::object();
    for (const auto &m : opts->obj)
      if (m.first != "add_generation_prompt") kw.set(m.first, m.second);
    opt.kwargs = &kw;
    std::string prompt, rerr;
    const bool ok = fmt->render(*req->get("messages"), req->get("tools"), opt, prompt, rerr);
    json::Value o = json::Value::object();
    o.set("name", json::Value::string(*name));
    o.set("request", *req);
    o.set("options", *opts);
    if (ok) {
      o.set("prompt", json::Value::string(prompt));
    } else {
      o.set("error", json::Value::string(rerr));
      o.set("error_exact", json::Value::boolean(true));
    }
    if (const std::string *want = c.get_string("prompt")) {
      if (!ok) {
        ++failed;
        std::printf("[FAIL] %s: rejected (%s), the cases file has a prompt\n", name->c_str(), rerr.c_str());
      } else if (prompt != *want) {
        ++diffs;
        size_t i = 0;
        while (i < prompt.size() && i < want->size() && prompt[i] == (*want)[i]) ++i;
        std::printf("[DIFF] %s at byte %zu\n    want: %s\n    got:  %s\n", name->c_str(), i, show(want->substr(i, 100)).c_str(),
                    show(prompt.substr(i, 100)).c_str());
      }
    } else if (const std::string *werr = c.get_string("error"); werr && check) {
      if (ok) {
        ++diffs;
        std::printf("[DIFF] %s: rendered, the cases file has an error\n", name->c_str());
      } else if (const json::Value *ex = c.get("error_exact"); ex && ex->is_bool() && ex->b && rerr != *werr) {
        ++diffs;
        std::printf("[DIFF] %s: error\n    want: %s\n    got:  %s\n", name->c_str(), werr->c_str(), rerr.c_str());
      }
    }
    render.push(std::move(o));
  }
  golden.set("render", std::move(render));
  golden.set("parser", in.get("parser") ? *in.get("parser") : json::Value::array());
  std::printf("gen-chat-golden: %d render cases, %d differ from the cases file, %d failed\n", n, diffs, failed);
  if (!out.empty()) {
    std::ofstream f(out, std::ios::binary);
    if (!f) {
      std::fprintf(stderr, "gen-chat-golden: cannot write %s\n", out.c_str());
      return 2;
    }
    std::string s;
    json::write(golden, s, json::Style::python);
    f << s << '\n';
  }
  return (check && diffs) || failed ? 1 : 0;
}
