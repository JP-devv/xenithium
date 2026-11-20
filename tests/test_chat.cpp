// Host-only gate for qwen38's chat format against the golden
// (tests/data/chat_golden.json):
// render: every case byte-equal to the golden's prompt, or rejected (with the golden's exact
// message where the case asks for it).
// parser: every model output parsed to the golden's non-streaming reasoning / content /
// tool calls, fed whole, one byte at a time and in random chunks (all identical).
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "xenithium/text/chat_formats.hpp"
#include "xenithium/text/chat.hpp"
#include "xenithium/util/json.hpp"

namespace {
namespace json = xenithium::json;
namespace text = xenithium::text;

int g_checks = 0, g_failures = 0;
bool g_verbose = false;

bool check(bool ok, const std::string &what) {
  ++g_checks;
  if (!ok) {
    ++g_failures;
    std::printf("[FAIL] %s\n", what.c_str());
  } else if (g_verbose) {
    std::printf("[PASS] %s\n", what.c_str());
  }
  return ok;
}

std::string show(const std::string &s) {
  std::string o;
  json::write_string(s, o);
  return o;
}

// Python str.lstrip over the whitespace a model actually emits around content.
std::string lstrip_py(const std::string &s) {
  size_t i = 0;
  while (i < s.size()) {
    const auto c = static_cast<unsigned char>(s[i]);
    if (c == ' ' || (c >= 0x09 && c <= 0x0d) || (c >= 0x1c && c <= 0x1f)) {
      ++i;
    } else if (s.compare(i, 2, "\xc2\xa0") == 0 || s.compare(i, 2, "\xc2\x85") == 0) {
      i += 2;
    } else if (s.compare(i, 3, "\xe3\x80\x80") == 0 || s.compare(i, 3, "\xe2\x80\xa8") == 0 ||
               s.compare(i, 3, "\xe2\x80\xa9") == 0) {
      i += 3;
    } else {
      break;
    }
  }
  return s.substr(i);
}

// ---- render ------------------------------------------------------------------------------------

void render_case(const text::ChatFormat &fmt, const json::Value &c) {
  const std::string name = "render/" + *c.get_string("name");
  const json::Value &req = *c.get("request");
  const json::Value *opts = c.get("options");
  text::ChatOptions opt;
  if (const json::Value *v = opts->get("add_generation_prompt")) opt.add_generation_prompt = v->is_bool() && v->b;
  opt.enable_thinking = text::flag_of(opts->get("enable_thinking"));
  opt.preserve_thinking = text::flag_of(opts->get("preserve_thinking"));
  opt.reasoning_effort = opts->get("reasoning_effort");
  // every option but add_generation_prompt is a chat_template_kwargs member (the Jinja renderer passes them all)
  json::Value kw = json::Value::object();
  for (const auto &m : opts->obj)
    if (m.first != "add_generation_prompt") kw.set(m.first, m.second);
  opt.kwargs = &kw;
  std::string out, err;
  const bool ok = fmt.render(*req.get("messages"), req.get("tools"), opt, out, err);
  if (const std::string *want = c.get_string("prompt")) {
    if (!check(ok && out == *want, name)) {
      if (!ok) std::printf("    rejected: %s\n", err.c_str());
      size_t i = 0;
      while (i < out.size() && i < want->size() && out[i] == (*want)[i]) ++i;
      std::printf("    first difference at byte %zu\n    want: %s\n    got:  %s\n", i,
                  show(want->substr(i, 120)).c_str(), show(out.substr(i, 120)).c_str());
    }
    return;
  }
  const json::Value *exact = c.get("error_exact");
  const std::string &want = *c.get_string("error");
  if (!check(!ok, name + " rejected")) return;
  if (exact && exact->is_bool() && exact->b && !check(err == want, name + " message"))
    std::printf("    want: %s\n    got:  %s\n", want.c_str(), err.c_str());
}

// ---- parser ------------------------------------------------------------------------------------

struct Parsed {
  std::string reasoning, content;
  std::vector<std::pair<std::string, std::string>> calls;
  bool well_formed = true;  // tool_start then its tool_args, indices 0, 1...
  int count = 0;
  bool markup = false;  // the parser's tool_markup
  bool operator==(const Parsed &o) const {
    return reasoning == o.reasoning && content == o.content && calls == o.calls;
  }
};

bool g_no_tool_calls = false;  // the case's request has tool_choice "none" (parser_no_tool_calls)

Parsed run_parser(const text::ChatFormat &fmt, bool thinking, const json::Value *tools,
                  const std::string &output, const std::vector<size_t> &cuts) {
  auto p = g_no_tool_calls ? fmt.parser_no_tool_calls(thinking) : fmt.parser(thinking, tools);
  std::vector<text::Delta> d;
  size_t from = 0;
  for (size_t cut : cuts) {
    p->feed(std::string_view(output).substr(from, cut - from), d);
    from = cut;
  }
  p->feed(std::string_view(output).substr(from), d);
  p->finish(d);
  Parsed r;
  for (const auto &x : d) {
    switch (x.kind) {
      case text::Delta::reasoning: r.reasoning += x.text; break;
      case text::Delta::content: r.content += x.text; break;
      case text::Delta::tool_start:
        r.well_formed &= x.index == static_cast<int>(r.calls.size());
        r.calls.push_back({x.text, ""});
        break;
      case text::Delta::tool_args:
        r.well_formed &= !r.calls.empty() && x.index == static_cast<int>(r.calls.size()) - 1 &&
                         r.calls.back().second.empty();
        if (!r.calls.empty()) r.calls.back().second = x.text;
        break;
    }
  }
  r.count = p->tool_calls();
  r.markup = p->tool_markup();
  return r;
}

std::string describe(const Parsed &p) {
  std::string s = "reasoning=" + show(p.reasoning) + " content=" + show(p.content) + " calls=[";
  for (const auto &c : p.calls) s += c.first + " " + c.second + "; ";
  return s + "]";
}

int g_deviations = 0;

void parser_case(const text::ChatFormat &fmt, const json::Value &c) {
  const std::string name = "parser/" + *c.get_string("name");
  if (const std::string *skip = c.get_string("skip")) {
    std::printf("[SKIP] %s: %s\n", name.c_str(), skip->c_str());
    return;
  }
  const std::string &output = *c.get_string("output");
  // The parser's view of enable_thinking must match the golden's truthiness.
  const bool thinking = fmt.thinking(c.get("enable_thinking"));
  check(thinking == c.get("thinking")->b, name + " thinking()");
  const json::Value *tools = c.get("tools");
  if (tools && tools->is_null()) tools = nullptr;
  const json::Value *req = c.get_object("request");
  const std::string *tc = req ? req->get_string("tool_choice") : nullptr;
  g_no_tool_calls = tc && *tc == "none" && req->get_array("tools") && !req->get_array("tools")->arr.empty();

  const json::Value &exp = *c.get("expected");
  Parsed want;
  if (const std::string *s = exp.get_string("reasoning")) want.reasoning = *s;
  if (const std::string *s = exp.get_string("content")) want.content = *s;
  for (const auto &tc : exp.get("tool_calls")->arr)
    want.calls.push_back({*tc.get_string("name"), *tc.get_string("arguments")});

  Parsed whole = run_parser(fmt, thinking, tools, output, {});
  // A whole answer goes through finish_content (as the server's non-streamed answers do), which
  // must give the golden's content exactly. The streamed content may differ in one
  // documented way: content followed by tool-call markup keeps its leading whitespace (released
  // before the call was known); the golden's content has it stripped.
  std::string fin = whole.content;
  fmt.finish_content(fin, whole.markup);
  bool same = whole.reasoning == want.reasoning && whole.calls == want.calls && fin == want.content;
  bool deviation = false;
  if (whole.content != want.content) {
    deviation = c.get("reference_tools_called")->b && lstrip_py(whole.content) == want.content;
    same &= deviation;
  }
  if (!check(same && whole.well_formed && whole.count == static_cast<int>(whole.calls.size()), name)) {
    std::printf("    want: %s\n    got:  %s\n", describe(want).c_str(), describe(whole).c_str());
  } else if (deviation) {
    ++g_deviations;
    std::printf("[DEV ] %s: content %s, reference %s (leading whitespace before a tool call)\n",
                name.c_str(), show(whole.content).c_str(), show(want.content).c_str());
  }

  // Chunking must not matter: one byte at a time, and random splits.
  std::vector<size_t> bytes;
  for (size_t i = 1; i < output.size(); ++i) bytes.push_back(i);
  const Parsed by_byte = run_parser(fmt, thinking, tools, output, bytes);
  if (!check(by_byte == whole && by_byte.well_formed, name + " 1-byte chunks"))
    std::printf("    whole: %s\n    bytes: %s\n", describe(whole).c_str(), describe(by_byte).c_str());
  std::mt19937 rng(static_cast<uint32_t>(output.size() * 2654435761u + 7));
  for (int round = 0; round < 8; ++round) {
    std::vector<size_t> cuts;
    for (size_t i = 1; i < output.size(); ++i)
      if (rng() % 5 == 0) cuts.push_back(i);
    const Parsed split = run_parser(fmt, thinking, tools, output, cuts);
    if (!check(split == whole && split.well_formed, name + " random chunks #" + std::to_string(round))) {
      std::printf("    whole: %s\n    split: %s\n", describe(whole).c_str(), describe(split).c_str());
      break;
    }
  }
}

std::string default_golden(const char *file) {
  std::string f = __FILE__;
  const size_t slash = f.find_last_of('/');
  return (slash == std::string::npos ? std::string(".") : f.substr(0, slash)) + "/data/" + file;
}

bool load_golden(const std::string &path, json::Value &golden) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    std::printf("chat: cannot open %s\n", path.c_str());
    return false;
  }
  std::stringstream ss;
  ss << in.rdbuf();
  std::string err;
  if (!json::parse(ss.str(), golden, &err)) {
    std::printf("chat: bad golden %s: %s\n", path.c_str(), err.c_str());
    return false;
  }
  return true;
}

}  // namespace

int main(int argc, char **argv) {
  std::string path = default_golden("chat_golden.json");
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "-v") g_verbose = true;
    else path = a;
  }
  json::Value golden;
  if (!load_golden(path, golden)) return 2;
  auto fmt = text::open_chat_format("qwen38");
  if (!check(fmt != nullptr, "open_chat_format(qwen38)")) return 1;

  const auto &render = golden.get_array("render")->arr;
  const auto &parser = golden.get_array("parser")->arr;
  for (const auto &c : render) render_case(*fmt, c);
  for (const auto &c : parser) parser_case(*fmt, c);

  // The bool overload is the schema-less form: same parse, arguments left as strings.
  {
    auto p = fmt->parser(true, true);
    std::vector<text::Delta> d;
    p->feed("r</think><tool_call><function=f><parameter=n>\n5\n</parameter></function></tool_call>", d);
    p->finish(d);
    check(d.size() == 3 && d[1].kind == text::Delta::tool_start && d[1].text == "f" &&
              d[2].text == "{\"n\": \"5\"}" && p->tool_calls() == 1,
          "parser(thinking, bool): untyped arguments");
  }

  // Nanbeige4.3 (arch llama): its template's renders (the parser is the same as above).
  size_t nb_render = 0;
  {
    json::Value nb;
    auto fmt_nb = text::open_chat_format("llama");
    if (check(fmt_nb != nullptr, "open_chat_format(llama)") && load_golden(default_golden("chat_golden_nanbeige.json"), nb)) {
      const auto &r = nb.get_array("render")->arr;
      for (const auto &c : r) render_case(*fmt_nb, c);
      nb_render = r.size();
    } else {
      check(false, "chat_golden_nanbeige.json");
    }
  }

  // Recursive Flint (arch flint): its template's renders and its own (GLM-4.7-style) parser.
  size_t fl_render = 0, fl_parser = 0;
  {
    json::Value fl;
    auto fmt_fl = text::open_chat_format("flint");
    if (check(fmt_fl != nullptr, "open_chat_format(flint)") && load_golden(default_golden("chat_golden_flint.json"), fl)) {
      check(fmt_fl->exclude_tools_on_none(), "flint: tool_choice none renders no tools");
      const auto &r = fl.get_array("render")->arr;
      for (const auto &c : r) render_case(*fmt_fl, c);
      const auto &pc = fl.get_array("parser")->arr;
      for (const auto &c : pc) parser_case(*fmt_fl, c);
      fl_render = r.size(), fl_parser = pc.size();
    } else {
      check(false, "chat_golden_flint.json");
    }
  }

  // The Jinja renderer (text/jinja.hpp) over each checkpoint's own template against the same golden renders: the
  // interpreter's gate. A checkpoint that is not on this machine is skipped.
  size_t jinja_cases = 0;
  {
    const char *home = std::getenv("HOME");
    const std::string models = std::getenv("XENITHIUM_MODELS") ? std::getenv("XENITHIUM_MODELS") : std::string(home ? home : "") + "/models";
    const struct {
      const char *arch, *dir, *golden;
    } sets[] = {{"qwen38", "qwen38-w4a16", "chat_golden.json"},
                {"llama", "nanbeige43-33b-int4", "chat_golden_nanbeige.json"},
                {"flint", "flint-int4", "chat_golden_flint.json"}};
    for (const auto &st : sets) {
      const std::string dir = models + "/" + st.dir;
      std::ifstream probe(dir + "/tokenizer_config.json");
      if (!probe) {
        std::printf("chat: [SKIP] jinja %s: %s not present\n", st.arch, dir.c_str());
        continue;
      }
      std::string note;
      auto jf = text::open_chat_format(st.arch, dir, "jinja", &note);
      if (!check(jf != nullptr, std::string("jinja ") + st.arch + ": " + note)) continue;
      std::printf("chat: jinja %s: %s\n", st.arch, note.c_str());
      json::Value g;
      if (!load_golden(default_golden(st.golden), g)) continue;
      for (const auto &c : g.get_array("render")->arr) {
        json::Value cc = c;
        cc.set("name", json::Value::string(std::string("jinja-") + st.arch + "/" + *c.get_string("name")));
        render_case(*jf, cc);
        ++jinja_cases;
      }
    }
  }

  std::printf("chat: %zu render cases, %zu parser cases, %d leading-whitespace deviations; nanbeige43 %zu render cases; flint %zu render, %zu parser; "
              "jinja %zu render cases\n",
              render.size(), parser.size(), g_deviations, nb_render, fl_render, fl_parser, jinja_cases);
  std::printf("chat: %d checks, %d failed\n", g_checks, g_failures);
  return g_failures ? 1 : 0;
}
