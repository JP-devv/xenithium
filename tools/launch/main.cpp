// xenithium-launch: full-screen launcher for xenithium-server. Pick a model, edit the server.env
// settings with their docs beside them, see the exact start-server.sh command, launch, and watch
// the server load and serve (status, metrics, live log). Shipped doc: docs/launcher.md.
// xenithium-launch [--root DIR] [--env FILE] [--models DIR]
// xenithium-launch --check settings catalog vs script vs docs, and the file's issues
// xenithium-launch --render 120x36 [--keys "<down><enter>..."] one frame as text (no terminal)
#include <limits.h>
#include <unistd.h>

#include <cstdio>
#include <cstring>
#include <string>

#include "app.hpp"
#include "settings.hpp"
#include "term.hpp"

using namespace xenithium::launch;

namespace {

std::string exe_root() {
  char buf[PATH_MAX];
  const ssize_t n = readlink("/proc/self/exe", buf, sizeof buf - 1);
  if (n <= 0) return ".";
  std::string p(buf, static_cast<size_t>(n));
  for (int i = 0; i < 2; ++i) {  // <root>/build/xenithium-launch -> <root>
    const size_t sl = p.find_last_of('/');
    if (sl == std::string::npos) return ".";
    p.resize(sl);
  }
  return p.empty() ? "/" : p;
}

// "<down><enter>abc" -> keys.
std::vector<Key> parse_keys(const std::string &s) {
  std::vector<Key> out;
  for (size_t i = 0; i < s.size();) {
    if (s[i] == '<') {
      const size_t e = s.find('>', i);
      if (e != std::string::npos) {
        const std::string n = s.substr(i + 1, e - i - 1);
        const std::pair<const char *, Key::Type> names[] = {{"up", Key::Up},     {"down", Key::Down},   {"left", Key::Left},  {"right", Key::Right},
                                                            {"enter", Key::Enter}, {"esc", Key::Esc},    {"tab", Key::Tab},    {"bs", Key::Backspace},
                                                            {"pgup", Key::PgUp}, {"pgdn", Key::PgDn},   {"home", Key::Home},  {"end", Key::End},
                                                            {"f5", Key::F5},     {"del", Key::Delete}};
        bool hit = false;
        for (const auto &[name, t] : names)
          if (n == name) {
            out.push_back(Key::of(t));
            hit = true;
          }
        if (!hit && n.size() == 3 && n[0] == 'c' && n[1] == '-') {
          out.push_back(Key::ctrl(n[2]));
          hit = true;
        }
        if (hit) {
          i = e + 1;
          continue;
        }
      }
    }
    out.push_back(Key::chr(utf8_next(s, i)));
  }
  return out;
}

void usage() {
  std::printf(
      "xenithium-launch: interactive launcher for xenithium-server (docs/launcher.md)\n"
      "  --root DIR      the repo (default: the parent of the binary's build/ directory)\n"
      "  --env FILE      settings file to open (default <root>/server.env)\n"
      "  --models DIR    the models folder to scan for one run (default: the settings file's XEN_MODELS_DIR; empty = the project's models/)\n"
      "  --bench-bin P   the xenithium-bench binary (default <root>/build/xenithium-bench)\n"
      "  --bench-runs D  the benchmark runs dir (default $XEN_BENCH_RUNS, else ~/.xenithium/bench/runs)\n"
      "  --experts DIR   MoE expert files, one folder per model (default <root>/.cache/experts)\n"
      "  --model-envs DIR per-model settings files <id>.env (default <root>/models)\n"
      "  --check         print catalog warnings and the file's invalid settings, exit 1 if any\n"
      "  --render WxH    draw one frame as text after --keys (tests); --live watches processes\n");
}

}  // namespace

int main(int argc, char **argv) {
  AppConfig cfg;
  cfg.root = exe_root();
  bool check = false, live_render = false;
  std::string render, keys;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto val = [&]() -> std::string {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "%s needs a value\n", a.c_str());
        std::exit(2);
      }
      return argv[++i];
    };
    if (a == "--root") cfg.root = val();
    else if (a == "--env") cfg.env_path = val();
    else if (a == "--models") cfg.models_dir = val();
    else if (a == "--bench-bin") cfg.bench_bin = val();
    else if (a == "--bench-runs") cfg.bench_runs = val();
    else if (a == "--experts") cfg.experts_dir = val();
    else if (a == "--model-envs") cfg.model_envs_dir = val();
    else if (a == "--check") check = true;
    else if (a == "--render") render = val();
    else if (a == "--keys") keys = val();
    else if (a == "--live") live_render = true;
    else if (a == "-h" || a == "--help") {
      usage();
      return 0;
    } else {
      std::fprintf(stderr, "unknown option %s\n", a.c_str());
      usage();
      return 2;
    }
  }
  if (!is_file(join_path(cfg.root, "scripts/start-server.sh"))) {
    std::fprintf(stderr, "xenithium-launch: %s has no scripts/start-server.sh (use --root)\n", cfg.root.c_str());
    return 2;
  }
  if (!cfg.env_path.empty() && cfg.env_path[0] != '/') {
    char cwd[PATH_MAX];
    if (getcwd(cwd, sizeof cwd)) cfg.env_path = join_path(cwd, cfg.env_path);
  }

  if (check) {
    const Catalog cat = load_catalog(cfg.root);
    int bad = 0;
    for (const auto &w : cat.warnings) {
      std::printf("catalog: %s\n", w.c_str());
      ++bad;
    }
    const std::string env = cfg.env_path.empty() ? join_path(cfg.root, "server.env") : cfg.env_path;
    std::string text;
    Values v = defaults_of(cat);
    if (is_pointer_env(env)) {  // server.env: the model it starts with that model's own settings under it (start-server.sh)
      const std::string md = cfg.model_envs_dir.empty() ? join_path(cfg.root, "models") : cfg.model_envs_dir;
      const ModelEnvView mv = read_model_env(cat, cfg.root, md, env);
      for (const auto &[k, x] : mv.file_values) v[k] = x;
      for (const auto &[k, x] : mv.setup_values) v[k] = x;  // the setup's file over the model's (start-server.sh)
      for (const auto &[k, x] : mv.env_values) v[k] = x;
      v["XEN_MODEL"] = mv.model;
      std::printf("settings: %s%s\n", mv.file.empty() ? "no per-model file" : tilde(mv.file_exists ? mv.file_source : mv.file).c_str(),
                  mv.file.empty() || mv.file_exists ? "" : " (not created yet)");
      if (!mv.setup_values.empty()) std::printf("setup: %s\n", tilde(mv.setup_file).c_str());
    } else if (read_text(env, text)) {
      for (const auto &[k, x] : EnvFile::parse(text).values())
        if (cat.find(k)) v[k] = x;
    }
    const ModelInfo m = probe_model(v["XEN_MODEL"]);
    std::printf("model: %s (%s)\n", tilde(m.path).c_str(), m.arch.empty() ? m.error.c_str() : m.arch.c_str());
    if (is_moe_arch(m.arch)) {  // the expert files: not problems (the default placement and a first run work), but said
      const std::string ed = cfg.experts_dir.empty() ? join_path(cfg.root, ".cache/experts") : cfg.experts_dir;
      const ExpertFile rf = expert_file(v["XEN_HOT_RANK"], "hot-rank.xrc", cfg.root, ed, m);
      std::printf("experts: ranking %s%s%s\n", rf.path.empty() ? "none (not recorded)" : tilde(rf.path).c_str(),
                  rf.path.empty() ? "" : rf.exists ? (rf.error.empty() ? " (recorded)" : " (unusable)") : " (none yet: the server records it while serving)",
                  rf.error.empty() ? "" : (": " + rf.error).c_str());
    }
    for (const auto &is : validate(cat, v, m.arch, cfg.root, max_context_of(m), 0, trained_positions_of(m))) {
      std::printf("%s %s: %s\n", is.error ? "error" : "warning", is.key.c_str(), is.msg.c_str());
      if (is.error) ++bad;
    }
    std::printf("%zu settings, %d problem(s)\n", cat.specs.size(), bad);
    return bad ? 1 : 0;
  }

  if (!render.empty()) {
    int w = 120, h = 36;
    if (std::sscanf(render.c_str(), "%dx%d", &w, &h) != 2) {
      std::fprintf(stderr, "--render WxH\n");
      return 2;
    }
    cfg.live = live_render;
    App app(cfg);
    Canvas c;
    c.resize(w, h);
    app.draw(c);
    if (live_render) usleep(1500 * 1000);  // the monitor's first polls (processes, readiness, metrics)
    for (const Key &k : parse_keys(keys)) {
      if (!app.handle(k)) break;
      app.draw(c);
      app.settle();  // the plans a key started are in before the next key (a popup lists what is known when it opens)
    }
    app.draw(c);
    app.settle();  // the memory plan runs off the UI thread
    app.draw(c);
    for (int y = 0; y < c.h(); ++y) std::printf("%s\n", c.row(y).c_str());
    return 0;
  }

  Terminal term;
  std::string err;
  if (!term.open(&err)) {
    std::fprintf(stderr, "xenithium-launch: %s\n", err.c_str());
    return 1;
  }
  std::string note;
  {
    App app(cfg);
    Canvas c;
    for (;;) {
      int w, h;
      term.size(w, h);
      if (c.w() != w || c.h() != h) c.resize(w, h);
      app.draw(c);
      term.present(c);
      Key k;
      // While an answer streams, poll fast (the deltas would otherwise land in ~200 ms
      // bursts); idle, sleep longer.
      if (!term.read_key(k, app.frame_wait_ms())) continue;
      if (k.type == Key::Ctrl && k.ch == 'z') {
        term.suspend();
        continue;
      }
      if (k.type == Key::Ctrl && k.ch == 'l') {
        term.invalidate();
        continue;
      }
      if (!app.handle(k)) break;
    }
    note = app.exit_note();
    term.close();
  }
  if (!note.empty()) std::printf("%s\n", note.c_str());
  return 0;
}
