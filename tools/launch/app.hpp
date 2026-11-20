// The launcher's screens: Settings (model, settings, help, command preview), Server (status, metrics, live log), Chat
// and Bench, with popups for choices, models, profiles, launch checks and confirmations.
// Drawing goes to a Canvas and keys come in as Key, so tests drive whole flows without a terminal.
#pragma once

#include <memory>
#include <string>

#include "settings.hpp"
#include "term.hpp"

namespace xenithium::launch {

struct AppConfig {
  std::string root;  // the repo (scripts/start-server.sh, docs/server-env.md, build/)
  std::string env_path;  // the file being edited at start (default <root>/server.env)
  std::string models_dir;  // scanned for checkpoints (default ~/models)
  std::string state_dir;  // launch.env, server.log, server.pid (default <root>/build/launch)
  std::string profiles_dir;  // named env files (default <root>/profiles)
  std::string bench_bin;  // the xenithium-bench binary (default <root>/build/xenithium-bench)
  std::string bench_runs;  // where its runs go and are listed from (default $XEN_BENCH_RUNS, else ~/.xenithium/bench/runs)
  std::string experts_dir;  // MoE expert files, one folder per model (default <root>/.cache/experts; docs/server-env.md)
  std::string model_envs_dir;  // per-model settings files <id>.env (default <root>/models; docs/server-env.md)
  bool live = true;  // watch processes, launch and stop servers (off in tests)
};

class App {
 public:
  explicit App(AppConfig cfg);
  ~App();
  App(const App &) = delete;
  App &operator=(const App &) = delete;

  void draw(Canvas &c);
  bool handle(const Key &k);  // false = quit
  std::string exit_note() const;

  // Test access.
  const Values &values() const;
  void set_value(const std::string &key, const std::string &v);
  int tab() const;
  bool dirty() const;
  std::string selected_key() const;
  bool popup_open() const;
  std::string models_dir() const;  // the folder the m list scans (the settings file's XEN_MODELS_DIR; --models wins)
  std::string preview() const;
  // Chat tab (tests): the port chat_send targets in test mode (0 = XEN_PORT's value).
  void set_chat_port(int port);
  // (tests) open a tab by its number, also one the header hides (the Chat tab is hidden for now; kChat = 2).
  void show_tab(int t);
  // (tests) a click point of the shown tab's header chip of the given tab number, from the last frame (false when none).
  bool tab_click_point(int t, int &x, int &y) const;
  bool chat_busy() const;
  std::string chat_answer() const;  // the last assistant answer's text ("" when none)
  bool chat_failed() const;
  // Bench tab (tests): the runs dir to scan, the model groups it found (newest run first),
  // the total run count, and the weighted decode of the selected run ("" when none selected).
  void set_bench_runs_dir(const std::string &dir);
  std::vector<std::string> bench_model_names() const;
  int bench_run_count() const;
  std::string bench_selected_decode() const;
  // How long the main loop may sleep before redrawing: short while an answer streams (the
  // deltas would otherwise land in ~200 ms bursts), long otherwise.
  int frame_wait_ms() const;
  // Waits for the background work a frame shows (the memory plan of a MoE model), then folds it in: --render and tests.
  void settle();
  MemPlanView memory_plan() const;  // the current plan (ok = false when none)

  struct Impl;

 private:
  std::unique_ptr<Impl> p_;
};

}  // namespace xenithium:launch
