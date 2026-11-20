// A background job the launcher starts and watches: one command in its own session, its output appended to a log
// file. A scripts/make-hot-rank.sh run is watched with it. The job outlives the launcher (its own session); a later
// launcher re-adopts it from its pid file and then only sees whether it is alive (no exit status).
#pragma once

#include <sys/types.h>

#include <chrono>
#include <climits>
#include <string>
#include <utility>
#include <vector>

namespace xenithium::launch {

class Job {
 public:
  using Clock = std::chrono::steady_clock;
  struct State {
    bool active = false;
    bool adopted = false;  // started by an earlier launcher: no exit status
    pid_t pid = -1;
    Clock::time_point started{};
    int exit_code = INT_MIN;  // INT_MIN = running / never ran; -1000 = ended, status unknown (adopted)
    bool cancelled = false;
    std::string title, log, error;
  };

  Job() = default;
  ~Job() = default;
  Job(const Job &) = delete;
  Job &operator=(const Job &) = delete;

  // argv[0] by PATH lookup, in cwd, extra environment, stdout + stderr appended to log. False (state.error) when one
  // runs already or it could not start.
  bool start(const std::vector<std::string> &argv, const std::vector<std::pair<std::string, std::string>> &env, const std::string &cwd,
             const std::string &log, const std::string &title);
  // A job an earlier launcher started (its pid file): watched by pid only.
  void adopt(pid_t pid, const std::string &log, const std::string &title, double age_s);
  void cancel();  // SIGTERM to its process group (the script and the probe it execs)
  bool poll();  // reaps; true when the job just ended
  const State &state() const { return st_; }
  bool busy() const { return st_.active; }

 private:
  State st_;
};

}  // namespace xenithium:launch
