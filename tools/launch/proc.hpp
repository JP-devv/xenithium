// Processes the launcher runs or watches: the start script (dry run and launch), running servers,
// the port, Docker containers holding the cards, and the server's /metrics.
#pragma once

#include <sys/types.h>

#include <map>
#include <string>
#include <utility>
#include <vector>

namespace xenithium::launch {

struct RunResult {
  int status = -1;  // exit code, or -1 when it could not run / timed out
  bool timed_out = false;
  std::string out, err;
};
// Runs argv[0] (PATH lookup) in `cwd` with extra environment variables; captures stdout and stderr.
RunResult run_capture(const std::vector<std::string> &argv, const std::vector<std::pair<std::string, std::string>> &env,
                      const std::string &cwd, int timeout_ms);

// scripts/start-server.sh detached (own session) with SERVE_ENV=env_file, output to log_file.
// The script execs the server, so the pid is the server's. Returns -1 and sets err on failure.
pid_t spawn_server(const std::string &root, const std::string &env_file, const std::string &log_file, std::string &err);

struct ProcInfo {
  pid_t pid = 0;
  std::string cmdline;  // arguments joined by spaces
  int port = 0;  // --port from the command line (8000 when absent)
};
// Running xenithium-server processes (by /proc/PID/comm, which the kernel cuts to 15 chars).
std::vector<ProcInfo> find_servers();
bool pid_alive(pid_t pid);
// Live processes whose /proc/PID/comm is exactly `comm` (the kernel cuts it to 15 chars: xenithium-probe, xenithium-serve).
std::vector<pid_t> pids_named(const std::string &comm);
// Seconds since the process started (/proc/PID/stat), -1 when unknown.
double proc_age(pid_t pid);
bool port_in_use(int port);
// Names of running Docker containers; docker_ok = false when docker could not be asked.
// Running containers' names, except this project's own (label org.xenithium); docker_ok = the docker CLI answered.
std::vector<std::string> docker_containers(bool &docker_ok);
// The first non-loopback IPv4 address (for the LAN URL), "" when none.
std::string lan_ip();

// Why a server ended, as one line for the launcher's header and a "launcher:" line in its log. code = the exit status
// (128 + N for signal N; -1000 = unknown, the server was adopted from its pid file). A SIGKILL (137) or an unknown end
// is looked up in the kernel log (journalctl -k, else dmesg) since `since_epoch`: an OOM kill of this pid names the RSS
// it had; a server cannot log its own SIGKILL. killed_here = the launcher sent the SIGKILL (Stop -> Kill now).
std::string exit_reason(pid_t pid, int code, long long since_epoch, bool killed_here);
// The kernel log's OOM kill of `pid` (pure, for tests): "Killed process PID (...)" with its anon-rss; false when absent.
bool find_oom_kill(const std::string &kernel_log, pid_t pid, double *anon_rss_gib, std::string *when);
// The plain-words part of exit_reason for a status (no kernel log), e.g. "crashed: segmentation fault (SIGSEGV)".
std::string describe_exit(int code);

// Prometheus text: metric name -> value summed over its label sets.
std::map<std::string, double> parse_metrics(const std::string &text);

}  // namespace xenithium:launch
