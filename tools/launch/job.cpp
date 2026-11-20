// Background jobs of the launcher.
#include "job.hpp"

#include <fcntl.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <cstdlib>
#include <cstring>

namespace xenithium::launch {

bool Job::start(const std::vector<std::string> &argv, const std::vector<std::pair<std::string, std::string>> &env, const std::string &cwd,
                const std::string &log, const std::string &title) {
  if (st_.active) {
    st_.error = "a job is already running";
    return false;
  }
  if (argv.empty()) {
    st_.error = "no command";
    return false;
  }
  const int lf = open(log.c_str(), O_WRONLY | O_CREAT | O_APPEND, 0644);
  if (lf < 0) {
    st_.error = "cannot open " + log + ": " + std::strerror(errno);
    return false;
  }
  const pid_t pid = fork();
  if (pid < 0) {
    st_.error = std::string("fork: ") + std::strerror(errno);
    close(lf);
    return false;
  }
  if (pid == 0) {
    setsid();  // its own session and process group: survives the launcher, cancel reaches the whole group
    dup2(lf, 1);
    dup2(lf, 2);
    const int dn = open("/dev/null", O_RDONLY);
    if (dn >= 0) dup2(dn, 0);
    if (!cwd.empty() && chdir(cwd.c_str()) != 0) _exit(126);
    for (const auto &[k, v] : env) setenv(k.c_str(), v.c_str(), 1);
    std::vector<char *> av;
    for (const auto &a : argv) av.push_back(const_cast<char *>(a.c_str()));
    av.push_back(nullptr);
    execvp(av[0], av.data());
    _exit(127);
  }
  close(lf);
  st_ = {};
  st_.active = true;
  st_.pid = pid;
  st_.started = Clock::now();
  st_.title = title;
  st_.log = log;
  return true;
}

void Job::adopt(pid_t pid, const std::string &log, const std::string &title, double age_s) {
  st_ = {};
  st_.active = true;
  st_.adopted = true;
  st_.pid = pid;
  st_.started = Clock::now() - std::chrono::milliseconds(static_cast<int64_t>(age_s * 1000));
  st_.title = title;
  st_.log = log;
}

void Job::cancel() {
  if (!st_.active || st_.pid <= 0) return;
  if (kill(-st_.pid, SIGTERM) != 0) kill(st_.pid, SIGTERM);
  st_.cancelled = true;
}

bool Job::poll() {
  if (!st_.active || st_.pid <= 0) return false;
  if (st_.adopted) {
    if (kill(st_.pid, 0) == 0 || errno == EPERM) return false;
    st_.active = false;
    st_.exit_code = -1000;
    return true;
  }
  int status = 0;
  const pid_t r = waitpid(st_.pid, &status, WNOHANG);
  if (r != st_.pid) return false;
  st_.active = false;
  st_.exit_code = WIFEXITED(status) ? WEXITSTATUS(status) : WIFSIGNALED(status) ? 128 + WTERMSIG(status) : -1;
  return true;
}

}  // namespace xenithium:launch
