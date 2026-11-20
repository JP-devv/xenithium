// Processes the launcher runs or watches.
#include "proc.hpp"

#include <algorithm>
#include <arpa/inet.h>
#include <dirent.h>
#include <fcntl.h>
#include <ifaddrs.h>
#include <net/if.h>
#include <netinet/in.h>
#include <poll.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>

#include "xenithium/util/host_mem.hpp"

namespace xenithium::launch {

RunResult run_capture(const std::vector<std::string> &argv, const std::vector<std::pair<std::string, std::string>> &env,
                      const std::string &cwd, int timeout_ms) {
  RunResult r;
  int po[2], pe[2];
  if (pipe2(po, O_CLOEXEC) != 0) return r;
  if (pipe2(pe, O_CLOEXEC) != 0) {
    close(po[0]);
    close(po[1]);
    return r;
  }
  const pid_t pid = fork();
  if (pid < 0) {
    for (int fd : {po[0], po[1], pe[0], pe[1]}) close(fd);
    return r;
  }
  if (pid == 0) {
    dup2(po[1], 1);
    dup2(pe[1], 2);
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
  close(po[1]);
  close(pe[1]);
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
  pollfd fds[2] = {{po[0], POLLIN, 0}, {pe[0], POLLIN, 0}};
  int open_fds = 2;
  char buf[4096];
  while (open_fds > 0) {
    const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - std::chrono::steady_clock::now()).count();
    if (left <= 0) {
      r.timed_out = true;
      kill(pid, SIGKILL);
      break;
    }
    const int n = poll(fds, 2, static_cast<int>(left));
    if (n < 0 && errno == EINTR) continue;
    if (n <= 0) continue;
    for (int i = 0; i < 2; ++i) {
      if (fds[i].fd < 0 || !(fds[i].revents & (POLLIN | POLLHUP | POLLERR))) continue;
      const ssize_t got = read(fds[i].fd, buf, sizeof buf);
      if (got > 0) (i == 0 ? r.out : r.err).append(buf, static_cast<size_t>(got));
      else {
        close(fds[i].fd);
        fds[i].fd = -1;
        --open_fds;
      }
    }
  }
  for (auto &f : fds)
    if (f.fd >= 0) close(f.fd);
  int st = 0;
  while (waitpid(pid, &st, 0) < 0 && errno == EINTR) {
  }
  if (!r.timed_out && WIFEXITED(st)) r.status = WEXITSTATUS(st);
  return r;
}

pid_t spawn_server(const std::string &root, const std::string &env_file, const std::string &log_file, std::string &err) {
  const int log = open(log_file.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
  if (log < 0) {
    err = "cannot open " + log_file + ": " + std::strerror(errno);
    return -1;
  }
  const pid_t pid = fork();
  if (pid < 0) {
    err = std::string("fork: ") + std::strerror(errno);
    close(log);
    return -1;
  }
  if (pid == 0) {
    setsid();  // its own session: survives the launcher, no SIGHUP from the terminal
    for (int s : {SIGINT, SIGTERM, SIGHUP, SIGPIPE, SIGWINCH, SIGTSTP}) signal(s, SIG_DFL);
    dup2(log, 1);
    dup2(log, 2);
    const int dn = open("/dev/null", O_RDONLY);
    if (dn >= 0) dup2(dn, 0);
    if (chdir(root.c_str()) != 0) _exit(126);
    setenv("SERVE_ENV", env_file.c_str(), 1);
    unsetenv("SERVE_DRY_RUN");
    execl("/bin/bash", "bash", "scripts/start-server.sh", static_cast<char *>(nullptr));
    _exit(127);
  }
  close(log);
  return pid;
}

bool find_oom_kill(const std::string &kernel_log, pid_t pid, double *anon_rss_gib, std::string *when) {
  const std::string key = "Killed process " + std::to_string(pid) + " (";
  std::istringstream in(kernel_log);
  std::string line, hit;
  while (std::getline(in, line))
    if (line.find(key) != std::string::npos) hit = line;  // the last one (a pid can be reused across boots in dmesg)
  if (hit.empty()) return false;
  if (anon_rss_gib) {
    *anon_rss_gib = -1;
    const size_t a = hit.find("anon-rss:");
    if (a != std::string::npos) *anon_rss_gib = std::atof(hit.c_str() + a + 9) / (1024.0 * 1024.0);  // kB
  }
  if (when) {  // journalctl short-iso: 2026-11-09T00:33:50+0000 host kernel...; dmesg -T: [Mon Nov 9 00:33:50 2026]...
    when->clear();
    const size_t t = hit.find('T');
    if (hit.size() > 19 && hit[4] == '-' && t == 10) *when = hit.substr(11, 8);
    else if (!hit.empty() && hit[0] == '[') {
      const size_t e = hit.find(']');
      if (e != std::string::npos && e > 9) *when = hit.substr(e - 13, 8);
    }
  }
  return true;
}

std::string describe_exit(int code) {
  if (code == 0) return "stopped cleanly";
  if (code == -1000) return "ended (started outside this launcher: its exit status is unknown)";
  if (code == 1) return "exited with status 1: the error is the last 'server:' line of its log";
  if (code == 2) return "refused its settings (status 2): the reason is the last line of its log";
  if (code > 128 && code < 128 + 65) {
    const int sig = code - 128;
    switch (sig) {
      case SIGKILL: return "killed by SIGKILL (exit status 137)";
      case SIGSEGV: return "crashed: segmentation fault (SIGSEGV, exit status 139); the backtrace is at the end of its log";
      case SIGABRT: return "aborted (SIGABRT, exit status 134); the backtrace is at the end of its log";
      case SIGBUS: return "crashed: bus error (SIGBUS, exit status 135); the backtrace is at the end of its log";
      case SIGILL: return "crashed: illegal instruction (SIGILL, exit status 132)";
      case SIGFPE: return "crashed: arithmetic trap (SIGFPE, exit status 136)";
      case SIGTERM: return "ended by SIGTERM before it could stop cleanly (exit status 143)";
      case SIGINT: return "ended by SIGINT (exit status 130)";
      case SIGHUP: return "ended by SIGHUP (exit status 129)";
      default: return "ended by signal " + std::to_string(sig) + " (exit status " + std::to_string(code) + ")";
    }
  }
  return "exited with status " + std::to_string(code) + ": see the end of its log";
}

std::string exit_reason(pid_t pid, int code, long long since_epoch, bool killed_here) {
  std::string base = describe_exit(code);
  if (code != 128 + SIGKILL && code != -1000) return base;
  // the kernel log: journalctl first (readable without root on most systemd hosts), else dmesg
  const std::string since = "@" + std::to_string(std::max(0LL, since_epoch));
  RunResult r = run_capture({"journalctl", "-k", "-o", "short-iso", "--no-pager", "-q", "--since", since}, {}, "/", 4000);
  std::string log = r.status == 0 ? r.out : std::string();
  if (log.empty()) {
    r = run_capture({"dmesg", "-T"}, {}, "/", 3000);
    if (r.status == 0) log = r.out;
  }
  double gib = -1;
  std::string when;
  if (find_oom_kill(log, pid, &gib, &when)) {
    const util::HostMem m = util::host_mem();
    char held[48] = "most", total[24] = "?", buf[360];
    if (gib >= 0) std::snprintf(held, sizeof held, "%.1f GiB", gib);
    if (m.ok) std::snprintf(total, sizeof total, "%.0f", static_cast<double>(m.total) / 1073741824.0);
    std::snprintf(buf, sizeof buf,
                  "killed by the kernel's OOM killer%s%s: the host ran out of RAM (the server held %s of %s GiB). A SIGKILL leaves no "
                  "message in the server log; its 'host RSS' and 'WARNING host RAM low' lines show the growth",
                  when.empty() ? "" : " at ", when.c_str(), held, total);
    return buf;
  }
  if (code == -1000) return base;
  if (killed_here) return "killed by SIGKILL from this launcher (Stop -> Kill now)";
  return base + (log.empty() ? ", not by the OOM killer as far as can be told (the kernel log is not readable here)"
                             : ": not the OOM killer (no kernel log entry for pid " + std::to_string(pid) + "), so another process sent it");
}

bool pid_alive(pid_t pid) {
  if (pid <= 0) return false;
  if (kill(pid, 0) != 0 && errno == ESRCH) return false;
  // A zombie of ours counts as gone.
  std::ifstream f("/proc/" + std::to_string(pid) + "/stat");
  std::string s;
  std::getline(f, s);
  const size_t rp = s.rfind(')');
  return !(rp != std::string::npos && rp + 2 < s.size() && s[rp + 2] == 'Z');
}

double proc_age(pid_t pid) {
  std::ifstream f("/proc/" + std::to_string(pid) + "/stat");
  std::string s;
  std::getline(f, s);
  const size_t rp = s.rfind(')');
  if (rp == std::string::npos) return -1;
  std::istringstream in(s.substr(rp + 2));
  std::string tok;
  for (int i = 3; i <= 22 && in >> tok; ++i)  // field 22: starttime, clock ticks after boot
    if (i == 22) {
      std::ifstream up("/proc/uptime");
      double uptime = 0;
      if (!(up >> uptime)) return -1;
      return uptime - std::strtod(tok.c_str(), nullptr) / static_cast<double>(sysconf(_SC_CLK_TCK));
    }
  return -1;
}

std::vector<pid_t> pids_named(const std::string &comm_want) {
  std::vector<pid_t> out;
  DIR *d = opendir("/proc");
  if (!d) return out;
  while (dirent *e = readdir(d)) {
    const char *n = e->d_name;
    if (*n < '1' || *n > '9') continue;
    std::ifstream c(std::string("/proc/") + n + "/comm");
    std::string comm;
    std::getline(c, comm);
    if (comm == comm_want && pid_alive(static_cast<pid_t>(std::atoi(n)))) out.push_back(static_cast<pid_t>(std::atoi(n)));
  }
  closedir(d);
  return out;
}

std::vector<ProcInfo> find_servers() {
  std::vector<ProcInfo> out;
  DIR *d = opendir("/proc");
  if (!d) return out;
  while (dirent *e = readdir(d)) {
    const char *n = e->d_name;
    if (*n < '1' || *n > '9') continue;
    const std::string base = std::string("/proc/") + n;
    std::ifstream c(base + "/comm");
    std::string comm;
    std::getline(c, comm);
    if (comm != "xenithium-serve") continue;
    ProcInfo p;
    p.pid = static_cast<pid_t>(std::atoi(n));
    if (!pid_alive(p.pid)) continue;
    std::ifstream cl(base + "/cmdline", std::ios::binary);
    std::stringstream ss;
    ss << cl.rdbuf();
    std::vector<std::string> args;
    std::string a;
    for (const char ch : ss.str()) {
      if (ch == '\0') {
        args.push_back(a);
        a.clear();
      } else {
        a += ch;
      }
    }
    if (!a.empty()) args.push_back(a);
    p.port = 8000;
    for (size_t i = 0; i < args.size(); ++i) {
      p.cmdline += (i ? " " : "") + args[i];
      if (args[i] == "--port" && i + 1 < args.size()) p.port = std::atoi(args[i + 1].c_str());
    }
    out.push_back(p);
  }
  closedir(d);
  return out;
}

bool port_in_use(int port) {
  const int fd = socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
  if (fd < 0) return false;
  sockaddr_in a{};
  a.sin_family = AF_INET;
  a.sin_port = htons(static_cast<uint16_t>(port));
  a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  const bool used = connect(fd, reinterpret_cast<sockaddr *>(&a), sizeof a) == 0;
  close(fd);
  return used;
}

std::vector<std::string> docker_containers(bool &docker_ok) {
  std::vector<std::string> out;
  const RunResult r = run_capture({"docker", "ps", "--format", "{{.Label \"org.xenithium\"}}\t{{.Names}}"}, {}, "", 3000);
  docker_ok = r.status == 0;
  if (!docker_ok) return out;
  std::istringstream in(r.out);
  std::string l;
  while (std::getline(in, l)) {
    const size_t tab = l.find('\t');
    if (tab == std::string::npos) continue;
    // this project's own containers (label org.xenithium from docker/Dockerfile: compose.yaml's server / launch / tools)
    // are not counted: they share the host's process table, so a server in one shows up as a server process
    if (l.rfind("1\t", 0) == 0) continue;
    out.push_back(l.substr(tab + 1));
  }
  return out;
}

std::string lan_ip() {
  ifaddrs *ifs = nullptr;
  if (getifaddrs(&ifs) != 0) return "";
  std::string best;
  for (ifaddrs *i = ifs; i; i = i->ifa_next) {
    if (!i->ifa_addr || i->ifa_addr->sa_family != AF_INET || (i->ifa_flags & IFF_LOOPBACK) || !(i->ifa_flags & IFF_UP)) continue;
    const std::string name = i->ifa_name ? i->ifa_name : "";
    if (name.rfind("docker", 0) == 0 || name.rfind("br-", 0) == 0 || name.rfind("veth", 0) == 0 || name.rfind("virbr", 0) == 0) continue;
    char buf[INET_ADDRSTRLEN];
    inet_ntop(AF_INET, &reinterpret_cast<sockaddr_in *>(i->ifa_addr)->sin_addr, buf, sizeof buf);
    best = buf;
    break;
  }
  freeifaddrs(ifs);
  return best;
}

std::map<std::string, double> parse_metrics(const std::string &text) {
  std::map<std::string, double> m;
  std::istringstream in(text);
  std::string l;
  while (std::getline(in, l)) {
    if (l.empty() || l[0] == '#') continue;
    size_t end = 0;
    while (end < l.size() && l[end] != '{' && l[end] != ' ') ++end;
    const std::string name = l.substr(0, end);
    size_t v = l.find_last_of(' ');
    if (v == std::string::npos || v < end) continue;
    char *e = nullptr;
    const double x = std::strtod(l.c_str() + v + 1, &e);
    if (e == l.c_str() + v + 1) continue;
    m[name] += x;
  }
  return m;
}

}  // namespace xenithium:launch
