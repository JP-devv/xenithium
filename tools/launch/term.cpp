// Terminal layer of the launcher.
#include "term.hpp"

#include <fcntl.h>
#include <poll.h>
#include <sys/ioctl.h>
#include <termios.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <csignal>
#include <cstdlib>
#include <cstring>

namespace xenithium::launch {

// ---- UTF-8 and widths ---------------------------------------------------------------------

int cp_width(char32_t c) {
  if (c == 0) return 0;
  if (c < 0x20 || (c >= 0x7f && c < 0xa0)) return 1;  // drawn as '?'
  if ((c >= 0x300 && c <= 0x36f) || (c >= 0x200b && c <= 0x200f) || (c >= 0xfe00 && c <= 0xfe0f)) return 0;
  if ((c >= 0x1100 && c <= 0x115f) || (c >= 0x2e80 && c <= 0xa4cf && c != 0x303f) || (c >= 0xac00 && c <= 0xd7a3) ||
      (c >= 0xf900 && c <= 0xfaff) || (c >= 0xfe30 && c <= 0xfe4f) || (c >= 0xff00 && c <= 0xff60) || (c >= 0xffe0 && c <= 0xffe6) ||
      (c >= 0x1f300 && c <= 0x1f64f) || (c >= 0x1f900 && c <= 0x1f9ff) || (c >= 0x20000 && c <= 0x3fffd))
    return 2;
  return 1;
}

char32_t utf8_next(std::string_view s, size_t &i) {
  const auto b = static_cast<unsigned char>(s[i]);
  const int n = b < 0x80 ? 0 : (b >> 5) == 6 ? 1 : (b >> 4) == 14 ? 2 : (b >> 3) == 30 ? 3 : -1;
  if (n == 0) {
    ++i;
    return b;
  }
  if (n < 0 || i + n >= s.size()) {  // a stray continuation byte or a cut sequence
    ++i;
    return 0xfffd;
  }
  char32_t c = b & (0x3f >> n);
  for (int k = 1; k <= n; ++k) {
    const auto x = static_cast<unsigned char>(s[i + k]);
    if ((x & 0xc0) != 0x80) {
      ++i;
      return 0xfffd;
    }
    c = (c << 6) | (x & 0x3f);
  }
  i += n + 1;
  return c;
}

void utf8_append(std::string &out, char32_t c) {
  if (c < 0x80) out += static_cast<char>(c);
  else if (c < 0x800) {
    out += static_cast<char>(0xc0 | (c >> 6));
    out += static_cast<char>(0x80 | (c & 0x3f));
  } else if (c < 0x10000) {
    out += static_cast<char>(0xe0 | (c >> 12));
    out += static_cast<char>(0x80 | ((c >> 6) & 0x3f));
    out += static_cast<char>(0x80 | (c & 0x3f));
  } else {
    out += static_cast<char>(0xf0 | (c >> 18));
    out += static_cast<char>(0x80 | ((c >> 12) & 0x3f));
    out += static_cast<char>(0x80 | ((c >> 6) & 0x3f));
    out += static_cast<char>(0x80 | (c & 0x3f));
  }
}

int text_width(std::string_view s) {
  int w = 0;
  for (size_t i = 0; i < s.size();) w += cp_width(utf8_next(s, i));
  return w;
}

std::string ellipsize(std::string_view s, int cols) {
  if (cols <= 0) return {};
  if (text_width(s) <= cols) return std::string(s);
  std::string out;
  int w = 0;
  for (size_t i = 0; i < s.size();) {
    const size_t at = i;
    const char32_t c = utf8_next(s, i);
    const int cw = cp_width(c);
    if (w + cw > cols - 1) break;
    out.append(s.substr(at, i - at));
    w += cw;
  }
  return out + "\xe2\x80\xa6";  // …
}

std::vector<std::string> wrap(std::string_view s, int cols) {
  std::vector<std::string> out;
  if (cols < 4) cols = 4;
  size_t p = 0;
  while (p <= s.size()) {
    const size_t nl = s.find('\n', p);
    const std::string_view para = s.substr(p, nl == std::string_view::npos ? std::string_view::npos : nl - p);
    std::string line;
    int lw = 0;
    size_t i = 0;
    while (i < para.size()) {
      while (i < para.size() && para[i] == ' ') ++i;
      size_t j = i;
      while (j < para.size() && para[j] != ' ') ++j;
      if (j == i) break;
      std::string_view word = para.substr(i, j - i);
      int ww = text_width(word);
      while (ww > cols) {  // a word longer than a line: hard split
        if (!line.empty()) {
          out.push_back(line);
          line.clear();
          lw = 0;
        }
        size_t k = 0;
        int w = 0;
        while (k < word.size()) {
          size_t at = k;
          const int cw = cp_width(utf8_next(word, k));
          if (w + cw > cols) {
            k = at;
            break;
          }
          w += cw;
        }
        out.emplace_back(word.substr(0, k));
        word = word.substr(k);
        ww = text_width(word);
      }
      if (!line.empty() && lw + 1 + ww > cols) {
        out.push_back(line);
        line.clear();
        lw = 0;
      }
      if (!line.empty()) {
        line += ' ';
        ++lw;
      }
      line.append(word);
      lw += ww;
      i = j;
    }
    out.push_back(line);
    if (nl == std::string_view::npos) break;
    p = nl + 1;
  }
  return out;
}

// ---- canvas ---------------------------------------------------------------------------------

void Canvas::resize(int w, int h) {
  w_ = std::max(0, w);
  h_ = std::max(0, h);
  cells_.assign(static_cast<size_t>(w_) * h_, Cell{});
}

void Canvas::clear(Style st) {
  for (auto &c : cells_) c = Cell{U' ', st};
}

int Canvas::put(int y, int x, std::string_view text, Style st, int max_cols) {
  if (y < 0 || y >= h_ || x >= w_) return 0;
  const int lim = max_cols < 0 ? w_ : std::min(w_, x + max_cols);
  int cx = x;
  for (size_t i = 0; i < text.size() && cx < lim;) {
    char32_t c = utf8_next(text, i);
    if (c == '\t') c = ' ';
    if (c < 0x20 || (c >= 0x7f && c < 0xa0)) c = '?';
    const int cw = cp_width(c);
    if (cw == 0) continue;
    if (cx + cw > lim) break;
    if (cx >= 0) {
      cells_[static_cast<size_t>(y) * w_ + cx] = Cell{c, st};
      if (cw == 2 && cx + 1 < w_) cells_[static_cast<size_t>(y) * w_ + cx + 1] = Cell{0, st};
    }
    cx += cw;
  }
  return cx - x;
}

int help_label(std::string_view p) {
  static const char *const words[] = {"Gives", "Takes", "Output", "Needs", "Tip", "Note", "Limits", "Why", "Applies", "Names",
                                      "Memory", "KV context", "What changes", "What does not", "What auto gives", "Careful", "This model"};
  for (const char *w : words) {
    const size_t n = std::strlen(w);
    if (p.size() <= n || p.compare(0, n, w) != 0) continue;
    size_t i = n;
    if (p.compare(i, 2, " (") == 0) {  // "Gives (int8-v4):"
      const size_t e = p.find(')', i);
      if (e == std::string_view::npos || e - i > 40) continue;
      i = e + 1;
    }
    if (i < p.size() && p[i] == ':' && (i + 1 == p.size() || p[i + 1] == ' ')) return static_cast<int>(i + 1);
  }
  return 0;
}

std::vector<HelpLine> help_lines(std::string_view text, int cols) {
  std::vector<HelpLine> out;
  if (cols < 8) cols = 8;
  auto width = [](std::string_view w) {  // as drawn: the backticks are markup
    int n = text_width(w);
    for (char c : w) n -= c == '`';
    return n;
  };
  std::vector<std::string_view> paras;
  for (size_t p = 0; p <= text.size();) {
    const size_t nl = text.find('\n', p);
    paras.push_back(text.substr(p, nl == std::string_view::npos ? std::string_view::npos : nl - p));
    if (nl == std::string_view::npos) break;
    p = nl + 1;
  }
  // a value paragraph: `value` + its text; a run of them shares one column for the text (values up to kValueCol wide)
  constexpr int kValueCol = 12;
  auto value_end = [](std::string_view p) -> size_t {  // one past the closing backtick, 0 = not a value paragraph
    if (p.empty() || p[0] != '`') return 0;
    const size_t e = p.find('`', 1);
    return e == std::string_view::npos ? 0 : e + 1;
  };
  std::vector<int> col(paras.size(), 0);
  for (size_t i = 0; i < paras.size();) {
    if (!value_end(paras[i])) {
      ++i;
      continue;
    }
    size_t j = i;
    int w = 0;
    for (; j < paras.size() && value_end(paras[j]); ++j)
      if (const int vw = width(paras[j].substr(0, value_end(paras[j]))); vw <= kValueCol) w = std::max(w, vw);
    for (size_t k = i; k < j; ++k) col[k] = w;
    i = j;
  }
  for (size_t pi = 0; pi < paras.size(); ++pi) {
    std::string_view para = paras[pi];
    std::string lead;
    int hang = 0;
    const int label = help_label(para);
    const size_t ve = value_end(para);
    if (para.rfind("- ", 0) == 0) lead = "\xe2\x80\xa2 ", para.remove_prefix(2), hang = 2;  // a bullet
    else if (ve) {  // a value: its text starts in the run's column (a longer value: two spaces after it)
      const int vw = width(para.substr(0, ve));
      hang = std::max(col[pi], vw <= kValueCol ? vw : 2) + 2;
      if (vw > kValueCol) hang = col[pi] > 0 ? col[pi] + 2 : 4;
      lead = std::string(para.substr(0, ve)) + std::string(static_cast<size_t>(std::max(2, col[pi] + 2 - vw)), ' ');
      para.remove_prefix(ve);
    } else if (label > 0) hang = 2;
    if (hang > cols / 2) hang = std::min(hang, 4);
    const size_t first = out.size();
    std::string line = lead;
    int lw = width(lead);
    bool fresh = true;  // no word of the text on the line yet
    auto flush = [&] {
      bool open = false;
      for (char c : line) open ^= c == '`';
      if (open) line += '`';  // a code span cut by the break: closed here, reopened on the next line
      out.push_back({line, 0});
      line = std::string(static_cast<size_t>(hang), ' ') + (open ? "`" : "");
      lw = hang, fresh = true;
    };
    size_t i = 0;
    while (i < para.size()) {
      size_t j = i;
      while (j < para.size() && para[j] == ' ') ++j;
      const int gap = fresh ? 0 : static_cast<int>(j - i);
      size_t e = j;
      while (e < para.size() && para[e] != ' ') ++e;
      if (e == j) break;
      // "12 %", "3.5-5.9 %": the sign stays with its number
      if (e + 1 < para.size() && para[e] == ' ' && para[e + 1] == '%') {
        e += 2;
        while (e < para.size() && para[e] != ' ') ++e;
      }
      const std::string_view word = para.substr(j, e - j);
      const int ww = width(word);
      if (!fresh && lw + gap + ww > cols) {
        flush();
        line.append(word);
        lw += ww;
      } else {
        line.append(static_cast<size_t>(gap), ' ');
        line.append(word);
        lw += gap + ww;
      }
      fresh = false;
      i = e;
    }
    out.push_back({line, 0});
    out[first].label = label;
  }
  return out;
}

int Canvas::put_md(int y, int x, std::string_view text, Style st, Style code, int max_cols) {
  int used = 0;
  bool in_code = false;
  size_t p = 0;
  while (p < text.size()) {
    const size_t q = text.find('`', p);
    const std::string_view seg = text.substr(p, q == std::string_view::npos ? std::string_view::npos : q - p);
    const int room = max_cols < 0 ? -1 : max_cols - used;
    if (room == 0) break;
    used += put(y, x + used, seg, in_code ? code : st, room);
    if (q == std::string_view::npos) break;
    in_code = !in_code;
    p = q + 1;
  }
  return used;
}

void Canvas::fill(int y, int x, int w, int h, Style st, char32_t ch) {
  for (int r = std::max(0, y); r < std::min(h_, y + h); ++r)
    for (int c = std::max(0, x); c < std::min(w_, x + w); ++c) cells_[static_cast<size_t>(r) * w_ + c] = Cell{ch, st};
}

void Canvas::tint(int y, int x, int w, uint32_t bg) {
  if (y < 0 || y >= h_) return;
  for (int c = std::max(0, x); c < std::min(w_, x + w); ++c) cells_[static_cast<size_t>(y) * w_ + c].st.bg = bg;
}

void Canvas::box(int y, int x, int h, int w, Style border, std::string_view title, Style title_st, std::string_view right) {
  if (w < 2 || h < 2) return;
  std::string top = "\xe2\x95\xad", bot = "\xe2\x95\xb0";  // ╭ ╰
  for (int i = 0; i < w - 2; ++i) {
    top += "\xe2\x94\x80";  // ─
    bot += "\xe2\x94\x80";
  }
  top += "\xe2\x95\xae";  // ╮
  bot += "\xe2\x95\xaf";  // ╯
  put(y, x, top, border);
  put(y + h - 1, x, bot, border);
  for (int r = 1; r < h - 1; ++r) {
    put(y + r, x, "\xe2\x94\x82", border);  // │
    put(y + r, x + w - 1, "\xe2\x94\x82", border);
  }
  if (!title.empty()) {
    const std::string t = " " + ellipsize(title, w - 6) + " ";
    put(y, x + 2, t, title_st);
  }
  if (!right.empty()) {
    const std::string t = " " + ellipsize(right, w - 6) + " ";
    put(y + h - 1, x + w - 2 - text_width(t), t, border);
  }
}

std::string Canvas::row(int y) const {
  std::string s;
  for (int x = 0; x < w_; ++x) {
    const char32_t c = at(y, x).ch;
    if (c) utf8_append(s, c);
  }
  while (!s.empty() && s.back() == ' ') s.pop_back();
  return s;
}

// ---- keys -----------------------------------------------------------------------------------

bool KeyDecoder::next(Key &k, bool more_soon) {
  k = Key{};
  while (!buf_.empty()) {
    const auto b = static_cast<unsigned char>(buf_[0]);
    if (b == 0x1b) {
      if (buf_.size() == 1) {
        if (more_soon) return false;
        buf_.erase(0, 1);
        k.type = Key::Esc;
        return true;
      }
      const char t = buf_[1];
      if (t == '[') {
        // Bracketed paste: everything up to ESC [ 201 ~.
        if (buf_.compare(0, 6, "\x1b[200~") == 0) {
          const size_t end = buf_.find("\x1b[201~", 6);
          if (end == std::string::npos) {
            if (more_soon) return false;
            k.type = Key::Paste;
            k.text = buf_.substr(6);
            buf_.clear();
            return true;
          }
          k.type = Key::Paste;
          k.text = buf_.substr(6, end - 6);
          buf_.erase(0, end + 6);
          return true;
        }
        size_t i = 2;
        while (i < buf_.size() && !(buf_[i] >= 0x40 && buf_[i] <= 0x7e)) ++i;
        if (i >= buf_.size()) {
          if (more_soon) return false;
          buf_.clear();  // a cut sequence: drop it
          continue;
        }
        const std::string params = buf_.substr(2, i - 2);
        const char fin = buf_[i];
        buf_.erase(0, i + 1);
        if (!params.empty() && params[0] == '<' && (fin == 'M' || fin == 'm')) {  // SGR mouse
          int v[3] = {0, 0, 0}, n = 0;
          for (size_t p = 1; p < params.size() && n < 3; ++p) {
            if (params[p] == ';') ++n;
            else if (params[p] >= '0' && params[p] <= '9') v[n] = v[n] * 10 + (params[p] - '0');
          }
          k.type = Key::Mouse;
          k.button = v[0] & ~(4 | 8 | 16 | 32);
          if (v[0] & 32) k.button = -1;  // motion
          k.mx = v[1] - 1;
          k.my = v[2] - 1;
          k.press = fin == 'M';
          return true;
        }
        const int num = std::atoi(params.c_str());
        switch (fin) {
          case 'A': k.type = Key::Up; return true;
          case 'B': k.type = Key::Down; return true;
          case 'C': k.type = Key::Right; return true;
          case 'D': k.type = Key::Left; return true;
          case 'H': k.type = Key::Home; return true;
          case 'F': k.type = Key::End; return true;
          case 'Z': k.type = Key::BackTab; return true;
          case 'P': k.type = Key::F1; return true;
          case 'Q': k.type = Key::F2; return true;
          case 'R': k.type = Key::F3; return true;
          case 'S': k.type = Key::F4; return true;
          case '~':
            switch (num) {
              case 1: case 7: k.type = Key::Home; return true;
              case 4: case 8: k.type = Key::End; return true;
              case 3: k.type = Key::Delete; return true;
              case 5: k.type = Key::PgUp; return true;
              case 6: k.type = Key::PgDn; return true;
              case 11: k.type = Key::F1; return true;
              case 12: k.type = Key::F2; return true;
              case 13: k.type = Key::F3; return true;
              case 14: k.type = Key::F4; return true;
              case 15: k.type = Key::F5; return true;
              default: continue;
            }
          default: continue;  // unknown: skip
        }
      }
      if (t == 'O') {
        if (buf_.size() < 3) {
          if (more_soon) return false;
          buf_.clear();
          continue;
        }
        const char f = buf_[2];
        buf_.erase(0, 3);
        switch (f) {
          case 'A': k.type = Key::Up; return true;
          case 'B': k.type = Key::Down; return true;
          case 'C': k.type = Key::Right; return true;
          case 'D': k.type = Key::Left; return true;
          case 'H': k.type = Key::Home; return true;
          case 'F': k.type = Key::End; return true;
          case 'P': k.type = Key::F1; return true;
          case 'Q': k.type = Key::F2; return true;
          case 'R': k.type = Key::F3; return true;
          case 'S': k.type = Key::F4; return true;
          default: continue;
        }
      }
      if (t == 0x1b) {  // ESC ESC: one Esc, keep the second
        buf_.erase(0, 1);
        k.type = Key::Esc;
        return true;
      }
      buf_.erase(0, 1);  // Alt+key: read as the key itself
      continue;
    }
    if (b == '\r' || b == '\n') {
      buf_.erase(0, 1);
      if (b == '\r' && !buf_.empty() && buf_[0] == '\n') buf_.erase(0, 1);
      k.type = Key::Enter;
      return true;
    }
    if (b == '\t') {
      buf_.erase(0, 1);
      k.type = Key::Tab;
      return true;
    }
    if (b == 0x7f || b == 0x08) {
      buf_.erase(0, 1);
      k.type = Key::Backspace;
      return true;
    }
    if (b < 0x20) {
      buf_.erase(0, 1);
      if (b == 0) continue;
      k = Key::ctrl(static_cast<char>('a' + b - 1));
      return true;
    }
    // UTF-8: wait for a cut multi-byte sequence.
    const int n = b < 0x80 ? 1 : (b >> 5) == 6 ? 2 : (b >> 4) == 14 ? 3 : (b >> 3) == 30 ? 4 : 1;
    if (static_cast<int>(buf_.size()) < n) {
      if (more_soon) return false;
      buf_.clear();
      continue;
    }
    size_t i = 0;
    const std::string one = buf_.substr(0, n);
    const char32_t c = utf8_next(one, i);
    buf_.erase(0, n);
    k = Key::chr(c);
    return true;
  }
  return false;
}

// ---- terminal -------------------------------------------------------------------------------

namespace {

termios g_saved{};
std::atomic<bool> g_raw{false};
int g_wake[2] = {-1, -1};

void write_all(const std::string &s) {
  size_t off = 0;
  while (off < s.size()) {
    const ssize_t n = ::write(STDOUT_FILENO, s.data() + off, s.size() - off);
    if (n < 0) {
      if (errno == EINTR) continue;
      return;
    }
    off += static_cast<size_t>(n);
  }
}

const char kEnter[] = "\x1b[?1049h\x1b[?25l\x1b[?7l\x1b[?1000h\x1b[?1006h\x1b[?2004h\x1b[H\x1b[2J";
const char kLeave[] = "\x1b[0m\x1b[?2004l\x1b[?1006l\x1b[?1000l\x1b[?7h\x1b[?25h\x1b[?1049l";

void restore_raw() {
  if (!g_raw.exchange(false)) return;
  (void)!::write(STDOUT_FILENO, kLeave, sizeof(kLeave) - 1);
  tcsetattr(STDIN_FILENO, TCSAFLUSH, &g_saved);
}

void on_fatal(int sig) {
  restore_raw();
  std::signal(sig, SIG_DFL);
  std::raise(sig);
}

void on_winch(int) {
  if (g_wake[1] >= 0) {
    const char c = 'w';
    (void)!::write(g_wake[1], &c, 1);
  }
}

bool enter_raw() {
  termios t = g_saved;
  t.c_iflag &= ~static_cast<tcflag_t>(BRKINT | ICRNL | INPCK | ISTRIP | IXON);
  t.c_cflag |= CS8;
  t.c_lflag &= ~static_cast<tcflag_t>(ECHO | ICANON | IEXTEN | ISIG);
  t.c_cc[VMIN] = 0;
  t.c_cc[VTIME] = 0;
  if (tcsetattr(STDIN_FILENO, TCSAFLUSH, &t) != 0) return false;
  g_raw = true;
  write_all(kEnter);
  return true;
}

// xterm 256-colour index nearest to an RGB colour (6x6x6 cube or the grey ramp).
int to256(uint32_t rgb) {
  const int r = (rgb >> 16) & 255, g = (rgb >> 8) & 255, b = rgb & 255;
  auto q = [](int v) { return v < 48 ? 0 : v < 115 ? 1 : (v - 35) / 40; };
  const int cr = q(r), cg = q(g), cb = q(b);
  static const int lv[6] = {0, 95, 135, 175, 215, 255};
  const int dc = (lv[cr] - r) * (lv[cr] - r) + (lv[cg] - g) * (lv[cg] - g) + (lv[cb] - b) * (lv[cb] - b);
  const int avg = (r + g + b) / 3;
  const int gi = avg > 238 ? 23 : std::max(0, (avg - 3) / 10);
  const int gv = 8 + gi * 10;
  const int dg = (gv - r) * (gv - r) + (gv - g) * (gv - g) + (gv - b) * (gv - b);
  return dg < dc ? 232 + gi : 16 + 36 * cr + 6 * cg + cb;
}

}  // namespace

void terminal_emergency_restore() { restore_raw(); }

Terminal::~Terminal() { close(); }

bool Terminal::open(std::string *err) {
  if (!isatty(STDIN_FILENO) || !isatty(STDOUT_FILENO)) {
    if (err) *err = "stdin and stdout must be a terminal";
    return false;
  }
  if (tcgetattr(STDIN_FILENO, &g_saved) != 0) {
    if (err) *err = std::string("tcgetattr: ") + std::strerror(errno);
    return false;
  }
  const char *ct = std::getenv("COLORTERM");
  truecolor_ = ct && (std::strstr(ct, "truecolor") || std::strstr(ct, "24bit"));
  if (g_wake[0] < 0 && pipe(g_wake) == 0) {
    fcntl(g_wake[0], F_SETFL, O_NONBLOCK);
    fcntl(g_wake[1], F_SETFL, O_NONBLOCK);
  }
  std::signal(SIGWINCH, on_winch);
  std::signal(SIGTERM, on_fatal);
  std::signal(SIGHUP, on_fatal);
  std::signal(SIGSEGV, on_fatal);
  std::signal(SIGABRT, on_fatal);
  std::atexit(restore_raw);
  if (!enter_raw()) {
    if (err) *err = std::string("tcsetattr: ") + std::strerror(errno);
    return false;
  }
  open_ = true;
  last_.resize(0, 0);
  return true;
}

void Terminal::close() {
  if (!open_) return;
  open_ = false;
  restore_raw();
}

void Terminal::suspend() {
  restore_raw();
  std::signal(SIGTSTP, SIG_DFL);
  std::raise(SIGTSTP);  // continues here after SIGCONT
  enter_raw();
  last_.resize(0, 0);
}

void Terminal::size(int &w, int &h) const {
  winsize ws{};
  if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0 && ws.ws_col > 0 && ws.ws_row > 0) {
    w = ws.ws_col;
    h = ws.ws_row;
  } else {
    w = 80;
    h = 24;
  }
}

bool Terminal::read_key(Key &k, int timeout_ms) {
  if (dec_.next(k, true)) return true;
  pollfd fds[2] = {{STDIN_FILENO, POLLIN, 0}, {g_wake[0], POLLIN, 0}};
  int wait = timeout_ms;
  for (;;) {
    const int r = poll(fds, g_wake[0] >= 0 ? 2 : 1, wait);
    if (r < 0) {
      if (errno == EINTR) continue;
      return false;
    }
    if (r == 0) {
      if (dec_.pending() && dec_.next(k, false)) return true;
      return false;
    }
    if (fds[1].revents & POLLIN) {
      char tmp[64];
      while (::read(g_wake[0], tmp, sizeof tmp) > 0) {
      }
      k = Key::of(Key::Resize);
      return true;
    }
    if (fds[0].revents & (POLLIN | POLLHUP)) {
      char buf[4096];
      const ssize_t n = ::read(STDIN_FILENO, buf, sizeof buf);
      if (n > 0) dec_.feed(std::string_view(buf, static_cast<size_t>(n)));
      if (dec_.next(k, true)) return true;
      if (!dec_.pending()) return false;
      wait = 30;  // a cut escape sequence: give its rest 30 ms
    }
  }
}

void Terminal::sgr(std::string &out, const Style &s) const {
  static const bool no_color = std::getenv("NO_COLOR") && *std::getenv("NO_COLOR");
  out += "\x1b[0";
  if (s.attr & kBold) out += ";1";
  if (s.attr & kDim) out += ";2";
  if (s.attr & kItalic) out += ";3";
  if (s.attr & kUnder) out += ";4";
  if (s.attr & kReverse) out += ";7";
  auto col = [&](uint32_t c, bool fg) {
    if (c == kDefault || no_color) return;
    if (truecolor_) {
      out += fg ? ";38;2;" : ";48;2;";
      out += std::to_string((c >> 16) & 255) + ";" + std::to_string((c >> 8) & 255) + ";" + std::to_string(c & 255);
    } else {
      out += fg ? ";38;5;" : ";48;5;";
      out += std::to_string(to256(c));
    }
  };
  col(s.fg, true);
  col(s.bg, false);
  out += 'm';
}

void Terminal::present(const Canvas &c) {
  std::string out;
  const bool full = last_.w() != c.w() || last_.h() != c.h();
  if (full) out += "\x1b[0m\x1b[2J";
  Style cur;
  bool have_style = false;
  for (int y = 0; y < c.h(); ++y) {
    int a = -1, b = -1;
    for (int x = 0; x < c.w(); ++x)
      if (full || !(c.at(y, x) == last_.at(y, x))) {
        if (a < 0) a = x;
        b = x;
      }
    if (a < 0) continue;
    while (a > 0 && c.at(y, a).ch == 0) --a;  // start on a whole wide char
    out += "\x1b[" + std::to_string(y + 1) + ";" + std::to_string(a + 1) + "H";
    for (int x = a; x <= b || (x < c.w() && c.at(y, x).ch == 0); ++x) {
      const Cell &cell = c.at(y, x);
      if (cell.ch == 0) continue;
      if (!have_style || !(cell.st == cur)) {
        sgr(out, cell.st);
        cur = cell.st;
        have_style = true;
      }
      utf8_append(out, cell.ch);
    }
  }
  if (!out.empty()) {
    out += "\x1b[0m";
    write_all(out);
  }
  last_ = c;
}

}  // namespace xenithium:launch
