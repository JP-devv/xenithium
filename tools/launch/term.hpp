// Terminal layer of the launcher: a cell canvas drawn by the screens, a terminal that shows it
// (raw mode, alternate screen, diff rendering, truecolor or 256 colours) and the key decoder.
// No curses: the build has no ncurses headers, and the user rule is C++ tools without deps.
#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace xenithium::launch {

// 0xRRGGBB, or kDefault for the terminal's own colour.
constexpr uint32_t kDefault = 0xFF000000u;

enum Attr : uint8_t { kBold = 1, kDim = 2, kItalic = 4, kUnder = 8, kReverse = 16 };

struct Style {
  uint32_t fg = kDefault, bg = kDefault;
  uint8_t attr = 0;
  bool operator==(const Style &) const = default;
  Style with_bg(uint32_t b) const { Style s = *this; s.bg = b; return s; }
  Style with_fg(uint32_t f) const { Style s = *this; s.fg = f; return s; }
  Style with(uint8_t a) const { Style s = *this; s.attr |= a; return s; }
};

struct Cell {
  char32_t ch = U' ';
  Style st;
  bool operator==(const Cell &) const = default;
};

// Columns a code point takes (0 for combining marks, 2 for wide CJK, else 1).
int cp_width(char32_t c);
// Decodes one UTF-8 code point at s[i] and advances i (invalid bytes give U+FFFD).
char32_t utf8_next(std::string_view s, size_t &i);
void utf8_append(std::string &out, char32_t c);
// Display width of a UTF-8 string.
int text_width(std::string_view s);
// The longest prefix of s that fits in `cols` columns, with "…" when cut.
std::string ellipsize(std::string_view s, int cols);
// Greedy word wrap to `cols` columns; '\n' starts a new paragraph line.
std::vector<std::string> wrap(std::string_view s, int cols);

// A setting's help text laid out for the help panel. The text is paragraphs split by '\n' (an empty one = a blank
// line), `code` between backticks (Canvas:put_md), and three kinds of paragraph that wrap with a hanging indent:
// `value` what it does a value of the setting
// Gives... a labelled line (help_label: Gives, Takes, Output, Needs, KV context...; a "(case)" may
// follow the word); the label is drawn apart
// - item a bullet
// A code span cut by a line break is closed on that line and reopened on the next, so every line draws alone.
struct HelpLine {
  std::string text;
  int label = 0;  // bytes at the start of text that are the label (0 = none)
};
std::vector<HelpLine> help_lines(std::string_view text, int cols);
// The bytes of the label a paragraph starts with ("Gives (int8): " -> 13, the colon included), 0 = none.
int help_label(std::string_view para);

class Canvas {
 public:
  void resize(int w, int h);
  int w() const { return w_; }
  int h() const { return h_; }
  void clear(Style st = {});
  // Writes text at (y, x) clipped to the canvas and to max_cols (-1 = to the edge). Returns the
  // columns written.
  int put(int y, int x, std::string_view text, Style st = {}, int max_cols = -1);
  // Text where `backticked spans` are drawn in `code` style (the docs' markdown).
  int put_md(int y, int x, std::string_view text, Style st, Style code, int max_cols = -1);
  void fill(int y, int x, int w, int h, Style st, char32_t ch = U' ');
  // Rounded box; title drawn into the top edge, `right` into the bottom-right edge.
  void box(int y, int x, int h, int w, Style border, std::string_view title = {}, Style title_st = {}, std::string_view right = {});
  // Sets the background of a span without touching its text.
  void tint(int y, int x, int w, uint32_t bg);
  const Cell &at(int y, int x) const { return cells_[static_cast<size_t>(y) * w_ + x]; }
  // A row as plain text (tests).
  std::string row(int y) const;

 private:
  int w_ = 0, h_ = 0;
  std::vector<Cell> cells_;
};

struct Key {
  enum Type {
    None, Char, Enter, Esc, Tab, BackTab, Backspace, Delete, Up, Down, Left, Right, Home, End, PgUp, PgDn,
    F1, F2, F3, F4, F5, Ctrl, Mouse, Paste, Resize
  };
  Type type = None;
  char32_t ch = 0;  // Char: the code point; Ctrl: the letter ('a'..'z')
  int mx = 0, my = 0;  // Mouse: 0-based cell
  int button = 0;  // Mouse: 0 left, 1 middle, 2 right, 64 wheel up, 65 wheel down
  bool press = false;  // Mouse: press (vs release)
  std::string text;  // Paste
  static Key of(Type t) { Key k; k.type = t; return k; }
  static Key chr(char32_t c) { Key k; k.type = Char; k.ch = c; return k; }
  static Key ctrl(char c) { Key k; k.type = Ctrl; k.ch = static_cast<char32_t>(c); return k; }
  bool is(char32_t c) const { return type == Char && ch == c; }
};

// Splits a byte stream into keys. `more_soon` = more bytes may follow: a lone ESC (or a cut
// sequence) is then kept for the next call instead of read as the Esc key.
class KeyDecoder {
 public:
  void feed(std::string_view bytes) { buf_.append(bytes); }
  bool next(Key &k, bool more_soon);
  bool pending() const { return !buf_.empty(); }

 private:
  std::string buf_;
};

class Terminal {
 public:
  Terminal() = default;
  ~Terminal();
  bool open(std::string *err);  // raw mode, alternate screen, mouse, bracketed paste
  void close();  // restores the terminal; safe to call twice
  void suspend();  // Ctrl-Z: restore, stop, re-enter on SIGCONT
  void size(int &w, int &h) const;
  // Waits up to timeout_ms for input; returns false on timeout. A window resize returns
  // Key:Resize.
  bool read_key(Key &k, int timeout_ms);
  void present(const Canvas &c);  // draws the cells that changed since the last frame
  void invalidate() { last_.resize(0, 0); }
  bool truecolor() const { return truecolor_; }

 private:
  void sgr(std::string &out, const Style &s) const;
  bool open_ = false, truecolor_ = true;
  KeyDecoder dec_;
  Canvas last_;
};

// Restores the terminal from a signal or exit path (registered by Terminal:open).
void terminal_emergency_restore();

}  // namespace xenithium:launch
