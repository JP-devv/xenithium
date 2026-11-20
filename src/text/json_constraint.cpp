#include "xenithium/text/json_constraint.hpp"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <map>
#include <mutex>
#include <unordered_map>

namespace xenithium::text {

namespace {

constexpr int kMaxDepth = 24;  // frames per configuration (nesting)
constexpr int kMaxWs = 32;  // whitespace bytes per run
constexpr int kMaxNumDigits = 24;  // digits per number part
constexpr int kMaxLits = 64;  // literals per enum / const set, properties per object
constexpr size_t kMaxConfigs = 32;
constexpr size_t kMaxMasks = 2048;  // cached states per spec (~31 KB each at 248K tokens)

// ---------------------------------------------------------------- the compiled schema

enum class K : uint8_t { object, array, string, number, integer, literals, union_ };

struct Prop {
  std::string name;
  int node = 0;
  bool required = false;
};

struct Node {
  K k = K::union_;
  // object
  std::vector<Prop> props;
  int additional = -1;  // node of additional property values; -1 = none allowed
  // array
  int items = -1;
  int64_t min_items = 0, max_items = -1;
  // string
  int64_t min_len = 0, max_len = -1;
  // literals: raw bytes each (JSON text, or a guided_choice's text)
  std::vector<std::string> lits;
  // union
  std::vector<int> branches;
};

bool is_ws(uint8_t b) { return b == ' ' || b == '\n' || b == '\t' || b == '\r'; }

class Compiler {
 public:
  explicit Compiler(const json::Value &root) : root_(root) {
    // fixed nodes: 0 any, 1 any object, 2 any array, 3 any string, 4 any number, 5 true/false/null
    nodes_.resize(6);
    nodes_[1].k = K::object;
    nodes_[1].additional = 0;
    nodes_[2].k = K::array;
    nodes_[2].items = 0;
    nodes_[3].k = K::string;
    nodes_[4].k = K::number;
    nodes_[5].k = K::literals;
    nodes_[5].lits = {"true", "false", "null"};
    nodes_[0].k = K::union_;
    nodes_[0].branches = {1, 2, 3, 4, 5};
  }
  std::vector<Node> nodes_;
  std::string err;

  int add(Node n) {
    nodes_.push_back(std::move(n));
    return static_cast<int>(nodes_.size()) - 1;
  }

  int compile(const json::Value &s, int depth) {
    if (!err.empty()) return 0;
    if (depth > kMaxDepth) return fail("the schema nests deeper than " + std::to_string(kMaxDepth));
    if (s.is_bool()) return s.b ? 0 : fail("a false schema matches nothing");
    if (!s.is_object()) return fail("a schema must be an object or a boolean");
    for (const char *k : {"pattern", "patternProperties", "prefixItems", "not", "if", "then", "else"})
      if (s.get(k)) return fail(std::string("the schema keyword '") + k + "' is not supported");
    if (const std::string *ref = s.get_string("$ref")) return compile_ref(*ref, depth);
    if (const json::Value *c = s.get("const")) {
      Node n;
      n.k = K::literals;
      n.lits.push_back(json::dump(*c));
      return add(std::move(n));
    }
    if (const json::Value *e = s.get_array("enum")) {
      if (e->arr.empty()) return fail("an empty enum matches nothing");
      if (e->arr.size() > kMaxLits) return fail("enum has more than " + std::to_string(kMaxLits) + " values");
      Node n;
      n.k = K::literals;
      for (const auto &v : e->arr) n.lits.push_back(json::dump(v));
      return add(std::move(n));
    }
    for (const char *k : {"anyOf", "oneOf"})
      if (const json::Value *a = s.get_array(k)) {
        if (a->arr.empty()) return fail(std::string(k) + " is empty");
        Node n;
        n.k = K::union_;
        for (const auto &b : a->arr) n.branches.push_back(compile(b, depth + 1));
        return add(std::move(n));
      }
    if (const json::Value *a = s.get_array("allOf")) {
      if (a->arr.size() != 1) return fail("allOf with more than one schema is not supported");
      return compile(a->arr[0], depth + 1);
    }
    const json::Value *t = s.get("type");
    if (!t) {
      if (s.get("properties") || s.get("additionalProperties") || s.get("required")) return compile_typed(s, "object", depth);
      if (s.get("items")) return compile_typed(s, "array", depth);
      return 0;  // no constraint: any value
    }
    if (t->is_string()) return compile_typed(s, t->str, depth);
    if (t->is_array()) {
      if (t->arr.empty()) return fail("an empty type list matches nothing");
      Node n;
      n.k = K::union_;
      for (const auto &x : t->arr) {
        if (!x.is_string()) return fail("type must be a string or a list of strings");
        n.branches.push_back(compile_typed(s, x.str, depth));
      }
      return add(std::move(n));
    }
    return fail("type must be a string or a list of strings");
  }

 private:
  const json::Value &root_;
  std::map<std::string, int> refs_;

  int fail(const std::string &m) {
    if (err.empty()) err = m;
    return 0;
  }

  int compile_ref(const std::string &ref, int depth) {
    if (const auto it = refs_.find(ref); it != refs_.end()) return it->second;
    const json::Value *v = nullptr;
    for (const char *pre : {"#/$defs/", "#/definitions/"}) {
      const std::string p = pre;
      if (ref.rfind(p, 0) == 0) {
        const json::Value *defs = root_.get_object(p.substr(2, p.size() - 3));
        v = defs ? defs->get(ref.substr(p.size())) : nullptr;
      }
    }
    if (ref == "#") v = &root_;
    if (!v) return fail("$ref '" + ref + "' does not resolve (only #, #/$defs/X and #/definitions/X)");
    // a placeholder first, so a recursive reference resolves to this node
    const int id = add(Node{});
    refs_[ref] = id;
    const int body = compile(*v, depth + 1);
    Node n;
    n.k = K::union_;
    n.branches = {body};
    nodes_[static_cast<size_t>(id)] = std::move(n);
    return id;
  }

  static int64_t int_of(const json::Value *v, int64_t dflt) { return v && v->is_number() ? static_cast<int64_t>(v->num) : dflt; }

  int compile_typed(const json::Value &s, const std::string &type, int depth) {
    Node n;
    if (type == "object") {
      n.k = K::object;
      const json::Value *props = s.get_object("properties");
      std::vector<std::string> req;
      if (const json::Value *r = s.get_array("required"))
        for (const auto &x : r->arr)
          if (x.is_string()) req.push_back(x.str);
      if (props) {
        if (props->obj.size() > kMaxLits) return fail("an object has more than " + std::to_string(kMaxLits) + " properties");
        for (const auto &[name, sub] : props->obj) {
          std::string esc;
          json::write_string(name, esc);
          if (esc.size() != name.size() + 2) return fail("property name '" + name + "' needs JSON escapes");
          Prop p;
          p.name = name;
          p.required = std::find(req.begin(), req.end(), name) != req.end();
          p.node = compile(sub, depth + 1);
          n.props.push_back(std::move(p));
        }
      }
      for (const auto &r : req) {
        bool found = false;
        for (const auto &p : n.props) found |= p.name == r;
        if (!found) return fail("required property '" + r + "' is not in properties");
      }
      const json::Value *ap = s.get("additionalProperties");
      if (ap && ap->is_bool()) n.additional = ap->b ? 0 : -1;
      else if (ap && ap->is_object()) n.additional = compile(*ap, depth + 1);
      else n.additional = props ? -1 : 0;  // strict: with "properties" (even empty) no other keys; without, any object
      return add(std::move(n));
    }
    if (type == "array") {
      n.k = K::array;
      const json::Value *it = s.get("items");
      n.items = it ? compile(*it, depth + 1) : 0;
      n.min_items = int_of(s.get("minItems"), 0);
      n.max_items = int_of(s.get("maxItems"), -1);
      if (n.max_items >= 0 && n.max_items < n.min_items) return fail("maxItems < minItems");
      if (n.min_items > 60000 || n.max_items > 60000) return fail("minItems / maxItems above 60000");
      return add(std::move(n));
    }
    if (type == "string") {
      n.k = K::string;
      n.min_len = int_of(s.get("minLength"), 0);
      n.max_len = int_of(s.get("maxLength"), -1);
      if (n.max_len >= 0 && n.max_len < n.min_len) return fail("maxLength < minLength");
      if (n.min_len > 60000 || n.max_len > 60000) return fail("minLength / maxLength above 60000");
      return add(std::move(n));
    }
    if (type == "number") return 4;
    if (type == "integer") {
      n.k = K::integer;
      return add(std::move(n));
    }
    if (type == "boolean") {
      n.k = K::literals;
      n.lits = {"true", "false"};
      return add(std::move(n));
    }
    if (type == "null") {
      n.k = K::literals;
      n.lits = {"null"};
      return add(std::move(n));
    }
    return fail("unknown type '" + type + "'");
  }
};

// ---------------------------------------------------------------- the automaton

enum FT : uint8_t { F_VAL, F_OBJ, F_ARR, F_STR, F_NUM, F_LIT, F_KEY };
// object phases
enum : uint8_t { O_OPEN, O_KEY_DONE, O_VALUE, O_AFTER, O_COMMA };
// array phases
enum : uint8_t { A_OPEN, A_VALUE, A_AFTER, A_COMMA };
// string phases
enum : uint8_t { S_NORMAL, S_ESC, S_U1, S_U2, S_U3, S_U4 };
// number phases
enum : uint8_t { N_START, N_MINUS, N_ZERO, N_INT, N_DOT, N_FRAC, N_E, N_ESIGN, N_EXP };

struct Frame {
  uint32_t node = 0;
  uint8_t t = 0, ph = 0;
  uint16_t a = 0;  // obj: chosen property (O_KEY_DONE / O_VALUE), 0xffff = an additional key; arr: items; str: chars; num: digits; lit / key: position
  uint16_t b = 0;  // obj: next property index (ordered); str: 1 = a key string of the object below
  uint16_t pad = 0;
  uint32_t pad2 = 0;  // no implicit padding: frames are compared and hashed as bytes
  uint64_t m = 0;  // lit / key: candidate mask
};
static_assert(sizeof(Frame) == 24, "Frame has no padding bytes");

struct Config {
  uint8_t depth = 0, ws = 0, done = 0;
  Frame f[kMaxDepth];
  Frame &top() { return f[depth - 1]; }
  bool same(const Config &o) const {
    if (depth != o.depth || ws != o.ws || done != o.done) return false;
    return std::memcmp(f, o.f, sizeof(Frame) * depth) == 0;
  }
  void key(std::string &out) const {
    out.push_back(static_cast<char>(depth));
    out.push_back(static_cast<char>(ws));
    out.push_back(static_cast<char>(done));
    out.append(reinterpret_cast<const char *>(f), sizeof(Frame) * depth);
  }
};

struct Grammar {
  std::vector<Node> nodes;
  int root = 0;

  static void add(std::vector<Config> &out, const Config &c) {
    for (const auto &o : out)
      if (o.same(c)) return;
    if (out.size() < kMaxConfigs) out.push_back(c);
  }

  // A value of `node` starts with byte b at the top of c (the VAL frame there is replaced).
  void start_value(Config c, int node, uint8_t b, std::vector<Config> &out) const {
    const Node &n = nodes[static_cast<size_t>(node)];
    Frame &f = c.top();
    f = Frame{};
    f.node = static_cast<uint32_t>(node);
    switch (n.k) {
      case K::union_:
        for (int br : n.branches) start_value(c, br, b, out);
        return;
      case K::object:
        if (b != '{' || c.depth >= kMaxDepth) return;
        f.t = F_OBJ;
        f.ph = O_OPEN;
        c.ws = 0;
        add(out, c);
        return;
      case K::array:
        if (b != '[' || c.depth >= kMaxDepth) return;
        f.t = F_ARR;
        f.ph = A_OPEN;
        c.ws = 0;
        add(out, c);
        return;
      case K::string:
        if (b != '"') return;
        f.t = F_STR;
        f.ph = S_NORMAL;
        c.ws = 0;
        add(out, c);
        return;
      case K::number:
      case K::integer:
        f.t = F_NUM;
        f.ph = N_START;
        c.ws = 0;
        feed(c, b, out);
        return;
      case K::literals:
        f.t = F_LIT;
        f.a = 0;
        f.m = n.lits.size() >= 64 ? ~0ull : (1ull << n.lits.size()) - 1;
        c.ws = 0;
        feed(c, b, out);
        return;
    }
  }

  // The top frame completed: pop it and let the parent advance.
  void complete(Config c, std::vector<Config> &out) const {
    --c.depth;
    if (c.depth == 0) {
      c.done = 1;
      c.ws = 0;
      add(out, c);
      return;
    }
    Frame &p = c.top();
    if (p.t == F_OBJ) {
      p.ph = O_AFTER;
    } else if (p.t == F_ARR) {
      const Node &n = nodes[p.node];
      p.ph = A_AFTER;
      if (n.min_items > 0 || n.max_items >= 0) p.a = static_cast<uint16_t>(std::min<int>(p.a + 1, 65535));
    }
    c.ws = 0;
    add(out, c);
  }

  // Candidate properties for the next key: index >= next, none required skipped.
  static uint64_t key_candidates(const Node &n, int next) {
    uint64_t m = 0;
    for (size_t i = static_cast<size_t>(next); i < n.props.size(); ++i) {
      m |= 1ull << i;
      if (n.props[i].required) break;
    }
    return m;
  }
  static bool required_left(const Node &n, int next) {
    for (size_t i = static_cast<size_t>(next); i < n.props.size(); ++i)
      if (n.props[i].required) return true;
    return false;
  }

  // A key starts (the opening quote was read) in object frame p of c.
  void start_key(Config c, std::vector<Config> &out) const {
    const Frame p = c.top();
    const Node &n = nodes[p.node];
    const uint64_t cand = key_candidates(n, p.b);
    c.ws = 0;
    if (cand) {  // a declared property
      Config k = c;
      Frame &f = k.f[k.depth++];
      f = Frame{};
      f.t = F_KEY;
      f.node = p.node;
      f.m = cand;
      add(out, k);
    }
    if (n.additional >= 0) {  // any other key: a free string
      Config k = c;
      k.top().a = 0xffff;
      Frame &f = k.f[k.depth++];
      f = Frame{};
      f.t = F_STR;
      f.node = 3;
      f.b = 1;
      add(out, k);
    }
  }

  bool ws_ok(Config &c) const {
    if (c.ws >= kMaxWs) return false;
    ++c.ws;
    return true;
  }

  // Can the configuration end here (end of input)? Numbers and literals at the top end on it.
  bool can_end(const Config &c) const {
    if (c.done) return true;
    if (c.depth != 1) return false;
    const Frame &f = c.f[0];
    if (f.t == F_NUM) return f.ph == N_ZERO || f.ph == N_INT || f.ph == N_FRAC || f.ph == N_EXP;
    if (f.t == F_LIT) return lit_complete(f);
    return false;
  }
  bool lit_complete(const Frame &f) const {
    const Node &n = nodes[f.node];
    for (size_t i = 0; i < n.lits.size(); ++i)
      if (((f.m >> i) & 1) && n.lits[i].size() == f.a) return true;
    return false;
  }

  // Feeds one byte to c; the configurations that accept it go to out.
  void feed(Config c, uint8_t b, std::vector<Config> &out) const {
    // No whitespace around the root value: the root rule is the value alone, so after </think> the
    // first token is the value's, not "\n\n". Whitespace inside the value is allowed.
    if (c.done) return;
    Frame &f = c.top();
    switch (f.t) {
      case F_VAL:
        if (is_ws(b)) {
          if (c.depth > 1 && ws_ok(c)) add(out, c);
          return;
        }
        start_value(c, static_cast<int>(f.node), b, out);
        return;
      case F_OBJ: {
        const Node &n = nodes[f.node];
        switch (f.ph) {
          case O_OPEN:
          case O_COMMA:
            if (is_ws(b)) {
              if (ws_ok(c)) add(out, c);
            } else if (b == '"') {
              start_key(c, out);
            } else if (b == '}' && f.ph == O_OPEN && !required_left(n, f.b)) {
              complete(c, out);
            }
            return;
          case O_KEY_DONE:
            if (is_ws(b)) {
              if (ws_ok(c)) add(out, c);
            } else if (b == ':' && c.depth < kMaxDepth) {
              const int vn = f.a == 0xffff ? n.additional : n.props[f.a].node;
              f.ph = O_VALUE;
              c.ws = 0;
              Frame &v = c.f[c.depth++];
              v = Frame{};
              v.t = F_VAL;
              v.node = static_cast<uint32_t>(vn);
              add(out, c);
            }
            return;
          case O_AFTER:
            if (is_ws(b)) {
              if (ws_ok(c)) add(out, c);
            } else if (b == ',') {
              if (key_candidates(n, f.b) || n.additional >= 0) {
                f.ph = O_COMMA;
                c.ws = 0;
                add(out, c);
              }
            } else if (b == '}' && !required_left(n, f.b)) {
              complete(c, out);
            }
            return;
          default:
            return;
        }
      }
      case F_ARR: {
        const Node &n = nodes[f.node];
        switch (f.ph) {
          case A_OPEN:
          case A_COMMA:
            if (is_ws(b)) {
              if (ws_ok(c)) add(out, c);
              return;
            }
            if (b == ']' && f.ph == A_OPEN && n.min_items <= 0) {
              complete(c, out);
              return;
            }
            if (f.ph == A_OPEN && n.max_items == 0) return;
            f.ph = A_VALUE;
            if (c.depth >= kMaxDepth) return;
            {
              Frame &v = c.f[c.depth++];
              v = Frame{};
              v.t = F_VAL;
              v.node = static_cast<uint32_t>(n.items);
              c.ws = 0;
              start_value(c, n.items, b, out);
            }
            return;
          case A_AFTER:
            if (is_ws(b)) {
              if (ws_ok(c)) add(out, c);
            } else if (b == ',') {
              if (n.max_items < 0 || f.a < n.max_items) {
                f.ph = A_COMMA;
                c.ws = 0;
                add(out, c);
              }
            } else if (b == ']' && f.a >= n.min_items) {
              complete(c, out);
            }
            return;
          default:
            return;
        }
      }
      case F_STR: {
        const Node &n = nodes[f.node];
        const bool bounded = n.min_len > 0 || n.max_len >= 0;
        switch (f.ph) {
          case S_NORMAL:
            if (b == '"') {
              if (bounded && f.a < n.min_len) return;
              if (f.b == 1) {  // a free key: the object below takes it
                --c.depth;
                c.top().ph = O_KEY_DONE;
                c.ws = 0;
                add(out, c);
                return;
              }
              complete(c, out);
              return;
            }
            if (b < 0x20) return;
            if (bounded && (b & 0xC0) != 0x80) {  // a new character
              if (n.max_len >= 0 && f.a >= n.max_len) return;
              f.a = static_cast<uint16_t>(std::min<int>(f.a + 1, 65535));
            }
            if (b == '\\') f.ph = S_ESC;
            add(out, c);
            return;
          case S_ESC:
            if (b == 'u') f.ph = S_U1;
            else if (std::strchr("\"\\/bfnrt", b) && b) f.ph = S_NORMAL;
            else return;
            add(out, c);
            return;
          default:  // \uXXXX
            if (!std::isxdigit(b)) return;
            f.ph = f.ph == S_U4 ? S_NORMAL : static_cast<uint8_t>(f.ph + 1);
            add(out, c);
            return;
        }
      }
      case F_NUM: {
        const bool integer = nodes[f.node].k == K::integer;
        const bool digit = b >= '0' && b <= '9';
        auto next = [&](uint8_t ph, bool count) {
          if (count) {
            if (f.a >= kMaxNumDigits) return;
            ++f.a;
          } else {
            f.a = 0;
          }
          f.ph = ph;
          add(out, c);
        };
        switch (f.ph) {
          case N_START:
            if (b == '-') next(N_MINUS, false);
            else if (b == '0') next(N_ZERO, true);
            else if (digit) next(N_INT, true);
            return;
          case N_MINUS:
            if (b == '0') next(N_ZERO, true);
            else if (digit) next(N_INT, true);
            return;
          case N_ZERO:
          case N_INT:
            if (digit && f.ph == N_INT) next(N_INT, true);
            else if (b == '.' && !integer) next(N_DOT, false);
            else if ((b == 'e' || b == 'E') && !integer) next(N_E, false);
            else if (!digit) end_then(c, b, out);
            return;
          case N_DOT:
            if (digit) next(N_FRAC, true);
            return;
          case N_FRAC:
            if (digit) next(N_FRAC, true);
            else if (b == 'e' || b == 'E') next(N_E, false);
            else end_then(c, b, out);
            return;
          case N_E:
            if (b == '+' || b == '-') next(N_ESIGN, false);
            else if (digit) next(N_EXP, true);
            return;
          case N_ESIGN:
            if (digit) next(N_EXP, true);
            return;
          case N_EXP:
            if (digit) next(N_EXP, true);
            else end_then(c, b, out);
            return;
          default:
            return;
        }
      }
      case F_LIT: {
        const Node &n = nodes[f.node];
        uint64_t keep = 0;
        bool whole = false;
        for (size_t i = 0; i < n.lits.size(); ++i) {
          if (!((f.m >> i) & 1)) continue;
          const std::string &l = n.lits[i];
          if (l.size() == f.a) whole = true;
          else if (static_cast<uint8_t>(l[f.a]) == b) keep |= 1ull << i;
        }
        if (whole) end_then(c, b, out);  // a complete literal ends at this byte (fork)
        if (keep) {
          Config k = c;
          Frame &g = k.top();
          g.m = keep;
          ++g.a;
          k.ws = 0;
          add(out, k);
        }
        return;
      }
      case F_KEY: {
        const Node &n = nodes[f.node];
        if (b == '"') {
          for (size_t i = 0; i < n.props.size(); ++i)
            if (((f.m >> i) & 1) && n.props[i].name.size() == f.a) {
              --c.depth;
              Frame &p = c.top();
              p.ph = O_KEY_DONE;
              p.a = static_cast<uint16_t>(i);
              p.b = static_cast<uint16_t>(i + 1);
              c.ws = 0;
              add(out, c);
              return;
            }
          return;
        }
        uint64_t keep = 0;
        for (size_t i = 0; i < n.props.size(); ++i)
          if (((f.m >> i) & 1) && n.props[i].name.size() > f.a && static_cast<uint8_t>(n.props[i].name[f.a]) == b) keep |= 1ull << i;
        if (!keep) return;
        f.m = keep;
        ++f.a;
        add(out, c);
        return;
      }
      default:
        return;
    }
  }

  // The top frame (a number or literal) ends before byte b: pop it, then feed b to what is below.
  void end_then(Config c, uint8_t b, std::vector<Config> &out) const {
    std::vector<Config> tmp;
    complete(c, tmp);
    for (auto &x : tmp) feed(x, b, out);
  }

  Config initial() const {
    Config c;
    c.depth = 1;
    c.f[0].t = F_VAL;
    c.f[0].node = static_cast<uint32_t>(root);
    return c;
  }
};

}  // namespace

// ---------------------------------------------------------------- the vocabulary

struct ConstraintVocab::Impl {
  int64_t vocab = 0;
  std::vector<std::string> bytes;  // per id; empty = never allowed inside a document
  std::vector<int32_t> eos;
  // byte trie: node 0 = root; children as (byte, node) runs; ids ending at a node
  struct TNode {
    uint32_t child = 0, nchild = 0;  // into edges
    uint32_t ids = 0, nids = 0;  // into tok_ids
  };
  std::vector<TNode> nodes;
  std::vector<std::pair<uint8_t, uint32_t>> edges;
  std::vector<int32_t> tok_ids;
  // the tokens without a quote, backslash or control byte (free inside a plain string), as a mask,
  // and the trie of the others
  std::vector<uint64_t> plain;
  std::vector<TNode> s_nodes;
  std::vector<std::pair<uint8_t, uint32_t>> s_edges;
  std::vector<int32_t> s_tok_ids;

  static void build_trie(const std::vector<std::pair<std::string, int32_t>> &toks, std::vector<TNode> &nodes,
                         std::vector<std::pair<uint8_t, uint32_t>> &edges, std::vector<int32_t> &ids) {
    // toks sorted by bytes: build level by level with an explicit recursion over ranges
    nodes.assign(1, TNode{});
    edges.clear();
    ids.clear();
    struct Work {
      size_t lo, hi, depth;
      uint32_t node;
    };
    std::vector<Work> st{{0, toks.size(), 0, 0}};
    while (!st.empty()) {
      const Work w = st.back();
      st.pop_back();
      size_t i = w.lo;
      // ids ending exactly here
      TNode &tn = nodes[w.node];
      tn.ids = static_cast<uint32_t>(ids.size());
      while (i < w.hi && toks[i].first.size() == w.depth) ids.push_back(toks[i++].second);
      tn.nids = static_cast<uint32_t>(ids.size()) - tn.ids;
      // children
      std::vector<Work> kids;
      const uint32_t first_edge = static_cast<uint32_t>(edges.size());
      while (i < w.hi) {
        const uint8_t b = static_cast<uint8_t>(toks[i].first[w.depth]);
        size_t j = i;
        while (j < w.hi && static_cast<uint8_t>(toks[j].first[w.depth]) == b) ++j;
        const uint32_t child = static_cast<uint32_t>(nodes.size());
        nodes.push_back(TNode{});
        edges.push_back({b, child});
        kids.push_back({i, j, w.depth + 1, child});
        i = j;
      }
      nodes[w.node].child = first_edge;
      nodes[w.node].nchild = static_cast<uint32_t>(edges.size()) - first_edge;
      for (auto it = kids.rbegin(); it != kids.rend(); ++it) st.push_back(*it);
    }
  }
};

ConstraintVocab::ConstraintVocab() : impl_(std::make_unique<Impl>()) {}
ConstraintVocab::~ConstraintVocab() = default;
int64_t ConstraintVocab::vocab() const { return impl_->vocab; }

std::shared_ptr<const ConstraintVocab> ConstraintVocab::build(const Tokenizer &tok, int64_t model_vocab, const std::vector<int32_t> &eos) {
  std::shared_ptr<ConstraintVocab> v(new ConstraintVocab());
  Impl &m = *v->impl_;
  m.vocab = model_vocab;
  m.eos = eos;
  m.bytes.assign(static_cast<size_t>(model_vocab), {});
  m.plain.assign(static_cast<size_t>((model_vocab + 63) / 64), 0);
  std::vector<std::pair<std::string, int32_t>> all, special;
  const int32_t n = std::min<int64_t>(tok.vocab_size(), model_vocab);
  for (int32_t id = 0; id < n; ++id) {
    if (tok.is_special(id)) continue;
    std::string s;
    tok.decode_piece(id, s, false);
    if (s.empty()) continue;
    bool plain = true;
    for (unsigned char ch : s) plain &= ch != '"' && ch != '\\' && ch >= 0x20;
    if (plain) m.plain[static_cast<size_t>(id) >> 6] |= 1ull << (id & 63);
    else special.push_back({s, id});
    all.push_back({s, id});
    m.bytes[static_cast<size_t>(id)] = std::move(s);
  }
  auto cmp = [](const auto &a, const auto &b) { return a.first < b.first || (a.first == b.first && a.second < b.second); };
  std::sort(all.begin(), all.end(), cmp);
  std::sort(special.begin(), special.end(), cmp);
  Impl::build_trie(all, m.nodes, m.edges, m.tok_ids);
  Impl::build_trie(special, m.s_nodes, m.s_edges, m.s_tok_ids);
  return v;
}

// ---------------------------------------------------------------- spec + per-request constraint

namespace {

class Spec;

class Constraint final : public core::TokenConstraint {
 public:
  Constraint(const Spec &s, int32_t start_after);
  const uint64_t *allowed() override;
  void accept(int32_t token) override;
  bool failed() const { return failed_; }
  bool complete() const;
  int computed = 0;

 private:
  const Spec &s_;
  int32_t wait_;
  bool failed_ = false;
  std::vector<Config> set_;
  std::shared_ptr<const std::vector<uint64_t>> mask_;  // keeps the cached mask alive
};

class Spec final : public core::ConstraintSpec {
 public:
  Spec(Grammar g, std::shared_ptr<const ConstraintVocab> v, int32_t start_after) : g_(std::move(g)), v_(std::move(v)), start_after_(start_after) {}
  std::unique_ptr<core::TokenConstraint> start() const override { return std::make_unique<Constraint>(*this, start_after_); }
  const Grammar &grammar() const { return g_; }
  const ConstraintVocab::Impl &vocab() const { return v_->impl(); }

  // The mask of a state, cached by its configurations.
  std::shared_ptr<const std::vector<uint64_t>> mask(const std::vector<Config> &set, bool &computed) const {
    std::string key;
    for (const auto &c : set) c.key(key);
    {
      std::lock_guard<std::mutex> lk(mu_);
      if (const auto it = cache_.find(key); it != cache_.end()) {
        computed = false;
        return it->second;
      }
    }
    computed = true;
    auto m = std::make_shared<std::vector<uint64_t>>(compute(set));
    std::lock_guard<std::mutex> lk(mu_);
    if (cache_.size() >= kMaxMasks) cache_.clear();
    cache_.emplace(std::move(key), m);
    return m;
  }

  bool can_end(const std::vector<Config> &set) const {
    for (const auto &c : set)
      if (g_.can_end(c)) return true;
    return false;
  }

  void feed_all(const std::vector<Config> &in, uint8_t b, std::vector<Config> &out) const {
    out.clear();
    for (const auto &c : in) g_.feed(c, b, out);
  }

 private:
  std::vector<uint64_t> compute(const std::vector<Config> &set) const {
    const auto &V = vocab();
    std::vector<uint64_t> m(static_cast<size_t>((V.vocab + 63) / 64), 0);
    auto allow = [&](int32_t id) { m[static_cast<size_t>(id) >> 6] |= 1ull << (id & 63); };
    // A plain string (no length bounds, not in an escape): every plain token keeps it; walk only the
    // tokens with a quote / backslash / control byte.
    bool plain = set.size() == 1 && !set[0].done && set[0].depth > 0;
    if (plain) {
      const Frame &f = set[0].f[set[0].depth - 1];
      const Node &n = g_.nodes[f.node];
      plain = f.t == F_STR && f.ph == S_NORMAL && n.min_len == 0 && n.max_len < 0;
    }
    if (plain) {
      m = V.plain;
      walk(set, V.s_nodes, V.s_edges, V.s_tok_ids, allow);
    } else {
      walk(set, V.nodes, V.edges, V.tok_ids, allow);
    }
    if (can_end(set))
      for (int32_t e : V.eos)
        if (e >= 0 && e < V.vocab) allow(e);
    return m;
  }

  template <class Allow>
  void walk(const std::vector<Config> &root, const std::vector<ConstraintVocab::Impl::TNode> &nodes,
            const std::vector<std::pair<uint8_t, uint32_t>> &edges, const std::vector<int32_t> &ids, Allow allow) const {
    if (nodes.empty()) return;
    std::vector<std::vector<Config>> level(2);
    level[0] = root;
    struct It {
      uint32_t node, e;
    };
    std::vector<It> st{{0, 0}};
    while (!st.empty()) {
      It &it = st.back();
      const auto &tn = nodes[it.node];
      if (it.e == tn.nchild) {
        st.pop_back();
        continue;
      }
      const auto [b, child] = edges[tn.child + it.e];
      ++it.e;
      const size_t d = st.size();
      if (level.size() <= d) level.resize(d + 1);
      feed_all(level[d - 1], b, level[d]);
      if (level[d].empty()) continue;
      const auto &cn = nodes[child];
      for (uint32_t k = 0; k < cn.nids; ++k) allow(ids[cn.ids + k]);
      if (cn.nchild) st.push_back({child, 0});
    }
  }

  Grammar g_;
  std::shared_ptr<const ConstraintVocab> v_;
  int32_t start_after_;
  mutable std::mutex mu_;
  mutable std::unordered_map<std::string, std::shared_ptr<const std::vector<uint64_t>>> cache_;
};

Constraint::Constraint(const Spec &s, int32_t start_after) : s_(s), wait_(start_after) { set_.push_back(s.grammar().initial()); }

const uint64_t *Constraint::allowed() {
  if (wait_ >= 0 || failed_) return nullptr;
  bool computed = false;
  mask_ = s_.mask(set_, computed);
  if (computed) ++this->computed;
  return mask_->data();
}

bool Constraint::complete() const { return wait_ < 0 && !failed_ && s_.can_end(set_); }

void Constraint::accept(int32_t token) {
  if (wait_ >= 0) {
    if (token == wait_) wait_ = -1;
    return;
  }
  if (failed_) return;
  const auto &V = s_.vocab();
  if (std::find(V.eos.begin(), V.eos.end(), token) != V.eos.end()) return;
  if (token < 0 || token >= V.vocab || V.bytes[static_cast<size_t>(token)].empty()) {
    failed_ = true;
    return;
  }
  std::vector<Config> next;
  for (unsigned char b : V.bytes[static_cast<size_t>(token)]) {
    s_.feed_all(set_, b, next);
    set_.swap(next);
    if (set_.empty()) {
      failed_ = true;
      return;
    }
  }
}

}  // namespace

std::shared_ptr<const core::ConstraintSpec> compile_json_constraint(const json::Value &schema, bool raw_choice,
                                                                    std::shared_ptr<const ConstraintVocab> vocab, int32_t start_after,
                                                                    std::string &err) {
  Grammar g;
  if (raw_choice) {
    if (!schema.is_array() || schema.arr.empty()) {
      err = "guided_choice must be a non-empty list of strings";
      return nullptr;
    }
    if (schema.arr.size() > kMaxLits) {
      err = "guided_choice has more than " + std::to_string(kMaxLits) + " choices";
      return nullptr;
    }
    Compiler c(schema);
    Node n;
    n.k = K::literals;
    for (const auto &v : schema.arr) {
      if (!v.is_string() || v.str.empty()) {
        err = "guided_choice must be a list of non-empty strings";
        return nullptr;
      }
      n.lits.push_back(v.str);
    }
    g.nodes = std::move(c.nodes_);
    g.nodes.push_back(std::move(n));
    g.root = static_cast<int>(g.nodes.size()) - 1;
  } else {
    Compiler c(schema);
    const int root = c.compile(schema, 0);
    if (!c.err.empty()) {
      err = c.err;
      return nullptr;
    }
    g.nodes = std::move(c.nodes_);
    g.root = root;
  }
  return std::make_shared<Spec>(std::move(g), std::move(vocab), start_after);
}

WalkResult walk_constraint(const core::ConstraintSpec &spec, const std::vector<int32_t> &ids) {
  WalkResult r;
  auto c = spec.start();
  auto *k = static_cast<Constraint *>(c.get());
  for (size_t i = 0; i < ids.size(); ++i) {
    const uint64_t *m = k->allowed();
    if (m && !((m[static_cast<size_t>(ids[i]) >> 6] >> (ids[i] & 63)) & 1)) {
      r.first_rejected = static_cast<int>(i);
      break;
    }
    k->accept(ids[i]);
  }
  r.complete = k->complete();
  r.masks_computed = k->computed;
  return r;
}

}  // namespace xenithium:text
