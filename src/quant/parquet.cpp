#include "xenithium/quant/parquet.hpp"

#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <stdexcept>

namespace xenithium::quant {

namespace {

[[noreturn]] void fail(const std::string &what) { throw std::runtime_error("parquet: " + what); }

// ---- snappy ----------------------------------------------------------------------------------
uint64_t read_varint(const uint8_t *&p, const uint8_t *end) {
  uint64_t v = 0;
  for (int s = 0; s < 64; s += 7) {
    if (p >= end) fail("truncated varint");
    const uint8_t c = *p++;
    v |= static_cast<uint64_t>(c & 0x7f) << s;
    if (c < 0x80) return v;
  }
  fail("varint too long");
}

}  // namespace

std::string snappy_decompress(const uint8_t *src, size_t n) {
  const uint8_t *p = src, *end = src + n;
  const uint64_t len = read_varint(p, end);
  std::string out;
  out.reserve(len);
  while (p < end) {
    const uint8_t tag = *p++;
    const int kind = tag & 3;
    if (kind == 0) {
      size_t l = tag >> 2;
      if (l >= 60) {
        const int extra = static_cast<int>(l) - 59;
        if (end - p < extra) fail("snappy: truncated literal length");
        l = 0;
        for (int i = 0; i < extra; ++i) l |= static_cast<size_t>(p[i]) << (8 * i);
        p += extra;
      }
      l += 1;
      if (static_cast<size_t>(end - p) < l) fail("snappy: truncated literal");
      out.append(reinterpret_cast<const char *>(p), l);
      p += l;
      continue;
    }
    size_t l = 0, off = 0;
    if (kind == 1) {
      if (p >= end) fail("snappy: truncated copy1");
      l = ((tag >> 2) & 7) + 4;
      off = (static_cast<size_t>(tag >> 5) << 8) | *p++;
    } else if (kind == 2) {
      if (end - p < 2) fail("snappy: truncated copy2");
      l = (tag >> 2) + 1;
      off = p[0] | (static_cast<size_t>(p[1]) << 8);
      p += 2;
    } else {
      if (end - p < 4) fail("snappy: truncated copy4");
      l = (tag >> 2) + 1;
      off = p[0] | (static_cast<size_t>(p[1]) << 8) | (static_cast<size_t>(p[2]) << 16) | (static_cast<size_t>(p[3]) << 24);
      p += 4;
    }
    if (off == 0 || off > out.size()) fail("snappy: copy offset out of range");
    const size_t from = out.size() - off;
    for (size_t i = 0; i < l; ++i) out.push_back(out[from + i]);  // overlapping copies are byte-wise
  }
  if (out.size() != len) fail("snappy: length mismatch");
  return out;
}

namespace {

// ---- Thrift compact protocol: a generic tree -------------------------------------------------
struct TVal {
  int type = 0;  // compact type id
  int64_t i = 0;
  std::string s;
  std::vector<TVal> list;
  std::map<int, TVal> fields;  // struct
  const TVal *f(int id) const {
    auto it = fields.find(id);
    return it == fields.end() ? nullptr : &it->second;
  }
  int64_t fi(int id, int64_t dflt = 0) const {
    const TVal *v = f(id);
    return v ? v->i : dflt;
  }
};

struct TReader {
  const uint8_t *p, *end;
  uint64_t varint() { return read_varint(p, end); }
  static int64_t zz(uint64_t n) { return static_cast<int64_t>(n >> 1) ^ -static_cast<int64_t>(n & 1); }
  TVal value(int t, int depth) {
    if (depth > 64) fail("thrift: nesting too deep");
    TVal v;
    v.type = t;
    switch (t) {
      case 1: v.i = 1; break;  // bool true (in struct field headers)
      case 2: v.i = 0; break;
      case 3:
        if (p >= end) fail("thrift: truncated byte");
        v.i = static_cast<int8_t>(*p++);
        break;
      case 4:
      case 5:
      case 6: v.i = zz(varint()); break;
      case 7:
        if (end - p < 8) fail("thrift: truncated double");
        p += 8;
        break;
      case 8: {
        const uint64_t n = varint();
        if (static_cast<uint64_t>(end - p) < n) fail("thrift: truncated binary");
        v.s.assign(reinterpret_cast<const char *>(p), n);
        p += n;
        break;
      }
      case 9:
      case 10: {
        if (p >= end) fail("thrift: truncated list");
        const uint8_t h = *p++;
        uint64_t n = h >> 4;
        const int et = h & 15;
        if (n == 15) n = varint();
        for (uint64_t k = 0; k < n; ++k) {
          if (et == 1 || et == 2) {  // bools in a list are one byte each
            if (p >= end) fail("thrift: truncated bool list");
            TVal b;
            b.type = 1;
            b.i = *p++ == 1;
            v.list.push_back(b);
          } else {
            v.list.push_back(value(et, depth + 1));
          }
        }
        break;
      }
      case 11: fail("thrift: maps are not used by the parquet footer");
      case 12: {
        int fid = 0;
        for (;;) {
          if (p >= end) fail("thrift: truncated struct");
          const uint8_t h = *p++;
          if (h == 0) break;
          const int ft = h & 15, dl = h >> 4;
          fid = dl ? fid + dl : static_cast<int>(zz(varint()));
          v.fields[fid] = value(ft, depth + 1);
        }
        break;
      }
      default: fail("thrift: unknown type " + std::to_string(t));
    }
    return v;
  }
};

// ---- RLE / bit-packed hybrid ----------------------------------------------------------------
void rle_decode(const uint8_t *p, const uint8_t *end, int bw, size_t count, std::vector<uint32_t> &out) {
  out.clear();
  out.reserve(count);
  if (bw == 0) {
    out.assign(count, 0);
    return;
  }
  const int bytes = (bw + 7) / 8;
  while (out.size() < count) {
    if (p >= end) fail("rle: truncated");
    const uint64_t h = read_varint(p, end);
    if (h & 1) {
      const size_t groups = h >> 1, n = groups * 8;
      const size_t need = groups * bw;
      if (static_cast<size_t>(end - p) < need) fail("rle: truncated bit-packed run");
      uint64_t acc = 0;
      int have = 0;
      size_t byte = 0;
      for (size_t i = 0; i < n; ++i) {
        while (have < bw) {
          acc |= static_cast<uint64_t>(p[byte++]) << have;
          have += 8;
        }
        const uint32_t v = static_cast<uint32_t>(acc & ((1ull << bw) - 1));
        acc >>= bw;
        have -= bw;
        if (out.size() < count) out.push_back(v);
      }
      p += need;
    } else {
      const size_t n = h >> 1;
      if (end - p < bytes) fail("rle: truncated run value");
      uint32_t v = 0;
      for (int i = 0; i < bytes; ++i) v |= static_cast<uint32_t>(p[i]) << (8 * i);
      p += bytes;
      for (size_t i = 0; i < n && out.size() < count; ++i) out.push_back(v);
    }
  }
}

int bit_width(int max_level) {
  int b = 0;
  while ((1 << b) <= max_level) ++b;
  return max_level == 0 ? 0 : b;
}

std::string read_file_range(FILE *f, const std::string &path, uint64_t off, uint64_t n) {
  std::string b(n, '\0');
  if (fseeko(f, static_cast<off_t>(off), SEEK_SET) != 0 || fread(b.data(), 1, n, f) != n) fail(path + ": short read");
  return b;
}

}  // namespace

ParquetFile ParquetFile::open(const std::string &path) {
  std::unique_ptr<FILE, int (*)(FILE *)> f(std::fopen(path.c_str(), "rb"), &std::fclose);
  if (!f) fail(path + ": cannot open");
  fseeko(f.get(), 0, SEEK_END);
  const uint64_t size = static_cast<uint64_t>(ftello(f.get()));
  if (size < 12) fail(path + ": too small");
  const std::string tail = read_file_range(f.get(), path, size - 8, 8);
  if (std::memcmp(tail.data() + 4, "PAR1", 4) != 0) fail(path + ": no PAR1 magic");
  uint32_t mlen = 0;
  std::memcpy(&mlen, tail.data(), 4);
  if (mlen + 12ull > size) fail(path + ": footer length out of range");
  const std::string meta = read_file_range(f.get(), path, size - 8 - mlen, mlen);
  TReader r{reinterpret_cast<const uint8_t *>(meta.data()), reinterpret_cast<const uint8_t *>(meta.data()) + meta.size()};
  const TVal fm = r.value(12, 0);

  ParquetFile pf;
  pf.path_ = path;
  pf.rows_ = fm.fi(3);
  // Schema: a depth-first list; levels accumulate along each leaf's path.
  const TVal *schema = fm.f(2);
  if (!schema || schema->list.empty()) fail(path + ": no schema");
  struct Frame {
    int left;
    int def, rep;
    std::string prefix;
  };
  std::vector<Frame> stack;
  std::map<std::string, ParquetColumn> leaves;
  for (size_t i = 0; i < schema->list.size(); ++i) {
    const TVal &e = schema->list[i];
    const int children = static_cast<int>(e.fi(5, 0));
    const int rt = static_cast<int>(e.fi(3, 0));
    if (i == 0) {
      stack.push_back({children, 0, 0, ""});
      continue;
    }
    Frame &top = stack.back();
    const std::string name = e.f(4) ? e.f(4)->s : "";
    const std::string full = top.prefix.empty() ? name : top.prefix + "." + name;
    const int def = top.def + (rt != 0 ? 1 : 0), rep = top.rep + (rt == 2 ? 1 : 0);
    --top.left;
    if (children > 0) {
      stack.push_back({children, def, rep, full});
    } else {
      ParquetColumn c;
      c.path = full;
      c.max_def = def;
      c.max_rep = rep;
      c.physical = static_cast<int>(e.fi(1, -1));
      leaves[full] = c;
    }
    while (stack.size() > 1 && stack.back().left == 0) stack.pop_back();
  }
  // Column order = the row group's column chunk order.
  const TVal *rgs = fm.f(4);
  if (!rgs) fail(path + ": no row groups");
  bool first = true;
  for (const TVal &rg : rgs->list) {
    const TVal *cols = rg.f(1);
    if (!cols) fail(path + ": row group without columns");
    int ci = 0;
    for (const TVal &cc : cols->list) {
      const TVal *md = cc.f(3);
      if (!md) fail(path + ": column chunk without metadata (external files unsupported)");
      std::string p;
      for (const TVal &s : md->f(3)->list) p += (p.empty() ? "" : ".") + s.s;
      if (first) {
        auto it = leaves.find(p);
        if (it == leaves.end()) fail(path + ": column " + p + " not in the schema");
        pf.cols_.push_back(it->second);
      } else if (ci >= static_cast<int>(pf.cols_.size()) || pf.cols_[static_cast<size_t>(ci)].path != p) {
        fail(path + ": row groups disagree on the column order");
      }
      Chunk ch;
      ch.col = ci;
      ch.codec = static_cast<int>(md->fi(4));
      ch.values = md->fi(5);
      const int64_t data_off = md->fi(9), dict_off = md->f(11) ? md->fi(11) : 0;
      ch.offset = static_cast<uint64_t>(dict_off > 0 && dict_off < data_off ? dict_off : data_off);
      ch.bytes = static_cast<uint64_t>(md->fi(7));
      if (ch.offset + ch.bytes > size) fail(path + ": column chunk out of range");
      pf.chunks_.push_back(ch);
      ++ci;
    }
    first = false;
  }
  return pf;
}

void ParquetFile::read_column(const std::string &column, std::vector<uint8_t> &rep, std::vector<uint8_t> &def,
                              std::vector<std::string> &vals) const {
  int ci = -1;
  for (size_t i = 0; i < cols_.size(); ++i)
    if (cols_[i].path == column) ci = static_cast<int>(i);
  if (ci < 0) fail(path_ + ": no column " + column);
  const ParquetColumn &C = cols_[static_cast<size_t>(ci)];
  if (C.physical != 6) fail(path_ + ": column " + column + " is not BYTE_ARRAY");
  if (C.max_rep > 1) fail(path_ + ": column " + column + " is nested deeper than one list");
  std::unique_ptr<FILE, int (*)(FILE *)> f(std::fopen(path_.c_str(), "rb"), &std::fclose);
  if (!f) fail(path_ + ": cannot open");
  const int dbw = bit_width(C.max_def), rbw = bit_width(C.max_rep);
  std::vector<uint32_t> lv, idx;
  for (const Chunk &ch : chunks_) {
    if (ch.col != ci) continue;
    if (ch.codec != 0 && ch.codec != 1) fail(path_ + ": codec " + std::to_string(ch.codec) + " unsupported (SNAPPY or none)");
    const std::string raw = read_file_range(f.get(), path_, ch.offset, ch.bytes);
    TReader r{reinterpret_cast<const uint8_t *>(raw.data()), reinterpret_cast<const uint8_t *>(raw.data()) + raw.size()};
    std::vector<std::string> dict;
    int64_t seen = 0;
    while (seen < ch.values) {
      const TVal ph = r.value(12, 0);
      const int ptype = static_cast<int>(ph.fi(1));
      const size_t usize = static_cast<size_t>(ph.fi(2)), csize = static_cast<size_t>(ph.fi(3));
      if (static_cast<size_t>(r.end - r.p) < csize) fail(path_ + ": truncated page");
      const uint8_t *body = r.p;
      r.p += csize;
      auto decompress = [&](const uint8_t *b, size_t n, size_t want) {
        if (ch.codec == 0) return std::string(reinterpret_cast<const char *>(b), n);
        std::string s = snappy_decompress(b, n);
        if (s.size() != want) fail(path_ + ": page size mismatch after decompression");
        return s;
      };
      auto plain_strings = [&](const uint8_t *p, const uint8_t *end, size_t n, std::vector<std::string> &out) {
        for (size_t i = 0; i < n; ++i) {
          if (end - p < 4) fail(path_ + ": truncated PLAIN value");
          uint32_t l = 0;
          std::memcpy(&l, p, 4);
          p += 4;
          if (static_cast<size_t>(end - p) < l) fail(path_ + ": truncated PLAIN value");
          out.emplace_back(reinterpret_cast<const char *>(p), l);
          p += l;
        }
      };
      if (ptype == 2) {  // dictionary page
        const TVal *dh = ph.f(7);
        const std::string d = decompress(body, csize, usize);
        const auto *dp = reinterpret_cast<const uint8_t *>(d.data());
        dict.clear();
        plain_strings(dp, dp + d.size(), static_cast<size_t>(dh->fi(1)), dict);
        continue;
      }
      if (ptype != 0 && ptype != 3) continue;  // index pages
      std::string page;
      const uint8_t *p = nullptr, *end = nullptr;
      size_t n = 0;
      int enc = 0;
      std::vector<uint32_t> rl, dl;
      if (ptype == 0) {
        const TVal *dh = ph.f(5);
        n = static_cast<size_t>(dh->fi(1));
        enc = static_cast<int>(dh->fi(2));
        page = decompress(body, csize, usize);
        p = reinterpret_cast<const uint8_t *>(page.data());
        end = p + page.size();
        auto levels = [&](int max, int bw, std::vector<uint32_t> &out) {
          if (max == 0) {
            out.assign(n, 0);
            return;
          }
          if (end - p < 4) fail(path_ + ": truncated level block");
          uint32_t l = 0;
          std::memcpy(&l, p, 4);
          p += 4;
          if (static_cast<size_t>(end - p) < l) fail(path_ + ": truncated level block");
          rle_decode(p, p + l, bw, n, out);
          p += l;
        };
        levels(C.max_rep, rbw, rl);
        levels(C.max_def, dbw, dl);
      } else {
        const TVal *dh = ph.f(8);
        n = static_cast<size_t>(dh->fi(1));
        enc = static_cast<int>(dh->fi(4));
        const size_t dlen = static_cast<size_t>(dh->fi(5)), rlen = static_cast<size_t>(dh->fi(6));
        const bool compressed = dh->f(7) ? dh->fi(7) != 0 : true;
        if (dlen + rlen > csize) fail(path_ + ": v2 level lengths exceed the page");
        if (C.max_rep) rle_decode(body, body + rlen, rbw, n, rl); else rl.assign(n, 0);
        if (C.max_def) rle_decode(body + rlen, body + rlen + dlen, dbw, n, dl); else dl.assign(n, 0);
        const uint8_t *vb = body + rlen + dlen;
        const size_t vn = csize - rlen - dlen;
        page = compressed ? decompress(vb, vn, usize - rlen - dlen) : std::string(reinterpret_cast<const char *>(vb), vn);
        p = reinterpret_cast<const uint8_t *>(page.data());
        end = p + page.size();
      }
      size_t present = 0;
      for (size_t i = 0; i < n; ++i) present += static_cast<int>(dl[i]) == C.max_def;
      std::vector<std::string> pv;
      pv.reserve(present);
      if (enc == 0) {
        plain_strings(p, end, present, pv);
      } else if (enc == 8 || enc == 2) {  // RLE_DICTIONARY / PLAIN_DICTIONARY
        if (p >= end) fail(path_ + ": empty dictionary-encoded page");
        const int bw = *p++;
        rle_decode(p, end, bw, present, idx);
        for (const uint32_t k : idx) {
          if (k >= dict.size()) fail(path_ + ": dictionary index out of range");
          pv.push_back(dict[k]);
        }
      } else {
        fail(path_ + ": value encoding " + std::to_string(enc) + " unsupported");
      }
      size_t vi = 0;
      for (size_t i = 0; i < n; ++i) {
        rep.push_back(static_cast<uint8_t>(rl[i]));
        def.push_back(static_cast<uint8_t>(dl[i]));
        vals.push_back(static_cast<int>(dl[i]) == C.max_def ? std::move(pv[vi++]) : std::string());
      }
      seen += static_cast<int64_t>(n);
    }
  }
}

std::vector<std::string> ParquetFile::strings(const std::string &column) const {
  std::vector<uint8_t> rep, def;
  std::vector<std::string> vals;
  read_column(column, rep, def, vals);
  for (const ParquetColumn &c : cols_)
    if (c.path == column && c.max_rep != 0) fail(path_ + ": " + column + " is a list column");
  if (static_cast<int64_t>(vals.size()) != rows_) fail(path_ + ": " + column + " value count != rows");
  return vals;
}

std::vector<std::vector<std::string>> ParquetFile::string_lists(const std::string &column) const {
  std::vector<uint8_t> rep, def;
  std::vector<std::string> vals;
  read_column(column, rep, def, vals);
  int max_def = 0;
  for (const ParquetColumn &c : cols_)
    if (c.path == column) {
      if (c.max_rep != 1) fail(path_ + ": " + column + " is not a one-level list column");
      max_def = c.max_def;
    }
  // The repeated level sits at def depth (max_def - 1) or (max_def - 2) depending on whether the
  // element itself is optional; an entry is a list element when def reaches the repeated node.
  // Both Arrow layouts (list.element optional / required leaf) put the repeated node at def >= 2
  // under an optional list: count an element whenever def >= max_def - (max_def >= 3 ? 2: 1).
  const int elem_def = max_def >= 3 ? max_def - 2 : max_def - 1;
  std::vector<std::vector<std::string>> rows;
  for (size_t i = 0; i < vals.size(); ++i) {
    if (rep[i] == 0) rows.emplace_back();
    if (rows.empty()) fail(path_ + ": list column starts with a continuation");
    if (def[i] >= elem_def) rows.back().push_back(std::move(vals[i]));
  }
  if (static_cast<int64_t>(rows.size()) != rows_) fail(path_ + ": " + column + " row count mismatch");
  return rows;
}

}  // namespace xenithium:quant
