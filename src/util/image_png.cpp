// Xenithium · PNG decoder and zlib inflate (RFC 1950 / 1951), written for the engine's image input
// (xenithium/util/image.hpp). Output: 8-bit gray / gray + alpha / RGB / RGBA; palette images expand to RGB, or RGBA
// when a tRNS chunk gives the palette alphas; 1-, 2- and 4-bit gray scale to 0..255 (x 255, 85, 17); 16-bit samples
// keep their high byte; a gray / RGB tRNS colour key is kept for the RGB conversion.
#include <cstring>

#include "xenithium/util/image.hpp"

namespace xenithium::util {

namespace {

// ---- inflate ---------------------------------------------------------------------------------------------------------

struct Bits {
  const uint8_t *p, *end;
  uint64_t buf = 0;
  int n = 0;
  int pad = 0;  // zero bits appended past the end of the data
  void fill() {
    while (n <= 56) {
      uint64_t b = 0;
      if (p < end) b = *p++;
      else pad += 8;
      buf |= b << n;
      n += 8;
    }
  }
  bool over() const { return pad > n; }  // some appended zero bits were consumed: the stream was truncated
  uint32_t get(int k) {
    if (k == 0) return 0;
    if (n < k) fill();
    const uint32_t v = static_cast<uint32_t>(buf & ((uint64_t(1) << k) - 1));
    buf >>= k;
    n -= k;
    return v;
  }
  void align() {
    const int r = n & 7;
    buf >>= r;
    n -= r;
  }
};

// Canonical Huffman code: a 9-bit first-level table, longer codes by the counts (slow path).
struct Huff {
  int16_t fast[512];  // symbol | (length << 9), -1 = longer than 9 bits
  uint16_t count[16] = {};
  uint16_t sym[288] = {};
  bool build(const uint8_t *lens, int n) {
    std::memset(count, 0, sizeof count);
    for (int i = 0; i < n; ++i) ++count[lens[i]];
    count[0] = 0;
    int left = 1;
    for (int l = 1; l < 16; ++l) {
      left <<= 1;
      left -= count[l];
      if (left < 0) return false;  // over-subscribed
    }
    uint16_t offs[16];
    offs[1] = 0;
    for (int l = 1; l < 15; ++l) offs[l + 1] = static_cast<uint16_t>(offs[l] + count[l]);
    for (int i = 0; i < n; ++i)
      if (lens[i]) sym[offs[lens[i]]++] = static_cast<uint16_t>(i);
    for (auto &f : fast) f = -1;
    // first-level table: walk codes in canonical order
    int code = 0, k = 0;
    for (int l = 1; l <= 9; ++l) {
      for (int c = 0; c < count[l]; ++c, ++code, ++k) {
        // reverse the l-bit code (deflate sends codes MSB first in the bit stream)
        int r = 0;
        for (int b = 0; b < l; ++b) r |= ((code >> b) & 1) << (l - 1 - b);
        for (int x = r; x < 512; x += 1 << l) fast[x] = static_cast<int16_t>(sym[k] | (l << 9));
      }
      code <<= 1;
    }
    return true;
  }
  int decode(Bits &b) const {
    if (b.n < 16) b.fill();
    const int f = fast[b.buf & 511];
    if (f >= 0) {
      const int l = f >> 9;
      b.buf >>= l;
      b.n -= l;
      return f & 511;
    }
    // slow path: canonical decode bit by bit
    int code = 0, first = 0, index = 0;
    for (int l = 1; l < 16; ++l) {
      code |= static_cast<int>(b.get(1));
      const int c = count[l];
      if (code - c < first) return sym[index + (code - first)];
      index += c;
      first += c;
      first <<= 1;
      code <<= 1;
    }
    return -1;
  }
};

const uint16_t kLenBase[29] = {3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258};
const uint8_t kLenExtra[29] = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0};
const uint16_t kDistBase[30] = {1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577};
const uint8_t kDistExtra[30] = {0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13};

bool inflate_raw(Bits &b, std::vector<uint8_t> &out, size_t limit, std::string &err) {
  Huff lit, dist;
  for (;;) {
    const uint32_t final = b.get(1), type = b.get(2);
    if (type == 0) {
      b.align();
      const uint32_t len = b.get(16), nlen = b.get(16);
      if ((len ^ 0xFFFF) != nlen) return err = "PNG: corrupt stored block", false;
      for (uint32_t i = 0; i < len; ++i) out.push_back(static_cast<uint8_t>(b.get(8)));
    } else if (type == 1 || type == 2) {
      uint8_t lens[320];
      int nlit = 288, ndist = 30;
      if (type == 1) {
        for (int i = 0; i < 144; ++i) lens[i] = 8;
        for (int i = 144; i < 256; ++i) lens[i] = 9;
        for (int i = 256; i < 280; ++i) lens[i] = 7;
        for (int i = 280; i < 288; ++i) lens[i] = 8;
        for (int i = 0; i < 30; ++i) lens[288 + i] = 5;
      } else {
        nlit = static_cast<int>(b.get(5)) + 257;
        ndist = static_cast<int>(b.get(5)) + 1;
        const int ncl = static_cast<int>(b.get(4)) + 4;
        static const uint8_t order[19] = {16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15};
        uint8_t cl[19] = {};
        for (int i = 0; i < ncl; ++i) cl[order[i]] = static_cast<uint8_t>(b.get(3));
        Huff clh;
        if (!clh.build(cl, 19)) return err = "PNG: corrupt code lengths", false;
        int i = 0;
        while (i < nlit + ndist) {
          const int s = clh.decode(b);
          if (s < 0) return err = "PNG: corrupt code lengths", false;
          if (s < 16) {
            lens[i++] = static_cast<uint8_t>(s);
            continue;
          }
          int rep = 0;
          uint8_t v = 0;
          if (s == 16) {
            if (i == 0) return err = "PNG: corrupt code lengths", false;
            v = lens[i - 1], rep = 3 + static_cast<int>(b.get(2));
          } else if (s == 17) {
            rep = 3 + static_cast<int>(b.get(3));
          } else {
            rep = 11 + static_cast<int>(b.get(7));
          }
          if (i + rep > nlit + ndist) return err = "PNG: corrupt code lengths", false;
          while (rep--) lens[i++] = v;
        }
        // the distance lengths follow the literal ones in the array; move them to slot 288
        std::memmove(lens + 288, lens + nlit, static_cast<size_t>(ndist));
        for (int k = nlit; k < 288; ++k) lens[k] = 0;
        for (int k = 288 + ndist; k < 318; ++k) lens[k] = 0;
      }
      if (!lit.build(lens, 288) || !dist.build(lens + 288, 30)) return err = "PNG: corrupt Huffman table", false;
      for (;;) {
        const int s = lit.decode(b);
        if (s < 0 || b.over()) return err = "PNG: corrupt compressed data", false;
        if (s < 256) {
          out.push_back(static_cast<uint8_t>(s));
          continue;
        }
        if (s == 256) break;
        if (s > 285) return err = "PNG: corrupt compressed data", false;
        const size_t len = kLenBase[s - 257] + b.get(kLenExtra[s - 257]);
        const int ds = dist.decode(b);
        if (ds < 0 || ds > 29) return err = "PNG: corrupt compressed data", false;
        const size_t d = kDistBase[ds] + b.get(kDistExtra[ds]);
        if (d > out.size()) return err = "PNG: corrupt compressed data (distance)", false;
        const size_t from = out.size() - d;
        for (size_t k = 0; k < len; ++k) out.push_back(out[from + k]);
      }
    } else {
      return err = "PNG: corrupt compressed data (block type)", false;
    }
    if (out.size() > limit) return err = "PNG: image data larger than its header says", false;
    if (b.over()) return err = "PNG: truncated compressed data", false;
    if (final) return true;
  }
}

uint32_t be32(const uint8_t *p) { return uint32_t(p[0]) << 24 | uint32_t(p[1]) << 16 | uint32_t(p[2]) << 8 | p[3]; }

int paeth(int a, int b, int c) {
  const int p = a + b - c, pa = p > a ? p - a : a - p, pb = p > b ? p - b : b - p, pc = p > c ? p - c : c - p;
  return pa <= pb && pa <= pc ? a : pb <= pc ? b : c;
}

// Undo the row filters of one (sub)image of w x h pixels; bpp = bytes per pixel (>= 1), stride = bytes per row.
bool unfilter(uint8_t *data, const uint8_t *src, int h, size_t stride, int bpp, std::string &err) {
  std::vector<uint8_t> zero(stride, 0);
  const uint8_t *prev = zero.data();
  for (int y = 0; y < h; ++y) {
    const int f = src[0];
    const uint8_t *in = src + 1;
    uint8_t *o = data + static_cast<size_t>(y) * stride;
    for (size_t i = 0; i < stride; ++i) {
      const int a = i >= static_cast<size_t>(bpp) ? o[i - bpp] : 0, b = prev[i], c = i >= static_cast<size_t>(bpp) ? prev[i - bpp] : 0;
      int v = in[i];
      switch (f) {
        case 0: break;
        case 1: v += a; break;
        case 2: v += b; break;
        case 3: v += (a + b) >> 1; break;
        case 4: v += paeth(a, b, c); break;
        default: return err = "PNG: bad filter type", false;
      }
      o[i] = static_cast<uint8_t>(v);
    }
    prev = o;
    src += 1 + stride;
  }
  return true;
}

}  // namespace

bool zlib_inflate(const uint8_t *data, size_t n, std::vector<uint8_t> &out, size_t expect, std::string &err) {
  if (n < 2 || ((data[0] << 8) | data[1]) % 31 != 0 || (data[0] & 15) != 8) return err = "PNG: bad zlib header", false;
  if (data[1] & 0x20) return err = "PNG: zlib preset dictionary", false;
  Bits b{data + 2, data + n};
  out.clear();
  if (expect) out.reserve(expect);
  return inflate_raw(b, out, expect ? expect + 1 : SIZE_MAX, err);
}

bool decode_png(const uint8_t *d, size_t n, RawImage &out, std::string &err) {
  size_t p = 8;
  int w = 0, h = 0, depth = 0, ct = -1, interlace = 0;
  std::vector<uint8_t> idat, pal, trns;
  bool have_trns = false;
  int orientation = 1;
  while (p + 12 <= n) {
    const uint32_t len = be32(d + p);
    const uint8_t *type = d + p + 4, *body = d + p + 8;
    if (len > n - p - 12) return err = "PNG: truncated chunk", false;
    if (!std::memcmp(type, "IHDR", 4)) {
      if (len < 13) return err = "PNG: bad IHDR", false;
      w = static_cast<int>(be32(body)), h = static_cast<int>(be32(body + 4));
      depth = body[8], ct = body[9], interlace = body[12];
      if (body[10] != 0 || body[11] != 0 || interlace > 1) return err = "PNG: unsupported compression / filter / interlace method", false;
    } else if (!std::memcmp(type, "PLTE", 4)) {
      pal.assign(body, body + len);
    } else if (!std::memcmp(type, "tRNS", 4)) {
      trns.assign(body, body + len), have_trns = true;
    } else if (!std::memcmp(type, "IDAT", 4)) {
      idat.insert(idat.end(), body, body + len);
    } else if (!std::memcmp(type, "eXIf", 4)) {
      orientation = exif_orientation(body, len);
    } else if (!std::memcmp(type, "IEND", 4)) {
      break;
    }
    p += 12 + len;
  }
  if (w <= 0 || h <= 0 || ct < 0) return err = "PNG: no IHDR", false;
  if (static_cast<int64_t>(w) * h > kImageMaxPixels) return err = "image too large (" + std::to_string(w) + " x " + std::to_string(h) + " pixels)", false;
  int samples;
  switch (ct) {
    case 0: samples = 1; break;
    case 2: samples = 3; break;
    case 3: samples = 1; break;
    case 4: samples = 2; break;
    case 6: samples = 4; break;
    default: return err = "PNG: bad color type", false;
  }
  const bool depth_ok = ct == 0 ? (depth == 1 || depth == 2 || depth == 4 || depth == 8 || depth == 16)
                        : ct == 3 ? (depth == 1 || depth == 2 || depth == 4 || depth == 8)
                                  : (depth == 8 || depth == 16);
  if (!depth_ok) return err = "PNG: bad bit depth", false;
  if (ct == 3 && pal.size() < 3) return err = "PNG: palette image without PLTE", false;
  const int bits_px = samples * depth;
  const int bpp = bits_px >= 8 ? bits_px / 8 : 1;
  // the (sub)images and their decoded size
  struct Pass {
    int x0, y0, dx, dy, w, h;
  };
  std::vector<Pass> passes;
  if (interlace) {
    const int P[7][4] = {{0, 0, 8, 8}, {4, 0, 8, 8}, {0, 4, 4, 8}, {2, 0, 4, 4}, {0, 2, 2, 4}, {1, 0, 2, 2}, {0, 1, 1, 2}};
    for (const auto &q : P) passes.push_back({q[0], q[1], q[2], q[3], (w - q[0] + q[2] - 1) / q[2], (h - q[1] + q[3] - 1) / q[3]});
  } else {
    passes.push_back({0, 0, 1, 1, w, h});
  }
  size_t total = 0;
  for (auto &q : passes)
    if (q.w > 0 && q.h > 0) total += static_cast<size_t>(q.h) * (1 + (static_cast<size_t>(q.w) * bits_px + 7) / 8);
  std::vector<uint8_t> raw;
  if (!zlib_inflate(idat.data(), idat.size(), raw, total, err)) return false;
  if (raw.size() < total) return err = "PNG: truncated image data", false;

  // samples -> 8-bit, then the output channels
  const bool pal_alpha = ct == 3 && have_trns;
  out.w = w, out.h = h;
  out.channels = ct == 3 ? (pal_alpha ? 4 : 3) : samples;
  out.px.assign(static_cast<size_t>(w) * h * out.channels, 0);
  out.orientation = orientation;
  if (have_trns && (ct == 0 || ct == 2)) {
    // the key compares at the file's bit depth; at 16 bits a sample is kept by its high byte, so compare that
    auto key_at = [&](int i) -> uint8_t {
      const uint16_t v = static_cast<uint16_t>(trns.size() >= static_cast<size_t>(2 * i + 2) ? (trns[2 * i] << 8 | trns[2 * i + 1]) : 0);
      if (depth == 16) return static_cast<uint8_t>(v >> 8);
      if (depth == 8) return static_cast<uint8_t>(v);
      return static_cast<uint8_t>(v * (depth == 1 ? 255 : depth == 2 ? 85 : 17));
    };
    out.key = true;
    for (int c = 0; c < (ct == 0 ? 1 : 3); ++c) out.key_rgb[c] = key_at(c);
  }
  size_t off = 0;
  std::vector<uint8_t> img;
  for (const auto &q : passes) {
    if (q.w <= 0 || q.h <= 0) continue;
    const size_t stride = (static_cast<size_t>(q.w) * bits_px + 7) / 8;
    img.assign(stride * q.h, 0);
    if (!unfilter(img.data(), raw.data() + off, q.h, stride, bpp, err)) return false;
    off += static_cast<size_t>(q.h) * (1 + stride);
    for (int y = 0; y < q.h; ++y) {
      const uint8_t *row = img.data() + static_cast<size_t>(y) * stride;
      for (int x = 0; x < q.w; ++x) {
        uint8_t *o = out.px.data() + (static_cast<size_t>(q.y0 + y * q.dy) * w + (q.x0 + x * q.dx)) * out.channels;
        if (depth < 8) {
          const int bit = x * depth, v = (row[bit >> 3] >> (8 - depth - (bit & 7))) & ((1 << depth) - 1);
          if (ct == 3) {
            const size_t e = static_cast<size_t>(v) * 3;
            if (e + 3 > pal.size()) return err = "PNG: palette index out of range", false;
            o[0] = pal[e], o[1] = pal[e + 1], o[2] = pal[e + 2];
            if (pal_alpha) o[3] = static_cast<size_t>(v) < trns.size() ? trns[v] : 255;
          } else {
            o[0] = static_cast<uint8_t>(v * (depth == 1 ? 255 : depth == 2 ? 85 : 17));
          }
        } else if (ct == 3) {
          const int v = row[x];
          const size_t e = static_cast<size_t>(v) * 3;
          if (e + 3 > pal.size()) return err = "PNG: palette index out of range", false;
          o[0] = pal[e], o[1] = pal[e + 1], o[2] = pal[e + 2];
          if (pal_alpha) o[3] = static_cast<size_t>(v) < trns.size() ? trns[v] : 255;
        } else {
          const int step = depth / 8;
          for (int s = 0; s < samples; ++s) o[s] = row[(static_cast<size_t>(x) * samples + s) * step];  // 16-bit: the high byte
        }
      }
    }
  }
  return true;
}

}  // namespace xenithium:util
