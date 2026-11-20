// Xenithium · JPEG decoder (ITU T.81 Huffman: baseline / extended SOF0-1, progressive SOF2), written for the engine's
// image input (xenithium/util/image.hpp). Every scan's coefficients are kept per component, then each block is
// dequantised and inverse-transformed by the accurate integer IDCT (13-bit constants, 2 extra pass-1 bits: the
// islow form), chroma upsampled by the triangle filter for 2x horizontal / 2x both / 2x vertical
// subsampling (with alternating rounding biases) and replication otherwise, then YCbCr -> RGB by
// 16-bit fixed-point tables. Arithmetic coding, lossless and hierarchical modes are refused.
#include <algorithm>
#include <cstring>

#include "xenithium/util/image.hpp"

namespace xenithium::util {

namespace {

const uint8_t kZigzag[64 + 16] = {0,  1,  8,  16, 9,  2,  3,  10, 17, 24, 32, 25, 18, 11, 4,  5,  12, 19, 26, 33, 40, 48,
                                  41, 34, 27, 20, 13, 6,  7,  14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23,
                                  30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
                                  // a corrupt run past 63 lands here (63) instead of out of the block
                                  63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63};

struct HuffTable {
  bool defined = false;
  uint8_t look_len[256];  // 8-bit lookahead: code length (0 = longer)
  uint8_t look_sym[256];
  int32_t maxcode[18];  // per length: the largest code, -1 = none
  int32_t valptr[17];
  int32_t mincode[17];
  uint8_t vals[256];
  bool build(const uint8_t *counts, const uint8_t *symbols, int nsym) {
    std::memcpy(vals, symbols, static_cast<size_t>(nsym));
    std::memset(look_len, 0, sizeof look_len);
    int code = 0, k = 0;
    for (int l = 1; l <= 16; ++l) {
      valptr[l] = k;
      mincode[l] = code;
      code += counts[l - 1];
      k += counts[l - 1];
      maxcode[l] = counts[l - 1] ? code - 1 : -1;
      if (code > (1 << l)) return false;
      code <<= 1;
    }
    maxcode[17] = 0x7FFFFFFF;
    // the 8-bit lookahead
    code = 0, k = 0;
    for (int l = 1; l <= 8; ++l) {
      for (int i = 0; i < counts[l - 1]; ++i, ++k, ++code) {
        const int lo = code << (8 - l), n = 1 << (8 - l);
        for (int j = 0; j < n; ++j) look_len[lo + j] = static_cast<uint8_t>(l), look_sym[lo + j] = vals[k];
      }
      code <<= 1;
    }
    defined = true;
    return true;
  }
};

// MSB-first entropy-coded segment reader: 0xFF 0x00 is a data 0xFF; a marker stops the data (zeros follow).
struct BitReader {
  const uint8_t *d;
  size_t n, pos;
  uint64_t buf = 0;
  int bits = 0;
  bool marker = false;
  void fill() {
    while (bits <= 56) {
      uint64_t b = 0;
      if (!marker && pos < n) {
        b = d[pos];
        if (b == 0xFF) {
          const uint8_t nx = pos + 1 < n ? d[pos + 1] : 0xD9;
          if (nx == 0x00) {
            pos += 2;
          } else {
            marker = true;  // leave pos at the marker
            b = 0;
          }
        } else {
          ++pos;
        }
      }
      buf |= b << (56 - bits);
      bits += 8;
    }
  }
  uint32_t peek(int k) {
    if (bits < k) fill();
    return static_cast<uint32_t>(buf >> (64 - k));
  }
  void skip(int k) {
    buf <<= k;
    bits -= k;
  }
  uint32_t get(int k) {
    if (k == 0) return 0;
    const uint32_t v = peek(k);
    skip(k);
    return v;
  }
  int decode(const HuffTable &h) {
    const uint32_t p = peek(16);
    const int l8 = h.look_len[p >> 8];
    if (l8) {
      skip(l8);
      return h.look_sym[p >> 8];
    }
    for (int l = 9; l <= 16; ++l) {
      const int32_t c = static_cast<int32_t>(p >> (16 - l));
      if (h.maxcode[l] >= 0 && c <= h.maxcode[l]) {
        skip(l);
        return h.vals[h.valptr[l] + c - h.mincode[l]];
      }
    }
    skip(16);
    return 0;  // corrupt: 0 is substituted
  }
  int extend(int s) {
    if (s == 0) return 0;
    const int v = static_cast<int>(get(s));
    return v < (1 << (s - 1)) ? v - (1 << s) + 1 : v;
  }
  // a restart: drop the buffered bits, consume the RSTn marker
  void restart() {
    buf = 0, bits = 0, marker = false;
    while (pos + 1 < n && !(d[pos] == 0xFF && d[pos + 1] >= 0xD0 && d[pos + 1] <= 0xD7)) {
      if (d[pos] == 0xFF && d[pos + 1] != 0 && d[pos + 1] != 0xFF) return;  // another marker: leave it
      ++pos;
    }
    if (pos + 1 < n) pos += 2;
  }
};

struct Comp {
  int id = 0, h = 1, v = 1, tq = 0;
  int bw = 0, bh = 0;  // blocks stored (whole MCUs)
  int w = 0, hgt = 0;  // samples (the component's own size)
  int td = 0, ta = 0;  // the current scan's tables
  int dc_pred = 0;
  std::vector<int16_t> coef;  // bw x bh blocks of 64 (natural order)
  std::vector<uint8_t> pix;  // after the IDCT: (bw x 8) x (bh x 8)
};

inline uint8_t clamp8(int v) { return static_cast<uint8_t>(v < 0 ? 0 : v > 255 ? 255 : v); }

// The accurate integer IDCT (CONST_BITS 13, PASS1_BITS 2), dequantising on the way in.
void idct_islow(const int16_t *in, const uint16_t *q, uint8_t *out, int stride) {
  constexpr int CB = 13, P1 = 2;
  constexpr int32_t F0298 = 2446, F0390 = 3196, F0541 = 4433, F0765 = 6270, F0899 = 7373, F1175 = 9633, F1501 = 12299, F1847 = 15137, F1961 = 16069,
                    F2053 = 16819, F2562 = 20995, F3072 = 25172;
  auto descale = [](int64_t x, int n) { return static_cast<int32_t>((x + (int64_t(1) << (n - 1))) >> n); };
  int32_t ws[64];
  for (int c = 0; c < 8; ++c) {
    auto Q = [&](int r) { return static_cast<int32_t>(in[r * 8 + c]) * q[r * 8 + c]; };
    int32_t z2 = Q(2), z3 = Q(6);
    int32_t z1 = (z2 + z3) * F0541;
    int32_t tmp2 = z1 + z3 * -F1847, tmp3 = z1 + z2 * F0765;
    z2 = Q(0), z3 = Q(4);
    int32_t tmp0 = (z2 + z3) * (1 << CB), tmp1 = (z2 - z3) * (1 << CB);
    const int32_t t10 = tmp0 + tmp3, t13 = tmp0 - tmp3, t11 = tmp1 + tmp2, t12 = tmp1 - tmp2;
    tmp0 = Q(7), tmp1 = Q(5), tmp2 = Q(3), tmp3 = Q(1);
    z1 = tmp0 + tmp3, z2 = tmp1 + tmp2, z3 = tmp0 + tmp2;
    int32_t z4 = tmp1 + tmp3;
    const int32_t z5 = (z3 + z4) * F1175;
    tmp0 *= F0298, tmp1 *= F2053, tmp2 *= F3072, tmp3 *= F1501;
    z1 *= -F0899, z2 *= -F2562, z3 *= -F1961, z4 *= -F0390;
    z3 += z5, z4 += z5;
    tmp0 += z1 + z3, tmp1 += z2 + z4, tmp2 += z2 + z3, tmp3 += z1 + z4;
    ws[0 * 8 + c] = descale(t10 + tmp3, CB - P1);
    ws[7 * 8 + c] = descale(t10 - tmp3, CB - P1);
    ws[1 * 8 + c] = descale(t11 + tmp2, CB - P1);
    ws[6 * 8 + c] = descale(t11 - tmp2, CB - P1);
    ws[2 * 8 + c] = descale(t12 + tmp1, CB - P1);
    ws[5 * 8 + c] = descale(t12 - tmp1, CB - P1);
    ws[3 * 8 + c] = descale(t13 + tmp0, CB - P1);
    ws[4 * 8 + c] = descale(t13 - tmp0, CB - P1);
  }
  for (int r = 0; r < 8; ++r) {
    const int32_t *w = ws + r * 8;
    int32_t z2 = w[2], z3 = w[6];
    int32_t z1 = (z2 + z3) * F0541;
    int32_t tmp2 = z1 + z3 * -F1847, tmp3 = z1 + z2 * F0765;
    int32_t tmp0 = (w[0] + w[4]) * (1 << CB), tmp1 = (w[0] - w[4]) * (1 << CB);
    const int32_t t10 = tmp0 + tmp3, t13 = tmp0 - tmp3, t11 = tmp1 + tmp2, t12 = tmp1 - tmp2;
    tmp0 = w[7], tmp1 = w[5], tmp2 = w[3], tmp3 = w[1];
    z1 = tmp0 + tmp3, z2 = tmp1 + tmp2, z3 = tmp0 + tmp2;
    int32_t z4 = tmp1 + tmp3;
    const int32_t z5 = (z3 + z4) * F1175;
    tmp0 *= F0298, tmp1 *= F2053, tmp2 *= F3072, tmp3 *= F1501;
    z1 *= -F0899, z2 *= -F2562, z3 *= -F1961, z4 *= -F0390;
    z3 += z5, z4 += z5;
    tmp0 += z1 + z3, tmp1 += z2 + z4, tmp2 += z2 + z3, tmp3 += z1 + z4;
    constexpr int S = CB + P1 + 3;
    uint8_t *o = out + static_cast<size_t>(r) * stride;
    o[0] = clamp8(descale(t10 + tmp3, S) + 128);
    o[7] = clamp8(descale(t10 - tmp3, S) + 128);
    o[1] = clamp8(descale(t11 + tmp2, S) + 128);
    o[6] = clamp8(descale(t11 - tmp2, S) + 128);
    o[2] = clamp8(descale(t12 + tmp1, S) + 128);
    o[5] = clamp8(descale(t12 - tmp1, S) + 128);
    o[3] = clamp8(descale(t13 + tmp0, S) + 128);
    o[4] = clamp8(descale(t13 - tmp0, S) + 128);
  }
}

// One component plane upsampled to the image's size (W x H) by (fh, fv) (the max factors over this one's).
void upsample(const Comp &c, int fh, int fv, int W, int H, std::vector<uint8_t> &out) {
  out.assign(static_cast<size_t>(W) * H, 0);
  const int stride = c.bw * 8, cw = c.w, ch = c.hgt;
  auto at = [&](int y, int x) -> int { return c.pix[static_cast<size_t>(std::clamp(y, 0, ch - 1)) * stride + std::clamp(x, 0, cw - 1)]; };
  const bool fancy = cw > 2;
  if (fh == 1 && fv == 1) {
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x) out[static_cast<size_t>(y) * W + x] = static_cast<uint8_t>(at(y, x));
    return;
  }
  // a row of the horizontally-upsampled plane at component row `y` (h2v1 fancy or replication), width 2 x cw
  std::vector<int> row(static_cast<size_t>(cw) * 2 + 2);
  if (fancy && fh == 2 && fv == 1) {
    for (int y = 0; y < H; ++y) {
      const int sy = y;
      int *o = row.data();
      o[0] = at(sy, 0);
      o[1] = (at(sy, 0) * 3 + at(sy, 1) + 2) >> 2;
      for (int x = 1; x < cw - 1; ++x) {
        const int v = at(sy, x) * 3;
        o[2 * x] = (v + at(sy, x - 1) + 1) >> 2;
        o[2 * x + 1] = (v + at(sy, x + 1) + 2) >> 2;
      }
      o[2 * cw - 2] = (at(sy, cw - 1) * 3 + at(sy, cw - 2) + 1) >> 2;
      o[2 * cw - 1] = at(sy, cw - 1);
      for (int x = 0; x < W; ++x) out[static_cast<size_t>(y) * W + x] = static_cast<uint8_t>(o[x]);
    }
    return;
  }
  if (fancy && fh == 2 && fv == 2) {
    for (int y = 0; y < H; ++y) {
      const int sy = y >> 1, other = (y & 1) ? sy + 1 : sy - 1;  // nearer row, further row
      auto col = [&](int x) { return at(sy, x) * 3 + at(other, x); };
      int *o = row.data();
      int last = 0, cur = col(0), next = col(1);
      o[0] = (cur * 4 + 8) >> 4;
      o[1] = (cur * 3 + next + 7) >> 4;
      last = cur, cur = next;
      for (int x = 1; x < cw - 1; ++x) {
        next = col(x + 1);
        o[2 * x] = (cur * 3 + last + 8) >> 4;
        o[2 * x + 1] = (cur * 3 + next + 7) >> 4;
        last = cur, cur = next;
      }
      o[2 * cw - 2] = (cur * 3 + last + 8) >> 4;
      o[2 * cw - 1] = (cur * 4 + 7) >> 4;
      for (int x = 0; x < W; ++x) out[static_cast<size_t>(y) * W + x] = static_cast<uint8_t>(o[x]);
    }
    return;
  }
  if (fh == 1 && fv == 2) {  // h1v2: the vertical triangle filter
    for (int y = 0; y < H; ++y) {
      const int sy = y >> 1, other = (y & 1) ? sy + 1 : sy - 1, bias = (y & 1) ? 2 : 1;
      for (int x = 0; x < W; ++x)
        out[static_cast<size_t>(y) * W + x] = static_cast<uint8_t>(fancy ? (at(sy, x) * 3 + at(other, x) + bias) >> 2 : at(sy, x));
    }
    return;
  }
  for (int y = 0; y < H; ++y)  // replication (any other ratio, or a plane too narrow for the filter)
    for (int x = 0; x < W; ++x) out[static_cast<size_t>(y) * W + x] = static_cast<uint8_t>(at(y / fv, x / fh));
}

uint16_t be16(const uint8_t *p) { return static_cast<uint16_t>(p[0] << 8 | p[1]); }

}  // namespace

bool decode_jpeg(const uint8_t *d, size_t n, RawImage &out, std::string &err) {
  uint16_t qt[4][64] = {};
  HuffTable dc[4], ac[4];
  std::vector<Comp> comps;
  int W = 0, H = 0, hmax = 1, vmax = 1, mcux = 0, mcuy = 0, restart = 0;
  bool progressive = false, frame = false;
  int adobe = -1;  // the Adobe marker's transform (-1: none)
  int orientation = 1;
  size_t p = 2;
  auto fail = [&](const char *m) { err = std::string("JPEG: ") + m; return false; };
  for (;;) {
    while (p < n && d[p] != 0xFF) ++p;  // tolerate garbage between markers
    while (p < n && d[p] == 0xFF) ++p;  // fill bytes
    if (p >= n) break;
    const uint8_t m = d[p++];
    if (m == 0xD9) break;  // EOI
    if (m >= 0xD0 && m <= 0xD7) continue;
    if (m == 0x01) continue;
    if (p + 2 > n) return fail("truncated marker");
    const size_t len = be16(d + p);
    if (len < 2 || p + len > n) return fail("truncated segment");
    const uint8_t *s = d + p + 2;
    const size_t sl = len - 2;
    p += len;
    if (m == 0xDB) {  // DQT
      size_t i = 0;
      while (i < sl) {
        const int pq = s[i] >> 4, tq = s[i] & 15;
        if (tq > 3) return fail("bad DQT");
        ++i;
        for (int k = 0; k < 64; ++k) {
          if (i + (pq ? 2 : 1) > sl) return fail("truncated DQT");
          qt[tq][kZigzag[k]] = pq ? be16(s + i) : s[i];
          i += pq ? 2 : 1;
        }
      }
    } else if (m == 0xC4) {  // DHT
      size_t i = 0;
      while (i + 17 <= sl) {
        const int tc = s[i] >> 4, th = s[i] & 15;
        if (tc > 1 || th > 3) return fail("bad DHT");
        int total = 0;
        for (int k = 0; k < 16; ++k) total += s[i + 1 + k];
        if (total > 256 || i + 17 + total > sl) return fail("bad DHT");
        if (!(tc ? ac[th] : dc[th]).build(s + i + 1, s + i + 17, total)) return fail("bad Huffman table");
        i += 17 + total;
      }
    } else if (m == 0xDD) {  // DRI
      if (sl < 2) return fail("bad DRI");
      restart = be16(s);
    } else if (m == 0xEE && sl >= 12 && !std::memcmp(s, "Adobe", 5)) {
      adobe = s[11];
    } else if (m == 0xE1 && sl >= 6 && !std::memcmp(s, "Exif\0\0", 6)) {
      if (orientation == 1) orientation = exif_orientation(s + 6, sl - 6);
    } else if (m == 0xC0 || m == 0xC1 || m == 0xC2) {
      if (frame) return fail("more than one frame");
      frame = true;
      progressive = m == 0xC2;
      if (sl < 6 || s[0] != 8) return fail("only 8-bit samples are supported");
      H = be16(s + 1), W = be16(s + 3);
      const int nc = s[5];
      if (W <= 0 || H <= 0) return fail("bad dimensions (DNL is not supported)");
      if (static_cast<int64_t>(W) * H > kImageMaxPixels) return err = "image too large (" + std::to_string(W) + " x " + std::to_string(H) + " pixels)", false;
      if ((nc != 1 && nc != 3 && nc != 4) || sl < 6 + static_cast<size_t>(nc) * 3) return fail("unsupported component count");
      comps.resize(static_cast<size_t>(nc));
      for (int c = 0; c < nc; ++c) {
        Comp &k = comps[static_cast<size_t>(c)];
        k.id = s[6 + 3 * c], k.h = s[7 + 3 * c] >> 4, k.v = s[7 + 3 * c] & 15, k.tq = s[8 + 3 * c];
        if (k.h < 1 || k.h > 4 || k.v < 1 || k.v > 4 || k.tq > 3) return fail("bad component");
        hmax = std::max(hmax, k.h), vmax = std::max(vmax, k.v);
      }
      mcux = (W + 8 * hmax - 1) / (8 * hmax), mcuy = (H + 8 * vmax - 1) / (8 * vmax);
      for (auto &k : comps) {
        k.bw = mcux * k.h, k.bh = mcuy * k.v;
        k.w = (W * k.h + hmax - 1) / hmax, k.hgt = (H * k.v + vmax - 1) / vmax;
        k.coef.assign(static_cast<size_t>(k.bw) * k.bh * 64, 0);
      }
    } else if (m == 0xDA) {  // SOS
      if (!frame) return fail("scan before frame");
      const int ns = s[0];
      if (ns < 1 || ns > 4 || sl < 4 + static_cast<size_t>(ns) * 2) return fail("bad SOS");
      std::vector<Comp *> sc;
      for (int i = 0; i < ns; ++i) {
        const int id = s[1 + 2 * i];
        Comp *k = nullptr;
        for (auto &c : comps)
          if (c.id == id) k = &c;
        if (!k) return fail("scan names an unknown component");
        k->td = s[2 + 2 * i] >> 4, k->ta = s[2 + 2 * i] & 15;
        if (k->td > 3 || k->ta > 3) return fail("bad SOS tables");
        sc.push_back(k);
      }
      const int Ss = s[1 + 2 * ns], Se = s[2 + 2 * ns], Ah = s[3 + 2 * ns] >> 4, Al = s[3 + 2 * ns] & 15;
      if (!progressive && (Ss != 0 || Se != 63)) return fail("bad baseline scan");
      if (progressive && (Se > 63 || Ss > Se || (Ss == 0 && Se != 0) || (Ss > 0 && ns != 1) || Al > 13)) return fail("bad progressive scan");
      for (auto *k : sc) {
        if ((!progressive || Ss == 0) && Ah == 0 && !dc[k->td].defined) return fail("missing DC table");
        if ((!progressive || Ss > 0) && !ac[k->ta].defined) return fail("missing AC table");
        k->dc_pred = 0;
      }
      BitReader br{d, n, p};
      int eobrun = 0;
      auto block_at = [](Comp &k, int bx, int by) { return k.coef.data() + (static_cast<size_t>(by) * k.bw + bx) * 64; };
      auto decode_block = [&](Comp &k, int16_t *b) {
        if (!progressive) {
          const int t = br.decode(dc[k.td]);
          k.dc_pred += br.extend(t);
          b[0] = static_cast<int16_t>(k.dc_pred);
          const HuffTable &A = ac[k.ta];
          for (int z = 1; z < 64;) {
            const int rs = br.decode(A), r = rs >> 4, sz = rs & 15;
            if (sz == 0) {
              if (r != 15) break;
              z += 16;
              continue;
            }
            z += r;
            b[kZigzag[std::min(z, 79)]] = static_cast<int16_t>(br.extend(sz));
            ++z;
          }
          return;
        }
        if (Ss == 0) {  // DC scans
          if (Ah == 0) {
            const int t = br.decode(dc[k.td]);
            k.dc_pred += br.extend(t);
            b[0] = static_cast<int16_t>(k.dc_pred * (1 << Al));
          } else if (br.get(1)) {
            b[0] = static_cast<int16_t>(b[0] | (1 << Al));
          }
          return;
        }
        const HuffTable &A = ac[k.ta];
        if (Ah == 0) {  // AC first
          if (eobrun > 0) {
            --eobrun;
            return;
          }
          for (int z = Ss; z <= Se;) {
            const int rs = br.decode(A), r = rs >> 4, sz = rs & 15;
            if (sz == 0) {
              if (r < 15) {
                eobrun = (1 << r) - 1;
                if (r) eobrun += static_cast<int>(br.get(r));
                break;
              }
              z += 16;
              continue;
            }
            z += r;
            b[kZigzag[std::min(z, 79)]] = static_cast<int16_t>(br.extend(sz) * (1 << Al));
            ++z;
          }
          return;
        }
        // AC refinement
        const int p1 = 1 << Al, m1 = -(1 << Al);
        int z = Ss;
        auto refine = [&](int16_t &c) {
          if (br.get(1) && (c & p1) == 0) c = static_cast<int16_t>(c >= 0 ? c + p1 : c + m1);
        };
        if (eobrun == 0) {
          for (; z <= Se; ++z) {
            const int rs = br.decode(A);
            int r = rs >> 4, sz = rs & 15, val = 0;
            if (sz) {
              val = br.get(1) ? p1 : m1;
            } else if (r != 15) {
              eobrun = 1 << r;
              if (r) eobrun += static_cast<int>(br.get(r));
              break;
            }
            do {
              int16_t &c = b[kZigzag[z]];
              if (c != 0) refine(c);
              else if (--r < 0) break;
              ++z;
            } while (z <= Se);
            if (val && z <= Se) b[kZigzag[z]] = static_cast<int16_t>(val);
          }
        }
        if (eobrun > 0) {
          for (; z <= Se; ++z) {
            int16_t &c = b[kZigzag[z]];
            if (c != 0) refine(c);
          }
          --eobrun;
        }
      };
      int todo = restart;
      auto restart_point = [&]() {
        if (!restart) return;
        if (--todo == 0) {
          br.restart();
          todo = restart;
          eobrun = 0;
          for (auto &c : comps) c.dc_pred = 0;
        }
      };
      if (ns == 1) {  // non-interleaved: the component's own blocks (ceil(w / 8) x ceil(h / 8)), one per "MCU"
        Comp &k = *sc[0];
        const int nbx = (k.w + 7) / 8, nby = (k.hgt + 7) / 8;
        for (int by = 0; by < nby; ++by)
          for (int bx = 0; bx < nbx; ++bx) {
            decode_block(k, block_at(k, bx, by));
            if (bx + 1 < nbx || by + 1 < nby) restart_point();
          }
      } else {
        for (int my = 0; my < mcuy; ++my)
          for (int mx = 0; mx < mcux; ++mx) {
            for (auto *k : sc)
              for (int v = 0; v < k->v; ++v)
                for (int h = 0; h < k->h; ++h) decode_block(*k, block_at(*k, mx * k->h + h, my * k->v + v));
            if (mx + 1 < mcux || my + 1 < mcuy) restart_point();
          }
      }
      // continue after the entropy-coded data: the next marker that is not a restart
      p = br.pos;
      while (p + 1 < n && !(d[p] == 0xFF && d[p + 1] != 0 && !(d[p + 1] >= 0xD0 && d[p + 1] <= 0xD7))) ++p;
    } else if (m >= 0xC3 && m <= 0xCF && m != 0xC4 && m != 0xC8 && m != 0xCC) {
      return fail("unsupported coding (lossless, hierarchical or arithmetic)");
    }
  }
  if (!frame) return fail("no frame");
  for (auto &k : comps) {
    k.pix.assign(static_cast<size_t>(k.bw) * 8 * k.bh * 8, 0);
    for (int by = 0; by < k.bh; ++by)
      for (int bx = 0; bx < k.bw; ++bx)
        idct_islow(k.coef.data() + (static_cast<size_t>(by) * k.bw + bx) * 64, qt[k.tq], k.pix.data() + static_cast<size_t>(by) * 8 * k.bw * 8 + bx * 8, k.bw * 8);
  }
  const int nc = static_cast<int>(comps.size());
  std::vector<std::vector<uint8_t>> pl(static_cast<size_t>(nc));
  for (int c = 0; c < nc; ++c) upsample(comps[static_cast<size_t>(c)], hmax / comps[static_cast<size_t>(c)].h, vmax / comps[static_cast<size_t>(c)].v, W, H, pl[static_cast<size_t>(c)]);
  out.w = W, out.h = H, out.orientation = orientation;
  const size_t px = static_cast<size_t>(W) * H;
  if (nc == 1) {
    out.channels = 1;
    out.px = std::move(pl[0]);
    return true;
  }
  // YCbCr -> RGB tables (16-bit fixed point)
  static int cr_r[256], cb_b[256], cr_g[256], cb_g[256];
  static bool tables = [] {
    auto FIX = [](double x) { return static_cast<int>(x * 65536 + 0.5); };
    for (int i = 0; i < 256; ++i) {
      const int x = i - 128;
      cr_r[i] = (FIX(1.40200) * x + 32768) >> 16;
      cb_b[i] = (FIX(1.77200) * x + 32768) >> 16;
      cr_g[i] = -FIX(0.71414) * x;
      cb_g[i] = -FIX(0.34414) * x + 32768;
    }
    return true;
  }();
  (void)tables;
  // RGB by the Adobe marker (transform 0) or the component ids 'R' 'G' 'B'; otherwise YCbCr (YCCK for 4 with transform 2)
  const bool rgb = nc == 3 && (adobe == 0 || (comps[0].id == 'R' && comps[1].id == 'G' && comps[2].id == 'B'));
  const bool ycc = nc == 3 ? !rgb : adobe == 2;
  out.channels = 3;
  out.px.resize(px * 3);
  for (size_t i = 0; i < px; ++i) {
    int r, g, b;
    if (ycc) {
      const int y = pl[0][i], cb = pl[1][i], cr = pl[2][i];
      r = clamp8(y + cr_r[cr]), g = clamp8(y + ((cb_g[cb] + cr_g[cr]) >> 16)), b = clamp8(y + cb_b[cb]);
    } else {
      r = pl[0][i], g = pl[1][i], b = pl[2][i];
    }
    if (nc == 4) {
      // CMYK (YCCK converted to CMY: 255 - RGB): Adobe files store inverted values; they are read as
      // C = 255 - stored, then RGB = (255 - K') - C x (255 - K') / 255 with K' = 255 - stored K (DIV255 rounding)
      int c, mm, yy;
      if (ycc) c = 255 - r, mm = 255 - g, yy = 255 - b;
      else c = r, mm = g, yy = b;
      int k = pl[3][i];
      if (adobe >= 0) c = 255 - c, mm = 255 - mm, yy = 255 - yy, k = 255 - k;
      const int nk = 255 - k;
      auto md = [](int a, int bb) { const int t = a * bb + 128; return ((t >> 8) + t) >> 8; };
      r = clamp8(nk - md(c, nk)), g = clamp8(nk - md(mm, nk)), b = clamp8(nk - md(yy, nk));
    }
    out.px[3 * i] = static_cast<uint8_t>(r), out.px[3 * i + 1] = static_cast<uint8_t>(g), out.px[3 * i + 2] = static_cast<uint8_t>(b);
  }
  return true;
}

}  // namespace xenithium:util
