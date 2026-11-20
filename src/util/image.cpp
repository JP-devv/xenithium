// Xenithium · image input: base64 / data: URLs, the EXIF orientation, the conversion to RGB
// (xenithium/util/image.hpp). The format decoders are in image_png.cpp and image_jpeg.cpp.
#include "xenithium/util/image.hpp"

#include <cstring>

namespace xenithium::util {

namespace {

int b64_value(unsigned char c) {
  if (c >= 'A' && c <= 'Z') return c - 'A';
  if (c >= 'a' && c <= 'z') return c - 'a' + 26;
  if (c >= '0' && c <= '9') return c - '0' + 52;
  if (c == '+') return 62;
  if (c == '/') return 63;
  return -1;
}

// The DIV255 rounding of the alpha blend: (a + 128 + ((a + 128) >> 8)) >> 8.
inline uint8_t div255(int a) {
  const int t = a + 128;
  return static_cast<uint8_t>(((t >> 8) + t) >> 8);
}

// Applies the EXIF orientation: 2 mirror, 3 rotate 180, 4 flip, 5 transpose,
// 6 rotate 90 clockwise, 7 transverse, 8 rotate 90 counter-clockwise.
void orient(Image &im, int o) {
  if (o < 2 || o > 8) return;
  const int w = im.w, h = im.h;
  const bool swap = o >= 5;
  const int W = swap ? h : w, H = swap ? w : h;
  std::vector<uint8_t> out(static_cast<size_t>(W) * H * 3);
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x) {
      int sx = 0, sy = 0;  // the source pixel of output (x, y)
      switch (o) {
        case 2: sx = w - 1 - x, sy = y; break;
        case 3: sx = w - 1 - x, sy = h - 1 - y; break;
        case 4: sx = x, sy = h - 1 - y; break;
        case 5: sx = y, sy = x; break;  // transpose
        case 6: sx = y, sy = h - 1 - x; break;  // rotate 90 clockwise
        case 7: sx = w - 1 - y, sy = h - 1 - x; break;  // transverse
        case 8: sx = w - 1 - y, sy = x; break;  // rotate 90 counter-clockwise
      }
      std::memcpy(&out[(static_cast<size_t>(y) * W + x) * 3], &im.rgb[(static_cast<size_t>(sy) * w + sx) * 3], 3);
    }
  im.rgb.swap(out);
  im.w = W, im.h = H;
}

uint16_t rd16(const uint8_t *p, bool be) { return be ? static_cast<uint16_t>(p[0] << 8 | p[1]) : static_cast<uint16_t>(p[1] << 8 | p[0]); }
uint32_t rd32(const uint8_t *p, bool be) {
  return be ? (uint32_t(p[0]) << 24 | uint32_t(p[1]) << 16 | uint32_t(p[2]) << 8 | p[3]) : (uint32_t(p[3]) << 24 | uint32_t(p[2]) << 16 | uint32_t(p[1]) << 8 | p[0]);
}

}  // namespace

bool base64_decode(std::string_view in, std::string &out) {
  out.clear();
  if (in.size() % 4 != 0) return false;
  out.reserve(in.size() / 4 * 3);
  for (size_t i = 0; i < in.size(); i += 4) {
    int v[4];
    int pad = 0;
    for (int k = 0; k < 4; ++k) {
      const auto c = static_cast<unsigned char>(in[i + k]);
      if (c == '=') {
        if (i + 4 != in.size() || k < 2) return false;  // padding only in the last quantum's last two places
        v[k] = 0, ++pad;
        continue;
      }
      if (pad) return false;  // data after '='
      v[k] = b64_value(c);
      if (v[k] < 0) return false;
    }
    const uint32_t q = static_cast<uint32_t>(v[0]) << 18 | static_cast<uint32_t>(v[1]) << 12 | static_cast<uint32_t>(v[2]) << 6 | static_cast<uint32_t>(v[3]);
    out.push_back(static_cast<char>(q >> 16));
    if (pad < 2) out.push_back(static_cast<char>((q >> 8) & 255));
    if (pad < 1) out.push_back(static_cast<char>(q & 255));
  }
  return true;
}

std::string base64_encode(std::string_view in) {
  static const char *a = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string out;
  out.reserve((in.size() + 2) / 3 * 4);
  for (size_t i = 0; i < in.size(); i += 3) {
    const size_t n = in.size() - i < 3 ? in.size() - i : 3;
    uint32_t q = static_cast<uint32_t>(static_cast<unsigned char>(in[i])) << 16;
    if (n > 1) q |= static_cast<uint32_t>(static_cast<unsigned char>(in[i + 1])) << 8;
    if (n > 2) q |= static_cast<unsigned char>(in[i + 2]);
    out += a[q >> 18 & 63];
    out += a[q >> 12 & 63];
    out += n > 1 ? a[q >> 6 & 63] : '=';
    out += n > 2 ? a[q & 63] : '=';
  }
  return out;
}

bool parse_data_url(std::string_view url, std::string &bytes, std::string &err) {
  if (url.substr(0, 5) != "data:") {
    err = "only data: URLs (base64) are accepted for images by this server";
    return false;
  }
  const size_t comma = url.find(',');
  if (comma == std::string_view::npos) {
    err = "malformed data URL: no ','";
    return false;
  }
  const std::string_view meta = url.substr(5, comma - 5);
  if (meta.size() < 7 || meta.substr(meta.size() - 7) != ";base64") {
    err = "data URL images must be base64-encoded (';base64,')";
    return false;
  }
  if (!base64_decode(url.substr(comma + 1), bytes)) {
    err = "invalid base64 in the image data URL";
    return false;
  }
  return true;
}

int exif_orientation(const uint8_t *t, size_t n) {
  if (n < 8) return 1;
  bool be;
  if (t[0] == 'M' && t[1] == 'M') be = true;
  else if (t[0] == 'I' && t[1] == 'I') be = false;
  else return 1;
  if (rd16(t + 2, be) != 42) return 1;
  const uint32_t ifd = rd32(t + 4, be);
  if (ifd + 2 > n) return 1;
  const unsigned count = rd16(t + ifd, be);
  for (unsigned i = 0; i < count; ++i) {
    const size_t e = ifd + 2 + static_cast<size_t>(i) * 12;
    if (e + 12 > n) return 1;
    if (rd16(t + e, be) == 0x0112 && rd16(t + e + 2, be) == 3) {  // Orientation, SHORT
      const int o = rd16(t + e + 8, be);
      return o >= 1 && o <= 8 ? o : 1;
    }
  }
  return 1;
}

bool decode_image(const uint8_t *data, size_t n, Image &out, std::string &err) {
  RawImage raw;
  static const uint8_t png_sig[8] = {0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n'};
  if (n >= 8 && std::memcmp(data, png_sig, 8) == 0) {
    if (!decode_png(data, n, raw, err)) return false;
  } else if (n >= 3 && data[0] == 0xFF && data[1] == 0xD8 && data[2] == 0xFF) {
    if (!decode_jpeg(data, n, raw, err)) return false;
  } else {
    err = "unsupported image format (PNG and JPEG are accepted)";
    return false;
  }
  out.w = raw.w, out.h = raw.h;
  const size_t px = static_cast<size_t>(raw.w) * raw.h;
  out.rgb.resize(px * 3);
  const uint8_t *s = raw.px.data();
  uint8_t *d = out.rgb.data();
  switch (raw.channels) {
    case 1:
      for (size_t i = 0; i < px; ++i) {
        const uint8_t v = s[i];
        if (raw.key && v == raw.key_rgb[0]) d[3 * i] = d[3 * i + 1] = d[3 * i + 2] = 255;  // alpha 0 over white
        else d[3 * i] = d[3 * i + 1] = d[3 * i + 2] = v;
      }
      break;
    case 2:
      for (size_t i = 0; i < px; ++i) {
        const int a = s[2 * i + 1];
        d[3 * i] = d[3 * i + 1] = d[3 * i + 2] = div255(255 * (255 - a) + s[2 * i] * a);
      }
      break;
    case 3:
      for (size_t i = 0; i < px; ++i) {
        const uint8_t *p = s + 3 * i;
        if (raw.key && p[0] == raw.key_rgb[0] && p[1] == raw.key_rgb[1] && p[2] == raw.key_rgb[2]) d[3 * i] = d[3 * i + 1] = d[3 * i + 2] = 255;
        else d[3 * i] = p[0], d[3 * i + 1] = p[1], d[3 * i + 2] = p[2];
      }
      break;
    case 4:
      for (size_t i = 0; i < px; ++i) {
        const uint8_t *p = s + 4 * i;
        const int a = p[3];
        for (int c = 0; c < 3; ++c) d[3 * i + c] = div255(255 * (255 - a) + p[c] * a);
      }
      break;
    default:
      err = "image decoder produced no pixels";
      return false;
  }
  orient(out, raw.orientation);
  return true;
}

}  // namespace xenithium:util
