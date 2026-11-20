#include "image.hpp"

#include <algorithm>
#include <array>

#include "xenithium/util/image.hpp"

namespace xenithium::bench {
namespace {

constexpr NamedColor kGridColors[] = {
    {"red", {220, 30, 30}}, {"green", {30, 160, 60}}, {"blue", {40, 80, 220}}, {"yellow", {245, 215, 40}}, {"black", {15, 15, 15}}, {"white", {250, 250, 250}},
};
constexpr Rgb kGround{128, 128, 128};
constexpr int kGapShare = 16;  // the gap between cells is 1/16 of a cell's pitch

// ---- zlib / deflate, write side only (RFC 1950, RFC 1951) ----
// One block with the fixed Huffman code. The only matches coded are runs: a byte equal to the one before it, as
// (length, distance 1). After the PNG "sub" filter a flat-coloured row is almost all zeros, so this is enough.

constexpr int kMinMatch = 3, kMaxMatch = 258;
constexpr uint32_t kAdlerModulus = 65521;

class BitWriter {
 public:
  explicit BitWriter(std::string &out) : out_(out) {}
  // `count` bits of `value, least significant first (how deflate packs everything but Huffman codes).
  void bits(uint32_t value, int count) {
    for (int i = 0; i < count; ++i) bit((value >> i) & 1);
  }
  // A Huffman code of `length` bits, most significant first.
  void code(uint32_t value, int length) {
    for (int i = length - 1; i >= 0; --i) bit((value >> i) & 1);
  }
  void flush() {
    if (used_ > 0) out_.push_back(static_cast<char>(byte_));
    byte_ = 0, used_ = 0;
  }

 private:
  void bit(uint32_t b) {
    byte_ |= b << used_;
    if (++used_ == 8) flush();
  }
  std::string &out_;
  uint32_t byte_ = 0;
  int used_ = 0;
};

// The fixed code's literal / length symbols (RFC 1951, 3.2.6).
void put_symbol(BitWriter &w, int symbol) {
  if (symbol <= 143) w.code(0b00110000u + static_cast<uint32_t>(symbol), 8);
  else if (symbol <= 255) w.code(0b110010000u + static_cast<uint32_t>(symbol - 144), 9);
  else if (symbol <= 279) w.code(static_cast<uint32_t>(symbol - 256), 7);
  else w.code(0b11000000u + static_cast<uint32_t>(symbol - 280), 8);
}

// A match of `length` bytes at distance 1: the length's symbol and extra bits (RFC 1951, 3.2.5), then distance code 0.
void put_run(BitWriter &w, int length) {
  constexpr int kEndOfLengths = 285, kDistanceCodeBits = 5;
  if (length == kMaxMatch) {
    put_symbol(w, kEndOfLengths);
  } else {
    // symbols 257..264 code lengths 3..10 without extra bits; from 265 on each group of four doubles its span
    int symbol = 257, base = kMinMatch, extra_bits = 0;
    while (true) {
      const int span = 1 << extra_bits;
      const int group = extra_bits == 0 ? 8 : 4;
      if (length < base + span * group) {
        symbol += (length - base) / span;
        put_symbol(w, symbol);
        w.bits(static_cast<uint32_t>((length - base) % span), extra_bits);
        break;
      }
      symbol += group, base += span * group, ++extra_bits;
    }
  }
  w.code(0, kDistanceCodeBits);
}

uint32_t adler32(std::string_view data) {
  uint32_t a = 1, b = 0;
  for (const char c : data) {
    a = (a + static_cast<uint8_t>(c)) % kAdlerModulus;
    b = (b + a) % kAdlerModulus;
  }
  return (b << 16) | a;
}

void put_be32(std::string &out, uint32_t value) {
  for (int shift = 24; shift >= 0; shift -= 8) out.push_back(static_cast<char>((value >> shift) & 0xFF));
}

std::string zlib_of(std::string_view data) {
  constexpr char kZlibHeader[] = {0x78, 0x01};  // deflate, 32K window, no preset dictionary, fastest
  constexpr int kEndOfBlock = 256;
  std::string out(kZlibHeader, sizeof(kZlibHeader));
  BitWriter w(out);
  w.bits(1, 1);  // the last block
  w.bits(0b01, 2);  // fixed Huffman code
  for (size_t at = 0; at < data.size();) {
    size_t run = 0;  // bytes from `at` equal to the byte before `at`
    while (at > 0 && at + run < data.size() && data[at + run] == data[at - 1] && run < static_cast<size_t>(kMaxMatch)) ++run;
    if (run >= static_cast<size_t>(kMinMatch)) {
      put_run(w, static_cast<int>(run));
      at += run;
    } else {
      put_symbol(w, static_cast<uint8_t>(data[at]));
      ++at;
    }
  }
  put_symbol(w, kEndOfBlock);
  w.flush();
  put_be32(out, adler32(data));
  return out;
}

// ---- PNG (ISO/IEC 15948) ----

std::array<uint32_t, 256> crc_table() {
  std::array<uint32_t, 256> table{};
  for (uint32_t n = 0; n < 256; ++n) {
    uint32_t c = n;
    for (int k = 0; k < 8; ++k) c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
    table[n] = c;
  }
  return table;
}

uint32_t crc32(std::string_view data) {
  static const std::array<uint32_t, 256> table = crc_table();
  uint32_t c = 0xFFFFFFFFu;
  for (const char ch : data) c = table[(c ^ static_cast<uint8_t>(ch)) & 0xFF] ^ (c >> 8);
  return c ^ 0xFFFFFFFFu;
}

void put_chunk(std::string &png, std::string_view type, std::string_view data) {
  put_be32(png, static_cast<uint32_t>(data.size()));
  const size_t checked_from = png.size();
  png.append(type).append(data);
  put_be32(png, crc32(std::string_view(png).substr(checked_from)));
}

// Every row as PNG stores it: a filter byte, then the pixels with the "sub" filter (each byte minus the one a pixel
// to its left), which turns a flat run of colour into zeros.
std::string filtered_rows(const Image &image) {
  constexpr char kFilterSub = 1;
  std::string rows;
  rows.reserve(static_cast<size_t>(image.height) * (1 + 3 * static_cast<size_t>(image.width)));
  for (int y = 0; y < image.height; ++y) {
    rows.push_back(kFilterSub);
    Rgb left{};
    for (int x = 0; x < image.width; ++x) {
      const Rgb p = image.pixels[static_cast<size_t>(y) * static_cast<size_t>(image.width) + static_cast<size_t>(x)];
      rows.push_back(static_cast<char>(p.r - left.r));
      rows.push_back(static_cast<char>(p.g - left.g));
      rows.push_back(static_cast<char>(p.b - left.b));
      left = p;
    }
  }
  return rows;
}

}  // namespace

std::span<const NamedColor> grid_colors() { return kGridColors; }

Image grid_image(int side, int columns, std::span<const Rgb> cells) {
  Image image{side, side, std::vector<Rgb>(static_cast<size_t>(side) * static_cast<size_t>(side), kGround)};
  const int pitch = side / columns, gap = std::max(1, pitch / kGapShare);
  for (int row = 0; row < columns; ++row)
    for (int column = 0; column < columns; ++column) {
      const Rgb colour = cells[static_cast<size_t>(row * columns + column)];
      for (int y = row * pitch + gap; y < (row + 1) * pitch - gap; ++y)
        std::fill_n(image.pixels.begin() + static_cast<std::ptrdiff_t>(y) * side + column * pitch + gap, pitch - 2 * gap, colour);
    }
  return image;
}

std::string png_of(const Image &image) {
  constexpr std::string_view kSignature = "\x89PNG\r\n\x1a\n";
  constexpr char kBitDepth = 8, kColorTypeRgb = 2;
  std::string header;
  put_be32(header, static_cast<uint32_t>(image.width));
  put_be32(header, static_cast<uint32_t>(image.height));
  header.append({kBitDepth, kColorTypeRgb, 0, 0, 0});  // deflate, the standard filters, not interlaced
  std::string png(kSignature);
  put_chunk(png, "IHDR", header);
  put_chunk(png, "IDAT", zlib_of(filtered_rows(image)));
  put_chunk(png, "IEND", {});
  return png;
}

std::string png_data_url(const Image &image) { return "data:image/png;base64," + util::base64_encode(png_of(image)); }

}  // namespace xenithium:bench
