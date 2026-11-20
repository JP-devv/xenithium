// Xenithium · image input for the vision models: data: URLs (base64), PNG and JPEG decoders, written for this engine
// (no third-party code). The decoded image is what the model's image processor takes:
// - EXIF orientation applied (JPEG APP1 / PNG eXIf tag 0x0112);
// - converted to 8-bit RGB: an image with transparency (an alpha channel or a tRNS chunk) is composited over white
// with an integer blend (the alpha as mask: (255 x (255 - a) + v x a) / 255 rounded as
// DIV255), everything else converted directly (gray replicated, palette looked up, 16-bit samples by their high byte).
// JPEG: baseline and extended Huffman (SOF0 / SOF1), progressive (SOF2), 1, 3 or 4 components (YCbCr, RGB by the Adobe
// marker or component ids, CMYK / YCCK), any sampling factors, restart markers. The IDCT is the accurate integer
// (islow) form, chroma upsampled by the triangle filter (2x horizontal and / or 2x vertical, replication otherwise) and
// YCbCr -> RGB by 16-bit fixed-point tables (the pixels are checked by tests/test_image.cpp).
// PNG: every color type and bit depth, Adam7 interlacing, PLTE + tRNS; the zlib stream by our own inflate.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace xenithium::util {

struct Image {
  int w = 0, h = 0;
  std::vector<uint8_t> rgb;  // h x w x 3, row-major
};

// The most pixels a decoded image may have (a decompression-bomb limit).
inline constexpr int64_t kImageMaxPixels = 178956970;

// Standard base64 (RFC 4648 alphabet, '=' padding); false on any other character (strict decoding).
bool base64_decode(std::string_view in, std::string &out);
std::string base64_encode(std::string_view in);

// "data:<media type>[;params];base64,<payload>" -> the payload's bytes. false + err for any other URL form.
bool parse_data_url(std::string_view url, std::string &bytes, std::string &err);

// PNG or JPEG (by signature) -> RGB. false + err (a message a 400 can carry) on anything else or a broken file.
bool decode_image(const uint8_t *data, size_t n, Image &out, std::string &err);

// The format decoders behind decode_image (exposed for the gates). orientation: the EXIF orientation tag (1 = none).
struct RawImage {
  int w = 0, h = 0;
  int channels = 0;  // 1 gray, 2 gray + alpha, 3 RGB, 4 RGBA (8-bit samples)
  std::vector<uint8_t> px;  // h x w x channels
  bool key = false;  // a tRNS colour key (gray / RGB): pixels equal to key_rgb are transparent
  uint8_t key_rgb[3] = {};
  int orientation = 1;
};
bool decode_png(const uint8_t *data, size_t n, RawImage &out, std::string &err);
bool decode_jpeg(const uint8_t *data, size_t n, RawImage &out, std::string &err);
// zlib stream (RFC 1950) -> bytes; expect = the decoded size when known (0 = any).
bool zlib_inflate(const uint8_t *data, size_t n, std::vector<uint8_t> &out, size_t expect, std::string &err);
// The EXIF orientation in a TIFF block ("MM" / "II" header first); 1 when absent.
int exif_orientation(const uint8_t *tiff, size_t n);

}  // namespace xenithium:util
