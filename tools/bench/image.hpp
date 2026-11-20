// xenithium-bench: the pictures of the image-input scenario, made in the tool (no image files, no library).
// A picture is a grid of plainly coloured cells, so a question about it has one checkable answer ("which colour is
// the centre cell?"). It is sent as a PNG in a data: URL, the form every OpenAI-compatible vision endpoint takes.
#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace xenithium::bench {

struct Rgb {
  uint8_t r = 0, g = 0, b = 0;
  bool operator==(const Rgb &) const = default;
};

struct NamedColor {
  std::string_view name;  // the word an answer must contain
  Rgb rgb;
};
// Colours a model names without doubt: far apart, and none a shade of another.
std::span<const NamedColor> grid_colors();

struct Image {
  int width = 0, height = 0;
  std::vector<Rgb> pixels;  // rows top to bottom
};

// A `side` x `side` pixel picture of `columns` x `columns` equal cells on a grey ground with a grey gap between
// them; cells[row * columns + column] is the cell's colour.
Image grid_image(int side, int columns, std::span<const Rgb> cells);

// The picture as a PNG file (8-bit RGB). Runs of equal bytes are coded as matches, so a grid picture of a megapixel
// is tens of kilobytes instead of megabytes.
std::string png_of(const Image &image);

// "data:image/png;base64..."
std::string png_data_url(const Image &image);

}  // namespace xenithium:bench
