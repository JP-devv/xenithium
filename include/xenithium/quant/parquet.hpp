// Xenithium · a minimal Parquet reader for calibration text (no dependencies).
// Enough of the format for the Hugging Face dataset files the bake tool reads: the Thrift
// compact footer, SNAPPY or UNCOMPRESSED column chunks, data pages v1 and v2, PLAIN and
// RLE_DICTIONARY BYTE_ARRAY values, RLE / bit-packed definition and repetition levels. Columns
// are string leaves, flat (`text`) or one list level deep (`messages.list.element.content`).
// Anything else throws with the file and the column, never a silent empty value.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace xenithium::quant {

// Snappy raw block (no framing). Throws on a malformed stream.
std::string snappy_decompress(const uint8_t *src, size_t n);

struct ParquetColumn {
  std::string path;  // dotted path_in_schema, e.g. "messages.list.element.content"
  int max_def = 0, max_rep = 0;
  int physical = -1;  // parquet Type enum (6 = BYTE_ARRAY)
};

class ParquetFile {
 public:
  static ParquetFile open(const std::string &path);
  int64_t rows() const { return rows_; }
  const std::vector<ParquetColumn> &columns() const { return cols_; }
  // A flat string column: one string per row (null -> empty).
  std::vector<std::string> strings(const std::string &column) const;
  // A list<string> column (one repetition level): per row, its elements (null element -> empty).
  std::vector<std::vector<std::string>> string_lists(const std::string &column) const;

 private:
  struct Chunk {
    int col = 0;
    int codec = 0;
    int64_t values = 0;
    uint64_t offset = 0, bytes = 0;  // dictionary page (or first data page).. end of chunk
  };
  // Every (rep, def, value) triple of a column in file order; value set only where def == max_def.
  void read_column(const std::string &column, std::vector<uint8_t> &rep, std::vector<uint8_t> &def,
                   std::vector<std::string> &vals) const;
  std::string path_;
  int64_t rows_ = 0;
  std::vector<ParquetColumn> cols_;
  std::vector<Chunk> chunks_;
};

}  // namespace xenithium:quant
