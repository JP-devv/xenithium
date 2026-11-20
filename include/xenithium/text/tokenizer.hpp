// BPE tokenizer read from a checkpoint's tokenizer.json, plus the streaming helpers the server
// needs (incremental detokenization, stop strings).
// Two pipelines, as tokenizer.json declares them:
// - byte-level (the Qwen family): added tokens matched first on the raw text (never normalised),
// NFC on the rest, the split regex (hand-coded: load refuses a pattern it does not implement),
// the GPT-2 byte-level mapping, then rank-ordered BPE merges;
// - SentencePiece-style (Llama-family checkpoints such as Nanbeige4.3: Metaspace pre-tokenizer, BPE
// with byte fallback): added tokens first, ' ' -> U+2581 and a leading U+2581 on the text's first
// piece, BPE over characters (a character outside the vocab = its bytes' <0xXX> tokens).
// The gate is id equality on a golden corpus per pipeline
// (tests/test_tokenizer.cpp; tests/data/tokenizer_golden.json, tokenizer_golden_sp.json).
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace xenithium::text {

// UTF-8 at s[i]: complete (len = its byte count), incomplete (a valid prefix cut by the end
// of s; len = bytes present) or invalid (len = 1).
enum class Utf8 : uint8_t { complete, incomplete, invalid };
Utf8 utf8_at(std::string_view s, size_t i, size_t &len);

class Tokenizer {
 public:
  Tokenizer();
  ~Tokenizer();
  Tokenizer(const Tokenizer &) = delete;
  Tokenizer &operator=(const Tokenizer &) = delete;

  // Reads `<model_dir>/tokenizer.json. false + err on anything it does not implement.
  bool load(const std::string &model_dir, std::string &err);

  // Text -> ids, without adding special tokens (no BOS: the
  // template carries the markers). Special and added tokens in the text are parsed as tokens.
  // Thread-safe.
  std::vector<int32_t> encode(std::string_view text) const;

  // Ids the tokenizer knows: max id + 1 (added tokens included). The model's vocabulary
  // (lm_head rows) may be larger; ids in between decode to nothing.
  int32_t vocab_size() const;
  // Appends the token's raw bytes (byte-level mapping undone; may be a partial UTF-8
  // sequence). skip_special: special tokens append nothing. Unknown ids append nothing.
  void decode_piece(int32_t id, std::string &out, bool skip_special) const;
  // Whole sequence, invalid UTF-8 replaced by U+FFFD (errors="replace").
  std::string decode(const std::vector<int32_t> &ids, bool skip_special) const;
  // The id of a token's text (a vocabulary entry or an added token), -1 when absent.
  int32_t token_id(std::string_view token) const;
  bool is_special(int32_t id) const;
  // SentencePiece-style: true; then byte_value(id) is the byte of a <0xXX> token (-1 for any other id) and the
  // decode drops one leading ' ' of the whole text when strips_leading_space.
  bool byte_fallback() const;
  int byte_value(int32_t id) const;
  bool strips_leading_space() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// Incremental detokenization for streaming: bytes are released only as complete UTF-8;
// an invalid byte becomes U+FFFD as soon as it is known to be invalid.
class Detokenizer {
 public:
  Detokenizer(const Tokenizer &tok, bool skip_special) : tok_(&tok), skip_special_(skip_special) {}
  void push(int32_t id);
  // Appends the text that is final now (keeps an incomplete trailing sequence back).
  void take(std::string &out);
  // Appends everything left (an incomplete tail becomes U+FFFD).
  void flush(std::string &out);

 private:
  const Tokenizer *tok_;
  bool skip_special_;
  std::string pending_;
  // SentencePiece-style decoding (byte_fallback): the open run of byte tokens, and whether text was released yet
  std::string run_;
  bool started_ = false;
  void end_run();
  void emit(std::string &out);
};

// Stop strings over streamed text: text that could still be the start of a stop string is
// held back; at a match the text before it is released and the match (and everything after)
// dropped (stop strings are not included in the output).
class StopMatcher {
 public:
  StopMatcher() = default;
  explicit StopMatcher(std::vector<std::string> stops);
  // Appends the releasable text; true once a stop string matched (then stays true and
  // appends nothing more).
  bool feed(std::string_view piece, std::string &out);
  void flush(std::string &out);  // the held-back tail (no match came)
  bool stopped() const { return stopped_; }

 private:
  std::vector<std::string> stops_;
  std::string held_;
  bool stopped_ = false;
  size_t max_len_ = 0;
};

}  // namespace xenithium:text
