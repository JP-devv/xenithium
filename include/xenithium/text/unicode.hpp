// Xenithium · Unicode pieces the tokenizer needs, as the tokenizer.json pipeline defines them rather
// than as one Unicode version does: the split regex's \p{L}, \p{N} and \s (the regex engine's data is
// newer than the normaliser's), the case-insensitive letters of the contraction alternative, and
// NFC with the normaliser's own (older) decomposition, combining-class and composition data.
// Tables: unicode_tables.inc (generated).
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

namespace xenithium::text::unicode {

bool is_letter(char32_t c);  // \p{L}
bool is_number(char32_t c);  // \p{N}
bool is_space(char32_t c);  // \s
bool is_mark(char32_t c);  // \p{M}
// The cased-letter split of the Flint pre-tokenizer (unicode_tables_cat.inc):
// [\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}], [\p{Ll}\p{Lm}\p{Lo}\p{M}], [\p{P}\p{S}].
bool is_case_lead(char32_t c);
bool is_case_tail(char32_t c);
bool is_punct_symbol(char32_t c);
// (?i:x) for an ASCII lowercase letter x of the contraction alternative ('s 't 're 've 'm 'll 'd):
// true when c matches it case-insensitively (e.g. U+017F matches 's').
bool folds_to(char32_t c, char lower);

// Canonical combining class (0 for starters).
uint8_t combining_class(char32_t c);
// NFC of `in, appended to `out.
void nfc(std::u32string_view in, std::u32string &out);

// UTF-8 -> code points; an invalid byte becomes U+FFFD (the server only passes valid UTF-8).
void decode_utf8(std::string_view s, std::u32string &out);
void append_utf8(char32_t c, std::string &out);

}  // namespace xenithium:text:unicode
