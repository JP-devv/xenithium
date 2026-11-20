// Xenithium · Unicode tables and NFC for the tokenizer.
#include "xenithium/text/unicode.hpp"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <utility>
#include <vector>

namespace xenithium::text::unicode {
namespace {

struct CpRange {
  uint32_t lo, hi;
};
struct CccRange {
  uint32_t lo, hi;
  uint8_t ccc;
};
struct Decomp {
  uint32_t cp;
  uint16_t offset, length;
};
struct Compose {
  uint32_t a, b, c;
};
struct FoldEntry {
  char lower;
  uint32_t cp;
};

#include "xenithium/text/unicode_tables.inc"
#include "xenithium/text/unicode_tables_cat.inc"

static_assert(sizeof(char32_t) == sizeof(uint32_t));

bool in_ranges(const CpRange *r, size_t n, uint32_t c) {
  const CpRange *end = r + n;
  const CpRange *it = std::upper_bound(r, end, c, [](uint32_t v, const CpRange &x) { return v < x.lo; });
  return it != r && c <= (it - 1)->hi;
}

// Class bits for the low planes in one flat table (the hot path of pre-tokenisation); above it
// a binary search over the ranges.
constexpr uint32_t kFlatLimit = 0x40000;
enum : uint8_t { kL = 1, kN = 2, kS = 4, kM = 8, kCL = 16, kCT = 32, kPS = 64 };

const std::vector<uint8_t> &flat() {
  static const std::vector<uint8_t> t = [] {
    std::vector<uint8_t> v(kFlatLimit, 0);
    auto fill = [&](const CpRange *r, size_t n, uint8_t bit) {
      for (size_t i = 0; i < n; ++i)
        for (uint32_t c = r[i].lo; c <= r[i].hi && c < kFlatLimit; ++c) v[c] |= bit;
    };
    fill(kLetter, std::size(kLetter), kL);
    fill(kNumber, std::size(kNumber), kN);
    fill(kSpace, std::size(kSpace), kS);
    fill(kMark, std::size(kMark), kM);
    fill(kCaseLead, std::size(kCaseLead), kCL);
    fill(kCaseTail, std::size(kCaseTail), kCT);
    fill(kPunctSym, std::size(kPunctSym), kPS);
    return v;
  }();
  return t;
}

// Hangul syllables compose and decompose algorithmically (Unicode ch. 3.12).
constexpr uint32_t kSBase = 0xAC00, kLBase = 0x1100, kVBase = 0x1161, kTBase = 0x11A7;
constexpr uint32_t kLCount = 19, kVCount = 21, kTCount = 28, kNCount = kVCount * kTCount;
constexpr uint32_t kSCount = kLCount * kNCount;

void decompose(char32_t c, std::u32string &out) {
  uint32_t u = static_cast<uint32_t>(c);
  if (u - kSBase < kSCount) {
    uint32_t s = u - kSBase;
    out.push_back(kLBase + s / kNCount);
    out.push_back(kVBase + (s % kNCount) / kTCount);
    if (s % kTCount) out.push_back(kTBase + s % kTCount);
    return;
  }
  if (u >= 0xC0) {
    const Decomp *end = kDecomp + std::size(kDecomp);
    const Decomp *it = std::lower_bound(kDecomp, end, u, [](const Decomp &d, uint32_t v) { return d.cp < v; });
    if (it != end && it->cp == u) {
      for (uint16_t k = 0; k < it->length; ++k) out.push_back(kDecompData[it->offset + k]);
      return;
    }
  }
  out.push_back(c);
}

// The primary composite of a pair, 0 when there is none.
char32_t compose(char32_t a, char32_t b) {
  uint32_t ua = static_cast<uint32_t>(a), ub = static_cast<uint32_t>(b);
  if (ua - kLBase < kLCount && ub - kVBase < kVCount)
    return kSBase + ((ua - kLBase) * kVCount + (ub - kVBase)) * kTCount;
  if (ua - kSBase < kSCount && (ua - kSBase) % kTCount == 0 && ub - (kTBase + 1) < kTCount - 1)
    return ua + (ub - kTBase);
  const Compose *end = kCompose + std::size(kCompose);
  const Compose *it = std::lower_bound(kCompose, end, std::pair{ua, ub},
                                       [](const Compose &x, std::pair<uint32_t, uint32_t> v) {
                                         return x.a < v.first || (x.a == v.first && x.b < v.second);
                                       });
  if (it != end && it->a == ua && it->b == ub) return it->c;
  return 0;
}

}  // namespace

bool is_letter(char32_t c) {
  uint32_t u = c;
  return u < kFlatLimit ? (flat()[u] & kL) != 0 : in_ranges(kLetter, std::size(kLetter), u);
}
bool is_number(char32_t c) {
  uint32_t u = c;
  return u < kFlatLimit ? (flat()[u] & kN) != 0 : in_ranges(kNumber, std::size(kNumber), u);
}
bool is_space(char32_t c) {
  uint32_t u = c;
  return u < kFlatLimit ? (flat()[u] & kS) != 0 : in_ranges(kSpace, std::size(kSpace), u);
}
bool is_mark(char32_t c) {
  uint32_t u = c;
  return u < kFlatLimit ? (flat()[u] & kM) != 0 : in_ranges(kMark, std::size(kMark), u);
}

bool is_case_lead(char32_t c) {
  uint32_t u = c;
  return u < kFlatLimit ? (flat()[u] & kCL) != 0 : in_ranges(kCaseLead, std::size(kCaseLead), u);
}
bool is_case_tail(char32_t c) {
  uint32_t u = c;
  return u < kFlatLimit ? (flat()[u] & kCT) != 0 : in_ranges(kCaseTail, std::size(kCaseTail), u);
}
bool is_punct_symbol(char32_t c) {
  uint32_t u = c;
  return u < kFlatLimit ? (flat()[u] & kPS) != 0 : in_ranges(kPunctSym, std::size(kPunctSym), u);
}

bool folds_to(char32_t c, char lower) {
  for (const FoldEntry &f : kContractionFold)
    if (f.lower == lower && f.cp == static_cast<uint32_t>(c)) return true;
  return false;
}

uint8_t combining_class(char32_t c) {
  uint32_t u = c;
  if (u < 0x300) return 0;
  const CccRange *end = kCcc + std::size(kCcc);
  const CccRange *it = std::upper_bound(kCcc, end, u, [](uint32_t v, const CccRange &x) { return v < x.lo; });
  return (it != kCcc && u <= (it - 1)->hi) ? (it - 1)->ccc : 0;
}

void nfc(std::u32string_view in, std::u32string &out) {
  // Below U+0300 NFC changes nothing: no decomposition there differs from its recomposition
  // and nothing combines.
  if (std::all_of(in.begin(), in.end(), [](char32_t c) { return c < 0x300; })) {
    out.append(in);
    return;
  }
  std::u32string d;
  d.reserve(in.size() + 8);
  for (char32_t c : in) decompose(c, d);
  // Canonical ordering: a stable sort of each run of non-starters by class.
  std::vector<uint8_t> cc(d.size());
  for (size_t i = 0; i < d.size(); ++i) cc[i] = combining_class(d[i]);
  for (size_t i = 0; i < d.size();) {
    if (!cc[i]) {
      ++i;
      continue;
    }
    size_t j = i;
    while (j < d.size() && cc[j]) ++j;
    for (size_t k = i + 1; k < j; ++k)  // insertion sort: runs are short, and it is stable
      for (size_t m = k; m > i && cc[m - 1] > cc[m]; --m) {
        std::swap(cc[m - 1], cc[m]);
        std::swap(d[m - 1], d[m]);
      }
    i = j;
  }
  // Canonical composition: a character combines with the last starter unless something in
  // between blocks it (a class >= its own, or anything at all when it is a starter itself).
  // Whatever follows a starter that was kept is a non-starter, so `last_cc` > 0 when not adjacent.
  long starter = -1;
  uint8_t last_cc = 0;
  for (size_t i = 0; i < d.size(); ++i) {
    char32_t c = d[i];
    uint8_t k = cc[i];
    if (starter >= 0) {
      bool adjacent = out.size() - 1 == static_cast<size_t>(starter);
      if (adjacent || last_cc < k) {
        if (char32_t p = compose(out[starter], c)) {
          out[starter] = p;
          continue;
        }
      }
    }
    if (k == 0) starter = static_cast<long>(out.size());
    last_cc = k;
    out.push_back(c);
  }
}

void decode_utf8(std::string_view s, std::u32string &out) {
  size_t i = 0, n = s.size();
  while (i < n) {
    uint8_t b = static_cast<uint8_t>(s[i]);
    if (b < 0x80) {
      out.push_back(b);
      ++i;
      continue;
    }
    // Strict decoding (shortest form, no surrogates, <= U+10FFFF), as Python's str.
    uint32_t c = 0, need = 0, lo = 0x80, hi = 0xBF;
    if (b >= 0xC2 && b <= 0xDF) {
      c = b & 0x1F, need = 1;
    } else if (b >= 0xE0 && b <= 0xEF) {
      c = b & 0x0F, need = 2, lo = b == 0xE0 ? 0xA0 : 0x80, hi = b == 0xED ? 0x9F : 0xBF;
    } else if (b >= 0xF0 && b <= 0xF4) {
      c = b & 0x07, need = 3, lo = b == 0xF0 ? 0x90 : 0x80, hi = b == 0xF4 ? 0x8F : 0xBF;
    }
    size_t j = i + 1;
    bool ok = need > 0;
    for (uint32_t k = 0; ok && k < need; ++k, ++j) {
      uint8_t t = j < n ? static_cast<uint8_t>(s[j]) : 0;
      if (j >= n || t < (k == 0 ? lo : 0x80) || t > (k == 0 ? hi : 0xBF)) {
        ok = false;
        break;
      }
      c = (c << 6) | (t & 0x3F);
    }
    out.push_back(ok ? c : 0xFFFD);  // the maximal valid prefix becomes one U+FFFD
    i = j;
  }
}

void append_utf8(char32_t c, std::string &out) {
  uint32_t u = c;
  if (u < 0x80) {
    out.push_back(static_cast<char>(u));
  } else if (u < 0x800) {
    out.push_back(static_cast<char>(0xC0 | (u >> 6)));
    out.push_back(static_cast<char>(0x80 | (u & 0x3F)));
  } else if (u < 0x10000) {
    out.push_back(static_cast<char>(0xE0 | (u >> 12)));
    out.push_back(static_cast<char>(0x80 | ((u >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (u & 0x3F)));
  } else {
    out.push_back(static_cast<char>(0xF0 | (u >> 18)));
    out.push_back(static_cast<char>(0x80 | ((u >> 12) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | ((u >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (u & 0x3F)));
  }
}

}  // namespace xenithium:text:unicode
