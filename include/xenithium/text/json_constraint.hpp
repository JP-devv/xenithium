// Structured output: a JSON schema (response_format json_schema / json_object, guided_json)
// or a list of choices (guided_choice) compiled into a token-level constraint for the engine.
// The grammar is a byte-level pushdown automaton over JSON, specialised by the schema. A state is a
// small set of configurations (anyOf / type lists / literal prefixes fork it); each configuration is
// a stack of frames (object, array, string, number, literal) plus the current whitespace run. The
// allowed tokens of a state are found by walking the vocabulary's byte trie from that state and
// keeping every token whose bytes the automaton accepts; masks are cached per state in the compiled
// spec, which the server shares between requests with the same schema.
// Where the semantics differ from plain JSON
// Schema: object properties appear in schema order (each optional one may be skipped, every
// required one must appear), and additionalProperties defaults to false; an object schema without
// a "properties" keyword accepts any object. Whitespace is allowed between tokens inside the value, at
// most 32 bytes per run (a greedy model would otherwise emit whitespace until max_tokens), and not
// before or after the root value. Not supported (400 at compile):
// pattern, patternProperties, prefixItems, not, if/then/else, enum/const sets over 64 values,
// more than 64 properties in one object, property names that need JSON escapes, nesting over 24.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "xenithium/core/engine.hpp"
#include "xenithium/text/tokenizer.hpp"
#include "xenithium/util/json.hpp"

namespace xenithium::text {

// The vocabulary as bytes, with its trie; built once per tokenizer and shared by every grammar.
class ConstraintVocab {
 public:
  // model_vocab: the logits row length (may exceed the tokenizer's ids; those are never allowed).
  // eos: the ids that may end a complete document. Special tokens are never allowed inside one.
  static std::shared_ptr<const ConstraintVocab> build(const Tokenizer &tok, int64_t model_vocab, const std::vector<int32_t> &eos);
  ~ConstraintVocab();
  int64_t vocab() const;
  struct Impl;
  const Impl &impl() const { return *impl_; }

 private:
  ConstraintVocab();
  std::unique_ptr<Impl> impl_;
};

// Compiles a schema (raw_choice: `schema` is a JSON array of strings, matched as raw text:
// guided_choice). start_after >= 0: the constraint starts after this token id has been
// emitted (the reasoning end, `</think>, for a thinking chat); every token is allowed before it.
// nullptr + err (a message for a 400) when the schema uses something the grammar cannot express.
std::shared_ptr<const core::ConstraintSpec> compile_json_constraint(const json::Value &schema, bool raw_choice,
                                                                    std::shared_ptr<const ConstraintVocab> vocab, int32_t start_after,
                                                                    std::string &err);

// For the gates: walk `text` through a fresh constraint of `spec` token by token (the tokenizer's
// ids), checking at each token that it is in the allowed set. Returns the index of the first token
// that was not allowed, or -1 when every token was; `complete` = the document may end there (EOS
// allowed). Also counts the states whose mask had to be computed.
struct WalkResult {
  int first_rejected = -1;
  bool complete = false;
  int masks_computed = 0;
};
WalkResult walk_constraint(const core::ConstraintSpec &spec, const std::vector<int32_t> &ids);

}  // namespace xenithium:text
