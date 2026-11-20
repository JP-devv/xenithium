// Xenithium · calibration set for the bake tool: a fixed mix of datasets rendered and tokenized
// the way the server sees them, packed into samples of exactly `seqlen` tokens.
// Sources (all read natively, no Python): parquet text columns (pile-10k), parquet chat
// columns `messages.list.{item,element}.{content,role}` (ultrachat_200k, Mixture-of-Thoughts),
// JSONL chats with OpenAI messages + tools (Fable agentic traces; their `tool_calls` are Python
// literals and are converted), and JSON arrays of ShareGPT `conversations` (hermes function
// calling). Chats go through the arch's chat template (the serving path); raw text is
// tokenized as is.
// A sample is documents of ONE source concatenated until seqlen tokens (raw text joined by the
// EOS token), cut at seqlen; documents are taken in a seeded order, so a build is reproducible.
// File (little endian): 'XCB1', int32 samples, int32 seqlen, int32 sources, then per source a
// 64-byte NUL-padded name, then per sample int32 source index and int32 tokens[seqlen].
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace xenithium::quant {

struct CalibSource {
  std::string name;  // label in the file and the report
  std::string kind;  // "parquet-text" | "parquet-chat" | "jsonl-chat" | "sharegpt-json"
  std::vector<std::string> files;
  std::string column;  // parquet-text: the text column; parquet-chat: the list prefix ("messages.list.item")
  int samples = 0;  // samples of seqlen tokens this source contributes
};

struct CalibSet {
  int seqlen = 0;
  std::vector<std::string> sources;
  std::vector<int32_t> source_of;  // [samples]
  std::vector<int32_t> tokens;  // [samples][seqlen]
  int samples() const { return static_cast<int>(source_of.size()); }
  const int32_t *sample(int s) const { return tokens.data() + static_cast<size_t>(s) * seqlen; }
};

// The default mix under a datasets root (one directory per dataset), scaled to `samples` in total:
// pile-10k 160, ultrachat 96, Mixture-of-Thoughts 112, hermes function calling 48, Fable agentic 96
// (per 512). Sources whose files are missing throw.
std::vector<CalibSource> default_mix(const std::string &root, int samples);

struct CalibBuildStats {
  std::vector<int> docs_used, docs_failed;  // per source (failed = the template rejected it)
};
// model_dir: tokenizer.json + the arch's chat format (arch id "qwen38"). threads: tokenizer threads.
CalibSet build_calib(const std::vector<CalibSource> &mix, const std::string &model_dir, int seqlen, uint64_t seed, int threads,
                     CalibBuildStats *stats = nullptr);

void write_calib(const std::string &path, const CalibSet &c);
CalibSet read_calib(const std::string &path);

// A Python literal (dict / list / str / int / float / True / False / None) as JSON text.
// Throws on anything else. For datasets that stored repr of their tool calls.
std::string python_literal_to_json(const std::string &src);

}  // namespace xenithium:quant
