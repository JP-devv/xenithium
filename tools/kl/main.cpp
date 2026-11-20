// xenithium-kl: the host side of the KL gate (no GPU, no dependencies beyond the repo).
// xenithium-kl dump --tokens FILE --out FILE [--base-url http://localhost:8000] [--model NAME] [--k 20] [--timeout S]
// Reference log-probs: each chunk of the token file (the
// `probe model --ppl` format) is sent as token ids to /v1/completions with prompt_logprobs =
// logprobs = k, return_tokens_as_token_ids, max_tokens 1, temperature 0. No tokenizer between
// the engines; the rows line up with `probe model --ppl` row for row (n rows for n tokens; the
// last row's target is the reference's greedy pick, as the ppl gate scores). k is capped by
// the server's --max-logprobs (default 20). Prints the ppl of the dump: on the 548-token
// set it should reproduce the stored reference baseline, 14.4613.
// xenithium-kl compare REF TEST [--buckets 128,512,2048,8192,32768]
// Dump vs dump (either side from `dump` or `probe model --lp-out`): KL(REF || TEST) per
// position bucket on REF's top k + a tail bucket, ppl of each, top-1 agreement. TEST is known
// only on its own top k here (REF ids outside it take its last log-prob, counted in "bound",
// KL understated there); `probe model --lp-ref` gives the same table exact on its side.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <sstream>
#include <string>
#include <vector>

#include "xenithium/eval/logprobs.hpp"
#include "xenithium/eval/reference.hpp"

namespace ev = xenithium::eval;

namespace {

int usage() {
  std::fprintf(stderr,
               "usage: xenithium-kl dump --tokens FILE --out FILE [--base-url URL] [--model NAME] [--k 20] [--timeout S]\n"
               "       xenithium-kl compare REF.xlp TEST.xlp [--buckets 128,512,2048,8192,32768]\n");
  return 2;
}

std::string base_name(const std::string &p) { return p.substr(p.find_last_of('/') + 1); }

int cmd_dump(const std::vector<std::string> &a) {
  std::string tokens, out, base = "http://localhost:8000", model;
  int k = 20;
  double timeout = 1800;
  for (size_t i = 0; i < a.size(); i += 2) {
    if (i + 1 >= a.size()) return usage();
    const std::string &o = a[i], &v = a[i + 1];
    if (o == "--tokens") tokens = v;
    else if (o == "--out") out = v;
    else if (o == "--base-url") base = v;
    else if (o == "--model") model = v;
    else if (o == "--k") k = std::atoi(v.c_str());
    else if (o == "--timeout") timeout = std::atof(v.c_str());
    else return usage();
  }
  if (tokens.empty() || out.empty() || k < 1) return usage();
  if (model.empty()) model = ev::served_model(base);
  const auto chunks = ev::read_tokens(tokens);
  ev::LpSet set;
  set.k = k;
  double nll = 0;
  int64_t n = 0;
  const auto t0 = std::chrono::steady_clock::now();
  for (size_t c = 0; c < chunks.size(); ++c) {
    set.chunks.push_back(ev::dump_reference_chunk(base, model, chunks[c], k, timeout));
    double cn = 0;
    for (const auto &r : set.chunks.back()) cn -= r.target_lp;
    nll += cn;
    n += static_cast<int64_t>(set.chunks.back().size());
    std::printf("  chunk %zu: %zu rows, ppl %.4f\n", c, set.chunks.back().size(), std::exp(cn / static_cast<double>(set.chunks.back().size())));
    std::fflush(stdout);
  }
  std::string err;
  if (!ev::write_lp(out, set, &err)) {
    std::fprintf(stderr, "%s\n", err.c_str());
    return 1;
  }
  std::printf("%s: ppl %.4f over %lld rows (%zu chunks, top %d, %.1f s) -> %s\n", model.c_str(), std::exp(nll / static_cast<double>(n)),
              static_cast<long long>(n), chunks.size(), k, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), out.c_str());
  return 0;
}

int cmd_compare(const std::vector<std::string> &a) {
  std::vector<std::string> files;
  std::vector<int64_t> edges{128, 512, 2048, 8192, 32768};
  for (size_t i = 0; i < a.size(); ++i) {
    if (a[i] == "--buckets" && i + 1 < a.size()) {
      edges.clear();
      std::stringstream ss(a[++i]);
      std::string t;
      while (std::getline(ss, t, ','))
        if (!t.empty()) edges.push_back(std::atoll(t.c_str()));
    } else {
      files.push_back(a[i]);
    }
  }
  if (files.size() != 2) return usage();
  ev::LpSet ref, test;
  std::string err;
  if (!ev::read_lp(files[0], ref, &err) || !ev::read_lp(files[1], test, &err)) {
    std::fprintf(stderr, "%s\n", err.c_str());
    return 1;
  }
  const auto d = ev::compare_dumps(ref, test, edges);
  std::printf("KL(%s || %s): ref top %d + tail bucket, test top %d; ppl columns use each side's own last-row target "
              "(%d of %zu last rows pick differently)\n%s",
              base_name(files[0]).c_str(), base_name(files[1]).c_str(), ref.k, test.k, d.last_rows_differ, ref.chunks.size(), d.stats.table().c_str());
  return 0;
}

}  // namespace

int main(int argc, char **argv) {
  if (argc < 2) return usage();
  const std::string cmd = argv[1];
  const std::vector<std::string> args(argv + 2, argv + argc);
  try {
    if (cmd == "dump") return cmd_dump(args);
    if (cmd == "compare") return cmd_compare(args);
  } catch (const std::exception &e) {
    std::fprintf(stderr, "xenithium-kl: %s\n", e.what());
    return 1;
  }
  return usage();
}
