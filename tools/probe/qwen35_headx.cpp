// xenithium-probe head35x: the lm_head's inputs at generation time, for the int4 head's GPTQ.
// xenithium-probe head35x --model DIR --ppl TRAIN.bin,HELDOUT.bin [--generate 128] --head-x-out FILE
// Per prompt of each token file (`xenithium-bake prompts`): reset, prefill the prompt, then greedy decode, 1 row per
// step, until `generate` tokens are out; the final-normed hidden state (the lm_head's input, card 0's copy) of every
// row whose pick is a generated token is kept: the prompt's last row, then each step's row = `generate` rows per
// prompt. FILE = XHI1 (`xenithium-bake head4-calib`): uint32 'XHI1', int32 hidden, int64 training rows, int64 held-out
// rows, then bf16 rows (the first file's prompts, then the second's). The model runs as served (int8 head unless
// --head-w4 1; the head's input does not depend on the head).
#include <chrono>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "xenithium/arch/qwen35/model.hpp"
#include "xenithium/rt/fabric.hpp"
#include "xenithium/rt/l0.hpp"
#include "xenithium/text/tokenizer.hpp"

namespace q35 = xenithium::arch::qwen35;

#include "probe.hpp"  // the run-wide flags
void q35_apply_r26_options(q35::ModelOptions &mo);  // qwen35_model.cpp

namespace {

bool read_prompts(const std::string &path, std::vector<std::vector<int32_t>> &chunks) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return false;
  int32_t n = 0;
  if (!f.read(reinterpret_cast<char *>(&n), 4) || n <= 0 || n > 100000) return false;
  chunks.resize(static_cast<size_t>(n));
  for (auto &c : chunks) {
    int32_t len = 0;
    if (!f.read(reinterpret_cast<char *>(&len), 4) || len <= 1 || len > (1 << 24)) return false;
    c.resize(static_cast<size_t>(len));
    if (!f.read(reinterpret_cast<char *>(c.data()), static_cast<std::streamsize>(len) * 4)) return false;
  }
  return true;
}

}  // namespace

int cmd_head35x(const std::string &model, const std::string &ppl, int generate, int threads) {
  std::vector<std::string> files;
  {
    std::stringstream ss(ppl);
    std::string f;
    while (std::getline(ss, f, ','))
      if (!f.empty()) files.push_back(f);
  }
  if (files.size() != 2 || g_head_x_out.empty() || generate < 1) {
    std::fprintf(stderr, "head35x: --ppl TRAIN.bin,HELDOUT.bin, --head-x-out FILE and --generate N are required\n");
    return 2;
  }
  std::vector<std::vector<std::vector<int32_t>>> sets(2);
  size_t longest = 0;
  for (int i = 0; i < 2; ++i) {
    if (!read_prompts(files[static_cast<size_t>(i)], sets[static_cast<size_t>(i)])) {
      std::fprintf(stderr, "head35x: cannot read %s\n", files[static_cast<size_t>(i)].c_str());
      return 2;
    }
    for (const auto &p : sets[static_cast<size_t>(i)]) longest = std::max(longest, p.size());
  }
  q35::ModelOptions mo;
  mo.context = std::max<int64_t>(8192, static_cast<int64_t>((longest + static_cast<size_t>(generate) + 64 + 63) / 64 * 64));
  mo.kv = xenithium::kernels::QsaKv::i8;
  mo.threads = threads;
  q35_apply_r26_options(mo);
  const auto loader = xenithium::l0::Loader::open();
  xenithium::Fabric fab(loader, {0, 1});
  mo.loader = &loader;
  q35::Model m(fab, model, mo);
  const int H = m.config().hidden;
  xenithium::text::Tokenizer tok;
  std::string terr;
  const bool have_tok = tok.load(model, terr);
  std::ofstream out(g_head_x_out, std::ios::binary);
  const uint32_t magic = 0x31494858u;
  int64_t counts[2] = {0, 0};
  out.write(reinterpret_cast<const char *>(&magic), 4).write(reinterpret_cast<const char *>(&H), 4);
  out.write(reinterpret_cast<const char *>(counts), 16);  // rewritten at the end
  const auto t0 = std::chrono::steady_clock::now();
  for (int si = 0; si < 2; ++si) {
    for (size_t pi = 0; pi < sets[static_cast<size_t>(si)].size(); ++pi) {
      const auto &p = sets[static_cast<size_t>(si)][pi];
      m.reset();
      m.prefill(p.data(), static_cast<int>(p.size()), {});
      const int lr = m.prefill_last_row();
      auto row = m.final_hidden(0, lr + 1);
      out.write(reinterpret_cast<const char *>(row.data() + static_cast<size_t>(lr) * H), static_cast<std::streamsize>(H) * 2);
      int32_t next = m.picks(0, lr + 1)[static_cast<size_t>(lr)];
      std::vector<int32_t> gen{next};
      for (int i = 1; i < generate; ++i) {
        m.step(&next, 1);
        m.accept(1);
        row = m.final_hidden(0, 1);
        out.write(reinterpret_cast<const char *>(row.data()), static_cast<std::streamsize>(H) * 2);
        next = m.picks(0, 1)[0];
        gen.push_back(next);
      }
      counts[si] += generate;
      if (pi == 0 && have_tok) {
        std::string t = tok.decode(gen, false), e;
        for (char ch : t) e += ch == '\n' ? std::string("\\n") : std::string(1, ch);
        std::printf("  %s prompt 0 (%zu tokens) -> %s\n", files[static_cast<size_t>(si)].c_str(), p.size(), e.substr(0, 400).c_str());
      }
      if ((pi + 1) % 32 == 0)
        std::printf("  %s: %zu / %zu prompts, %.0f s\n", files[static_cast<size_t>(si)].c_str(), pi + 1, sets[static_cast<size_t>(si)].size(),
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
      std::fflush(stdout);
    }
  }
  out.seekp(8);
  out.write(reinterpret_cast<const char *>(counts), 16);
  out.close();
  if (!out) {
    std::fprintf(stderr, "head35x: cannot write %s\n", g_head_x_out.c_str());
    return 1;
  }
  std::printf("head35x: %lld training + %lld held-out rows of %d -> %s (%.0f s); wire errors %u\n", static_cast<long long>(counts[0]),
              static_cast<long long>(counts[1]), H, g_head_x_out.c_str(), std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(),
              m.wire_errors());
  return m.wire_errors() == 0 ? 0 : 1;
}
