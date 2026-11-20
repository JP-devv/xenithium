// xenithium-bake: offline quantisation + calibration of a checkpoint into the engine's formats.
// xenithium-bake calib --datasets DIR --model MODEL_DIR --out FILE [--samples 512] [--seqlen 2048]
// [--seed 42] [--threads 32]
// Builds the calibration mix (calib.hpp) and writes it as an XCB1 token file.
// xenithium-bake synth --src BF16_DIR [...] (synth.cpp)
// AutoRound on one real layer's experts with synthetic inputs: RTN vs tuned.
// xenithium-bake rtn --src BF16_DIR --out DIR [--layers A-B] (experts.cpp)
// The RTN baseline bake of the routed experts, in the export layout.
// xenithium-bake check-layout --src BF16_DIR --ref W4A16_DIR (experts.cpp)
// Our export vs a reference int4 checkpoint, both against BF16.
// xenithium-bake ple --src BF16_DIR --out DIR [--shards A-B] (ple.cpp)
// The PLE n-gram table as int8 rows (fp32 scale per row).
// Tuning runs on the GPUs: --gpus N (default every GPU). The host cores feed them (weight
// streaming, the per-step reduction, packing); there is no CPU tuning worker.
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>

#include "bake.hpp"
#include "xenithium/quant/calib.hpp"

namespace q = xenithium::quant;
using xenithium::bake::Args;
using xenithium::bake::home;

namespace {

Args parse(int argc, char **argv, int from) {
  Args a;
  for (int i = from; i < argc; ++i) {
    if (std::strncmp(argv[i], "--", 2) != 0) {
      std::fprintf(stderr, "bake: unexpected argument %s\n", argv[i]);
      std::exit(2);
    }
    const std::string k = argv[i] + 2;
    if (i + 1 < argc && std::strncmp(argv[i + 1], "--", 2) != 0) a.kv[k] = argv[++i];
    else a.kv[k] = "1";
  }
  return a;
}

int cmd_calib(const Args &a) {
  const std::string root = a.str("datasets", home() + "/ai/datasets"), model = a.str("model", home() + "/models/qwen38-w4a16");
  const std::string out = a.str("out");
  if (out.empty()) {
    std::fprintf(stderr, "bake calib: --out FILE is required\n");
    return 2;
  }
  const int samples = static_cast<int>(a.num("samples", 512)), seqlen = static_cast<int>(a.num("seqlen", 2048));
  const int threads = static_cast<int>(a.num("threads", std::thread::hardware_concurrency()));
  const auto t0 = std::chrono::steady_clock::now();
  const auto mix = q::default_mix(root, samples);
  q::CalibBuildStats st;
  const q::CalibSet c = q::build_calib(mix, model, seqlen, static_cast<uint64_t>(a.num("seed", 42)), threads, &st);
  q::write_calib(out, c);
  const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  std::printf("%-26s %8s %8s %8s\n", "source", "samples", "docs", "rejected");
  for (size_t i = 0; i < mix.size(); ++i)
    std::printf("%-26s %8d %8d %8d\n", mix[i].name.c_str(), mix[i].samples, st.docs_used[i], st.docs_failed[i]);
  std::printf("calib: %d samples x %d tokens -> %s (%.1f s)\n", c.samples(), c.seqlen, out.c_str(), s);
  return 0;
}

}  // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: xenithium-bake calib|synth|rtn|check-layout|ple|selftest-calib|tune|assemble|pack-native|mtp|mtp-tune|mtp-states|head-capture|head-calib|kld|prompts|head4-calib|gdn4-calib [--options]\n");
    return 2;
  }
  const std::string cmd = argv[1];
  const Args a = parse(argc, argv, 2);
  try {
    if (cmd == "calib") return cmd_calib(a);
    if (cmd == "synth") return xenithium::bake::cmd_synth(a);
    if (cmd == "rtn") return xenithium::bake::cmd_rtn(a);
    if (cmd == "check-layout") return xenithium::bake::cmd_check_layout(a);
    if (cmd == "ple") return xenithium::bake::cmd_ple(a);
    if (cmd == "selftest-calib") return xenithium::bake::cmd_selftest_calib(a);
    if (cmd == "tune") return xenithium::bake::cmd_tune(a);
    if (cmd == "assemble") return xenithium::bake::cmd_assemble(a);
    if (cmd == "pack-native") return xenithium::bake::cmd_pack_native(a);
    if (cmd == "mtp") return xenithium::bake::cmd_mtp(a);
    if (cmd == "mtp-tune") return xenithium::bake::cmd_mtp_tune(a);
    if (cmd == "mtp-states") return xenithium::bake::cmd_mtp_states(a);
    if (cmd == "head-capture") return xenithium::bake::cmd_head_capture(a);
    if (cmd == "head-calib") return xenithium::bake::cmd_head_calib(a);
    if (cmd == "kld") return xenithium::bake::cmd_kld(a);
    if (cmd == "prompts") return xenithium::bake::cmd_prompts(a);
    if (cmd == "head4-calib") return xenithium::bake::cmd_head4_calib(a);
    if (cmd == "gdn4-calib") return xenithium::bake::cmd_gdn4_calib(a);
    if (cmd == "dense-tune") return xenithium::bake::cmd_dense_tune(a);
    if (cmd == "dense-assemble") return xenithium::bake::cmd_dense_assemble(a);
  } catch (const std::exception &e) {
    std::fprintf(stderr, "bake %s: %s\n", cmd.c_str(), e.what());
    return 1;
  }
  std::fprintf(stderr, "bake: unknown command %s\n", cmd.c_str());
  return 2;
}
