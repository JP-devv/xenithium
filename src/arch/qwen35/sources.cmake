# qwen35: the qwen35 runtime: the dense Qwen3.8-27B (Gated DeltaNet + gated GQA), which also runs the llama and flint archs.
xenithium_arch_sources(
  HOST   arch.cpp trained_qsa.cpp
  DEVICE head.cpp mlp.cpp gdn.cpp attn.cpp model.cpp mtp.cpp trained_qsa_device.cpp
  SERVE  step_model.cpp serve.cpp)
