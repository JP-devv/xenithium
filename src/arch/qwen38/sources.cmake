# qwen38: Qwen3.8 Flash-Next (MoE + Gated DeltaNet + QSA + PLE + HC; TP2 + EP2).
xenithium_arch_sources(
  HOST   arch.cpp experts.cpp ple_table.cpp mtp.cpp
  DEVICE expert_store.cpp head.cpp hc.cpp moe.cpp gdn.cpp qsa.cpp ple.cpp model.cpp drafter.cpp prefill.cpp prefix.cpp
  SERVE  step_model.cpp serve.cpp)
