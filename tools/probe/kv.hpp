// Probe helper: the KV dtype a probe command runs. --kv-bytes 2 = bf16, 1 = int8; with --kv-v4 1 qwen38's
// QSA model35 / gqa35) 1 = int8 K + int4 V.
#pragma once

#include "xenithium/kernels/attention/qsa.hpp"

extern bool g_kv_v4;  // --kv-v4 0|1 (flags.cpp)

inline xenithium::kernels::QsaKv probe_kv(int kv_bytes) {
  using xenithium::kernels::QsaKv;
  return kv_bytes == 2 ? QsaKv::bf16 : g_kv_v4 ? QsaKv::i8v4 : QsaKv::i8;
}
inline const char *probe_kv_name(xenithium::kernels::QsaKv kv) {
  using xenithium::kernels::QsaKv;
  return kv == QsaKv::i8 ? "int8" : kv == QsaKv::i8v4 ? "int8-v4" : "bf16";
}
