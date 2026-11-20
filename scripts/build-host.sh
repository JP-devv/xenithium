#!/usr/bin/env bash
# Incremental native build (icpx, AOT for Battlemage). Usage: scripts/build-host.sh [target...]
# A build prints no warnings: the C++ sources are warning-free under -Wall -Wextra, and the device compiler's notes
# (a remark per loop it does not vectorise, per-kernel register-spill notes) are dropped at link time by
# scripts/quiet-device-link.sh. To read them: cmake -S . -B build -DXENITHIUM_DEVICE_COMPILER_NOTES=ON, then build.
set -euo pipefail
cd "$(dirname "$0")/.."
if ! command -v icpx >/dev/null; then
  # shellcheck disable=SC1091
  source scripts/oneapi-env.sh
fi
[[ -f build/CMakeCache.txt ]] || cmake -S . -B build -DCMAKE_CXX_COMPILER=icpx -DCMAKE_BUILD_TYPE=Release
if (($#)); then
  for t in "$@"; do cmake --build build -j "$(nproc)" --target "$t"; done
else
  cmake --build build -j "$(nproc)"
fi
