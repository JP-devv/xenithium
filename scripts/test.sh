#!/usr/bin/env bash
# Runs every gate (tests/CMakeLists.txt registers them with ctest, labelled host or gpu). Host gates always; GPU gates
# only when no other GPU process holds the cards. BUILD_DIR=DIR tests another build tree (default build/).
# Extra arguments go to ctest (e.g. -R gqa to run the gates whose name matches).
set -uo pipefail
cd "$(dirname "$0")/.."
build=${BUILD_DIR:-build}
[[ -f "$build/CTestTestfile.cmake" ]] || { echo "test: $build is not a configured build tree (scripts/build-host.sh)" >&2; exit 1; }
rc=0
ctest --test-dir "$build" -L host --output-on-failure "$@" || rc=1
if ps -eo comm | grep -qE '^xenithium-(serv|prob|bake)'; then
  echo "GPU gates skipped: a xenithium server/probe/bake is running"; exit $rc
fi
if [[ -n "$(docker ps -q 2>/dev/null)" ]]; then
  echo "GPU gates skipped: containers are running ($(docker ps --format '{{.Names}}' | tr '\n' ' '))"; exit $rc
fi
ctest --test-dir "$build" -L gpu --output-on-failure "$@" || rc=1
exit $rc
