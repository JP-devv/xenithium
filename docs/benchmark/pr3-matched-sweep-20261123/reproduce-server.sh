#!/usr/bin/env bash
# Historical PR #3 matched C1 fixture. Only run under a separately authorized exclusive GPU lease.
set -euo pipefail
arm=${1:?usage: bash reproduce-server.sh base\|dense\|trained\|xmx}
: "${MODEL_DIR:?set MODEL_DIR to the pinned compatible checkpoint}"
: "${DRAFTER_DIR:?set DRAFTER_DIR to the pinned DSpine checkpoint}"
: "${ONEAPI_ENV:?set ONEAPI_ENV to the pinned oneapi-env.sh}"
export XEN_ONEAPI=/opt/intel/oneapi/2025.3
printf '%s  %s\n' 44770d1b1aa2520dcbf9b28c7501ad59c381b667cdf55fa885bb93ba61038da3 "$ONEAPI_ENV" | sha256sum -c -
# The original managed job started from env -i. Use a fresh clean shell as documented.
source "$ONEAPI_ENV"
while IFS='=' read -r name _; do case "$name" in XEN_*) [[ $name == XEN_ONEAPI ]] || unset "$name";; esac; done < <(env)
unset LD_PRELOAD ZE_AFFINITY_MASK ONEAPI_DEVICE_SELECTOR UR_ADAPTERS_FORCE_LOAD ZE_ENABLE_PCI_ID_DEVICE_ORDER ZET_ENABLE_METRICS
case "$arm" in
  base) : "${BASE_SERVER:?set BASE_SERVER}"; server=$BASE_SERVER; hash=59d9af05c5ca719a9fab55f372bd79400fcb9802a020eb0272b4e7e07d3e2e33; xl=5e4248f8da87823c116fdfc5d41d1b14dc175da3d0ab4ec09936114bf5671f13 ;;
  dense|trained|xmx) : "${CLEANUP_SERVER:?set CLEANUP_SERVER}"; server=$CLEANUP_SERVER; hash=4131140b53f935f27969ce91133b4d3f5e13c49737b70fe90d1e148338f74c7a; xl=7250149afe3da67a7e5818a23b8ea00bd5b442879043089ac8baee32d40e7eb8 ;;
  *) echo "invalid arm: $arm" >&2; exit 2 ;;
esac
printf '%s  %s\n' "$hash" "$server" | sha256sum -c -
printf '%s  %s\n' "$xl" "$(dirname -- "$server")/libxenithium_kernels_xl.so" | sha256sum -c -
if [[ $arm == trained || $arm == xmx ]]; then
  : "${QSA_BANK:?set QSA_BANK to the pinned sidecar directory}"
  printf '%s  %s\n' 241f04cd8db6f2820fb50ff4d8f4cfbd1a1be1fe2993dcdb9caccf17a3e83d9a "$QSA_BANK/qsa-indexers.safetensors" | sha256sum -c -
  export XEN_TRAINED_QSA_DELTA_DIR="$QSA_BANK" XEN_TRAINED_QSA_TILED_SCORE=1 XEN_TRAINED_QSA_POOL_FP16=1
  if [[ $arm == xmx ]]; then export XEN_TRAINED_QSA_PREFILL_XMX=1; else export XEN_TRAINED_QSA_PREFILL_XMX=0; fi
fi
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
mapfile -t args <"$here/reproduce-server.argv"
model=0;drafter=0
for i in "${!args[@]}"; do
  case ${args[i]} in
    @MODEL_DIR@) args[i]=$MODEL_DIR; model=$((model+1)) ;;
    @DRAFTER_DIR@) args[i]=$DRAFTER_DIR; drafter=$((drafter+1)) ;;
  esac
done
[[ $model == 1 && $drafter == 1 ]]
exec "$server" "${args[@]}"
