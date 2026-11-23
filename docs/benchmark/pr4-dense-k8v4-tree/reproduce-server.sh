#!/usr/bin/env bash
# Planned PR #4 sweep only; run solely inside an authorized exclusive GPU window.
set -euo pipefail
arm=${1:?usage: bash reproduce-server.sh base\|head-historical\|head-merged 0\|7 [--check-only]}
tree=${2:?usage: bash reproduce-server.sh base\|head-historical\|head-merged 0\|7 [--check-only]}
mode=${3:-}
[[ $# -le 3 && ( -z $mode || $mode == --check-only ) ]] || { echo 'invalid option' >&2; exit 2; }
: "${INPUTS:?set INPUTS to the pinned sweep-inputs directory}"
: "${MODEL_DIR:?set MODEL_DIR to the pinned compatible checkpoint}"
: "${DRAFTER_DIR:?set DRAFTER_DIR to the pinned drafter}"
: "${ONEAPI_ENV:?set ONEAPI_ENV to the pinned oneapi-env.sh}"
case "$tree" in
  0) argv_sha=b79480f2295a343423ad6bd4b58a55fe1541a86ded991610df12448d63303dde ;;
  7) argv_sha=b24e2a285589461da53fa342e2f9c82a55cfcba3cef977aa0b1b31e31cd6890d ;;
  *) echo 'tree must be 0 or 7' >&2; exit 2 ;;
esac
case "$arm" in
  base) : "${BASE_SERVER:?set BASE_SERVER}"; server=$BASE_SERVER; hash=59d9af05c5ca719a9fab55f372bd79400fcb9802a020eb0272b4e7e07d3e2e33; xl=5e4248f8da87823c116fdfc5d41d1b14dc175da3d0ab4ec09936114bf5671f13 ;;
  head-historical) : "${HEAD_SERVER:?set HEAD_SERVER to the f0e84a98 server}"; server=$HEAD_SERVER; hash=beb341998d43c3648e96fb1fba2021f590073e4a4db4fb15e9bbfa197680870b; xl=f3ef900e4e1b5769742215c533b2982652ced3d7187c322551ef40a6ef9ed66c ;;
  head-merged) : "${HEAD_SERVER:?set HEAD_SERVER to the 3bfc10e server}"; server=$HEAD_SERVER; hash=8b1b39196fe1cf9ba8baa72e60f75fbd32a644f355ad0b7b36a0e7dd68fb2027; xl=21668ec86a1a29c91320b37b10737416ad76646ae7babfe87d2d4a8b78a93cab ;;
  head) echo 'head is ambiguous: use head-historical or head-merged' >&2; exit 2 ;;
  *) echo 'arm must be base or head' >&2; exit 2 ;;
esac
printf '%s  %s\n' "$argv_sha" "$INPUTS/tree$tree.argv" | sha256sum -c -
printf '%s  %s\n' "$hash" "$server" | sha256sum -c -
printf '%s  %s\n' "$xl" "$(dirname -- "$server")/libxenithium_kernels_xl.so" | sha256sum -c -
printf '%s  %s\n' 44770d1b1aa2520dcbf9b28c7501ad59c381b667cdf55fa885bb93ba61038da3 "$ONEAPI_ENV" | sha256sum -c -
if [[ $mode == --check-only ]]; then echo "pin checks passed: $arm tree$tree"; exit 0; fi
export XEN_ONEAPI=/opt/intel/oneapi/2025.3
source "$ONEAPI_ENV"
while IFS='=' read -r name _; do case "$name" in XEN_*) [[ $name == XEN_ONEAPI ]] || unset "$name";; esac; done < <(env)
unset LD_PRELOAD ZE_AFFINITY_MASK ONEAPI_DEVICE_SELECTOR UR_ADAPTERS_FORCE_LOAD ZE_ENABLE_PCI_ID_DEVICE_ORDER ZET_ENABLE_METRICS
mapfile -t args <"$INPUTS/tree$tree.argv"
model=0; drafter=0
for i in "${!args[@]}"; do
  case ${args[i]} in
    --model) args[i+1]=$MODEL_DIR; model=$((model+1)) ;;
    --drafter-model) args[i+1]=$DRAFTER_DIR; drafter=$((drafter+1)) ;;
  esac
done
[[ $model == 1 && $drafter == 1 ]]
exec "$server" "${args[@]}"
