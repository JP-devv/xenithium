#!/usr/bin/env bash
# MTP drafter bake: the drafter of a BF16 checkpoint (W8 trunk at load, int4 g128
# experts, int2 draft head), its routed experts AutoRound-tuned on the quantised target's hiddens.
#   scripts/bake-mtp.sh            # after scripts/bake.sh finished: mtp-tune reads its WORK states
# Env: SRC (BF16 source), WORK (scripts/bake.sh's), RTN_OUT (the RTN drafter, `xenithium-bake mtp`),
#      OUT (the tuned drafter: RTN_OUT's files hard-linked + the tuned experts), SAMPLES (default: all
#      of WORK's), ITERS (200), GPUS (all), ROTATE (1; must match the target bake's).
# Needs the cards free (two engine models with drafters + the tuner). Build first: scripts/build-host.sh
set -euo pipefail
cd "$(dirname "$0")/.."
SRC=${SRC:-$HOME/models/Qwen3.8-Flash-Next-Uncensored}
WORK=${WORK:-$HOME/models/.bake-work}
RTN_OUT=${RTN_OUT:-$HOME/models/qwen38-uncensored-w4a8-mtp}
OUT=${OUT:-$HOME/models/qwen38-uncensored-w4a8-mtp-ar}
ITERS=${ITERS:-200}
ROTATE=${ROTATE:-1}
BAKE=./build/xenithium-bake
ARGS=(--iters "$ITERS")
[[ -n "${GPUS:-}" ]] && ARGS+=(--gpus "$GPUS")
[[ -n "${SAMPLES:-}" ]] && ARGS+=(--samples "$SAMPLES")
log() { echo "[bake-mtp $(date +%H:%M:%S)] $*"; }

# 1. the drafter directory with RTN experts (host only, ~10 s)
if [[ ! -f "$RTN_OUT/mtp-experts.safetensors" ]]; then
  log "drafter (RTN experts, int2 head) -> $RTN_OUT"
  $BAKE mtp --src "$SRC" --out "$RTN_OUT" --rotate "$ROTATE"
fi
# 2. AutoRound of its experts on the quantised target's hiddens (~10 min at 512 samples)
if [[ ! -f "$OUT/mtp-experts.safetensors" ]]; then
  if [[ -n "$(docker ps -q 2>/dev/null)" ]] || pgrep -x xenithium-probe >/dev/null || pgrep -x xenithium-serve >/dev/null || pgrep -x xenithium-bake >/dev/null; then
    echo "cards busy (docker ps / xenithium-probe / xenithium-serve / xenithium-bake): not starting" >&2
    exit 1
  fi
  log "mtp-tune: $RTN_OUT + $WORK -> $OUT"
  $BAKE mtp-tune --src "$SRC" --mtp "$RTN_OUT" --work "$WORK" --out "$OUT" "${ARGS[@]}"
fi
log "done: $OUT (ModelOptions::mtp_dir; server: XEN_MTP)"
