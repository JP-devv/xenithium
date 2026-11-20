#!/usr/bin/env bash
# MTP drafter calibration on a served checkpoint: a W4A16 export without a BF16 target to tune against. The
# drafter's routed experts (RTN g128 from its BF16 experts, `xenithium-bake mtp --experts-only 1`) AutoRound-tuned on the
# served target's states after its last layer, the BF16 drafter as the target.
#   scripts/bake-mtp-served.sh
# Env: SRC (the served checkpoint), BF16 (the drafter's BF16 experts: Intel's model_extra_tensors directory), RTN (the RTN
#      drafter), OUT (the tuned drafter: RTN's files hard-linked + the tuned experts), WORK (states: ~52 MB per sample),
#      CALIB (calibration file), SAMPLES (512), ITERS (200).
# Needs the cards free and ~45 GiB of host memory (mtp-states loads the served model alone). Build first: scripts/build-host.sh
set -euo pipefail
cd "$(dirname "$0")/.."
SRC=${SRC:-$HOME/models/qwen38-w4a16}
BF16=${BF16:-$HOME/models/qwen38-w4a16-mtp-bf16}
RTN=${RTN:-$HOME/models/qwen38-w4a16-mtp-g128}
OUT=${OUT:-$HOME/models/qwen38-w4a16-mtp-g128-ar}
WORK=${WORK:-$HOME/models/.mtp-work-w4a16}
CALIB=${CALIB:-$HOME/models/.bake-work/calib.xcb}
SAMPLES=${SAMPLES:-512}
ITERS=${ITERS:-200}
BAKE=./build/xenithium-bake
log() { echo "[bake-mtp-served $(date +%H:%M:%S)] $*"; }
if [[ -n "$(docker ps -q 2>/dev/null)" ]] || pgrep -x xenithium-probe >/dev/null || pgrep -x xenithium-serve >/dev/null || pgrep -x xenithium-bake >/dev/null; then
  echo "cards busy (docker ps / xenithium-probe / xenithium-serve / xenithium-bake): not starting" >&2
  exit 1
fi
if [[ ! -f "$RTN/mtp-experts.safetensors" ]]; then
  log "RTN drafter experts -> $RTN"
  $BAKE mtp --src "$BF16" --out "$RTN" --rotate 0 --experts-only 1
fi
[[ -f "$RTN/mtp-dense.safetensors" ]] || { echo "$RTN has no mtp-dense.safetensors: hard-link the served drafter's (its trunk)" >&2; exit 1; }
if [[ ! -f "$WORK/progress" ]]; then
  log "the served target's states: $SAMPLES samples -> $WORK (~5 min)"
  $BAKE mtp-states --src "$SRC" --work "$WORK" --calib "$CALIB" --samples "$SAMPLES" --eval 16
fi
log "mtp-tune --served 1: $RTN + $WORK -> $OUT (~4 min)"
$BAKE mtp-tune --served 1 --src "$SRC" --mtp "$RTN" --bf16-experts "$BF16" --work "$WORK" --out "$OUT" --samples "$SAMPLES" --eval 16 --iters "$ITERS"
log "done: $OUT (server: XEN_MTP)"
