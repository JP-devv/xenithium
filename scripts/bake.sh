#!/usr/bin/env bash
# Offline bake: BF16 checkpoint -> W4A8 AutoRound experts + int8 PLE, one model directory.
#   scripts/bake.sh                    # every stage; re-run to resume (finished stages are skipped)
# Env: SRC (BF16 source), OUT (the baked model), WORK (calibration states, ~54 GB at 512 x 2,048),
#      SAMPLES (512), ITERS (200), GPUS (all), SEQLEN (2048), ROTATE (1: experts in a block-Hadamard
#      basis; a WORK started with the other setting refuses to resume), HOLDOUT_SPREAD (1: the 16
#      held-out samples spread over every source; 0 = the file's last ones, i.e. one source; a fresh
#      WORK records it, a resume uses the recorded choice).
# Needs the cards free (the tuner and two engine models use both). Build first: scripts/build-host.sh
set -euo pipefail
cd "$(dirname "$0")/.."
SRC=${SRC:-$HOME/models/Qwen3.8-Flash-Next-Uncensored}
OUT=${OUT:-$HOME/models/qwen38-uncensored-w4a8}
WORK=${WORK:-$HOME/models/.bake-work}
SAMPLES=${SAMPLES:-512}
ITERS=${ITERS:-200}
SEQLEN=${SEQLEN:-2048}
ROTATE=${ROTATE:-1}
HOLDOUT_SPREAD=${HOLDOUT_SPREAD:-1}
BAKE=./build/xenithium-bake
GPU_ARG=(); [[ -n "${GPUS:-}" ]] && GPU_ARG=(--gpus "$GPUS")
mkdir -p "$OUT" "$WORK/ple"
log() { echo "[bake $(date +%H:%M:%S)] $*"; }

# 1. calibration set (token ids; ~15 s)
if [[ ! -f "$WORK/calib.xcb" ]]; then
  log "calibration set: $SAMPLES x $SEQLEN"
  $BAKE calib --model "$SRC" --out "$WORK/calib.xcb" --samples "$SAMPLES" --seqlen "$SEQLEN"
fi
# 2. PLE table -> int8 rows (host cores)
if [[ ! -f "$WORK/ple/.done" ]]; then
  log "PLE int8"
  $BAKE ple --src "$SRC" --out "$WORK/ple" && touch "$WORK/ple/.done"
fi
# 3. block-wise AutoRound of the routed experts, layer by layer (resumable)
LAYERS=$(python3 -c "import json;c=json.load(open('$SRC/config.json'));print(c.get('text_config',c)['num_hidden_layers'])")
done_l=-1; [[ -f "$WORK/progress" ]] && done_l=$(cat "$WORK/progress")
if (( done_l < LAYERS - 1 )); then
  RES=(--holdout-spread "$HOLDOUT_SPREAD"); (( done_l >= 0 )) && RES=(--resume 1)
  log "tune: layers $((done_l + 1))..$((LAYERS - 1)), $SAMPLES samples, $ITERS steps, rotate $ROTATE"
  $BAKE tune --src "$SRC" --calib "$WORK/calib.xcb" --ple "$WORK/ple" --out "$OUT" --work "$WORK" \
    --samples "$SAMPLES" --iters "$ITERS" --rotate "$ROTATE" "${GPU_ARG[@]}" "${RES[@]}"
fi
# 4. one model directory (trunk, PLE, config, tokenizer)
if [[ ! -f "$OUT/config.json" ]]; then
  log "assemble"
  $BAKE assemble --src "$SRC" --out "$OUT" --ple "$WORK/ple" --rotate "$ROTATE"
fi
log "done: $OUT (the calibration states in $WORK can go: rm -rf $WORK)"
