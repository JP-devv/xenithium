#!/usr/bin/env bash
# Optional: the server records the hot-expert ranking itself while serving (XEN_HOT_RANK=auto, nothing needed
# before the first start). This script seeds or replaces that file with a calibration run instead: it loads a MoE
# checkpoint (qwen38 Flash-Next) on both cards, runs the chat prompts of
# scripts/data/hot-rank-prompts.jsonl (greedy decode), counts which experts every decode step routes to and writes the
# counts as an XRC1 file (it overwrites what the server recorded). The launcher runs this from its Experts tab.
#
#   scripts/make-hot-rank.sh [--model DIR] [--out FILE] [--prompts F.jsonl] [--gen N]
#
# --model   default: XEN_MODEL of SERVE_ENV / ./server.env, else ~/models/qwen38-w4a16
# --out     default: .cache/experts/<model folder>/hot-rank.xrc in this repo, where XEN_HOT_RANK=auto looks
# --prompts default: scripts/data/hot-rank-prompts.jsonl (40 chats: chat, code, maths, writing, agents, 6 languages);
#           your own: one OpenAI request per line, {"id", "messages", "max_tokens"}
# --gen     decode tokens per prompt at most (default 192)
# Needs build/xenithium-probe (scripts/build-host.sh xenithium-probe) and both cards free: no xenithium server or probe
# running, no Docker container. Takes the model's load time plus ~1-3 minutes of decoding.
set -euo pipefail
cd "$(dirname "$0")/.."
model="" out="" prompts=scripts/data/hot-rank-prompts.jsonl gen=192
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) model=$2; shift 2 ;;
    --out) out=$2; shift 2 ;;
    --prompts) prompts=$2; shift 2 ;;
    --gen) gen=$2; shift 2 ;;
    -h|--help) sed -n '2,17p' "$0"; exit 0 ;;
    *) echo "make-hot-rank: unknown option $1 (--help)" >&2; exit 2 ;;
  esac
done
if [[ -z "$model" ]]; then
  env_file=${SERVE_ENV:-server.env}
  [[ -f "$env_file" ]] && model=$(set +u; set -a; source "$env_file" >/dev/null 2>&1; echo "${XEN_MODEL:-}")
  model=${model:-$HOME/models/qwen38-w4a16}
fi
model=${model/#\~/$HOME}
model=${model%/}
name=$(basename "$model")
out=${out:-$PWD/.cache/experts/$name/hot-rank.xrc}
out=${out/#\~/$HOME}

[[ -f "$model/config.json" ]] || { echo "make-hot-rank: $model has no config.json" >&2; exit 1; }
if ! grep -qE '"model_type": *"qwen4_exp"' "$model/config.json"; then
  echo "make-hot-rank: $name is not a MoE model with a hot-expert placement (qwen38): nothing to rank" >&2
  exit 1
fi
[[ -f "$prompts" ]] || { echo "make-hot-rank: prompts $prompts not found" >&2; exit 1; }
[[ -x build/xenithium-probe ]] || { echo "make-hot-rank: build/xenithium-probe missing; run scripts/build-host.sh xenithium-probe" >&2; exit 1; }
if ldd build/xenithium-probe 2>/dev/null | grep -q 'not found'; then
  echo "make-hot-rank: build/xenithium-probe cannot find the oneAPI runtime; source it first (e.g. source /opt/intel/oneapi/2026.1/oneapi-vars.sh)" >&2
  exit 1
fi
# one model load at a time: a running server, probe or bake holds the cards (and two loads exhaust host memory)
for p in xenithium-serve xenithium-probe xenithium-bake; do
  if pid=$(pgrep -x "$p" | head -1) && [[ -n "$pid" ]]; then
    echo "make-hot-rank: $p (pid $pid) holds the cards; stop it first" >&2
    exit 1
  fi
done
others=$(docker ps --format '{{.Label "org.xenithium"}}	{{.Names}}' 2>/dev/null | awk -F'\t' '$1 != "1" {print $2}' | tr '\n' ' ')
if [[ -n "$others" ]]; then  # this project's own containers (label org.xenithium) are not counted (start-server.sh)
  echo "make-hot-rank: containers hold the cards ($others); stop them first" >&2
  exit 1
fi
echo "make-hot-rank: $name -> $out"
echo "make-hot-rank: $(grep -c . "$prompts") prompts from $prompts, up to $gen tokens each"
exec ./build/xenithium-probe rank --model "$model" --prompts "$prompts" --out "$out" --gen "$gen"
