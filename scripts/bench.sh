#!/usr/bin/env bash
# xenithium-bench (docs/bench.md) against a fresh xenithium server, or against any other endpoint.
#
#   scripts/bench.sh                       standard suite on a server started here from server.env (~18 min)
#   scripts/bench.sh dspine                the run's name suffix (default: default)
#   BENCH_ARGS="--suite long" scripts/bench.sh long      another suite, scenarios, --set scenario.key=value, --seed N
#   URL=http://192.0.2.10:8000/v1 scripts/bench.sh    an endpoint that already runs: nothing is started or stopped
#   URL=https://api.example.com/v1 MODEL=some-model XEN_BENCH_API_KEY=... scripts/bench.sh other
#
# With no URL: checks the cards are free, starts the server (scripts/start-server.sh, so server.env and exported XEN_*
# apply), waits for /v1/models, runs the benchmark with every effective server flag and the host recorded as notes
# (the tool samples the cards' board power itself: the server is on this machine), and stops the server
# (KEEP_SERVER=1 leaves it up).
# Results: ~/.xenithium/bench/runs/<time>-xenithium-<name>/ (results.json, report.txt, report.html); logs in
# build/bench/<time>-<name>/.
# Exit code: the benchmark's (0 = every request answered, 1 = some failed, 2 = setup), 3 = the cards are busy.
set -uo pipefail
cd "$(dirname "$0")/.."
NAME=${1:-default}
BENCH=${BENCH:-build/xenithium-bench}
read -ra BENCH_EXTRA <<<"${BENCH_ARGS:---suite standard}"
TS=$(date +%Y%m%d-%H%M%S)
OUT=build/bench/$TS-$NAME
SLOG=$OUT/server.log BLOG=$OUT/bench.log
# seconds to wait for the server's /v1/models (a load with the warm-up and an expert cache build stays under it)
readonly SERVER_START_TIMEOUT_S=720

[[ -x $BENCH ]] || { echo "bench: $BENCH missing (scripts/build-host.sh xenithium-bench, or BENCH=path)" >&2; exit 2; }
mkdir -p "$OUT"

NOTES=()
note() { NOTES+=(--note "$1=$2"); echo "  $1=$2"; }

# ---- another endpoint: run and leave
if [[ -n ${URL:-} ]]; then
  echo "== xenithium-bench $TS ($NAME) against $URL"
  "$BENCH" run --url "$URL" ${MODEL:+--model "$MODEL"} --name "$NAME" "${BENCH_EXTRA[@]}" 2>&1 | tee "$BLOG"
  exit "${PIPESTATUS[0]}"
fi

# ---- a fresh server of this tree
[[ -x build/xenithium-server ]] || { echo "bench: build/xenithium-server missing (scripts/build-host.sh xenithium-server)" >&2; exit 2; }
# the cards must be free: no container, no other xenithium process
if [[ -n "$(docker ps -q 2>/dev/null)" ]]; then echo "bench: containers hold the cards: $(docker ps --format '{{.Names}}' | tr '\n' ' ')" >&2; exit 3; fi
if ps -eo comm | grep -qE '^xenithium-(serv|prob|bake)'; then echo "bench: a xenithium server / probe / bake is running" >&2; exit 3; fi

# every effective server flag, from the start script's dry run
read -ra CMD <<<"$(SERVE_DRY_RUN=1 scripts/start-server.sh)"
flag() { local k; for ((k = 1; k < ${#CMD[@]}; k++)); do [[ ${CMD[k]} == "$1" ]] && { echo "${CMD[k+1]}"; return; }; done; }
PORT=$(flag --port) SERVED=$(flag --served-model-name)
SERVER_URL=http://127.0.0.1:$PORT/v1

echo "== xenithium-bench $TS ($NAME)"
note engine xenithium-beta
note binary-sha256 "$(sha256sum build/xenithium-server | cut -c1-16)"
note binary-built "$(stat -c %y build/xenithium-server | cut -d. -f1)"
note repo-commit "$(git rev-parse --short HEAD 2>/dev/null || echo none) ($(git status --porcelain 2>/dev/null | wc -l) uncommitted paths)"
note server-env "$([[ -f ${SERVE_ENV:-server.env} ]] && echo "${SERVE_ENV:-server.env}" || echo defaults) + exported XEN_*: $(env | grep -o '^XEN_[A-Z_0-9]*' | tr '\n' ' ')"
for ((k = 1; k < ${#CMD[@]}; k++)); do
  [[ ${CMD[k]} == --* ]] || continue
  v=true; [[ $((k + 1)) -lt ${#CMD[@]} && ${CMD[k+1]} != --* ]] && v=${CMD[k+1]}
  note "srv${CMD[k]#-}" "$v"
done
note gpus "$(xpu-smi discovery 2>/dev/null | grep -c 'Device Name') x $(xpu-smi discovery 2>/dev/null | sed -n 's/.*Device Name: *//p' | head -1)"
i=0; for f in /sys/class/drm/card*/device/hwmon/hwmon*/power1_cap; do
  [[ -f $f ]] && { note "gpu$i-power1_cap-W" "$(($(cat "$f") / 1000000))"; i=$((i + 1)); }; done
note cpu "$(lscpu | sed -n 's/^Model name: *//p')"
note kernel "$(uname -r)"

scripts/start-server.sh >"$SLOG" 2>&1 &
SPID=$!
stop_server() {
  [[ ${KEEP_SERVER:-0} == 1 ]] && { echo "== server left running (pid $(pgrep -x xenithium-serve | head -1))"; return; }
  local p; p=$(pgrep -x xenithium-serve | head -1)
  [[ -n $p ]] && kill -INT "$p" 2>/dev/null
  for _ in $(seq 30); do pgrep -x xenithium-serve >/dev/null || break; sleep 1; done
}
trap stop_server EXIT
echo "server starting (log $SLOG)"
for _ in $(seq 1 $((SERVER_START_TIMEOUT_S / 2))); do
  curl -sf "$SERVER_URL/models" 2>/dev/null | grep -q "\"$SERVED\"" && break
  kill -0 $SPID 2>/dev/null || { echo "server died"; tail -30 "$SLOG"; exit 2; }
  sleep 2
done
curl -sf "$SERVER_URL/models" | grep -q "\"$SERVED\"" || { echo "server not up after $SERVER_START_TIMEOUT_S s"; exit 2; }
note server-load "$(grep -m1 'model loaded' "$SLOG" | sed 's/^server: //')"
note kv-pool "$(grep -m1 'KV pool sized' "$SLOG" | sed 's/^server: //')"

"$BENCH" run --url "$SERVER_URL" --model "$SERVED" --name "xenithium-$NAME" "${BENCH_EXTRA[@]}" "${NOTES[@]}" 2>&1 | tee "$BLOG"
exit "${PIPESTATUS[0]}"
