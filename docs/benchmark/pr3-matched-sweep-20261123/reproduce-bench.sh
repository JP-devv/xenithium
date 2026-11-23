#!/usr/bin/env bash
# Invoke only against the matching isolated server in an authorized GPU window.
set -euo pipefail
case_name=${1:?usage: bash reproduce-bench.sh short\|middle\|near base\|dense\|trained\|xmx}
arm=${2:?usage: bash reproduce-bench.sh short\|middle\|near base\|dense\|trained\|xmx}
: "${RUNNER:?set RUNNER to the pinned fixed-fixture xenithium-bench}"
: "${FIXTURE_DIR:?set FIXTURE_DIR to the pinned request fixtures}"
: "${OUTPUT_ROOT:?set OUTPUT_ROOT to a new writable results directory}"
: "${ONEAPI_ENV:?set ONEAPI_ENV to the pinned oneapi-env.sh}"
case "$arm" in base|dense|trained|xmx) ;; *) echo "invalid arm: $arm" >&2; exit 2;; esac
case "$case_name" in
  short) depth=8007; request_sha=e74a5962a9b93630b537ce11766dc20d01342e5ab783d2b5a0d7ea09058ce64f ;;
  middle) depth=36864; request_sha=fc2d6f26958d19d9e31b060e0eb50e51d9395038752035df34eba901bc54ef8e ;;
  near) depth=176073; request_sha=5a626fb8c27c1fcf1d9c5548348e2829b9f4eaf0df8bb6a45c61f02b8c64be8c ;;
  *) echo "invalid case: $case_name" >&2; exit 2 ;;
esac
printf '%s  %s\n' 44770d1b1aa2520dcbf9b28c7501ad59c381b667cdf55fa885bb93ba61038da3 "$ONEAPI_ENV" | sha256sum -c -
printf '%s  %s\n' 568c219dd4d961c2a6875ecf624520d92bb0e7eeb3bf580ffaf9d7c717778538 "$RUNNER" | sha256sum -c -
printf '%s  %s\n' "$request_sha" "$FIXTURE_DIR/$case_name.request.json" | sha256sum -c -
export XEN_ONEAPI=/opt/intel/oneapi/2025.3
source "$ONEAPI_ENV"
exec "$RUNNER" run --url http://127.0.0.1:18102/v1 --model swift15-baked --scenario burst \
  --set "burst.fixture_file=$FIXTURE_DIR/$case_name.request.json" --context 177216 \
  --set "burst.depth=$depth" --set burst.requests=1 --set burst.reply_tokens=128 \
  --set "burst.fixture_prompt_tokens=$depth" --set burst.fixture_reserve_tokens=256 \
  --thinking off --power off --name "$case_name-$arm" --out "$OUTPUT_ROOT/$case_name/$arm/results"
