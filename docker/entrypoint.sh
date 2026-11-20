#!/usr/bin/env bash
# The image's entry point: sets up the toolchain environment (scripts/oneapi-env.sh: oneAPI 2026.2 with its oneDNN),
# points scripts/start-server.sh and the launcher at the settings file mounted from the host, then runs the command
# (default scripts/start-server.sh). compose.yaml mounts the project folder at /work.
#   XEN_SETTINGS = the settings file (default /work/server.env; a missing file = start-server.sh's defaults)
set -euo pipefail
cd /opt/xenithium
export PATH=/opt/xenithium/build:$PATH  # every tool the image ships (xenithium-inspect, -bench, -launch, ...)
# shellcheck disable=SC1091
source scripts/oneapi-env.sh
settings=${XEN_SETTINGS:-/work/server.env}
if [[ -f "$settings" ]]; then
  export SERVE_ENV=$settings
elif [[ -n "${XEN_SETTINGS:-}" ]]; then
  echo "entrypoint: XEN_SETTINGS=$XEN_SETTINGS not found" >&2
  exit 1
fi
# the launcher takes the same file (--env), unless the caller names another
if [[ "${1:-}" == scripts/launch.sh && -n "${SERVE_ENV:-}" && " $* " != *" --env "* ]]; then
  set -- "$@" --env "$SERVE_ENV"
fi
exec "$@"
