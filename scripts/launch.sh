#!/usr/bin/env bash
# Starts the interactive launcher (build it first: scripts/build-host.sh xenithium-launch).
# Pick a model, edit the server.env settings, launch xenithium-server and watch it; docs/launcher.md.
# Arguments go to the launcher (--env FILE, --models DIR, --check, --help).
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ ! -x build/xenithium-launch ]]; then
  echo "launch: build/xenithium-launch missing; run scripts/build-host.sh xenithium-launch" >&2
  exit 1
fi
exec ./build/xenithium-launch "$@"
