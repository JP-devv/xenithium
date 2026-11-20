#!/usr/bin/env bash
# `docker compose` for this project (compose.yaml) with the host's ids and folders filled in:
#   scripts/docker.sh build | up -d server | logs -f server | down | run --rm launch | run --rm tools CMD ...
# Sets XEN_UID / XEN_GID (you), XEN_RENDER_GID / XEN_VIDEO_GID (the groups of /dev/dri/renderD*), XEN_DOCKER_GID (the
# docker socket's group), and mounts the folders of XEN_MODELS_DIR and XEN_MODEL (from the settings file and the model's
# settings file) when they lie outside $HOME (compose.yaml mounts $HOME itself). A variable already set in the shell
# wins. Settings file: XEN_SETTINGS (a path inside the container, /work/... = this folder) or ./server.env.
set -euo pipefail
cd "$(dirname "$0")/.."
export XEN_UID=${XEN_UID:-$(id -u)} XEN_GID=${XEN_GID:-$(id -g)}
render=$(ls /dev/dri/renderD* 2>/dev/null | head -1 || true)
export XEN_RENDER_GID=${XEN_RENDER_GID:-$( [[ -n "$render" ]] && stat -c %g "$render" || getent group render | cut -d: -f3)}
export XEN_VIDEO_GID=${XEN_VIDEO_GID:-$(getent group video | cut -d: -f3)}
export XEN_DOCKER_GID=${XEN_DOCKER_GID:-$(stat -c %g /var/run/docker.sock 2>/dev/null || getent group docker | cut -d: -f3)}
# folders the containers write to: created here, so Docker does not create them owned by root
mkdir -p .cache build/launch

settings=server.env
[[ -n "${XEN_SETTINGS:-}" ]] && settings=${XEN_SETTINGS/#\/work\//}
# a key's value from env files (later files win), $HOME / ~ expanded
read_key() {
  local key=$1; shift
  env -i HOME="$HOME" bash -c 'k=$1; shift; for f; do [[ -f "$f" ]] && { set -a; source "$f" >/dev/null 2>&1; set +a; }; done
    v=${!k-}; printf "%s" "${v/#\~/$HOME}"' _ "$key" "$@"
}
model_env=$(read_key XEN_MODEL_ENV "$settings")
files=()
if [[ -n "$model_env" && "$model_env" != auto ]]; then
  [[ "$model_env" == */* || "$model_env" == *.env ]] || model_env=models/$model_env.env
  if [[ -f "$model_env" ]]; then files+=("$model_env"); else files+=("$(dirname "$model_env")/presets/$(basename "$model_env")"); fi
fi
files+=("$settings")
dirs=()
for key in XEN_MODELS_DIR XEN_MODEL XEN_DRAFTER_MODEL; do
  v=${!key:-$(read_key "$key" "${files[@]}")}
  [[ -z "$v" ]] && continue
  [[ $key == XEN_MODELS_DIR ]] || v=$(dirname "$v")
  v=$(readlink -f "$v" 2>/dev/null || echo "$v")
  [[ -d "$v" && "$v/" != "$HOME/"* ]] && dirs+=("$v")
done
compose=(docker compose -f compose.yaml)
if ((${#dirs[@]})); then
  over=.cache/compose.mounts.yaml
  {
    echo "# written by scripts/docker.sh: checkpoint folders outside \$HOME"
    echo "services:"
    for s in server launch tools; do
      echo "  $s:"
      echo "    volumes:"
      printf '%s\n' "${dirs[@]}" | sort -u | while read -r d; do echo "      - \"$d:$d\""; done
    done
  } > "$over"
  compose+=(-f "$over")
fi
exec "${compose[@]}" "$@"
