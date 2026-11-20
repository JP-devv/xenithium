#!/usr/bin/env bash
# Installs the engine's system packages on Ubuntu 26.04 (the release it is built and tested on). Needs root (runs apt
# through sudo when not root). docker/Dockerfile runs the same script, so the image and a host install get the same set.
#   scripts/install-deps.sh            build + run: oneAPI 2026.2 DPC++ compiler, oneDNN, Level Zero, GPU driver, cmake, ffmpeg
#   scripts/install-deps.sh --runtime  run only: the SYCL runtime, oneDNN, Level Zero, GPU driver, ffmpeg (no compiler)
# Adds two apt sources when they are not configured yet (keys from docker/keys/, checked by sha256): Intel's oneAPI
# repository (compiler, SYCL runtime) and the kobuk-team intel-graphics PPA (Level Zero loader + GPU user-mode driver).
# oneDNN: oneAPI's package (3.13.2).
# ONEAPI_VERSION=2026.2.0-863 / DNNL_VERSION=2026.1.0-617 pin the compiler and oneDNN package versions.
set -euo pipefail
cd "$(dirname "$0")/.."
mode=build
case "${1:-}" in
  "") ;;
  --runtime) mode=runtime ;;
  -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
  *) echo "install-deps: unknown argument $1 (--runtime or nothing)" >&2; exit 1 ;;
esac
ONEAPI_VERSION=${ONEAPI_VERSION:-2026.2.0-863}
DNNL_VERSION=${DNNL_VERSION:-2026.1.0-617}
ONEAPI_KEY_SHA256=db932ba032a71f732dc415eef5f4f185fc7ae768487d7a357bc2772746b6c48c
GRAPHICS_KEY_SHA256=42502b2402f28f3ed5a66bef8c29618f16bdb6e8f5a706eeef1ddcae49c56955
sudo=""
[[ $(id -u) == 0 ]] || sudo=sudo
export DEBIAN_FRONTEND=noninteractive

. /etc/os-release
[[ "${VERSION_ID:-}" == 26.04 ]] || echo "install-deps: tested on Ubuntu 26.04, this is ${PRETTY_NAME:-unknown}; continuing" >&2

$sudo apt-get update -q
$sudo apt-get install -y -q --no-install-recommends ca-certificates gnupg

# A source already configured (any file under /etc/apt) is left alone: a second entry with another keyring is an apt error.
configured() { grep -rqs -- "$1" /etc/apt/sources.list /etc/apt/sources.list.d/; }
add_source() {  # key file, its sha256, keyring name, list name, deb line ({KEYRING} = the keyring path)
  local key=$1 sum=$2 ring=/usr/share/keyrings/$3.gpg list=/etc/apt/sources.list.d/$4.list line=$5
  echo "$sum  $key" | sha256sum -c --quiet - || { echo "install-deps: $key does not match its sha256" >&2; exit 1; }
  gpg --dearmor < "$key" | $sudo tee "$ring" > /dev/null
  echo "${line//\{KEYRING\}/$ring}" | $sudo tee "$list" > /dev/null
  echo "install-deps: added $list"
}
added=0
if ! configured apt.repos.intel.com/oneapi; then
  add_source docker/keys/oneapi.pub "$ONEAPI_KEY_SHA256" oneapi-archive-keyring oneapi \
    "deb [signed-by={KEYRING}] https://apt.repos.intel.com/oneapi all main"
  added=1
fi
if ! configured kobuk-team/intel-graphics; then
  add_source docker/keys/intel-graphics.asc "$GRAPHICS_KEY_SHA256" intel-graphics intel-graphics \
    "deb [signed-by={KEYRING}] https://ppa.launchpadcontent.net/kobuk-team/intel-graphics/ubuntu ${VERSION_CODENAME} main"
  added=1
fi
((added)) && $sudo apt-get update -q

# Level Zero loader + the GPU user-mode driver (the PPA keeps only its newest build, so these are not pinned);
# intel-opencl-icd: oneDNN builds its GPU kernels through the card's OpenCL device (without it every oneDNN GPU primitive
# fails: "OpenCL device not found"); hwloc / numa: the SYCL Level Zero adapter loads them; procps: pgrep
# (start-server.sh, the launcher); ffmpeg: video input.
common=(libze1 libze-intel-gpu1 intel-opencl-icd libhwloc15 libnuma1 procps curl ffmpeg)
if [[ $mode == runtime ]]; then
  pkgs=("intel-oneapi-compiler-dpcpp-cpp-runtime-2026.2=$ONEAPI_VERSION" "intel-oneapi-dnnl-2026.1=$DNNL_VERSION" "${common[@]}")
else
  pkgs=("intel-oneapi-compiler-dpcpp-cpp-2026.2=$ONEAPI_VERSION" "intel-oneapi-dnnl-devel-2026.1=$DNNL_VERSION" intel-ocloc libze-dev
        build-essential cmake ninja-build git pkg-config "${common[@]}")
fi
$sudo apt-get install -y -q --no-install-recommends "${pkgs[@]}"
echo "install-deps: $mode packages installed (oneAPI under /opt/intel/oneapi/2026.2; source scripts/oneapi-env.sh)"
