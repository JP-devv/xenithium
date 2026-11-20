#!/usr/bin/env bash
# Sets the power limit of the Intel GPUs with xpu-smi (card power domain, "burst" = the driver's
# power1_cap). xpu-smi needs root to write (ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS otherwise), so
# the write runs through sudo.
#   scripts/gpu-power.sh            limit every GPU to 220 W
#   scripts/gpu-power.sh 250        limit every GPU to 250 W
#   scripts/gpu-power.sh status     limit and power now of every GPU (no root)
# The limit is lost on reboot or driver reload; run the script again afterwards.
set -euo pipefail
WATTS=${GPU_POWER_W:-220}
arg=${1:-}
[[ -n "$arg" && "$arg" != status ]] && WATTS=$arg
command -v xpu-smi >/dev/null || { echo "gpu-power: xpu-smi not found" >&2; exit 1; }

# Device ids from the discovery table: lines like "  0   Device Name: ..."
mapfile -t ids < <(xpu-smi discovery | awk '$1 ~ /^[0-9]+$/ && $2 == "Device" {print $1}')
((${#ids[@]})) || { echo "gpu-power: xpu-smi lists no GPU" >&2; exit 1; }

# The card power domain's limits in W from `xpu-smi config -d N`: "burst" or "peak".
limit() { xpu-smi config -d "$1" | awk -v t="$2" '/Power domain card/ {c=1} /Power domain package/ {c=0} c && $1 == t"(w):" {print $2; exit}'; }

# Power now (W) from the board's energy counter over one second; the xe hwmon files, since xpu-smi
# reports N/A for power on this card. "?" when the files are not there.
now_w() {
  local h e0 e1
  for h in /sys/class/drm/card*/device/hwmon/hwmon*; do
    [[ "$(cat "$h/name" 2>/dev/null)" == xe ]] || continue
    [[ "$(basename "$(readlink -f "${h%/hwmon/*}")")" == "$1" ]] || continue
    e0=$(cat "$h/energy1_input"); sleep 1; e1=$(cat "$h/energy1_input")
    echo $(((e1 - e0) / 1000000)); return
  done
  echo "?"
}

show() {
  local i bdf
  for i in "${ids[@]}"; do
    bdf=$(xpu-smi discovery -d "$i" 2>/dev/null | awk '/PCI BDF Address/ {print $NF}')
    printf 'GPU %s (%s)  limit %s W  peak %s W  now %s W\n' "$i" "$bdf" "$(limit "$i" burst)" "$(limit "$i" peak)" "$(now_w "$bdf")"
  done
}

if [[ "$arg" == status ]]; then show; exit 0; fi

[[ "$WATTS" =~ ^[0-9]+$ ]] || { echo "gpu-power: '$WATTS' is not a whole number of watts" >&2; exit 2; }
for i in "${ids[@]}"; do
  peak=$(limit "$i" peak)
  [[ "$peak" =~ ^[0-9]+$ ]] || continue
  ((WATTS >= 50 && WATTS <= peak)) || { echo "gpu-power: $WATTS W is outside 50..$peak W (GPU $i)" >&2; exit 2; }
done

echo "gpu-power: setting ${#ids[@]} GPU(s) to $WATTS W (sudo xpu-smi)"
for i in "${ids[@]}"; do
  sudo xpu-smi config -d "$i" --powerlimit "$WATTS" --powertype burst
done
rc=0
for i in "${ids[@]}"; do
  got=$(limit "$i" burst)
  [[ "$got" == "$WATTS" ]] || { echo "gpu-power: GPU $i reports $got W, not $WATTS W" >&2; rc=1; }
done
show
exit $rc
