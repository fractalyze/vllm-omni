#!/usr/bin/env bash
# Two screens that decide where the remaining GPU hours go.
#
# 1. Residency probe: is the streamed BF16 arm stream-bound or compute-bound?
#    `--resident-layers 16 --offload-text-encoder` removes sixteen 0.899 GiB
#    blocks from every step's stream and is numerically identical to the plain
#    arm, so a change in step time is the stream and nothing else. It decides
#    whether INT8-as-storage (offline quantize, dequantize after the H2D copy)
#    is worth two hours -- and if it wins, it is itself the saving, with no new
#    code and no new checkpoint.
#
# 2. The layer schedule: SageAttention2 through blocks 6-53 with the platform
#    default at both ends. Both Sage arms measured so far fail the user's gate
#    on the whole stack -- default Sage2 at 0.1682/0.3693 and the accuracy
#    variant worse on both screened prompts (a1 0.0563 against 0.0495, a2 0.1867
#    against 0.1551) while also being 193 s against 166 s. Nothing between "all
#    60 blocks approximate" and "none of them" had been measurable until
#    AttentionSpec.layers landed.
#
# Two prompts each, a1 and a2: a1 is the easiest prompt for every arm so far and
# a2 is the one that separates them. A two-prompt screen decides what to run, it
# never decides a gate -- the set is nine prompts and one of them told me the
# opposite of the truth earlier tonight.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${K6_PYTHON:-/data/jooman/k6/venv/bin/python}"
OUT="${K6_GATE_PRO:-/data/jooman/k6/results/gate-pro}"
failures=0
run() {
    local name="$1"; shift
    echo "=== $(date +%H:%M:%S) $name"
    if "$@"; then echo "=== $(date +%H:%M:%S) $name ok"
    else echo "=== $(date +%H:%M:%S) $name FAILED (exit $?)"; failures=$((failures + 1)); fi
}

( while true; do
      printf '%s ' "$(date +%H:%M:%S)"
      iostat -d nvme0n1 1 2 2>/dev/null | awk '/nvme0n1/{r=$3} END{printf "nvme_read_kBps=%s ", r}'
      nvidia-smi --query-gpu=memory.used --format=csv,noheader
      sleep 10
  done ) > "$OUT/residency-probe-io.log" 2>&1 &
sampler=$!
trap 'kill "$sampler" 2>/dev/null' EXIT

run residency-probe "$PY" "$HERE/gate_pro.py" \
    --arm "$HERE/arms/tuned.json" --name residency-probe --offload dlo-mmap \
    --resident-layers 16 --offload-text-encoder \
    --limit 2 --no-locks --out-dir "$OUT/residency-probe"

kill "$sampler" 2>/dev/null

run sage2-mid-screen "$PY" "$HERE/gate_pro.py" \
    --arm "$HERE/arms/sage2-mid.json" --name sage2-mid-screen --offload dlo-mmap \
    --limit 2 --no-locks --out-dir "$OUT/sage2-mid-screen"

echo "=== $(date +%H:%M:%S) session done, $failures failure(s)"
exit "$failures"
