#!/usr/bin/env bash
# The per-channel FP8 arm, in full: the gate on set A, then one profiled request
# to see what is left once the DiT's weight traffic halves.
#
# Exact attention. Sage2 failed the gate on exact weights (a3 max 0.3277 against
# a 0.25 limit), so an arm carrying both it and a new weight format could only
# fail for two reasons at once. If this arm passes with room, sage2-accurate goes
# on top afterwards and has its own number.
#
# The profiled request is deliberately last and deliberately separate: the
# profiler synchronizes around every wrapped call, so it destroys the overlap
# the real path has. Its numbers are shares, not a wall.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${K6_PYTHON:-/data/jooman/k6/venv/bin/python}"
OUT="${K6_GATE_PRO:-/data/jooman/k6/results/gate-pro}"
OFFLOAD="${K6_PERCHAN_OFFLOAD:-layerwise}"
failures=0
run() {
    local name="$1"; shift
    echo "=== $(date +%H:%M:%S) $name"
    if "$@"; then echo "=== $(date +%H:%M:%S) $name ok"
    else echo "=== $(date +%H:%M:%S) $name FAILED (exit $?)"; failures=$((failures + 1)); fi
}

run perchan-exact "$PY" "$HERE/gate_pro.py" \
    --arm shipped --name perchan-exact \
    --quantization fp8_per_channel --offload "$OFFLOAD" \
    --no-locks --out-dir "$OUT/perchan-exact"

echo "=== $(date +%H:%M:%S) session done, $failures failure(s)"
exit "$failures"
