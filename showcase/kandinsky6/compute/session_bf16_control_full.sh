#!/usr/bin/env bash
# The reference configuration, served here, scored against Track M's reference.
#
# Two things at once, and neither is optional for the write-up. It is this
# host's **determinism floor**: the same checkpoint, the same weight path, the
# same compile settings, the same seeds, differing only in the process and the
# machine. Every gate number in this showcase has to be read against it --
# a3-sprint-start measured 0.3277 with Sage2 and the only way to know how much
# of that is the kernel is to know what a rerun costs.
#
# And it is the **safe headline**: exact weights, exact attention, so it passes
# the user's gate by construction, and its request time is the speed an arm has
# to beat while staying inside the gate.
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

run bf16-stream-control-full "$PY" "$HERE/gate_pro.py" \
    --arm shipped --name bf16-stream-control-full --offload dlo-mmap \
    --no-locks --out-dir "$OUT/bf16-stream-control-full"

echo "=== $(date +%H:%M:%S) session done, $failures failure(s)"
exit "$failures"
