#!/usr/bin/env bash
# The gate-passing candidate, if there is one: exact BF16 weights streamed from
# NVMe (the reference's own weight path) with Sage2 attention on the two visual
# roles. It isolates the question the FP8 arms cannot answer -- what does
# attention alone cost in quality? -- because everything except the attention
# config is the reference server's configuration, including compile settings,
# which are left at the platform default exactly as the reference left them.
#
# Order: the Sage2 arm first, because it is the deliverable and the GPU is free
# now. Then two prompts of the control (the reference configuration, served by
# me in a fresh process) as this host's determinism floor against Track M's
# reference: without it a small Sage2 number cannot be told from autotune
# noise, and a large one cannot be blamed on the kernel.
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

run bf16-stream-sage2 "$PY" "$HERE/gate_pro.py" \
    --arm "$HERE/arms/tuned.json" --name bf16-stream-sage2 --offload dlo-mmap \
    --no-locks --out-dir "$OUT/bf16-stream-sage2"

run bf16-stream-control "$PY" "$HERE/gate_pro.py" \
    --arm shipped --name bf16-stream-control --offload dlo-mmap \
    --no-locks --limit 2 --out-dir "$OUT/bf16-stream-control"

echo "=== $(date +%H:%M:%S) session done, $failures failure(s)"
exit "$failures"
