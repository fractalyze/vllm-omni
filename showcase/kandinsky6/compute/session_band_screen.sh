#!/usr/bin/env bash
# How wide should the exact band at the ends of the stack be?
#
# The schedule is a dial, not a switch: `AttentionSpec.layers` picks how many
# blocks at each end keep exact attention, and the two ends of the dial are
# measured -- Sage2 on all 60 blocks is 165.9 s and misses the floor-relative
# gate by 2%, the platform default is 231.8 s and is the reference. The band is
# the only free parameter left, so this screens two settings either side of the
# first one on the two prompts that separate arms.
#
#   sage2-wide    blocks 3-56 approximate, 6 exact   -- fastest, least accurate
#   sage2-mid     blocks 6-53 approximate, 12 exact  -- already screened
#   sage2-narrow  blocks 12-47 approximate, 24 exact -- slowest, most accurate
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

for band in wide narrow; do
    run "sage2-$band-screen" "$PY" "$HERE/gate_pro.py" \
        --arm "$HERE/arms/sage2-$band.json" --name "sage2-$band-screen" --offload dlo-mmap \
        --limit 2 --no-warmup --no-locks --out-dir "$OUT/sage2-$band-screen"
done

echo "=== $(date +%H:%M:%S) session done, $failures failure(s)"
exit "$failures"
