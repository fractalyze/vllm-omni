#!/usr/bin/env bash
# The fastest arm worth gating, plus the scoring of every arm generated so far.
#
# On Pro against Track M's canonical BF16 reference, the tuned arm's first
# prompt scored LPIPS mean 0.0790 / max 0.1180 against limits of 0.15 / 0.25 --
# so there is error budget left, and the objective is the fastest config that
# passes rather than the safest one. sage3-tuned spends that budget: Sage3's
# kernel is 1.27x Sage2's on the dominant call.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${K6_PYTHON:-/data/jooman/k6/venv/bin/python}"
OUT="${K6_GATE_PRO:-/data/jooman/k6/results/gate-pro}"
SCORES="$OUT/scores"
failures=0
run() {
    local name="$1"; shift
    echo "=== $(date +%H:%M:%S) $name"
    if "$@"; then echo "=== $(date +%H:%M:%S) $name ok"; else
        status=$?; echo "=== $(date +%H:%M:%S) $name FAILED (exit $status)"; failures=$((failures + 1))
    fi
}

run gate-pro-sage3-tuned "$PY" "$HERE/gate_pro.py" \
    --arm "$HERE/arms/sage3-tuned.json" --name sage3-tuned --no-locks \
    --out-dir "$OUT/sage3-tuned"

# Score whatever generated, newest arms included. Scoring is GPU work, so it
# runs inside this session's lock rather than racing the next generation.
for arm in tuned sage3-tuned sage2-visual-only sage2-accurate lossless; do
    [ -f "$OUT/$arm/manifest.json" ] || continue
    run "score-pro-$arm" "$PY" "$HERE/gate_pro_score.py" \
        --arm-dir "$OUT/$arm" --no-locks --json "$SCORES/pro-a-$arm.json"
done

echo "=== $(date +%H:%M:%S) session done, $failures failure(s)"
exit "$failures"
