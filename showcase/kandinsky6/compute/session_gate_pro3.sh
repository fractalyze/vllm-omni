#!/usr/bin/env bash
# THE diagnostic: how much of the error is the FP8 weights, before any
# attention choice?
#
# The tuned arm failed set A hard on Pro -- set mean 0.2791 against a 0.15
# limit, max 0.5649 against 0.25, six of nine prompts over -- and that number
# is the WHOLE stack against BF16: FP8 weights and Sage2 attention together.
# If FP8 alone already spends the budget then no attention arm can pass and the
# lever is precision, which is Track M's. If FP8 alone is cheap, the attention
# arms are worth tuning and sage2-accurate is the one to beat.
#
# So the control arm goes first: FP8 weights with the platform's own cuDNN
# attention. It is the same server configuration as Track M's 173.6 s
# baseline, so its gate number is the baseline's quality as well as the
# diagnostic.
#
# Then sage2-accurate, because if attention is worth tuning it is the arm most
# likely to fit; then lossless, the one that should sit at the floor whatever
# else happens.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${K6_PYTHON:-/data/jooman/k6/venv/bin/python}"
OUT="${K6_GATE_PRO:-/data/jooman/k6/results/gate-pro}"
failures=0
run() {
    local name="$1"
    shift
    echo "=== $(date +%H:%M:%S) $name"
    if "$@"; then
        echo "=== $(date +%H:%M:%S) $name ok"
    else
        status=$?
        echo "=== $(date +%H:%M:%S) $name FAILED (exit $status)"
        failures=$((failures + 1))
    fi
}

run gate-pro-control-fp8 "$PY" "$HERE/gate_pro.py" \
    --arm shipped --name control-fp8 --no-locks --out-dir "$OUT/control-fp8"

run gate-pro-sage2-accurate "$PY" "$HERE/gate_pro.py" \
    --arm "$HERE/arms/sage2-accurate.json" --name sage2-accurate --no-locks \
    --out-dir "$OUT/sage2-accurate"

run gate-pro-lossless "$PY" "$HERE/gate_pro.py" \
    --arm "$HERE/arms/lossless.json" --name lossless --no-locks \
    --out-dir "$OUT/lossless"

echo "=== $(date +%H:%M:%S) session done, $failures failure(s)"
exit "$failures"
