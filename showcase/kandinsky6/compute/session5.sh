#!/usr/bin/env bash
# Does the fast arm compute the right answer?
#
# k6c-g01's max-autotune block came out at 134.70 ms, and its profile contains
# no kernel capable of a 50,220-token self-attention: 41.32 TFLOP cannot be
# done in the 4.35 ms the attention category got, at any rate this GPU has.
# Either the profiler is hiding work inside a CUDA graph, or the arm is not
# doing the work. The device total matches the wall time, which rules out
# hidden work -- so the number is suspect until the outputs are compared.
#
# `reduce-overhead` is included because it also enables CUDA graphs and logged
# "The CUDA Graph is empty" warnings during capture.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${K6_PYTHON:-/data/jooman/k6/venv/bin/python}"
OUT="${K6_RESULTS:-/data/jooman/k6/results}"
failures=0
run() {
    local name="$1"; shift
    echo "=== $(date +%H:%M:%S) $name"
    if "$@" --no-locks; then echo "=== $(date +%H:%M:%S) $name ok"
    else echo "=== $(date +%H:%M:%S) $name FAILED (exit $?)"; failures=$((failures + 1)); fi
}

run k6c-g01e-numerics "$PY" "$HERE/block_profile.py" \
    --config pro --geometry w1 --target fused \
    --compare-arm "tuned-eager=$HERE/arms/tuned.json@eager" \
    --compare-arm "tuned-default=$HERE/arms/tuned.json@default" \
    --compare-arm "tuned-reduce-overhead=$HERE/arms/tuned.json@reduce-overhead" \
    --compare-arm "tuned-max-autotune=$HERE/arms/tuned.json@max-autotune" \
    --check-numerics --repeats 3 --rounds 2 --warmups 3 --profile-iters 0 \
    --json "$OUT/k6c-g01e-numerics-w1.json"

echo "=== $(date +%H:%M:%S) session done, $failures failure(s)"
exit "$failures"
