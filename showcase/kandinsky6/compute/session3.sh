#!/usr/bin/env bash
# k6c-g01: after the fusion, the block is 64.1% GEMM on Ampere-generation
# CUTLASS kernels. This asks whether Inductor's GEMM autotuning or CUDA graphs
# beat mode=default, with all four arms timed in ABBA order in one process so
# the ratios are same-session.
#
# max-autotune compiles slowly (it benchmarks Triton GEMM templates against
# cuBLAS), so the warm-up count is raised and the whole thing is given its own
# GPU window.
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

run k6c-g01-compile-modes "$PY" "$HERE/block_profile.py" \
    --config pro --geometry w1 --target fused \
    --attention-config "$HERE/arms/tuned.json" \
    --compare-compile --compile default --compile reduce-overhead --compile max-autotune \
    --repeats 3 --rounds 2 --warmups 3 --profile-iters 0 \
    --json "$OUT/k6c-g01-compile-modes-w1.json"

echo "=== $(date +%H:%M:%S) session done, $failures failure(s)"
exit "$failures"
