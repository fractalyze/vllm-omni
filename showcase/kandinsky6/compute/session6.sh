#!/usr/bin/env bash
# Re-measure every block number with correct RoPE tables.
#
# make_inputs drew the RoPE table from `randn` until now. A RoPE table's 2x2
# blocks are [[cos, -sin], [sin, cos]], so apply_rotary is a rotation and
# preserves the per-head norm query_norm/key_norm just set; random entries make
# it an arbitrary linear map. The dense bf16 backends stayed finite under that,
# which is why it went unnoticed -- but SageAttention's per-block INT8 scale is
# set by the largest entry in a block, and the visual self-attention returned
# NaN at W1. Every Sage-arm block number taken before this fix is void.
#
# Order matters: the 2x2 with --check-numerics comes first, because until the
# arms are shown to compute the same thing the timings mean nothing.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${K6_PYTHON:-/data/jooman/k6/venv/bin/python}"
OUT="${K6_RESULTS:-/data/jooman/k6/results}"
TUNED="$HERE/arms/tuned.json"
failures=0
run() {
    local name="$1"; shift
    echo "=== $(date +%H:%M:%S) $name"
    if "$@" --no-locks; then echo "=== $(date +%H:%M:%S) $name ok"
    else echo "=== $(date +%H:%M:%S) $name FAILED (exit $?)"; failures=$((failures + 1)); fi
}

# The baseline 2x2 plus the two CUDA-graph modes, with the output check.
run k6c-b02-baseline-and-numerics "$PY" "$HERE/block_profile.py" \
    --config pro --geometry w1 --target fused \
    --compare-arm "shipped=default@default" \
    --compare-arm "shipped-eager=default@eager" \
    --compare-arm "tuned=$TUNED@default" \
    --compare-arm "tuned-reduce-overhead=$TUNED@reduce-overhead" \
    --compare-arm "tuned-max-autotune=$TUNED@max-autotune" \
    --check-numerics --repeats 3 --rounds 2 --warmups 3 --profile-iters 0 \
    --json "$OUT/k6c-b02-baseline-numerics-w1.json"

# The splits, one per stack, so what is left is named rather than guessed.
run k6c-p02-split-shipped "$PY" "$HERE/block_profile.py" \
    --config pro --geometry w1 --target fused --compile default \
    --repeats 5 --warmups 3 --profile-iters 3 \
    --json "$OUT/k6c-p02-split-w1-shipped.json"

run k6c-p03-split-tuned "$PY" "$HERE/block_profile.py" \
    --config pro --geometry w1 --target fused --attention-config "$TUNED" --compile default \
    --repeats 5 --warmups 3 --profile-iters 3 \
    --json "$OUT/k6c-p03-split-w1-tuned.json"

run k6c-p04-split-tuned-max "$PY" "$HERE/block_profile.py" \
    --config pro --geometry w1 --target fused --attention-config "$TUNED" --compile max-autotune \
    --repeats 5 --warmups 3 --profile-iters 3 \
    --json "$OUT/k6c-p04-split-w1-tuned-max-autotune.json"

echo "=== $(date +%H:%M:%S) session done, $failures failure(s)"
exit "$failures"
