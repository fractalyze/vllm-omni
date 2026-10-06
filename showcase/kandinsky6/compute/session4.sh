#!/usr/bin/env bash
# The 2x2 this study's baseline depends on.
#
# vLLM-Omni compiles the DiT's repeated blocks by default
# (`enforce_eager=False`, `diffusion_compile_granularity=regional`), and
# Kandinsky6FusedTransformerDecoderBlock is in the model's `_repeated_blocks`.
# So `default@default` -- platform attention, compiled -- is what upstream
# actually serves, and an eager measurement is not the shipped baseline. This
# times all four corners in ABBA order in one process so every ratio is
# same-session.
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

run k6c-b01-baseline-2x2 "$PY" "$HERE/block_profile.py" \
    --config pro --geometry w1 --target fused \
    --compare-arm "shipped=default@default" \
    --compare-arm "shipped-eager=default@eager" \
    --compare-arm "tuned=$HERE/arms/tuned.json@default" \
    --compare-arm "tuned-max-autotune=$HERE/arms/tuned.json@max-autotune" \
    --repeats 3 --rounds 2 --warmups 3 --profile-iters 0 \
    --json "$OUT/k6c-b01-baseline-2x2-w1.json"

# The split of the fastest arm, so what is left is named rather than guessed.
run k6c-g01b-split-max-autotune "$PY" "$HERE/block_profile.py" \
    --config pro --geometry w1 --target fused \
    --attention-config "$HERE/arms/tuned.json" \
    --compile max-autotune --repeats 5 --warmups 3 --profile-iters 3 \
    --json "$OUT/k6c-g01b-split-w1-tuned-max-autotune.json"

echo "=== $(date +%H:%M:%S) session done, $failures failure(s)"
exit "$failures"
