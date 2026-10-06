#!/usr/bin/env bash
# Is the port's NABLA block-sparse attention worth it on sm_120?
#
# Not measurable at W1: the fractal reordering patches the latent grid in 8x8
# tiles, so both patched latent dims must be multiples of 8, and W1's 30x54
# fail the reshape outright (so does the 512x320 smoke). --geometry nabla-ok
# is 1024x512, the nearest working geometry, at 63,488 visual tokens against
# W1's 50,220.
#
# Three arms at that geometry, in one window so the clocks match: the platform
# default, the Sage2 arm the showcase would serve, and NABLA at its usual
# threshold. The question is whether a sparse bf16 kernel can beat a dense FP8
# one -- Sage2 is already 2.52x cuDNN, so NABLA needs more than the 1.2-1.6x
# the recipe DB credits block-sparse with to be interesting at all.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${K6_PYTHON:-/data/jooman/k6/venv/bin/python}"
OUT="${K6_RESULTS:-/data/jooman/k6/results}"
failures=0
run() {
    local name="$1"; shift
    echo "=== $(date +%H:%M:%S) $name"
    if "$@" --no-locks; then echo "=== $(date +%H:%M:%S) $name ok"; else
        status=$?; echo "=== $(date +%H:%M:%S) $name FAILED (exit $status)"; failures=$((failures + 1))
    fi
}

common=(--config pro --geometry nabla-ok --target fused --repeats 5 --warmups 2 --profile-iters 3)

run nabla-dense-default "$PY" "$HERE/block_profile.py" "${common[@]}" \
    --json "$OUT/k6c-s01-dense-default-nablaok.json"

run nabla-dense-sage2 "$PY" "$HERE/block_profile.py" "${common[@]}" \
    --attention-config "$HERE/arms/tuned.json" \
    --json "$OUT/k6c-s01-dense-sage2-nablaok.json"

run nabla-sparse-p09 "$PY" "$HERE/block_profile.py" "${common[@]}" --sparse 0.9 \
    --json "$OUT/k6c-s01-nabla-p09-nablaok.json"

echo "=== $(date +%H:%M:%S) session done, $failures failure(s)"
exit "$failures"
