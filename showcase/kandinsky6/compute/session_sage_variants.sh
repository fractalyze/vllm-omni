#!/usr/bin/env bash
# Which SageAttention variant can reach the adoption gate?
#
# The registered SAGE_ATTN backend calls the top-level dispatcher, which on
# sm_120 picks qk_int8_pv_fp8_cuda with per-warp INT8 granularity and
# fp32+fp16 PV accumulation -- the fastest variant, and the one that failed the
# gate at set max LPIPS 0.3745 against a 0.25 limit. These are the knobs that
# trade the error back, raced at W1's visual self-attention shape against the
# dense cuDNN kernel and an fp32 reference.
#
# Kernel-level first on purpose: an image-level gate run costs about 25 minutes
# of GPU per arm, and this says in two minutes which arms are worth one.
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

run sage-variants-visual-self "$PY" "$HERE/attn_race.py" \
    --sage-variants --roles visual_self --repeats 3 --rounds 2 \
    --json "$OUT/k6c-v01-sage-variants-w1.json"

echo "=== $(date +%H:%M:%S) session done, $failures failure(s)"
exit "$failures"
