#!/usr/bin/env bash
# One GPU session: the step split under the tuned attention arm, re-taken with
# the fixed kernel-name table, and the same split after torch.compile.
#
# The first p01b run put 23.5% of the block in `unclassified` because the
# table was missing the names SageAttention's sm_120 path emits
# (`qk_int_sv_f8_attn_kernel` and its two quantization prologues). The numbers
# were right and the classification was wrong, so this re-takes it rather than
# re-labelling a recorded run by hand.
#
# The compiled split is what says *what* torch.compile fused: if the -20.75%
# came from the elementwise chain, that category must be the one that shrank.
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

run k6c-p01b2-split-tuned "$PY" "$HERE/block_profile.py" \
    --config pro --geometry w1 --target fused \
    --attention-config "$HERE/arms/tuned.json" \
    --repeats 5 --warmups 2 --profile-iters 3 \
    --json "$OUT/k6c-p01b-block-profile-w1-tuned.json"

run k6c-f01b-split-compiled "$PY" "$HERE/block_profile.py" \
    --config pro --geometry w1 --target fused \
    --attention-config "$HERE/arms/tuned.json" \
    --compile default --repeats 5 --warmups 2 --profile-iters 3 \
    --json "$OUT/k6c-f01b-block-profile-w1-tuned-compiled.json"

echo "=== $(date +%H:%M:%S) session done, $failures failure(s)"
exit "$failures"
