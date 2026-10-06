#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
#
# One GPU session: several runs under a single lock acquisition.
#
# The GPU of build-server-3 is shared and its free windows are short and
# unpredictable. Queueing each run separately means each one waits for its own
# turn, and a window big enough for three runs gets used for one. So
# `run_when_free.py` takes the locks once and this script spends the window:
#
#     python run_when_free.py --need-free-gib 16 -- bash session.sh
#
# Every run is independent and a failure does not stop the rest: a window is
# too scarce to lose to one bad arm. Each run's exit status is logged and the
# script's own status is the number that failed.
#
# `run_when_free.py` appends `--no-locks` to the command it runs, so this
# script is handed one argument it does not need — it already runs under the
# parent's locks, and passes `--no-locks` to each tool itself. That argument
# is ignored on purpose rather than treated as an error.

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${K6_PYTHON:-/data/jooman/k6/venv/bin/python}"
OUT="${K6_RESULTS:-/data/jooman/k6/results}"
mkdir -p "$OUT"

failures=0

run() {
    local name="$1"
    shift
    echo "=== $(date +%H:%M:%S) $name"
    if "$@" --no-locks; then
        echo "=== $(date +%H:%M:%S) $name ok"
    else
        local status=$?
        echo "=== $(date +%H:%M:%S) $name FAILED (exit $status)"
        failures=$((failures + 1))
    fi
}

# k6c-f01: eager vs torch.compile on one Pro block, ABBA inside one process.
# Both arms carry arms/tuned.json, so the fusion is measured on the stack the
# showcase would actually serve rather than on the platform default.
run k6c-f01-compile "$PY" "$HERE/block_profile.py" \
    --config pro --geometry w1 --target fused \
    --attention-config "$HERE/arms/tuned.json" \
    --compare-compile --compile default --repeats 3 --rounds 2 --warmups 2 \
    --profile-iters 0 \
    --json "$OUT/k6c-f01-compile-vs-eager-w1.json"

# k6c-p01b: the step split again, with arms/tuned.json instead of the platform
# default. p01 profiled the shipped-as-is stack; this one profiles the stack
# the attention race chose, which is where the next hypothesis has to be
# ranked from.
run k6c-p01b-split-tuned "$PY" "$HERE/block_profile.py" \
    --config pro --geometry w1 --target fused \
    --attention-config "$HERE/arms/tuned.json" \
    --repeats 5 --warmups 2 --profile-iters 3 \
    --json "$OUT/k6c-p01b-block-profile-w1-tuned.json"

# k6c-a01b: the headline attention role again, now that FlashAttention-4 is
# installed. FLASHINFER_ATTN is left out: on sm_120 it cannot run and costs
# about 90 s of 404 retries before saying so.
run k6c-a01b-fa4 "$PY" "$HERE/attn_race.py" \
    --roles visual_self --arms CUDNN_ATTN,FLASH_ATTN,SAGE_ATTN,SAGE_ATTN_3 \
    --repeats 3 --rounds 2 \
    --json "$OUT/k6c-a01b-attn-race-w1-fa4.json"

echo "=== $(date +%H:%M:%S) session done, $failures failure(s)"
exit "$failures"
