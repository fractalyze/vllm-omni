#!/usr/bin/env bash
# The quality gate for the attention arms, on prompt set A.
#
# My headline -- the tuned attention arm, -26% on a Lite request -- has been
# ungated all session: its only accuracy evidence was an attention-kernel
# error on synthetic activations. This scores it the way PLAN.md's gate does:
# same checkpoint, same prompts, same seeds, LPIPS per frame against the arm
# the model ships with. The shipped arm is cuDNN in bf16, so it *is* the BF16
# reference the gate asks for; what is missing against PLAN.md is only that
# this is Lite rather than Pro, because Pro does not fit.
#
# Three arms: the reference, Sage2 (what the showcase would serve), and Sage3,
# whose kernel error was 4.8x Sage2's and which has never been scored on an
# image.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${K6_PYTHON:-/data/jooman/k6/venv/bin/python}"
GATE="${K6_RESULTS:-/data/jooman/k6/results}/gate"
export HF_HOME=/data/jooman/hf
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
failures=0
run() {
    local name="$1"; shift
    echo "=== $(date +%H:%M:%S) $name"
    if "$@"; then echo "=== $(date +%H:%M:%S) $name ok"; else
        status=$?; echo "=== $(date +%H:%M:%S) $name FAILED (exit $status)"; failures=$((failures + 1))
    fi
}

run gate-a-shipped "$PY" "$HERE/gate_run.py" --arm shipped \
    --prompts "$HERE/prompts_set_a.json" --out-dir "$GATE/a-shipped"

run gate-a-tuned "$PY" "$HERE/gate_run.py" --arm "$HERE/arms/tuned.json" \
    --prompts "$HERE/prompts_set_a.json" --out-dir "$GATE/a-tuned"

run gate-a-sage3 "$PY" "$HERE/gate_run.py" --arm "$HERE/arms/sage3.json" \
    --prompts "$HERE/prompts_set_a.json" --out-dir "$GATE/a-sage3"

echo "=== $(date +%H:%M:%S) session done, $failures failure(s)"
exit "$failures"
