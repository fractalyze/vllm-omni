#!/usr/bin/env bash
# The chosen arm's full set A, then its set B outputs.
#
# K6_ARM names an arms/*.json; set A is scored against ref-eager/setA with Track
# M's floors, set B exists so it can be scored the moment their eager set B
# references land. Nine prompts each, warm-up discarded, one server per set so
# neither set inherits the other's allocator state.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${K6_PYTHON:-/data/jooman/k6/venv/bin/python}"
OUT="${K6_GATE_PRO:-/data/jooman/k6/results/gate-pro}"
ARM="${K6_ARM:?set K6_ARM to an arms/*.json basename, e.g. sage2-wide}"
# The output directory is named by K6_LABEL, not by the arm, because the same
# arm file can be two different arms: `sage2-mid` with
# VLLM_OMNI_K6_EXACT_ATTN_STEPS=1 is not `sage2-mid`, and writing both into one
# directory would silently overwrite the first with the second and leave a
# manifest that describes neither.
LABEL="${K6_LABEL:-$ARM}"
failures=0
run() {
    local name="$1"; shift
    echo "=== $(date +%H:%M:%S) $name"
    if "$@"; then echo "=== $(date +%H:%M:%S) $name ok"
    else echo "=== $(date +%H:%M:%S) $name FAILED (exit $?)"; failures=$((failures + 1)); fi
}

echo "=== arm $ARM, label $LABEL, exact steps ${VLLM_OMNI_K6_EXACT_ATTN_STEPS:-0}"

run "$LABEL-setA" "$PY" "$HERE/gate_pro.py" \
    --arm "$HERE/arms/$ARM.json" --name "$LABEL-setA" --offload dlo-mmap \
    --no-locks --out-dir "$OUT/$LABEL-setA"

run "$LABEL-setB" "$PY" "$HERE/gate_pro.py" \
    --arm "$HERE/arms/$ARM.json" --name "$LABEL-setB" --offload dlo-mmap \
    --prompts "$HERE/../bench/prompts/setB.json" --reference-manifest /nonexistent \
    --no-locks --out-dir "$OUT/$LABEL-setB"

echo "=== $(date +%H:%M:%S) session done, $failures failure(s)"
exit "$failures"
