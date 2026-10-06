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
failures=0
run() {
    local name="$1"; shift
    echo "=== $(date +%H:%M:%S) $name"
    if "$@"; then echo "=== $(date +%H:%M:%S) $name ok"
    else echo "=== $(date +%H:%M:%S) $name FAILED (exit $?)"; failures=$((failures + 1)); fi
}

run "$ARM-setA" "$PY" "$HERE/gate_pro.py" \
    --arm "$HERE/arms/$ARM.json" --name "$ARM-setA" --offload dlo-mmap \
    --no-locks --out-dir "$OUT/$ARM-setA"

run "$ARM-setB" "$PY" "$HERE/gate_pro.py" \
    --arm "$HERE/arms/$ARM.json" --name "$ARM-setB" --offload dlo-mmap \
    --prompts "$HERE/../bench/prompts/setB.json" --reference-manifest /nonexistent \
    --no-locks --out-dir "$OUT/$ARM-setB"

echo "=== $(date +%H:%M:%S) session done, $failures failure(s)"
exit "$failures"
