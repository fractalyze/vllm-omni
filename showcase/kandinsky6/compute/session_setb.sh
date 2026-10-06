#!/usr/bin/env bash
# Set B, both halves, in one session: the user's gate is per prompt set and
# nobody has generated B's BF16 reference.
#
# A reference is a gate run on exact weights with the platform's own attention,
# which is what `--arm shipped --offload dlo-mmap` is -- the same server
# configuration that produced Track M's set A references. Generated here first,
# then the candidate is scored against it, so B's numbers are self-consistent
# even though they do not share A's reference process.
#
# K6_CAND_ARM and K6_CAND_NAME select which candidate to score, because which
# arm is the headline is decided by set A and the timing, not here.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${K6_PYTHON:-/data/jooman/k6/venv/bin/python}"
OUT="${K6_GATE_PRO:-/data/jooman/k6/results/gate-pro}"
REF="${K6_SETB_REF:-/data/jooman/k6/ref/setB-bs3}"
PROMPTS="$HERE/../bench/prompts/setB.json"
CAND_ARM="${K6_CAND_ARM:-$HERE/arms/tuned.json}"
CAND_NAME="${K6_CAND_NAME:-bf16-stream-sage2-setb}"
CAND_CKPT="${K6_CAND_CKPT:-}"
failures=0
run() {
    local name="$1"; shift
    echo "=== $(date +%H:%M:%S) $name"
    if "$@"; then echo "=== $(date +%H:%M:%S) $name ok"
    else echo "=== $(date +%H:%M:%S) $name FAILED (exit $?)"; failures=$((failures + 1)); fi
}

# The reference. No reference manifest exists yet, so this run falls back to
# seed 42 and W1 -- deliberately the same seed Track M used for set A.
if [ ! -f "$REF/manifest.json" ]; then
    run setb-bf16-reference "$PY" "$HERE/gate_pro.py" \
        --arm shipped --name setb-bf16-reference --offload dlo-mmap \
        --prompts "$PROMPTS" --reference-manifest /nonexistent \
        --no-locks --out-dir "$REF"
else
    echo "=== $(date +%H:%M:%S) setb-bf16-reference exists, skipped"
fi

run "$CAND_NAME" "$PY" "$HERE/gate_pro.py" \
    --arm "$CAND_ARM" --name "$CAND_NAME" --offload dlo-mmap \
    ${CAND_CKPT:+--ckpt "$CAND_CKPT"} \
    --prompts "$PROMPTS" --reference-manifest "$REF/manifest.json" \
    --no-locks --out-dir "$OUT/$CAND_NAME"

echo "=== $(date +%H:%M:%S) session done, $failures failure(s)"
exit "$failures"
