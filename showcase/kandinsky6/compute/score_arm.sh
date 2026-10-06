#!/usr/bin/env bash
# All three gates for one arm's outputs, on the CPU.
#
# G1 and G2 come from gate_pro_score.py (the floor makes G2 decidable; without
# it the scorer says "undecided" rather than silently restating G1), G3 and the
# contact sheet from clip_gate.py, and a side-by-side sheet of the arm against
# the reference on the prompt that scored worst -- which is the one worth
# looking at, and not always the one a mean would point to.
#
# CUDA_VISIBLE_DEVICES is cleared throughout: the GPU belongs to whatever is
# being timed, and a scorer that takes memory from a timed run has already cost
# this study one measurement.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${K6_PYTHON:-/data/jooman/k6/venv/bin/python}"
ARM_DIR="${1:?usage: score_arm.sh ARM_DIR [REFERENCE_DIR] [PROMPTS]}"
REF="${2:-/data/jooman/k6/ref-eager/setA}"
PROMPTS="${3:-$HERE/../bench/prompts/setA.json}"
NAME="$(basename "$ARM_DIR")"
SCORES="${K6_GATE_PRO:-/data/jooman/k6/results/gate-pro}/scores"
SAMPLES="${K6_SAMPLES:-/data/jooman/k6/showcase-samples}"
export CUDA_VISIBLE_DEVICES=""
export HF_HOME="${HF_HOME:-/data/jooman/hf}"
mkdir -p "$SCORES" "$SAMPLES"

echo "=== $(date +%H:%M:%S) G1 + G2: $NAME against $(basename "$REF")"
"$PY" "$HERE/gate_pro_score.py" --arm-dir "$ARM_DIR" --reference-dir "$REF" --prompts "$PROMPTS" \
    --g2-floor-mean "${K6_FLOOR_MEAN:-0.1455}" --g2-floor-max "${K6_FLOOR_MAX:-0.4160}" \
    --no-locks --json "$SCORES/$NAME.json"

echo "=== $(date +%H:%M:%S) G3: $NAME"
"$PY" "$HERE/clip_gate.py" --arm-dir "$ARM_DIR" --reference-dir "$REF" --prompts "$PROMPTS" \
    --json "$SCORES/g3-$NAME.json" --contact-sheet "$SAMPLES/sheet-$NAME.jpg"

# The scorer writes {"verdict": ..., "detail": ...} and the worst prompt is the
# scorer's own, not a re-derivation here: two definitions of "worst" that drift
# apart would put the wrong clip in front of the reader.
worst="$("$PY" - "$SCORES/$NAME.json" <<'PYEOF'
import json, sys
print(json.load(open(sys.argv[1])).get("detail", {}).get("worst_prompt") or "")
PYEOF
)"
if [ -n "$worst" ]; then
    echo "=== $(date +%H:%M:%S) a look at the worst prompt, $worst"
    "$PY" "$HERE/compare_sheet.py" --prompt "$worst" \
        --arm "BF16 reference=$REF" --arm "$NAME=$ARM_DIR" \
        --out "$SAMPLES/compare-$worst-$NAME.jpg"
fi
echo "=== $(date +%H:%M:%S) done"
