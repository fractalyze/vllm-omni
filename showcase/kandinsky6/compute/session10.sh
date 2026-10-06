#!/usr/bin/env bash
# The end-to-end A/B this whole track points at: the two switches on a real
# Kandinsky 6 request, not on a synthetic block.
#
# Lite at W1's geometry (Pro does not fit), so the DiT sees W1's 50,220 visual
# tokens. Arm A is the shipped default; arm B is arms/tuned.json plus
# mode="max-autotune". Both in one GPU window, A then B then A again, so a
# clock or co-tenant drift shows up as disagreement between the two A runs
# rather than as a win for B.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
PY="${K6_PYTHON:-/data/jooman/k6/venv/bin/python}"
OUT="${K6_RESULTS:-/data/jooman/k6/results}"
export HF_HOME=/data/jooman/hf
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
T2V="$REPO/examples/offline_inference/text_to_video/text_to_video.py"
PROMPT="a golden retriever running on a sunny beach, breaking waves"
failures=0
run() {
    local name="$1"; shift
    echo "=== $(date +%H:%M:%S) $name"
    if "$@"; then echo "=== $(date +%H:%M:%S) $name ok"
    else echo "=== $(date +%H:%M:%S) $name FAILED (exit $?)"; failures=$((failures + 1)); fi
}

common=(
    --model kandinskylab/Kandinsky-6.0-Lite-5s-Diffusers
    --model-class-name Kandinsky6TI2VAPipeline
    --enable-cpu-offload --seed 42
    --height 480 --width 864 --num-frames 121 --num-inference-steps 10
    --prompt "$PROMPT"
)

run lite-w1-A1-shipped "$PY" "$T2V" "${common[@]}" --output "$OUT/lite-w1-A1.mp4"
run lite-w1-B-tuned-maxautotune "$PY" "$T2V" "${common[@]}" \
    --diffusion-attention-config "$HERE/arms/tuned.json" \
    --diffusion-compile-mode max-autotune \
    --output "$OUT/lite-w1-B.mp4"
run lite-w1-A2-shipped "$PY" "$T2V" "${common[@]}" --output "$OUT/lite-w1-A2.mp4"

echo "=== $(date +%H:%M:%S) session done, $failures failure(s)"
exit "$failures"
