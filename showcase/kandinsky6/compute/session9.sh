#!/usr/bin/env bash
# Kandinsky 6 Lite end to end at W1's geometry, baseline against both
# switches, with the pipeline profiler on so the per-stage split is recorded.
#
# Lite rather than Pro because Pro does not fit; W1's geometry rather than the
# smoke one because the switches are shape-dependent -- SageAttention needs
# many queries to amortize its per-block quantization prologue
# (c-k6c-sage-quant-prologue-dominates-at-few-queries), and the smoke
# geometry's 4,480 visual tokens is where it loses. At W1's geometry Lite sees
# the same 50,220 visual tokens as Pro, at 1792 model dim instead of 4096.
#
# Two arms, one after the other in one GPU window so the clocks match:
# the shipped default, then the tuned attention arm with max-autotune.
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
    --enable-diffusion-pipeline-profiler
)

run lite-w1-shipped "$PY" "$T2V" "${common[@]}" \
    --output "$OUT/lite-w1-shipped.mp4"

echo "=== $(date +%H:%M:%S) session done, $failures failure(s)"
exit "$failures"
