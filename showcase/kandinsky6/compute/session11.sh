#!/usr/bin/env bash
# Does the attention arm survive a real pipeline, and does reduce-overhead
# fail the same way max-autotune did?
#
# session10 found that mode="max-autotune" raises inside a real request:
#
#   RuntimeError: accessing tensor output of CUDAGraphs that has been
#   overwritten by a subsequent run
#
# from Kandinsky6TransformerEncoderBlock.forward -> apply_gate_sum. A DiT's
# residual stream crosses block boundaries, and cudagraph trees assume a
# graph's output is consumed before the next replay. So the -38.5% measured on
# one block in isolation may not be reachable here at all.
#
# Two questions, in one window, with a shipped control either side:
#   B1  the attention arm alone, no compile-mode change  -> the real headline
#   B2  mode="reduce-overhead"                           -> same failure?
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
    if "$@"; then echo "=== $(date +%H:%M:%S) $name ok"; else
        status=$?; echo "=== $(date +%H:%M:%S) $name FAILED (exit $status)"; failures=$((failures + 1))
    fi
}

common=(
    --model kandinskylab/Kandinsky-6.0-Lite-5s-Diffusers
    --model-class-name Kandinsky6TI2VAPipeline
    --enable-cpu-offload --seed 42
    --height 480 --width 864 --num-frames 121 --num-inference-steps 10
    --prompt "$PROMPT"
)

run lite-w1-B1-tuned-attention-only "$PY" "$T2V" "${common[@]}" \
    --diffusion-attention-config "$HERE/arms/tuned.json" \
    --output "$OUT/lite-w1-B1.mp4"

run lite-w1-B2-reduce-overhead "$PY" "$T2V" "${common[@]}" \
    --diffusion-compile-mode reduce-overhead \
    --output "$OUT/lite-w1-B2.mp4"

run lite-w1-A3-shipped "$PY" "$T2V" "${common[@]}" --output "$OUT/lite-w1-A3.mp4"

echo "=== $(date +%H:%M:%S) session done, $failures failure(s)"
exit "$failures"
