#!/usr/bin/env bash
# max-autotune-no-cudagraphs: the Triton GEMM autotuning without the graph
# capture that makes the other two modes raise on this model.
#
# session10 and session11 established that mode="max-autotune" and
# mode="reduce-overhead" both raise "accessing tensor output of CUDAGraphs
# that has been overwritten by a subsequent run" on a real Kandinsky 6
# request: a DiT replays many captured graphs back to back while the residual
# stream holds a reference across block boundaries. This mode keeps the GEMM
# templates and drops the capture, so it is the one value of the flag that
# should be usable here.
#
# Paired with the attention arm, since that is what the showcase would serve,
# and with a shipped control after it.
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

run lite-w1-C1-tuned-plus-matnocg "$PY" "$T2V" "${common[@]}" \
    --diffusion-attention-config "$HERE/arms/tuned.json" \
    --diffusion-compile-mode max-autotune-no-cudagraphs \
    --output "$OUT/lite-w1-C1.mp4"

run lite-w1-B1b-tuned-attention-only "$PY" "$T2V" "${common[@]}" \
    --diffusion-attention-config "$HERE/arms/tuned.json" \
    --output "$OUT/lite-w1-B1b.mp4"

echo "=== $(date +%H:%M:%S) session done, $failures failure(s)"
exit "$failures"
