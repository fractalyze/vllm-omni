#!/usr/bin/env bash
# Kandinsky 6 Lite end to end, the first whole-pipeline run of this track.
#
# Everything measured so far is one synthetic DiT block. Lite (3.7B) is the
# largest Kandinsky 6 that fits a 32 GB card with its text encoder, so it is
# what can check that the two switches survive a real pipeline and what gives
# the first measured shares for the Hunyuan VAE decode and the Qwen2.5-VL text
# encoder -- both still unmeasured, and both outside the DiT the block work
# has been about.
#
# The smoke geometry (512x320, 25 frames, 10 steps) first: a correctness and
# fit check before anything is timed.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
PY="${K6_PYTHON:-/data/jooman/k6/venv/bin/python}"
OUT="${K6_RESULTS:-/data/jooman/k6/results}"
export HF_HOME=/data/jooman/hf
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
T2V="$REPO/examples/offline_inference/text_to_video/text_to_video.py"
failures=0
run() {
    local name="$1"; shift
    echo "=== $(date +%H:%M:%S) $name"
    if "$@"; then echo "=== $(date +%H:%M:%S) $name ok"
    else echo "=== $(date +%H:%M:%S) $name FAILED (exit $?)"; failures=$((failures + 1)); fi
}

run lite-smoke-baseline "$PY" "$T2V" \
    --model kandinskylab/Kandinsky-6.0-Lite-5s-Diffusers \
    --model-class-name Kandinsky6TI2VAPipeline \
    --enable-cpu-offload --seed 42 \
    --height 320 --width 512 --num-frames 25 --num-inference-steps 10 \
    --prompt "a golden retriever running on a sunny beach, breaking waves" \
    --output "$OUT/lite-smoke-baseline.mp4"

echo "=== $(date +%H:%M:%S) session done, $failures failure(s)"
exit "$failures"
