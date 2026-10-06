#!/usr/bin/env bash
# The last attention arm that can still pass the user's literal gate, and then
# the floor every gate number is read against.
#
# Why this arm and why now. One-byte weights are out on this GPU: INT8's W8A8
# GEMM does not exist on sm_120 ("Int8 not supported on SM120. Use FP8
# quantization instead" from CUTLASS's dispatch_scaled_mm) and vLLM's online
# registry has no INT8 weight-only path either, while FP8 E4M3 -- the one-byte
# format sm_120 does support -- carries 2.6% relative weight error at any scale
# granularity and measures LPIPS 0.262 against BF16. So the weights have to stay
# BF16, and the only lever left on a BF16 arm is attention.
#
# Sage2 at its default settings is too lossy (set A 0.1682/0.3693 against
# 0.15/0.25). arms/sage2-accurate.json is the same kernel family at its accurate
# settings: INT8 QK at per-thread granularity, PV in FP16 with FP32 accumulation
# instead of FP8, smooth_k on. Slower than sage2 and far faster than cuDNN, so if
# it lands inside the gate it is the fastest G1-passing configuration there is.
#
# The control follows in the same session rather than before it, because it is
# the floor rather than a candidate and the candidate is what the clock is for.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${K6_PYTHON:-/data/jooman/k6/venv/bin/python}"
OUT="${K6_GATE_PRO:-/data/jooman/k6/results/gate-pro}"
failures=0
run() {
    local name="$1"; shift
    echo "=== $(date +%H:%M:%S) $name"
    if "$@"; then echo "=== $(date +%H:%M:%S) $name ok"
    else echo "=== $(date +%H:%M:%S) $name FAILED (exit $?)"; failures=$((failures + 1)); fi
}

run bf16-stream-accurate "$PY" "$HERE/gate_pro.py" \
    --arm "$HERE/arms/sage2-accurate.json" --name bf16-stream-accurate --offload dlo-mmap \
    --no-locks --out-dir "$OUT/bf16-stream-accurate"

run bf16-stream-control-full "$PY" "$HERE/gate_pro.py" \
    --arm shipped --name bf16-stream-control-full --offload dlo-mmap \
    --no-locks --out-dir "$OUT/bf16-stream-control-full"

echo "=== $(date +%H:%M:%S) session done, $failures failure(s)"
exit "$failures"
