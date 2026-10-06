#!/usr/bin/env bash
# Which one-byte weight format can pass the gate? Measured offline first, then
# end to end here.
#
# `tools/weight_quant_error.py` quantizes the real DiT tensors and reports the
# relative error each format introduces. On 14 sampled 2-D weights, median:
#
#   FP8 E4M3  per-tensor 0.02645   per-row 0.02643
#   INT8      per-tensor 0.02057   per-row 0.00908
#
# Two findings there. FP8's error does not care about scale granularity -- E4M3
# carries its own 4-bit exponent, so a finer scale only slides the matrix along
# the exponent ladder while the error stays at the 3-bit mantissa's step. That
# kills the per-row-FP8 hypothesis before it costs a GPU hour, and it explains
# why Track M's 0.262 did not move with the keep profile and why their
# FP8_DYNAMIC (per-row weights plus per-token activations) was *worse* at
# 0.338-0.364: it added activation error on top of a weight error that was never
# about the scale. INT8 is fixed point, so there the scale *is* the step, and
# per-row INT8 costs 2.9x less error than FP8 at the same one byte per weight.
#
# So INT8 per-output-row is the format to test. Two variants, because the
# control arm just showed the two bottlenecks are closer than they looked --
# 21.9 s/step with the platform's cuDNN attention against 14.5 s/step with
# Sage2, on a stream that is worth about 15 s/step. Bytes alone therefore do not
# make this workload fast; the GEMMs have to get faster too.
#
#   `int8`  -- vLLM-Omni's own DiffusionInt8Config: per-output-channel weight
#              scales *and* dynamic per-token activation scales, so the GEMMs run
#              on INT8 tensor cores. The fast arm, and the one with an activation
#              approximation to answer for.
#   weight-only -- INT8 weights, activations left in BF16. Half the bytes, BF16
#              GEMMs, so about the streamed arm's speed with a third of FP8's
#              weight error. The quality fallback.
#
# Both arms run with the platform's own attention: Sage2 failed the gate on
# exact weights (a3 max LPIPS 0.3277 against a 0.25 limit), so an arm carrying
# both it and a new weight format could only fail for two reasons at once.
#
# The FP8 arm runs second, two prompts, as the end-to-end check on the offline
# prediction: if the offline numbers mean what they say, its LPIPS should be
# roughly 3x the INT8 arm's.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${K6_PYTHON:-/data/jooman/k6/venv/bin/python}"
OUT="${K6_GATE_PRO:-/data/jooman/k6/results/gate-pro}"
INT8_W8A8='int8'
INT8_WEIGHT_ONLY='{"method": "int8_per_channel_weight_only", "linear": "int8_per_channel_static"}'
failures=0
run() {
    local name="$1"; shift
    echo "=== $(date +%H:%M:%S) $name"
    if "$@"; then echo "=== $(date +%H:%M:%S) $name ok"
    else echo "=== $(date +%H:%M:%S) $name FAILED (exit $?)"; failures=$((failures + 1)); fi
}

run int8-w8a8-smoke "$PY" "$HERE/gate_pro.py" \
    --arm shipped --name int8-w8a8-smoke \
    --quantization "$INT8_W8A8" --offload layerwise \
    --limit 2 --no-warmup --no-locks --out-dir "$OUT/int8-w8a8-smoke"

# If 26 GiB of INT8 weights will not pin beside the server, the same weights
# stream from the mmapped BF16 checkpoint instead; same arithmetic, slower load.
if [ "$failures" -ne 0 ]; then
    run int8-w8a8-mmap-smoke "$PY" "$HERE/gate_pro.py" \
        --arm shipped --name int8-w8a8-mmap-smoke \
        --quantization "$INT8_W8A8" --offload dlo-mmap \
        --limit 2 --no-warmup --no-locks --out-dir "$OUT/int8-w8a8-mmap-smoke"
fi

run int8-weight-only-smoke "$PY" "$HERE/gate_pro.py" \
    --arm shipped --name int8-weight-only-smoke \
    --quantization "$INT8_WEIGHT_ONLY" --offload layerwise \
    --limit 2 --no-warmup --no-locks --out-dir "$OUT/int8-weight-only-smoke"

echo "=== $(date +%H:%M:%S) session done, $failures failure(s)"
exit 0
