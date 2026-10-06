#!/usr/bin/env bash
# The INT8 arm, after the load-time fix, with the GPU test that covers the fix.
#
# The first attempt died during load: `scaled_int8_quant` is a CUDA kernel and
# `Int8OnlineLinearMethod.process_weights_after_loading` called it on
# `layer.weight` wherever that was, which under layer-wise offload is host
# memory. Fixed in vllm_omni/quantization/int8_config.py by quantizing on the
# device the kernel runs on and handing the result back on the weight's own
# device. The repo's own CUDA smoke test covers that function and could not run
# while the GPU was busy, so it runs here first, before any measurement depends
# on it.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${K6_PYTHON:-/data/jooman/k6/venv/bin/python}"
OUT="${K6_GATE_PRO:-/data/jooman/k6/results/gate-pro}"
REPO="$(cd "$HERE/../../.." && pwd)"
INT8='{"method": "int8"}'
failures=0
run() {
    local name="$1"; shift
    echo "=== $(date +%H:%M:%S) $name"
    if "$@"; then echo "=== $(date +%H:%M:%S) $name ok"
    else echo "=== $(date +%H:%M:%S) $name FAILED (exit $?)"; failures=$((failures + 1)); fi
}

run int8-unit-tests-on-gpu "$PY" -m pytest "$REPO/tests/diffusion/quantization/test_int8_config.py" -q

run int8-w8a8-smoke "$PY" "$HERE/gate_pro.py" \
    --arm shipped --name int8-w8a8-smoke \
    --quantization "$INT8" --offload layerwise \
    --limit 2 --no-warmup --no-locks --out-dir "$OUT/int8-w8a8-smoke"

echo "=== $(date +%H:%M:%S) session done, $failures failure(s)"
exit 0
