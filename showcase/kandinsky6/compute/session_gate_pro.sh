#!/usr/bin/env bash
# Generate the served Pro-distill arms' W1 outputs on Track M's prompt set A,
# so every arm is scored against the one canonical BF16 reference with the one
# canonical scorer (bench/quality.py).
#
# Arms, fastest first, because the objective is the fastest config that still
# passes the gate and a faster arm that passes makes the slower ones moot:
#
#   tuned             Sage2 on visual_self and video_audio_cross, SDPA on the
#                     two cheap audio roles. 109.92 s on W1 against the
#                     control's 175.54 s. Fails the gate's max on Lite.
#   sage2-visual-only drops Sage from video_audio_cross (0.19 ms a block), so
#                     it should keep nearly all the speed with one fewer
#                     quantized path.
#   sage2-accurate    FP16 PV with FP32 accumulation and per-thread INT8
#                     granularity on the same roles.
#   lossless          only the two cheap audio roles on SDPA; dense bf16
#                     either way, so it should sit at the floor.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${K6_PYTHON:-/data/jooman/k6/venv/bin/python}"
OUT="${K6_GATE_PRO:-/data/jooman/k6/results/gate-pro}"
failures=0
run() {
    local name="$1"; shift
    echo "=== $(date +%H:%M:%S) $name"
    if "$@"; then echo "=== $(date +%H:%M:%S) $name ok"; else
        status=$?; echo "=== $(date +%H:%M:%S) $name FAILED (exit $status)"; failures=$((failures + 1))
    fi
}

for arm in tuned sage2-visual-only sage2-accurate lossless; do
    run "gate-pro-$arm" "$PY" "$HERE/gate_pro.py" \
        --arm "$HERE/arms/$arm.json" --name "$arm" --no-locks \
        --out-dir "$OUT/$arm"
done

echo "=== $(date +%H:%M:%S) session done, $failures failure(s)"
exit "$failures"
