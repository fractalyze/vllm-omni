#!/usr/bin/env bash
# Gate the two candidate arms against the user's bar (set mean LPIPS <= 0.15,
# max <= 0.25) on prompt set A.
#
# arms/tuned.json passed the mean at 0.1178 and failed the max at 0.3745. The
# two candidates attack that max from different directions:
#
#   sage2-visual-only  drops Sage from video_audio_cross, which is 0.19 ms a
#                      block, so it removes a quantized path for almost no
#                      speed.
#   sage2-accurate     keeps both roles on Sage but with the FP16 PV path,
#                      FP32 accumulation and per-thread INT8 granularity.
#
# The reference is already on disk (gate/a-shipped), so only the candidates
# need generating.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${K6_PYTHON:-/data/jooman/k6/venv/bin/python}"
GATE="${K6_RESULTS:-/data/jooman/k6/results}/gate"
export HF_HOME=/data/jooman/hf
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
failures=0
run() {
    local name="$1"; shift
    echo "=== $(date +%H:%M:%S) $name"
    if "$@"; then echo "=== $(date +%H:%M:%S) $name ok"; else
        status=$?; echo "=== $(date +%H:%M:%S) $name FAILED (exit $status)"; failures=$((failures + 1))
    fi
}

for arm in sage2-visual-only sage2-accurate; do
    run "gate-a-$arm" "$PY" "$HERE/gate_run.py" --arm "$HERE/arms/$arm.json" \
        --prompts "$HERE/prompts_set_a.json" --out-dir "$GATE/a-$arm"
    run "score-a-$arm" "$PY" "$HERE/gate_score.py" \
        --reference "$GATE/a-shipped" --candidate "$GATE/a-$arm" \
        --noise-floor-max 0.0030 --no-locks \
        --json "$GATE/scores/a-$arm.json"
done

echo "=== $(date +%H:%M:%S) session done, $failures failure(s)"
exit "$failures"
