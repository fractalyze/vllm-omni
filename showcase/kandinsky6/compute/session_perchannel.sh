#!/usr/bin/env bash
# Per-output-row FP8 scales, quantized at load from the exact BF16 checkpoint.
#
# Both FP8 checkpoints on this host carry **per-tensor** scales: one scalar for a
# 4096x16384 matrix, so a single outlier weight sets the range for every row.
# `tools/quantize_dit_fp8.py` says in its own docstring that per-row scales are
# the better recipe and that it cannot ship them, because vLLM's serialized fp8
# method builds a PerTensorScaleParameter and a per-row scale fails its shape
# assertion. That is a limitation of the *serialized* path only: vLLM's online
# methods carry a group shape, and `fp8_per_channel` is row=-1 col=1 -- one
# scale per output row -- with `activation=None`, so it is weight-only FP8 and
# the activations stay BF16.
#
# Two reasons that is the most promising arm left. Quality: per-row scales and
# unquantized activations are both strictly better than what scored
# 0.262/0.551. Speed: this arm is weight-traffic-bound, 60.3 GB re-read every
# step at ~4 GB/s, and FP8 weights halve the bytes -- to ~30 GB, which also
# fits this host's page cache, so the NVMe re-read should go away rather than
# merely halve.
#
# Smoke first (two prompts, no warm-up) because it may not load at all: online
# quantization has to happen tensor by tensor as the checkpoint streams, and the
# loader has restrictions on combining it with offload. Then the full set.
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

# Does it load, and from pinned host memory (the fast path: 30 GB of FP8 fits
# where 60 GB of BF16 does not)?
run perchan-pinned-smoke "$PY" "$HERE/gate_pro.py" \
    --arm "$HERE/arms/tuned.json" --name perchan-pinned-smoke \
    --quantization fp8_per_channel --offload layerwise \
    --limit 2 --no-warmup --no-locks --out-dir "$OUT/perchan-pinned-smoke"

# If pinning 30 GB beside the server does not fit, the same weights stream from
# the mmapped BF16 checkpoint instead; slower to load, same arithmetic.
if [ "$failures" -ne 0 ]; then
    run perchan-mmap-smoke "$PY" "$HERE/gate_pro.py" \
        --arm "$HERE/arms/tuned.json" --name perchan-mmap-smoke \
        --quantization fp8_per_channel --offload dlo-mmap \
        --limit 2 --no-warmup --no-locks --out-dir "$OUT/perchan-mmap-smoke"
fi

echo "=== $(date +%H:%M:%S) session done, $failures failure(s)"
exit 0
