#!/usr/bin/env bash
# The FP8 keep profile that has never been gated, served the reference's way.
#
# Two facts from the checkpoints' own quantization_report.json: Track M's
# 173.6 s baseline is `pro-distill-fp8-min` (keep profile `minimal`, 30.2 GB,
# only the embeddings and output heads in BF16), and `pro-distill-fp8` next to
# it is the `sensitive` profile (36.7 GB) which also keeps every block's
# modulation, both text towers and visual blocks 0 and 59. The 0.262/0.551
# failure is the minimal profile's; the sensitive one has no gate number at all.
# Its own tool says why it was shelved -- 36.7 GB does not fit pinned host
# memory beside the server on a 60 GB host.
#
# Which is exactly the constraint `--offload dlo-mmap` removes. Streaming it
# from the mmapped checkpoint pins nothing, and 36.7 GB is comfortably inside
# this host's ~57 GB page cache, so unlike the 60.3 GB BF16 DiT it should be
# read once and then served from RAM rather than re-read from NVMe every step.
# If the sensitive profile passes the gate, this is the fast gate-passing
# configuration; if it fails, per-tensor scales are the remaining suspect and
# that is Track M's.
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

run fp8sens-stream-sage2 "$PY" "$HERE/gate_pro.py" \
    --arm "$HERE/arms/tuned.json" --name fp8sens-stream-sage2 \
    --ckpt /data/jooman/k6/ckpt/pro-distill-fp8 --offload dlo-mmap \
    --no-locks --out-dir "$OUT/fp8sens-stream-sage2"

echo "=== $(date +%H:%M:%S) session done, $failures failure(s)"
exit "$failures"
