#!/usr/bin/env bash
# Is the streamed BF16 arm stream-bound or compute-bound? Ten minutes, no new code.
#
# The question matters because it decides a two-hour build. INT8-as-storage
# (quantize offline, dequantize after the H2D copy, run BF16 GEMMs) halves the
# bytes per step and is worth doing *if* bytes are what the step is waiting on.
# The arithmetic does not settle it: the NVMe re-read is about 14.4 s/step
# (1.67 GB/s over the ~24 GB that misses a 57 GB page cache) and Sage2's compute
# is about 15.4 s/step (cuDNN's 21.9 less the 6.5 s/step the kernel race says
# Sage2 saves over 60 blocks). Those are the same number within the error of the
# estimate, and only one of them is the binding constraint.
#
# So remove bytes from the stream and see whether the step gets shorter.
# `--resident-layers` keeps leading DiT blocks on the device (0.899 GiB each) and
# `--offload-text-encoder` frees the board memory that decides how many fit. Both
# are numerically identical to the plain arm -- the same weights, the same
# kernels, the same order -- so any change in step time is the stream and
# nothing else.
#
#   stream-bound  -> step time falls by about 0.3 s per resident block, and
#                    `iostat` shows the NVMe read dropping. INT8 storage is worth
#                    building.
#   compute-bound -> step time does not move. INT8 storage buys nothing on this
#                    arm and the two hours go elsewhere.
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

( while true; do
      printf '%s ' "$(date +%H:%M:%S)"
      iostat -d nvme0n1 1 2 2>/dev/null | awk '/nvme0n1/{r=$3} END{printf "nvme_read_kBps=%s ", r}'
      nvidia-smi --query-gpu=memory.used --format=csv,noheader
      sleep 10
  done ) > "$OUT/residency-probe-io.log" 2>&1 &
sampler=$!
trap 'kill "$sampler" 2>/dev/null' EXIT

run residency-probe "$PY" "$HERE/gate_pro.py" \
    --arm "$HERE/arms/tuned.json" --name residency-probe --offload dlo-mmap \
    --resident-layers 16 --offload-text-encoder \
    --limit 2 --no-locks --out-dir "$OUT/residency-probe"

echo "=== $(date +%H:%M:%S) session done, $failures failure(s)"
exit "$failures"
