#!/usr/bin/env bash
# Residency probe, second attempt: resident DiT blocks only.
#
# The first attempt asked for the text encoder to be offloaded as well, to free
# the board memory for more resident blocks, and the pipeline refused: "Selected
# text encoder 'text_encoder' requires a model-declared streamable or on-demand
# plan". Kandinsky 6's port does not declare one, so the encoder stays resident
# and the probe has to fit inside the ~7 GiB the streamed arm leaves free.
#
# Six blocks at 0.899 GiB is 5.4 GiB, which fits, and it is enough to answer the
# question in either of two ways:
#
#   linear  -> about -0.3 s/step per block, so -1.8 s/step, if the stream is
#              simply serialized against compute.
#   cliff   -> much more than that. The re-read working set falls from 56.14 GiB
#              to 50.7 GiB, under what this host's page cache actually holds, so
#              a hit rate measured at ~0.6 should go to nearly 1 and the NVMe
#              read should collapse rather than shrink.
#   flat    -> the arm is compute-bound, bytes are not the constraint, and
#              INT8-as-storage is not worth building.
#
# Numerically identical to the plain arm either way: the same weights in the same
# order through the same kernels, only staged from a different place.
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
  done ) > "$OUT/residency-io.log" 2>&1 &
sampler=$!
trap 'kill "$sampler" 2>/dev/null' EXIT

run residency-6 "$PY" "$HERE/gate_pro.py" \
    --arm "$HERE/arms/tuned.json" --name residency-6 --offload dlo-mmap \
    --resident-layers 6 \
    --limit 2 --no-locks --out-dir "$OUT/residency-6"

echo "=== $(date +%H:%M:%S) session done, $failures failure(s)"
exit "$failures"
