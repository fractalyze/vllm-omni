#!/usr/bin/env bash
# Two schedules stacked, on the two prompts that decide the gate.
#
# `sage2-mid` fails set B's floor-relative max on b6-train-platform (0.4989
# against a 0.4345 limit), with b3-storefront-sign behind it. Both are
# text-plus-motion. Two ways to buy the ~13% that b6 needs:
#
#   sage2-narrow        24 exact blocks instead of 12. Costs ~12 s a request.
#   sage2-mid + step 1  the same 12-block band, plus the first sampler step
#                       exact (Track M's VLLM_OMNI_K6_EXACT_ATTN_STEPS). One
#                       step of ten, so ~18 s of DiT if the step were free of
#                       the band, less in practice because the band already
#                       makes part of that step exact.
#
# The step dial is orthogonal to the block dial -- one picks *when* in the
# trajectory to be exact, the other *where* in the stack -- so if the error is
# concentrated early as well as at the ends, stacking should beat either alone.
# Track M measures exact-step-1 to be worth passing G2 on set A for their INT8
# arm, so the dial is known to work; what is open is whether it moves b6.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${K6_PYTHON:-/data/jooman/k6/venv/bin/python}"
OUT="${K6_GATE_PRO:-/data/jooman/k6/results/gate-pro}"
export VLLM_OMNI_K6_EXACT_ATTN_STEPS=1
echo "=== $(date +%H:%M:%S) sage2-mid + exact step 1, on b3 and b6"
echo "    VLLM_OMNI_K6_EXACT_ATTN_STEPS=$VLLM_OMNI_K6_EXACT_ATTN_STEPS"
"$PY" "$HERE/gate_pro.py" \
    --arm "$HERE/arms/sage2-mid.json" --name sage2-mid-step1-hard2 --offload dlo-mmap \
    --prompts /data/jooman/k6/results/setB-hard2.json --reference-manifest /nonexistent \
    --no-locks --out-dir "$OUT/sage2-mid-step1-hard2"
status=$?
echo "=== $(date +%H:%M:%S) exit $status"
exit "$status"
