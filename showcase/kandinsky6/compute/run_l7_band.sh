#!/bin/bash
# L7: is the exact-attention edge band still worth anything now that sampler
# step 1 is exact?
#
# Round 1 found the band worth 24% of the error at 12 exact blocks and nothing
# at 6 ([[k6c-s04]]), and separately found that making step 1 exact beat twelve
# exact blocks at half the cost. Those two levers may be buying the same thing:
# both protect the moment the trajectory picks which sample it lands on. If they
# do, the band is now redundant and costs ~1 s a request per exact block -- 12 s
# of a 182.6 s request for the 12 the headline arm keeps.
#
# Three points on the dial, every other setting identical, step 1 exact in all
# three. The 12-block point is the shipped headline arm and already has full-set
# numbers, so only 6 and 0 are generated here.
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HOME=/data/jooman/hf
export VLLM_OMNI_K6_EXACT_ATTN_STEPS=1
R=/data/jooman/k6/results/gate-pro
for arm in sage2-wide:l7-edge6 sage2-edge0:l7-edge0; do
  file="${arm%%:*}"; name="${arm##*:}"
  echo "=== $name (compute/arms/$file.json) ==="
  /data/jooman/k6/venv/bin/python compute/gate_pro.py \
    --arm "compute/arms/$file.json" --name "$name" --offload dlo-mmap \
    --prompts /data/jooman/k6/prompts/L7-screen.json \
    --reference-manifest /data/jooman/k6/ref/L7-screen/manifest.json \
    --out-dir "$R/$name" 2>&1 | grep -E "s ->|ready|warm|FAIL|Error" || true
done
