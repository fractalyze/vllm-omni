#!/bin/bash
# R0: the BF16 reference for the quality gate. The Pro DiT is 60.3 GB in BF16,
# more than this host's RAM, so it cannot be copied into pinned host memory the
# way the FP8 arm is. Distributed layerwise offload without AllGather binds every
# DiT tensor to the mmapped checkpoint (rank-local mmap) and stages one block at a
# time through two host slots, so the weights stream from NVMe through the page
# cache. Slow, and only run once per prompt and seed. Not a timed arm.
set -uo pipefail
export HF_HOME=${HF_HOME:-/data/jooman/hf}
export PATH=/data/jooman/k6/venv/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MALLOC_MMAP_THRESHOLD_=${MALLOC_MMAP_THRESHOLD_:-131072}
exec /data/jooman/k6/venv/bin/vllm serve ${K6_BF16_MODEL:-kandinskylab/Kandinsky-6.0-Pro-distill-5s-Diffusers} \
  --omni --host 127.0.0.1 --port ${K6_PORT:-8094} \
  --num-gpus 1 \
  --enable-distributed-layerwise-offload --dlo-no-use-allgather \
  --disable-multithread-weight-load \
  "$@"
