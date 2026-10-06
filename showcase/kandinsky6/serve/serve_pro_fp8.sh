#!/bin/bash
# W1 arm: FP8 DiT (32.3 GB, host-resident) streamed per block to the 5090.
set -uo pipefail
export HF_HOME=/data/jooman/hf
export PATH=/data/jooman/k6/venv/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# glibc raises its mmap threshold dynamically (up to 32 MB) after each large
# free, so most DiT tensors (<= 32 MB) come from heap arenas whose frees are
# never returned to the OS. The layerwise backend copies every block into pinned
# memory and frees the original: without a fixed threshold that free is kept
# and host RSS grows by the size of the DiT during hook installation.
export MALLOC_MMAP_THRESHOLD_=${MALLOC_MMAP_THRESHOLD_:-131072}
export VLLM_OMNI_K6_PIFLOW_DEBUG=${VLLM_OMNI_K6_PIFLOW_DEBUG:-0}
exec /data/jooman/k6/venv/bin/vllm serve ${K6_CKPT:?set K6_CKPT to the FP8 model root} \
  --omni --host 127.0.0.1 --port 8091 \
  --num-gpus 1 \
  --enable-layerwise-offload \
  --disable-multithread-weight-load \
  "$@"
