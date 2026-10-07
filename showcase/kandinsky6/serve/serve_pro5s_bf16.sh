#!/bin/bash
# W2: Kandinsky 6.0 **Pro-5s** (not the distill), streamed BF16.
#
# Same placement as the W1 headline arm -- distributed layerwise offload with
# rank-local mmap, so every DiT tensor is bound to the mmapped checkpoint and
# one block at a time is staged through two host slots. The DiT is the same size
# as the distill's (30.1B parameters, 60.3 GB all-BF16) and still does not fit a
# 32 GB board, so it streams from NVMe exactly as W1 does.
#
# What differs from W1 and is the reason this script exists separately:
#
# - The checkpoint is **mixed dtype**: 18.9 GB of its 69.7 GB on disk is FP32
#   (1267 tensors, 4.73B parameters), the rest BF16. Streaming it as stored
#   would move 69.7 GB a step instead of 60.3 GB, so the run is worth watching
#   for whether the loader casts on the way in.
# - The sampler is `FlowMatchEulerDiscreteScheduler` with shift 5.0, not the
#   distill's PiFlow, and the headline request is 50 steps at CFG 5.0 rather
#   than 10 steps at guidance 1.0.
# - The pipeline is `Kandinsky6TI2VAPipeline`, which takes an optional image.
#   A text-only request leaves it out.
set -uo pipefail
export HF_HOME=${HF_HOME:-/data/jooman/hf}
export PATH=/data/jooman/k6/venv/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MALLOC_MMAP_THRESHOLD_=${MALLOC_MMAP_THRESHOLD_:-131072}
# The same Inductor cache every arm and reference on this host reads, so that a
# compiled arm and a compiled reference make the same autotune choices.
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-/data/jooman/k6/inductor-cache}
exec /data/jooman/k6/venv/bin/vllm serve ${K6_W2_MODEL:-kandinskylab/Kandinsky-6.0-Pro-5s-Diffusers} \
  --omni --host 127.0.0.1 --port ${K6_PORT:-8095} \
  --num-gpus 1 \
  --enable-distributed-layerwise-offload --dlo-no-use-allgather \
  --disable-multithread-weight-load \
  "$@"
