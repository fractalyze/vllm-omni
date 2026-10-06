#!/bin/bash
# W1 arm: the exact BF16 Pro-distill checkpoint, quantized to FP8 at load with
# one scale per output row.
#
# Why at load and not from a converted checkpoint. vLLM's *serialized* fp8
# method builds a PerTensorScaleParameter, so a checkpoint with per-row scales
# fails its shape assertion -- which is why the two converted checkpoints on
# these hosts carry `scale_granularity: tensor`, one scalar for a 4096x16384
# matrix. vLLM's online methods carry a group shape instead, and
# `fp8_per_channel` is row=-1 col=1 with `activation=None`: one scale per output
# row, weights in FP8, activations left in BF16. No converter, no second
# checkpoint, and the better recipe.
#
# Why it is also the speed arm. At W1 this pipeline is weight-traffic-bound, not
# compute-bound: the BF16 DiT is 60.3 GB and all 10 steps re-read all of it. FP8
# weights are ~30 GB, which both halves the traffic and fits in this host's page
# cache, so the NVMe re-read stops rather than merely halves.
set -uo pipefail
export HF_HOME=${HF_HOME:-/data/jooman/hf}
export PATH=/data/jooman/k6/venv/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# glibc raises its mmap threshold dynamically after each large free, so most DiT
# tensors come from heap arenas whose frees are never returned to the OS; the
# offload backend frees a DiT's worth during hook installation. Pinning the
# threshold keeps host RSS from growing by the size of the model.
export MALLOC_MMAP_THRESHOLD_=${MALLOC_MMAP_THRESHOLD_:-131072}
exec /data/jooman/k6/venv/bin/vllm serve ${K6_BF16_MODEL:-kandinskylab/Kandinsky-6.0-Pro-distill-5s-Diffusers} \
  --omni --host 127.0.0.1 --port ${K6_PORT:-8095} \
  --num-gpus 1 \
  --diffusion-quantization-config ${K6_QUANT:-fp8_per_channel} \
  --enable-layerwise-offload \
  --disable-multithread-weight-load \
  "$@"
