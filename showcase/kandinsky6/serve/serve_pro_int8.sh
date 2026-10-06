#!/bin/bash
# W1 arm: the exact BF16 Pro-distill checkpoint, quantized to INT8 at load with
# one scale per output row.
#
# Why INT8 and not FP8. Measured on these tensors
# (showcase/kandinsky6/tools/weight_quant_error.py), the median relative error of
# one quantize/dequantize round trip is:
#
#     FP8 E4M3   per-tensor 0.02645   per-row 0.02643
#     INT8       per-tensor 0.02057   per-row 0.00908
#
# FP8's error does not care about the scale, because E4M3 carries its own 4-bit
# exponent and a finer scale only slides the matrix along the exponent ladder
# while the step stays at the 3-bit mantissa. INT8 is fixed point, so there the
# scale *is* the step: per-output-row INT8 costs 2.9x less error for the same
# one byte per weight. The converted FP8 checkpoints beside this script measured
# LPIPS 0.262 against the BF16 reference, against a 0.15 limit, and no keep
# profile or scale granularity moved it.
#
# Why at load and not from a converted checkpoint: nothing has to be converted.
# DiffusionInt8Config quantizes each tensor as the checkpoint streams, so this
# serves the exact published weights, and `ignored_layers` can keep named layers
# in BF16 without rebuilding anything.
#
# W8A8: per-output-channel weight scales with dynamic per-token activation
# scales, so the GEMMs run on INT8 tensor cores. That matters because at W1 this
# pipeline is not purely bandwidth-bound -- with the platform's cuDNN attention
# the streamed BF16 arm runs 21.9 s/step against a stream worth about 15 s/step,
# so the GEMMs have to get faster, not just smaller.
set -uo pipefail
export HF_HOME=${HF_HOME:-/data/jooman/hf}
export PATH=/data/jooman/k6/venv/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# glibc raises its mmap threshold dynamically after each large free, so most DiT
# tensors come from heap arenas whose frees are never returned to the OS, and the
# offload backend frees a DiT's worth during hook installation. Pinning the
# threshold keeps host RSS from growing by the size of the model.
export MALLOC_MMAP_THRESHOLD_=${MALLOC_MMAP_THRESHOLD_:-131072}
exec /data/jooman/k6/venv/bin/vllm serve ${K6_BF16_MODEL:-kandinskylab/Kandinsky-6.0-Pro-distill-5s-Diffusers} \
  --omni --host 127.0.0.1 --port ${K6_PORT:-8095} \
  --num-gpus 1 \
  --diffusion-quantization-config ${K6_QUANT:-int8} \
  --enable-layerwise-offload \
  --disable-multithread-weight-load \
  "$@"
