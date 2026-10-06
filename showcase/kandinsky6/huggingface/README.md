<!--
DRAFT, NOT PUBLISHED. Nothing here is uploaded: no weights leave the
fractalyze org, and publishing needs the user's approval and the upstream
license check below.
-->
---
base_model: kandinskylab/Kandinsky-6.0-Pro-distill-5s-Diffusers
library_name: vllm-omni
tags:
- text-to-video
- text-to-audio-video
- int8
- weight-only-quantization
- rtx-5090
---

# Kandinsky 6.0 Pro-distill 5s: INT8 weight-only DiT for one RTX 5090 (draft)

This is the Kandinsky 6.0 Pro-distill DiT with its linear weights stored as
INT8, with one FP32 scale per output row. Every other component (Qwen2.5-VL and
CLIP text encoders, video VAE, audio VAE) is unchanged and loaded from the base
model. Built to serve 5-second 864x480 video with audio on a single RTX 5090
(32 GB) with [vLLM-Omni](https://github.com/fractalyze/vllm-omni).

## What is quantized

| | |
|---|---|
| Format | INT8 symmetric (+-127), one FP32 scale per output row; `quant_method: int8, weight_only: true` |
| Layers in INT8 | 1976 linear weights of the DiT |
| Layers kept BF16 | 15 (embeddings and output heads: `--keep minimal`) |
| DiT size | 30.2 GB (BF16: 60.3 GB) |
| Compute | BF16. Weights are dequantized on the GPU before each GEMM, since sm_120 has no INT8 GEMM |

The served model is bit-identical to the BF16 model with its weights
fake-quantized to this INT8 grid. The only quality change is the rounding.

## Serving

```bash
VLLM_OMNI_K6_EXACT_ATTN_STEPS=1 K6_MEMMAX=44G K6_CKPT=<this checkpoint's model root> \
  showcase/kandinsky6/serve/run_capped.sh showcase/kandinsky6/serve/serve_pro_fp8.sh \
  --diffusion-attention-config "$(cat showcase/kandinsky6/compute/arms/tuned.json)"
```

That is the INT8 + Sage2 + exact-first-step arm; drop the attention config and
the environment switch for platform attention.

Host requirements: about 44 GB of host RAM for the server (the DiT is pinned
in host memory and streamed per block), and one 32 GB GPU.

## Quality and speed (W1: 864x480, 121 frames, 10 pi-Flow steps, guidance 1.0)

One RTX 5090 (build-server-2), one mirrored session (n=4 timed per arm):

| configuration | W1 request | vs FP8 baseline |
|---|---:|---:|
| FP8 per-tensor DiT (baseline) | 174.10 s | -- |
| **this checkpoint + Sage2 + exact first step** | **168.07 s** | **-3.5%** |

LPIPS against the BF16 model (set mean / worst frame):

| prompt set | vs eager BF16 | working gate (floor x 1.25) | verdict |
|---|---|---|---|
| A (9 prompts) | 0.1735 / 0.4426 | 0.1819 / 0.5200 | pass |
| B (b3, b6 screened) | b6 0.579 / 0.596, b3 0.387 / 0.436 | 0.1922 / 0.4245 | **fail** |

The working gate passes on set A and fails on set B. On b6 the INT8 rounding
changes the scene from the first frame. Set B's working gate is met only by
keeping the weights BF16 (BF16 streamed + Sage2 on blocks 6-53 + exact first
step, 188.9 s). The user's gate (mean <= 0.15, worst frame <= 0.25) is not
met by either. Full tables: `showcase/kandinsky6/measurements.md`. **Publish
this card only with that limitation stated.**

## Reproduce the checkpoint

```bash
python showcase/kandinsky6/tools/quantize_dit_fp8.py \
  --src <Kandinsky-6.0-Pro-distill-5s-Diffusers snapshot> \
  --dst <out> --keep minimal --format int8
```

## License and attribution

TBD: confirm that the base model's license permits redistributing a quantized
derivative, and copy its attribution requirements here, before anything is
published.
