---
license: mit
base_model: kandinskylab/Kandinsky-6.0-Pro-distill-5s-Diffusers
pipeline_tag: text-to-video
tags:
  - vllm-omni
  - rtx-5090
  - kandinsky6
  - sageattention
  - fp16-accumulate
  - nvfp4
  - inference-optimization
---

# Kandinsky 6 Pro with audio on one RTX 5090: a 5-second clip in 151 s

Kandinsky 6.0 Pro-distill (29B joint video+audio DiT, 56 GiB in BF16) generating 864x480, 121 frames at 24 fps with audio on a single RTX 5090 (32 GB) in a 60 GB host, with [vLLM-Omni](https://github.com/vllm-project/vllm-omni). Upstream cannot serve this checkpoint on that machine; this branch does it in **151 s**, 36% faster than the BF16 reference configuration, at LPIPS 0.11 against that reference. A fast, lossy mode takes **105 s** ([Results](#results)).

- **Code:** [fractalyze/vllm-omni @ `kandinsky6/showcase`](https://github.com/fractalyze/vllm-omni/tree/9424bbbbe04e0e601f913f254e1aaee032ccfe89/showcase/kandinsky6) (pinned commit)
- **This repo:** results, samples and how to reproduce them. **No model weights**: use [kandinskylab/Kandinsky-6.0-Pro-distill-5s-Diffusers](https://huggingface.co/kandinskylab/Kandinsky-6.0-Pro-distill-5s-Diffusers) (MIT).

## What's inside

- **PiFlow sampler** for the distilled checkpoint, which upstream vLLM-Omni lacks.
- **BF16 weights streamed per block** from the mmapped checkpoint, 93% hidden behind compute.
- **SageAttention2 on visual blocks 6-53**, exact attention on the first and last six blocks and on the first sampler step.
- **PiFlow cache:** step 8 reuses step 7's prediction.
- **Hybrid FP16-accumulate GEMM** (Triton, sm_120) on the large linears.
- **`torch.compile`** with a pinned Inductor cache.
- **Fast, lossy mode:** NVFP4 GEMMs and SageAttention3 from step 3 on INT8 weights.

Every change is behind a switch, off by default.

## Results

W1: 864x480, 121 frames, 10 PiFlow steps, guidance 1.0, audio on, one RTX 5090, batch 1, seed 42. Speed: median of a mirrored A B B A session. Quality: LPIPS mean / worst frame over nine prompts, against a compiled BF16 reference generated on the same code and host.

| configuration | W1 request | set A LPIPS | set B LPIPS |
| --- | --: | --- | --- |
| BF16 reference, platform attention | 234.6 s | the reference | the reference |
| round-3 stack (Sage2 mid-blocks, exact step 1, step-8 cache) | 158.9 s | 0.1128 / 0.3675 | -- |
| **final stack**: + hybrid FP16-accumulate GEMM | **151.05 s** | **0.1114 / 0.3492** | **0.1307 / 0.3576** |
| **fast, lossy** (INT8 + NVFP4 + Sage3) | **104.8 s** | CLIP within 0.42% of its base; some shots change | -- |

The reference's wall time is from its own gate run, not the same session. W2 (Pro-5s, 50 steps, CFG 5.0): 1611 s vs 2587 s with platform attention, quality screened on one prompt only. Protocols, commits and hosts: [`measurements.md`](https://github.com/fractalyze/vllm-omni/blob/9424bbbbe04e0e601f913f254e1aaee032ccfe89/showcase/kandinsky6/measurements.md).

## Quick start

Requires an RTX 5090, CUDA 13, Python 3.12, 60 GB of host RAM, ~110 GB of NVMe, and `uv`, `git`, `jq`.

```bash
git clone https://github.com/fractalyze/vllm-omni.git && cd vllm-omni && git checkout 9424bbbbe
uv venv --python 3.12 .venv
VIRTUAL_ENV=.venv uv pip install setuptools_scm
VIRTUAL_ENV=.venv uv pip install vllm==0.31.0 --torch-backend=auto
VIRTUAL_ENV=.venv uv pip install -e .
git clone https://github.com/thu-ml/SageAttention.git ../SageAttention && git -C ../SageAttention checkout d1a57a5
TORCH_CUDA_ARCH_LIST=12.0 VIRTUAL_ENV=.venv uv pip install --no-build-isolation ../SageAttention
echo "export HF_HOME=/path/with/room TORCHINDUCTOR_CACHE_DIR=$PWD/.inductor-cache" > env.sh && source env.sh

PATH=$PWD/.venv/bin:$PATH \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True MALLOC_MMAP_THRESHOLD_=131072 \
VLLM_OMNI_K6_EXACT_ATTN_STEPS=1 VLLM_OMNI_K6_EXACT_ATTN_BLOCKS=6 \
VLLM_OMNI_DLO_STAGE_AHEAD=1 VLLM_OMNI_K6_PIFLOW_CACHE_STEPS=8 VLLM_OMNI_K6_HYBRID_GEMM=1 \
vllm serve kandinskylab/Kandinsky-6.0-Pro-distill-5s-Diffusers --omni --port 8094 --num-gpus 1 \
    --enable-distributed-layerwise-offload --dlo-no-use-allgather --disable-multithread-weight-load \
    --diffusion-attention-config "$(cat showcase/kandinsky6/compute/arms/sage2.json)"
```

When `curl -sf localhost:8094/health` succeeds:

```bash
ID=$(curl -s localhost:8094/v1/videos -F prompt="Close-up portrait of an elderly fisherman ..." \
    -F size=864x480 -F num_frames=121 -F num_inference_steps=10 -F guidance_scale=1.0 -F seed=42 | jq -r .id)
until [ "$(curl -s localhost:8094/v1/videos/$ID | jq -r .status)" = completed ]; do sleep 10; done
curl -s localhost:8094/v1/videos/$ID/content -o clip.mp4
```

The fast mode, the quality-gate scripts and every switch: [`showcase/kandinsky6/README.md`](https://github.com/fractalyze/vllm-omni/blob/9424bbbbe04e0e601f913f254e1aaee032ccfe89/showcase/kandinsky6/README.md).

## Samples

Seed 42, prompts in [`prompts-setA.json`](prompts-setA.json) (b6 is from set B). The reference and final-stack clips are the pairs the LPIPS numbers score.

| prompt | BF16 reference | final stack | fast, lossy |
| --- | --- | --- | --- |
| a1, portrait with speech | [mp4](samples/bf16-reference/a1-portrait-speech.mp4) | [mp4](samples/final-stack/a1-portrait-speech.mp4) | [mp4](samples/fast-lossy-int8/a1-portrait-speech.mp4) |
| a3, sprint start | [mp4](samples/bf16-reference/a3-sprint-start.mp4) | [mp4](samples/final-stack/a3-sprint-start.mp4) | [mp4](samples/fast-lossy-int8/a3-sprint-start.mp4) |
| a7, café menu board | [mp4](samples/bf16-reference/a7-cafe-menu.mp4) | [mp4](samples/final-stack/a7-cafe-menu.mp4) | -- |
| b6, train platform | [mp4](samples/bf16-reference/b6-train-platform.mp4) | [mp4](samples/final-stack/b6-train-platform.mp4) | [mp4](samples/fast-lossy-int8/b6-train-platform.mp4) |

Contact sheets in [`sheets/`](sheets): the final stack's attention arm against the reference on both prompt sets, and the fast mode against its base. On b6 the fast mode's INT8 weights change the shot; the final stack keeps it.

## Limitations

- **One GPU, batch 1, one geometry** (864x480 x 121 frames, sm_120 only).
- **The strict gate's worst-frame bar (0.25) is not met:** the final stack's worst frame is 0.35. Recompiling the BF16 reference alone moves a worst frame by 0.42, so the pipeline's own noise exceeds that bar.
- **The hybrid GEMM changes numerics**; results are always scored against a reference built on the same code.
- **The fast mode is lossy** and changes shots on some prompts.
- **Host needs** 60 GB of RAM and the checkpoint on NVMe.
- **W2 is not quality-gated.**

## Links

- Code: [`showcase/kandinsky6/` @ 9424bbbbe](https://github.com/fractalyze/vllm-omni/tree/9424bbbbe04e0e601f913f254e1aaee032ccfe89/showcase/kandinsky6)
- Record: [`measurements.md`](https://github.com/fractalyze/vllm-omni/blob/9424bbbbe04e0e601f913f254e1aaee032ccfe89/showcase/kandinsky6/measurements.md)
- Where the next 20 s are: [ncu rooflines](https://github.com/fractalyze/vllm-omni/tree/9424bbbbe04e0e601f913f254e1aaee032ccfe89/showcase/kandinsky6/compute) (`ncu-roofline*.md`)
- Checkpoint: [kandinskylab/Kandinsky-6.0-Pro-distill-5s-Diffusers](https://huggingface.co/kandinskylab/Kandinsky-6.0-Pro-distill-5s-Diffusers) (MIT)
- Earlier showcase: [Fractalyze/qwen3-omni-rtx5090-showcase](https://huggingface.co/Fractalyze/qwen3-omni-rtx5090-showcase)
