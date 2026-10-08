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

Kandinsky 6.0 Pro-distill (a 29B joint video+audio DiT, 56 GiB in BF16) generating 864x480, 121 frames at 24 fps with audio on a single RTX 5090 (32 GB) in a 60 GB host, with [vLLM-Omni](https://github.com/vllm-project/vllm-omni). Upstream cannot serve this checkpoint on that machine; this branch does it in **151.05 s, -36% against the BF16 reference configuration's own gate run** (round-1 code, not the same session), at set-A LPIPS 0.1114 against a BF16 reference built from the same code. A separate fast, lossy mode takes **104.8 s** ([Results](#results)).

- **Code:** [fractalyze/vllm-omni @ `kandinsky6/showcase`](https://github.com/fractalyze/vllm-omni/tree/9310bdad259dd75af0098526a34fa4105d704b26/showcase/kandinsky6) (pinned commit)
- **This repo:** results, samples and how to reproduce them. **No model weights**: use [kandinskylab/Kandinsky-6.0-Pro-distill-5s-Diffusers](https://huggingface.co/kandinskylab/Kandinsky-6.0-Pro-distill-5s-Diffusers) (MIT). The fast mode's INT8 checkpoint is built locally from it with one command.

## What's inside

- **A PiFlow port:** the 10-step policy-rollout sampler the distilled checkpoint needs, which upstream vLLM-Omni lacks.
- **Exact BF16 weights streamed per block** from the mmapped checkpoint: one 0.9 GiB block at a time through two GPU slots, the block after next packed on a worker thread; ~57 GB/s over PCIe, 93% hidden behind compute.
- **SageAttention2 on visual blocks 6-53** from sampler step 2, cuDNN attention on the first and last six blocks and on the whole first step.
- **PiFlow cache:** step 8 reuses step 7's prediction instead of running the DiT.
- **A hybrid FP16-accumulate GEMM** (Triton, sm_120). FP16 accumulation inside each 32-element K block, FP32 across blocks, on the large-M linears. An FP16 range audit keeps in BF16 the layers (modulation, time embeddings) that would underflow in FP16.
- **`torch.compile`** with a pinned Inductor cache, so the server and its quality reference use the same kernels.
- **Fast, lossy mode:** NVFP4 GEMMs and SageAttention3 from step 3, on an INT8 weight-only DiT held in pinned host RAM.

Every change is behind a `VLLM_OMNI_K6_*` / `VLLM_OMNI_DLO_*` switch or an attention-config file, off by default.

## Results

W1: 864x480, 121 frames, 10 PiFlow steps, guidance 1.0, audio on. One RTX 5090, batch 1, prompt a1, seed 42. Each speed row comes from one mirrored session (servers visited A B B A, warm-up then timed requests) and is compared with that session's own control, except the BF16 reference, whose wall time is its own gate run on round-1 code. Quality: LPIPS mean / worst frame over nine set-A prompts, against a compiled BF16 reference generated on the same code, host and Inductor cache.

| configuration | W1 request | change | set A LPIPS, mean / worst frame |
| --- | --: | --: | --- |
| BF16 reference, platform attention | 234.6 s | -- | the reference |
| final stack, round 3 (Sage2 mid-blocks, exact step 1, step-8 cache) | 158.90 s | -32% vs the reference's gate run (another session) | 0.1128 / 0.3675 |
| **final stack (head)**: + hybrid FP16-accumulate GEMM | **151.05 s** | **-36% vs the reference's gate run (another session)** | **0.1114 / 0.3492** |
| **fast, lossy (INT8 + NVFP4 + Sage3)** | **104.8 s** | -36% vs its base, the round-3 final stack on bs1 (164.3 s) | lossy: CLIP within 0.42% of base; b6 changes shot |

On set B (nine harder prompts) the final stack scores **0.1307 / 0.3576** against the same kind of reference: mean inside 0.15, worst frame (b6) over 0.25 as on set A.

- The fast mode was measured on a second 5090 host (bs1), on prompts a3, b6 and a1, against the round-3 final stack on that host; it does not include the hybrid GEMM.
- W2 (the non-distilled Pro-5s checkpoint, 50 steps, CFG 5.0) takes 1610.8 s against 2587.4 s with platform attention; its quality was screened on one prompt only.
- Every protocol, commit and host: [`measurements.md`, "Headline"](https://github.com/fractalyze/vllm-omni/blob/9310bdad259dd75af0098526a34fa4105d704b26/showcase/kandinsky6/measurements.md).

## Quick start

Requires an RTX 5090, CUDA 13, Python 3.12, 60 GB of host RAM, ~110 GB of NVMe, and `uv`, `git` and `jq` on `PATH`. `093ed22b5` is the serving code every number here was measured on.

```bash
git clone https://github.com/fractalyze/vllm-omni.git && cd vllm-omni && git checkout 093ed22b5
uv venv --python 3.12 .venv
VIRTUAL_ENV=.venv uv pip install setuptools_scm
VIRTUAL_ENV=.venv uv pip install vllm==0.31.0 --torch-backend=auto
VIRTUAL_ENV=.venv uv pip install -e .
git clone https://github.com/thu-ml/SageAttention.git ../SageAttention && git -C ../SageAttention checkout d1a57a5
TORCH_CUDA_ARCH_LIST=12.0 VIRTUAL_ENV=.venv uv pip install --no-build-isolation ../SageAttention
# Source env.sh in every shell (server, request): the cache must be the same one the reference used.
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

The fast mode, the quality-gate scripts and every switch are in the branch's [`showcase/kandinsky6/README.md`](https://github.com/fractalyze/vllm-omni/blob/9310bdad259dd75af0098526a34fa4105d704b26/showcase/kandinsky6/README.md).

## Samples

Seed 42, W1, the prompts in [`prompts-setA.json`](prompts-setA.json) (b6 is from the second set).

| prompt | BF16 reference | final stack | fast, lossy |
| --- | --- | --- | --- |
| a1, portrait with speech | [mp4](samples/bf16-reference/a1-portrait-speech.mp4) | [mp4](samples/final-stack/a1-portrait-speech.mp4) | [mp4](samples/fast-lossy-int8/a1-portrait-speech.mp4) |
| a3, sprint start | [mp4](samples/bf16-reference/a3-sprint-start.mp4) | [mp4](samples/final-stack/a3-sprint-start.mp4) | [mp4](samples/fast-lossy-int8/a3-sprint-start.mp4) |
| a7, café menu board | [mp4](samples/bf16-reference/a7-cafe-menu.mp4) | [mp4](samples/final-stack/a7-cafe-menu.mp4) | -- |
| b6, train platform | [mp4](samples/bf16-reference/b6-train-platform.mp4) | [mp4](samples/final-stack/b6-train-platform.mp4) | [mp4](samples/fast-lossy-int8/b6-train-platform.mp4) |

- **a1:** the easiest prompt. Every arm stays close to the reference; listen for the voice and the gulls.
- **a3:** fast motion, where the attention approximation and the compiler's own variance show most; compare the runner's legs and the bystanders. In the fast mode the shorts change colour.
- **a7:** rendered text. Compare the menu board's lettering between the reference and the final stack; signage glyphs are what an approximation loses first.
- **b6:** the prompt that decided the design. Its framing is set in the first sampler step, which the final stack keeps exact, and the final stack keeps the reference's framing; b6 is still its worst set-B prompt (0.308 mean). The fast mode's INT8 weights change it anyway: a different shot down the platform, with the "WEST" board barely legible.

The BF16-reference and final-stack clips are the exact pairs the G1 numbers score: the same-code reference and the head's outputs for set A and set B (code `093ed22b5`).

Contact sheets:

- [`sheets/sheet-sage2-mid-step1-setA.jpg`](sheets/sheet-sage2-mid-step1-setA.jpg), [`sheets/sheet-sage2-mid-step1-setB.jpg`](sheets/sheet-sage2-mid-step1-setB.jpg): the attention arm (Sage2 on blocks 6-53, exact step 1) against the reference, every prompt of each set.
- [`sheets/compare-a3-sprint-start-sage2-mid-step1-setA.jpg`](sheets/compare-a3-sprint-start-sage2-mid-step1-setA.jpg): eight frames of a3, the arm's worst prompt, reference above arm. Same sprinter, track, camera move and crowd; the arm frames the shot slightly tighter. LPIPS scores that global shift near its worst, though a viewer would call both takes correct.
- [`sheets/compare-b6-the-prompt-that-decided-it.jpg`](sheets/compare-b6-the-prompt-that-decided-it.jpg): b6 across the reference, Sage2 on blocks 6-53 alone, and the same with an exact first step. The approximate first step frames the platform differently; the exact one restores the reference's framing and signage. That is why step 1 stays exact: an error in the first step changes which sample the trajectory lands on.
- [`sheets/h5-int8-vs-base.jpg`](sheets/h5-int8-vs-base.jpg), [`sheets/h5-bf16-vs-base.jpg`](sheets/h5-bf16-vs-base.jpg): the fast mode on INT8 and on BF16 weights against its base. The BF16 variant (139.4 s) keeps the scenes.

## Limitations

- **One GPU, batch 1, one geometry.** Every number is one request at a time at 864x480 x 121 frames on an RTX 5090 (sm_120). The hybrid kernel, SageAttention builds and NVFP4 path target that GPU.
- **The strict quality gate's worst-frame bar is not met, and cannot be measured against.** The gate asks for a set mean <= 0.15 and every frame <= 0.25. The final stack's mean (0.1114) is inside; its worst frame (0.3492) is not. Recompiling the BF16 reference alone, compiled against eager, moves a worst frame by 0.416, so this pipeline's own noise exceeds that bar.
- **The hybrid GEMM changes numerics.** FP16 operands and FP16 accumulation per 32-term block move the video about as much as recompiling does; results here are always scored against a reference built on the same code.
- **The fast mode is lossy.** It passes the CLIP check but changes the shot on some prompts (b6) and details on others (a3). It is a labelled fast mode, never a quality-gated one.
- **No INT8 GEMM on sm_120**, so INT8 is a storage format here (dequantized to BF16 for the BF16 steps).
- **Host needs:** 60 GB of RAM, and the checkpoint on NVMe, since the final stack streams 56 GiB a step.
- **W2 is not quality-gated**, and set B (a second, harder prompt set) was scored for the head only.

## Links

- Code: [`showcase/kandinsky6/` @ 9310bdad259dd75af0098526a34fa4105d704b26](https://github.com/fractalyze/vllm-omni/tree/9310bdad259dd75af0098526a34fa4105d704b26/showcase/kandinsky6)
- Record: [`measurements.md` @ 9310bdad259dd75af0098526a34fa4105d704b26](https://github.com/fractalyze/vllm-omni/blob/9310bdad259dd75af0098526a34fa4105d704b26/showcase/kandinsky6/measurements.md)
- Where the next 20 s are: the [ncu rooflines](https://github.com/fractalyze/vllm-omni/tree/9310bdad259dd75af0098526a34fa4105d704b26/showcase/kandinsky6/compute) (`ncu-roofline*.md`)
- Checkpoint: [kandinskylab/Kandinsky-6.0-Pro-distill-5s-Diffusers](https://huggingface.co/kandinskylab/Kandinsky-6.0-Pro-distill-5s-Diffusers) (MIT)
- Earlier showcase on this fork: [Fractalyze/qwen3-omni-rtx5090-showcase](https://huggingface.co/Fractalyze/qwen3-omni-rtx5090-showcase)
