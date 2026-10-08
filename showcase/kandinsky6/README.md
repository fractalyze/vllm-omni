# Kandinsky 6 Pro on one RTX 5090

Kandinsky 6.0 **Pro-distill** (a 29B joint video+audio DiT, **56 GiB in BF16**)
serving 864x480, 121 frames at 24 fps with audio, on one RTX 5090 (32 GB) in a
host with 60 GB of RAM, from this branch of vLLM-Omni. Upstream cannot serve
this checkpoint on that machine at all. It has no PiFlow scheduler for the
distilled model, and its recipe needs an H100 80 GB and about 90 GB of free host
memory for its CPU offload.

On that box a W1 request (10 PiFlow steps, guidance 1.0, audio on) takes
**151.05 s** with the final stack. That is **-36% against the BF16 reference
configuration's own gate run (234.6 s; round-1 code, not the same session)**,
and its set-A quality is LPIPS **0.1114 mean /
0.3492 worst frame** against a BF16 reference compiled from the same code
(see [Limitations](#limitations) for what that worst frame means). A separate
**fast, lossy** mode takes it to **104.8 s**. It is judged on text-video
agreement (CLIP within 0.42% of the base) and contact sheets, not on LPIPS, and
it changes the shot on some prompts.

[measurements.md](measurements.md) is the record: every number, how it was
taken, and what each one superseded. [PLAN.md](PLAN.md) states the goal and the
rules. [huggingface/README.md](huggingface/README.md) is the Hugging Face card
for these results.

## What's inside

The final stack, mechanism first. Each piece is in the request path only when its
switch is set; with none set, the branch serves the BF16 reference.

- **A PiFlow port.** The distilled checkpoint samples with PiFlow (a 10-step
  policy-rollout sampler), which upstream vLLM-Omni does not have
  (`vllm_omni/diffusion/models/kandinsky6/scheduling_kandinsky6_piflow.py`).
- **BF16 weights streamed per block from the mmapped checkpoint.** Distributed
  layerwise offload without AllGather binds every DiT tensor to the mmapped
  safetensors and copies one 0.9 GiB block at a time to the GPU through two
  device slots, while the previous block computes. A worker thread packs the
  block after next (`VLLM_OMNI_DLO_STAGE_AHEAD=1`). The weights stay exact: 504
  GiB cross PCIe per request at ~57 GB/s, 93% hidden behind compute.
- **SageAttention2 on visual blocks 6-53, exact attention elsewhere.** INT8 QK /
  FP8 PV attention on the 50,220-token visual self-attention of blocks 6-53 and
  sampler steps 2-10. cuDNN attention on the first and last six blocks and the
  whole first step, where an approximation changes the video most.
- **PiFlow cache on step 8.** Step 8 skips the DiT and reuses step 7's
  prediction (`VLLM_OMNI_K6_PIFLOW_CACHE_STEPS=8`).
- **The bias unfused on the large projections.** cuBLAS picks a ~21% slower
  tile for `addmm` with a bias at these shapes, so q/k/v/out run bias-free and
  the bias is added by the next fused kernel.
- **A hybrid FP16-accumulate GEMM.** On sm_120 an FP16 MMA that accumulates in
  FP16 runs about 1.5x the FP32-accumulate rate. The Triton kernel accumulates
  each 32-element K block in FP16 and sums the blocks in FP32, on the 12
  large-M linears of each visual block; calls under 2048 rows stay on cuBLAS.
  An FP16 range audit keeps in BF16 the layers (modulation, time embeddings)
  that would underflow in FP16. A placeholder bias pointer stops Inductor
  writing every bias-free call's FP16 input twice.
- **`torch.compile` with a pinned Inductor cache.** Every DiT block is compiled.
  One cache directory per host keeps the server and the BF16 reference on the
  same autotune choices, which the quality gate needs.

The **fast, lossy mode** adds, from sampler step 3:
- NVFP4 GEMMs, with weight and activation quantized on the device per call
  (vLLM's CUTLASS sm_120 kernel).
- SageAttention3 on every visual block.
- An INT8 weight-only checkpoint (30 GB) staged from pinned host RAM instead of
  streaming BF16 from NVMe.

**Every speed path is behind a `VLLM_OMNI_K6_*` / `VLLM_OMNI_DLO_*` switch or an
attention-config JSON, off by default** ([Switches](#switches)).

## Results

W1, one RTX 5090, batch 1, prompt a1, seed 42. Every speed row comes from one
mirrored session (`bench/abba.py`, visits A B B A, a warm-up then timed requests,
every GPU lock held) and is compared with that session's own control; the
exception is the BF16 reference, whose wall time is its own gate run on round-1
code. Quality is
set A (nine prompts) against a compiled BF16 reference generated on the same code
and host from the same Inductor cache. The full protocol is in
[measurements.md, "Headline"](measurements.md#headline-the-numbers-the-readme-and-the-model-card-quote).

| configuration | W1 request | change | set A LPIPS, mean / worst frame | numerics | measured on |
|---|---:|---:|---|---|---|
| BF16 reference: platform cuDNN attention, compiled | 234.6 s | -- | the reference | exact | bs2, round 1, n=9 |
| sage2-mid + exact step 1 + background staging + unfused bias | 173.42 s | -26% vs the reference's gate run (another session) | -- | approximate (Sage2) | bs2, `2c7b899dc`, n=4 |
| **final stack, round 3**: + PiFlow cache on step 8 | **158.90 s** | -8.4% vs the row above | **0.1128 / 0.3675** | approximate (cache) | bs2, `2c7b899dc`, n=4 |
| + hybrid FP16-accumulate GEMM, every linear | 153.82 s | -4.2% vs its control (160.54 s) | 0.1126 / 0.4597 | changes numerics | bs2, `747ef44f8`, n=4 |
| **final stack, round 5 (head)**: hybrid on large M only + placeholder bias pointer | **151.05 s** | -1.1% vs its control (152.70 s) | **0.1114 / 0.3492** | placeholder exact; the gate moves small-M linears back to BF16 | bs2, `794693892`, n=4 |
| **fast, lossy (H5-INT8)**: NVFP4 GEMMs + Sage3 from step 3, INT8 weights | **104.8 s** | **-36.2%** vs its base, the round-3 final stack on bs1 (164.3 s) | lossy: CLIP +0.42% vs base; b6 changes shot | lossy | bs1, `b9ede41fb`, n=3 |

Set B (nine harder prompts, scored at the write-up for the head only): **0.1307 / 0.3576** (worst frame b6) against
the same-code reference. The mean is inside 0.15; the worst frame, as on set A, is not.

- **Read each change against its own row's control.** The sessions ran on a
  shared host hours apart, and absolute walls drift by 1-2 s between them.
- **The 0.4597 worst frame** belongs to the all-linears configuration. The
  shipped default keeps small-M linears on BF16 and scores 0.3492.
- **The fast mode was measured on bs1**, on a3, b6 and a1, against the round-3
  final stack, without the hybrid GEMM.

**W2** (the non-distilled Pro-5s checkpoint: 125 frames, 50 steps, CFG 5.0):

| configuration | W2 request | per step | quality |
|---|---:|---:|---|
| BF16 streamed, platform attention | 2587.4 s | 51.75 s | the reference |
| sage2-mid + exact step 1 | 1610.8 s | 32.22 s | screened on a1 only (0.0543 / 0.0920); not gated |
| upstream recipe, one H100 80 GB (for scale) | 751.7 s | 14.4 s | -- |

## Quick start

Requires an RTX 5090 (sm_120), CUDA 13, Python 3.12, a host with 60 GB of RAM,
an NVMe disk with ~110 GB free for the checkpoint (76 GB) and, for the fast
mode, its INT8 transformer (30 GB), and `uv`, `git` and `jq` on `PATH` (`uv`:
`curl -LsSf https://astral.sh/uv/install.sh | sh`).

`093ed22b5` is the serving code every number below was measured on; later
commits on the branch change only documentation and measurement scripts.

```bash
git clone https://github.com/fractalyze/vllm-omni.git && cd vllm-omni
git checkout 093ed22b5
uv venv --python 3.12 .venv
VIRTUAL_ENV=.venv uv pip install setuptools_scm
VIRTUAL_ENV=.venv uv pip install vllm==0.31.0 --torch-backend=auto
VIRTUAL_ENV=.venv uv pip install -e .
VIRTUAL_ENV=.venv uv pip install lpips av          # only for the gate scripts

# SageAttention 2 (both modes) and 3 (fast mode), built for sm_120.
git clone https://github.com/thu-ml/SageAttention.git ../SageAttention
git -C ../SageAttention checkout d1a57a5
TORCH_CUDA_ARCH_LIST=12.0 VIRTUAL_ENV=.venv uv pip install --no-build-isolation ../SageAttention
TORCH_CUDA_ARCH_LIST=12.0 VIRTUAL_ENV=.venv uv pip install --no-build-isolation ../SageAttention/sageattention3_blackwell

# Every shell below (server, request, gate) needs these two variables.
cat > env.sh <<ENV
export HF_HOME=/path/with/room                        # the checkpoint is downloaded here on first start
export TORCHINDUCTOR_CACHE_DIR=$PWD/.inductor-cache   # one cache for the reference and every arm
ENV
```

Run `source env.sh` from the clone's root in every new shell, before each block
below. A server started without it downloads the checkpoint to
`~/.cache/huggingface` and compiles into a cache of its own, and the gate's
reference and arm then no longer share kernels.

**The final stack.**

```bash
source env.sh
PATH=$PWD/.venv/bin:$PATH \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True MALLOC_MMAP_THRESHOLD_=131072 \
VLLM_OMNI_K6_EXACT_ATTN_STEPS=1 VLLM_OMNI_K6_EXACT_ATTN_BLOCKS=6 \
VLLM_OMNI_DLO_STAGE_AHEAD=1 VLLM_OMNI_K6_PIFLOW_CACHE_STEPS=8 \
VLLM_OMNI_K6_HYBRID_GEMM=1 \
vllm serve kandinskylab/Kandinsky-6.0-Pro-distill-5s-Diffusers --omni --port 8094 --num-gpus 1 \
    --enable-distributed-layerwise-offload --dlo-no-use-allgather --disable-multithread-weight-load \
    --diffusion-attention-config "$(cat showcase/kandinsky6/compute/arms/sage2.json)"
```

Drop every `VLLM_OMNI_*` switch and the attention config and the same command
serves the BF16 reference. `MALLOC_MMAP_THRESHOLD_` stops glibc keeping freed
staging buffers in its heap, which otherwise grows host RSS by the size of the
DiT. On a host shared with other work, `systemd-run --user --scope -p
MemoryMax=40G -p MemorySwapMax=0 <command>` caps the server instead of the host
(`serve/run_capped.sh`). The server is ready when `curl -sf localhost:8094/health`
succeeds; its first request also compiles the DiT blocks and takes ~25 s longer.

**The fast, lossy mode.** Build the INT8 weight-only checkpoint once. The
destination is a full model root: the INT8 transformer plus symlinks to the
other components.

```bash
source env.sh
SNAP=$(ls -d $HF_HOME/hub/models--kandinskylab--Kandinsky-6.0-Pro-distill-5s-Diffusers/snapshots/*/ | head -1)
.venv/bin/python showcase/kandinsky6/tools/quantize_dit_fp8.py --src $SNAP --dst ckpt/pro-distill-int8 \
    --keep minimal --format int8

PATH=$PWD/.venv/bin:$PATH \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True MALLOC_MMAP_THRESHOLD_=131072 \
VLLM_OMNI_K6_EXACT_ATTN_STEPS=2 VLLM_OMNI_K6_FP8_GEMM_AFTER_STEP=2 VLLM_OMNI_K6_STEP_GEMM_FORMAT=nvfp4 \
VLLM_OMNI_K6_PIFLOW_CACHE_STEPS=8 \
vllm serve ckpt/pro-distill-int8 --omni --port 8094 --num-gpus 1 \
    --enable-layerwise-offload --disable-multithread-weight-load \
    --diffusion-attention-config "$(cat showcase/kandinsky6/compute/arms/h5-sage3.json)"
```

The INT8 DiT is pinned in host memory (about 44 GB of host RAM for the server).

**A request** (prompt a1 of set A), from any shell. `POST /v1/videos` takes form
fields and returns a job id; fetch the MP4 when its status is `completed`.

```bash
ID=$(curl -s localhost:8094/v1/videos \
    -F prompt="Close-up portrait of an elderly fisherman with a weathered face and white stubble, looking into the camera and speaking slowly in a low voice. Shallow depth of field, soft overcast daylight, salt spray in the air. Audio: a calm male voice speaking, gulls in the distance, gentle waves." \
    -F size=864x480 -F num_frames=121 -F num_inference_steps=10 -F guidance_scale=1.0 -F seed=42 | jq -r .id)
until [ "$(curl -s localhost:8094/v1/videos/$ID | jq -r .status)" = completed ]; do sleep 10; done
curl -s localhost:8094/v1/videos/$ID/content -o a1.mp4      # H.264 video + AAC audio, 121 frames at 24 fps
```

**The quality gate.** Both prompt sets are generated against port 8094, so the
server behind it must be swapped between them. Generating both against the same
server scores it against itself: LPIPS near 0, a gate that cannot fail.

1. Start the **BF16 reference** server: the final-stack command with every
   `VLLM_OMNI_*` switch and the `--diffusion-attention-config` removed. Wait for
   `/health`.
2. Generate the reference set (about 9 x 235 s):
   ```bash
   source env.sh; cd showcase/kandinsky6/bench
   ../../../.venv/bin/python make_refs.py --base-url http://127.0.0.1:8094 --prompts prompts/setA.json \
       --out runs/ref/setA --label bf16-reference
   ```
3. **Stop the reference server** and start the **final stack** in its place. Wait
   for `/health`.
4. Generate the arm's set (about 9 x 151 s) and score it against the reference:
   ```bash
   source env.sh; cd showcase/kandinsky6/bench
   ../../../.venv/bin/python make_refs.py --base-url http://127.0.0.1:8094 --prompts prompts/setA.json \
       --out runs/final/setA --label final-stack
   ../../../.venv/bin/python quality.py runs/ref/setA runs/final/setA --prompts prompts/setA.json \
       --out runs/final/setA/vs_ref.json      # G1: LPIPS mean / worst frame, per prompt and set
   ```

Generate the reference on the same code, on the same host and with the same
`TORCHINDUCTOR_CACHE_DIR` as the arm, or the score measures the code change too
([measurements.md, "Round 3 / attribution"](measurements.md)). Speed claims use
`bench/abba.py` with two arm JSONs (`arms/`), which holds the GPU locks and
interleaves the servers A B B A. The arm JSONs name this project's host paths;
edit `command` for yours.

## Switches

| switch | effect | final stack | fast mode |
|---|---|:-:|:-:|
| `--enable-distributed-layerwise-offload --dlo-no-use-allgather` | stream the BF16 DiT one block at a time from the mmapped checkpoint | yes | |
| `--enable-layerwise-offload` | stage a pre-quantized DiT from pinned host RAM per block | | yes |
| `--diffusion-attention-config <json>` | per-role attention backends: `compute/arms/sage2.json` (Sage2 on visual self-attention), `h5-sage3.json` (Sage3) | `sage2.json` | `h5-sage3.json` |
| `VLLM_OMNI_K6_EXACT_ATTN_STEPS=k` | the first k sampler steps use exact attention everywhere | 1 | 2 |
| `VLLM_OMNI_K6_EXACT_ATTN_BLOCKS=n` | the first and last n visual blocks use exact attention every step | 6 | |
| `VLLM_OMNI_DLO_STAGE_AHEAD=1` | a worker thread packs the block after next while the current one computes (exact) | yes | |
| `VLLM_OMNI_K6_PIFLOW_CACHE_STEPS=8` | sampler step 8 reuses step 7's prediction instead of running the DiT | 8 | 8 |
| `VLLM_OMNI_K6_HYBRID_GEMM=1` | large-M DiT linears on the hybrid FP16-accumulate GEMM | yes | |
| `VLLM_OMNI_K6_HYBRID_GEMM_EXCLUDE` | regex of layers kept BF16; default `modulation\|time_embeddings` (from the FP16 range audit) | default | |
| `VLLM_OMNI_K6_HYBRID_GEMM_MIN_ROWS` | calls with fewer rows stay on cuBLAS; default 2048, 0 wraps every linear | default | |
| `VLLM_OMNI_K6_FP8_GEMM_AFTER_STEP=k` | steps after k run low-precision GEMMs (FP8 per tensor) | | 2 |
| `VLLM_OMNI_K6_STEP_GEMM_FORMAT=nvfp4` | those steps run NVFP4 instead of FP8 | | yes |
| `VLLM_OMNI_K6_HYBRID_STAGE_FP16=1` | stage the hybrid linears' weights as FP16 on the offload copy stream (measured no gain; off) | | |
| `VLLM_OMNI_K6_FP16_AUDIT=<file>` | write a per-layer FP16 range audit of every DiT linear after each request (diagnostic) | | |

`arms/final-B-cache8.json` (round-3 final stack), `arms/h1-hybrid.json` (round-5
head) and `arms/bf16-reference.json` are the exact configurations measured.

## Limitations

- **One GPU, batch 1, W1 geometry.** Every number is one request at a time at
  864x480 x 121 frames. Other sizes and concurrent requests were not measured.
- **The user's gate is not met on its worst frame, and cannot be.** G1 asks for
  a set mean <= 0.15 and every frame <= 0.25. The final stack's mean is inside
  (0.1114); its worst frame (0.3492) is not. Recompiling the BF16 reference
  itself, compiled against eager with the same weights, already moves a worst
  frame by **0.416**, so this pipeline's own noise is above that bar
  ([measurements.md](measurements.md)).
- **The hybrid GEMM changes numerics.** FP16 operands and FP16 accumulation
  inside each 32-term K block move the output by about as much as recompiling
  does. It needs its own reference, as every result here has one.
- **The fast mode is lossy.** It passes the CLIP check, but on b6 the INT8
  weights turn the shot into a different view with an illegible sign, and on a3
  the runner's shorts change colour (contact sheets on the card). It is never a
  G1 configuration.
- **sm_120 only.** The hybrid kernel, SageAttention 2/3 builds and NVFP4 path
  target the RTX 5090. sm_120 has no INT8 GEMM, which is why INT8 is a storage
  format here, not a compute one.
- **W2 is not gated.** Its timing is measured; its quality was screened on one
  prompt.
- **Set B** (a second, harder nine-prompt set) was scored for the head only
  (0.1307 / 0.3576); the earlier rows have set-A scores alone.
- **Host needs.** 60 GB of RAM and the checkpoint on NVMe: the final stack
  streams 56 GiB a step through the page cache.

## Where to look next

- [measurements.md](measurements.md): every number, newest first, with its
  protocol and what it superseded.
- [PLAN.md](PLAN.md): the goal, the gates and the rules of the study.
- [compute/README.md](compute/README.md): the kernel-level work (attention
  races, GEMM census, the hybrid kernel).
- [compute/ncu-roofline.md](compute/ncu-roofline.md),
  [compute/ncu-roofline-outside-block.md](compute/ncu-roofline-outside-block.md),
  [compute/ncu-roofline-h5.md](compute/ncu-roofline-h5.md): where the next 20 s
  are.
  - Exact attention and the GEMMs sit at the tensor-core ceiling.
  - The VAE decode does ~2.7x the work it needs.
  - GroupNorm runs on 32 CTAs.
