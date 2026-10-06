# Kandinsky 6 Pro on one RTX 5090: optimization plan

Branch `kandinsky6/showcase` of `fractalyze/vllm-omni`, based on upstream
`5f95115e7` (Kandinsky 6 port, vllm-project/vllm-omni#8537). The earlier
showcase on this fork, `qwen3omni/showcase`, is the model for the layout:
deploy configs, a `measurements.md` from one session, a Hugging Face card.

## Goal

The fastest end-to-end text-to-video-and-audio (T2VA) request for
Kandinsky 6.0 Pro on **one RTX 5090 (32 GB, 575 W) in a host with 60 GB of
RAM**, served by vLLM-Omni from this branch, with output that passes a
quality gate against the same checkpoint's BF16 output.

Neither half fits as shipped. The Pro DiT has 29B parameters: 60 GB in BF16,
which exceeds both the GPU and the host RAM. Upstream's recipe needs an H100
and about 90 GB of free host RAM for `--enable-cpu-offload`. The distilled
checkpoint (`kandinskylab/Kandinsky-6.0-Pro-distill-5s-Diffusers`, 10 steps,
guidance 1.0, `PiflowScheduler`) is not served by the upstream port at all:
`scheduling_kandinsky6.py` has no PiFlow.

## Workloads

| id | checkpoint | geometry | steps / guidance | status |
|---|---|---|---|---|
| W1 (headline) | Pro-distill-5s | 864x480, 121 frames, 24 fps, audio on | 10 / 1.0 | anonymous download works |
| W2 | Pro-5s | 864x480, 125 frames, audio on | 50 / CFG 5.0 | Hub repo is gated: needs `HF_TOKEN` (not on the hosts yet) |
| W3 | Pro-distill-5s, TI2VA | W1 plus a reference image | 10 / 1.0 | after W1 |
| smoke | either | 512x320, 25 frames | 4-10 | correctness only, never a headline |

Scale of W1: the Hunyuan VAE compresses 4x in time and 8x in space and the
DiT patches 1x2x2, so one forward sees 31 x 30 x 54 = 50,220 visual tokens
through 60 blocks (d_model 4096, 32 heads of 128, FFN 16384), plus 4 text
blocks and an audio branch (d 2048). By shape, self-attention is about
2.5 PFLOP a forward and the block GEMMs about 1.4 PFLOP, so **attention is
roughly two thirds of the DiT's work**. That is a derived bound: profile
before acting on it.

## Metrics

- **Headline:** request wall time, from the POST to `/v1/videos` until the
  MP4 is downloadable, on a warm server. Report median, min and max over at
  least 3 timed requests after 1 warm-up.
- **Breakdown, same session:** text encoding, denoise loop (per step),
  video VAE decode, audio decode, muxing, and any weight movement.
- Peak GPU memory (allocator and `nvidia-smi`), peak host RSS, cold start.

## Quality gate

- **Reference:** the same checkpoint in BF16, same seeds and prompts. It
  cannot be resident on these hosts, so the reference runner streams each
  DiT block from the safetensors file on NVMe. It is slow and only has to
  run once per prompt and seed. Store reference latents and MP4s under
  `/data/jooman/k6/ref/`.
- **Video:** LPIPS per frame (mean and max over frames), PSNR, SSIM against
  the reference. **Audio:** log-mel L1 and SI-SDR against the reference.
- **Adoption gate (set by the user, 2026-10-06):** per prompt set, LPIPS
  mean <= 0.15 **and** max <= 0.25 against the same-checkpoint BF16
  reference, on **both** prompt sets. "Mean" is the mean over the set's
  prompts of each prompt's mean over frames; "max" is the single worst frame
  anywhere in the set. An arm that passes both sets is a headline candidate,
  however narrowly; the fastest such stack is the showcase's answer.
- **Tiers** follow the world-model vocabulary (exact, reorder, approx,
  lossy) and are still reported alongside the gate, because the two answer
  different questions: the gate says whether an arm may ship, the tier says
  what to call it. Approx is mean LPIPS <= 0.05 and max <= 0.10, as in the
  Qwen-Image studies; `reorder` additionally requires being inside the
  pipeline's own noise floor, which on this host is **max LPIPS 0.0030**
  (measured by running one arm twice at the same seed). An arm can be
  adoptable and `lossy` at once, and saying so is the point: a showcase that
  calls a lossy arm near-lossless is the failure this pair of numbers exists
  to prevent. A lossy change additionally needs a distributional check (CLIP
  score) and a look at the frames.
- **Measure the noise floor before quoting any gate number.** Run one arm
  twice at the same seed and score it against itself. On this pipeline that
  is LPIPS mean 0.0023 / max 0.0030 on the video, which makes a 0.118 result
  50x the floor rather than a number to argue about — and **SI-SDR -1.4 dB on
  the audio**, which means the audio branch is not reproducible and the
  gate's audio half cannot distinguish an arm from a rerun until it is.
- **Prompts:** two disjoint sets of at least 8 prompts each. Each must
  include a face or person, rendered text, fast motion, and speech or a
  sharp sound event. The vault shows why: an FP8 recipe passed one 8-prompt
  set at LPIPS 0.034 and failed another at 0.162, worst on text and faces
  (`c-qwen-image21-fp8-quality-is-prompt-dependent-2026-09`).

## Prior evidence

From the world-model vault (`~/fractalyze/optimization-world-model`) and
the recipe DB (`s3://fractalyze-hfrecipe-...`, summarized in
huggingface-crawler `docs/FINDINGS.md`). Treat every item as a prior to
test, not a result.

- Step reduction dominates diffusion speedups (LightX2V 4-step: 11-22x on
  Wan2.1); feature caching gives about 2-3x with a quality cliff past 3-5x;
  FP8 or SageAttention give 1.1-1.5x each; torch.compile gives 1.14x on
  Wan-14B. W1 already takes the step lever.
- Consumer 24-32 GB GPUs are memory-bound first. Group offload costs 3.7x
  on FLUX but only 1.09x on Wan-14B: a big video DiT has enough compute per
  layer to hide the weight stream. At 50k tokens this DiT is compute-bound,
  so **FP8 weights streamed layer by layer from pinned host memory** may
  match NVFP4 weights resident on the GPU, at better quality. Test it.
- On Qwen-Image 2.1, an error at an early denoise step grows about 20x by
  the final latent. Approximating early steps does not pay under a
  same-seed gate; with 10 distilled steps every step is early-ish, so step
  caching has a low prior on W1 and a reasonable one on W2.
- RTX 5090 FP8 `mma.sync` runs at half rate (494 TF) unless the block-scaled
  form is used (982 TF). cuBLASLt's `nvjet` kernels already use it; CUTLASS
  `OpClassTensorOp` FP8 does not
  (`rtx5090-sm120-fp8-mma-block-scaled-full-rate`).
- Moving the text encoder in and out per prompt costs PCIe time; encoding
  once per batch saved 23% latency on Qwen-Image
  (`memsched-m01-te-once-per-batch`).
- FlashAttention-3 is Hopper-only. Upstream's H100 run was 1.87x slower on
  SDPA than on FA3, so on sm_120 the attention backend is open: SDPA
  (cuDNN or flash), FlashAttention-2, SageAttention 2/2++/3 (FP8 and FP4
  attention on Blackwell), FlashInfer.
- The port already carries NABLA / sliding-tile attention masks
  (`nabla_block_mask`), off by default (`sparse_params: None`). Whether the
  checkpoint tolerates them is a quality question.

## Tracks

Two agents, one per host. Each owns one track and works on its own branches.

### Track M: memory, precision, enablement (build-server-2)

Owns the harness and the quality gate, which Track C consumes.

1. **M0 harness.** `showcase/kandinsky6/bench/`: server launcher, request
   driver, stage timers, co-tenant sampler, quality scorer, ledger rows.
2. **R0 reference runner.** BF16 with blocks streamed from NVMe; produce the
   reference set for both prompt sets on W1.
3. **PiFlow.** Port `PiflowScheduler` so the pipeline serves Pro-distill.
4. **Fit.** Make W1 run on one 5090 with 60 GB of host RAM. Candidates:
   NVFP4 (W4A4) resident, FP8 W8A8 resident with the text encoder off the
   GPU, FP8 streamed from pinned host memory, mixed precision that keeps
   sensitive layers (first and last blocks, modulation, text blocks, audio
   branch) wider. Offline-quantized checkpoints go to
   `/data/jooman/k6/ckpt/<name>/`; publish only after the user approves.
5. Text encoder (Qwen2.5-VL-7B) placement and precision; VAE decode memory
   and time (upstream logged 7 GiB allocation failures in decode).
6. W2 once a token is available.

### Track C: compute (build-server-3)

1. **Profile** one W1 forward, split by attention, GEMM, elementwise and
   norm, VAE, text encoder, launch gaps. Until Track M ships a fitting
   checkpoint, use what fits: Kandinsky 6 Lite (3B, same block structure)
   for the whole pipeline, and Pro-shaped single blocks with random or
   FP8 weights for kernel work.
2. **Attention backend on sm_120** at Pro shapes: SDPA backends,
   FlashAttention-2, SageAttention 2++ and 3, FlashInfer. Register the
   winner in vLLM-Omni's diffusion attention layer.
3. **Sparse attention:** NABLA / sliding-tile through a registered
   backend, gated on quality.
4. GEMM kernels for whatever format Track M picks (block-scaled FP8 or
   NVFP4 on sm_120).
5. Fusion of modulation, RoPE, norms and gates (torch.compile first);
   CUDA graphs per step; VAE decode kernels.
6. Step caching (MagCache) for W2.

## Rules for both tracks

- **World-model loop for every trial:** consult the vault, preregister the
  prediction (write-once), run, record the verdict. Study page
  `study-kandinsky6-5090`; trial prefixes `k6m-` and `k6c-`. The vault copy
  on each host is local: commit to branch `k6/<track>` there, never push.
  Use the `wm` tool as `uv run --quiet --project
  ~/fractalyze/optimization-world-model python "$(cat
  ~/.local/state/world-model/wm-path)" ...` (see
  `wm-tool-old-jsonschema-uv-run`).
- **GPU:** check `nvidia-smi` before every timed run and hold every lock
  file on the host (`flock` each one), including other studies' locks.
  Discard a timed run that had a foreign GPU process.
- **Measurement:** warm-up, at least 3 timed repeats, median with min and
  max, ABBA for A/B pairs, same session for any ratio, a delta inside the
  control's own spread is null. A profiled run is not a wall.
- **Disk:** weights and caches under `/data/jooman/` (`HF_HOME=/data/jooman/hf`),
  never on `/`.
- **Code:** feature branches `kandinsky6/<m|c>-<topic>` from
  `kandinsky6/showcase`; PRs into `kandinsky6/showcase` on
  `fractalyze/vllm-omni`, never upstream. Every behavior change gets a test;
  a speed path sits behind a switch until it passes the gate. Merge your
  own PR into the showcase branch once tests pass and the measurement is
  recorded. No force pushes.
- **Status:** keep `/data/jooman/k6/STATUS.md` on your host current: what
  runs now, the last result, the next step, blockers.
