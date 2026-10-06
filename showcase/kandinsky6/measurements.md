# Kandinsky 6 measurements

Measurements for the single-RTX-5090 showcase ([PLAN.md](PLAN.md)). One
section per decided question, newest first. Track C (compute, build-server-3)
writes the kernel sections; Track M (build-server-2) writes the end-to-end and
quality sections.

Every number here was taken with all five of build-server-3's GPU lock files
held and `nvidia-smi` showing no foreign compute process. The GPU is shared,
and a run that had a co-tenant is discarded rather than reported.

## Where a Pro block's time goes at W1, and what two config values do to it

> Every number in this section was taken after two bugs in the measuring
> harness were found and fixed (see "Two bugs in the harness" below). Earlier
> revisions of this file carried pre-fix numbers; those are superseded.

One `Kandinsky6FusedTransformerDecoderBlock` at W1's token counts, and x60 for
the 60 visual blocks of one forward. All four arms timed in ABBA order in
**one process**, n=12 each, so every ratio is same-session.

| arm | block | x60 blocks | vs shipped | spread |
|---|---:|---:|---:|---:|
| `shipped` — platform attention, compiled | 340.86 ms | 20.45 s | — | 0.95% |
| `shipped-eager` — `--enforce-eager` | 393.08 ms | 23.58 s | +15.32% | 0.57% |
| `tuned` — `arms/tuned.json`, compiled | 230.74 ms | 13.84 s | **−32.31%** | 0.73% |
| `tuned` + `mode="max-autotune"` | 141.61 ms | 8.50 s | **−58.45%** | 0.59% |

`shipped` is the right baseline: vLLM-Omni compiles the DiT's repeated blocks
by default — `enforce_eager=False`,
`diffusion_compile_granularity="regional"`, and
`Kandinsky6FusedTransformerDecoderBlock` is in the model's
`_repeated_blocks` — so an eager measurement describes nothing anyone serves.

Both changes are configuration, not code:

```bash
--diffusion-attention-config "$(cat showcase/kandinsky6/compute/arms/tuned.json)"
```

and `--diffusion-compile-mode max-autotune`, which this branch adds:
`regionally_compile` forwarded only `dynamic`, so there was previously no way
to ask for CUDA graphs or GEMM autotuning from a config. Unset leaves torch's
own default, so the flag changes nothing until it is set.

### The compiled arms compute the same answer, which is why they are adoptable

Each compiled arm against the **same module** run eager, on the same inputs:

| arm | vs eager | rel L2 (video) | rel L2 (audio) |
|---|---:|---:|---:|
| `tuned`, `mode="default"` | −18.09% | 5.200e-03 | 4.436e-03 |
| `tuned`, `mode="reduce-overhead"` | −42.31% | 5.200e-03 | 4.438e-03 |
| `tuned`, `mode="max-autotune"` | −49.65% | 5.161e-03 | 4.444e-03 |
| `shipped`, `mode="default"` | −13.17% | 4.590e-03 | 4.375e-03 |

All three tuned arms land at the same error, so `max-autotune`'s Triton GEMM
templates add nothing beyond the reordering `mode="default"` already does.
5.2e-03 against a control of RMS 1.26 is bf16 reduction reordering.

**This check is the only reason the CUDA-graph arms are reportable at all.**
Their profiles put attention at 3.1% of the block, which cannot be true —
41.32 TFLOP does not run in 4.4 ms at any rate this GPU has. The profiler
mis-attributes work captured inside a CUDA graph, so the *category split* of a
graph-captured arm must not be quoted. Its wall time and its output are both
sound.

### The split, and where the next lever is

Profiled separately (the profiler changes the timing, so a profiled run is not
a wall), both at `mode="default"`:

| category | shipped | tuned |
|---|---:|---:|
| block | 339.35 ms | 228.76 ms |
| attention | 185.89 ms (55.0%) | 71.97 ms (31.6%) |
| GEMM | 145.55 ms (43.1%) | 146.61 ms (64.4%) |
| elementwise | 3.12 ms (0.9%) | 3.39 ms (1.5%) |
| norm | 3.41 ms (1.0%) | 5.08 ms (2.2%) |
| copy | ~0 | 0.67 ms (0.3%) |
| unclassified | 0.01 ms | 0.01 ms |
| launch gap | 6.05 ms | 6.34 ms |

**PLAN.md's "attention is about two thirds of the DiT's work" is a modest
overstatement, not a refutation.** It is 55.0% of the shipped block. An
earlier revision of this file reported 47.5% and called the bound refuted;
that came from an *eager* profile, where the unfused fp32 AdaLN/RoPE/gate
chain contributed 17.8% and inflated the denominator. Compiled — which is what
runs — that chain is 1.9%. The lesson is not about the bound: it is that a
profile taken in a mode the system does not run in attributes time to the
wrong place, by enough to change which lever you pick next.

After the tuned arm the block is **64.4% GEMM**, on
`cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_*` kernels — the Ampere
`s16816` HMMA generation. `max-autotune` takes that 145.55 ms to 125.15
(−14%) with Triton templates, which is most of its win. Whether a block-scaled
FP8 or NVFP4 route beats bf16 here needs Track M's format.

### Two bugs in the harness

Both were in `block_profile.py`'s synthetic block, both invalidated earlier
block numbers, and **neither touched the attention race** — `attn_race.py`
always built its activations with the port's own RoPE module and normalized
q/k itself.

1. **The RoPE table was drawn from `randn`.** A RoPE table's 2x2 blocks are
   `[[cos, −sin], [sin, cos]]`, so `apply_rotary` is a rotation and preserves
   the per-head norm `query_norm`/`key_norm` just set. Random entries make it
   an arbitrary linear map that stretches some rows far more than others.
2. **The weights were never random.** vLLM's `ColumnParallelLinear` and
   `RowParallelLinear` allocate with `torch.empty` and expect a checkpoint
   loader; unlike `torch.nn.Linear` they run no initializer, and the
   allocation read back as **zeros**. Every projection in the block returned
   zero.

Dense GEMM and attention kernels take the same time on zeros, which is why
both bugs were invisible in the timings. They are not invisible to a
quantizing kernel: SageAttention's per-block scale is the block's maximum
absolute value, so a zero query gives a zero scale and the kernel returns NaN.

They surfaced because `--check-numerics` was added before trusting a 49%
speedup, and **the control's own output came back NaN**. The first version of
that check was itself wrong: it compared parameter *names* to decide whether
two arms share weights, and `torch.compile` returns an `OptimizedModule` whose
names are all prefixed `_orig_mod.`, so it answered "not comparable" for
exactly the arms it existed to check.

Each fix carries a test: every RoPE 2x2 block has determinant 1 and unit rows;
the initializer returns how many parameters it filled, so a layer-type change
cannot make it silently stop applying; a non-finite control is reported as a
broken control rather than as NaN deltas.

## Attention backend on sm_120, at W1's visual self-attention

`kandinsky6.visual_self` at W1 is 50,220 queries against 50,220 keys, 32 heads
of 128, bf16: **41.32 TFLOP a call**, 98.8% of a fused block's attention FLOPs
and 2.48 PFLOP over the 60 blocks of one forward. It is the largest single
compute decision on this GPU, and on sm_120 it was open — FlashAttention-3 is
Hopper-only and the platform default is `CUDNN_ATTN`.

Median of 12 samples an arm, in ABBA order (2 rounds x 2 visits x 3 repeats)
in one session, after 2 warm-ups per visit. `rel L2` is against softmax
attention computed in fp32 on the same q/k/v with TF32 off.

| arm | median | min–max | spread | TFLOP/s | vs cuDNN | rel L2 vs fp32 | cosine |
|---|---:|---:|---:|---:|---:|---:|---:|
| `SAGE_ATTN_3` (FP4) | 55.53 ms | 54.91–55.81 | 1.62% | 744.1 | **3.20x** | 0.188251 | 0.98229 |
| `SAGE_ATTN` (2++) | 70.53 ms | 70.22–70.84 | 0.88% | 585.9 | **2.52x** | 0.038957 | 0.99924 |
| `CUDNN_ATTN` (default) | 177.66 ms | 176.74–178.77 | 1.14% | 232.6 | 1.00x | 0.002322 | 0.9999974 |
| `TORCH_SDPA` | 194.12 ms | 193.57–194.65 | 0.55% | 212.9 | 0.91x | 0.002322 | 0.9999974 |

`FLASHINFER_ATTN` cannot run here at all (below), so the bf16 control is cuDNN.

A second session replicated the table within 0.6% (`SAGE_ATTN` 70.93 ms,
`CUDNN_ATTN` 178.58 ms, `SAGE_ATTN_3` 55.52 ms), which is the better evidence
that these are kernel properties and not a session artefact. `FLASH_ATTN` was
still unavailable in it: with `flash-attn-4==4.0.0b18` installed the arch gate
passes but the backend raises `AttributeError: 'NoneType' object has no
attribute '_trait'` from inside the CuTe DSL. Unresolved, and it is a bf16
kernel, so it cannot beat an FP8 one on rate — the question it would answer is
only whether cuDNN leaves anything on the table as the control.

Arm spreads are 0.55–1.62%, so the 2.52x is not a noise artefact.

### What makes Sage2 fast here

On sm_120 the `sageattn` dispatcher picks `sageattn_qk_int8_pv_fp8_cuda` with
`pv_accum_dtype="fp32+fp16"`: INT8 for QK, FP8 for PV. There is no separate
sm_120 extension — the kernels come from `_qattn_sm89`, compiled with
`-gencode arch=compute_120a,code=sm_120a`, because consumer Blackwell runs the
Ada INT8/FP8 mma path. 586 TFLOP/s is above the 494 TF rate of non-block-scaled
FP8 `mma.sync` on this architecture, consistent with the QK half running at the
INT8 rate rather than the FP8 one.

At 50,220 tokens the call is mma-bound, so that rate is realized almost in
full. This is also where the published SageAttention2 figure ("about 560 TOPS
on an RTX 5090") is confirmed at a video DiT's sequence length rather than an
image one.

### Why the pick is Sage2 and not the faster Sage3

Sage3's FP4 kernel is another 1.27x, for **4.8x the error** (rel L2 0.188
against fp32, vs Sage2's 0.039). The showcase gate is mean LPIPS ≤ 0.05 and max
≤ 0.10 against the BF16 checkpoint; spending 4.8x the numerical error for 1.27x
is not a trade to make before the gate can score it. Sage3 stays an arm
(`compute/arms/sage3.json`) and is re-decided once Track M's reference set
exists.

> The accuracy column is the attention kernel's error, not the video's, and the
> activations were synthetic — Pro does not fit on this GPU, so there is no real
> Pro forward to capture from. They are not `randn`: the port RMS-normalizes q
> and k per head before RoPE and RoPE is a rotation, so every row a kernel sees
> has unit RMS, which `make_activations` reproduces with the port's own RoPE
> module. What it cannot reproduce is the correlation between neighbouring
> tokens in a real video latent, which makes the real softmax peakier — and a
> peakier softmax is kinder to a quantizing kernel. These errors are therefore
> an **upper bound**: safe for ranking arms, not a substitute for the gate.

### Adopting it is a config value, not a code change

`AttentionConfig` already resolves a backend per call site (exact `per_role`
match, then role category, then `default`, then the platform), and the port
already passes a distinct role string at each of Kandinsky 6's five attention
call sites. So:

```bash
vllm serve kandinskylab/Kandinsky-6.0-Pro-distill-5s-Diffusers --omni \
    --diffusion-attention-config "$(cat showcase/kandinsky6/compute/arms/tuned.json)" \
    --diffusion-compile-mode max-autotune
```

Only `kandinsky6.visual_self` is pinned, and that is forced, not tidy:
`SageAttentionImpl.forward_cuda` raises `"SAGE_ATTN does not support
attn_mask"`, both bundles set `text_token_padding: true`, and the fused block
hands `attn_mask` to the text and audio cross-attention while visual
self-attention is called with only `rotary_emb` and `sparse_params`. A global
`default: SAGE_ATTN` would not be slower on the cheap call sites — it would
raise on the first padded prompt.

## One backend for the whole model is wrong whatever it is set to

The same race over all five call sites. Reversing the shape inverts the
ranking: SageAttention pays its per-block quantization prologue per *query*
block, and at 218 queries there is no mma work to amortize it.

| call site | q | kv | heads | `TORCH_SDPA` | `SAGE_ATTN` | `CUDNN_ATTN` | `SAGE_ATTN_3` |
|---|---:|---:|---|---:|---:|---:|---:|
| `visual_self` | 50,220 | 50,220 | 32x128 | 194.12 ms | **70.53 ms** | 177.66 ms | 55.53 ms |
| `text_cross` | 50,220 | 256 | 32x128 | 1.176 ms | **0.946 ms** | 1.127 ms | 3.728 ms |
| `video_audio_cross` | 50,220 | 218 | 32x128 | 1.168 ms | **0.945 ms** | 1.131 ms | 3.736 ms |
| `audio_video_cross` | 218 | 50,220 | 16x128 | **0.593 ms** | 1.882 ms | 2.498 ms | 2.711 ms |
| `audio_self` | 218 | 218 | 16x128 | **0.0253 ms** | 0.0556 ms | 0.0344 ms | 0.1447 ms |

Two consequences. `audio_video_cross` is on the **worst** of the three arms as
shipped: cuDNN is the sm_120 platform default and is 4.2x `TORCH_SDPA` there,
about 2.5 ms a block or 150 ms a step over 60 blocks. And a kernel ranked on a
model's dominant call site must not be extrapolated to its cheap ones — at
0.09 TFLOP the audio cross-attention is prologue- and launch-bound, not
mma-bound. (`text_cross` and `video_audio_cross` carry masks in production, so
`SAGE_ATTN`'s small win there is not available.)

## `FLASHINFER_ATTN` does not run on consumer Blackwell

flashinfer 0.7.0.post1 raises from
`flashinfer/cute_dsl/attention/fmha/fmha.py:291`, where `_make_qk_tiled_mma`
calls `sm100_utils.make_trivial_tiled_mma`:

```
expects arch to be one of [sm_100a, sm_100f, sm_110a, sm_110f, sm_103a, sm_103f],
but got sm_120a
```

with its own suggestion that "tcgen05 MMA requires datacenter Blackwell
(sm_100 / sm_103)". Before raising it spends about 90 s per attempt on
exponential-backoff 404s fetching
`.../fmha/cute-dsl/x86_64/sm_120f/checksums.txt` — NVIDIA does not publish
cubins for this arch. On a shared GPU that comes out of the measurement
window, so `FLASHINFER_ATTN` belongs out of an sm_120 arm list rather than in
it and failing.

This is the **dense** path only. It says nothing about FlashInfer's
block-sparse `bsa_attn_sm120_blk64_fwd`, which is reported on sm_120 hardware
and remains the candidate for the sparse-attention line of the plan.

## Environment

| | |
|---|---|
| GPU | RTX 5090, 32 GB, 575 W, driver 610.43.02, CUDA UMD 13.3 |
| Host | build-server-3, 60 GB RAM, 32 cores; GPU shared with other studies |
| venv | `/data/jooman/k6/venv`, Python 3.12.3 |
| | `vllm==0.31.0`, `torch==2.13.0+cu132`, `vllm-omni` editable at `517f27b04` |
| | `diffusers==0.40.0`, `transformers==5.14.1` |
| | `sageattention==2.2.0` and `sageattn3==1.0.0`, both built from `thu-ml/SageAttention` source with CUDA 13.3 and `TORCH_CUDA_ARCH_LIST=12.0` |
| | `flashinfer==0.7.0.post1` (from vLLM), `flash-attn-4==4.0.0b18` |
| Checkpoint config | `kandinskylab/Kandinsky-6.0-Pro-distill-5s-Diffusers`, snapshot `7a1a4033` |
| Harness | `showcase/kandinsky6/compute/attn_race.py`, run under `run_when_free.py` |
| Raw records | `compute/results/*.json` (`k6c-b04`, `k6c-g02`, `k6c-b03`, `k6c-p05-*` are the post-fix runs); vault root `k6c-bs3` |
| Vault trials | `k6c-a01` (attention backend), `k6c-f01` (`torch.compile`), `k6c-g01` (compile modes). Every prediction frozen before its run. |

## Reproduce

From a clone of this branch, with the venv above:

```bash
cd showcase/kandinsky6/compute
# The attention backend race.
python run_when_free.py --need-free-gib 12 -- \
    python attn_race.py --all-roles --repeats 3 --rounds 2 --json race.json
# The step split, and eager vs compiled in one process.
python run_when_free.py --need-free-gib 16 -- \
    python block_profile.py --attention-config arms/tuned.json --json split.json
python run_when_free.py --need-free-gib 16 -- \
    python block_profile.py --attention-config arms/tuned.json \
        --compare-compile --compile default --json fusion.json
```

`session.sh` and `session2.sh` are the exact sequences the recorded runs used:
on a shared GPU a free window is often big enough for three runs but gets used
for one if each queues separately, so `run_when_free.py` takes the locks once
and a session script spends the window.

`attn_race.py --arms` takes the arm list; dropping `FLASHINFER_ATTN` on sm_120
saves about 90 s a role. `python block_profile.py --list-shapes` prints the
derived token counts the table's shapes come from.
