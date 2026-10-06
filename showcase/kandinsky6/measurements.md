# Kandinsky 6 measurements

Measurements for the single-RTX-5090 showcase ([PLAN.md](PLAN.md)). One
section per decided question, newest first. Track C (compute, build-server-3)
writes the kernel sections; Track M (build-server-2) writes the end-to-end and
quality sections.

Every number here was taken with all five of build-server-3's GPU lock files
held and `nvidia-smi` showing no foreign compute process. The GPU is shared,
and a run that had a co-tenant is discarded rather than reported.

## Where a Pro block's time goes at W1, and what two switches do to it

One `Kandinsky6FusedTransformerDecoderBlock` at W1's token counts, median of
5 timed forwards. Spreads are 0.13–0.30%, so every delta below is two orders
of magnitude outside the noise.

| stack | block | x60 blocks | vs shipped | peak allocated |
|---|---:|---:|---:|---:|
| as shipped (platform default, eager) | 375.67 ms | 22.54 s | — | 6.74 GiB |
| `arms/tuned.json`, eager | 259.69 ms | 15.58 s | **−30.9%** | 6.74 GiB |
| `arms/tuned.json` + `torch.compile` | 203.01 ms | 12.18 s | **−45.9%** | **5.20 GiB** |

The split, by kernel name, with the launch gap measured as wall minus device:

| category | shipped | tuned | tuned + compile |
|---|---:|---:|---:|
| attention | 178.53 ms (47.5%) | 62.92 ms (24.2%) | 63.71 ms (31.4%) |
| GEMM | 130.45 ms (34.7%) | 129.82 ms (49.8%) | 130.00 ms (64.1%) |
| elementwise | 44.35 ms (11.8%) | 44.63 ms (17.1%) | 3.38 ms (1.7%) |
| copy | 13.14 ms (3.5%) | 13.82 ms (5.3%) | 0.65 ms (0.3%) |
| norm | 9.31 ms (2.5%) | 9.29 ms (3.6%) | 5.01 ms (2.5%) |
| unclassified | 0.01 ms | 0.01 ms | 0.01 ms |
| launch gap | 10.83 ms | 8.18 ms | 6.24 ms |

### PLAN.md's "attention is about two thirds of the DiT's work" does not hold

It is 47.5% of the shipped block, not ~65%. The bound is not wrong about
attention *versus GEMM* — measured, attention is 178.53 / (178.53 + 130.45) =
57.8% of the attention-plus-GEMM time, close to the 64% the FLOP counts give.
It is wrong about attention versus the **step**, because it priced no
elementwise, norm, copy or launch time at all, and those are 17.8% of the
device time plus a 2.9% gap. A plan that budgets attention and GEMM is
budgeting 82% of the block.

### `torch.compile` is worth 21.8%, and the categories say why

Same block, same inputs, same attention arm, eager and compiled timed in ABBA
order **in one process** (2 rounds x 2 visits x 3 repeats, n=12 each): eager
263.79 ms (spread 0.51%), compiled 209.04 ms (spread 0.77%), **−20.75%** — a
delta 40x the control's spread. Profiled separately, the mechanism is not
inferred but visible: elementwise −92%, copy −95%, norm −46%, while **attention
and GEMM do not move** (+1.3% and +0.1%).

That is exactly the chain the port upcasts to fp32. `apply_scale_shift_norm`
runs three times a block, `apply_rotary` twice and `apply_gate_sum` three
times, each on the whole residual stream: at 50,220 x 4096 one upcast is a
823 MB fp32 tensor, and `apply_rotary`'s broadcast intermediate — `(N, H, D/2,
2, 2)` before its `sum(-1)` — is 1.65 GB. Eager mode fuses none of them, so
each was a full HBM round trip. Inductor graph-breaks at every attention call
(the backends enter extensions Dynamo cannot trace) and leaves the GEMMs to
cuBLAS, which is why those two categories are untouched.

`fullgraph=False` is required rather than convenient: a full graph either
fails at the attention extension or silently falls back to a traceable,
slower attention path and then measures the wrong thing.

**Peak allocation falls 1.54 GiB**, from 6.74 to 5.20. This study exists
because the model does not fit, so that is worth naming: the fp32
intermediates are transient, so it is per-block headroom rather than resident
capacity, but it is 1.54 GiB a residency or streaming plan no longer has to
leave free.

### What is left

After both switches the block is 64.1% GEMM. The GEMMs run on
`cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_*` kernels — the Ampere
`s16816` HMMA generation — with the largest at a `64x64_32x6` tile taking
73.2 ms of the 130. The next compute lever is therefore the GEMM: whatever
format Track M picks, and possibly the tile selection. Attention is down to
31.4% and the elementwise chain to 4.5%.

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
    --diffusion-attention-config "$(cat showcase/kandinsky6/compute/arms/sage2.json)"
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
| Raw records | `compute/results/*.json`; vault root `k6c-bs3`, snapshot `20261006-1940` |
| Vault trials | `k6c-a01` (attention backend, `outcome_vs_vault: vault-right`), `k6c-f01` (`torch.compile`, predicted -8%, measured -20.75%). Both predictions frozen before their runs. |

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
