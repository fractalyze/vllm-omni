# Final stack, set A on build-server-3 — the gate row

Delivered as a file because **bs3 cannot reach bs2**: the tailnet policy refuses
`ssh jooman@build-server-2`, and the brief grants bs1 -> bs2/bs3 and bs2 -> bs3
only. Git is the channel both hosts share, so the row lives here rather than in
bs2's `HANDOFF.md`. It is also in bs3's `/data/jooman/k6/HANDOFF.md`.

Measured 2026-10-07 11:53–12:21 KST, Track C.

## Stack

`kandinsky6/showcase` head `d79c0acb1` (includes **#40** stage-ahead and **#38**
bias unfused) with:

```bash
HF_HOME=/data/jooman/hf \
TORCHINDUCTOR_CACHE_DIR=/data/jooman/k6/inductor-cache \
VLLM_OMNI_K6_EXACT_ATTN_STEPS=1 \
VLLM_OMNI_K6_PIFLOW_CACHE_STEPS=8 \
VLLM_OMNI_K6_PIFLOW_CACHE_MODE=reuse \
serve/serve_pro_bf16_ref.sh \
  --diffusion-attention-config "$(cat compute/arms/sage2-mid.json)"
```

BF16 streamed via distributed layerwise offload with rank-local mmap. Reference:
`results/gate-pro/bf16-stream-control-full` — same host, same pinned Inductor
cache, the reference the shipped arm's primary verdict used. No foreign GPU
process in the run.

## The row

| | value |
|---|---|
| request, median of 9 | **158.7 s** (min 158.0, max 163.2, spread 3.3%) |
| **G1 (primary, the user's gate)** | set mean **0.1393** (limit 0.15) **inside**; set max **0.4675** (limit 0.25) **over** → **fails** |
| prompts over the 0.25 max | **2** — a3-sprint-start 0.4675, a8-skateboard-crash 0.4124 |
| **G2** (vs the 0.1455 / 0.4160 floor, 1.25×) | **passes** — 0.96× the floor's mean, 1.12× its max |
| **G3** (CLIP, ±2%) | **passes** — set CLIP 0.3092 against 0.3107, **−0.47%** |
| tier | lossy |

Per prompt, LPIPS mean / max:

| prompt | mean | max |
|---|---:|---:|
| a1-portrait-speech | 0.0464 | 0.0822 |
| a2-neon-signage | 0.1736 | 0.2279 |
| **a3-sprint-start** | 0.2776 | **0.4675** |
| a4-chalkboard | 0.0814 | 0.1046 |
| a5-waterfall-drone | 0.0638 | 0.0758 |
| a6-blacksmith | 0.0898 | 0.1672 |
| a7-cafe-menu | 0.1138 | 0.1500 |
| **a8-skateboard-crash** | 0.2696 | **0.4124** |
| a9-violinist | 0.1374 | 0.2381 |

Request times, in run order: 163.2, 159.4, 158.5, 158.7, 159.0, 158.0, 160.3,
158.1, 158.0 s.

## Against the shipped arm, same host, same reference

| arm | request | set mean | set max | over the max |
|---|---:|---:|---:|---:|
| `sage2-mid` + exact1 (shipped, pre-head) | 182.3 s | **0.1117** | **0.3446** | **1** (a3) |
| **final stack (+ head + cache step 8)** | **158.7 s** | 0.1393 | 0.4675 | **2** (a3, a8) |
| change | **−23.6 s, −12.9%** | +25% | +36% | 1 → 2 |

**Cross-host agreement is the tightest this study has had.** Track M's bs2 B
median was 158.9 s against bs3's 158.7 s — 0.1% apart.

## What this does and does not establish

The stack is **12.9% faster** and its set mean stays inside the user's limit, but
the set max rises 36% and a8 joins a3 above 0.25.

**The quality change is not attributed here.** Head also brings #40 (stage-ahead,
which should be numerically neutral) and #38 (bias unfused, a reassociation whose
effect should be ~1e-3), so by elimination the cached step 8 is the likely driver
— but **head-without-cache was not run on bs3**, so that attribution needs
Track M's A row, not this one.

**Suggested placement:** the fastest stack that passes G2 and G3, with its G1 max
failure and the 1 → 2 count stated beside it, the same way `sage2-edge0` is
listed. On the user's own criterion — the fastest method that passes *their* gate
— the shipped `sage2-mid` + exact1 still has the best G1, and 158.7 s buys a 36%
worse worst frame.

## Verifying a stack is actually the stack

This nearly cost the run, so it is worth writing down. **`/proc/<worker>/environ`
does not list `VLLM_OMNI_K6_*` even when the pipeline is honouring them.** Reading
it and concluding two flags had not propagated led to aborting a correct run. The
checks that do work, from the server log:

- `VLLM_OMNI_K6_EXACT_ATTN_STEPS` → grep for **`visual_self_exact`**. That role
  is only constructed when the flag is set.
- `VLLM_OMNI_K6_PIFLOW_CACHE_STEPS=8` → the denoise progress bar **jumps
  `7/10` → `9/10`**, step 8 absent.

Any table row resting on an environment flag should be confirmed from the log,
not from the launch command: a stack that silently drops a flag yields a
plausible number rather than an error.
