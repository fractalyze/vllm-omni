# Kandinsky 6 measurements

Measurements for the single-RTX-5090 showcase ([PLAN.md](PLAN.md)). One
section per decided question, newest first. Track C (compute, build-server-3)
writes the kernel sections; Track M (build-server-2) writes the end-to-end and
quality sections.

Every number here was taken with all five of build-server-3's GPU lock files
held and `nvidia-smi` showing no foreign compute process. The GPU is shared,
and a run that had a co-tenant is discarded rather than reported.

## The block schedule: speed is linear in the band, quality is not resolvable by a screen

`AttentionSpec.layers` turns "which attention kernel" into "which blocks get the
fast kernel", so the band of exact blocks at the two ends of the stack is a dial.
Three settings, all on the exact BF16 streamed arm with Sage2 on `visual_self`:

| arm | exact blocks | request (steady) |
|---|---:|---:|
| Sage2 everywhere (`tuned.json`) | 0 | 164.8 s |
| `sage2-wide` (3:57) | 6 | 170.6 s |
| `sage2-mid` (6:54) | 12 | 179.1 s |
| platform default everywhere | 60 | 231.8 s |

**Speed is close to linear in the band**, at 0.97 s a request per block over the
first six and 1.42 s over the next six. The kernel race predicts 1.07 s a block:
60 blocks x (177.66 - 70.53) ms of attention over 10 steps is 64 s across the
whole stack. The two intervals bracket that rather than matching it, which is
what three points measured once each can support -- the useful form of the
result is that the band is a latency dial of roughly 1 s a block, not that it is
exactly linear.

Each number above is the **second** request of its run. The first carries
compilation -- `sage2-wide`'s was 185.9 s against its steady 170.6 s -- so an
arm screened without a warm-up and compared against one screened with a warm-up
reads 15 s slower than it is.

### Quality, and why these screens cannot order the bands

Two-prompt screens against the eager reference, with the full set for the one
arm that has one:

| arm | a1 | a2 | screen mean | full set A |
|---|---:|---:|---:|---:|
| Sage2 everywhere | 0.0676 | 0.2462 | 0.1569 (1.08x floor) | **0.1858 (1.28x)** |
| `sage2-wide` | 0.0577 | **0.2828** | 0.1703 (1.17x) | — |
| `sage2-mid` | 0.0579 | 0.2291 | 0.1435 (0.99x) | — |

Two things are wrong with reading a gate off that table, and both are worth
keeping.

**The screen underestimates the set.** For the one arm where both numbers exist,
two prompts gave 0.1569 and the nine-prompt set gave 0.1858 -- 18% higher, and
across the bar. a1 is the easiest prompt for every arm measured here and a2 the
one that separates them, which makes them a good pair for *choosing what to run*
and a bad pair for deciding anything.

**And the ordering inverts.** `sage2-wide` approximates six blocks fewer than
Sage2-everywhere and scored *worse* on a2, 0.2828 against 0.2462. That cannot be
a property of the band. It is what a difference of 0.02-0.05 looks like when the
pipeline's own compile floor has a set mean of 0.1455: the bands are separated
by less than the noise they are measured through, so a single sample per prompt
cannot order them.

So the band was chosen on the only signal that survives -- `sage2-mid` is the
one band better than Sage2-everywhere on *both* screened prompts -- and the
decision is made on its full nine-prompt set, not on the screen.

## Every arm measured tonight, and which reference each number is against

The reference column is not bookkeeping. The same outputs score differently
against a compiled reference, an eager one, and a same-host one, by more than
most of the arms differ from each other -- so a number without its reference is
not a measurement. All on Kandinsky 6 Pro-distill at W1 on one RTX 5090.

**Floors** (each set's compiled reference against its own eager one, same host):
set A **0.1455 / 0.4160**, set B **0.1615 / 0.3476**. G2 limits at 1.25x are
therefore **0.1819 / 0.5200** for set A and **0.2019 / 0.4345** for set B.

| arm | weights | request | set A mean / max | set B mean / max | ref | G1 | G2 | G3 |
|---|---|---:|---:|---:|---|---|---|---|
| platform default attention | BF16 streamed | 231.8 s | — it *is* the reference — | — | — | — | — | — |
| Sage2 all 60 blocks | BF16 streamed | **165.9 s** | 0.1858 / 0.4738 | — | eager | fail | **1.28x fail** | +0.39% pass |
| " (same outputs) | " | " | 0.1682 / 0.3693 | — | compiled | fail | — | +0.35% pass |
| Sage2 blocks 3-56 (6 exact) | BF16 streamed | 170.6 s | a1 0.0577, a2 0.2828 (screen) | — | eager | — | — | — |
| **Sage2 blocks 6-53 (12 exact)** | BF16 streamed | **177.7 s** | **0.1543 / 0.4591** | 0.1976 / 0.4989 | eager | fail | A **1.06x pass**, B 1.22x/**1.44x fail** | +0.00% pass |
| Sage2 blocks 12-47 (24 exact) | BF16 streamed | ~190 s | — | b3 0.3349, b6 0.2767 (screen) | eager | — | — | — |
| Sage2 "accurate" knobs | BF16 streamed | 192.8 s | a1 0.0563, a2 0.1867 (screen) | — | eager | — | — | — |
| **blocks 6-53 + exact step 1** | BF16 streamed | **182.6 s** | **0.1296 / 0.3560** | *running* | eager | mean inside, max over | A **0.89x / 0.86x PASS** | **-0.07% pass** |
| FP8 + Sage2 | FP8-min pinned | ~110 s | 0.2791 / 0.5649 | — | compiled | fail | ~1.9x fail | — |
| FP8 + platform attention | FP8-min pinned | 173.6 s | 0.262 / 0.551 | — | compiled | fail | ~1.8x fail | — |

Track M's arms on the same gates, for the Pareto frontier rather than for
attribution: **INT8 storage + Sage2 + exact step 1 at 167.5 s** passes G2 and G3
on set A (0.1735 / 0.4426 against the eager reference), and an **FFN FP8 band at
148.9 s** fails G2 on set A (0.218 / 0.461), so that band is out.

### The best arm: one exact sampler step on top of the block schedule

`sage2-mid` plus `VLLM_OMNI_K6_EXACT_ATTN_STEPS=1` -- the 12-block band with the
**first sampler step exact** -- is the strongest arm either track measured:

| prompt | categories | LPIPS mean | max |
|---|---|---:|---:|
| a5-waterfall-drone | motion | 0.0319 | 0.0351 |
| a1-portrait-speech | face, speech | 0.0345 | 0.0661 |
| a4-chalkboard | text, face, speech | 0.0997 | 0.1096 |
| a7-cafe-menu | text, face | 0.1026 | 0.1113 |
| a9-violinist | face, motion | 0.1207 | 0.1909 |
| a6-blacksmith | sharp-sound, face, motion | 0.1525 | 0.2471 |
| a2-neon-signage | text, motion | 0.1670 | 0.1909 |
| a8-skateboard-crash | motion, sharp-sound | 0.2397 | **0.3120** |
| a3-sprint-start | motion, face, sharp-sound | 0.2179 | **0.3560** |
| **set** | | **0.1296** | **0.3560** |

**It sits below the pipeline's own floor on both halves** -- 0.89x the floor's
mean and 0.86x its max. The difference between this arm and the BF16 reference
is *smaller than the difference between compiling that reference and running it
eager*. It is also the first arm here inside G1's **mean** (0.1296 against
0.15), and ties the reference on prompt agreement (-0.07%).

**One exact sampler step is worth more than twelve exact blocks.** Against the
same band without it (0.1543 mean, 177.7 s), the step cut the set mean **16% for
4.9 s a request**; against doubling the band instead (`sage2-narrow`, ~190 s) it
is both better and 6 s cheaper. That is the vault's Qwen-Image result -- an error
injected at an early step grows about 20x by the final latent -- reproduced on a
different model and a different approximation, and it says the trajectory
position matters more than the stack position for attention error.

### Reading the frontier

Two arms are faster than the 173.6 s FP8 baseline *and* pass a gate: Track M's
INT8 arm at 167.5 s, and Sage2-everywhere at 165.9 s which fails G2 by 2%. Of
the arms measured here, the only one that passes G2 on set A is the 12-block
band at 177.7 s -- 2.4% slower than the baseline, which fails every gate.

**Nothing on this list is both fast and gate-passing by a wide margin.** The
frontier between 165 s and 190 s is where every candidate sits, and it is set by
one thing: how much exact attention the arm keeps, in blocks or in steps. The
174-184 s band is where the measured arms cross from failing to passing, and the
arm now running is the cheapest crossing found.

## The verdict: nothing here passes on both prompt sets

Each set's floor is its own compiled reference against its own eager one, both
on the same host, which is the only pairing that means anything after the
autotune result below.

| set | floor mean / max | `sage2-mid` mean / max | ratio | G2 (1.25x) |
|---|---:|---:|---:|---|
| A | 0.1455 / 0.4160 | 0.1543 / 0.4591 | 1.06x / 1.10x | **passes** |
| B | 0.1615 / 0.3476 | 0.1976 / 0.4989 | 1.22x / **1.44x** | **fails** |

The **mean** passes on both sets. The **max** fails on set B, and on one prompt:
b6-train-platform at 0.4989 against a 0.4345 limit, with b3-storefront-sign
behind it. Both are text-plus-motion.

The shape of the failure is worth stating because it is counter-intuitive. Set
B is the harder set -- its floor mean is higher, 0.1615 against 0.1455. But its
floor **max** is *lower*, 0.3476 against 0.4160. A floor's max is the single
worst frame the pipeline moves by on its own, and set A happens to contain a
clip where recompilation moves one frame a long way. So set B's max limit is
**tighter** than set A's, exactly where this arm is worst. Expecting the harder
set to be more forgiving is a reasonable guess and it is wrong: the mean and the
max are set by different clips and move independently.

**So the user's question -- the fastest single-RTX-5090 configuration that
produces W1 and passes the gate -- has no affirmative answer from this track
tonight.** `sage2-mid` at 177.7 s passes the floor-relative gate at 1.06x and the
distributional gate at +0.00% on set A, and fails the floor-relative gate's max
on set B. Every faster arm fails by more; the only arm that passes everything is
the reference configuration at 231.8 s, which is not an optimization.

## Set B is 28% harder than set A, which is why there are two sets

The same arm, the same eight-of-nine prompts it could be scored on (Track M's
eager set B reference is missing `b9-piano-tuner`), median 178.3 s a request:

| prompt | categories | LPIPS mean | max |
|---|---|---:|---:|
| b8-surf-barrel | motion | 0.0536 | 0.0671 |
| b5-glassblower | motion, face | 0.0682 | 0.0714 |
| b1-newsreader | face, text, speech | 0.0972 | 0.1377 |
| b2-market-haggle | face, speech, motion | 0.1410 | 0.1597 |
| b7-child-birthday | face, sharp-sound | 0.1996 | **0.2699** |
| b4-tennis-serve | motion, sharp-sound, face | 0.2517 | **0.2736** |
| b3-storefront-sign | text, motion | 0.3307 | **0.3646** |
| b6-train-platform | text, motion, sharp-sound | **0.4387** | **0.4989** |
| **8-prompt set** | | **0.1976** | **0.4989** |

Set A was 0.1543 for the same arm. The reason the sets disagree is in the
categories: **B has two text-plus-motion prompts and they are its two worst**,
where A has one. PLAN.md requires two disjoint sets on exactly this evidence --
the vault's `c-qwen-image21-fp8-quality-is-prompt-dependent-2026-09`, where an
FP8 recipe passed one 8-prompt set at LPIPS 0.034 and failed another at 0.162 --
and this is that case reproduced on a different model and a different
approximation.

**Set B has no floor-relative verdict here.** A floor is the compiled reference
against the eager one *for that set*, and set B's compiled reference is still
generating. Substituting set A's floor would be the easy wrong thing: set B's
prompts are not set A's, and the difficulty ordering above is exactly what a
floor would also pick up. The scorer reports G2 as **undecided** when no floor
is supplied rather than falling back, and that is what it reports for set B.

`b9-piano-tuner` is generated here but unscored, and named rather than dropped:
a set mean over eight of nine prompts is a different gate, and this study has
already had a partial set read the opposite of its full one.

## The arm: SageAttention2 through blocks 6-53 of the stack

Exact BF16 weights streamed from the mmapped checkpoint, SageAttention2 on
`kandinsky6.visual_self` restricted to visual blocks 6-53, the platform default
on blocks 0-5 and 54-59, SDPA on the two audio roles, compile at the platform
default. **Median 177.7 s** a W1 request over nine timed requests (min 176.7,
max 180.2, spread 1.9%; cold start 24.3 s).

Set A against Track M's eager BF16 reference, with the measured floor
(set mean 0.1455, set max 0.4160):

| gate | number | limit | verdict |
|---|---:|---:|---|
| G1, the user's | mean **0.1543**, max 0.4591 | 0.15 / 0.25 | fails, mean over by 2.9% |
| G2, floor-relative | **1.06x** the floor's mean, 1.10x its max | 1.25x | **passes** |
| G3, distributional | set CLIP 0.3098 against 0.3098, **+0.00%** | +-2% | **passes** |

| prompt | categories | LPIPS mean | max |
|---|---|---:|---:|
| a5-waterfall-drone | motion | 0.0538 | 0.0590 |
| a1-portrait-speech | face, speech | 0.0579 | 0.1068 |
| a9-violinist | face, motion | 0.0837 | 0.1996 |
| a4-chalkboard | text, face, speech | 0.1219 | 0.1612 |
| a6-blacksmith | sharp-sound, face, motion | 0.1350 | 0.1849 |
| a7-cafe-menu | text, face | 0.1361 | 0.1530 |
| a2-neon-signage | text, motion | 0.2291 | **0.2601** |
| a8-skateboard-crash | motion, sharp-sound | 0.2710 | **0.3454** |
| a3-sprint-start | motion, face, sharp-sound | 0.2999 | **0.4591** |
| **set** | | **0.1543** | **0.4591** |

By category: sharp-sound 0.235, motion 0.179, text 0.162, face 0.139, speech
0.090 -- the same ordering every arm here produces, and the opposite of what an
attention-only screen on Kandinsky 6 **Lite** suggested.

Against Sage2 on all 60 blocks (0.1858 mean, G2 1.28x, 165.9 s) **the schedule
removes 17% of the error for 7% of the speed**, which is the difference between
failing the working gate and passing it with room.

### What a 0.4591 worst frame actually is

`showcase-samples/compare-a3-sprint-start-sage2-mid-setA.jpg` puts eight frames
of the worst prompt side by side, reference above arm. Same sprinter, same
track, same camera move, same lighting, same background crowd -- **the arm
frames the shot slightly tighter**. A global framing shift moves every pixel, so
a per-frame perceptual metric scores it near its worst while a viewer would call
both takes correct; the arm's CLIP agreement on that clip is -0.43%.

a2 and a8 are a different matter: those lose rendered-signage glyphs and
fast-motion detail that a viewer would notice. So this arm's error is **part
sampling divergence and part real detail loss**, and a claim resting on a
max-over-frames number has to say which it is made of. That is what G3 and the
contact sheets are in this file for.

## Where the reference was generated matters as much as how

Two of this host's server processes, started 40 minutes apart with compile on
and no determinism flags, produced **byte-identical MP4s** on the same prompts
and seeds. Compiled execution is therefore not run-to-run random. Inductor
benchmarks candidate kernels, picks by measured latency, and **caches the choice
on disk** (`/tmp/torchinductor_$USER`, 1.2 GB here); once that cache is warm the
choice is fixed and so is the output.

So this pipeline has three different floors, with three different causes:

| pairing | LPIPS mean / max | cause |
|---|---:|---|
| same host, warm cache, two processes | **0.0000 / 0.0000** | none; bit-identical |
| different hosts, both compiled | 0.0272 / 0.0636 (a1) | each host's autotune choices |
| same host, compiled vs eager | 0.1455 / 0.4160 | two fixed, different kernel sets |

### What that does to every number above

An arm scored against a reference from another machine is charged for that
machine's autotune choices as well as for its own approximation. Scoring four
arms against a reference generated **here**, on a1 and a2, the two prompts for
which both sides exist locally:

| arm | a1 mean / max | a2 mean / max | 2-prompt mean |
|---|---:|---:|---:|
| Sage2 everywhere | 0.0422 / 0.0486 | 0.1999 / 0.2134 | 0.1210 |
| **`sage2-mid`** (blocks 6-53) | **0.0194 / 0.0369** | **0.1657 / 0.1827** | **0.0925** |
| `sage2-wide` (blocks 3-56) | 0.0271 / 0.0354 | 0.2157 / 0.2283 | 0.1214 |
| `sage2-accurate` | 0.0582 / 0.1090 | 0.2099 / 0.2356 | 0.1340 |

The same Sage2 arm's a1 reads **0.0676** against the remote eager reference,
**0.0422** against the remote compiled one, and the scheduled arm reads
**0.0194** against a local control. The arm did not change; the reference did.

And the band ordering, which [the band section](#the-block-schedule-speed-is-linear-in-the-band-quality-is-not-resolvable-by-a-screen)
reports as unresolvable, resolves cleanly once the cross-host term is gone:
twelve exact blocks are worth 24% of the error, six are worth nothing, and the
accuracy knobs are the worst of the four. **That earlier conclusion was about
the measurement, not about the arms**, and it is left in place above rather than
rewritten, because the sequence is the point: a difference that sits under the
noise of one comparison can be plain in a better-conditioned one.

### What to do instead

Pin `TORCHINDUCTOR_CACHE_DIR` to a shared path and prime it once, for the
reference and every arm. It costs nothing, keeps compiled speed, and removes a
term that was larger than the effects being measured -- without
`TORCHINDUCTOR_DETERMINISTIC` and without falling back to eager.

## The pipeline's own numerical floor is larger than the gate

Track M measured it on set A: the same BF16 checkpoint through the same weight
path at the same seeds, **compiled against eager**.

| | set mean | set max |
|---|---:|---:|
| floor (BF16 compiled vs BF16 eager) | **0.1455** | **0.4160** |
| the user's gate (G1) | 0.15 | 0.25 |
| G2 limits (1.25x the floor) | 0.1819 | 0.5200 |

**Turning `torch.compile` on and off moves a single frame by LPIPS 0.416.** The
mechanism is Inductor's timing-based kernel selection, which Track M confirmed
by checking that two *eager* processes are byte-identical while two compiled ones
are not.

That single row reframes the whole exercise. An absolute bar of max 0.25 is
below the pipeline's own compile setting, so **the reference cannot be shown to
pass the user's literal gate against itself** -- not because the model is
unstable in any way a viewer would notice, but because LPIPS on a 121-frame clip
counts the worst frame anywhere and a differently-scheduled kernel moves it. A
gate that only a bit-exact change can pass is measuring the harness.

Hence the three bars reported side by side in this file. G1 is the user's and
stays. G2 is floor-relative: within 1.25x the floor on both halves. G3 is
distributional. **G2 and G3 are coordinator-chosen pending the user.**

### What that does to the arms

Scored against the eager reference, which is the pairing G2 is defined against:

| arm | request | G1 mean / max | G2 (x floor) | G3 |
|---|---:|---:|---:|---|
| BF16 streamed + Sage2 everywhere | 165.9 s | 0.1858 / 0.4738 | 1.28x / 1.14x — **fails** | +0.39%, passes |
| FP8-min + Sage2 | ~110 s | 0.2791 / 0.5649 (vs compiled ref) | ~1.9x — fails | — |
| FP8-min + platform attention | 175.5 s | 0.262 / 0.551 (vs compiled ref) | ~1.8x — fails | — |

Sage2 everywhere misses G2 **by 2%** on the mean. Every FP8 arm misses it by
nearly a factor of two, which is the same conclusion the weight-error table
reaches from the other end: FP8's error is a property of the format, not of
anything that can be tuned.

Note what a compiled arm scored against an eager reference is carrying: the
compile-vs-eager difference *and* its own approximation. The same Sage2 outputs
against the *compiled* set A reference are 0.1682 / 0.3693. Both pairings are in
the ledger and they answer different questions; the eager one is what G2 is
defined against.

## One-byte weights are closed on sm_120, so the lever is which blocks approximate

### INT8 would be the right format and has no kernel here

The weight-error table below says INT8 at per-output-row scales costs 2.9x less
than FP8 E4M3 at the same one byte per weight. It cannot be spent on this GPU.
CUTLASS's own dispatch refuses:

    RuntimeError: dispatch_scaled_mm, scaled_mm_helper.hpp:34,
    Int8 not supported on SM120. Use FP8 quantization instead, or run on
    older arch (SM < 100).

reached by the repo's CUDA smoke test for `Int8LinearMethod.apply` rather than
inferred from a serving failure. The weight-only route is closed too: vLLM's
online quantization registry has no INT8 linear, its supported online weight
keys being FP8 (per-tensor, per-channel, per-128-block) and the MX formats.

So the one-byte formats available on consumer Blackwell are FP8 E4M3 and the MX
family, which share E4M3's 3-bit mantissa, and NVFP4, which has fewer bits
still. With the scale-invariance result below, **no sub-BF16 weight format on
this GPU gets under ~2.6% relative weight error**, and 2.6% measures LPIPS 0.262
against the BF16 reference -- against a 0.15 limit. The weights have to stay
BF16, and precision is not the lever for either track.

One real bug came out of the attempt and is fixed:
`Int8OnlineLinearMethod.process_weights_after_loading` called the CUDA-only
`scaled_int8_quant` on `layer.weight` wherever it happened to be, which under
layer-wise offload is host memory -- so online INT8 plus offload died during
load on *any* architecture, not only this one.

### SageAttention's accuracy knobs are dominated

With the weights fixed at BF16, attention is the only lever, and Sage2's default
dispatch is too lossy (0.1682/0.3693, below). `arms/sage2-accurate.json` turns on
the accuracy settings -- INT8 QK at per-thread granularity, PV in FP16 with FP32
accumulation instead of FP8, smooth_k -- and is worse on both axes:

| arm | a1 mean/max | a2 mean/max | request |
|---|---:|---:|---:|
| Sage2, default dispatch | 0.0495 / 0.0865 | 0.1551 / 0.1780 | 163.9 s |
| Sage2, accuracy knobs on | 0.0563 / 0.0941 | 0.1867 / 0.2148 | 192.8 s |

A two-prompt screen, which is enough to stop an arm and never enough to pass
one. The mechanism is in the dispatcher: on sm_120 `sageattn` already selects
`pv_accum_dtype="fp32+fp16"`, a two-level accumulation, so FP16 PV with FP32
accumulation is not an upgrade over what the default already does, and
per-thread QK granularity costs time at 50,220 queries without buying it back.
**An accuracy knob is only an improvement relative to what the default actually
does**, which has to be read out of the dispatcher rather than assumed from the
knob's name.

### What was missing was a way to approximate *some* blocks

Both endpoints are measured and neither is adoptable -- Sage2 on all 60 blocks
is 165.9 s and LPIPS 0.1682/0.3693, the platform default is 232.2 s and exact --
and nothing in between could be expressed, because an attention config is
per-role and a role spans every block. `AttentionSpec.layers` now takes a
half-open range of layer indices, and a layer outside it falls through to the
rest of the existing lookup, so:

```json
{"per_role": {"kandinsky6": {"visual_self": {"backend": "SAGE_ATTN", "layers": "6:54"}}}}
```

puts the fast kernel through blocks 6-53 and leaves the platform default at both
ends, where a perturbation has the most of the network left to amplify it or
lands nearly in the output. The server log confirms it resolves both ways for
the same role:

    Resolved diffusion attention backend 'SAGE_ATTN' for role='kandinsky6.visual_self' via attention_config.per_role
    Resolved diffusion attention backend 'CUDNN_ATTN' for role='kandinsky6.visual_self' (platform default)

## What actually limits W1 on one 5090, and what a one-byte weight costs

Three measurements that together pick the arms worth running. All on
Kandinsky 6 **Pro-distill** at W1 (864x480, 121 frames, 10 PiFlow steps,
guidance 1.0, audio on; 50,220 visual tokens through 60 blocks), scored against
Track M's canonical BF16 reference for prompt set A.

### The streamed BF16 arm is compute-bound with the platform's attention

The Pro DiT is **56.14 GiB** in BF16 -- 60 visual blocks of 0.899 GiB, four text
blocks totalling 2.13 GiB, 0.09 GiB of embeddings and heads -- which fits
neither the 32 GB board nor the 60 GB host. It runs from the mmapped checkpoint
(`--enable-distributed-layerwise-offload --dlo-no-use-allgather`), so every one
of the 10 steps re-reads all of it.

| streamed BF16 arm | s/step | request (median) |
|---|---:|---:|
| platform default attention (cuDNN), = the reference's configuration | 21.9 | ~240 s |
| `arms/tuned.json` (Sage2 on `visual_self` and `video_audio_cross`) | 14.3-14.6 | **165.9 s** |

During the run `iostat` showed **1.67 GB/s** from `nvme0n1` with `Cached:
59.4 GB` and `Mapped: 57.0 GB`: about 24 GB of each step's 60.3 GB comes from
the device and 36 GB from the page cache, a hit rate near 0.6 -- which is what
LRU gives for a sequential rescan of a working set 1.06x the cache.

So the stream is worth about 15 s/step, cuDNN's compute about 22 s, and Sage2
takes the arm down **to the stream's floor and no further**. Two things follow.
A faster attention kernel is worth a third of this arm's wall time, not the
-37.4% it was worth on the pinned FP8 arm. And 14.5 s/step is the floor for BF16
weights however fast attention gets, so an arm that wants to be fast has to cut
bytes *and* keep the GEMMs fast.

### SageAttention2 fails the user's gate on exact weights

The same arm's quality, so the only difference from the reference is the
attention kernel:

| prompt | categories | LPIPS mean | max |
|---|---|---:|---:|
| G1 limits | | 0.15 | 0.25 |
| a5-waterfall-drone | motion | 0.0451 | 0.0478 |
| a1-portrait-speech | face, speech | 0.0495 | 0.0865 |
| a4-chalkboard | text, face, speech | 0.1391 | 0.1756 |
| a2-neon-signage | text, motion | 0.1551 | 0.1780 |
| a6-blacksmith | sharp-sound, face, motion | 0.1570 | 0.2128 |
| a9-violinist | face, motion | 0.1806 | **0.2822** |
| a3-sprint-start | motion, face, sharp-sound | 0.2435 | **0.3277** |
| a7-cafe-menu | text, face | 0.2634 | **0.2908** |
| a8-skateboard-crash | motion, sharp-sound | 0.2805 | **0.3693** |
| **set** | | **0.1682** | **0.3693** |

Four of nine prompts over the max. This is the Lite screen's failure (mean
0.118, max 0.375) reproduced on Pro against a real reference, and it refutes an
inference worth recording because it was wrong in an instructive way: the FP8
stack with Sage2 scored 0.2791 and FP8 alone 0.262, from which we had reasoned
that attention was worth about 0.017. It is worth 0.168. Perceptual errors of
this kind do not add -- the larger one hides the smaller -- so a stacked
measurement attributes nothing to its smaller component.

### FP8's weight error ignores scale granularity; INT8's does not

`tools/weight_quant_error.py` quantizes the checkpoint's own tensors and reports
the relative error of one quantize/dequantize round trip,
`||W - dequant(quant(W))||_F / ||W||_F`. It needs no GPU, no server and no video,
because it asks only about the weights. Median over 14 sampled 2-D weights
spread across the stack:

| format | per-tensor scale | per-output-row scale |
|---|---:|---:|
| FP8 E4M3 | 0.02645 | 0.02643 |
| INT8 | 0.02057 | **0.00908** |

**FP8's error does not care about the scale.** E4M3 carries its own 4-bit
exponent, so a finer scale only slides the matrix along the exponent ladder
while the quantization step stays at the 3-bit mantissa. It holds even for the
sampled tensor whose amax is 76x its median row's
(`va_modulation.out_layer`, where per-row INT8 is 2.5x better and per-row FP8 is
1.01x better).

That retires three hypotheses at once, before any of them cost a GPU hour: a
per-row FP8 checkpoint, a wider FP8 keep profile, and FP8 scale tuning in
general. It also explains the two things the gate numbers had made puzzling --
why the measured 0.262 did not move between the `minimal` (15 tensors kept in
BF16) and `sensitive` (367) keep profiles, since the error is per-weight and
uniform rather than concentrated; and why a per-row-weight plus
per-token-activation FP8 recipe scored *worse* at 0.338-0.364, since the weight
half bought nothing and the activation half added a second error.

INT8 is fixed point, so there the scale **is** the step, and a per-row scale buys
real precision: 2.9x less error than FP8 for the same one byte per weight. The
recipe DB's MiniMax-H3 entry, the nearest joint video+audio analogue, used INT8
linears for the same reason (inherited evidence,
`/data/jooman/k6/db/EVIDENCE.md`).

vLLM-Omni serves it from the exact published checkpoint with no conversion step:
`--diffusion-quantization-config int8` is `DiffusionInt8Config`, which quantizes
each tensor as the checkpoint streams -- per-output-channel weight scales with
dynamic per-token activation scales, so the GEMMs run on INT8 tensor cores, and
`ignored_layers` can hold named layers in BF16 without rebuilding anything.

## The quality gate: SageAttention on Kandinsky 6 is lossy, not approx

This is the section that decides whether the speed numbers below are
adoptable, and the answer is no for the attention arms.

Scored the way [PLAN.md](PLAN.md) sets the gate: the same checkpoint, the same
prompts, the same seeds, LPIPS per frame against the arm the model ships with
— which is cuDNN in bf16 and so *is* the BF16 reference the gate asks for.
Prompt set A, eight prompts, Kandinsky 6 Lite at W1's geometry. (Lite because
Pro does not fit a 32 GB card with its text encoder.)

| arm | set mean LPIPS | set max | verdict | prompts over the 0.10 max |
|---|---:|---:|---|---:|
| gate limits for `approx` | 0.05 | 0.10 | — | — |
| `arms/tuned.json` (Sage2) | **0.1178** | **0.3745** | **FAILS** | 6 of 8 |
| `arms/sage3.json` (Sage3) | **0.2530** | **0.5310** | **FAILS** | 8 of 8 |
| *the same arm twice* | 0.0023 | 0.0030 | passes | 0 |

### The last row is the one that makes the rest quotable

Running the shipped arm twice — same prompt, same seed, same configuration —
gives LPIPS mean 0.0023, max 0.0030, PSNR 52.3 dB. That is the **noise floor**
of this pipeline, and it cost one extra request. Without it, 0.1178 is a
number you have to argue about. With it, Sage2's mean is **50x the floor** and
its max is **125x**, so neither failure is measurement noise and neither needs
defending.

### The failure is prompt-dependent, by a factor of 62

Sage2, per prompt:

| prompt | LPIPS mean | LPIPS max | PSNR | SSIM |
|---|---:|---:|---:|---:|
| a1 face close-up | 0.0046 | 0.0060 | 47.8 dB | 0.995 |
| a2 person speaking | 0.0146 | 0.0173 | 41.2 dB | 0.991 |
| a7 sharp sound | 0.0883 | 0.1311 | 33.3 dB | 0.964 |
| a4 text on screen | 0.1006 | 0.1124 | 28.9 dB | 0.933 |
| a5 fast motion | 0.1228 | 0.1908 | 33.2 dB | 0.939 |
| a8 sharp sound + music | 0.1295 | 0.1506 | 28.9 dB | 0.881 |
| a6 fast motion, crowd | 0.1983 | 0.3072 | 25.2 dB | 0.860 |
| **a3 rendered text** | **0.2839** | **0.3745** | **20.4 dB** | **0.598** |

Best on a face at 0.0046 — twice the noise floor, which anyone would call
lossless. Worst on rendered text at 0.2839, with SSIM 0.598. **Had this been
scored on one prompt, and had that prompt been the face, the arm would have
been published as `reorder` tier.**

That is `c-qwen-image21-fp8-quality-is-prompt-dependent-2026-09` reproduced on
a different model family with a different kernel: an INT8/FP8 attention recipe
that is near-lossless on faces and destroys rendered text. It is the reason
PLAN.md demands two disjoint prompt sets and why both of them require a text
prompt.

### What the kernel-level error did and did not predict

The attention-kernel rel L2 against fp32 (0.039 for Sage2, 0.188 for Sage3)
got the **order** right: Sage3 is 4.8x the kernel error and lands at 2.1x the
set mean LPIPS. It got the **tier** wrong. A kernel rel L2 of 0.039 reads as
small, and on an image it is lossy. **A kernel error is not a tier**, and the
upper-bound argument in the attention-race section — that synthetic
activations overstate a quantizing kernel's error — did not save it.

### The audio half of the gate is not usable yet

SI-SDR is **−1.4 dB for the control**: the same arm, same seed, run twice. At
−1.4 dB the two audio tracks are substantially different signals, so the audio
metric cannot currently distinguish an arm from a rerun. The video half of the
same control is clean (LPIPS 0.0023, PSNR 52.3 dB), so this is the audio
branch, not the harness. No audio figure in this document is quoted as an arm
effect, and the audio branch needs a seeded deterministic path before the
gate's audio half means anything.

### What is adoptable

`arms/lossless.json` moves only the two cheap audio roles to `TORCH_SDPA` and
leaves `visual_self` on the platform default. Both are dense bf16 kernels
computing the same operation as cuDNN, so the output should sit at the noise
floor, and the swap was worth about 1.9 ms a block on a Pro-shaped block —
roughly 114 ms a step over 60 blocks. Small and free, which is the opposite
trade from the arm above.

## End to end: Kandinsky 6 Lite at W1's geometry

The measurements above are one synthetic Pro block. This is a whole request.
Lite (3.7B) because Pro does not fit on a 32 GB card with its text encoder,
and W1's geometry so the DiT sees the same 50,220 visual tokens as Pro would.
Offline `Omni(...)` through
`examples/offline_inference/text_to_video/text_to_video.py`, one request per
arm, `--enable-cpu-offload`, seed 42, all five host GPU locks held.

| arm | generation | vs control | steady-state step | peak reserved |
|---|---:|---:|---:|---:|
| shipped (platform attention, compiled) | 104.40 s | — | 7.38 s | 29.17 GiB |
| shipped | 101.56 s | — | — | 29.17 GiB |
| shipped | 99.61 s | — | — | 29.17 GiB |
| **`arms/tuned.json`** | **78.22 s** | **−23.0%** | **4.00 s** (−45.8%) | 29.17 GiB |
| `arms/tuned.json` again | **72.53 s** | **−28.6%** | — | 29.17 GiB |
| `arms/tuned.json` + `max-autotune-no-cudagraphs` | 142.56 s | +40.4% | 4.00 s | 29.17 GiB |
| `arms/tuned.json` + `mode="max-autotune"` | **raises** | — | — | — |
| `mode="reduce-overhead"` | **raises** | — | — | — |

Three controls spanning 99.61–104.40 s (median 101.56, spread 4.7%) and two
candidate runs at 78.22 and 72.53 s (median 75.38), so the effect is
**−25.8%** on the medians — against a control spread of 4.7%. The two
candidate runs differ by 7.3%, more than the controls do, which is worth
saying rather than hiding: both were cold processes and the spread of a
single-request cold measurement is simply wider than the steady-state step
figure beside it. Output verified as H.264 864x480, 121 frames,
5.06 s, plus AAC 44.1 kHz (219 audio frames) — the joint path, not video only.

The **steady-state step** is the slope of the progress bar between step 2 and
step 10, which separates the first step's compilation from the per-step
compute. It matters because the two columns tell different stories: the
attention arm takes a step from 7.38 s to 4.00 s (**−45.8%**) while taking the
request only −23.0%, because a request also carries the first step's compile
and about 19 s of stages outside the denoise loop. On a warm server serving
many requests the per-step figure is the one that compounds; for the
single-request headline the whole-request figure is the honest one.

### `max-autotune-no-cudagraphs` gives this model nothing

It is the one value of `--diffusion-compile-mode` that does not capture CUDA
graphs, so it was the remaining candidate after the other two raised. It costs
**+40.4%** on the request and its **steady-state step is identical to the
attention arm's, 4.00 s**: the entire difference is compile time, about 65 s
more in the first step, and the Triton GEMM templates buy nothing back.

That contradicts the block measurement, where `max-autotune` took the block
GEMMs from 145.55 ms to 125.15 (−14%) — and the resolution is a scale the two
measurements do not share. The block was **Pro-shaped** (`model_dim` 4096);
this request is **Lite** (1792). Triton beating cuBLAS at one GEMM width says
nothing about another. So the flag may still pay on Pro, and that is
untestable on this host until Pro fits.

The general caution is the one worth keeping: a Pro-shaped block result does
not transfer to a Lite-shaped request, in either direction.

### The block-level CUDA-graph win does not survive a real pipeline

`mode="max-autotune"` was −38.5% on one block in isolation. On a request both
it and `mode="reduce-overhead"` raise:

```
RuntimeError: Error: accessing tensor output of CUDAGraphs that has been
overwritten by a subsequent run.
  ... kandinsky6_transformer.py Kandinsky6TransformerEncoderBlock.forward
  ... kandinsky6_transformer.py apply_gate_sum
```

The cause looks structural, not incidental. `regionally_compile` compiles each
repeated block, so a DiT replays many captured graphs back to back — and the
**residual stream holds a reference across block boundaries**:
`apply_gate_sum(x, out, gate)` reads the previous block's output. CUDA-graph
trees assume a graph's output is consumed before the next replay, so that
reference points at reclaimed memory. Nothing here is Kandinsky-specific; any
model whose `_repeated_blocks` pass a residual through should be expected to
hit it.

Both modes fail identically, which is what identifies the graph capture rather
than `max-autotune`'s GEMM autotuning as the cause.

**This is the single most useful thing the end-to-end run produced.** A 38.5%
block-level win that raises on the first real request is worth less than
nothing if it is published as a speedup, and nothing in the block measurement
— timing, output comparison, or profile — could have revealed it. The arms
were all one block deep, and the failure needs two.

Taken with the `max-autotune-no-cudagraphs` result below, the honest summary
of `--diffusion-compile-mode` on Kandinsky 6 is that **no value of it helps**:
the two that capture graphs raise, and the one that does not costs compile
time for no steady-state gain at this model's GEMM widths.

### So the adoptable change is the attention arm

−23.0% on a whole request, from one config value and no code change:

```bash
python examples/offline_inference/text_to_video/text_to_video.py \
    --model kandinskylab/Kandinsky-6.0-Lite-5s-Diffusers \
    --model-class-name Kandinsky6TI2VAPipeline --enable-cpu-offload \
    --height 480 --width 864 --num-frames 121 --num-inference-steps 10 \
    --diffusion-attention-config showcase/kandinsky6/compute/arms/tuned.json
```

> **It fails the quality gate.** When this was written its only accuracy
> evidence was an attention-kernel error on synthetic activations. Scored
> properly (see the gate section above) it is set mean LPIPS 0.1178 against a
> 0.05 limit, so the arm is **lossy**, not `approx`, and the −23% is a lossy
> speed number rather than an adoptable default.

### Two numbers for Track M

**About 19 s of the request is outside the denoise loop** — text encode, video
VAE decode, audio decode, mux. Measured on the profiled run: 114.40 s total
against a 95.4 s denoise loop. That is 17% of a request that none of the block
work touches.

**Peak reserved is 29.17 GiB of 32, and the peak is the VAE decode, not the
DiT.** The process sat at 30.5 GB with the denoise already finished, and the
figure is identical across every arm — including the one that made the DiT 23%
faster. For a 3.7B model. Whatever fits Pro will be decided by the decode as
much as by the weights, so `--vae-use-slicing`, `--vae-use-tiling` and the
batch-parallel decode are worth pricing before any more DiT work.

## Making W1's checkpoint serve on one 5090 with a 60 GB host (Track M)

Track M, build-server-2, 2026-10-06. `kandinskylab/Kandinsky-6.0-Pro-distill-5s-Diffusers`
(29B) now serves W1 end to end on one RTX 5090 with an FP8 DiT and pi-Flow
sampling: the first correct W1 request (21:13 KST) returned a 2.4 MB MP4 with
121 frames and 44.1 kHz audio, every DiT module output finite. **Its 249.5 s is
not a headline**: it was one request on a server with per-module NaN probes on,
which synchronize after every module. The timed W1 baseline follows in its own
section. Two defects remain open and are stated below: the decoded video is
816x448 rather than 864x480, and host RSS is 33 GB against a 25 GB goal.

The first end-to-end requests returned a black, silent MP4 with
`status: completed`; the history of that bug and its fix is kept below because
the fix is a whole-class one (any FP8 layer meant to stay BF16).

### FP8 conversion of the DiT

`tools/quantize_dit_fp8.py` on this host:

| | |
|---|---:|
| Input (BF16, 4471 tensors) | 60.3 GB |
| Output (FP8 E4M3 + per-tensor scales) | 32.3 GB |
| Compression | 1.87x |
| Wall time | 75 s, then 63 s |
| Linear weights quantized / kept BF16 | 1856 / 135 |

Round-trip error, dequantizing sampled weights against the originals: 1.8% to
3.5% of each tensor's maximum (block 5 `to_query` 3.5%, block 5 FFN `in_layer`
1.8%, block 30 `visual_modulation` 3.3%). No NaN or Inf in a full sweep of a
4 GB shard, so the checkpoint itself is sound.

Scales are per **tensor**, not per channel, because vLLM's native `fp8` method
builds a `PerTensorScaleParameter` and a per-channel scale fails its shape
assertion on load. `--scale channel` is implemented for a later
`compressed-tensors` checkpoint and should be the better-quality arm: one scale
over a 4096x16384 matrix means every row loses range to that matrix's single
largest weight.

### Host RAM is the binding constraint, not the 32 GB of GPU

| arm | DiT weights | fits 32 GB GPU | fits ~55 GB usable host RAM |
|---|---:|---|---|
| BF16 resident | 60.3 GB | no | no |
| BF16 streamed per block | 60.3 GB | n/a | **no** -- needs NVMe, not host RAM |
| FP8 streamed from host (this arm) | 32.3 GB | n/a | **marginally** |
| NVFP4 resident | ~15 GB | plausible, untested | yes |

Measured on the FP8 arm with `--enable-layerwise-offload`:

| | |
|---|---:|
| Worker RSS at steady state | 52.4 GB |
| Worker swapped out | 5.6 GB |
| Host swap in use | 7 GB of 7 GB |
| Host memory available | 1-4 GB |
| GPU in use after load | 18.1 GB |
| GPU peak during a smoke request | 26.3 GB |

The DiT's 32.3 GB and the Qwen2.5-VL-7B text encoder (~16 GB BF16) are both
host-resident, with the layerwise backend's flattened copies beside the
checkpoint's mapped pages. Three consequences, all measured:

- **Load time is bimodal.** About 100 s with the checkpoint warm in the page
  cache from having just been written; 5-9 minutes cold with swap in use. Cold
  start is a property of the page cache's state here, not of the checkpoint, and
  has to be reported with that state named.
- **The BF16 reference cannot reuse this path.** BF16 does not fit host RAM, so
  R0 needs block-by-block streaming from NVMe, not the headline arm's offload.
- **The next precision step should target the text encoder.** Its 16 GB is a
  bigger prize than the DiT's remaining 2 GB of BF16 weights.

This also made the host unusable for anything else: at 20:05 the box had 3 GB
available, swap full and a load average of 110 with a build running next to the
server. One server at a time on a 60 GB host, and nothing heavy beside it.

### Smoke geometry, served (plumbing evidence, not a workload)

512x320, 25 frames, 10 steps, guidance 1.0, seed 42, audio on.

| | |
|---|---:|
| Request wall time, first request on a fresh server | 32.6 s |
| Request wall time, warm | 6.65 / 6.64 / 6.70 s, three separately started servers |
| Denoise loop | 10/10 steps at 1.86 it/s |
| GPU peak (allocator) | 26.3 GB |

The smoke geometry has about 4.5k visual tokens against W1's 50k, so it
exercises the plumbing and not the workload.

### The zero-output bug: fixed (`ca247172d`), and how it was found

| hypothesis | tested by | verdict |
|---|---|---|
| The FP8 checkpoint is corrupt | dequantized sampled weights; NaN/Inf sweep | ruled out |
| The pi-Flow math is wrong | 23 unit tests against a naive oracle built from the published recursion | ruled out at the math level |
| The pi-Flow loop never runs | it logs its sampler; the request logs `sampler=piflow steps=10` with a 10/10 progress bar | ruled out, it runs |
| The Euler loop runs instead | same log line | ruled out |
| Latents promoted to FP32 into an FP16 VAE | found, fixed, re-measured | a real bug, fixed, **not** the cause |
| A dtype or shape fault in the loop | GPU integration test against a small real DiT | ruled out |
| The VAE or muxer is at fault | Track C's BF16 Lite run produces a correct MP4 on the same code | ruled out |

Traced to its origin: the **DiT forward returns NaN from step 0** (2,867,200 of
2,867,200 elements), and the VAE then clamps NaN to zero, which is why both
streams are exactly zero and the file is byte-identical across prompts. So the
fault is upstream of the sampler, in the FP8 DiT itself or in the wide
pi-Flow head. The two differ from Track C's working arm in exactly those two
ways, which is what the next experiment separates: Lite-**distill** is BF16 *and*
pi-Flow *and* has the same n_grid-10 heads, so a correct video there indicts FP8
and a NaN there indicts the head handling.

**Cause and fix.** It was neither of those: Lite-distill in BF16 produced correct
video on this branch, so pi-Flow and the wide heads were fine, and FP8 Lite
reproduced the NaN in a one-minute loop. An in-worker locator then named the
first non-finite module -- `video_text_embeddings.in_layer`, a layer the
checkpoint meant to keep in BF16. Two faults, both in naming the unquantized
layers: every layer prefix began with "." (the DiT's root prefix is empty and
`f"{prefix}.{name}"` was unguarded), and the quantizer wrote `ignored_layers` as
checkpoint keys while vLLM matches module paths, which differ for the three
remapped families. Either way the layer was built as FP8, its `weight_scale`
was absent from the checkpoint, and nothing requires a scale to be filled, so it
kept its `torch.empty` contents. Ruled out on the way, each by a run:
modulation quantized, text padding into cross-attention, 0-dim scales, and
CLIP (the pooled embedding entering the DiT was finite). After the fix FP8 Lite
returns a 281 KB MP4 (pixels mean 150.6, std 49.0; audio rms 0.186).

Instrumentation left behind, gated on `VLLM_OMNI_K6_PIFLOW_DEBUG=1`: per-step
latent and DiT-output statistics inside the pi-Flow loop, plus the final latents,
the decoded video and the decoded audio. That is what localized this, and it
separates "degenerate latents" from "a decode that zeroes a good latent" in one
request.

### What the port could not do before this session

Each was a hard failure, not a slowdown:

1. **No pi-Flow.** `model_index.json` names `PiflowScheduler`; the port had only
   the shifted-Euler stepper, and `PiflowScheduler` is not in diffusers 0.40.0 --
   it exists only in Kandinsky's patched fork.
2. **A 10x wider output head** (`out_visual_dim` 160 for `in_visual_dim` 16).
   `n_grid` is derived from the head so the two cannot be configured apart.
3. **`out_audio_dim` was not a DiT parameter.** The audio head was always sized
   from `in_audio_dim` -- right for plain Pro (40 == 40), wrong for distilled
   (400 != 40).
4. **Pre-quantized checkpoints could not load.** `OmniDiffusionConfig`
   auto-detects a checkpoint's `quantization_config` and resolves it through
   vLLM-Omni's factory, which maps `fp8` to the *online* `DiffusionFp8Config` --
   which cannot load the serialized checkpoint it was detected from. Any
   pre-quantized diffusion checkpoint hits this, not just Kandinsky.
5. **Weight names.** Covered by Track C's PR #21, which this branch adopts.

### Two process findings, each of which cost measurable time

- **Worker stdout is not forwarded into the server log**, while `logger.*` from
  the same module is. A missing `print` is therefore not evidence that a code
  path did not run, and about 40 minutes went into conclusions drawn that way.
- **Killing `vllm serve` leaves the `DiffusionWorker` alive and serving** on the
  port. One round of measurements was taken against the previous build before
  this was noticed; the harness's `serve.py` signals the process group and waits
  for the GPU to drain for exactly this reason.

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

The served arms and their gate numbers:

```bash
cd showcase/kandinsky6/compute
# One arm's W1 outputs for a prompt set, from real served requests.
#   --offload dlo-mmap    exact BF16 weights, the reference's own weight path
#   --offload layerwise   a pre-quantized checkpoint staged from pinned host RAM
#   --quantization '{"method": "..."}'   quantize at load from the exact checkpoint
#   --resident-layers N   keep N leading DiT blocks on the device. Lossless, and
#                         it does not survive compilation on this model -- see
#                         gate_pro.py's docstring.
python run_when_free.py --need-free-gib 30 -- python gate_pro.py \
    --arm arms/sage2-mid.json --name sage2-mid --offload dlo-mmap \
    --out-dir /data/jooman/k6/results/gate-pro/sage2-mid

# All three gates at once, with a contact sheet for whichever prompt scored worst.
./score_arm.sh /data/jooman/k6/results/gate-pro/sage2-mid /data/jooman/k6/ref-eager/setA

# G1 and G2. The floor comes from the reference measured against itself,
# compiled against eager; without it G2 reports "undecided" rather than a pass.
python gate_pro_score.py --arm-dir /data/jooman/k6/results/gate-pro/sage2-mid \
    --reference-dir /data/jooman/k6/ref-eager/setA \
    --g2-floor-mean 0.1455 --g2-floor-max 0.4160

# G3 and the contact sheet. CPU by default: the GPU belongs to whatever is timed.
CUDA_VISIBLE_DEVICES= python clip_gate.py \
    --arm-dir /data/jooman/k6/results/gate-pro/sage2-mid \
    --reference-dir /data/jooman/k6/ref-eager/setA --contact-sheet sheet.jpg
```

Scoring runs on the GPU unless `CUDA_VISIBLE_DEVICES=` is set, and it takes the
host locks for that reason: a scorer that ignores them steals memory from a
timed run, which happened once here and cost a measurement.

What a one-byte weight costs, without a GPU at all:

```bash
python showcase/kandinsky6/tools/weight_quant_error.py --limit 24
```
