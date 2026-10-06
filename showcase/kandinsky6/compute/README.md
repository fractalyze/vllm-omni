# Kandinsky 6 compute tooling (Track C)

Kernel-level measurement for the single-RTX-5090 showcase
([PLAN.md](../PLAN.md), Track C). Track M's end-to-end harness lives in
`../bench/`; nothing here serves a request or scores quality.

The problem these solve: **Kandinsky 6.0 Pro does not fit on the GPU it is
being optimized for.** 29B parameters is 60 GB in BF16, against 32 GB of
device memory and 60 GB of host RAM. Waiting for a whole pipeline before
touching a kernel would mean waiting for Track M's quantized checkpoint, so
instead these build the port's *own* blocks at the Pro config's dimensions
with random weights and feed them W1's token counts. Every shape, launch
configuration and kernel is then the checkpoint's; only the weight *values*
are not, which costs nothing for timing and is why nothing here reports a
quality number.

## The shared GPU

This host's GPU is shared with other studies and other people's jobs. Each
study keeps an advisory lock file, so a timed run must hold **all** of them
and must still check `nvidia-smi` — a job that ignores the locks shows up
there and invalidates the run.

```bash
python gpulock.py --list                       # which locks exist, who is on the GPU
python run_when_free.py --need-free-gib 12 -- python attn_race.py --json race.json
```

`run_when_free.py` queues on the locks with a **blocking** `flock`, which is
what makes it fair. A non-blocking retry loop starves against a co-tenant
that re-takes its lock within the poll interval — observed on this host, where
the GPU came free and the lock was gone again inside 30 s. Once it holds the
locks it waits a bounded `--hold-wait-min` for the memory and then releases
them and queues again, so one stuck job cannot deadlock the host behind it.

`--no-locks` on either runner skips all of that. It exists for correctness
checks on a busy GPU; **a timing taken without the locks is not a
measurement.** When `run_when_free.py` invokes a runner it passes
`--no-locks` (it already holds them) together with `K6_LOCKS_HELD_BY`, so the
recorded run still names the locks it ran under.

## Tools

| file | what it does |
|---|---|
| `gpulock.py` | Holds every `gpu.lock` on the host, found by glob so a later study is honoured without a code change; names foreign GPU processes. |
| `run_when_free.py` | Queues for the GPU, then runs a command under the locks. |
| `block_profile.py` | Times one Pro- or Lite-shaped DiT block at W1's token counts and splits the step into attention / GEMM / elementwise / norm / copy, plus the launch gap. |
| `attn_race.py` | Races the registered diffusion attention backends at each of Kandinsky 6's five attention call sites, with accuracy against a chunked fp32 reference. |
| `kernel_classes.py` | The ordered kernel-name table the split comes from. |
| `session.sh`, `session2.sh` | The run sequences the recorded measurements used, one per GPU window. |
| `arms/` | One `--diffusion-attention-config` value per attention arm. |
| `results/` | The JSON each run wrote, so a published table can be audited from the tree. |
| `test_compute_tools.py`, `test_role_masks.py` | 53 tests, no GPU and no checkpoint. |

```bash
python block_profile.py --list-shapes                  # the derived token counts
python block_profile.py --attention-config arms/tuned.json   # one T2VA block at W1
python block_profile.py --attention-config arms/tuned.json \
    --compare-compile --compile default                # eager vs compiled, ABBA, one process
python attn_race.py --all-roles --json race.json       # the backend race
/data/jooman/k6/venv/bin/python -m pytest . -q
```

`--attention-config` takes an arm file, the same JSON a server is given as
`--diffusion-attention-config`, so a profiled block used the per-role
selection production would use. `--backend` is the blunter form (one backend
for every role) and the two are mutually exclusive, because they would
disagree per role.

`--compare-compile` times eager and compiled **in one process**, on the same
module and the same inputs, in ABBA order. A ratio taken across two processes
is a ratio across two clock states; this one is not.

On a shared GPU a free window is often big enough for several runs but gets
used for one if each queues separately. `session.sh` and `session2.sh` are the
sequences the recorded measurements used: `run_when_free.py` takes the locks
once and the script spends the window, with each run independent so one
failure does not cost the rest of it.

## W1's shapes

`--list-shapes` derives them rather than hardcoding them, so a different
geometry stays consistent: the Hunyuan VAE compresses 4x in time and 8x in
space and the DiT patches 1x2x2, so 121 frames at 864x480 become
31 x 30 x 54 = **50,220 visual tokens**, and the audio VAE's 44.1 kHz / 1024
downsample gives **218 audio latent frames** (the same arithmetic as
`pipeline_kandinsky6.audio_latent_duration`).

The five attention call sites at those counts, per fused block:

| role | queries | keys | heads x dim | TFLOP a call |
|---|---:|---:|---|---:|
| `kandinsky6.visual_self` | 50,220 | 50,220 | 32 x 128 | 41.32 |
| `kandinsky6.text_cross` | 50,220 | 256 | 32 x 128 | 0.21 |
| `kandinsky6.video_audio_cross` | 50,220 | 218 | 32 x 128 | 0.18 |
| `kandinsky6.audio_video_cross` | 218 | 50,220 | 16 x 128 | 0.09 |
| `kandinsky6.audio_self` | 218 | 218 | 16 x 128 | 0.0004 |

Visual self-attention is 98.8% of the attention FLOPs in a block, which is why
the race defaults to it. Over 60 blocks it is 2.48 PFLOP a forward, which is
where PLAN.md's "about 2.5 PFLOP" comes from.

## Why the block configs are not the port's defaults

`Kandinsky6Transformer3DModel.__init__` defaults to Pro's *widths*, but not to
the checkpoint's *flags*. Both bundles set `ca_rope`, `cross_gates`,
`fix_modulation` and `text_token_padding` true, and `cross_gates` changes the
cross-modal modulation's output width from `3*model_dim` to
`2*model_dim + model_dim_a`. A block built from the defaults is a different
block, so `PRO` and `LITE` in `block_profile.py` are read from the bundles'
`transformer/config.json` (snapshots `7a1a4033` and `6510114a`) and carry
those flags.

Pro-distill's `out_visual_dim` is 160 and `out_audio_dim` 400, ten times the
input dims: the distilled output layer emits `PiflowScheduler`'s 10-point
policy grid, not one velocity.

## A kernel no rule names is reported, never bucketed

`kernel_classes.py` leaves an unmatched kernel in `unclassified` and prints it
by name, and that design earned itself in the first profile of the tuned
attention arm: 61.43 ms of a 260 ms block (23.5%) landed there and attention
read as 0.7%, because the table had SageAttention's `qk_int8_sv` name from its
other variants but not the `qk_int_sv_f8_attn_kernel` its sm_120 path actually
emits, nor the `QuantInt8Kernel` and `MeanScaleKernel` prologues. Had the
table folded unknown names into elementwise, the number would have been wrong
and plausible. Those names are now in the table with a test each.

## What the accuracy numbers are and are not

`attn_race.py` scores each arm against softmax attention computed in fp32 on
the same q/k/v, with TF32 off and chunked over heads and query blocks — the
full score matrix would be 10 TB at W1, and leaving TF32 on would give the
"fp32" reference 10 mantissa bits, barely better than bf16's 8.

The activations are synthetic, because there is no real Pro forward on this
GPU to capture from. They are not `randn`: the port RMS-normalizes q and k per
head *before* RoPE, and RoPE is a rotation, so every row a kernel sees has
unit RMS by construction, and `make_activations` reproduces that with the
port's own RoPE module. What it cannot reproduce is the *correlation* between
neighbouring tokens in a real video latent, which makes the real softmax
peakier — and a peakier softmax is kinder to a quantizing kernel. A
SageAttention error measured here is therefore an **upper bound** on the real
one: the safe direction for ranking arms, but not a substitute for the
end-to-end quality gate.
