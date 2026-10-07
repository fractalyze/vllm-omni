# ncu roofline outside the DiT block (Track M, build-server-2, 2026-10-07)

Round-6 prep, measurement only. Scope: everything a W1 request runs outside the
DiT block (video VAE decode, audio decode, text encoders, the DLO weight
stream, per-step scheduler/RoPE kernels), plus the exact-attention kernel
(cuDNN SDPA) for comparison with SageAttention2. The DiT block itself is Track
C's section.

**Setup.** RTX 5090 (sm_120), Nsight Compute 2026.2.1, `--clock-control none`:
the card runs at its real, 575 W-capped clocks rather than locked base clocks.
Every GPU lock was held, and no foreign GPU process was present before or after
any profile.

**Request shares** come from the round-5 nsys trace of the served head
(`runs/profile-r5/h1-hybrid`, one a1 request: a 158.7 s window, with 0.6 s
before the first DiT kernel and 19.2 s after the last). ncu profiled two units,
never a whole request:

- **One W1 VAE decode**, `ncu/vae_decode_unit.py`. Eager FP16, as served, with
  the tile plan the server logs at W1: tile (1, 17, 256, 448), stride
  (8, 224, 416). It takes 18.2 s, matching the served 18.7-18.9 s. Weights are
  random (scaled into FP16 range), since conv and GroupNorm cost does not depend
  on values.
- **One cuDNN SDPA call** at W1's visual self-attention shape (B 1, H 32,
  S 50,220, D 128, BF16), `ncu/sdpa_unit.py`: 175.5 ms under ncu, 177-178 ms
  timed. The served trace has 156 such calls at 180 ms each.

## Table

| kernel | share of a request | unit time | SOL SM / mem % | tensor pipe | occupancy | top warp stalls | class |
|---|---:|---:|---|---:|---:|---|---|
| cuDNN SDPA `sm120_flash_fprop_f16` (exact attention: step 1 and blocks 0-5, 54-59) | 28.15 s, 17.7% | 175.5 ms / call | 98 / 25 | 98% | 19% | math_pipe_throttle 8.0, wait 2.2, long_scoreboard 1.9 | compute-bound at peak |
| VAE conv3d, cuDNN `sm80_xmma_fprop_implicit_gemm_f16f16_f16f32` (128x128 and 256x128 tiles) | 9.50 s, 6.0% | 0.07-29 ms / launch | 97-98 / 27-54 on the large shapes; 59-88 / 22-47 on the sliver tiles | 95-99% | 8-17% | math_pipe_throttle 4.8-15, wait 4 | compute-bound at peak (FP32-accumulate rate) |
| VAE GroupNorm statistics, `RowwiseMomentsCUDAKernel<Half,float>` | 3.10 s, 2.0% | 0.8-1.4 ms / launch | 5 / 6 | 0 | 33% | long_scoreboard 33 | latency-bound: 32 CTAs on 170 SMs |
| DLO weight stream (H2D on the copy engine; not a kernel, from nsys) | 11.0 s of copy time, 93% overlapped | 920 MiB a block, 612 copies, 504 GiB a request | 57.4 GB/s median (p10 33.9), ~91% of PCIe 5.0 x16 | - | - | - | bandwidth-bound, hidden |
| replication pad, NCHW<->NHWC transposes, SiLU/add/upsample | 0.7% + 0.7% + ~1% | 3-110 us / launch (sampled) | 24-73 / 23-83 | 0 | 73-91% | long_scoreboard | memory-bound |
| cuDNN SDPA `sm80_flash_fprop_wmma` (text/audio shapes) | 2.46 s, 1.5% | - | not profiled (< 2%) | | | | |
| text encoders (all kernels before the first DiT step) | 0.6 s, 0.4% | - | not profiled (< 2%) | | | | |
| audio decode + mux | < 1 s, < 0.7% | - | not profiled (< 2%) | | | | |

Scheduler, RoPE and embedding kernels run inside the compiled step and are each
far below 1%.

## Why each sits where it does

- **cuDNN SDPA** keeps the tensor pipe 98% busy and stalls mostly on
  `math_pipe_throttle`: it is at its compute ceiling. 233 TFLOP/s counting
  4·S²·D·H, which is the FP32-accumulate MMA ceiling at the clocks it ran at,
  not the FP16-accumulate one. Only a precision change moves it:
  FP16-accumulate dots, as Track S's H3 screen measured (1.21x for both
  products in Triton). Occupancy is 19% (288 threads, 168 registers), which is
  normal for a flash kernel that is MMA-bound.
- **VAE convs** are at the same ceiling. cuDNN's `f16f16_f16f32` kernels take
  FP16 operands and accumulate in FP32, and the pipe is 95-99% active on every
  shape that matters. The kernels are not the problem; **the amount of work
  is**:
  - **Temporal:** the served plan decodes 5-latent-frame chunks every 2 latent
    frames (`_temporal_tiled_decode`). That is 14 chunks, 70 latent-frame
    decodes for 31 frames: **2.26x**.
  - **Spatial:** each chunk is cut into 3x3 tiles (latent 32x56, stride 28x52),
    whose last row and column are 4-latent slivers. That is 68x116 latent area
    for 60x108, **1.21x**. The slivers also run the small, less efficient conv
    grids (SOL SM 59-88%).
  - Together the convs do about **2.7x** the work of a non-overlapping decode.
- **GroupNorm statistics** launch one CTA per (batch, group): 32 CTAs on a
  170-SM GPU. SOL SM and memory both sit near 5%, and warps wait on
  `long_scoreboard`, so this is latency-bound by its launch shape, not by
  bandwidth.
- **The weight stream** moves 920 MiB a block at about 57 GB/s, about 91% of
  PCIe 5.0 x16. 93% of its copy time overlaps kernels and the GPU idles 7%. It
  is not a lever while the step stays compute-bound.
- **Pad, layout transposes and elementwise** are memory-bound at 57-83% of DRAM
  on the launches sampled. The NCHW<->NHWC transposes exist only because the VAE
  runs NCHW and cuDNN's conv wants NHWC.

## Ranked hypotheses for round 6 (outside the block)

1. **Remove the VAE's redundant work.** A causal, cached temporal decode
   carries the causal conv state from chunk to chunk instead of re-decoding
   overlapping chunks, and the tiles should be non-sliver. Estimate: convs 9.5 s
   -> ~4 s, decode 18.2 -> ~10 s, **about -8 s a request**. The pixels change
   against today's blended tiles (towards an untiled decode), so it needs its own
   G1 against the same-code reference, and memory decides how large a chunk fits
   beside the DLO buffers.
2. **FP16-accumulate exact attention.** It is at the FP32-accumulate ceiling;
   Track S measured 1.21x for a Triton FP16-accumulate attention at this shape.
   **About -4.7 s.** A numerics change, so it needs its own reference.
3. **Parallelise GroupNorm.** Split each group's reduction across CTAs (Welford
   partials), or use a channels-last fused GroupNorm + SiLU in Triton.
   **About -2.5 s**, exact up to summation order.
4. **channels_last VAE.** Removes the NCHW<->NHWC transposes (1.14 s) and lets
   the elementwise kernels run in the conv layout. **About -1 s**, exact.

Not a lever: the DLO weight stream, the text encoders, audio decode, and the
small-shape SDPA (each under 2%, or hidden).

**Limits of this measurement.**
- The pad/transpose/elementwise rows come from the first 60 matching launches
  of the decode, which are the early, smaller ones. Their request shares come
  from nsys, not ncu.
- The VAE profile used random weights. The conv and GroupNorm work is the same
  as served; data-dependent kernels would not be, and none appear in the
  decode.
- The redundancy estimates are arithmetic on the tile plan, not a measured
  rewrite.

## Reproducing

```bash
cd showcase/kandinsky6/compute/ncu
export PYTHON=/data/jooman/k6/venv/bin/python
./ncu_run.sh sdpa_w1    "sdpa"                 0   1  sdpa_unit.py
./ncu_run.sh vae_conv_a "implicit_gemm"        0   60 vae_decode_unit.py
./ncu_run.sh vae_conv_b "implicit_gemm"        60  60 vae_decode_unit.py
./ncu_run.sh vae_conv_c "implicit_gemm"        120 40 vae_decode_unit.py
./ncu_run.sh vae_gn     "RowwiseMoments"       0   40 vae_decode_unit.py
./ncu_run.sh vae_misc   "replication_pad|nchwToNhwc|nhwcToNchw|silu|upsample|elementwise" 0 60 vae_decode_unit.py
python summarize.py vae_conv_a.ncu-rep vae_conv_b.ncu-rep vae_conv_c.ncu-rep
```

`ncu_run.sh` runs `ncu --target-processes all --profile-from-start off
--clock-control none` with the SpeedOfLight, MemoryWorkloadAnalysis,
ComputeWorkloadAnalysis, SchedulerStats, WarpStateStats, Occupancy and
LaunchStats sections. It adds the tensor-pipe, DRAM, L2 and bank-conflict
metrics, a kernel-name regex, `--launch-skip` and `-c` (at most 60). The
harnesses bracket the measured unit with `torch.cuda.profiler.start()/stop()`
after a warm-up and hold the GPU locks (`bench/gpu_guard.py`). The reports are
on bs2 under `/data/jooman/k6/ncu/`.
