# ncu roofline of the DiT block at W1 (Track C, build-server-3)

Measurement only. Nsight Compute 2026.2.1, `--clock-control none` so the card
runs at its real 575 W-capped clocks (the default locks base clock and changes
the picture). GPU locks held, no foreign GPU process before or after either run.

**Unit profiled:** one *compiled* fused block at W1 (50,220 visual tokens, text,
audio), hybrid GEMM installed on the 12 large-M linears, warmed three times so
Triton/Inductor JIT is out of the region, then one forward inside
`torch.cuda.profiler.start()/stop()`. Harness: `compute/ncu_block.py`.

**Attention arm is `sage2-edge0.json`, not `sage2-mid.json`, deliberately.** A
block built at prefix `visual_transformer_blocks.0` is *outside* sage2-mid's
`6:54` band and would resolve to the platform default, so profiling it would
measure cuDNN rather than the SageAttention2 kernel the served arm actually runs
on 48 of its 60 blocks.

**Durations and request shares come from the served nsys trace, not from ncu.**
ncu replays each kernel 8–40× in isolation and reports a different wall time (it
read 249 us for a cast the served trace timed at 1449 us), so ncu is used only
for where a kernel sits against its bound, as the brief prescribes.

## Only three kernels carry ≥2% of a W1 request, and they are 70% of it

| kernel | s / request | % request | class |
|---|---:|---:|---|
| `_hybrid_mm` (hybrid FP16-acc GEMM) | 56.7 | **37.7%** | compute-bound below peak |
| `qk_int_sv_f8_attn_kernel` (SageAttention2, INT8-QK / FP8-PV) | 28.1 | **18.7%** | compute/latency-bound below peak |
| `cudnn_…sdpa_sm120_flash_fprop_f16` (exact attention) | 20.5 | **13.6%** | Track M's section |
| — everything below here is under 2% individually — | | | |
| `to_copy_gelu_t_view_24` | 1.8 | 1.2% | memory-bound |
| `to_copy_t_view_0` (standalone FP16 cast) | 1.6 | 1.0% | memory-bound at 87% DRAM |
| `…native_layer_norm…` (the 2279 us one) | 1.2 | 0.8% | memory-bound |
| `to_copy_permute_t_view_4` (cast + permute) | 0.9 | 0.6% | memory-bound at 87% DRAM |

That shape is itself the headline: **there is no long tail to optimise.** Any
round-6 win has to come out of the GEMM or the attention.

## Roofline position (ncu, medians over the launches captured)

| metric | `_hybrid_mm` | SageAttention2 | FP16 cast | cast+permute |
|---|---:|---:|---:|---:|
| Compute (SM) throughput | **75.0%** | **71.6%** | 10.7% | 2.7% |
| Memory throughput (SOL) | 118.3% | 26.4% | 42.5% | 44.4% |
| **DRAM throughput** | 8.9% | **0.9%** | **87.3%** | **87.2%** |
| L2 throughput | 81.8% | 37.3% | 26.5% | 26.3% |
| L1/TEX throughput | 38.7% | 25.3% | 12.2% | 12.1% |
| Mem pipes busy | 17.4% | 18.5% | 10.7% | 2.7% |
| Achieved / theoretical occupancy | 16.6 / 16.7% | 16.7 / 16.7% | 76.5 / 100% | 87.5 / 100% |
| Eligible warps per scheduler | 0.44 | 0.44 | 0.07 | 0.03 |
| **No eligible warp** | **68.0%** | **68.2%** | 93.9% | 97.5% |
| Executed IPC active | 1.28 | 1.27 | 0.24 | 0.10 |
| Avg active threads / warp | 32.0 | 32.0 | 32.0 | 32.0 |

## Ranked hypotheses for round 6

### 1. SageAttention2 is compute-limited with memory idle — 18.7% of the request

**DRAM at 0.9% and L2 at 37%** while SM throughput is 71.6%: this kernel is not
waiting on memory at all. It is pinned at **16.7% occupancy** (achieved equals
theoretical, so the cap is the launch configuration, not an eligibility problem)
with **68% of cycles having no eligible warp** and 0.44 eligible warps per
scheduler. That is a latency-hiding shortfall inside the compute pipe.

This matches the prior the brief flagged: the `sageattention` sm_120 build is an
sm_80/89/90-era kernel configuration (`<128, 64, 32, 64, 128, …>`), which on this
card has less shared memory per SM to work with and no TMA or wgmma to pipeline
with.

**Concrete change:** this is an external kernel, so the lever is configuration
rather than code — sweep the Sage variants already exposed through
`AttnQuantSpec` (`qk_quant_gran=per_thread`, `pv_accum_dtype=fp32+fp16` vs
`fp16`) measuring *occupancy and SM throughput*, not just wall time, and compare
against SageAttention3 on the same shape. **Upper bound if it reached 90% SM:
about 6 s a request.** Track S has Sage3 numbers at W1; those should be read
alongside this table rather than separately.

### 2. The hybrid GEMM is above the line; its remaining gap is L2, not the kernel

**75.0% SM with L2 at 81.8%** and DRAM at 8.9%. It is the largest single item
(37.7%) and the *least* promising: it is already above the brief's ~70% bar, and
independent of ncu it measures **84% of the 351 TF FP16-accumulate ceiling** on
this card. Occupancy 16.6% is by design for a 128×128×32 tile at 4 warps.

The honest read is that **L2 pressure, not the MMA, is what is left** — and the
operand traffic that creates it is the same traffic that made in-register casting
and Strassen's prologue sums lose. **No change recommended**; I would not spend
round 6 here.

### 3. Both standalone casts are already at 87% of DRAM — not optimisable, only removable

**87.3% and 87.2% DRAM throughput**, SM at 10.7% and 2.7%, occupancy 76–88%.
These are textbook memory-bound streaming kernels running near their bound. **A
better cast kernel cannot win anything**; only removing the traffic can.

And PR #71 established that the traffic cannot be removed by an exact change:
one producer is a graph input, the other is an extern attention output, and the
permuted view has no 2-D stride pair so the copy is what makes the operand
addressable. **This closes the lever for a second, independent reason** — even a
perfect 3-D-A kernel is competing against a copy already at 87% of peak
bandwidth, so the win is the 0.88 s of traffic, not a speedup.

## What this says about round 6 overall

Three kernels are 70% of the request; one of them (37.7%) is near its ceiling and
one (13.6%) is Track M's. **The only Track C target with real headroom is
SageAttention2's 18.7%**, and its problem is occupancy and warp eligibility on
sm_120 rather than arithmetic. If that kernel cannot be reconfigured, the DiT
block is close to done and the remaining levers are outside it.

## Reproduce

```
cd showcase/kandinsky6/compute

# ranked list + bound classification
/usr/local/cuda/bin/ncu --target-processes all --profile-from-start off \
  --clock-control none --section SpeedOfLight -c 60 \
  --export ncu-sol --force-overwrite python ncu_block.py

# the full section set, restricted to the kernels that matter
/usr/local/cuda/bin/ncu --target-processes all --profile-from-start off \
  --clock-control none \
  --section SpeedOfLight --section MemoryWorkloadAnalysis \
  --section ComputeWorkloadAnalysis --section Occupancy \
  --section WarpStateStats --section SchedulerStats --section LaunchStats \
  --kernel-name "regex:_hybrid_mm|qk_int_sv_f8_attn_kernel|cudnn_generated|s16816gemm|to_copy_t_view|to_copy_permute|to_copy_gelu|native_layer_norm|fused_rms_norm" \
  -c 40 --export ncu-full --force-overwrite python ncu_block.py

/usr/local/cuda/bin/ncu --import ncu-full.ncu-rep --csv --page details
```

**A note on `-c`:** the first run's 60-launch window caught only small early
kernels (6.7 ms total, nothing above 498 us) — the GEMM and attention launch
later in the forward. If you reuse this, filter with `--kernel-name` rather than
raising `-c`, which is also what keeps the replay cost down.
