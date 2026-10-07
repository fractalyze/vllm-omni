# Nsight Compute roofline: the lossy H5 fast mode (Track S, build-server)

2026-10-07, RTX 5090 (sm_120). Measurement only.

ncu 2026.2.0, `--clock-control none` (real 575 W clocks), sections SpeedOfLight, MemoryWorkloadAnalysis,
ComputeWorkloadAnalysis, SchedulerStats, WarpStateStats, Occupancy, LaunchStats plus
`--metrics sm__pipe_tensor_cycles_active...,dram__throughput...,l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum`;
`--profile-from-start off` around one unit (compute/ncu_h5_units.py gemm|attn). All bs1 locks held; no other GPU
process before or after. (A first attempt overlapped a film-render server shutting down and was discarded.)
Units at W1 (M = 50,220 visual tokens), unprofiled CUDA-event time: q/k/v/out-shaped linear 2.54 ms, FF1 8.50 ms,
FF2 8.46 ms, one served Sage3 call 56.4 ms. Request share assumes H5-INT8 (104.5 s): 7 NVFP4 DiT calls/request
(steps 3-10, step 8 cached) x 60 blocks; per block ~9 q/k/v/out-equivalent linears on video tokens + FF1 + FF2 + 1 Sage3.

| # | kernel (per block) | time/unit | s/request (share) | SOL SM / mem / DRAM % | tensor % | class | round-6 change |
|---|---|---:|---:|---|---:|---|---|
| 1 | Sage3 `compute_attn_ws` | 41.6 ms | 17.5 s (16.7%) | 57 / 46 / 5 | 58 | compute-bound BELOW peak: 1 persistent CTA/SM (170 x 384, occ 21%), low stalls -> issue/pipeline-bound | tune the FP4 attention tile / pipelining for sm_120 (the kernel is the Blackwell-datacenter design); 58% -> 80% would save ~5 s |
| 2 | Sage3 pre/post (4 transposes ~525 us each, K group_mean 516 us, Kernel2 2.2 ms, 2 x 1.28 ms quant/elementwise, 3 x scaled_fp4_quant ~295 us, ~12 elementwise) | 14.8 ms | 6.2 s (6%) | each 80-93% DRAM | 0 | memory-bound at DRAM peak, but avoidable traffic | call the kernel in the model's (B,S,H,D) layout (drop 4 transposes, ~0.9 s) and fuse smoothing + quantization into one pass (~4-5 s) |
| 3 | NVFP4 GEMM (CUTLASS `device_kernel`): q/k/v/out 1.23 ms, FF1 4.73 ms (1.43 PF), FF2 ~4.7 ms | ~20.5 ms | 8.6 s (8.2%) | 87 / 78 / 22 | 88 | compute-bound AT peak | none at the kernel; fewer GEMMs (cache) only |
| 4 | bias add after the GEMM (`elementwise_kernel`): 527 us (N 4096), 2.13 ms (N 16384) | ~7.4 ms | 3.1 s (3.0%) | 33 / 87 / 87 | 0 | memory-bound at DRAM peak, avoidable | fuse the bias into the FP4 GEMM epilogue (CUTLASS EVT) or into the next elementwise op: ~3 s |
| 5 | activation scale + quant: `aminmax` reduce 266 us (K 4096) + `cvt_fp16_to_fp4` 300 us, per linear | ~6.2 ms | 2.6 s (2.5%) | 8-15 / 89-92 / 89-92 | 0 | memory-bound at DRAM peak, redundant | quantize x once for q, k and v (same input: saves 2/3 of those); fold amax into the quant kernel or use a calibrated static scale: ~1.5-2 s |
| 6 | INT8 row dequant of the weight (3 elementwise kernels: 153 us at 4096^2, 755 us at FF size) | ~2.9 ms | 1.2 s (1.2%) | 3-37 / 53-95 | 0 | memory-bound | quantize to FP4 straight from INT8 in one kernel, or keep FP4 weights in pinned RAM: ~1 s |
| 7 | weight scale + quant: `aminmax` 25-108 us + `cvt` 22-84 us | ~1.0 ms | 0.4 s | ~76-95 DRAM | 0 | memory-bound | precompute per weight once per request (weights do not change across steps) |
| 8 | ~20 one-CTA scalar kernels per linear (global-scale arithmetic, ~2 us each, 6-16% occupancy) | ~0.5 ms + launch gaps | ~0.3-0.5 s | ~0 | 0 | latency-bound | compute the scale inside the quant kernel |
Not covered: steps 1-2 (BF16 GEMMs + exact cuDNN attention, Track C/M), VAE decode (~19 s, Track M), the INT8 stream.
Ranked hypotheses: (1) fuse Sage3 preprocessing + native layout, ~5-6 s; (2) Sage3 main kernel tuning toward 80% tensor, ~5 s;
(3) fused bias epilogue, ~3 s; (4) shared/fused activation quant, ~2 s; (5) INT8->FP4 direct, ~1 s. Together ~15 s of 104.5 s.
Reports: bs1 /home/jooman/k6/runs/ncu-h5/{gemm,attn}.ncu-rep; full per-launch table runs/ncu-h5/table.md.

## Reproduce

```bash
# All GPU locks held (compute/gpulock.py); nothing else on the GPU.
NCU=/usr/local/cuda/bin/ncu   # 2026.2.0
SECT="--section SpeedOfLight --section MemoryWorkloadAnalysis --section ComputeWorkloadAnalysis \
      --section SchedulerStats --section WarpStateStats --section Occupancy --section LaunchStats"
MET="--metrics sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,\
sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_active,sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum"
python compute/ncu_h5_units.py gemm --time; python compute/ncu_h5_units.py attn --time
$NCU --target-processes all --profile-from-start off --clock-control none $SECT $MET -c 60 \
     --export gemm --force-overwrite python compute/ncu_h5_units.py gemm
$NCU --target-processes all --profile-from-start off --clock-control none $SECT $MET -c 20 \
     --export attn --force-overwrite python compute/ncu_h5_units.py attn
python compute/ncu_table.py gemm.ncu-rep attn.ncu-rep
```

The 60-launch cap ends inside the FF2 unit, so FF2's GEMM is not in the export; its FLOPs equal FF1's and its
unprofiled unit time (8.46 ms) matches FF1's (8.50 ms).
