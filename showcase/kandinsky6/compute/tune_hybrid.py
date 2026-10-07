# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""H2: per-shape configs for the hybrid GEMM, against the right baseline.

The baseline is **cuBLAS on BF16**, because that is what the served path runs.
Benchmarking against cuBLAS-on-FP16 would flatter the kernel: FP16 and BF16 move
the same bytes and issue the same MMA shape, but quoting the wrong one makes a
library-versus-library difference look like the kernel's doing.

Only the large-M visual shapes are tuned. At W1 the audio branch runs 218 rows
and the text tower 256, together 0.02 s of a 7.98 s/step census, and a Triton
kernel's launch and tail behaviour at M=218 is not where this lever lives --
those stay on cuBLAS.
"""

from __future__ import annotations

import argparse
import contextlib
import itertools
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gpulock import GpuLocks, foreign_gpu_procs  # noqa: E402

# The distinct large-M shapes the census lists, as (label, K, N). M is W1's
# 50,220 visual tokens. `visual.qkv` is three d->d calls and `cross.kv_from_visual`
# two d->d_a calls, so both are covered by the 4096->4096 row.
SHAPES = [
    ("d->d        (attn_out, text_cross x2, cross q/out, qkv x3)", 4096, 4096),
    ("d->ff       (ff1)", 4096, 16384),
    ("ff->d       (ff2)", 16384, 4096),
]

M = 50_220


def _median(run, warmup: int = 3, iters: int = 10) -> float:
    import torch

    for _ in range(warmup):
        run()
    torch.cuda.synchronize()
    s = []
    for _ in range(iters):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        run()
        e1.record()
        torch.cuda.synchronize()
        s.append(e0.elapsed_time(e1) / 1e3)
    return statistics.median(s)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--json", type=Path, default=None)
    p.add_argument("--no-locks", action="store_true")
    p.add_argument("--quick", action="store_true", help="a smaller grid, for a sanity pass")
    args = p.parse_args()

    import torch
    import triton

    from hybrid_gemm import _hybrid_mm

    bm_s = (128, 256) if args.quick else (64, 128, 256)
    bn_s = (128, 256) if args.quick else (64, 128, 256)
    bk_s = (32, 64)
    warp_s = (4, 8)
    stage_s = (3,) if args.quick else (2, 3, 4)

    report = {"M": M, "device": torch.cuda.get_device_name(0), "shapes": []}
    with contextlib.nullcontext() if args.no_locks else GpuLocks():
        foreign = foreign_gpu_procs()
        report["foreign_gpu_procs"] = [str(x) for x in foreign]
        if foreign:
            print(f"warning: foreign GPU process(es): {foreign}", file=sys.stderr)

        for label, K, N in SHAPES:
            flop = 2 * M * K * N
            xb = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
            wb = (torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02)
            # The served baseline: cuBLAS, BF16 operands, no bias (PR #38).
            base = _median(lambda: torch.nn.functional.linear(xb, wb))
            base_tf = flop / base / 1e12

            xh = xb.to(torch.float16)
            wh = wb.to(torch.float16).t()
            c = torch.empty((M, N), device="cuda", dtype=torch.bfloat16)
            bias = torch.randn(N, device="cuda", dtype=torch.float32)

            print(f"\n=== {label}  M={M} K={K} N={N}")
            print(f"    cuBLAS BF16 baseline: {base*1e6:9.0f} us  {base_tf:7.1f} TF")

            rows, best = [], None
            for BM, BN, BK, w, st in itertools.product(bm_s, bn_s, bk_s, warp_s, stage_s):
                def run(BM=BM, BN=BN, BK=BK, w=w, st=st):
                    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
                    _hybrid_mm[grid](xh, wh, bias, c, M, N, K,
                                     xh.stride(0), xh.stride(1), wh.stride(0), wh.stride(1),
                                     c.stride(0), c.stride(1),
                                     BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK, GROUP_M=8,
                                     HAS_BIAS=True, OUT_FP16=False,
                                     num_warps=w, num_stages=st)
                try:
                    t = _median(run)
                except Exception as exc:  # noqa: BLE001 - OOM on shared memory is expected and common
                    rows.append({"BM": BM, "BN": BN, "BK": BK, "warps": w, "stages": st,
                                 "skipped": type(exc).__name__})
                    continue
                tf = flop / t / 1e12
                row = {"BM": BM, "BN": BN, "BK": BK, "warps": w, "stages": st,
                       "seconds": t, "tflops": tf, "speedup_vs_cublas_bf16": base / t}
                rows.append(row)
                if best is None or tf > best["tflops"]:
                    best = row

            viable = [r for r in rows if "tflops" in r]
            viable.sort(key=lambda r: -r["tflops"])
            for r in viable[:5]:
                print(f"    {r['BM']:3d}x{r['BN']:3d}x{r['BK']:3d} w{r['warps']} s{r['stages']}: "
                      f"{r['seconds']*1e6:9.0f} us  {r['tflops']:7.1f} TF  "
                      f"{r['speedup_vs_cublas_bf16']:.2f}x vs cuBLAS BF16")
            print(f"    ({len(rows) - len(viable)} of {len(rows)} configs did not fit)")
            report["shapes"].append({"label": label, "K": K, "N": N,
                                     "cublas_bf16_seconds": base, "cublas_bf16_tflops": base_tf,
                                     "best": best, "rows": rows})

        print("\n=== per-shape best, against the served cuBLAS BF16 path ===")
        total_base = total_best = 0.0
        for s in report["shapes"]:
            b = s["best"]
            # Counting the GEMMs each row stands for at W1, per visual block.
            n_calls = {4096: 7, 16384: 1}[s["N"]] if s["K"] == 4096 else 1
            total_base += s["cublas_bf16_seconds"] * n_calls
            total_best += b["seconds"] * n_calls
            print(f"  {s['label']:58s} {b['BM']:3d}x{b['BN']:3d}x{b['BK']:3d} "
                  f"w{b['warps']} s{b['stages']}  {b['speedup_vs_cublas_bf16']:.2f}x")
        print(f"\n  weighted over one visual block's large GEMMs: "
              f"{total_base*60:.3f} -> {total_best*60:.3f} s/step  "
              f"({total_base/total_best:.2f}x, {(total_best-total_base)*60*10:+.1f} s a request)")
        report["weighted"] = {"base_step_s": total_base * 60, "hybrid_step_s": total_best * 60,
                              "speedup": total_base / total_best,
                              "request_delta_s": (total_best - total_base) * 60 * 10}

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + "\n")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
