# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Why the profile's GEMM bucket is 45% larger than the GEMMs.

`gemm_census.py` puts every GEMM one block issues at 6.50 s/step; the profile
attributes 9.40 s to BF16 GEMMs. The kernels are at 98-99% of peak, so the extra
2.90 s is not `mma` and not something a better kernel recovers.

This measures the three things that stand between a matmul benchmark and a
matmul in this model, each at W1's real shapes:

**fp32 operands.** `apply_scale_shift_norm`, `apply_gate_sum` and `apply_rotary`
upcast the residual stream to fp32. At 50,220 x 4096 that is an 823 MB tensor,
and a GEMM fed from it pays a cast back to BF16 first.

**Strided operands.** A stream that has been chunked, transposed or sliced
arrives as a view. cuBLAS wants a contiguous operand and will materialise one.

**The epilogue itself.** `gate * out + residual` over the same 823 MB, counted
in whichever bucket the profiler decides it is nearest to.

Each is reported as the cost *added* to the same GEMM, so the three can be
compared against the 2.90 s the profile is missing.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gemm_census import W1_VISUAL_TOKENS, VISUAL_BLOCKS, time_matmul  # noqa: E402
from gpulock import GpuLocks  # noqa: E402

SHAPES = [
    ("visual.ff1", 4096, 16384),
    ("visual.ff2", 16384, 4096),
    ("visual.qkv", 4096, 12288),
    ("visual.attn_out", 4096, 4096),
]


def timed(fn, *, warmup: int = 5, iters: int = 20) -> float:
    import torch

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) / 1e3)
    return statistics.median(samples)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--no-locks", action="store_true")
    args = parser.parse_args()

    import torch

    M = W1_VISUAL_TOKENS
    rows = []
    with contextlib.nullcontext() if args.no_locks else GpuLocks():
        for name, K, N in SHAPES:
            b = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
            a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
            a32 = torch.randn(M, K, device="cuda", dtype=torch.float32)
            # A view the way a chunked/transposed stream arrives: same values,
            # non-contiguous in the reduction dimension.
            a_strided = torch.randn(K, M, device="cuda", dtype=torch.bfloat16).t()
            residual = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
            gate = torch.randn(1, N, device="cuda", dtype=torch.bfloat16)

            base = time_matmul(a, b)
            from_fp32 = timed(lambda: a32.to(torch.bfloat16) @ b)
            strided = timed(lambda: a_strided @ b)
            with_epilogue = timed(lambda: gate * (a @ b) + residual)

            row = {
                "name": name, "M": M, "K": K, "N": N,
                "base_us": base * 1e6,
                "from_fp32_us": from_fp32 * 1e6,
                "strided_us": strided * 1e6,
                "with_epilogue_us": with_epilogue * 1e6,
                "cast_overhead_pct": (from_fp32 / base - 1) * 100,
                "stride_overhead_pct": (strided / base - 1) * 100,
                "epilogue_overhead_pct": (with_epilogue / base - 1) * 100,
                "epilogue_step_s": (with_epilogue - base) * VISUAL_BLOCKS,
                "cast_step_s": (from_fp32 - base) * VISUAL_BLOCKS,
            }
            rows.append(row)
            print(f"{name:18s} base {base*1e6:8.0f} us | fp32-in {from_fp32*1e6:8.0f} "
                  f"({row['cast_overhead_pct']:+5.1f}%) | strided {strided*1e6:8.0f} "
                  f"({row['stride_overhead_pct']:+5.1f}%) | +epilogue {with_epilogue*1e6:8.0f} "
                  f"({row['epilogue_overhead_pct']:+5.1f}%)")
            del a, b, a32, a_strided, residual, gate
            torch.cuda.empty_cache()

    cast = sum(r["cast_step_s"] for r in rows)
    epi = sum(r["epilogue_step_s"] for r in rows)
    print(f"\nover {VISUAL_BLOCKS} blocks: fp32->bf16 casts add {cast:.3f} s/step, "
          f"epilogues add {epi:.3f} s/step")
    print(f"the profile's unexplained GEMM time is 2.90 s/step")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({"rows": rows, "cast_step_s": cast, "epilogue_step_s": epi}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
