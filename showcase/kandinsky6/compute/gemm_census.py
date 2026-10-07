# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Every GEMM in one Kandinsky 6 block at W1, against a measured BF16 peak.

The denoise loop spends 54% of its time in BF16 GEMMs (9.40 s of a 17.4 s step),
and they run on CUTLASS `s16816gemm` -- an Ampere-style `mma.sync.m16n8k16`
kernel family -- on sm_120. Before writing a kernel by hand, two numbers are
needed and neither is in the profile:

**What the hardware actually does.** A BF16 peak quoted from a spec sheet is not
a target; this measures one, at the clocks and power limit the GPU runs these
shapes at, and records both so a later run can tell a kernel regression from a
thermal one.

**What each shape achieves against it.** 1213 TFLOP a step is spread over four
distinct shapes with very different aspect ratios, and a kernel family can be at
peak on one and half-rate on another. The gap, per shape, weighted by how much
of the step that shape owns, is the only thing that says which GEMM is worth
rewriting -- or whether a library already has a better kernel and nothing needs
writing at all.

That last question comes first, because it is the cheapest: on this GPU
cuBLASLt's nvjet kernels beat CUTLASS's OpClassTensorOp at every FP8 tile
(vault: `rtx5090-sm120-fp8-mma-block-scaled-full-rate`), and Track M measured
cuBLASLt 1.42-1.50x over CUTLASS for FP8 on these very shapes. Whether BF16 has
the same story is a flag, not a kernel.

    python gemm_census.py --json census.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gpulock import GpuLocks, foreign_gpu_procs  # noqa: E402

# W1: 31 x 30 x 54 = 50,220 visual tokens through 60 blocks. The audio branch
# carries 218 latent frames at d 2048, and the text towers 256 padded tokens.
W1_VISUAL_TOKENS = 50_220
W1_AUDIO_TOKENS = 218
W1_TEXT_TOKENS = 256
VISUAL_BLOCKS = 60


def block_gemms(d: int = 4096, ff: int = 16384, d_a: int = 2048, ff_a: int = 7168) -> list[dict]:
    """Every GEMM one fused block issues, with how many of them a step runs.

    Shapes come from the checkpoint's own `transformer/config.json` (model_dim
    4096, ff_dim 16384, model_dim_a 2048, ff_dim_a 7168) rather than from the
    code, so a config change is visible here instead of silently changing what
    is being benchmarked.

    QKV is listed fused although the port issues it as **three** separate
    ``ColumnParallelLinear``s of d -> d (it deliberately avoids
    ``QKVParallelLinear``; see the class docstring). The two forms are within
    0.2% at W1 -- 3 x 7759 us against 23230 us for the fused shape -- so the
    fused row is kept for brevity and this note exists so nobody reads the row
    as a claim about the port.

    ``bias`` is the port's own setting, not a guess: every attention projection
    is ``bias=True`` (``Kandinsky6Attention``) and every FFN is ``bias=False``
    (``Kandinsky6FeedForward``). It is carried here because on this GPU the bias
    is not free -- see ``--bias`` -- so a census that benchmarked bare ``a @ b``
    everywhere would under-report the attention projections and over-report how
    close the served model runs to peak.
    """
    M, Ma, Mt = W1_VISUAL_TOKENS, W1_AUDIO_TOKENS, W1_TEXT_TOKENS
    return [
        # name,                    M,  K,     N,         per step
        dict(name="visual.qkv", m=M, k=d, n=3 * d, count=VISUAL_BLOCKS, bias=True, branch="visual"),
        dict(name="visual.attn_out", m=M, k=d, n=d, count=VISUAL_BLOCKS, bias=True, branch="visual"),
        # The decoder block also runs a *text* cross-attention (visual queries
        # against the 256 text tokens), whose query and output projections are
        # both full-size at M=50,220. Found by enumerating `named_modules()`
        # rather than reading the forward -- `gemm_insitu.py` lists 12 large-M
        # linears where this census first listed 7. Missing these two plus
        # `cross.out_from_visual` was 1.40 s/step, and was the whole of what
        # looked like an unexplained in-situ GEMM gap.
        dict(name="visual.text_cross_q", m=M, k=d, n=d, count=VISUAL_BLOCKS, bias=True, branch="visual"),
        dict(name="visual.text_cross_out", m=M, k=d, n=d, count=VISUAL_BLOCKS, bias=True, branch="visual"),
        dict(name="visual.ff1", m=M, k=d, n=ff, count=VISUAL_BLOCKS, bias=False, branch="visual"),
        dict(name="visual.ff2", m=M, k=ff, n=d, count=VISUAL_BLOCKS, bias=False, branch="visual"),
        # Cross-attention: the query comes from the visual stream, the key and
        # value from the audio stream, so the two sides have wildly different M.
        dict(name="cross.q_from_visual", m=M, k=d, n=d, count=VISUAL_BLOCKS, bias=True, branch="cross"),
        dict(name="cross.kv_from_audio", m=Ma, k=d_a, n=2 * d, count=VISUAL_BLOCKS, bias=True, branch="cross"),
        dict(name="cross.q_from_audio", m=Ma, k=d_a, n=d_a, count=VISUAL_BLOCKS, bias=True, branch="cross"),
        dict(name="cross.kv_from_visual", m=M, k=d, n=2 * d_a, count=VISUAL_BLOCKS, bias=True, branch="cross"),
        # Each cross-attention also has an output projection, on the side its
        # *query* came from, so `va_cross_attention.out_layer` is another
        # full-size d -> d GEMM at M=50,220. Leaving it out understated this
        # census by 0.47 s/step -- a third of the gap the tool was built to
        # explain. A census that misses a GEMM makes the unexplained remainder
        # look bigger than it is, which is the failure mode to guard here.
        dict(name="cross.out_from_visual", m=M, k=d, n=d, count=VISUAL_BLOCKS, bias=True, branch="cross"),
        dict(name="cross.out_from_audio", m=Ma, k=d_a, n=d_a, count=VISUAL_BLOCKS, bias=True, branch="cross"),
        # The audio branch is the same structure at 218 rows: a shape where a
        # kernel tuned for 50,220 rows can be far off peak and it may not matter.
        dict(name="audio.qkv", m=Ma, k=d_a, n=3 * d_a, count=VISUAL_BLOCKS, bias=True, branch="audio"),
        dict(name="audio.attn_out", m=Ma, k=d_a, n=d_a, count=VISUAL_BLOCKS, bias=True, branch="audio"),
        dict(name="audio.ff1", m=Ma, k=d_a, n=ff_a, count=VISUAL_BLOCKS, bias=False, branch="audio"),
        dict(name="audio.ff2", m=Ma, k=ff_a, n=d_a, count=VISUAL_BLOCKS, bias=False, branch="audio"),
        # Text towers run four blocks, not sixty.
        dict(name="text.qkv", m=Mt, k=d, n=3 * d, count=4, bias=True, branch="text"),
        dict(name="text.ff1", m=Mt, k=d, n=ff, count=4, bias=False, branch="text"),
        # The decoder block's text cross-attention reads the 256 text tokens on
        # its key/value side, so these run 60 times at M=256 rather than 4 times.
        # Listed for completeness rather than for the 0.001 s/step: this census is
        # tested against the block's module tree, and a row missing here is the
        # error that produced a phantom 1.4 s/step gap once already.
        dict(name="text_cross.kv", m=Mt, k=d, n=2 * d, count=VISUAL_BLOCKS, bias=True, branch="text"),
    ]


def time_linear(a, w, bias, *, warmup: int = 5, iters: int = 20) -> float:
    """Median seconds for one `F.linear(a, w, bias)` -- the call the port makes.

    `w` is in `nn.Linear`'s own `(out, in)` layout, so this is the TN GEMM the
    served model issues rather than the NN one `a @ b` issues. Layout turns out
    not to matter on this GPU (+-0.3%); passing `bias` does.
    """
    import torch

    return _median(lambda: torch.nn.functional.linear(a, w, bias), warmup, iters)


def _median(call, warmup: int, iters: int) -> float:
    import statistics

    import torch

    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        call()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) / 1e3)
    return statistics.median(samples)


def time_matmul(a, b, *, warmup: int = 5, iters: int = 20) -> float:
    """Median seconds for one `a @ b`, on CUDA events.

    Events rather than wall time: the launch is asynchronous, and a wall clock
    around it measures the Python, not the kernel. Median rather than mean
    because a single preempted iteration should not move the number.
    """
    import torch

    for _ in range(warmup):
        a @ b
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        a @ b
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) / 1e3)
    return statistics.median(samples)


def board() -> dict:
    """Clocks and power *right now*."""
    import subprocess

    query = "clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu"
    out = subprocess.run(
        ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=False,
    ).stdout.strip().split(", ")
    keys = ["sm_mhz", "mem_mhz", "power_w", "temp_c", "util_pct"]
    return {k: float(v) for k, v in zip(keys, out) if v.replace(".", "", 1).isdigit()}


def board_under_load(a, b, seconds: float = 2.0) -> dict:
    """Clocks and power sampled *while the GEMM is running*.

    Reading them after `torch.cuda.synchronize()` returns measures an idle GPU:
    this card drops to ~1100 MHz and ~19 W within the time it takes to launch a
    subprocess, so a sample taken after the benchmark reports conditions no
    kernel ever ran under. The first version of this file did exactly that and
    recorded a 230 TFLOP/s peak "at 1087 MHz, 18.8 W", which is not a thing that
    happened.

    This keeps the GPU busy on the same shape and samples in the middle.
    """
    import torch

    deadline = time.perf_counter() + seconds
    while time.perf_counter() < deadline:
        for _ in range(10):
            a @ b
        torch.cuda.synchronize() if False else None  # keep the queue full, do not drain it
        if time.perf_counter() > deadline - seconds / 2:
            sample = board()
    torch.cuda.synchronize()
    return sample


def measure_peak(dtype, sizes=(4096, 8192, 12288)) -> dict:
    """The best BF16 rate this GPU sustains on a square GEMM, with its conditions.

    Square and large: the shape most likely to reach the hardware's rate, so the
    number is an upper bound the real shapes are measured against rather than a
    claim about any particular kernel.
    """
    import torch

    best = {"tflops": 0.0}
    for n in sizes:
        a = torch.randn(n, n, device="cuda", dtype=dtype)
        b = torch.randn(n, n, device="cuda", dtype=dtype)
        seconds = time_matmul(a, b)
        tflops = 2 * n**3 / seconds / 1e12
        if tflops > best["tflops"]:
            best = {"tflops": tflops, "n": n, "seconds": seconds, "board": board_under_load(a, b)}
        del a, b
        torch.cuda.empty_cache()
    return best


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--no-locks", action="store_true")
    parser.add_argument("--backends", default="default,cublaslt", help="comma-separated BLAS backends to compare")
    args = parser.parse_args()

    import torch

    import contextlib
    with contextlib.nullcontext() if args.no_locks else GpuLocks():
        foreign = foreign_gpu_procs()
        if foreign:
            print(f"warning: foreign GPU process(es): {foreign}", file=sys.stderr)

        dtype = torch.bfloat16
        report = {
            "device": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
            "torch": torch.__version__,
            "foreign_gpu_procs": [str(p) for p in foreign],
            "backends": {},
        }

        for backend in args.backends.split(","):
            backend = backend.strip()
            if backend != "default":
                try:
                    torch.backends.cuda.preferred_blas_library(backend)
                except Exception as exc:  # noqa: BLE001
                    print(f"{backend}: unavailable ({exc})", file=sys.stderr)
                    continue
            peak = measure_peak(dtype)
            print(f"\n=== {backend}: BF16 peak {peak['tflops']:.1f} TFLOP/s "
                  f"at {peak['n']}^3, sm {peak['board'].get('sm_mhz')} MHz, {peak['board'].get('power_w')} W")
            rows = []
            total_s = 0.0
            bias_penalty_s = 0.0
            for gemm in block_gemms():
                a = torch.randn(gemm["m"], gemm["k"], device="cuda", dtype=dtype)
                b = torch.randn(gemm["k"], gemm["n"], device="cuda", dtype=dtype)
                seconds = time_matmul(a, b)
                flop = 2 * gemm["m"] * gemm["k"] * gemm["n"]
                step_s = seconds * gemm["count"]
                total_s += step_s
                row = {**gemm, "seconds": seconds, "tflops": flop / seconds / 1e12,
                       "pct_peak": flop / seconds / 1e12 / peak["tflops"] * 100,
                       "step_seconds": step_s}
                # The served GEMM is `F.linear`, not `a @ b`. Where the port
                # passes a bias, time that too: the bias is N elements against
                # an M*N output, so it ought to be free, and on sm_120 it is
                # not. Without this column the census would claim the model
                # runs at 98% of peak when the attention projections do not.
                if gemm["bias"]:
                    w = b.t().contiguous()
                    bias = torch.randn(gemm["n"], device="cuda", dtype=dtype)
                    nobias_s = time_linear(a, w, None)
                    withbias_s = time_linear(a, w, bias)
                    row["linear_seconds"] = nobias_s
                    row["linear_bias_seconds"] = withbias_s
                    row["bias_penalty_pct"] = (withbias_s / nobias_s - 1) * 100
                    row["bias_penalty_step_s"] = (withbias_s - nobias_s) * gemm["count"]
                    bias_penalty_s += row["bias_penalty_step_s"]
                    del w, bias
                rows.append(row)
                del a, b
                torch.cuda.empty_cache()

            rows.sort(key=lambda r: -r["step_seconds"])
            print(f"{'gemm':24s} {'M':>6s} {'K':>6s} {'N':>6s} {'us':>8s} {'TFLOP/s':>8s} {'%peak':>6s} {'s/step':>7s}")
            for r in rows:
                print(f"{r['name']:24s} {r['m']:6d} {r['k']:6d} {r['n']:6d} {r['seconds']*1e6:8.0f} "
                      f"{r['tflops']:8.1f} {r['pct_peak']:5.0f}% {r['step_seconds']:7.3f}")
            print(f"{'TOTAL':24s} {'':6s} {'':6s} {'':6s} {'':8s} {'':8s} {'':6s} {total_s:7.3f}")
            biased = [r for r in rows if r.get("bias_penalty_pct") is not None]
            if biased:
                print(f"\nthe bias, on the GEMMs the port gives one "
                      f"(F.linear(x, W) vs F.linear(x, W, bias)):")
                print(f"{'gemm':24s} {'no bias':>9s} {'+bias':>9s} {'penalty':>8s} {'s/step':>8s}")
                for r in sorted(biased, key=lambda r: -r["bias_penalty_step_s"]):
                    print(f"{r['name']:24s} {r['linear_seconds']*1e6:8.0f}us "
                          f"{r['linear_bias_seconds']*1e6:8.0f}us {r['bias_penalty_pct']:+7.1f}% "
                          f"{r['bias_penalty_step_s']:+8.3f}")
                print(f"{'TOTAL bias penalty':24s} {'':9s} {'':9s} {'':8s} {bias_penalty_s:+8.3f}")
            report["backends"][backend] = {"peak": peak, "gemms": rows, "step_seconds": total_s,
                                           "bias_penalty_step_seconds": bias_penalty_s}

        if len(report["backends"]) > 1:
            names = list(report["backends"])
            a_s = report["backends"][names[0]]["step_seconds"]
            b_s = report["backends"][names[1]]["step_seconds"]
            print(f"\n{names[1]} vs {names[0]}: {a_s:.3f} -> {b_s:.3f} s/step  ({(a_s/b_s - 1)*100:+.1f}%)")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + "\n")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
