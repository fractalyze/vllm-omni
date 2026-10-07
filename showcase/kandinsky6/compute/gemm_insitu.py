# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""What the GEMM operands actually look like inside a block, not in a benchmark.

`gemm_census.py` times every GEMM at its W1 shape and finds them at 98-99% of a
measured BF16 peak, yet the served GEMM bucket is larger than the census total.
Part of that was the bias (`k6c-k1b`); a remainder is unexplained, and the
remaining suspects are all properties of the *call* rather than the shape:

**Alignment and contiguity.** cuBLAS picks a kernel partly from the operand
pointers and leading dimensions. A view, a slice, or a leading dim that is not a
multiple of 16 bytes can drop it to a config with a smaller tile or a narrower
load -- exactly the `_align8` suffix visible in the kernel names.

**dtype.** A projection fed by an fp32 modulation or norm output pays a cast
first, and depending on where the cast lands a profiler may bill it to either
side.

**Who issues the GEMM.** Under `torch.compile`, Inductor either falls back to an
extern `addmm`/`mm` (cuBLAS, the same kernel as eager) or emits its own Triton
template. Which one it chose decides whether a GEMM finding is even actionable.

This registers a pre-hook on every parallel linear in one fused block at W1 and
records, for each, the operand metadata a kernel choice can depend on. It
asserts nothing: it is the record the next hypothesis gets tested against.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import block_profile as bp  # noqa: E402
from gpulock import GpuLocks, foreign_gpu_procs  # noqa: E402


def describe(t) -> dict:
    """The operand facts a cuBLAS kernel choice can depend on."""
    import torch

    if not isinstance(t, torch.Tensor):
        return {"not_a_tensor": type(t).__name__}
    itemsize = t.element_size()
    # Leading dimension in elements, and whether its byte length is a multiple
    # of 16 -- the alignment cuBLAS's wider vector loads require.
    lead = t.stride(-2) if t.dim() >= 2 else t.numel()
    return {
        "shape": list(t.shape),
        "stride": list(t.stride()),
        "dtype": str(t.dtype).replace("torch.", ""),
        "contiguous": bool(t.is_contiguous()),
        "leading_dim": int(lead),
        "leading_dim_bytes": int(lead * itemsize),
        "leading_dim_16B_aligned": (lead * itemsize) % 16 == 0,
        "ptr_256B_aligned": (t.data_ptr() % 256) == 0,
        "ptr_16B_aligned": (t.data_ptr() % 16) == 0,
        "is_view": t._base is not None,
        "mb": round(t.numel() * itemsize / 1e6, 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", choices=sorted(bp.CONFIGS), default="pro")
    parser.add_argument("--geometry", choices=sorted(bp.GEOMETRIES), default="w1")
    parser.add_argument("--compile", dest="compile_mode", default=None,
                        help="compile the block first, to see what Inductor does to these calls")
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--no-locks", action="store_true")
    args = parser.parse_args()

    import torch
    from vllm.model_executor.layers.linear import LinearBase

    cfg = bp.CONFIGS[args.config]
    shapes = bp.Shapes(**bp.GEOMETRIES[args.geometry])
    device, dtype = torch.device("cuda"), torch.bfloat16

    with contextlib.nullcontext() if args.no_locks else GpuLocks():
        foreign = foreign_gpu_procs()
        if foreign:
            print(f"warning: foreign GPU process(es): {foreign}", file=sys.stderr)

        with bp.single_process_parallel():
            module = bp.build_target("fused", cfg, None, device, dtype)
            inputs = bp.make_inputs("fused", cfg, shapes, device, dtype)

            seen: list[dict] = []

            def hook(name):
                def fn(mod, call_args):
                    x = call_args[0]
                    seen.append({
                        "layer": name,
                        "class": type(mod).__name__,
                        "bias": getattr(mod, "bias", None) is not None,
                        "skip_bias_add": bool(getattr(mod, "skip_bias_add", False)),
                        "x": describe(x),
                        "w": describe(mod.weight),
                    })
                return fn

            handles = [m.register_forward_pre_hook(hook(n))
                       for n, m in module.named_modules() if isinstance(m, LinearBase)]
            if args.compile_mode:
                module = bp.maybe_compile(module, args.compile_mode)
            with torch.no_grad():
                module(**inputs)
            for h in handles:
                h.remove()

    big = [r for r in seen if r["x"]["shape"][-2] > 1000]
    print(f"{len(seen)} linear calls in one fused block; {len(big)} at large M\n")
    hdr = f"{'layer':46s} {'M':>6s} {'K':>6s} {'N':>6s} {'x dtype':>8s} {'x ctg':>6s} {'view':>5s} {'ld16B':>6s} {'bias':>5s}"
    print(hdr)
    for r in sorted(big, key=lambda r: -r["x"]["shape"][-2] * r["x"]["shape"][-1]):
        x, w = r["x"], r["w"]
        print(f"{r['layer']:46s} {x['shape'][-2]:6d} {x['shape'][-1]:6d} {w['shape'][0]:6d} "
              f"{x['dtype']:>8s} {str(x['contiguous']):>6s} {str(x['is_view']):>5s} "
              f"{str(x['leading_dim_16B_aligned']):>6s} {str(r['bias']):>5s}")

    odd = [r for r in big if not r["x"]["contiguous"] or not r["x"]["leading_dim_16B_aligned"]
           or r["x"]["dtype"] != "bfloat16"]
    print(f"\nlarge-M calls with a non-contiguous, misaligned or non-BF16 input: {len(odd)}")
    for r in odd:
        print(f"  {r['layer']}: {r['x']}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(
            {"compile_mode": args.compile_mode, "foreign_gpu_procs": [str(p) for p in foreign],
             "calls": seen}, indent=2) + "\n")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
