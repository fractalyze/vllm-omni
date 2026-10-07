# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""One unit of each H5 fast-mode kernel path at W1's shapes, for an Nsight Compute profile.

``gemm``: one NVFP4 step linear per W1 visual-block shape, exactly as the H5-INT8
mode runs it (``StepFp8LinearMethod`` with ``nvfp4`` on an INT8 weight-only
layer): INT8 row dequantize -> two aminmax global scales -> two
``scaled_fp4_quant`` -> ``cutlass_scaled_fp4_mm`` -> bias add.
``attn``: one served SageAttention3 call (the backend's four transposes and
``sageattn3_blackwell``) at the visual self-attention shape.

The script warms every kernel up, then wraps exactly one unit in
``torch.cuda.profiler.start()/stop()``; run it under
``ncu --profile-from-start off``. Without ncu it prints CUDA-event times per unit.

    python ncu_h5_units.py gemm|attn [--time]
"""

from __future__ import annotations

import argparse

import torch

from vllm_omni.diffusion.models.kandinsky6.step_precision import nvfp4_linear
from vllm_omni.quantization.int8_config import dequantize_int8_rows

M = 50220
# (name, K, N): the visual block's linears; q, k, v and out are separate 4096x4096 layers.
GEMMS = (("qkvo", 4096, 4096), ("ff1", 4096, 16384), ("ff2", 16384, 4096))
HEADS, HEAD_DIM = 32, 128


def gemm_units():
    units = []
    for name, k, n in GEMMS:
        x = torch.randn(M, k, device="cuda", dtype=torch.bfloat16)
        w8 = torch.randint(-127, 128, (n, k), device="cuda", dtype=torch.int8)
        scale = torch.rand(n, 1, device="cuda", dtype=torch.float32) * 1e-3
        bias = torch.randn(n, device="cuda", dtype=torch.bfloat16)

        def unit(x=x, w8=w8, scale=scale, bias=bias):
            return nvfp4_linear(x, dequantize_int8_rows(w8, scale, x.dtype), bias)

        units.append((name, unit))
    return units


def attn_units():
    from sageattn3 import sageattn3_blackwell

    # The backend receives (B, S, H, D) and transposes to (B, H, S, D) and back.
    q, k, v = (torch.randn(1, M, HEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16) for _ in range(3))

    def unit():
        qt, kt, vt = (t.transpose(1, 2).contiguous() for t in (q, k, v))
        return sageattn3_blackwell(qt, kt, vt, is_causal=False).transpose(1, 2).contiguous()

    return [("sage3", unit)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("which", choices=("gemm", "attn"))
    parser.add_argument("--time", action="store_true", help="CUDA-event time per unit instead of a profile region")
    args = parser.parse_args()
    units = gemm_units() if args.which == "gemm" else attn_units()
    with torch.no_grad():
        for _, unit in units:
            for _ in range(3):
                unit()
        torch.cuda.synchronize()
        if args.time:
            for name, unit in units:
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(10):
                    unit()
                end.record()
                torch.cuda.synchronize()
                print(f"{name}: {start.elapsed_time(end) / 10:.3f} ms per unit", flush=True)
            return
        torch.cuda.profiler.start()
        for _, unit in units:
            unit()
        torch.cuda.synchronize()
        torch.cuda.profiler.stop()


if __name__ == "__main__":
    main()
