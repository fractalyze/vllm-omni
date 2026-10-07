# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""cuDNN SDPA (the exact-attention path) at W1's visual self-attention shape, for ncu."""

import sys
from pathlib import Path

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "bench"))
from gpu_guard import GpuGuard

guard = GpuGuard()
guard.acquire(timeout_s=1800)
try:
    q, k, v = (torch.randn(1, 32, 50220, 128, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    with sdpa_kernel([SDPBackend.CUDNN_ATTENTION]), torch.no_grad():
        for _ in range(2):
            torch.nn.functional.scaled_dot_product_attention(q, k, v)
        torch.accelerator.synchronize()
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        for _ in range(3):
            torch.nn.functional.scaled_dot_product_attention(q, k, v)
        e.record()
        torch.accelerator.synchronize()
        ms = s.elapsed_time(e) / 3
        print(
            f"sdpa {ms:.1f} ms, {4 * 50220 * 50220 * 128 * 32 / ms / 1e9:.0f} TFLOP/s (non-causal 4*S^2*D*H)",
            flush=True,
        )
        torch.cuda.profiler.start()
        torch.nn.functional.scaled_dot_product_attention(q, k, v)
        torch.accelerator.synchronize()
        torch.cuda.profiler.stop()
finally:
    guard.release()
