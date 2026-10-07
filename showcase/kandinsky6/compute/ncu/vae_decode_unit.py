# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""One W1 video-VAE decode as served (eager fp16, served tile plan), for ncu. Warm decode, then a profiled decode."""

import json
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "bench"))
from gpu_guard import GpuGuard

from vllm_omni.diffusion.models.kandinsky6.modeling_kandinsky6_vae import AutoencoderKLHunyuanVideo

HUB = (
    Path(os.environ.get("HF_HOME", "/data/jooman/hf"))
    / "hub/models--kandinskylab--Kandinsky-6.0-Pro-distill-5s-Diffusers"
)
SNAP = sorted((HUB / "snapshots").iterdir())[-1]  # only vae/config.json is read
guard = GpuGuard()
guard.acquire(timeout_s=1800)
try:
    torch.set_grad_enabled(False)
    vae = (
        AutoencoderKLHunyuanVideo.from_config(json.loads((SNAP / "vae/config.json").read_text()))
        .to("cuda", torch.float16)
        .eval()
    )
    # Weights: values do not change conv/GroupNorm cost; random init is fine (scaled down to keep fp16 finite).
    for p in vae.parameters():
        p.mul_(0.5)
    plan = ((1, 17, 256, 448), (8, 224, 416))  # the served plan at W1 (server logs)
    vae.get_dec_optimal_tiling = lambda shape, device=None: plan
    z = (torch.randn(1, 16, 31, 60, 108, device="cuda") * 0.5).to(torch.float16)
    t = time.time()
    vae.decode(z)
    torch.accelerator.synchronize()
    print(f"warm decode {time.time() - t:.1f}s", flush=True)
    torch.cuda.profiler.start()
    t = time.time()
    out = vae.decode(z).sample
    torch.accelerator.synchronize()
    torch.cuda.profiler.stop()
    print(
        f"profiled decode {time.time() - t:.1f}s out {tuple(out.shape)} finite {bool(torch.isfinite(out).all())}",
        flush=True,
    )
finally:
    guard.release()
