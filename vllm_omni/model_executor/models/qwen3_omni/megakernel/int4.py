# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# Copyright 2026 Fractalyze Inc. All rights reserved.
"""Symmetric int4 weights in groups of 32, as Qwen3-Omni's compressed-tensors
W4A16 checkpoint stores them and vLLM loads them, as the int4 cores
read them (csrc/int4_*_core.cuh).

A weight row of k values is `packed` int32 [k / 8], value j in bits
4 (j % 8) of word j / 8 as q + 8, with q in [-8, 7], and `scales` bf16
[k / 32]: w = q × scale over the group of 32 a scale covers.
"""

from __future__ import annotations

import torch

GROUP = 32


def quantize(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(packed, scales) for a float [n, k] weight: each group's scale maps its
    largest magnitude to 7."""
    n, k = w.shape
    groups = w.float().view(n, k // GROUP, GROUP)
    scales = (groups.abs().amax(dim=-1, keepdim=True) / 7).clamp(min=1e-8)
    scales = scales.bfloat16()
    q = (groups / scales.float()).round().clamp(-8, 7).to(torch.int64) + 8
    shifts = torch.arange(0, 32, 4, device=w.device, dtype=torch.int64)
    words = (q.view(n, k // 8, 8) << shifts).sum(dim=-1)
    # The same 32 bits as a signed int32, as the checkpoint stores them.
    packed = torch.where(words >= 2**31, words - 2**32, words).to(torch.int32)
    return packed.contiguous(), scales.view(n, k // GROUP).contiguous()


def dequantize(packed: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """The fp32 [n, k] weight (packed, scales) stand for."""
    n = packed.shape[0]
    shifts = torch.arange(0, 32, 4, device=packed.device, dtype=torch.int32)
    q = ((packed.view(n, -1, 1) >> shifts) & 0xF) - 8
    q = q.view(n, -1, GROUP).float()
    return (q * scales.float().view(n, -1, 1)).view(n, -1)
