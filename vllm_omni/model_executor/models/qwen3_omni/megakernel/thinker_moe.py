# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# Copyright 2026 Fractalyze Inc. All rights reserved.
"""Qwen3-Omni's thinker MoE block for one token in one launch
(csrc/thinker_moe.cu), and the PyTorch model it is held to.

Experts are compressed-tensors W4A16 (int4.py); the norm and router are
bf16, as vLLM loads them.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from vllm_omni.model_executor.models.qwen3_omni.megakernel import _ext, int4
from vllm_omni.model_executor.models.qwen3_omni.megakernel.barrier import (
    DEFAULT_TIMEOUT_NS,
    ErrorRecord,
    num_ctas,
    sync_words,
)

DIM, EXPERTS, TOP_K, EXPERT_FFN = 2048, 128, 8, 768


@dataclass(frozen=True)
class MoeWeights:
    norm: torch.Tensor  # bf16 [DIM]: post_attention_layernorm
    router: torch.Tensor  # bf16 [EXPERTS, DIM]
    w13_packed: torch.Tensor  # int32 [EXPERTS, 2 × EXPERT_FFN, DIM / 8]: gate, then up
    w13_scales: torch.Tensor  # bf16 [EXPERTS, 2 × EXPERT_FFN, DIM / 32]
    w2_packed: torch.Tensor  # int32 [EXPERTS, DIM, EXPERT_FFN / 8]
    w2_scales: torch.Tensor  # bf16 [EXPERTS, DIM, EXPERT_FFN / 32]
    eps: float = 1e-6

    @classmethod
    def random(cls, seed: int) -> MoeWeights:
        gen = torch.Generator(device="cuda").manual_seed(seed)

        def normal(*shape):
            return torch.randn(*shape, generator=gen, device="cuda")

        w13 = [int4.quantize(normal(2 * EXPERT_FFN, DIM) / DIM**0.5) for _ in range(EXPERTS)]
        w2 = [int4.quantize(normal(DIM, EXPERT_FFN) / EXPERT_FFN**0.5) for _ in range(EXPERTS)]
        return cls(
            norm=(1 + 0.1 * normal(DIM)).bfloat16(),
            router=(normal(EXPERTS, DIM) / DIM**0.5).bfloat16(),
            w13_packed=torch.stack([p for p, _ in w13]),
            w13_scales=torch.stack([s for _, s in w13]),
            w2_packed=torch.stack([p for p, _ in w2]),
            w2_scales=torch.stack([s for _, s in w2]),
        )


@dataclass
class MoeOutput:
    residual: torch.Tensor  # fp32 [DIM]: residual_in + the block's output
    experts: torch.Tensor  # int32 [TOP_K]: the chosen experts, in rank order
    weights: torch.Tensor  # fp32 [TOP_K]: their renormalized weights


def reference(
    w: MoeWeights, residual: torch.Tensor, dtype: torch.dtype, route: tuple[torch.Tensor, torch.Tensor] | None = None
) -> MoeOutput:
    """vLLM's Qwen3-MoE block in `dtype` (fp32 for ground truth, bf16 for the
    budget) on the dequantized experts; router logits are rounded to bf16 as
    vLLM's bf16 gate produces them. `route` ((experts, weights), rank order)
    replaces the block's own routing, so a test can hold the experts' math to
    account apart from which side of a bf16 tie routing fell on."""
    x = residual.float()
    h = (x * torch.rsqrt(x.pow(2).mean() + w.eps)).to(dtype) * w.norm.to(dtype)
    if route is None:
        logits = F.linear(h, w.router.to(dtype)).bfloat16().float()
        probs = torch.softmax(logits, dim=-1)
        top, experts = probs.topk(TOP_K)
        weights = top / top.sum()
    else:
        experts, weights = route
    out = torch.zeros(DIM, dtype=dtype, device=residual.device)
    for k in range(TOP_K):
        e = int(experts[k])
        w13 = int4.dequantize(w.w13_packed[e], w.w13_scales[e]).to(dtype)
        w2 = int4.dequantize(w.w2_packed[e], w.w2_scales[e]).to(dtype)
        gate_up = F.linear(h, w13)
        act = F.silu(gate_up[:EXPERT_FFN]) * gate_up[EXPERT_FFN:]
        out = out + weights[k].to(dtype) * F.linear(act, w2)
    return MoeOutput(x + out.float(), experts.int(), weights)


class ThinkerMoe:
    """Runs the MoE block on the kernel, one token a launch."""

    def __init__(self, weights: MoeWeights, timeout_ns: int = DEFAULT_TIMEOUT_NS, ctas: int | None = None) -> None:
        self.weights = weights
        device = weights.norm.device
        self._ctas = ctas or num_ctas(device)
        self._timeout_ns = timeout_ns
        self._logits = torch.zeros(EXPERTS, dtype=torch.float32, device=device)
        self._act = torch.zeros(TOP_K, EXPERT_FFN, dtype=torch.bfloat16, device=device)
        self._sync = sync_words(device)
        self._error = ErrorRecord()

    def run(self, residual: torch.Tensor) -> MoeOutput:
        out = MoeOutput(
            torch.empty(DIM, dtype=torch.float32, device=residual.device),
            torch.empty(TOP_K, dtype=torch.int32, device=residual.device),
            torch.empty(TOP_K, dtype=torch.float32, device=residual.device),
        )
        residual_in = residual.float().contiguous()
        ext = _ext.load()
        params = self.params(residual_in, out.residual, out.experts, out.weights)
        ext.run_thinker_moe(params=params, num_ctas=self._ctas)
        self._error.synchronize()
        return out

    def params(
        self, residual_in: torch.Tensor, residual: torch.Tensor, experts: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        """The block's launch params as CPU bytes, reading `residual_in` and
        writing `residual`, `experts` and `weights`."""
        w = self.weights
        return _ext.load().thinker_moe_params(
            residual_in=residual_in,
            norm=w.norm,
            router=w.router,
            w13_packed=w.w13_packed,
            w13_scales=w.w13_scales,
            w2_packed=w.w2_packed,
            w2_scales=w.w2_scales,
            eps=w.eps,
            timeout_ns=self._timeout_ns,
            router_logits=self._logits,
            act=self._act,
            residual=residual,
            experts=experts,
            weights=weights,
            sync=self._sync,
            error=self._error.tensor,
            num_ctas=self._ctas,
        )
