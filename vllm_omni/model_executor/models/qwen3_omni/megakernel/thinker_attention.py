# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# Copyright 2026 Fractalyze Inc. All rights reserved.
"""Qwen3-Omni's thinker attention block for one token in one launch
(csrc/thinker_attention.cu), and the PyTorch model it is held to.

qkv_proj and o_proj are compressed-tensors W4A16 (int4.py); the norms
are bf16, as vLLM loads them. The KV cache is vLLM's paged cache, read and
written in place through a block table.
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

DIM, Q_HEADS, KV_HEADS, HEAD_DIM = 2048, 32, 4, 128
Q_DIM, KV_DIM = Q_HEADS * HEAD_DIM, KV_HEADS * HEAD_DIM
QKV_ROWS = Q_DIM + 2 * KV_DIM
ROPE_THETA = 1_000_000.0
# Interleaved M-RoPE (vLLM's apply_interleaved_rope with sections 24/20/20):
# frequency i < 60 turns at the height position when i mod 3 = 1, at the
# width position when i mod 3 = 2, and at the temporal position otherwise.
MROPE_INTERLEAVED = 60


def cos_sin_table(positions: int, device: torch.device | str = "cuda") -> torch.Tensor:
    """vLLM's cos_sin_cache for the thinker: bf16 [positions, HEAD_DIM], each
    position's 64 cosines, then its 64 sines."""
    inv_freq = 1.0 / ROPE_THETA ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float, device=device) / HEAD_DIM)
    freqs = torch.outer(torch.arange(positions, dtype=torch.float, device=device), inv_freq)
    return torch.cat([freqs.cos(), freqs.sin()], dim=-1).bfloat16()


def mrope_axes(device: torch.device | str = "cuda") -> torch.Tensor:
    """For each of the HEAD_DIM / 2 frequencies, the index (0 temporal, 1
    height, 2 width) of the position it turns at."""
    i = torch.arange(HEAD_DIM // 2, device=device)
    return torch.where(i < MROPE_INTERLEAVED, i % 3, torch.zeros_like(i))


@dataclass(frozen=True)
class AttentionWeights:
    norm: torch.Tensor  # bf16 [DIM]: input_layernorm
    wqkv_packed: torch.Tensor  # int32 [QKV_ROWS, DIM / 8]: q rows, k rows, v rows
    wqkv_scales: torch.Tensor  # bf16 [QKV_ROWS, DIM / 32]
    q_norm: torch.Tensor  # bf16 [HEAD_DIM]
    k_norm: torch.Tensor  # bf16 [HEAD_DIM]
    wo_packed: torch.Tensor  # int32 [DIM, Q_DIM / 8]
    wo_scales: torch.Tensor  # bf16 [DIM, Q_DIM / 32]
    eps: float = 1e-6

    @classmethod
    def random(cls, seed: int) -> AttentionWeights:
        gen = torch.Generator(device="cuda").manual_seed(seed)

        def normal(*shape):
            return torch.randn(*shape, generator=gen, device="cuda")

        wqkv = int4.quantize(normal(QKV_ROWS, DIM) / DIM**0.5)
        wo = int4.quantize(normal(DIM, Q_DIM) / Q_DIM**0.5)
        return cls(
            norm=(1 + 0.1 * normal(DIM)).bfloat16(),
            wqkv_packed=wqkv[0],
            wqkv_scales=wqkv[1],
            q_norm=(1 + 0.1 * normal(HEAD_DIM)).bfloat16(),
            k_norm=(1 + 0.1 * normal(HEAD_DIM)).bfloat16(),
            wo_packed=wo[0],
            wo_scales=wo[1],
        )


@dataclass(frozen=True)
class PagedCache:
    """One layer of vLLM's paged KV cache: key and value views [blocks,
    block_size, KV_HEADS, HEAD_DIM] in any strides that keep a head
    contiguous, and the sequence's block table."""

    key: torch.Tensor
    value: torch.Tensor
    block_table: torch.Tensor  # int32 [blocks of the sequence]

    def gather(self, length: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Positions [0, length) as ([length, KV_HEADS, HEAD_DIM] keys, values)."""
        block_size = self.key.shape[1]
        t = torch.arange(length, device=self.key.device)
        blocks = self.block_table.long()[t // block_size]
        return self.key[blocks, t % block_size], self.value[blocks, t % block_size]


def _rms(x: torch.Tensor, weight: torch.Tensor, eps: float, dtype: torch.dtype) -> torch.Tensor:
    x = x.float()
    return (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)).to(dtype) * weight.to(dtype)


@dataclass
class AttentionOutput:
    residual: torch.Tensor  # fp32 [DIM]: residual_in + the block's output
    key: torch.Tensor  # [KV_HEADS, HEAD_DIM]: this step's key, after QK-norm and RoPE
    value: torch.Tensor  # [KV_HEADS, HEAD_DIM]


def reference(
    w: AttentionWeights,
    residual: torch.Tensor,
    cache: PagedCache,
    pos: int,
    positions: torch.Tensor,
    cos_sin: torch.Tensor,
    dtype: torch.dtype,
) -> AttentionOutput:
    """vLLM's Qwen3-MoE attention in `dtype` (fp32 for ground truth, bf16 for
    the budget) on the dequantized weights, attending to the cache's positions
    [0, pos) and this step's key and value."""
    x = residual.float()
    h = _rms(x, w.norm, w.eps, dtype)
    qkv = F.linear(h, int4.dequantize(w.wqkv_packed, w.wqkv_scales).to(dtype))
    q, k, v = qkv.split([Q_DIM, KV_DIM, KV_DIM])
    q = _rms(q.view(Q_HEADS, HEAD_DIM), w.q_norm, w.eps, dtype)
    k = _rms(k.view(KV_HEADS, HEAD_DIM), w.k_norm, w.eps, dtype)
    v = v.view(KV_HEADS, HEAD_DIM)
    turn_at = positions.long().to(cos_sin.device)[mrope_axes(cos_sin.device)]
    freq = torch.arange(HEAD_DIM // 2, device=cos_sin.device)
    cos = cos_sin[turn_at, freq].to(dtype)
    sin = cos_sin[turn_at, HEAD_DIM // 2 + freq].to(dtype)

    def rotate(x):
        x0, x1 = x[..., : HEAD_DIM // 2], x[..., HEAD_DIM // 2 :]
        return torch.cat([x0 * cos - x1 * sin, x1 * cos + x0 * sin], dim=-1)

    q, k = rotate(q), rotate(k)
    past_k, past_v = cache.gather(pos)
    keys = torch.cat([past_k.to(dtype), k[None]])  # [pos + 1, KV_HEADS, HEAD_DIM]
    values = torch.cat([past_v.to(dtype), v[None]])
    group = torch.arange(Q_HEADS, device=q.device) // (Q_HEADS // KV_HEADS)
    scores = torch.einsum("hd,thd->ht", q.float(), keys[:, group].float()) / HEAD_DIM**0.5
    attn = torch.einsum("ht,thd->hd", torch.softmax(scores, -1), values[:, group].float())
    out = F.linear(attn.to(dtype).flatten(), int4.dequantize(w.wo_packed, w.wo_scales).to(dtype))
    return AttentionOutput(x + out.float(), k, v)


class ThinkerAttention:
    """Runs the attention block on the kernel, one token a launch."""

    def __init__(
        self,
        weights: AttentionWeights,
        cos_sin: torch.Tensor,
        timeout_ns: int = DEFAULT_TIMEOUT_NS,
        ctas: int | None = None,
    ) -> None:
        self.weights = weights
        self.cos_sin = cos_sin
        device = weights.norm.device
        self._ctas = ctas or num_ctas(device)
        self.splits = self._ctas // Q_HEADS
        self._timeout_ns = timeout_ns
        self._qkv = torch.zeros(QKV_ROWS, dtype=torch.float32, device=device)
        items = Q_HEADS * self.splits
        self._partial_ml = torch.zeros(items, 2, dtype=torch.float32, device=device)
        self._partial_o = torch.zeros(items, HEAD_DIM, dtype=torch.float32, device=device)
        self._sync = sync_words(device)
        self._error = ErrorRecord()

    def run(self, residual: torch.Tensor, cache: PagedCache, pos: int, positions: torch.Tensor) -> torch.Tensor:
        """residual + the block's output, fp32 [DIM], for a token at cache
        position `pos` with M-RoPE `positions` (int32 [3]: temporal, height,
        width). Writes the token's key and value into `cache` at `pos`."""
        out = torch.empty(DIM, dtype=torch.float32, device=residual.device)
        residual_in = residual.float().contiguous()
        positions = positions.int().contiguous()
        params = self.params(residual_in, out, cache, pos, positions)
        _ext.load().run_thinker_attention(params=params, num_ctas=self._ctas)
        self._error.synchronize()
        return out

    def params(
        self, residual_in: torch.Tensor, residual: torch.Tensor, cache: PagedCache, pos: int, positions: torch.Tensor
    ) -> torch.Tensor:
        """The block's launch params as CPU bytes, reading `residual_in` and
        writing `residual` and the cache at `pos`."""
        w = self.weights
        return _ext.load().thinker_attention_params(
            residual_in=residual_in,
            norm=w.norm,
            wqkv_packed=w.wqkv_packed,
            wqkv_scales=w.wqkv_scales,
            q_norm=w.q_norm,
            k_norm=w.k_norm,
            wo_packed=w.wo_packed,
            wo_scales=w.wo_scales,
            cos_sin=self.cos_sin,
            positions=positions,
            key_cache=cache.key,
            value_cache=cache.value,
            block_table=cache.block_table,
            pos=pos,
            splits=self.splits,
            eps=w.eps,
            timeout_ns=self._timeout_ns,
            qkv=self._qkv,
            partial_ml=self._partial_ml,
            partial_o=self._partial_o,
            residual=residual,
            sync=self._sync,
            error=self._error.tensor,
            num_ctas=self._ctas,
        )
