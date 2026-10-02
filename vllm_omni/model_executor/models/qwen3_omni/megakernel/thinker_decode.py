# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# Copyright 2026 Fractalyze Inc. All rights reserved.
"""One decode step of Qwen3-Omni's thinker in one launch
(csrc/thinker_decode.cu), and the PyTorch model it is held to.

Weights are as vLLM serves them: compressed-tensors W4A16 layers as
stored (int4.py), q, k and v rows fused in that order, every expert's
gate rows then up rows, and every fp16 tensor (norms, router, scales, LM head,
embeddings) in bf16.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from vllm_omni.model_executor.models.qwen3_omni.megakernel import _ext
from vllm_omni.model_executor.models.qwen3_omni.megakernel.barrier import (
    DEFAULT_TIMEOUT_NS,
    ErrorRecord,
    num_ctas,
    sync_words,
)
from vllm_omni.model_executor.models.qwen3_omni.megakernel.thinker_attention import (
    AttentionWeights,
    PagedCache,
    ThinkerAttention,
)
from vllm_omni.model_executor.models.qwen3_omni.megakernel.thinker_attention import reference as attention_reference
from vllm_omni.model_executor.models.qwen3_omni.megakernel.thinker_moe import MoeWeights, ThinkerMoe
from vllm_omni.model_executor.models.qwen3_omni.megakernel.thinker_moe import reference as moe_reference

DIM, LAYERS, VOCAB = 2048, 48, 152064


@dataclass(frozen=True)
class ThinkerWeights:
    embed: torch.Tensor | None  # bf16 [VOCAB, DIM]; None leaves embedding to the caller
    layers: list[tuple[AttentionWeights, MoeWeights]]
    final_norm: torch.Tensor  # bf16 [DIM]
    lm_head: torch.Tensor | None  # bf16 [VOCAB, DIM]; None leaves logits to the caller


def new_caches(num_layers: int, positions: int, block_size: int = 16, device: str = "cuda") -> list[PagedCache]:
    """Empty caches for `positions` positions per layer in vLLM's
    FlashAttention layout, on a shuffled block table shared by the layers."""
    blocks = (positions + block_size - 1) // block_size
    table = torch.randperm(blocks, device=device).int()
    caches = []
    for _ in range(num_layers):
        kv = torch.zeros(blocks, 4, block_size, 256, dtype=torch.bfloat16, device=device)
        key, value = kv.transpose(1, 2).split(128, dim=-1)
        caches.append(PagedCache(key, value, table))
    return caches


class ThinkerDecoder:
    """Runs decode steps on the kernel, one launch a step, over `caches`, on
    `ctas` CTAs (default one per SM). Beside other processes under MPS, fewer
    CTAs than SMs leave SMs to them: a step holds every SM it runs on for
    its whole length."""

    def __init__(
        self,
        weights: ThinkerWeights,
        caches: list[PagedCache],
        cos_sin: torch.Tensor,
        timeout_ns: int = DEFAULT_TIMEOUT_NS,
        ctas: int | None = None,
    ) -> None:
        device = weights.final_norm.device
        self.weights = weights
        self.caches = caches
        self.num_ctas = ctas or num_ctas(device)
        self.prefetch = True
        self._timeout_ns = timeout_ns
        self.residual = torch.zeros(DIM, dtype=torch.float32, device=device)
        self.logits = (
            None
            if weights.lm_head is None
            else torch.zeros(weights.lm_head.shape[0], dtype=torch.float32, device=device)
        )
        # step()'s own copy of what launch() reads per step.
        self._seq_len = torch.ones(1, dtype=torch.int32, device=device)
        self._slot = torch.zeros(1, dtype=torch.int64, device=device)
        self._positions = torch.zeros(3, 1, dtype=torch.int64, device=device)
        # Per layer: the chosen experts and their weights, for inspection.
        self.experts = torch.zeros(len(weights.layers), 8, dtype=torch.int32, device=device)
        self.expert_weights = torch.zeros(len(weights.layers), 8, device=device)
        self._blocks = []
        attention, moe = [], []
        # The kernel takes the position and residual per step instead of from
        # these.
        unused_positions = torch.zeros(3, dtype=torch.int32, device=device)
        for i, (aw, mw) in enumerate(weights.layers):
            a = ThinkerAttention(aw, cos_sin, timeout_ns, self.num_ctas)
            m = ThinkerMoe(mw, timeout_ns, self.num_ctas)
            self._blocks.append((a, m))
            attention.append(a.params(self.residual, self.residual, caches[i], 0, unused_positions))
            moe.append(m.params(self.residual, self.residual, self.experts[i], self.expert_weights[i]))
        self._attention = torch.stack(attention).to(device)
        self._moe = torch.stack(moe).to(device)
        self._sync = sync_words(device)
        self._error = ErrorRecord()

    def launch(
        self,
        embedding: torch.Tensor,
        seq_len: torch.Tensor,
        slot_mapping: torch.Tensor,
        positions: torch.Tensor,
        final_hidden: torch.Tensor | None = None,
        hidden: torch.Tensor | None = None,
        profile: torch.Tensor | None = None,
    ) -> None:
        """Queues one step without waiting for it, so a CUDA graph can capture
        it. The token (input `embedding`, [DIM]) sits at position
        seq_len[0] - 1 (int32), writes its keys and values to cache slot
        slot_mapping[0] (int64; negative writes nothing) and turns at M-RoPE
        `positions` (int64 [3, tokens], column 0). Writes the logits when the
        weights have an LM head; with `final_hidden` (bf16 [DIM]), the final
        norm's output; with `hidden` ([layers + 1, DIM] fp32), every layer's
        input and the last layer's output; with `profile` (int64 [CTAs,
        layers × 6, 2]), each CTA's globaltimer at every grid barrier."""
        self.residual.copy_(embedding.reshape(DIM))
        _ext.load().run_thinker_decode(
            attention=self._attention,
            moe=self._moe,
            seq_len=seq_len,
            slot_mapping=slot_mapping,
            positions=positions,
            final_norm=self.weights.final_norm,
            lm_head=self.weights.lm_head,
            eps=1e-6,
            prefetch=self.prefetch,
            timeout_ns=self._timeout_ns,
            residual=self.residual,
            logits=self.logits,
            final_hidden=final_hidden,
            hidden=hidden,
            profile=profile,
            sync=self._sync,
            error=self._error.tensor,
            num_ctas=self.num_ctas,
        )

    def step(
        self,
        embedding: torch.Tensor,
        pos: int,
        positions: torch.Tensor,
        hidden: torch.Tensor | None = None,
        profile: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Logits (fp32 [VOCAB]) for the token whose input `embedding` sits
        at cache position `pos` with M-RoPE `positions` ([3]), through the
        caches' block table; writes its keys and values. Waits for it."""
        cache = self.caches[0]
        block_size = cache.key.shape[1]
        self._seq_len.fill_(pos + 1)
        self._slot.copy_(cache.block_table[pos // block_size].long().view(1) * block_size + pos % block_size)
        self._positions.copy_(positions.view(3, 1))
        self.launch(embedding, self._seq_len, self._slot, self._positions, hidden=hidden, profile=profile)
        self._error.synchronize()
        return self.logits


def reference_step(
    weights: ThinkerWeights,
    caches: list[PagedCache],
    embedding: torch.Tensor,
    pos: int,
    positions: torch.Tensor,
    cos_sin: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """vLLM's thinker decode step in `dtype` on the dequantized weights:
    logits (fp32 [VOCAB]). Writes each layer's key and value, bf16 as a cache
    holds them, into `caches` at `pos`."""
    x = embedding.float()
    block_size = caches[0].key.shape[1]
    for (aw, mw), cache in zip(weights.layers, caches):
        out = attention_reference(aw, x, cache, pos, positions, cos_sin, dtype)
        block = int(cache.block_table[pos // block_size])
        cache.key[block, pos % block_size] = out.key.bfloat16()
        cache.value[block, pos % block_size] = out.value.bfloat16()
        x = moe_reference(mw, out.residual, dtype).residual
    h = (x * torch.rsqrt(x.pow(2).mean() + 1e-6)).to(dtype) * weights.final_norm.to(dtype)
    return F.linear(h, weights.lm_head.to(dtype)).float()
