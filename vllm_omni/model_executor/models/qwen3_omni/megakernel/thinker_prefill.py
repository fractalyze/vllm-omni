# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# Copyright 2026 Fractalyze Inc. All rights reserved.
"""One prefill chunk of Qwen3-Omni's thinker in one launch
(csrc/thinker_prefill.cu), and the PyTorch model it is held to.

The prefill shares a ThinkerDecoder's per-layer params and caches, so the
decode steps that follow read the keys and values the prefill wrote.
"""

from __future__ import annotations

import torch

from vllm_omni.model_executor.models.qwen3_omni.megakernel import _ext
from vllm_omni.model_executor.models.qwen3_omni.megakernel.barrier import ErrorRecord, num_ctas, sync_words
from vllm_omni.model_executor.models.qwen3_omni.megakernel.thinker_attention import HEAD_DIM, KV_HEADS, PagedCache
from vllm_omni.model_executor.models.qwen3_omni.megakernel.thinker_attention import reference as attention_reference
from vllm_omni.model_executor.models.qwen3_omni.megakernel.thinker_decode import DIM, ThinkerDecoder, ThinkerWeights
from vllm_omni.model_executor.models.qwen3_omni.megakernel.thinker_moe import EXPERT_FFN, EXPERTS, TOP_K
from vllm_omni.model_executor.models.qwen3_omni.megakernel.thinker_moe import reference as moe_reference

MAX_TOKENS = 64
QKV_ROWS, Q_DIM = 5120, 4096
BARRIERS_PER_LAYER = 9


class ThinkerPrefiller:
    """Runs prefill chunks of up to MAX_TOKENS tokens on the kernel, on a
    ThinkerDecoder's weights and caches, on `ctas` CTAs (default one per SM,
    whatever the decoder's own count: a prompt runs before the stages that
    share the GPU have work)."""

    def __init__(self, decoder: ThinkerDecoder, ctas: int | None = None) -> None:
        self.decoder = decoder
        device = decoder.residual.device
        self.num_ctas = ctas or num_ctas(device)
        layers = len(decoder.weights.layers)

        def buf(*shape, dtype=torch.float32):
            return torch.zeros(*shape, dtype=dtype, device=device)

        self.residual = buf(MAX_TOKENS, DIM)
        self._h = buf(MAX_TOKENS, DIM, dtype=torch.bfloat16)
        self._qkv = buf(MAX_TOKENS, QKV_ROWS)
        self._attn = buf(MAX_TOKENS, Q_DIM, dtype=torch.bfloat16)
        self._kv = buf(MAX_TOKENS, KV_HEADS, 2, HEAD_DIM, dtype=torch.bfloat16)
        self._logits = buf(MAX_TOKENS, EXPERTS)
        self._act = buf(MAX_TOKENS * TOP_K, EXPERT_FFN, dtype=torch.bfloat16)
        self._partial = buf(MAX_TOKENS, TOP_K, DIM)
        # Every layer's routing, [layers, tokens, TOP_K] for the last chunk.
        self.experts = buf(layers * MAX_TOKENS * TOP_K, dtype=torch.int32)
        self.weights = buf(layers * MAX_TOKENS * TOP_K)
        self._sync = sync_words(device)
        self._error = ErrorRecord()

    def launch(
        self,
        embeddings: torch.Tensor,
        seq_len: torch.Tensor,
        slot_mapping: torch.Tensor,
        positions: torch.Tensor,
        final_hidden: torch.Tensor,
        hidden: torch.Tensor | None = None,
        hidden_layer: int = -1,
        profile: torch.Tensor | None = None,
    ) -> None:
        """Queues one chunk without waiting: `embeddings` ([tokens, DIM]) sit
        at cache positions seq_len[0] - tokens on (int32), write their keys
        and values to slot_mapping (int64 [tokens]) and turn at M-RoPE
        `positions` (int64 [3, tokens], contiguous rows). Writes the final
        norm's output to final_hidden (bf16 [tokens, DIM]) and, with
        `hidden`, the residual entering layer `hidden_layer`. `profile`
        (int64 [num_ctas, layers, BARRIERS_PER_LAYER, 2]) takes each CTA's
        globaltimer on arriving at and leaving every grid barrier."""
        tokens = embeddings.shape[0]
        if not 0 < tokens <= MAX_TOKENS:
            raise ValueError(f"a chunk takes 1 to {MAX_TOKENS} tokens, got {tokens}")
        d = self.decoder
        self.residual[:tokens].copy_(embeddings)
        _ext.load().run_thinker_prefill(
            attention=d._attention,
            moe=d._moe,
            tokens=tokens,
            seq_len=seq_len,
            slot_mapping=slot_mapping,
            positions=positions,
            final_norm=d.weights.final_norm,
            eps=1e-6,
            timeout_ns=d._timeout_ns,
            residual=self.residual,
            h=self._h,
            qkv=self._qkv,
            attn=self._attn,
            kv=self._kv,
            router_logits=self._logits,
            act=self._act,
            partial=self._partial,
            experts=self.experts,
            weights=self.weights,
            final_hidden=final_hidden,
            hidden=hidden,
            hidden_layer=hidden_layer,
            profile=profile,
            sync=self._sync,
            error=self._error.tensor,
            num_ctas=self.num_ctas,
        )

    def run(self, embeddings: torch.Tensor, pos0: int, positions: torch.Tensor) -> torch.Tensor:
        """The final norm's output (bf16 [tokens, DIM]) of a chunk at cache
        positions pos0 on, through the caches' block table. Waits for it."""
        tokens = embeddings.shape[0]
        device = embeddings.device
        cache = self.decoder.caches[0]
        block_size = cache.key.shape[1]
        pos = torch.arange(pos0, pos0 + tokens, device=device)
        slots = cache.block_table.long()[pos // block_size] * block_size + pos % block_size
        seq_len = torch.tensor([pos0 + tokens], dtype=torch.int32, device=device)
        final = torch.empty(tokens, DIM, dtype=torch.bfloat16, device=device)
        self.launch(embeddings.float(), seq_len, slots, positions.long().contiguous(), final)
        self._error.synchronize()
        return final

    def routing(self, layer: int, tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Layer `layer`'s (experts, weights), [tokens, TOP_K] each, of the
        last chunk."""
        span = slice(layer * tokens * TOP_K, (layer + 1) * tokens * TOP_K)
        return (self.experts[span].view(tokens, TOP_K), self.weights[span].view(tokens, TOP_K))


def reference_prefill(
    weights: ThinkerWeights,
    caches: list[PagedCache],
    embeddings: torch.Tensor,
    pos0: int,
    positions: torch.Tensor,
    cos_sin: torch.Tensor,
    dtype: torch.dtype,
    routes: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
) -> torch.Tensor:
    """vLLM's thinker over a chunk in `dtype`, token by token: each token
    attends to the cache and the chunk's tokens before it, as a causal prefill
    does. Returns the final norm's output (fp32 [tokens, DIM]) and writes the
    keys and values to `caches`. routes[layer] = (experts, weights) [tokens,
    TOP_K] replaces the layers' own routing."""
    block_size = caches[0].key.shape[1]
    outs = []
    for i in range(embeddings.shape[0]):
        pos = pos0 + i
        x = embeddings[i].float()
        for layer, ((aw, mw), cache) in enumerate(zip(weights.layers, caches)):
            att = attention_reference(aw, x, cache, pos, positions[:, i], cos_sin, dtype)
            block = int(cache.block_table[pos // block_size])
            cache.key[block, pos % block_size] = att.key.bfloat16()
            cache.value[block, pos % block_size] = att.value.bfloat16()
            route = None
            if routes is not None:
                experts, w = routes[layer]
                route = (experts[i].long(), w[i])
            x = moe_reference(mw, att.residual, dtype, route).residual
        h = (x * torch.rsqrt(x.pow(2).mean() + 1e-6)).to(dtype) * weights.final_norm.to(dtype)
        outs.append(h.float())
    return torch.stack(outs)
