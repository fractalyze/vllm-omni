# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# Copyright 2026 Fractalyze Inc. All rights reserved.
"""One decode step of Qwen3-Omni's talker in one launch
(csrc/talker_decode.cu), and the PyTorch model it is held to.

The talker is vLLM's Qwen3MoeForCausalLM over codec tokens, all bf16: q, k
and v rows fused in that order, every expert's gate rows then up rows, and
the shared expert's gate rows then up rows, as vLLM loads them. The kernel
stops at the final norm; the codec head and sampling stay with the caller.
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
from vllm_omni.model_executor.models.qwen3_omni.megakernel.thinker_attention import PagedCache, mrope_axes

DIM = 1024
Q_HEADS, KV_HEADS, HEAD_DIM = 16, 2, 128
Q_DIM, KV_DIM = Q_HEADS * HEAD_DIM, KV_HEADS * HEAD_DIM
EXPERTS, TOP_K, EXPERT_FFN, SHARED_FFN = 128, 6, 384, 768
EPS = 1e-6


@dataclass(frozen=True)
class LayerWeights:
    norm: torch.Tensor  # [DIM]: input_layernorm
    wqkv: torch.Tensor  # [Q_DIM + 2 KV_DIM, DIM]
    q_norm: torch.Tensor  # [HEAD_DIM]
    k_norm: torch.Tensor  # [HEAD_DIM]
    wo: torch.Tensor  # [DIM, Q_DIM]
    moe_norm: torch.Tensor  # [DIM]: post_attention_layernorm
    router: torch.Tensor  # [EXPERTS, DIM]
    w13: torch.Tensor  # [EXPERTS, 2 EXPERT_FFN, DIM]: gate rows, then up rows
    w2: torch.Tensor  # [EXPERTS, DIM, EXPERT_FFN]
    shared_w13: torch.Tensor  # [2 SHARED_FFN, DIM]: gate rows, then up rows
    shared_w2: torch.Tensor  # [DIM, SHARED_FFN]
    shared_gate: torch.Tensor  # [1, DIM]


def new_caches(num_layers: int, positions: int, block_size: int = 16, device: str = "cuda") -> list[PagedCache]:
    """Empty caches for `positions` positions per layer in vLLM's
    FlashAttention layout, on a shuffled block table shared by the layers."""
    blocks = (positions + block_size - 1) // block_size
    table = torch.randperm(blocks, device=device).int()
    caches = []
    for _ in range(num_layers):
        kv = torch.zeros(blocks, KV_HEADS, block_size, 2 * HEAD_DIM, dtype=torch.bfloat16, device=device)
        key, value = kv.transpose(1, 2).split(HEAD_DIM, dim=-1)
        caches.append(PagedCache(key, value, table))
    return caches


def _layer_params(w: LayerWeights, cache: PagedCache) -> torch.Tensor:
    """A layer's weights and cache as the kernel reads them, CPU bytes."""
    return _ext.load().talker_layer_params(
        norm=w.norm,
        wqkv=w.wqkv,
        q_norm=w.q_norm,
        k_norm=w.k_norm,
        wo=w.wo,
        key_cache=cache.key,
        value_cache=cache.value,
        block_table=cache.block_table,
        moe_norm=w.moe_norm,
        router=w.router,
        w13=w.w13,
        w2=w.w2,
        shared_w13=w.shared_w13,
        shared_w2=w.shared_w2,
        shared_gate=w.shared_gate,
    )


class TalkerDecoder:
    """Runs talker decode steps on the kernel, one launch a step, on `ctas`
    CTAs (default one per SM), each query head's attention in `splits`
    chunks (default as many as the CTAs hold). Beside other processes under
    MPS, fewer CTAs than SMs leave SMs to them; the launch needs every CTA
    resident."""

    def __init__(
        self,
        layers: list[LayerWeights],
        caches: list[PagedCache],
        final_norm: torch.Tensor,
        cos_sin: torch.Tensor,
        timeout_ns: int = DEFAULT_TIMEOUT_NS,
        ctas: int | None = None,
        splits: int | None = None,
    ) -> None:
        device = final_norm.device
        # The kernel holds raw pointers to every layer's tensors.
        self.layers = layers
        self.caches = caches
        self.final_norm = final_norm
        self.cos_sin = cos_sin
        self.num_ctas = ctas or num_ctas(device)
        self.splits = splits or self.num_ctas // Q_HEADS
        self._timeout_ns = timeout_ns
        self._layers = torch.stack([_layer_params(w, c) for w, c in zip(layers, caches)]).to(device)
        self.residual = torch.zeros(DIM, dtype=torch.float32, device=device)
        self._qkv = torch.zeros(Q_DIM + 2 * KV_DIM, dtype=torch.float32, device=device)
        items = Q_HEADS * self.splits
        self._partial_ml = torch.zeros(items, 2, dtype=torch.float32, device=device)
        self._partial_o = torch.zeros(items, HEAD_DIM, dtype=torch.float32, device=device)
        self._router_logits = torch.zeros(EXPERTS + 1, dtype=torch.float32, device=device)
        self._act = torch.zeros(TOP_K * EXPERT_FFN + SHARED_FFN, dtype=torch.bfloat16, device=device)
        self._sync = sync_words(device)
        self._error = ErrorRecord()
        # step()'s own copy of what launch() reads per step.
        self._seq_len = torch.ones(1, dtype=torch.int32, device=device)
        self._slot = torch.zeros(1, dtype=torch.int64, device=device)
        self._positions = torch.zeros(3, 1, dtype=torch.int64, device=device)

    def launch(
        self,
        embedding: torch.Tensor,
        seq_len: torch.Tensor,
        slot_mapping: torch.Tensor,
        positions: torch.Tensor,
        final_hidden: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> None:
        """Queues one step without waiting for it, so a CUDA graph can capture
        it. The token (input `embedding`, [DIM]) sits at position
        seq_len[0] - 1 (int32), writes its keys and values to cache slot
        slot_mapping[0] (int64; negative writes nothing) and turns at M-RoPE
        `positions` (int64 [3, tokens], column 0). Writes the final norm's
        output to `final_hidden` (bf16 [DIM]); with `hidden` ([layers + 1,
        DIM] fp32), every layer's input and the last layer's output."""
        self.residual.copy_(embedding.reshape(DIM))
        _ext.load().run_talker_decode(
            layers=self._layers,
            seq_len=seq_len,
            slot_mapping=slot_mapping,
            positions=positions,
            cos_sin=self.cos_sin,
            final_norm=self.final_norm,
            splits=self.splits,
            eps=EPS,
            timeout_ns=self._timeout_ns,
            residual=self.residual,
            final_hidden=final_hidden,
            qkv=self._qkv,
            partial_ml=self._partial_ml,
            partial_o=self._partial_o,
            router_logits=self._router_logits,
            act=self._act,
            hidden=hidden,
            profile=None,
            sync=self._sync,
            error=self._error.tensor,
            num_ctas=self.num_ctas,
        )

    def step(self, embedding: torch.Tensor, pos: int, positions: torch.Tensor) -> torch.Tensor:
        """The final norm's output (bf16 [DIM]) for the token whose input
        `embedding` sits at cache position `pos` with M-RoPE `positions`
        ([3]), through the caches' block table; writes its keys and values.
        Waits for it."""
        cache = self.caches[0]
        block_size = cache.key.shape[1]
        self._seq_len.fill_(pos + 1)
        self._slot.copy_(cache.block_table[pos // block_size].long().view(1) * block_size + pos % block_size)
        self._positions.copy_(positions.view(3, 1))
        out = torch.empty(DIM, dtype=torch.bfloat16, device=embedding.device)
        self.launch(embedding, self._seq_len, self._slot, self._positions, out)
        self._error.synchronize()
        return out


def _rms(x: torch.Tensor, weight: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    x = x.float()
    return (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS)).to(dtype) * weight.to(dtype)


def _attention(w, x, cache: PagedCache, pos: int, positions: torch.Tensor, cos_sin: torch.Tensor, dtype):
    h = _rms(x, w.norm, dtype)
    q, k, v = F.linear(h, w.wqkv.to(dtype)).split([Q_DIM, KV_DIM, KV_DIM])
    q = _rms(q.view(Q_HEADS, HEAD_DIM), w.q_norm, dtype)
    k = _rms(k.view(KV_HEADS, HEAD_DIM), w.k_norm, dtype)
    v = v.view(KV_HEADS, HEAD_DIM)
    turn_at = positions.long().to(cos_sin.device)[mrope_axes(cos_sin.device)]
    freq = torch.arange(HEAD_DIM // 2, device=cos_sin.device)
    cos = cos_sin[turn_at, freq].to(dtype)
    sin = cos_sin[turn_at, HEAD_DIM // 2 + freq].to(dtype)

    def rotate(t):
        t0, t1 = t[..., : HEAD_DIM // 2], t[..., HEAD_DIM // 2 :]
        return torch.cat([t0 * cos - t1 * sin, t1 * cos + t0 * sin], dim=-1)

    q, k = rotate(q), rotate(k)
    past_k, past_v = cache.gather(pos)
    keys = torch.cat([past_k.to(dtype), k[None].bfloat16().to(dtype)])
    values = torch.cat([past_v.to(dtype), v[None].bfloat16().to(dtype)])
    group = torch.arange(Q_HEADS, device=q.device) // (Q_HEADS // KV_HEADS)
    scores = torch.einsum("hd,thd->ht", q.float(), keys[:, group].float()) / HEAD_DIM**0.5
    attn = torch.einsum("ht,thd->hd", torch.softmax(scores, -1), values[:, group].float())
    out = F.linear(attn.to(dtype).flatten(), w.wo.to(dtype))
    return x + out.float(), k, v


def _moe(w: LayerWeights, x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    h = _rms(x, w.moe_norm, dtype)
    logits = F.linear(h, w.router.to(dtype)).float()
    weights, experts = torch.softmax(logits, -1).topk(TOP_K)
    weights = weights / weights.sum()
    out = torch.zeros(DIM, dtype=torch.float32, device=x.device)
    for weight, e in zip(weights, experts.tolist()):
        gate, up = F.linear(h, w.w13[e].to(dtype)).split(EXPERT_FFN)
        act = (F.silu(gate.float()) * up.float()).to(dtype)
        out += weight * F.linear(act, w.w2[e].to(dtype)).float()
    gate, up = F.linear(h, w.shared_w13.to(dtype)).split(SHARED_FFN)
    act = (F.silu(gate.float()) * up.float()).to(dtype)
    scale = torch.sigmoid(F.linear(h, w.shared_gate.to(dtype)).float())
    out += scale * F.linear(act, w.shared_w2.to(dtype)).float()
    return x + out


def reference_step(
    layers: list[LayerWeights],
    final_norm: torch.Tensor,
    caches: list[PagedCache],
    embedding: torch.Tensor,
    pos: int,
    positions: torch.Tensor,
    cos_sin: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """vLLM's talker decode step in `dtype` (fp32 for ground truth, bf16 for
    the budget): the final norm's output, fp32 [DIM]. Writes each layer's key
    and value, bf16 as a cache holds them, into `caches` at `pos`."""
    x = embedding.float()
    block_size = caches[0].key.shape[1]
    for w, cache in zip(layers, caches):
        x, k, v = _attention(w, x, cache, pos, positions, cos_sin, dtype)
        block = int(cache.block_table[pos // block_size])
        cache.key[block, pos % block_size] = k.bfloat16()
        cache.value[block, pos % block_size] = v.bfloat16()
        x = _moe(w, x, dtype)
    return _rms(x, final_norm, dtype).float()
