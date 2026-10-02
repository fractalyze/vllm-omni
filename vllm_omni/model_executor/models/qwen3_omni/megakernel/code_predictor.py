# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# Copyright 2026 Fractalyze Inc. All rights reserved.
"""Qwen3-Omni's code predictor in one launch (csrc/qwen3omni_cp.cu): codes
1 to 15 of N audio frames, and the PyTorch model it is held to.

A frame runs the code predictor's layers over positions 0 to 15 with a KV
cache, the incremental equivalent of CodePredictorWrapper's re-prefill, and
draws code s from head s - 1 with the wrapper's "stored" sampler (top-k, then
top-p, then a Gumbel-max draw on the caller's uniforms) on every CTA.

The kernel reads the checkpoint's layout: rotate-half RoPE, q, k and v rows
fused in that order, and every gate row then every up row. vLLM-Omni's
qkv_proj and gate_up_proj hold exactly that, so the kernel reads them in
place and serving it beside the module costs no layer memory.
"""

from __future__ import annotations

import math
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

DIM, Q_HEADS, KV_HEADS, HEAD_DIM, FFN = 1024, 16, 8, 128, 3072
Q_DIM, KV_DIM = Q_HEADS * HEAD_DIM, KV_HEADS * HEAD_DIM
CODE_GROUPS = 16
# Heads, and the codes they draw: codes 1 to 15.
HEADS = CODE_GROUPS - 1
VOCAB = 2048
# Position 0 is the talker's hidden state, position s >= 1 the embedding of
# code s - 1; position 15 is the last to feed a head.
POSITIONS = HEADS + 1
# The largest top-k the kernel's sampler takes.
MAX_TOP_K = 50
# Bytes of its next GEMV slice each CTA prefetches into L2 before a barrier.
PREFETCH_BYTES = 96 * 1024


@dataclass(frozen=True)
class Layer:
    """One decoder layer as vLLM-Omni holds it: bf16, linear weights [out, in]."""

    wqkv: torch.Tensor  # [Q_DIM + 2 KV_DIM, DIM]: q rows, k rows, v rows
    wo: torch.Tensor  # [DIM, Q_DIM]
    w13: torch.Tensor  # [2 FFN, DIM]: gate rows, then up rows
    w2: torch.Tensor  # [DIM, FFN]
    input_layernorm: torch.Tensor  # [DIM]
    post_attention_layernorm: torch.Tensor  # [DIM]
    q_norm: torch.Tensor  # [HEAD_DIM]
    k_norm: torch.Tensor  # [HEAD_DIM]

    def to(self, dtype: torch.dtype) -> Layer:
        return Layer(**{name: t.to(dtype) for name, t in vars(self).items()})


@dataclass(frozen=True)
class Weights:
    layers: list[Layer]
    norm: torch.Tensor  # [DIM]
    lm_heads: torch.Tensor  # [HEADS, VOCAB, DIM]: head s - 1 draws code s
    codec_embeddings: torch.Tensor  # [HEADS, VOCAB, DIM]: table s - 1 embeds code s
    eps: float
    rope_theta: float

    def to(self, dtype: torch.dtype) -> Weights:
        return Weights(
            [layer.to(dtype) for layer in self.layers],
            self.norm.to(dtype),
            self.lm_heads.to(dtype),
            self.codec_embeddings.to(dtype),
            self.eps,
            self.rope_theta,
        )


def _check_bf16_cuda(t: torch.Tensor, shape: tuple[int, ...], name: str) -> None:
    if t.dtype != torch.bfloat16 or not t.is_cuda or tuple(t.shape) != shape or not t.is_contiguous():
        raise ValueError(
            f"{name} must be a contiguous bf16 CUDA tensor of shape {shape}, "
            f"got {t.dtype} {tuple(t.shape)} on {t.device}"
        )


def _check_layer(layer: Layer) -> None:
    for name, shape in {
        "wqkv": (Q_DIM + 2 * KV_DIM, DIM),
        "wo": (DIM, Q_DIM),
        "w13": (2 * FFN, DIM),
        "w2": (DIM, FFN),
        "input_layernorm": (DIM,),
        "post_attention_layernorm": (DIM,),
        "q_norm": (HEAD_DIM,),
        "k_norm": (HEAD_DIM,),
    }.items():
        _check_bf16_cuda(getattr(layer, name), shape, name)


def rope(theta: float, positions: int = POSITIONS) -> tuple[torch.Tensor, torch.Tensor]:
    """fp32 cos and sin, [positions, HEAD_DIM], in rotate-half's layout."""
    inv_freq = 1.0 / theta ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float32) / HEAD_DIM)
    freqs = torch.outer(torch.arange(positions, dtype=torch.float32), inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


def _rope_table(theta: float, device: torch.device) -> torch.Tensor:
    """[POSITIONS, HEAD_DIM / 2, (cos, sin)] in bf16, as the wrapper casts its
    tables: frequency i turns dims i and i + HEAD_DIM / 2."""
    cos, sin = rope(theta)
    half = HEAD_DIM // 2
    return torch.stack([cos[:, :half], sin[:, :half]], dim=-1).bfloat16().to(device)


class CodePredictor:
    """Runs frames of the code predictor on the kernel, one launch a call, on
    `ctas` CTAs (default one per SM), keeping its own KV caches. `weights`
    are read in place: the kernel holds raw pointers to them.

    The launch is plain, not cooperative: under CUDA MPS a cooperative launch
    waits until no other client's work is on any SM, which hung vLLM-Omni's
    stages. With at most one CTA per SM every CTA becomes resident as SMs
    free up, and one that never does trips the barrier watchdog. A launch
    holds every SM it runs on until it ends, so beside other processes fewer
    CTAs than SMs leave SMs to them.
    """

    def __init__(
        self,
        weights: Weights,
        top_k: int,
        top_p: float,
        timeout_ns: int = DEFAULT_TIMEOUT_NS,
        ctas: int | None = None,
    ) -> None:
        device = weights.norm.device
        table_shape = (HEADS, VOCAB, DIM)
        for layer in weights.layers:
            _check_layer(layer)
        _check_bf16_cuda(weights.norm, (DIM,), "norm")
        _check_bf16_cuda(weights.lm_heads, table_shape, "lm_heads")
        _check_bf16_cuda(weights.codec_embeddings, table_shape, "codec_embeddings")
        self.weights = weights
        self.set_sampling(top_k, top_p)
        self.num_ctas = ctas or num_ctas(device)
        self._timeout_ns = timeout_ns
        # One row of csrc/layer.h's LayerWeights pointers per layer.
        self._layer_table = torch.tensor(
            [
                [
                    layer.wqkv.data_ptr(),
                    layer.wo.data_ptr(),
                    layer.w13.data_ptr(),
                    layer.w2.data_ptr(),
                    layer.input_layernorm.data_ptr(),
                    layer.post_attention_layernorm.data_ptr(),
                    layer.q_norm.data_ptr(),
                    layer.k_norm.data_ptr(),
                ]
                for layer in weights.layers
            ],
            dtype=torch.int64,
            device=device,
        )
        self._rope = _rope_table(weights.rope_theta, device)

        def caches() -> list[torch.Tensor]:
            return [
                torch.zeros(1, KV_HEADS, POSITIONS, HEAD_DIM, dtype=torch.bfloat16, device=device)
                for _ in weights.layers
            ]

        self._k_caches, self._v_caches = caches(), caches()
        self._k_pointers = torch.tensor([t.data_ptr() for t in self._k_caches], dtype=torch.int64, device=device)
        self._v_pointers = torch.tensor([t.data_ptr() for t in self._v_caches], dtype=torch.int64, device=device)
        self._residual = torch.zeros(DIM, dtype=torch.float32, device=device)
        self._qkv = torch.zeros(Q_DIM + 2 * KV_DIM, dtype=torch.float32, device=device)
        self._act = torch.zeros(FFN, dtype=torch.bfloat16, device=device)
        self._sync = sync_words(device)
        self._error = ErrorRecord()

    def set_sampling(self, top_k: int, top_p: float) -> None:
        if not 0 < top_k <= MAX_TOP_K:
            raise ValueError(f"top_k must be in [1, {MAX_TOP_K}], got {top_k}")
        self.top_k, self.top_p = top_k, top_p

    def launch(
        self,
        talker_hidden: torch.Tensor,
        code0_embed: torch.Tensor,
        uniforms: torch.Tensor,
        forced_codes: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Queues `talker_hidden.shape[0]` frames without waiting for them,
        so a CUDA graph can capture it; returns (codes, logits), int64
        [frames, HEADS] and fp32 [frames, HEADS, VOCAB], filled when the
        launch finishes.

        `talker_hidden` and `code0_embed` are bf16 [frames, DIM]; `uniforms`
        is fp32 [frames, HEADS, VOCAB], each pass's Gumbel uniforms. With
        `forced_codes` (int64 [frames, HEADS]), pass s + 1 embeds code
        forced_codes[:, s - 1] instead of the drawn one.
        """
        frames, device = talker_hidden.shape[0], self.weights.norm.device
        _check_bf16_cuda(talker_hidden, (frames, DIM), "talker_hidden")
        _check_bf16_cuda(code0_embed, (frames, DIM), "code0_embed")
        codes = torch.empty(frames, HEADS, dtype=torch.int64, device=device)
        logits = torch.empty(frames, HEADS, VOCAB, dtype=torch.float32, device=device)
        _ext.load().run_code_predictor(
            layers=self._layer_table,
            final_norm=self.weights.norm,
            heads=self.weights.lm_heads,
            embeddings=self.weights.codec_embeddings,
            rope=self._rope,
            k_caches=self._k_pointers,
            v_caches=self._v_pointers,
            eps=self.weights.eps,
            top_k=self.top_k,
            top_p=self.top_p,
            talker_hidden=talker_hidden,
            code0_embed=code0_embed,
            uniforms=uniforms,
            forced_codes=forced_codes,
            prefetch_bytes=PREFETCH_BYTES,
            timeout_ns=self._timeout_ns,
            residual=self._residual,
            qkv=self._qkv,
            act=self._act,
            logits=logits,
            codes=codes,
            dump=None,
            profile=None,
            sync=self._sync,
            error=self._error.tensor,
            num_ctas=self.num_ctas,
            cooperative=False,
        )
        return codes, logits

    def run(self, *args, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        """`launch`, then waits for it; raises if a barrier's watchdog fired."""
        out = self.launch(*args, **kwargs)
        self._error.synchronize()
        return out


def _rms(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    h = x.float()
    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps)
    return weight * h.to(x.dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def _transformer(w: Weights, x: torch.Tensor) -> torch.Tensor:
    """The final norm's output for inputs `x` [seq, DIM], causally, as one prefill."""
    seq = x.shape[0]
    cos, sin = (t[:seq].to(device=x.device, dtype=x.dtype) for t in rope(w.rope_theta))
    for layer in w.layers:
        h = _rms(x, layer.input_layernorm, w.eps)
        q, k, v = F.linear(h, layer.wqkv).split([Q_DIM, KV_DIM, KV_DIM], dim=-1)
        q = _rms(q.view(seq, Q_HEADS, HEAD_DIM), layer.q_norm, w.eps)
        k = _rms(k.view(seq, KV_HEADS, HEAD_DIM), layer.k_norm, w.eps)
        v = v.view(seq, KV_HEADS, HEAD_DIM)
        q, k, v = (t.transpose(0, 1) for t in (q, k, v))
        q = q * cos + _rotate_half(q) * sin
        k = k * cos + _rotate_half(k) * sin
        y = F.scaled_dot_product_attention(
            q[None], k[None], v[None], is_causal=True, scale=1 / math.sqrt(HEAD_DIM), enable_gqa=True
        )
        x = x + F.linear(y[0].transpose(0, 1).reshape(seq, -1), layer.wo)
        h = _rms(x, layer.post_attention_layernorm, w.eps)
        gate, up = F.linear(h, layer.w13).split(FFN, dim=-1)
        x = x + F.linear(F.silu(gate) * up, layer.w2)
    return _rms(x, w.norm, w.eps)


def reference_sample(logits: torch.Tensor, uniforms: torch.Tensor, top_k: int, top_p: float) -> torch.Tensor:
    """CodePredictorWrapper's "stored" draw for each row of `logits` [..., VOCAB]."""
    shape = logits.shape[:-1]
    logits = logits.reshape(-1, logits.shape[-1])
    kth = logits.topk(top_k, dim=-1).values[:, -1:]
    logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p < 1.0:
        sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)
        probs = F.softmax(sorted_logits, dim=-1, dtype=torch.float32)
        remove = (probs.cumsum(dim=-1) - probs) >= top_p
        sorted_logits[remove] = float("-inf")
        logits = sorted_logits.scatter(1, sorted_idx, sorted_logits)
    noise = torch.log(-torch.log(uniforms.reshape(logits.shape)))
    return (logits.float() - noise).argmax(dim=-1).reshape(shape)


def reference(
    w: Weights,
    talker_hidden: torch.Tensor,
    code0_embed: torch.Tensor,
    uniforms: torch.Tensor,
    top_k: int,
    top_p: float,
    forced_codes: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """CodePredictorWrapper.forward for one frame, re-prefilling every pass,
    in the dtype of `w` (fp32 for ground truth, bf16 for the budget): (codes,
    logits), int64 [HEADS] and [HEADS, VOCAB].

    Pass s draws code s from head s - 1 with `uniforms[s - 1]`. Code s's
    embedding feeds pass s + 1: the drawn code, or `forced_codes[s - 1]`.
    """
    dtype = w.norm.dtype
    x = torch.zeros(POSITIONS, DIM, dtype=dtype, device=talker_hidden.device)
    x[0], x[1] = talker_hidden.to(dtype), code0_embed.to(dtype)
    logits, codes = [], []
    for s in range(1, CODE_GROUPS):
        hidden = _transformer(w, x[: s + 1])
        step_logits = F.linear(hidden[s], w.lm_heads[s - 1])
        code = reference_sample(step_logits, uniforms[s - 1], top_k, top_p)
        logits.append(step_logits)
        codes.append(code)
        if s < HEADS:
            fed = code if forced_codes is None else forced_codes[s - 1]
            x[s + 1] = w.codec_embeddings[s - 1][fed]
    return torch.stack(codes), torch.stack(logits)
