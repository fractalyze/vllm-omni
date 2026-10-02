# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# Copyright 2026 Fractalyze Inc. All rights reserved.
"""Runs Qwen3-Omni's thinker steps on the megakernels, for
Qwen3OmniMoeThinkerForConditionalGeneration when VLLM_OMNI_THINKER_MEGAKERNEL
is set.

A step of one token goes to the decode kernel (thinker_decode.py): all 48
layers and the final norm in one launch, which a CUDA graph captures like the
model it replaces. With VLLM_OMNI_THINKER_MEGAKERNEL_PREFILL, a step of one
request's 2 to 64 prompt tokens goes to the prefill kernel
(thinker_prefill.py), also one launch. vLLM runs such steps outside its full
CUDA graphs (they are not uniform decode batches), so each one launches
eagerly; the tokens past num_actual_tokens are a piecewise graph's padding.
Every other step keeps the thinker's own forward: prompts with vision inputs
(the kernels do not add the deepstack embeddings), longer chunks, batches of
two, and the profile run. vLLM's LM head and sampler still turn the returned
hidden state into the token.

Weights, as vLLM holds them after loading:
- qkv_proj and o_proj are repacked for Marlin, so their checkpoint packing is
  copied at the end of load_weights, where vLLM's memory profile counts it;
- the experts are read in place: moe_backend triton keeps the checkpoint
  packing, as bytes. Marlin repacks them, and a copy of every expert does not
  fit beside the model, so the kernels serve only a triton thinker;
- norms and the router are read in place.

The kernels read the step's position, cache slot and M-RoPE positions from
vLLM's persistent buffers, and the caches through the first step's block
table, whose storage vLLM keeps for its lifetime.

A step holds every SM it runs on until it ends, and the thinker's steps run
back to back, so on all SMs the talker and code2wav would wait out whole
steps under MPS. VLLM_OMNI_THINKER_MEGAKERNEL_CTAS sets how many CTAs (SMs) a
decode step takes; prefill chunks take every SM, as they run before the
talker has work, unless VLLM_OMNI_THINKER_MEGAKERNEL_PREFILL_CTAS caps them.
The cap changes how the kernels split their sums, so the text a prompt gets
depends on it; at a fixed cap every repeat is identical.
"""

from __future__ import annotations

import torch
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger

from vllm_omni.model_executor.models.qwen3_omni.megakernel.thinker_attention import (
    HEAD_DIM,
    AttentionWeights,
    PagedCache,
)
from vllm_omni.model_executor.models.qwen3_omni.megakernel.thinker_decode import ThinkerDecoder, ThinkerWeights
from vllm_omni.model_executor.models.qwen3_omni.megakernel.thinker_moe import DIM, EXPERT_FFN, EXPERTS, MoeWeights
from vllm_omni.model_executor.models.qwen3_omni.megakernel.thinker_prefill import MAX_TOKENS, ThinkerPrefiller

logger = init_logger(__name__)


def _attn_metadata():
    """This step's attention metadata for the thinker's layers, or None
    outside a forward with KV caches (the profile run) or for a layout the
    kernels do not take (speculative decoding's list)."""
    metadata = get_forward_context().attn_metadata
    if not isinstance(metadata, dict) or not metadata:
        return None
    return next(iter(metadata.values()))


def _linears(model) -> list[tuple[torch.Tensor, ...]]:
    """Each layer's (qkv words, qkv scales, o words, o scales), copied while
    still in checkpoint packing."""
    out = []
    for layer in model.layers:
        attn = layer.self_attn
        out.append(
            tuple(
                t.detach().clone()
                for t in (
                    attn.qkv_proj.weight_packed,
                    attn.qkv_proj.weight_scale,
                    attn.o_proj.weight_packed,
                    attn.o_proj.weight_scale,
                )
            )
        )
    return out


def _experts(experts) -> tuple[torch.Tensor, ...]:
    """(w13 words, w13 scales, w2 words, w2 scales) of a FusedMoE as the
    kernels read them: triton's post-load uint8 bytes, viewed as int32."""
    # vLLM 0.30 wraps the expert weights in a MoERunner.
    experts = getattr(experts, "routed_experts", experts)
    w13, w2 = experts.w13_weight_packed, experts.w2_weight_packed
    if w13.dtype != torch.uint8 or tuple(w13.shape) != (EXPERTS, 2 * EXPERT_FFN, DIM // 2):
        raise RuntimeError(
            "the thinker megakernels read the experts as moe_backend triton holds them; got "
            f"w13 {tuple(w13.shape)} {w13.dtype} (marlin repacks them)"
        )
    return (w13.view(torch.int32), experts.w13_weight_scale, w2.view(torch.int32), experts.w2_weight_scale)


def _decode_eligible(inputs_embeds, intermediate_tensors, metadata) -> bool:
    return (
        inputs_embeds is not None
        and inputs_embeds.shape[0] == 1
        and intermediate_tensors is None
        and metadata is not None
        and getattr(metadata, "block_table", None) is not None
    )


def _prefill_eligible(thinker, inputs_embeds, intermediate_tensors, metadata, capture_layer_indices) -> bool:
    """One request's 2 to 64 prompt tokens, eager, without vision inputs, and
    at most one captured layer past the embedding (the prefill keeps one)."""
    if (
        inputs_embeds is None
        or intermediate_tensors is not None
        or metadata is None
        or getattr(metadata, "block_table", None) is None
        or torch.cuda.is_current_stream_capturing()
    ):
        return False
    tokens = getattr(metadata, "num_actual_tokens", None)
    return (
        tokens is not None
        and 1 < tokens <= MAX_TOKENS
        and metadata.seq_lens.shape[0] == 1
        and metadata.max_query_len == tokens
        and getattr(thinker, "deepstack_input_embeds_num_tokens", 0) == 0
        and sum(i > 0 for i in capture_layer_indices or []) <= 1
    )


def _output(final, inputs_embeds, capture_layer_indices, return_hidden_states, layer_input):
    """The thinker forward's return value for a kernel step: the final hidden
    state, with the captured layer inputs when vLLM asks for them.
    `layer_input(i)` is the residual entering layer i > 0."""
    if capture_layer_indices is None and not return_hidden_states:
        return final
    captured = None
    if return_hidden_states:
        captured = {}
        if capture_layer_indices:
            # Layer 0's input is the embedding itself, as vLLM captures it.
            layers = {i: inputs_embeds.clone() if i == 0 else layer_input(i) for i in capture_layer_indices}
            captured = {"hidden_states": {"layers": layers}}
    return final, captured


class _Kernels:
    """The decode kernel, and the prefill kernel once a prompt chunk needs it,
    on a loaded thinker's weights and KV caches."""

    def __init__(
        self,
        thinker,
        linears,
        cos_sin,
        block_table: torch.Tensor,
        decode_ctas: int | None,
        prefill_ctas: int | None,
    ) -> None:
        model = thinker.language_model.model
        layers, caches = [], []
        for layer, (qkv, qkv_scales, o, o_scales) in zip(model.layers, linears):
            attn = layer.self_attn
            w13, w13_scales, w2, w2_scales = _experts(layer.mlp.experts)
            layers.append(
                (
                    AttentionWeights(
                        norm=layer.input_layernorm.weight,
                        wqkv_packed=qkv,
                        wqkv_scales=qkv_scales,
                        q_norm=attn.q_norm.weight,
                        k_norm=attn.k_norm.weight,
                        wo_packed=o,
                        wo_scales=o_scales,
                        eps=layer.input_layernorm.variance_epsilon,
                    ),
                    MoeWeights(
                        norm=layer.post_attention_layernorm.weight,
                        router=layer.mlp.gate.weight,
                        w13_packed=w13,
                        w13_scales=w13_scales,
                        w2_packed=w2,
                        w2_scales=w2_scales,
                        eps=layer.post_attention_layernorm.variance_epsilon,
                    ),
                )
            )
            key, value = attn.attn.kv_cache.transpose(1, 2).split(HEAD_DIM, dim=-1)
            caches.append(PagedCache(key, value, block_table))
        weights = ThinkerWeights(embed=None, layers=layers, final_norm=model.norm.weight, lm_head=None)
        self.block_table = block_table
        self.decoder = ThinkerDecoder(weights, caches, cos_sin, ctas=decode_ctas)
        self._prefill_ctas = prefill_ctas
        self._prefiller: ThinkerPrefiller | None = None
        # The residual entering each layer; vLLM's talker takes layer 24's.
        self.hidden = torch.zeros(len(layers) + 1, weights.final_norm.shape[0], device=block_table.device)
        logger.info("thinker megakernel: %d layers, %d CTAs a decode step", len(layers), self.decoder.num_ctas)

    def decode(self, inputs_embeds, positions, metadata, capture_layer_indices, return_hidden_states):
        final = torch.empty_like(inputs_embeds)
        self.decoder.launch(
            inputs_embeds,
            metadata.seq_lens,
            metadata.slot_mapping,
            positions,
            final_hidden=final.view(-1),
            hidden=self.hidden,
        )
        return _output(
            final,
            inputs_embeds,
            capture_layer_indices,
            return_hidden_states,
            lambda i: self.hidden[i].to(inputs_embeds.dtype).view(1, -1),
        )

    def prefill(self, inputs_embeds, positions, metadata, capture_layer_indices, return_hidden_states):
        if self._prefiller is None:
            self._prefiller = ThinkerPrefiller(self.decoder, ctas=self._prefill_ctas)
        tokens = metadata.num_actual_tokens
        final = torch.zeros_like(inputs_embeds)
        accept = max(capture_layer_indices or [0])
        hidden = None
        if return_hidden_states and accept > 0:
            hidden = torch.zeros(inputs_embeds.shape, dtype=torch.float32, device=inputs_embeds.device)
        self._prefiller.launch(
            inputs_embeds[:tokens].float(),
            metadata.seq_lens,
            metadata.slot_mapping,
            positions,
            final[:tokens],
            hidden=hidden,
            hidden_layer=accept if hidden is not None else -1,
        )
        return _output(
            final, inputs_embeds, capture_layer_indices, return_hidden_states, lambda i: hidden.to(inputs_embeds.dtype)
        )


class ThinkerMegakernel:
    """A thinker's megakernel path: `load_weights` keeps what the kernels need
    from the loaded weights, and `forward` runs an eligible step."""

    def __init__(self, decode_ctas: int | None, prefill: bool, prefill_ctas: int | None) -> None:
        self._decode_ctas = decode_ctas
        self._prefill = prefill
        self._prefill_ctas = prefill_ctas
        self._linears: list[tuple[torch.Tensor, ...]] | None = None
        self._cos_sin: torch.Tensor | None = None
        self._kernels: _Kernels | None = None
        # Each way once: whether real prompts reach the prefill kernel.
        self._logged: set[bool] = set()

    def load_weights(self, thinker) -> None:
        """Copies what vLLM repacks after loading; call once the thinker's
        weights are loaded."""
        with torch.inference_mode():
            model = thinker.language_model.model
            self._linears = _linears(model)
            rope = model.layers[0].self_attn.rotary_emb
            self._cos_sin = rope.cos_sin_cache.to(torch.bfloat16).contiguous()
        logger.info("thinker megakernel: copied %d layers' attention linears", len(model.layers))

    def forward(
        self,
        thinker,
        inputs_embeds,
        positions,
        intermediate_tensors,
        capture_layer_indices,
        return_hidden_states,
    ):
        """The thinker's forward output for this step from a kernel, or None
        when the step keeps the thinker's own forward."""
        metadata = _attn_metadata()
        prefilling = self._prefill and _prefill_eligible(
            thinker, inputs_embeds, intermediate_tensors, metadata, capture_layer_indices
        )
        tokens = getattr(metadata, "num_actual_tokens", 0)
        if self._prefill and tokens > 1 and prefilling not in self._logged:
            self._logged.add(prefilling)
            logger.info(
                "thinker megakernel: a %d-token step %s the prefill kernel", tokens, "takes" if prefilling else "skips"
            )
        if not prefilling and not _decode_eligible(inputs_embeds, intermediate_tensors, metadata):
            return None
        kernels = self._kernels
        if kernels is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("the thinker megakernel must first run eagerly, before any CUDA graph capture")
            if self._linears is None:
                raise RuntimeError("the thinker megakernel runs only after load_weights")
            kernels = self._kernels = _Kernels(
                thinker,
                self._linears,
                self._cos_sin,
                metadata.block_table[0],
                self._decode_ctas,
                self._prefill_ctas,
            )
        elif metadata.block_table.data_ptr() != kernels.block_table.data_ptr():
            raise RuntimeError("vLLM moved its block table; the thinker megakernel reads the first step's")
        with torch.inference_mode():
            run = kernels.prefill if prefilling else kernels.decode
            return run(inputs_embeds, positions, metadata, capture_layer_indices, return_hidden_states)
