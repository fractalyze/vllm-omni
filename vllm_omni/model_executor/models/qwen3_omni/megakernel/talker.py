# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# Copyright 2026 Fractalyze Inc. All rights reserved.
"""Runs Qwen3-Omni's talker on the megakernels: its decode steps for
Qwen3OmniMoeTalkerForConditionalGeneration when VLLM_OMNI_TALKER_MEGAKERNEL
is set, and its code predictor for Qwen3OmniMoeTalkerCodePredictor when
VLLM_OMNI_CODE_PREDICTOR_MEGAKERNEL is set.

Talker decode (TalkerMegakernel): a step of one token of one request goes to
the talker kernel (talker_decode.py), all 20 layers and the final norm in
one launch, which a CUDA graph captures like the model it replaces. vLLM's
codec head and sampler still turn the returned hidden state into the code.
Every other step keeps vLLM's forward: prompts, batches of two, and the
profile run. The kernel reads vLLM's bf16 weights in place, the step's
position, cache slot and M-RoPE positions from vLLM's persistent buffers, and
the caches through the first step's block table, whose storage vLLM keeps
for its lifetime. VLLM_OMNI_TALKER_MEGAKERNEL_CTAS sets its CTAs (default
96), which leaves the thinker's decode its SMs under MPS.

Code predictor (CodePredictorMegakernel): every call runs on the
code-predictor kernel (code_predictor.py), codes 1 to 15 of each frame in
one launch. The kernel is built right after load_weights, so vLLM's memory
profile counts what it allocates. It reads the layers' fused weights in
place; the heads and codec embeddings are stacked once, and the module's own
copies then alias the stacks, so nothing is held twice. It draws each pass's
code with the wrapper's sampler on uniforms drawn up front from the same
generators, one [frames, 15, 2048] draw instead of 15 [frames, 2048] ones:
the same distribution, not the same random stream.
VLLM_OMNI_CODE_PREDICTOR_MEGAKERNEL_CTAS caps its CTAs, which leaves SMs to
code2wav.

The CTA counts change how the kernels split their sums, so the audio a
prompt gets depends on them; which SMs the CTAs run on does not.
"""

from __future__ import annotations

import torch
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger

from vllm_omni.model_executor.models.common.qwen3_code_predictor import _UNIFORM_EPS
from vllm_omni.model_executor.models.qwen3_omni.megakernel.code_predictor import DIM as CODE_PREDICTOR_DIM
from vllm_omni.model_executor.models.qwen3_omni.megakernel.code_predictor import (
    HEADS,
    POSITIONS,
    VOCAB,
    CodePredictor,
    Layer,
    Weights,
)
from vllm_omni.model_executor.models.qwen3_omni.megakernel.talker_decode import (
    DIM,
    EXPERT_FFN,
    EXPERTS,
    HEAD_DIM,
    LayerWeights,
    TalkerDecoder,
)
from vllm_omni.model_executor.models.qwen3_omni.megakernel.thinker_attention import PagedCache
from vllm_omni.platforms import current_omni_platform

logger = init_logger(__name__)

# The talker kernel splits each query head's attention in two.
_TALKER_SPLITS = 2


def _attn_metadata():
    """This step's attention metadata for the talker's layers, or None outside
    a forward with KV caches (the profile run)."""
    metadata = get_forward_context().attn_metadata
    if not isinstance(metadata, dict) or not metadata:
        return None
    return next(iter(metadata.values()))


def _talker_layer(layer) -> LayerWeights:
    """A vLLM Qwen3MoeDecoderLayer's weights as the talker kernel reads them,
    in place."""
    attn, mlp = layer.self_attn, layer.mlp
    # vLLM 0.30 wraps the expert weights in a MoERunner.
    experts = getattr(mlp.experts, "routed_experts", mlp.experts)
    w13, w2 = experts.w13_weight, experts.w2_weight
    if (
        w13.dtype != torch.bfloat16
        or tuple(w13.shape) != (EXPERTS, 2 * EXPERT_FFN, DIM)
        or tuple(w2.shape) != (EXPERTS, DIM, EXPERT_FFN)
    ):
        raise RuntimeError(
            "the talker megakernel reads bf16 experts [E, 2I, H] and [E, H, I]; got "
            f"w13 {tuple(w13.shape)} {w13.dtype}, w2 {tuple(w2.shape)}"
        )
    return LayerWeights(
        norm=layer.input_layernorm.weight,
        wqkv=attn.qkv_proj.weight,
        q_norm=attn.q_norm.weight,
        k_norm=attn.k_norm.weight,
        wo=attn.o_proj.weight,
        moe_norm=layer.post_attention_layernorm.weight,
        router=mlp.gate.weight,
        w13=w13,
        w2=w2,
        shared_w13=mlp.shared_expert.gate_up_proj.weight,
        shared_w2=mlp.shared_expert.down_proj.weight,
        shared_gate=mlp.shared_expert_gate.weight,
    )


def _talker_eligible(inputs_embeds, positions, intermediate_tensors, metadata) -> bool:
    """One token of one request, with M-RoPE positions and this step's caches."""
    return (
        inputs_embeds is not None
        and inputs_embeds.shape[0] == 1
        and positions.dim() == 2
        and intermediate_tensors is None
        and metadata is not None
        and getattr(metadata, "block_table", None) is not None
    )


class TalkerMegakernel:
    """A talker's decode path: `forward` runs an eligible step on the kernel,
    built on the first one."""

    def __init__(self, ctas: int) -> None:
        self._ctas = ctas
        self._decoder: TalkerDecoder | None = None

    def _build(self, talker, block_table: torch.Tensor) -> TalkerDecoder:
        model = talker.language_model.model
        caches = []
        for layer in model.layers:
            key, value = layer.self_attn.attn.kv_cache.transpose(1, 2).split(HEAD_DIM, dim=-1)
            caches.append(PagedCache(key, value, block_table))
        cos_sin = model.layers[0].self_attn.rotary_emb.cos_sin_cache.to(torch.bfloat16).contiguous()
        decoder = TalkerDecoder(
            [_talker_layer(layer) for layer in model.layers],
            caches,
            model.norm.weight,
            cos_sin,
            ctas=self._ctas,
            splits=_TALKER_SPLITS,
        )
        logger.info("talker megakernel: %d layers, %d CTAs a decode step", len(model.layers), decoder.num_ctas)
        return decoder

    def forward(self, talker, inputs_embeds, positions, intermediate_tensors):
        """The talker's forward output (the final norm's output) for this step
        from the kernel, or None when the step keeps vLLM's forward."""
        metadata = _attn_metadata()
        if not _talker_eligible(inputs_embeds, positions, intermediate_tensors, metadata):
            return None
        if self._decoder is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("the talker megakernel must first run eagerly, before any CUDA graph capture")
            self._decoder = self._build(talker, metadata.block_table[0])
        elif metadata.block_table.data_ptr() != self._decoder.caches[0].block_table.data_ptr():
            raise RuntimeError("vLLM moved its block table; the talker megakernel reads the first step's")
        with torch.inference_mode():
            out = torch.empty_like(inputs_embeds)
            self._decoder.launch(
                inputs_embeds, metadata.seq_lens, metadata.slot_mapping, positions, final_hidden=out.view(-1)
            )
            return out


def _code_predictor_weights(wrapper) -> Weights:
    """A loaded CodePredictorWrapper's weights as the kernel reads them. The
    layers' fused weights are read in place; the heads and codec embeddings
    are stacked, and the module's own copies then alias the stacks."""
    model, cfg = wrapper.model, wrapper.config
    layers = [
        Layer(
            wqkv=layer.self_attn.qkv_proj.weight,
            wo=layer.self_attn.o_proj.weight,
            w13=layer.mlp.gate_up_proj.weight,
            w2=layer.mlp.down_proj.weight,
            input_layernorm=layer.input_layernorm.weight,
            post_attention_layernorm=layer.post_attention_layernorm.weight,
            q_norm=layer.self_attn.q_norm.weight,
            k_norm=layer.self_attn.k_norm.weight,
        )
        for layer in model.layers
    ]
    heads = torch.stack([h.weight[:VOCAB] for h in wrapper.lm_head])
    embeddings = torch.stack([e.weight[:VOCAB] for e in model.codec_embedding])
    for i, h in enumerate(wrapper.lm_head):
        if h.weight.shape[0] == VOCAB:
            h.weight.data = heads[i]
    for i, e in enumerate(model.codec_embedding):
        if e.weight.shape[0] == VOCAB:
            e.weight.data = embeddings[i]
    rope = getattr(cfg, "rope_parameters", None) or {}
    theta = rope.get("rope_theta", getattr(cfg, "rope_theta", 10000.0))
    return Weights(layers, model.norm.weight, heads, embeddings, cfg.rms_norm_eps, theta)


class CodePredictorMegakernel:
    """A code predictor's kernel path: `load_weights` builds the kernel on the
    loaded weights, and `forward` runs every call on it."""

    def __init__(self, ctas: int | None) -> None:
        self._ctas = ctas
        self._predictor: CodePredictor | None = None

    def load_weights(self, wrapper) -> None:
        """Builds the kernel; call once the wrapper's weights are loaded."""
        with torch.inference_mode():
            self._predictor = CodePredictor(
                _code_predictor_weights(wrapper), wrapper._top_k, wrapper._top_p, ctas=self._ctas
            )
        # The originals the stacks replaced are free but still reserved by the
        # caching allocator, which vLLM's memory profile would count as in use.
        current_omni_platform.empty_cache()
        logger.info(
            "code predictor megakernel: %d layers, %d CTAs a launch",
            len(wrapper.model.layers),
            self._predictor.num_ctas,
        )

    def forward(self, wrapper, layer0_code, layer0_embed, last_talker_hidden, generator, generators, sample_uniforms):
        """CodePredictorWrapper.forward's (codes, proj_buf) from the kernel."""
        predictor = self._predictor
        if predictor is None:
            raise RuntimeError("the code predictor megakernel runs only after load_weights")
        bsz = int(layer0_code.shape[0])
        device = layer0_code.device
        predictor.set_sampling(wrapper._top_k, wrapper._top_p)
        if sample_uniforms is None:
            uniforms = torch.empty(bsz, HEADS, VOCAB, dtype=torch.float32, device=device)
            row_generators = wrapper._normalize_generators(generators if generators is not None else generator, bsz)
            if isinstance(row_generators, list):
                for row, g in enumerate(row_generators):
                    uniforms[row].uniform_(_UNIFORM_EPS, 1.0 - _UNIFORM_EPS, generator=g)
            else:
                uniforms.uniform_(_UNIFORM_EPS, 1.0 - _UNIFORM_EPS, generator=row_generators)
        else:
            uniforms = sample_uniforms.float().contiguous()
        hidden = last_talker_hidden.reshape(bsz, CODE_PREDICTOR_DIM).to(torch.bfloat16).contiguous()
        code0_embed = layer0_embed.reshape(bsz, CODE_PREDICTOR_DIM).to(torch.bfloat16).contiguous()
        codes, _ = predictor.launch(hidden, code0_embed, uniforms)

        all_codes = torch.cat([layer0_code.reshape(bsz, 1).long(), codes], dim=1)
        # The wrapper's buffer: every position's decoder input, and the last
        # code's embedding after them.
        proj_buf = torch.empty(bsz, POSITIONS + 1, CODE_PREDICTOR_DIM, dtype=torch.bfloat16, device=device)
        proj_buf[:, 0] = hidden
        proj_buf[:, 1] = code0_embed
        tables = torch.arange(HEADS, device=device)
        proj_buf[:, 2:] = predictor.weights.codec_embeddings[tables, codes]
        return all_codes.unsqueeze(-1), proj_buf
