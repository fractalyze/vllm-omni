# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The published Kandinsky 6 bundles must map onto the port's module tree.

``Kandinsky6TI2VAPipeline``'s module names were chosen to match a
"patched Diffusers" export, but the bundles actually published on the Hub
(`Kandinsky-6.0-Lite-5s-Diffusers`, `Kandinsky-6.0-Pro-distill-5s-Diffusers`)
name three families differently. Before ``_adapt_k6_weight_name`` handled
them, loading stopped on the first text block with "There is no module or
parameter named 'transformer.audio_text_transformer_blocks.0.attn'": 184 of
Lite's 2,399 transformer tensors and 344 of Pro-distill's 4,471 had no module
to land in.

The key names below are a property of the published bundles, so they are
written out here rather than read from a checkpoint: the test then needs no
70 GB download and still fails if the mapping regresses.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

# Lite's transformer/config.json (snapshot 6510114a). Pro-distill differs only
# in width and depth, and its names are identical.
_LITE_CONFIG = {
    "in_visual_dim": 16,
    "out_visual_dim": 16,
    "in_text_dim": 3584,
    "in_text_dim2": 768,
    "time_dim": 512,
    "patch_size": (1, 2, 2),
    "model_dim": 1792,
    "ff_dim": 7168,
    "num_text_blocks": 2,
    "num_visual_blocks": 2,  # 32 in the bundle; 2 is enough to exercise block 0
    "axes_dims": (16, 24, 24),
    "visual_cond": True,
    "is_multimodal": True,
    "in_audio_dim": 40,
    "model_dim_a": 896,
    "ff_dim_a": 3584,
    "axes_dims_a": (16, 24, 24),
    "audio_freqs_scaling": 0.144,
    "ca_rope": True,
    "cross_gates": True,
    "fix_modulation": True,
    "text_token_padding": True,
    "visual_token_type_num_embeddings": 2,
}

# One representative key per family the published bundles use, with the
# module path it has to reach.
_PUBLISHED_KEYS = [
    # 1. A text encoder block's attention is `attn`, not `self_attention`.
    (
        "transformer.audio_text_transformer_blocks.0.attn.to_query.weight",
        "audio_text_transformer_blocks.0.self_attention.to_query.weight",
    ),
    (
        "transformer.video_text_transformer_blocks.0.attn.key_norm.weight",
        "video_text_transformer_blocks.0.self_attention.key_norm.weight",
    ),
    (
        "transformer.audio_text_transformer_blocks.0.attn.out_layer.bias",
        "audio_text_transformer_blocks.0.self_attention.out_layer.bias",
    ),
    # 2. The feed-forward is Diffusers' FeedForward: net.0.proj and net.2.
    (
        "transformer.audio_text_transformer_blocks.0.feed_forward.net.0.proj.weight",
        "audio_text_transformer_blocks.0.feed_forward.in_layer.weight",
    ),
    (
        "transformer.audio_text_transformer_blocks.0.feed_forward.net.2.weight",
        "audio_text_transformer_blocks.0.feed_forward.out_layer.weight",
    ),
    (
        "transformer.visual_transformer_blocks.0.audio_dec_block.feed_forward.net.2.weight",
        "visual_transformer_blocks.0.audio_dec_block.feed_forward.out_layer.weight",
    ),
    # 3. The time embedding's MLP is timestep_embedder.linear_1/_2.
    (
        "transformer.video_time_embeddings.timestep_embedder.linear_1.weight",
        "video_time_embeddings.in_layer.weight",
    ),
    (
        "transformer.audio_time_embeddings.timestep_embedder.linear_2.bias",
        "audio_time_embeddings.out_layer.bias",
    ),
    # Already-matching names must pass through untouched.
    (
        "transformer.visual_transformer_blocks.0.video_dec_block.self_attention.to_key.weight",
        "visual_transformer_blocks.0.video_dec_block.self_attention.to_key.weight",
    ),
    (
        "transformer.visual_transformer_blocks.0.va_cross_attention.to_value.bias",
        "visual_transformer_blocks.0.va_cross_attention.to_value.bias",
    ),
]


@pytest.fixture
def transformer_param_names():
    """Every parameter name of a Lite-shaped transformer, built on `meta`.

    `meta` so a 3.7B-shaped module costs no memory: only the names matter.
    """
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed.parallel_state import (
        cleanup_dist_env_and_memory,
        init_distributed_environment,
        initialize_model_parallel,
        model_parallel_is_initialized,
    )

    from vllm_omni.diffusion.config import set_current_diffusion_config
    from vllm_omni.diffusion.data import AttentionConfig, AttentionSpec
    from vllm_omni.diffusion.models.kandinsky6 import Kandinsky6Transformer3DModel

    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29541")
    started = False
    with set_current_vllm_config(VllmConfig()):
        if not model_parallel_is_initialized():
            init_distributed_environment(world_size=1, rank=0, local_rank=0, distributed_init_method="env://")
            initialize_model_parallel()
            started = True
        diffusion_config = SimpleNamespace(
            diffusion_attention_config=AttentionConfig(default=AttentionSpec(backend="TORCH_SDPA")),
            parallel_config=SimpleNamespace(ring_degree=1, allgather_degree=1),
        )
        with torch.device("meta"), set_current_diffusion_config(diffusion_config):
            model = Kandinsky6Transformer3DModel.from_diffusers_config(dict(_LITE_CONFIG))
        yield set(dict(model.named_parameters()))
        if started:
            cleanup_dist_env_and_memory()


@pytest.mark.parametrize(("published", "expected"), _PUBLISHED_KEYS, ids=[k for k, _ in _PUBLISHED_KEYS])
def test_published_key_maps_to_an_existing_parameter(published, expected, transformer_param_names):
    from vllm_omni.diffusion.models.kandinsky6.pipeline_kandinsky6 import _adapt_k6_weight_name

    mapped = _adapt_k6_weight_name(published)
    assert mapped == f"transformer.{expected}"
    # The mapping being "right" means landing on a parameter that exists. A
    # rule that produced a plausible-looking path with no slot would still
    # fail the loader at run time.
    assert expected in transformer_param_names, f"{mapped} is not a parameter of the transformer"


def test_pre_rename_block_names_still_map():
    """The older `videoT`/`audioT` export is still handled."""
    from vllm_omni.diffusion.models.kandinsky6.pipeline_kandinsky6 import _adapt_k6_weight_name

    assert (
        _adapt_k6_weight_name("transformer.visual_transformer_blocks.0.videoT.self_attention.to_key.weight")
        == "transformer.visual_transformer_blocks.0.video_dec_block.self_attention.to_key.weight"
    )


def test_non_transformer_prefixes_are_untouched_by_the_new_rules():
    """The three rules are scoped to `transformer.`: the text encoder has its
    own `attn` modules and a `feed_forward`, and renaming those would break
    Qwen2.5-VL loading."""
    from vllm_omni.diffusion.models.kandinsky6.pipeline_kandinsky6 import _adapt_k6_weight_name

    name = "text_encoder.model.language_model.layers.0.self_attn.q_proj.weight"
    assert "attn" in _adapt_k6_weight_name(name)
    assert _adapt_k6_weight_name("vae.decoder.conv_in.weight") == "vae.decoder.conv_in.weight"
