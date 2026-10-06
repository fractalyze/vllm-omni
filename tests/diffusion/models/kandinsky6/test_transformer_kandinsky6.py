# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import os
from types import SimpleNamespace

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.fixture(autouse=True)
def _init_distributed(monkeypatch):
    """The native transformer uses vLLM parallel linear layers, which require a
    tensor-parallel group; initialize a single-process group for CPU tests
    (mirrors tests/diffusion/models/sana_video/test_transformer_sana_video.py)."""
    from vllm.distributed.parallel_state import (
        cleanup_dist_env_and_memory,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.model_executor.layers.utils import default_unquantized_gemm

    monkeypatch.setattr(
        "vllm.model_executor.layers.linear.dispatch_unquantized_gemm",
        lambda *_args, **_kwargs: default_unquantized_gemm,
    )
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29502")
    init_distributed_environment(world_size=1, rank=0, local_rank=0, distributed_init_method="env://")
    initialize_model_parallel()
    yield
    cleanup_dist_env_and_memory()


# model_dim must be an exact multiple of head_dim = sum(axes_dims); this is
# an architectural constraint of the native model (DiffusionTransformer3D),
# not specific to this port.
_TINY_T2V_CONFIG = {
    "in_visual_dim": 4,
    "out_visual_dim": 4,
    "in_text_dim": 8,
    "in_text_dim2": 6,
    "time_dim": 16,
    "patch_size": (1, 2, 2),
    "model_dim": 24,
    "ff_dim": 32,
    "num_text_blocks": 1,
    "num_visual_blocks": 2,
    "axes_dims": (4, 4, 4),
    "visual_cond": False,
    "is_multimodal": False,
    "attention_engine": "sdpa",
}

# time_dim_a is intentionally omitted (defaults to time_dim): the fused
# block's cross-modal modulation (va_mod/av_mod) consumes the *other*
# modality's time embedding by default (fix_modulation=False), so time_dim
# and time_dim_a must match unless fix_modulation=True.
_TINY_T2VA_CONFIG = {
    **_TINY_T2V_CONFIG,
    "is_multimodal": True,
    "in_audio_dim": 6,
    "model_dim_a": 12,
    "ff_dim_a": 16,
    "axes_dims_a": (2, 2, 2),
}


def _sdpa_config():
    from vllm_omni.diffusion.data import AttentionConfig, AttentionSpec

    return SimpleNamespace(
        diffusion_attention_config=AttentionConfig(default=AttentionSpec(backend="TORCH_SDPA")),
        parallel_config=SimpleNamespace(ring_degree=1, allgather_degree=1),
    )


def _build_transformer(config):
    from vllm_omni.diffusion.config import set_current_diffusion_config
    from vllm_omni.diffusion.models.kandinsky6 import Kandinsky6Transformer3DModel

    with set_current_diffusion_config(_sdpa_config()):
        model = Kandinsky6Transformer3DModel(**config)
    model.eval()
    return model


def test_t2v_only_forward_shape():
    model = _build_transformer(_TINY_T2V_CONFIG)
    time_frames, height, width = 2, 4, 4
    x_video = torch.randn(time_frames, height, width, 4)
    text_embed = torch.randn(5, 8)
    pooled = torch.randn(1, 6)
    time = torch.tensor([500.0])
    visual_rope = model.visual_rope_embeddings(
        shape=(time_frames, height // 2, width // 2),
        pos=[torch.arange(time_frames), torch.arange(height // 2), torch.arange(width // 2)],
        scale_factor=(1.0, 1.0, 1.0),
    )
    text_rope = model.text_rope_embeddings(torch.arange(5))

    with torch.no_grad():
        out = model(
            x_video=x_video,
            x_audio=None,
            text_embed=text_embed,
            pooled_text_embed=pooled,
            time=time,
            visual_rope=visual_rope,
            audio_rope=None,
            text_rope=text_rope,
        )

    assert out.shape == (time_frames, height, width, 4)


def test_visual_cond_channel_concat_path():
    """visual_cond=True doubles+1 the visual input channel count (cond +
    mask channels appended by _build_video_input, stripped again by
    OutLayer) — the I2V conditioning path."""
    config = {**_TINY_T2V_CONFIG, "visual_cond": True}
    model = _build_transformer(config)
    time_frames, height, width = 2, 4, 4
    x_video = torch.randn(time_frames, height, width, 2 * 4 + 1)
    text_embed = torch.randn(5, 8)
    pooled = torch.randn(1, 6)
    time = torch.tensor([500.0])
    visual_rope = model.visual_rope_embeddings(
        shape=(time_frames, height // 2, width // 2),
        pos=[torch.arange(time_frames), torch.arange(height // 2), torch.arange(width // 2)],
        scale_factor=(1.0, 1.0, 1.0),
    )
    text_rope = model.text_rope_embeddings(torch.arange(5))

    with torch.no_grad():
        out = model(
            x_video=x_video,
            x_audio=None,
            text_embed=text_embed,
            pooled_text_embed=pooled,
            time=time,
            visual_rope=visual_rope,
            audio_rope=None,
            text_rope=text_rope,
        )

    assert out.shape == (time_frames, height, width, 4)


def test_t2va_joint_forward_shapes():
    """Exercises the fused cross-modal decoder blocks (Kandinsky6FusedTransformerDecoderBlock)."""
    model = _build_transformer(_TINY_T2VA_CONFIG)
    time_frames, height, width = 2, 4, 4
    audio_len = 7
    x_video = torch.randn(time_frames, height, width, 4)
    x_audio = torch.randn(audio_len, 6)
    text_embed = torch.randn(5, 8)
    pooled = torch.randn(1, 6)
    time = torch.tensor([500.0])
    visual_rope = model.visual_rope_embeddings(
        shape=(time_frames, height // 2, width // 2),
        pos=[torch.arange(time_frames), torch.arange(height // 2), torch.arange(width // 2)],
        scale_factor=(1.0, 1.0, 1.0),
    )
    audio_rope = model.audio_rope_embeddings(torch.arange(audio_len))
    video_text_rope = model.video_text_rope_embeddings(torch.arange(5))
    audio_text_rope = model.audio_text_rope_embeddings(torch.arange(5))

    with torch.no_grad():
        video_out, audio_out = model(
            x_video=x_video,
            x_audio=x_audio,
            text_embed=[text_embed, text_embed],
            pooled_text_embed=[pooled, pooled],
            time=[time, time],
            visual_rope=visual_rope,
            audio_rope=audio_rope,
            text_rope=[video_text_rope, audio_text_rope],
        )

    assert video_out.shape == (time_frames, height, width, 4)
    assert audio_out.shape == (audio_len, 6)


@pytest.mark.parametrize("modality", ["video", "audio"])
def test_single_modality_through_multimodal_model(modality):
    """A multimodal (is_multimodal=True) model must still support a pure
    video-only or audio-only forward call (denoise_loop's partial-sampling
    path uses this)."""
    model = _build_transformer(_TINY_T2VA_CONFIG)
    time_frames, height, width = 2, 4, 4
    audio_len = 7
    text_embed = torch.randn(5, 8)
    pooled = torch.randn(1, 6)
    time = torch.tensor([500.0])

    if modality == "video":
        x_video = torch.randn(time_frames, height, width, 4)
        visual_rope = model.visual_rope_embeddings(
            shape=(time_frames, height // 2, width // 2),
            pos=[torch.arange(time_frames), torch.arange(height // 2), torch.arange(width // 2)],
            scale_factor=(1.0, 1.0, 1.0),
        )
        text_rope = model.video_text_rope_embeddings(torch.arange(5))
        with torch.no_grad():
            out = model(
                x_video=x_video,
                x_audio=None,
                text_embed=text_embed,
                pooled_text_embed=pooled,
                time=time,
                visual_rope=visual_rope,
                audio_rope=None,
                text_rope=text_rope,
            )
        assert out.shape == (time_frames, height, width, 4)
    else:
        x_audio = torch.randn(audio_len, 6)
        audio_rope = model.audio_rope_embeddings(torch.arange(audio_len))
        text_rope = model.audio_text_rope_embeddings(torch.arange(5))
        with torch.no_grad():
            out = model(
                x_video=None,
                x_audio=x_audio,
                text_embed=text_embed,
                pooled_text_embed=pooled,
                time=time,
                visual_rope=None,
                audio_rope=audio_rope,
                text_rope=text_rope,
            )
        assert out.shape == (audio_len, 6)


@pytest.mark.parametrize("batched", [False, True])
def test_self_attention_attends_over_tokens_for_batched_and_unbatched_inputs(batched):
    """Self-attention must contract over the *token* axis whether its input is
    unbatched text ``(S, C)`` or batched=1 visual/audio ``(1, N, C)``.

    Regression: an unconditional ``unsqueeze(0)`` on the already-batched
    visual path handed SDPA a 5-D ``(1, 1, N, H, D)`` tensor, which it
    accepts silently but then treats the head axis as the sequence — every
    generated video decoded to pure noise."""
    from vllm_omni.diffusion.config import set_current_diffusion_config
    from vllm_omni.diffusion.models.kandinsky6.kandinsky6_transformer import Kandinsky6Attention, apply_rotary

    torch.manual_seed(0)
    channels, head_dim, tokens = 24, 12, 7
    with set_current_diffusion_config(_sdpa_config()):
        attn = Kandinsky6Attention(channels, head_dim, engine="sdpa").eval()
    with torch.no_grad():
        # vLLM parallel linears allocate with torch.empty (no init).
        for param in attn.parameters():
            param.normal_(std=0.2)
    x = torch.randn(tokens, channels)
    rope = torch.randn(tokens, 1, head_dim // 2, 2, 2)
    hidden = x.unsqueeze(0) if batched else x

    with torch.no_grad():
        out = attn(hidden, rotary_emb=rope)

        heads = channels // head_dim
        # The projections are `skip_bias_add=True`, so each returns
        # (output, bias) and `_add_bias` applies it -- see that method.
        q = attn.query_norm(attn._add_bias(attn.to_query(x)).reshape(tokens, heads, head_dim))
        k = attn.key_norm(attn._add_bias(attn.to_key(x)).reshape(tokens, heads, head_dim))
        v = attn._add_bias(attn.to_value(x)).reshape(tokens, heads, head_dim)
        q = apply_rotary(q, rope).type_as(q)
        k = apply_rotary(k, rope).type_as(k)
        ref = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)
        ).transpose(0, 1)
        expected = attn._add_bias(attn.out_layer(ref.reshape(tokens, channels)))

    assert out.shape == (hidden.shape[:-1] + (channels,))
    torch.testing.assert_close(out.reshape(tokens, channels), expected, rtol=1e-4, atol=1e-5)


def test_attention_biases_are_added_once_outside_the_gemm():
    """The four attention projections skip cuBLAS's bias epilogue and add the
    bias themselves, and the result must be unchanged by that.

    Why the projections are built this way: on sm_120 an `addmm` carrying a
    bias is 21.5% slower than the same GEMM without one, while adding the
    identical bias as a separate elementwise kernel costs 7.4% -- cuBLAS
    dispatches a worse kernel when a bias is present. At W1's 50,220 visual
    tokens that is about 4 s of a 183 s request.

    The regression this guards is the cheap one: `skip_bias_add=True` makes the
    layer *return* the bias instead of applying it, so a call site that forgets
    `_add_bias` silently drops the bias entirely and still produces
    correctly-shaped output. Comparing against the fused configuration catches
    both a dropped bias and a doubly-applied one."""
    from vllm_omni.diffusion.config import set_current_diffusion_config
    from vllm_omni.diffusion.models.kandinsky6.kandinsky6_transformer import Kandinsky6Attention

    torch.manual_seed(0)
    channels, head_dim, tokens = 24, 12, 7
    with set_current_diffusion_config(_sdpa_config()):
        attn = Kandinsky6Attention(channels, head_dim, engine="sdpa").eval()
    with torch.no_grad():
        for param in attn.parameters():
            param.normal_(std=0.2)

    projections = (attn.to_query, attn.to_key, attn.to_value, attn.out_layer)
    for layer in projections:
        assert layer.skip_bias_add, "the GEMM must not fold the bias in"
        assert layer.return_bias, "the layer must hand the bias back to be added"
        assert layer.bias is not None, "there is still a bias to add"

    x = torch.randn(tokens, channels)
    rope = torch.randn(tokens, 1, head_dim // 2, 2, 2)
    with torch.no_grad():
        out = attn(x, rotary_emb=rope)
        # The configuration this replaced: cuBLAS folds the bias into the GEMM
        # and `_add_bias` has nothing left to do.
        for layer in projections:
            layer.skip_bias_add = False
        fused = attn(x, rotary_emb=rope)

    torch.testing.assert_close(out, fused, rtol=1e-5, atol=1e-6)

    # And the bias is genuinely doing something, so the comparison above is not
    # two zeros agreeing.
    with torch.no_grad():
        for layer in projections:
            layer.skip_bias_add = True
            layer.bias.zero_()
        unbiased = attn(x, rotary_emb=rope)
    assert not torch.allclose(out, unbiased, rtol=1e-3, atol=1e-4)


def test_modulation_projections_are_zero_initialized():
    """Modulation layers are zero-initialized at construction (standard
    AdaLN-zero init) — confirms the zero-init in Kandinsky6Modulation.__init__
    reaches the tensor-parallel out_layer correctly (it's a
    ColumnParallelLinear, not plain nn.Linear)."""
    model = _build_transformer(_TINY_T2V_CONFIG)
    for block in model.text_transformer_blocks:
        assert torch.all(block.text_modulation.out_layer.weight == 0)
    for block in model.visual_transformer_blocks:
        assert torch.all(block.visual_modulation.out_layer.weight == 0)


def test_dit_constructs_from_a_diffusers_config_dict():
    from vllm_omni.diffusion.config import set_current_diffusion_config
    from vllm_omni.diffusion.models.kandinsky6 import Kandinsky6Transformer3DModel

    with set_current_diffusion_config(_sdpa_config()):
        model = Kandinsky6Transformer3DModel.from_diffusers_config(
            {**_TINY_T2VA_CONFIG, "patch_size": [1, 2, 2], "axes_dims": [4, 4, 4]}
        )
    assert tuple(model.patch_size) == (1, 2, 2)
    assert model._repeated_blocks == [
        "Kandinsky6FusedTransformerDecoderBlock",
        "Kandinsky6TransformerEncoderBlock",
        "Kandinsky6TransformerDecoderBlock",
    ]
    visual = model.visual_transformer_blocks[0].video_dec_block.self_attention.attn
    audio = model.visual_transformer_blocks[0].audio_dec_block.self_attention.attn
    text = model.video_text_transformer_blocks[0].self_attention.attn
    cross = model.visual_transformer_blocks[0].va_cross_attention.attn
    assert visual.role == "kandinsky6.visual_self"
    assert visual.skip_sequence_parallel is False
    assert audio.skip_sequence_parallel is True
    assert text.role == "kandinsky6.text_self"
    assert text.skip_sequence_parallel is True
    assert cross.role == "kandinsky6.video_audio_cross"
    assert cross.skip_sequence_parallel is True


def test_cfg_combine_matches_apply_cfg_without_normalization():
    from vllm_omni.diffusion.distributed.cfg_parallel import CFGParallelMixin
    from vllm_omni.diffusion.models.kandinsky6.pipeline_kandinsky6 import apply_cfg

    positive = torch.tensor([1.0, 3.0])
    negative = torch.tensor([0.25, -1.0])
    combined = CFGParallelMixin.combine_cfg_noise(object(), positive, negative, 5.0, False)
    torch.testing.assert_close(combined, apply_cfg(positive, negative, 5.0))
