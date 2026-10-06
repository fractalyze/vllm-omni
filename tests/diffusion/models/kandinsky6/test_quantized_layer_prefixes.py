# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Layer prefixes must be matchable by a quantization config's ``ignored_layers``.

A quantization config names the layers it must *not* quantize by their module
path, and vLLM's ``fp8`` method compares those names to each layer's ``prefix``
exactly. This DiT is built with an empty root prefix, and an unguarded
``f"{prefix}.{name}"`` then yields ``.video_text_embeddings.in_layer`` -- a
leading dot that appears in no checkpoint. Nothing raises: the layer is simply
built as an FP8 layer, its ``weight_scale`` is absent from a checkpoint that
meant it to stay BF16, and that parameter keeps the contents of the
``torch.empty`` it was allocated with. The first forward then returns NaN, the
VAE clamps NaN to zero, and the server answers with a black video and
``status: completed``.

That is a silent, whole-model failure produced by one character, which is why it
gets a test of its own: a missing *weight* raises in the loader, a missing
*scale* does not.
"""

from __future__ import annotations

import os

import torch
from absl.testing import absltest, parameterized

from vllm_omni.diffusion.models.kandinsky6.kandinsky6_transformer import child_prefix

_MASTER_PORT = "29583"

# Lite-distill's shape, shrunk in depth. The families that matter are the ones
# whose prefixes were built unguarded: the embeddings, the output heads and the
# per-block modules.
TINY_CONFIG = {
    "model_dim": 64,
    "time_dim": 32,
    "ff_dim": 128,
    "num_text_blocks": 1,
    "num_visual_blocks": 2,
    "patch_size": [1, 2, 2],
    "axes_dims": [8, 12, 12],
    "in_visual_dim": 4,
    "out_visual_dim": 16,
    "in_text_dim": 16,
    "in_text_dim2": 8,
    "visual_cond": True,
    "is_multimodal": True,
    "in_audio_dim": 6,
    "out_audio_dim": 24,
    "model_dim_a": 32,
    "time_dim_a": 32,
    "ff_dim_a": 64,
    "axes_dims_a": [8, 12, 12],
    "audio_freqs_scaling": 0.144,
    "text_token_padding": True,
    "ca_rope": True,
    "cross_gates": True,
    "fix_modulation": True,
    "visual_token_type_num_embeddings": 2,
}


def _init_single_rank() -> None:
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import init_distributed_environment, initialize_model_parallel
    from vllm.distributed.parallel_state import model_parallel_is_initialized

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", _MASTER_PORT)
    if model_parallel_is_initialized():
        return
    with set_current_vllm_config(VllmConfig()):
        init_distributed_environment(
            world_size=1,
            rank=0,
            distributed_init_method=f"tcp://127.0.0.1:{_MASTER_PORT}",
            local_rank=0,
            backend="gloo",
        )
        initialize_model_parallel(1, 1)


class ChildPrefixTest(parameterized.TestCase):
    @parameterized.named_parameters(
        ("empty_root", "", "in_layer", "in_layer"),
        ("nested_root", "transformer", "in_layer", "transformer.in_layer"),
        ("deep_root", "a.b", "c", "a.b.c"),
    )
    def test_no_leading_dot(self, prefix: str, name: str, expected: str) -> None:
        self.assertEqual(child_prefix(prefix, name), expected)


class LayerPrefixTest(absltest.TestCase):
    """Every built layer's prefix must be a plausible checkpoint path."""

    def setUp(self) -> None:
        super().setUp()
        _init_single_rank()
        from diffusers.models.modeling_utils import no_init_weights

        from vllm_omni.diffusion.models.kandinsky6.kandinsky6_transformer import (
            Kandinsky6Transformer3DModel,
        )

        with torch.device("meta"), no_init_weights():
            self.dit = Kandinsky6Transformer3DModel.from_diffusers_config(dict(TINY_CONFIG))

    def _linear_prefixes(self) -> list[str]:
        from vllm.model_executor.layers.linear import LinearBase

        return [getattr(m, "prefix", "") for n, m in self.dit.named_modules() if isinstance(m, LinearBase)]

    def test_no_prefix_starts_with_a_dot(self) -> None:
        offenders = [p for p in self._linear_prefixes() if p.startswith(".")]
        self.assertEqual(offenders, [], f"{len(offenders)} layer prefixes start with '.'")

    def test_every_prefix_matches_its_module_path(self) -> None:
        """A layer's prefix must equal the dotted path it actually sits at.

        That equality is what makes a checkpoint's ``ignored_layers`` -- written
        from the checkpoint's own key names -- match the layers it names.
        """
        from vllm.model_executor.layers.linear import LinearBase

        mismatched = [
            (name, module.prefix)
            for name, module in self.dit.named_modules()
            if isinstance(module, LinearBase) and getattr(module, "prefix", None) != name
        ]
        self.assertEqual(mismatched, [], f"{len(mismatched)} layers' prefix differs from their module path")

    def test_a_kept_layer_is_actually_skipped(self) -> None:
        """The end-to-end property: naming a layer in ``ignored_layers`` keeps it wide.

        This is the assertion that would have caught the bug. Checking the prefix
        strings alone leaves open whether the quantization config agrees with
        them, and it is the config's verdict that decides whether a layer gets a
        ``weight_scale`` the checkpoint has to supply.
        """
        from diffusers.models.modeling_utils import no_init_weights
        from vllm.model_executor.layers.quantization import get_quantization_config

        from vllm_omni.diffusion.models.kandinsky6.kandinsky6_transformer import (
            Kandinsky6Transformer3DModel,
        )

        kept = "video_text_embeddings.in_layer"
        quant_config = get_quantization_config("fp8").from_config(
            {
                "quant_method": "fp8",
                "activation_scheme": "dynamic",
                "weight_block_size": None,
                "ignored_layers": [kept],
            }
        )
        # Fp8LinearMethod reads its activation dtype from the ambient vLLM config
        # at construction, so the build has to happen inside one.
        from types import SimpleNamespace

        from vllm.config import VllmConfig, set_current_vllm_config

        config = VllmConfig()
        config.model_config = SimpleNamespace(dtype=torch.bfloat16, is_moe=False)
        with set_current_vllm_config(config), torch.device("meta"), no_init_weights():
            dit = Kandinsky6Transformer3DModel.from_diffusers_config(
                dict(TINY_CONFIG), quant_config=quant_config
            )

        params = {name for name, _ in dit.named_parameters()}
        self.assertIn(f"{kept}.weight", params)
        self.assertNotIn(
            f"{kept}.weight_scale",
            params,
            "a layer named in ignored_layers was still built as an FP8 layer, so its "
            "weight_scale would stay at the contents of its torch.empty allocation",
        )
        # A layer *not* named must still be quantized, or the test would pass for
        # a config that was ignored entirely.
        quantized = "visual_transformer_blocks.0.video_dec_block.self_attention.to_query"
        self.assertIn(f"{quantized}.weight_scale", params)


if __name__ == "__main__":
    absltest.main()
