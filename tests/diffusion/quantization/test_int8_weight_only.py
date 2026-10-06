# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Weight-only INT8: INT8 storage and transfer, BF16 compute.

sm_120 has no INT8 GEMM, so the INT8 checkpoint written by
``showcase/kandinsky6/tools/quantize_dit_fp8.py --format int8`` is served by
``Int8WeightOnlyLinearMethod``: the weight crosses the bus as INT8 plus one FP32
scale per output row and is dequantized on the device into the BF16 operand of
an ordinary GEMM. The contract these tests pin is that this is *exactly* a BF16
model whose weights were fake-quantized on the host -- bit for bit -- so any
quality difference measured against BF16 is the INT8 rounding and nothing else.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import torch
from absl.testing import absltest

from vllm_omni.diffusion.data import TransformerConfig
from vllm_omni.quantization.int8_config import (
    DiffusionInt8Config,
    Int8WeightOnlyLinearMethod,
    dequantize_int8_rows,
)

_REPO = Path(__file__).resolve().parents[3]
_MASTER_PORT = "29587"

INT8_DECLARED = {
    "in_visual_dim": 16,
    "quantization_config": {
        "quant_method": "int8",
        "weight_only": True,
        "ignored_layers": ["out_layer.out_layer"],
    },
}

# Multimodal, so the fused video/audio blocks and both cross-attentions run.
_TINY_T2VA_CONFIG = {
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
    "is_multimodal": True,
    "in_audio_dim": 6,
    "model_dim_a": 12,
    "ff_dim_a": 16,
    "axes_dims_a": (2, 2, 2),
    "attention_engine": "sdpa",
}


def _load_quantizer():
    """The checkpoint writer, loaded from the showcase tree it lives in."""
    path = _REPO / "showcase" / "kandinsky6" / "tools" / "quantize_dit_fp8.py"
    spec = importlib.util.spec_from_file_location("quantize_dit_fp8", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _init_single_rank(test: absltest.TestCase) -> None:
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import init_distributed_environment, initialize_model_parallel
    from vllm.distributed.parallel_state import cleanup_dist_env_and_memory, model_parallel_is_initialized

    if model_parallel_is_initialized():
        return
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    with set_current_vllm_config(VllmConfig()):
        init_distributed_environment(
            world_size=1,
            rank=0,
            distributed_init_method=f"tcp://127.0.0.1:{_MASTER_PORT}",
            local_rank=0,
            backend="gloo",
        )
        initialize_model_parallel(1, 1)
    test.addCleanup(cleanup_dist_env_and_memory)


class DiskDeclaredInt8Test(absltest.TestCase):
    def test_resolves_to_weight_only_serialized_config(self) -> None:
        config = TransformerConfig.from_dict(INT8_DECLARED)
        self.assertIsInstance(config.quant_config, DiffusionInt8Config)
        self.assertTrue(config.quant_config.is_checkpoint_int8_serialized)
        self.assertTrue(config.quant_config.weight_only)
        self.assertEqual(config.quant_config.ignored_layers, ["out_layer.out_layer"])

    def test_kandinsky6_resolver_accepts_int8(self) -> None:
        # vLLM's own registry has no "int8"; the K6 pipeline must not ask it.
        from vllm_omni.diffusion.models.kandinsky6.pipeline_kandinsky6 import _resolve_quant_config

        od_config = SimpleNamespace(quantization_config=None, quantization_config_is_auto_detected=False)
        resolved = _resolve_quant_config(od_config, INT8_DECLARED)
        self.assertIsInstance(resolved, DiffusionInt8Config)
        self.assertTrue(resolved.weight_only)

    def test_weight_only_needs_a_serialized_checkpoint(self) -> None:
        with self.assertRaises(ValueError):
            DiffusionInt8Config(is_checkpoint_int8_serialized=False, weight_only=True)


class QuantizeDequantizeTest(absltest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.quantizer = _load_quantizer()
        generator = torch.Generator().manual_seed(0)
        # One Pro-block-sized projection's shape class, with rows of very
        # different range so a per-tensor scale would visibly fail.
        self.weight = (torch.randn(256, 512, generator=generator) * torch.logspace(-3, 0, 256).unsqueeze(1)).to(
            torch.bfloat16
        )

    def test_round_trip_error_is_at_most_half_a_step(self) -> None:
        q, scale = self.quantizer.quantize_weight_int8(self.weight)
        self.assertEqual(q.dtype, torch.int8)
        self.assertEqual(scale.dtype, torch.float32)
        self.assertEqual(tuple(scale.shape), (256, 1))
        self.assertLessEqual(int(q.abs().max()), 127)
        error = (q.float() * scale - self.weight.float()).abs()
        # Round to nearest: at most half a quantization step, per row, plus the
        # FP32 rounding of w / scale (~1e-5 of a step near +-127).
        self.assertTrue(bool((error <= scale / 2 * (1 + 1e-4)).all()))

    def test_zero_row_quantizes_to_zeros(self) -> None:
        weight = self.weight.clone()
        weight[3] = 0
        q, scale = self.quantizer.quantize_weight_int8(weight)
        self.assertTrue(bool((q[3] == 0).all()))
        self.assertEqual(float(scale[3]), 1.0)

    def test_device_dequant_matches_cpu_bit_for_bit(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("needs a CUDA device")
        q, scale = self.quantizer.quantize_weight_int8(self.weight)
        on_host = dequantize_int8_rows(q, scale, torch.bfloat16)
        on_device = dequantize_int8_rows(q.cuda(), scale.cuda(), torch.bfloat16).cpu()
        self.assertTrue(torch.equal(on_host, on_device))


class WeightOnlyModelTest(absltest.TestCase):
    """The weight-only INT8 DiT computes exactly what a fake-quantized BF16 DiT computes."""

    def setUp(self) -> None:
        super().setUp()
        _init_single_rank(self)

    def _build(self, quant_config):
        from vllm_omni.diffusion.config import set_current_diffusion_config
        from vllm_omni.diffusion.data import AttentionConfig, AttentionSpec
        from vllm_omni.diffusion.models.kandinsky6 import Kandinsky6Transformer3DModel

        attention = SimpleNamespace(
            diffusion_attention_config=AttentionConfig(default=AttentionSpec(backend="TORCH_SDPA")),
            parallel_config=SimpleNamespace(ring_degree=1, allgather_degree=1),
        )
        previous = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            with set_current_diffusion_config(attention):
                model = Kandinsky6Transformer3DModel(**_TINY_T2VA_CONFIG, quant_config=quant_config)
        finally:
            torch.set_default_dtype(previous)
        return model.eval()

    def _forward(self, model):
        generator = torch.Generator().manual_seed(1)
        frames, height, width, audio_len = 2, 4, 4, 7

        def rand(*shape):
            return torch.randn(*shape, generator=generator).to(torch.bfloat16)

        text_embed, pooled = rand(5, 8), rand(1, 6)
        time = torch.tensor([500.0])
        visual_rope = model.visual_rope_embeddings(
            shape=(frames, height // 2, width // 2),
            pos=[torch.arange(frames), torch.arange(height // 2), torch.arange(width // 2)],
            scale_factor=(1.0, 1.0, 1.0),
        )
        with torch.no_grad():
            return model(
                x_video=rand(frames, height, width, 4),
                x_audio=rand(audio_len, 6),
                text_embed=[text_embed, text_embed],
                pooled_text_embed=[pooled, pooled],
                time=[time, time],
                visual_rope=visual_rope,
                audio_rope=model.audio_rope_embeddings(torch.arange(audio_len)),
                text_rope=[
                    model.video_text_rope_embeddings(torch.arange(5)),
                    model.audio_text_rope_embeddings(torch.arange(5)),
                ],
            )

    def test_matches_fake_quantized_bf16_model(self) -> None:
        quantizer = _load_quantizer()
        reference = self._build(quant_config=None)
        int8 = self._build(quant_config=DiffusionInt8Config(is_checkpoint_int8_serialized=True, weight_only=True))

        generator = torch.Generator().manual_seed(2)
        quantized_layers = {
            name
            for name, module in int8.named_modules()
            if isinstance(getattr(module, "quant_method", None), Int8WeightOnlyLinearMethod)
        }
        self.assertTrue(quantized_layers, "no layer was built weight-only INT8")

        reference_params = dict(reference.named_parameters())
        int8_params = dict(int8.named_parameters())
        with torch.no_grad():
            for name, param in reference_params.items():
                # Nonzero everywhere: the modulation projections are
                # zero-initialized, which would make most of the model inert.
                param.copy_(torch.randn(param.shape, generator=generator) * 0.1)
                layer = name.removesuffix(".weight")
                if layer in quantized_layers and name.endswith(".weight"):
                    q, scale = quantizer.quantize_weight_int8(param)
                    param.copy_(dequantize_int8_rows(q, scale, torch.bfloat16))
                    int8_params[name].copy_(q)
                    int8_params[f"{layer}.weight_scale"].copy_(scale)
                else:
                    int8_params[name].copy_(param)
        for module in int8.modules():
            method = getattr(module, "quant_method", None)
            if isinstance(method, Int8WeightOnlyLinearMethod):
                method.process_weights_after_loading(module)
                self.assertEqual(module.weight.dtype, torch.int8)
                self.assertEqual(module.weight_scale.dtype, torch.float32)

        video_ref, audio_ref = self._forward(reference)
        video_int8, audio_int8 = self._forward(int8)
        self.assertTrue(torch.equal(video_ref, video_int8))
        self.assertTrue(torch.equal(audio_ref, audio_int8))


class ScaleSourceDtypeTest(absltest.TestCase):
    def test_scales_keep_fp32(self) -> None:
        from vllm_omni.diffusion.models.kandinsky6.pipeline_kandinsky6 import _normalize_quantized_source

        scale = torch.tensor([[1.0 + 2**-12]], dtype=torch.float32)  # not representable in BF16
        self.assertIs(_normalize_quantized_source("blocks.0.ff.weight_scale", scale), scale)

    def test_other_fp32_tensors_become_bf16(self) -> None:
        from vllm_omni.diffusion.models.kandinsky6.pipeline_kandinsky6 import _normalize_quantized_source

        norm = torch.ones(4, dtype=torch.float32)
        self.assertEqual(_normalize_quantized_source("blocks.0.norm.weight", norm).dtype, torch.bfloat16)

    def test_int8_weights_pass_through(self) -> None:
        from vllm_omni.diffusion.models.kandinsky6.pipeline_kandinsky6 import _normalize_quantized_source

        weight = torch.zeros(2, 2, dtype=torch.int8)
        self.assertIs(_normalize_quantized_source("blocks.0.ff.weight", weight), weight)


if __name__ == "__main__":
    absltest.main()
