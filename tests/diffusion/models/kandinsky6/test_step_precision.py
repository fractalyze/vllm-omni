# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Step-scheduled GEMM precision: exact steps are untouched, FP8 steps stay within FP8 error."""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest import mock

import torch
from absl.testing import absltest

from vllm_omni.diffusion.models.kandinsky6.step_precision import (
    StepFp8LinearMethod,
    fp8_after_step,
    install_step_fp8,
    nvfp4_global_scale,
    quantize_weight_per_tensor,
    set_fp8_gemm_step,
    step_gemm_format,
)

_MASTER_PORT = "29589"

_TINY = {
    "in_visual_dim": 4,
    "out_visual_dim": 4,
    "in_text_dim": 8,
    "in_text_dim2": 6,
    "time_dim": 16,
    "patch_size": (1, 2, 2),
    "model_dim": 32,
    "ff_dim": 64,
    "num_text_blocks": 1,
    "num_visual_blocks": 2,
    "axes_dims": (8, 4, 4),
    "visual_cond": False,
    "is_multimodal": False,
    "attention_engine": "sdpa",
}


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


class SwitchTest(absltest.TestCase):
    def test_off_by_default(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("VLLM_OMNI_K6_FP8_GEMM_AFTER_STEP", None)
            self.assertEqual(fp8_after_step(), 0)

    def test_reads_the_step(self) -> None:
        with mock.patch.dict(os.environ, {"VLLM_OMNI_K6_FP8_GEMM_AFTER_STEP": "1"}):
            self.assertEqual(fp8_after_step(), 1)


class QuantizeTest(absltest.TestCase):
    def test_per_tensor_round_trip(self) -> None:
        weight = (torch.randn(64, 128, generator=torch.Generator().manual_seed(0)) * 0.05).to(torch.bfloat16)
        quantized, scale = quantize_weight_per_tensor(weight)
        self.assertEqual(quantized.dtype, torch.float8_e4m3fn)
        self.assertEqual(tuple(scale.shape), (1,))
        relative = (quantized.float() * scale - weight.float()).norm() / weight.float().norm()
        self.assertLess(float(relative), 0.05)

    def test_zero_weight(self) -> None:
        quantized, scale = quantize_weight_per_tensor(torch.zeros(16, 16, dtype=torch.bfloat16))
        self.assertEqual(float(scale), 1.0)
        self.assertEqual(float(quantized.float().abs().sum()), 0.0)


class InstallTest(absltest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        _init_single_rank(self)

    def _dit(self):
        from vllm_omni.diffusion.config import set_current_diffusion_config
        from vllm_omni.diffusion.data import AttentionConfig, AttentionSpec
        from vllm_omni.diffusion.models.kandinsky6 import Kandinsky6Transformer3DModel

        attention = SimpleNamespace(
            diffusion_attention_config=AttentionConfig(default=AttentionSpec(backend="TORCH_SDPA")),
            parallel_config=SimpleNamespace(ring_degree=1, allgather_degree=1),
        )
        with set_current_diffusion_config(attention):
            return Kandinsky6Transformer3DModel(**_TINY)

    def test_wraps_only_visual_block_linears_and_the_flag_reaches_them(self) -> None:
        from vllm.model_executor.layers.linear import LinearBase

        dit = self._dit()
        wrapped = install_step_fp8(dit)
        self.assertGreater(wrapped, 0)
        for name, module in dit.named_modules():
            if isinstance(module, LinearBase):
                is_step = isinstance(module.quant_method, StepFp8LinearMethod)
                self.assertEqual(is_step, name.startswith("visual_transformer_blocks."), name)
        set_fp8_gemm_step(dit, True)
        flags = {m.fp8_step for m in dit.modules() if isinstance(getattr(m, "quant_method", None), StepFp8LinearMethod)}
        self.assertEqual(flags, {True})

    def test_exact_step_is_bit_identical_and_fp8_step_is_close(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("needs a CUDA device")
        from vllm.model_executor.layers.linear import ReplicatedLinear

        torch.manual_seed(0)
        layer = ReplicatedLinear(
            256, 128, bias=True, params_dtype=torch.bfloat16, prefix="visual_transformer_blocks.0.x"
        )
        with torch.no_grad():
            layer.weight.copy_(torch.randn(128, 256) * 0.05)
            layer.bias.copy_(torch.randn(128) * 0.1)
        layer = layer.cuda()
        x = torch.randn(4, 64, 256, device="cuda", dtype=torch.bfloat16)
        with torch.no_grad():
            reference = layer(x)[0]
            layer.quant_method = StepFp8LinearMethod(layer.quant_method)
            layer.fp8_step = False
            exact = layer(x)[0]
            layer.fp8_step = True
            fp8 = layer(x)[0]
        self.assertTrue(torch.equal(exact, reference))
        relative = (fp8.float() - reference.float()).norm() / reference.float().norm()
        self.assertLess(float(relative), 0.08)
        self.assertFalse(torch.equal(fp8, reference))

    def test_nvfp4_step_is_close_and_keeps_the_bias(self) -> None:
        if not torch.cuda.is_available() or torch.cuda.get_device_capability() < (10, 0):
            self.skipTest("needs a Blackwell CUDA device for the NVFP4 kernel")
        from vllm.model_executor.layers.linear import ReplicatedLinear

        torch.manual_seed(0)
        layer = ReplicatedLinear(
            256, 128, bias=True, params_dtype=torch.bfloat16, prefix="visual_transformer_blocks.0.x"
        )
        with torch.no_grad():
            layer.weight.copy_(torch.randn(128, 256) * 0.05)
            layer.bias.copy_(torch.randn(128) * 0.1)
        layer = layer.cuda()
        x = torch.randn(4, 64, 256, device="cuda", dtype=torch.bfloat16)
        bias = layer.bias.detach().clone()
        with torch.no_grad():
            reference = layer(x)[0]
            layer.quant_method = StepFp8LinearMethod(layer.quant_method, "nvfp4")
            layer.fp8_step = False
            exact = layer(x)[0]
            layer.fp8_step = True
            fp4 = layer(x)[0]
            layer.bias.zero_()
            fp4_no_bias = layer(x)[0]
        self.assertTrue(torch.equal(exact, reference))
        self.assertEqual(fp4.shape, reference.shape)
        self.assertEqual(fp4.dtype, reference.dtype)
        # E2M1 on both operands: ~0.13 relative error per GEMM, an order above FP8's.
        relative = (fp4.float() - reference.float()).norm() / reference.float().norm()
        self.assertLess(float(relative), 0.3)
        # cutlass_scaled_fp4_mm has no bias epilogue; the bias is added after it.
        torch.testing.assert_close((fp4 - fp4_no_bias).float(), bias.float().expand_as(fp4), atol=0.02, rtol=0.0)


class FormatTest(absltest.TestCase):
    def test_default_is_fp8_and_unknown_is_rejected(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("VLLM_OMNI_K6_STEP_GEMM_FORMAT", None)
            self.assertEqual(step_gemm_format(), "fp8")
        with mock.patch.dict(os.environ, {"VLLM_OMNI_K6_STEP_GEMM_FORMAT": "nvfp4"}):
            self.assertEqual(step_gemm_format(), "nvfp4")
        with mock.patch.dict(os.environ, {"VLLM_OMNI_K6_STEP_GEMM_FORMAT": "int4"}):
            with self.assertRaisesRegex(ValueError, "STEP_GEMM_FORMAT"):
                step_gemm_format()

    def test_global_scale_is_448_times_6_over_amax_and_zero_safe(self) -> None:
        t = torch.tensor([[1.0, -4.0], [2.0, 0.5]], dtype=torch.bfloat16)
        torch.testing.assert_close(nvfp4_global_scale(t), torch.tensor([448.0 * 6.0 / 4.0]))
        torch.testing.assert_close(nvfp4_global_scale(torch.zeros(2, 2, dtype=torch.bfloat16)), torch.tensor([1.0]))


class Int8WrapTest(absltest.TestCase):
    """NVFP4 steps also take an INT8 weight-only layer: dequantized, then re-quantized."""

    def test_post_processing_goes_to_the_wrapped_method(self) -> None:
        from vllm_omni.quantization.int8_config import Int8WeightOnlyLinearMethod

        inner = mock.create_autospec(Int8WeightOnlyLinearMethod, instance=True)
        layer = SimpleNamespace()
        StepFp8LinearMethod(inner, "nvfp4").process_weights_after_loading(layer)
        inner.process_weights_after_loading.assert_called_once_with(layer)

    def test_int8_layer_computes_with_its_dequantized_weight(self) -> None:
        from vllm_omni.quantization.int8_config import Int8WeightOnlyLinearMethod, dequantize_int8_rows

        inner = mock.create_autospec(Int8WeightOnlyLinearMethod, instance=True)
        layer = SimpleNamespace(
            weight=torch.randint(-127, 128, (8, 16), dtype=torch.int8),
            weight_scale=torch.rand(8, 1, dtype=torch.float32),
        )
        got = StepFp8LinearMethod(inner, "nvfp4")._bf16_weight(layer, torch.bfloat16)
        torch.testing.assert_close(got, dequantize_int8_rows(layer.weight, layer.weight_scale, torch.bfloat16))


if __name__ == "__main__":
    absltest.main()
