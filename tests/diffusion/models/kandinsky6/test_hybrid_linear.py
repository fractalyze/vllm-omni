# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Hybrid-GEMM routing: which linears are wrapped, and that the wrapper calls the kernel like F.linear."""

from __future__ import annotations

import os
from unittest import mock

import torch
from absl.testing import absltest

from vllm_omni.diffusion.models.kandinsky6.hybrid_linear import (
    HybridFp16LinearMethod,
    hybrid_enabled,
    install_hybrid,
)


def _fake_matmul(x, w, bias=None, *, out_dtype=None):
    out = torch.nn.functional.linear(x.half().float(), w.half().float(), None if bias is None else bias.float())
    return out.to(out_dtype or x.dtype)


class FakeLinear(torch.nn.Module):
    """Stands in for a vLLM LinearBase with an UnquantizedLinearMethod."""


class SwitchTest(absltest.TestCase):
    def test_off_by_default(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("VLLM_OMNI_K6_HYBRID_GEMM", None)
            self.assertFalse(hybrid_enabled())

    def test_release_scratch_only_when_enabled(self) -> None:
        from vllm_omni.diffusion.models.kandinsky6 import hybrid_linear

        with mock.patch.object(hybrid_linear, "current_omni_platform") as platform:
            platform.is_available.return_value = True
            with mock.patch.dict(os.environ, {"VLLM_OMNI_K6_HYBRID_GEMM": "1"}):
                self.assertTrue(hybrid_linear.release_hybrid_scratch())
            platform.empty_cache.assert_called_once()
            with mock.patch.dict(os.environ, {"VLLM_OMNI_K6_HYBRID_GEMM": "0"}):
                self.assertFalse(hybrid_linear.release_hybrid_scratch())
            platform.empty_cache.assert_called_once()

    def test_on(self) -> None:
        with mock.patch.dict(os.environ, {"VLLM_OMNI_K6_HYBRID_GEMM": "1"}):
            self.assertTrue(hybrid_enabled())


class InstallTest(absltest.TestCase):
    def _model(self):
        from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod

        class Lin(LinearBase):
            def __init__(self):
                torch.nn.Module.__init__(self)
                self.quant_method = UnquantizedLinearMethod.__new__(UnquantizedLinearMethod)
                self.weight = torch.nn.Parameter(torch.randn(8, 16) * 0.1, requires_grad=False)

        model = torch.nn.Module()
        model.visual = torch.nn.Module()
        model.visual.ff = Lin()
        model.out_layer = Lin()
        return model

    def test_exclude_keeps_named_layers(self) -> None:
        model = self._model()
        wrapped, excluded = install_hybrid(model, exclude=r"^out_layer", matmul=_fake_matmul)
        self.assertEqual((wrapped, excluded), (1, 1))
        self.assertIsInstance(model.visual.ff.quant_method, HybridFp16LinearMethod)
        self.assertNotIsInstance(model.out_layer.quant_method, HybridFp16LinearMethod)

    def test_default_exclude_keeps_one_vector_layers(self) -> None:
        model = self._model()
        model.visual.va_modulation = type(model.visual.ff)()
        model.time_embeddings = torch.nn.Module()
        model.time_embeddings.out_layer = type(model.visual.ff)()
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("VLLM_OMNI_K6_HYBRID_GEMM_EXCLUDE", None)
            wrapped, excluded = install_hybrid(model, matmul=_fake_matmul)
        self.assertEqual((wrapped, excluded), (2, 2))
        self.assertNotIsInstance(model.visual.va_modulation.quant_method, HybridFp16LinearMethod)
        self.assertNotIsInstance(model.time_embeddings.out_layer.quant_method, HybridFp16LinearMethod)

    def test_empty_exclude_wraps_everything(self) -> None:
        model = self._model()
        model.visual.va_modulation = type(model.visual.ff)()
        self.assertEqual(install_hybrid(model, exclude="", matmul=_fake_matmul), (3, 0))

    def test_stage_fp16_flags_only_large_m_linears(self) -> None:
        from vllm_omni.diffusion.models.kandinsky6.hybrid_linear import LARGE_M_LINEARS

        model = torch.nn.Module()
        model.visual_transformer_blocks = torch.nn.ModuleList([torch.nn.Module()])
        block = model.visual_transformer_blocks[0]
        block.video_dec_block = torch.nn.Module()
        block.video_dec_block.feed_forward = torch.nn.Module()
        block.video_dec_block.feed_forward.in_layer = type(self._model().visual.ff)()
        block.av_cross_attention = torch.nn.Module()
        block.av_cross_attention.to_query = type(self._model().visual.ff)()
        with mock.patch.dict(os.environ, {"VLLM_OMNI_K6_HYBRID_STAGE_FP16": "1"}):
            install_hybrid(model, exclude="", matmul=_fake_matmul)
        self.assertEqual(block.video_dec_block.feed_forward.in_layer.dlo_stage_weight_dtype, torch.float16)
        self.assertFalse(hasattr(block.av_cross_attention.to_query, "dlo_stage_weight_dtype"))
        self.assertRegex("visual_transformer_blocks.59.av_cross_attention.to_value", LARGE_M_LINEARS)

    def test_stage_fp16_off_by_default(self) -> None:
        model = self._model()
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("VLLM_OMNI_K6_HYBRID_STAGE_FP16", None)
            install_hybrid(model, exclude="", matmul=_fake_matmul)
        self.assertFalse(hasattr(model.visual.ff, "dlo_stage_weight_dtype"))

    def test_fp16_weight_stays_hybrid_at_small_m(self) -> None:
        layer = self._model().visual.ff
        layer.weight = torch.nn.Parameter(layer.weight.half(), requires_grad=False)
        inner, matmul = mock.Mock(), mock.Mock()
        HybridFp16LinearMethod(inner, matmul, min_rows=2048).apply(layer, torch.zeros(1, 10, 16))
        matmul.assert_called_once()
        inner.apply.assert_not_called()

    def test_small_m_takes_the_original_method(self) -> None:
        layer = self._model().visual.ff
        inner = mock.Mock()
        matmul = mock.Mock()
        method = HybridFp16LinearMethod(inner, matmul, min_rows=2048)
        method.apply(layer, torch.zeros(1, 2047, 16))
        inner.apply.assert_called_once()
        matmul.assert_not_called()
        method.apply(layer, torch.zeros(2, 1024, 16))
        matmul.assert_called_once()

    def test_apply_is_a_linear(self) -> None:
        model = self._model()
        with mock.patch.dict(os.environ, {"VLLM_OMNI_K6_HYBRID_GEMM_MIN_ROWS": "0"}):
            install_hybrid(model, exclude="", matmul=_fake_matmul)
        layer = model.visual.ff
        x = torch.randn(3, 5, 16).to(torch.bfloat16)
        bias = torch.randn(8)
        out = layer.quant_method.apply(layer, x, bias)
        self.assertEqual(tuple(out.shape), (3, 5, 8))
        self.assertEqual(out.dtype, torch.bfloat16)
        reference = torch.nn.functional.linear(x.float(), layer.weight.float(), bias)
        self.assertLess(float((out.float() - reference).norm() / reference.norm()), 1e-2)


@absltest.skipUnless(torch.cuda.is_available(), "needs a CUDA GPU")
class KernelTest(absltest.TestCase):
    """The real kernel behind the wrapper, eager and inside a compiled region (as the DiT blocks run)."""

    def _layer(self):
        layer = InstallTest._model(self).visual.ff
        layer.weight = torch.nn.Parameter(
            (torch.randn(256, 512) * 0.05).to("cuda", torch.bfloat16), requires_grad=False
        )
        layer.quant_method = HybridFp16LinearMethod(layer.quant_method, min_rows=0)
        return layer

    def _check(self, out, x, layer, bias) -> None:
        self.assertEqual(out.dtype, torch.bfloat16)
        self.assertEqual(tuple(out.shape), (2, 300, 256))
        reference = torch.nn.functional.linear(x.double(), layer.weight.double(), bias.double())
        self.assertLess(float((out.double() - reference).norm() / reference.norm()), 5e-3)

    def test_eager(self) -> None:
        layer = self._layer()
        x = torch.randn(2, 300, 512, device="cuda").to(torch.bfloat16)
        bias = torch.randn(256, device="cuda").to(torch.bfloat16)
        self._check(layer.quant_method.apply(layer, x, bias), x, layer, bias)

    def test_compiled_producer_writes_the_fp16_input_once_per_call(self) -> None:
        """Bias-free calls must not get a second FP16 copy of x for the unused Bias pointer."""
        import re

        from torch._inductor.utils import run_and_get_code

        from vllm_omni.diffusion.models.kandinsky6.hybrid_gemm import hybrid_matmul

        x = torch.randn(256, 512, device="cuda").to(torch.bfloat16)
        w = (torch.randn(256, 512, device="cuda") * 0.05).to(torch.bfloat16)

        def project(t):
            return hybrid_matmul(torch.relu(t), w, out_dtype=torch.bfloat16)

        torch._dynamo.reset()
        with torch.no_grad():
            _, codes = run_and_get_code(torch.compile(project, fullgraph=True), x)
        # Kernel bodies only (each ends at its closing triple quote); the ReLU producer is the one with `maximum`.
        bodies = [k.split("'''")[0] for k in re.split(r"\ndef ", "\n".join(codes))[1:]]
        producer = [k for k in bodies if "triton_helpers.maximum" in k]
        self.assertLen(producer, 1)
        self.assertEqual(len(re.findall(r"tl\.store", producer[0])), 1)

    def test_compiled_without_graph_break(self) -> None:
        layer = self._layer()
        x = torch.randn(2, 300, 512, device="cuda").to(torch.bfloat16)
        bias = torch.randn(256, device="cuda").to(torch.bfloat16)
        compiled = torch.compile(lambda t: layer.quant_method.apply(layer, t, bias) * 2, fullgraph=True)
        self._check(compiled(x) / 2, x, layer, bias)


if __name__ == "__main__":
    absltest.main()
