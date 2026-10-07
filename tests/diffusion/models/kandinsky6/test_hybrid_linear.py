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

    def test_apply_is_a_linear(self) -> None:
        model = self._model()
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
        layer.quant_method = HybridFp16LinearMethod(layer.quant_method)
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

    def test_compiled_without_graph_break(self) -> None:
        layer = self._layer()
        x = torch.randn(2, 300, 512, device="cuda").to(torch.bfloat16)
        bias = torch.randn(256, device="cuda").to(torch.bfloat16)
        compiled = torch.compile(lambda t: layer.quant_method.apply(layer, t, bias) * 2, fullgraph=True)
        self._check(compiled(x) / 2, x, layer, bias)


if __name__ == "__main__":
    absltest.main()
