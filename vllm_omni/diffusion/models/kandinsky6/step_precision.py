# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Step-scheduled GEMM precision for the Kandinsky 6 DiT: exact early, FP8 late.

An error made at an early sampler step grows the most over the trajectory (one
exact attention step bought more quality than twelve exact blocks). The same
schedule applied to the GEMMs: keep the first ``k`` steps' linears exact BF16,
and run the later steps' as FP8 GEMMs (``torch._scaled_mm``, cuBLASLt), which
are 1.4-1.5x BF16 at W1's shapes on sm_120.

No second copy of the weights. The streamed path already brings each block's
BF16 weights to the GPU every step; on an FP8 step the wrapped layer quantizes
that weight per tensor on the device (one amax reduction and one cast, ~1 ms
for a 0.9 GiB block) and the activation per tensor, dynamically. On an exact
step the layer calls the method it wrapped, unchanged, so step 1 is
bit-identical to the unscheduled model.

The step is a plain attribute on each wrapped layer, set by the sampler loop,
so regionally compiled blocks hold two cached variants rather than recompiling.

``VLLM_OMNI_K6_FP8_GEMM_AFTER_STEP=k`` turns it on (FP8 from step index k, i.e.
after k exact steps; 0 = off). ``VLLM_OMNI_K6_FP8_GEMM_LAYERS`` is a regular
expression over layer prefixes choosing which linears may run FP8 (default:
every linear in the visual transformer blocks).
"""

from __future__ import annotations

import os
import re

import torch
from torch import nn
from vllm import _custom_ops as ops
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod

FP8_E4M3_MAX = 448.0
DEFAULT_LAYERS = r"^visual_transformer_blocks\."


def fp8_after_step() -> int:
    """Index of the first sampler step whose GEMMs run FP8; 0 means never."""
    return max(0, int(os.environ.get("VLLM_OMNI_K6_FP8_GEMM_AFTER_STEP", "0") or 0))


def quantize_weight_per_tensor(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``weight`` -> ``float8_e4m3fn`` and its FP32 scale of shape ``(1,)``; amax / 448, zero-safe."""
    as_float = weight.float()
    amax = as_float.abs().amax()
    scale = torch.where(amax > 0, amax / FP8_E4M3_MAX, torch.ones_like(amax))
    quantized = (as_float / scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn)
    return quantized, scale.reshape(1)


class StepFp8LinearMethod(UnquantizedLinearMethod):
    """A BF16 linear that runs an FP8 GEMM on the steps its layer is told to."""

    def __init__(self, inner: UnquantizedLinearMethod) -> None:
        super().__init__()
        self.inner = inner

    def apply(self, layer: nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        if not getattr(layer, "fp8_step", False):
            return self.inner.apply(layer, x, bias)
        weight_fp8, weight_scale = quantize_weight_per_tensor(layer.weight)
        x_2d = x.reshape(-1, x.shape[-1])
        x_fp8, x_scale = ops.scaled_fp8_quant(x_2d, None)
        out = torch._scaled_mm(
            x_fp8,
            weight_fp8.t(),
            scale_a=x_scale.reshape(1),
            scale_b=weight_scale,
            out_dtype=x.dtype,
            bias=bias,
        )
        return out.reshape(*x.shape[:-1], out.shape[-1])


def install_step_fp8(dit: nn.Module, layers: str | None = None) -> int:
    """Wrap every matching unquantized linear; returns how many were wrapped."""
    pattern = re.compile(layers or os.environ.get("VLLM_OMNI_K6_FP8_GEMM_LAYERS", DEFAULT_LAYERS))
    wrapped = 0
    for name, module in dit.named_modules():
        method = getattr(module, "quant_method", None)
        if not isinstance(module, LinearBase) or not pattern.search(name):
            continue
        if type(method) is not UnquantizedLinearMethod:
            continue
        module.quant_method = StepFp8LinearMethod(method)
        module.fp8_step = False
        wrapped += 1
    return wrapped


def set_fp8_gemm_step(dit: nn.Module, enabled: bool) -> None:
    """Make the wrapped linears run FP8 (``enabled``) or exact on the current step."""
    for module in dit.modules():
        if isinstance(getattr(module, "quant_method", None), StepFp8LinearMethod):
            module.fp8_step = bool(enabled)
