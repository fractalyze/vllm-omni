# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Route the Kandinsky 6 DiT's linears through the hybrid FP16-accumulate GEMM.

On sm_120 an FP16 MMA that accumulates in FP16 runs faster than one that
accumulates in FP32. The hybrid kernel (``hybrid_gemm.hybrid_matmul``) keeps
that instruction but promotes its partial sums to an FP32 running total every
``BLOCK_K`` products, so its error is bounded by ``BLOCK_K`` rather than by K.
It takes FP16 operands, which have far less range than the BF16 the model runs
in: a layer whose activations, weights or 64-term partial sums approach 65504
must stay on the current path. That is decided per layer from a range audit
(``fp16_audit.py``) and passed here as an exclude pattern.

``VLLM_OMNI_K6_HYBRID_GEMM=1`` turns it on. ``VLLM_OMNI_K6_HYBRID_GEMM_EXCLUDE``
is a regular expression over layer names kept on the current path; unset, it is
``DEFAULT_EXCLUDE``. With the switch off, nothing changes.

The audit on real activations (W1, all ten steps, a1/a3/b6) found no layer near
FP16 overflow: the largest output was ~5e3 and the largest 64-term partial-sum
bound ~2e4, both under 65504. The risk it did find is underflow, confined to
layers that act on one vector per step: the modulation projections (up to 84%
of ``va_modulation`` weights are FP16-subnormal and 32% flush to zero) and the
time-embedding output layer (82% of its inputs subnormal, 9% flushed). Those
are ``DEFAULT_EXCLUDE``; with M = 1 they cost nothing to keep in BF16.
"""

from __future__ import annotations

import os
import re

import torch
from torch import nn
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod

from vllm_omni.platforms import current_omni_platform

DEFAULT_EXCLUDE = r"modulation|time_embeddings"

# Below this many rows cuBLAS wins: at W1's audio (M=218) and text (M=256) shapes
# the hybrid is 1.7-2.7x slower, 1024 is where it stops losing and 2048 where it
# reliably wins (Track C's crossover, showcase/kandinsky6/compute/hybrid_gemm.py).
# Such calls take the layer's original method. M is static per compiled graph,
# so this costs no runtime branch. VLLM_OMNI_K6_HYBRID_GEMM_MIN_ROWS overrides it.
DEFAULT_MIN_ROWS = 2048


def hybrid_enabled() -> bool:
    return os.environ.get("VLLM_OMNI_K6_HYBRID_GEMM", "") not in ("", "0", "false", "False")


def release_hybrid_scratch() -> bool:
    """Return the hybrid GEMM's cached FP16 operand copies to the device; True if it did.

    ``hybrid_matmul`` casts each activation to FP16 (up to 1.6 GB for FF2 at
    W1), and the caching allocator keeps those blocks after the call. The VAE
    decoder plans its tiles from the device's free memory, so without this the
    hybrid arm decodes with smaller tiles than the BF16 path: different pixels
    and a slower decode. Called once before the decode; a no-op with the switch off.
    """
    if not hybrid_enabled() or not current_omni_platform.is_available():
        return False
    current_omni_platform.empty_cache()
    return True


def _kernel():
    from .hybrid_gemm import hybrid_matmul

    return hybrid_matmul


class HybridFp16LinearMethod(UnquantizedLinearMethod):
    """A BF16 linear whose GEMM runs through the hybrid FP16-accumulate kernel."""

    def __init__(self, inner: UnquantizedLinearMethod, matmul=None, min_rows: int | None = None) -> None:
        super().__init__()
        self.inner = inner
        self._matmul = matmul
        if min_rows is None:
            min_rows = int(os.environ.get("VLLM_OMNI_K6_HYBRID_GEMM_MIN_ROWS", DEFAULT_MIN_ROWS))
        self.min_rows = min_rows

    def apply(self, layer: nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        if x.numel() // x.shape[-1] < self.min_rows:
            return self.inner.apply(layer, x, bias)
        matmul = self._matmul or _kernel()
        return matmul(x, layer.weight, bias, out_dtype=x.dtype)


def install_hybrid(dit: nn.Module, exclude: str | None = None, matmul=None) -> tuple[int, int]:
    """Wrap every unquantized DiT linear not matching ``exclude`` ('' = none); returns (wrapped, excluded)."""
    if exclude is None:
        exclude = os.environ.get("VLLM_OMNI_K6_HYBRID_GEMM_EXCLUDE", DEFAULT_EXCLUDE)
    pattern = exclude
    excluded_re = re.compile(pattern) if pattern else None
    wrapped = excluded = 0
    for name, module in dit.named_modules():
        if (
            not isinstance(module, LinearBase)
            or type(getattr(module, "quant_method", None)) is not UnquantizedLinearMethod
        ):
            continue
        if excluded_re is not None and excluded_re.search(name):
            excluded += 1
            continue
        module.quant_method = HybridFp16LinearMethod(module.quant_method, matmul)
        wrapped += 1
    return wrapped, excluded
