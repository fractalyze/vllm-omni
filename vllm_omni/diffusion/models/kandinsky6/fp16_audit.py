# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Per-layer FP16 range audit of the Kandinsky 6 DiT's linears, on real activations.

A GEMM that takes FP16 inputs (and, in the hybrid kernel, accumulates 64-element
partials in FP16) is only safe for a layer whose operands and partial sums fit
FP16: finite up to 65504, normal down to 2^-14 (6.1e-5), and flushed to zero
below 2^-24 (6.0e-8). The model runs in BF16, which has FP32's range, so nothing
guarantees that. This records, per linear and over every call (all sampler
steps), what the operands actually are.

``VLLM_OMNI_K6_FP16_AUDIT=<path.json>`` installs forward hooks on every DiT
linear; after each request the pipeline writes the running statistics there.
Measurement only: the hooks synchronize on reductions and are never on a
served path.

Per layer:
- ``x_absmax`` / ``w_absmax`` / ``y_absmax``: largest |input|, |weight|, |output|.
- ``x_sub`` / ``w_sub``: fraction of nonzero elements below FP16's smallest
  normal (they lose precision as subnormals), from a 1-in-97 strided sample.
- ``x_flush`` / ``w_flush``: fraction of nonzero elements below FP16's smallest
  subnormal (they become zero).
- ``partial64_bound``: ``x_absmax * w_absmax * 64``, an upper bound on any
  64-term partial sum, the quantity the hybrid kernel holds in FP16.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from torch import nn

FP16_MAX = 65504.0
FP16_MIN_NORMAL = 2.0**-14
FP16_MIN_SUBNORMAL = 2.0**-24


def audit_path() -> str:
    return os.environ.get("VLLM_OMNI_K6_FP16_AUDIT", "")


# Fractions come from every SAMPLE_STRIDE-th element (a strided view, no copy of
# the activation): an FFN input at W1 is 1.6 GB, and a full float copy of it
# beside the model does not fit the board.
SAMPLE_STRIDE = 97


def _absmax(t: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(t.detach(), ord=float("inf")))


def _fractions(t: torch.Tensor) -> tuple[float, float]:
    a = t.detach().reshape(-1)[::SAMPLE_STRIDE].abs().float()
    nonzero = a > 0
    count = int(nonzero.sum())
    if not count:
        return 0.0, 0.0
    sub = int(((a < FP16_MIN_NORMAL) & nonzero).sum()) / count
    flush = int(((a < FP16_MIN_SUBNORMAL) & nonzero).sum()) / count
    return sub, flush


class Fp16Audit:
    """Running per-layer FP16 range statistics, fed by forward hooks."""

    def __init__(self) -> None:
        self.stats: dict[str, dict[str, float]] = {}
        self._handles: list = []

    def update(self, name: str, x: torch.Tensor, w: torch.Tensor, y: torch.Tensor) -> None:
        s = self.stats.setdefault(
            name,
            {
                "calls": 0,
                "x_absmax": 0.0,
                "w_absmax": 0.0,
                "y_absmax": 0.0,
                "x_sub": 0.0,
                "x_flush": 0.0,
                "w_sub": 0.0,
                "w_flush": 0.0,
            },
        )
        x_sub, x_flush = _fractions(x)
        w_sub, w_flush = _fractions(w)
        s["calls"] += 1
        s["x_absmax"] = max(s["x_absmax"], _absmax(x))
        s["w_absmax"] = max(s["w_absmax"], _absmax(w))
        s["y_absmax"] = max(s["y_absmax"], _absmax(y))
        s["x_sub"] = max(s["x_sub"], x_sub)
        s["x_flush"] = max(s["x_flush"], x_flush)
        s["w_sub"] = max(s["w_sub"], w_sub)
        s["w_flush"] = max(s["w_flush"], w_flush)

    def install(self, dit: nn.Module) -> int:
        from vllm.model_executor.layers.linear import LinearBase

        count = 0
        for name, module in dit.named_modules():
            if not isinstance(module, LinearBase):
                continue

            def hook(mod, args, output, _name=name):
                y = output[0] if isinstance(output, tuple) else output
                self.update(_name, args[0], mod.weight, y)

            self._handles.append(module.register_forward_hook(hook))
            count += 1
        return count

    def report(self) -> dict[str, dict[str, float]]:
        out = {}
        for name, s in self.stats.items():
            row = dict(s)
            row["partial64_bound"] = s["x_absmax"] * s["w_absmax"] * 64
            row["at_risk"] = bool(
                s["y_absmax"] > FP16_MAX
                or row["partial64_bound"] > FP16_MAX
                or s["x_absmax"] > FP16_MAX
                or s["w_absmax"] > FP16_MAX
            )
            out[name] = row
        return out

    def write(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.report(), indent=1, sort_keys=True))
