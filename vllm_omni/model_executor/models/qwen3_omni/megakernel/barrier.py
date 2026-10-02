# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# Copyright 2026 Fractalyze Inc. All rights reserved.
"""The grid barrier's watchdog record, and the grid size the kernels launch on.

A kernel that waits at a barrier longer than its timeout writes where it
stopped into a host-mapped `ErrorRecord` and traps. The trap leaves the CUDA
context unusable, so the process can report the record but must then exit.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from vllm_omni.model_executor.models.qwen3_omni.megakernel import _ext
from vllm_omni.platforms import current_omni_platform

DEFAULT_TIMEOUT_NS = 1_000_000_000
_TIMEOUT_STATUS = 1


def num_ctas(device: torch.device | None = None) -> int:
    """One persistent CTA per SM."""
    return torch.cuda.get_device_properties(device).multi_processor_count


@dataclass(frozen=True)
class Hang:
    """Where a launch stopped: `cta`'s watchdog fired at launch-wide barrier
    `barrier` of decode step `step`, with `arrived` of `expected` CTAs there."""

    cta: int
    barrier: int
    step: int
    arrived: int
    expected: int


class BarrierTimeoutError(RuntimeError):
    def __init__(self, hang: Hang) -> None:
        super().__init__(
            f"grid barrier {hang.barrier} of step {hang.step} timed out on CTA "
            f"{hang.cta}: {hang.arrived} of {hang.expected} CTAs arrived; the "
            "CUDA context is lost"
        )
        self.hang = hang


class ErrorRecord:
    """The watchdog's record, in pinned host memory the kernel writes directly."""

    def __init__(self) -> None:
        words = _ext.load().ERROR_RECORD_WORDS
        self.tensor = torch.zeros(words, dtype=torch.int32, pin_memory=True)

    def hang(self) -> Hang | None:
        status, *fields = self.tensor.tolist()
        return Hang(*fields) if status == _TIMEOUT_STATUS else None

    def synchronize(self) -> None:
        """Waits for the GPU; raises BarrierTimeoutError if a watchdog fired."""
        try:
            current_omni_platform.synchronize()
        except RuntimeError as err:
            hang = self.hang()
            if hang is not None:
                raise BarrierTimeoutError(hang) from err
            raise


def sync_words(device: torch.device | str = "cuda") -> torch.Tensor:
    """The zeroed-per-launch device words a launch's barriers use."""
    return torch.zeros(_ext.load().SYNC_WORDS, dtype=torch.int32, device=device)
