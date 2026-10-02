# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# Copyright 2026 Fractalyze Inc. All rights reserved.
"""The thinker stage's embedding lookup, for the code that runs beside its
model in the same process: the thinker-to-talker input processor
(VLLM_OMNI_EARLY_CHUNK) and the scheduler (VLLM_OMNI_TALKER_PREPREFILL).

A text prompt's talker input needs only thinker embeddings, which are a
table lookup, so those paths can build it before the thinker's forward has
captured them.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ThinkerEmbedding:
    # Token ids [n] on `device` → their thinker embeddings [n, hidden].
    embed_input_ids: Callable[[torch.Tensor], torch.Tensor]
    # Where the embedding table lives.
    device: torch.device
    # The TTS special tokens' ids [1, 3]: bos, eos, pad.
    tts_tokens: torch.Tensor


_registered: ThinkerEmbedding | None = None


def register(thinker: nn.Module, tts_tokens: torch.Tensor) -> None:
    """Keeps the loaded thinker's lookup for this process."""
    global _registered
    _registered = ThinkerEmbedding(thinker.embed_input_ids, next(thinker.parameters()).device, tts_tokens)


def registered() -> ThinkerEmbedding | None:
    """The thinker's lookup, or None outside a thinker stage with
    VLLM_OMNI_EARLY_CHUNK or VLLM_OMNI_TALKER_PREPREFILL on."""
    return _registered
