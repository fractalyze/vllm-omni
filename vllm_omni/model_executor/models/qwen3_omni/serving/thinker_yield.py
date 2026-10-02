# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# Copyright 2026 Fractalyze Inc. All rights reserved.
"""Holds the thinker's decode while the talker makes a request's frame 0
(VLLM_OMNI_THINKER_YIELD, with VLLM_OMNI_FRAME0).

After the thinker's first token, the talker's prefill step, the code
predictor and code2wav make frame 0's audio: the path to the first audio.
The thinker keeps decoding meanwhile, each step streaming its weights at
full bandwidth beside them when the stages share a GPU, which slows them. With
the switch on:

- in the thinker's stage, a request's first decode step starts a hold, and
  the next step waits before it launches until the talker has shipped one
  more frame 0, or HOLD_S has passed since it began waiting;
- in the talker's stage, the last frame-0 listener counts each frame 0, once
  the listeners before it (VLLM_OMNI_FRAME0_AUDIO's decode and send among
  them) have run.

The count lives in an 8-byte file in the run directory both stages map. The
hold costs the reply's later text the window, once per request; the first
token is out before it starts.
"""

from __future__ import annotations

import mmap
import os
import struct
import time
from collections.abc import Callable

from vllm_omni.model_executor.models.qwen3_omni.serving.frame0_audio import run_dir

# The longest the thinker waits for a frame 0.
HOLD_S = 0.012
_POLL_S = 0.00005


class Counter:
    """A shared int64 counter in a file, mapped by every process that opens
    it."""

    def __init__(self, path: str) -> None:
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            if os.fstat(fd).st_size < 8:
                os.ftruncate(fd, 8)
            self._map = mmap.mmap(fd, 8)
        finally:
            os.close(fd)

    def value(self) -> int:
        return struct.unpack_from("<q", self._map)[0]

    def bump(self) -> None:
        struct.pack_into("<q", self._map, 0, self.value() + 1)


def frame0_shipped() -> Counter:
    """The run directory's count of frames 0 the talker has shipped."""
    return Counter(os.path.join(run_dir(), "qwen3_omni_frame0_shipped"))


class FrameShippedCounter:
    """The talker's frame-0 listener: counts each frame 0 shipped."""

    def __init__(self) -> None:
        self._shipped = frame0_shipped()

    def __call__(self, runner, req_id, codes) -> None:
        self._shipped.bump()


class Hold:
    """The thinker's side: started at a request's first decode step, waited
    out before the next. The first decode step is queued behind the prefill,
    long before the talker has work, so the window runs from when the next
    step starts waiting."""

    def __init__(
        self,
        shipped: Counter,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._shipped = shipped
        self._clock = clock
        self._sleep = sleep
        self._target: int | None = None

    def start(self) -> None:
        self._target = self._shipped.value() + 1

    def wait(self) -> None:
        if self._target is None:
            return
        until = self._clock() + HOLD_S
        while self._shipped.value() < self._target and self._clock() < until:
            self._sleep(_POLL_S)
        self._target = None

    def before_step(self, scheduler_output, requests) -> None:
        """Waits out a started hold, then starts one if this step decodes
        some request's first token."""
        self.wait()
        if first_decode(scheduler_output, requests):
            self.start()


def first_decode(scheduler_output, requests) -> bool:
    """Whether the step decodes some request's first token after its
    prompt."""
    cached = scheduler_output.scheduled_cached_reqs
    for req_id, computed in zip(cached.req_ids, cached.num_computed_tokens):
        state = requests.get(req_id)
        if state is not None and computed == len(state.prompt_token_ids or ()):
            return True
    return False


def serves(runner) -> bool:
    """Whether `runner` is Qwen3-Omni's thinker stage."""
    model = getattr(runner, "model", None)
    return getattr(model, "thinker", None) is not None and getattr(model, "talker", None) is None
