# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""VLLM_OMNI_FAST_POLL in the stage engine loop: a step that ran nothing
while requests wait returns as soon as a chunk lands, instead of sleeping
1 ms. CPU-only, on a stand-in engine core.
"""

import threading
import time
from types import SimpleNamespace

import pytest
from vllm.v1.engine.core import EngineCoreProc

from vllm_omni.engine.stage_engine_core_proc import StageEngineCoreProc

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _engine(chunk_landed, model_executed=False, has_requests=True):
    engine = object.__new__(StageEngineCoreProc)
    engine._chunk_landed = chunk_landed
    engine.outputs = []
    engine.step_fn = lambda: ({0: "out"}, model_executed)
    engine.output_queue = SimpleNamespace(put_nowait=engine.outputs.append)
    engine.post_step = lambda executed: None
    engine.scheduler = SimpleNamespace(has_requests=lambda: has_requests)
    return engine


def test_switch_off_runs_vllms_step(monkeypatch):
    calls = []

    def vllm_step(self):
        calls.append(self)
        return True

    monkeypatch.setattr(EngineCoreProc, "_process_engine_step", vllm_step)
    engine = _engine(None)
    assert engine._process_engine_step() is True
    assert calls == [engine]


def test_an_idle_step_returns_once_a_chunk_lands():
    landed = threading.Event()
    engine = _engine(landed)
    threading.Timer(0.0002, landed.set).start()

    executed = engine._process_engine_step()

    assert executed is False
    assert engine.outputs == [(0, "out")]
    # The event is consumed, so the next idle step waits for the next chunk.
    assert not landed.is_set()


def test_an_idle_step_still_waits_at_most_1ms():
    engine = _engine(threading.Event())
    start = time.monotonic()
    engine._process_engine_step()
    assert time.monotonic() - start < 0.1


def test_a_step_that_ran_the_model_does_not_wait():
    landed = threading.Event()
    landed.set()
    engine = _engine(landed, model_executed=True)
    assert engine._process_engine_step() is True
    assert landed.is_set()
