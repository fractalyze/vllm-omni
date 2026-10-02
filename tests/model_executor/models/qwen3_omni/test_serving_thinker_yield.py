# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""VLLM_OMNI_THINKER_YIELD: the thinker's step after a request's first
decode step waits until the talker counts one more frame 0, or HOLD_S.
CPU-only, with a fake clock.
"""

from types import SimpleNamespace

import pytest

from vllm_omni.model_executor.models.qwen3_omni.serving import thinker_yield
from vllm_omni.model_executor.models.qwen3_omni.serving.frame0 import TalkerFrame0
from vllm_omni.model_executor.models.qwen3_omni.serving.thinker_yield import (
    HOLD_S,
    Counter,
    FrameShippedCounter,
    Hold,
    first_decode,
)
from vllm_omni.worker.gpu_ar_model_runner import GPUARModelRunner

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture(autouse=True)
def _run_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("VLLM_OMNI_QWEN3_OMNI_RUN_DIR", str(tmp_path))


def test_counter_is_shared_through_its_file(tmp_path):
    a, b = Counter(str(tmp_path / "c")), Counter(str(tmp_path / "c"))
    a.bump()
    a.bump()
    assert b.value() == 2


class _Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def test_hold_ends_when_the_talker_ships_a_frame0(tmp_path):
    shipped = Counter(str(tmp_path / "c"))
    clock = _Clock()

    def sleep(seconds):
        clock.sleep(seconds)
        if clock.now >= 0.002:
            shipped.bump()

    hold = Hold(shipped, clock=clock, sleep=sleep)
    hold.start()
    hold.wait()
    assert 0.002 <= clock.now < HOLD_S


def test_hold_gives_up_after_hold_s(tmp_path):
    clock = _Clock()
    hold = Hold(Counter(str(tmp_path / "c")), clock=clock, sleep=clock.sleep)
    hold.start()
    hold.wait()
    assert HOLD_S <= clock.now < HOLD_S + 0.001
    # A hold is waited out once.
    clock.now = 0.0
    hold.wait()
    assert clock.now == 0.0


def _step(computed):
    return SimpleNamespace(
        scheduled_cached_reqs=SimpleNamespace(req_ids=["r"], num_computed_tokens=[computed]),
    )


def test_first_decode():
    requests = {"r": SimpleNamespace(prompt_token_ids=[1, 2, 3])}
    assert first_decode(_step(3), requests)
    assert not first_decode(_step(4), requests)
    assert not first_decode(_step(3), {})


def test_before_step_starts_a_hold_at_the_first_decode(tmp_path):
    clock = _Clock()
    hold = Hold(Counter(str(tmp_path / "c")), clock=clock, sleep=clock.sleep)
    requests = {"r": SimpleNamespace(prompt_token_ids=[1, 2, 3])}

    hold.before_step(_step(3), requests)
    assert clock.now == 0.0
    hold.before_step(_step(4), requests)
    assert clock.now >= HOLD_S


def test_talker_counts_frame0_after_the_other_listeners(monkeypatch):
    monkeypatch.setenv("VLLM_OMNI_FRAME0_AUDIO", "1")
    monkeypatch.setenv("VLLM_OMNI_THINKER_YIELD", "1")
    listeners = TalkerFrame0.with_listeners().listeners
    assert isinstance(listeners[-1], FrameShippedCounter)

    listeners[-1](None, "r", None)
    assert thinker_yield.frame0_shipped().value() == 1


def _runner(talker):
    return SimpleNamespace(model=SimpleNamespace(thinker=object(), talker=object() if talker else None))


@pytest.mark.parametrize("switch,talker,built", [("1", False, True), ("1", True, False), ("0", False, False)])
def test_only_the_thinker_holds(monkeypatch, switch, talker, built):
    monkeypatch.setenv("VLLM_OMNI_THINKER_YIELD", switch)
    runner = _runner(talker)
    GPUARModelRunner._init_thinker_hold(runner)
    assert isinstance(getattr(runner, "thinker_hold", None), Hold) is built


def test_execute_model_waits_before_the_step():
    calls = []
    runner = SimpleNamespace(
        thinker_hold=SimpleNamespace(before_step=lambda output, requests: calls.append(output)),
        requests={},
        execute_model_state=object(),
    )
    with pytest.raises(RuntimeError, match="State error"):
        GPUARModelRunner.execute_model(runner, "step")
    assert calls == ["step"]
