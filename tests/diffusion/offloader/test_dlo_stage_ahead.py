# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Background staging for DLO rank-local mmap: the hand-over rules.

Every rank-local mmap hook shares two host staging slots. VLLM_OMNI_DLO_STAGE_AHEAD
moves the pack of the block after next onto one worker thread, which must never
write a slot while another pack (background or foreground) uses it, and must
hand its result only to the hook and slot it was staged for.
"""

from __future__ import annotations

import os
import threading
import time
from unittest import mock

import torch
from absl.testing import absltest

from vllm_omni.diffusion.offloader.distributed_layerwise_backend import (
    DLO_STAGE_AHEAD_ENV,
    StagingAhead,
    stage_ahead_enabled,
)


class FakeHook:
    """Stands in for a hook: records which slots it packed and on which thread."""

    def __init__(self, name: str, delay_s: float = 0.0, log: list | None = None) -> None:
        self.name = name
        self.delay_s = delay_s
        self.log = log if log is not None else []
        self.threads: list[str] = []

    def _stage_mmap_sources(self, slot: int) -> dict[torch.dtype, torch.Tensor]:
        self.log.append(("start", self.name, slot))
        self.threads.append(threading.current_thread().name)
        time.sleep(self.delay_s)
        self.log.append(("end", self.name, slot))
        return {torch.bfloat16: torch.full((4,), float(slot))}


class StagingAheadTest(absltest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.ahead = StagingAhead()
        self.addCleanup(self.ahead.shutdown)

    def test_hands_the_pack_to_its_own_hook_and_slot(self) -> None:
        hook = FakeHook("b2")
        self.ahead.submit(hook, 1)
        staged = self.ahead.take(hook, 1)
        self.assertIsNotNone(staged)
        self.assertTrue(torch.equal(staged[torch.bfloat16], torch.full((4,), 1.0)))
        self.assertTrue(hook.threads[0].startswith("dlo-stage-ahead"))

    def test_another_hook_or_slot_waits_and_gets_nothing(self) -> None:
        log: list = []
        hook = FakeHook("b2", delay_s=0.05, log=log)
        self.ahead.submit(hook, 1)
        self.assertIsNone(self.ahead.take(FakeHook("other"), 1))
        self.assertIn(("end", "b2", 1), log)  # waited for the pending pack
        self.ahead.submit(hook, 0)
        self.assertIsNone(self.ahead.take(hook, 1))

    def test_packs_never_overlap(self) -> None:
        log: list = []
        first, second = FakeHook("b2", 0.05, log), FakeHook("b3", 0.0, log)
        self.ahead.submit(first, 1)
        self.ahead.submit(second, 0)  # must wait for b2's pack before starting
        self.ahead.take(second, 0)
        self.assertEqual(
            [entry[:2] for entry in log],
            [("start", "b2"), ("end", "b2"), ("start", "b3"), ("end", "b3")],
        )

    def test_nothing_pending_returns_none(self) -> None:
        self.assertIsNone(self.ahead.take(FakeHook("b2"), 0))


class SwitchTest(absltest.TestCase):
    def test_off_by_default(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(DLO_STAGE_AHEAD_ENV, None)
            self.assertFalse(stage_ahead_enabled())

    def test_on(self) -> None:
        with mock.patch.dict(os.environ, {DLO_STAGE_AHEAD_ENV: "1"}):
            self.assertTrue(stage_ahead_enabled())


if __name__ == "__main__":
    absltest.main()
