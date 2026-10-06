# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""GpuGuard locks: a read-only lock file is still held, and held exclusively."""

from __future__ import annotations

import fcntl
import os
import sys
import tempfile
from pathlib import Path
from unittest import mock

from absl.testing import absltest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gpu_guard  # noqa: E402


class LockTest(absltest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.dir = Path(tempfile.mkdtemp())
        self.foreign = self.dir / "fractalyze-gpu0.lease"
        self.foreign.touch()
        self.foreign.chmod(0o444)
        self.addCleanup(self.foreign.chmod, 0o644)
        patches = [
            mock.patch.object(gpu_guard, "LOCK_GLOBS", (str(self.dir / "*.lease"),)),
            mock.patch.object(gpu_guard, "OWN_LOCK", str(self.dir / "own" / "gpu.lock")),
        ]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)

    def test_read_only_lock_file_is_held(self) -> None:
        if os.geteuid() == 0:
            self.skipTest("root ignores file modes")
        guard = gpu_guard.GpuGuard()
        guard.acquire(timeout_s=5)
        self.addCleanup(guard.release)
        self.assertIn(str(self.foreign), [path for path, _ in guard._locks])
        other = os.open(self.foreign, os.O_RDONLY)
        self.addCleanup(os.close, other)
        with self.assertRaises(BlockingIOError):
            fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def test_own_lock_is_created(self) -> None:
        guard = gpu_guard.GpuGuard()
        guard.acquire(timeout_s=5)
        self.addCleanup(guard.release)
        self.assertTrue((self.dir / "own" / "gpu.lock").exists())


if __name__ == "__main__":
    absltest.main()
