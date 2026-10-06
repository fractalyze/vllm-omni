# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""abba.visit_order: every arm's visits are symmetric about the session's midpoint."""

from __future__ import annotations

import sys
from pathlib import Path

from absl.testing import absltest, parameterized

sys.path.insert(0, str(Path(__file__).resolve().parent))

from abba import visit_order  # noqa: E402


class VisitOrderTest(parameterized.TestCase):
    def test_two_arms_is_abba(self) -> None:
        self.assertEqual(visit_order(2), [0, 1, 1, 0])

    @parameterized.parameters(2, 3, 4)
    def test_linear_drift_cancels(self, n_arms: int) -> None:
        order = visit_order(n_arms)
        midpoint = (len(order) - 1) / 2
        for arm in range(n_arms):
            positions = [i for i, a in enumerate(order) if a == arm]
            self.assertLen(positions, 2)
            self.assertAlmostEqual(sum(positions) / 2, midpoint)


if __name__ == "__main__":
    absltest.main()
