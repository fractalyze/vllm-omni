# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""compare_sheet: same frame indices on every row, one row per arm."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from absl.testing import absltest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from compare_sheet import LABEL_WIDTH, build_sheet, frame_indices  # noqa: E402


class FrameIndicesTest(absltest.TestCase):
    def test_spread_covers_first_and_last(self) -> None:
        self.assertEqual(frame_indices(121, 5, []), [0, 30, 60, 90, 120])

    def test_extra_frames_are_merged_and_bounded(self) -> None:
        self.assertEqual(frame_indices(121, 3, [22, 60, 500]), [0, 22, 60, 120])


class BuildSheetTest(absltest.TestCase):
    def test_one_row_per_arm_and_same_column_per_index(self) -> None:
        black = np.zeros((10, 48, 96, 3), dtype=np.uint8)
        white = np.full((10, 48, 96, 3), 255, dtype=np.uint8)
        sheet = build_sheet([("a", black), ("b", white)], [0, 9], width=96)
        self.assertEqual(sheet.size, (LABEL_WIDTH + 2 * 96, 2 * 48))
        self.assertEqual(sheet.getpixel((LABEL_WIDTH + 100, 10)), (0, 0, 0))
        self.assertEqual(sheet.getpixel((LABEL_WIDTH + 100, 48 + 10)), (255, 255, 255))


if __name__ == "__main__":
    absltest.main()
