# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""One prompt across several arms, as a PNG: one labelled row per arm.

The gate's numbers say how far an arm moved from the reference; this shows
what moved. Each row takes the same frame indices from its arm's MP4 for one
prompt id, so a column compares the same instant across arms. Frames are
spread over the clip, plus any ``--frame`` given (e.g. the worst LPIPS frame
from a gate report).

Usage::

    python compare_sheet.py --prompt b6-train-platform --out b6.png \\
        --arm "BF16 eager=/data/jooman/k6/ref-eager/setB" \\
        --arm "INT8+Sage2+exact1=/data/jooman/k6/arms/int8-sage2-exact1/setB"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from quality import read_video  # noqa: E402

LABEL_WIDTH = 220


def frame_indices(n_frames: int, columns: int, extra: list[int]) -> list[int]:
    """``columns`` indices spread over ``[0, n_frames)``, plus ``extra``, sorted and unique."""
    spread = [round(i * (n_frames - 1) / (columns - 1)) for i in range(columns)] if columns > 1 else [0]
    return sorted({*spread, *(i for i in extra if 0 <= i < n_frames)})


def build_sheet(rows: list[tuple[str, np.ndarray]], indices: list[int], *, width: int = 216):
    """``rows`` of (label, frames ``(T, H, W, 3)``) -> a PIL image, one row each."""
    from PIL import Image, ImageDraw

    height = int(width * rows[0][1].shape[1] / rows[0][1].shape[2])
    sheet = Image.new("RGB", (LABEL_WIDTH + width * len(indices), height * len(rows)), "white")
    draw = ImageDraw.Draw(sheet)
    for r, (label, frames) in enumerate(rows):
        y = r * height
        draw.text((6, y + height // 2 - 6), label, fill="black")
        for c, index in enumerate(indices):
            thumb = Image.fromarray(frames[min(index, len(frames) - 1)]).resize((width, height))
            sheet.paste(thumb, (LABEL_WIDTH + c * width, y))
    return sheet


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prompt", required=True, help="prompt id; each arm directory holds <id>.mp4")
    parser.add_argument("--arm", action="append", required=True, help="LABEL=DIRECTORY, one row each, in order")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--columns", type=int, default=5)
    parser.add_argument("--frame", type=int, action="append", default=[], help="an extra frame index to include")
    args = parser.parse_args()

    rows = []
    for spec in args.arm:
        label, _, directory = spec.partition("=")
        rows.append((label, read_video(Path(directory) / f"{args.prompt}.mp4")))
    indices = frame_indices(min(len(f) for _, f in rows), args.columns, args.frame)
    build_sheet(rows, indices).save(args.out)
    print(f"{args.out}: {len(rows)} arms x frames {indices}")


if __name__ == "__main__":
    main()
