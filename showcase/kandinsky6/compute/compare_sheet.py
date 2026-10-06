# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""A contact sheet comparing several arms on the same prompt, for a human look.

The last step of a quality gate is somebody looking at the frames, and the thing
worth looking at is not an arm against the reference -- `clip_gate.py` already
writes that -- but several arms against each other on the prompt where the
numbers disagree. LPIPS says `a8-skateboard-crash` is three times worse than
`a5-waterfall-drone` under the same kernel; whether that is visible, and what it
looks like, is not something the number answers.

    python compare_sheet.py --prompt a8-skateboard-crash \
        --arm "BF16 reference=/data/jooman/k6/ref-eager/setA" \
        --arm "Sage2 everywhere=/data/jooman/k6/results/gate-pro/bf16-stream-sage2" \
        --out sheet-a8.jpg

One row per arm, the same evenly spaced frame indices in every row, so a column
is one moment in the clip across all the arms.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from clip_gate import contact_sheet, sample_frames  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prompt", required=True, help="prompt id, e.g. a8-skateboard-crash")
    parser.add_argument(
        "--arm",
        action="append",
        required=True,
        metavar="LABEL=DIR",
        help="a labelled directory of MP4s named by prompt id; repeat for each arm",
    )
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    missing = []
    for entry in args.arm:
        if "=" not in entry:
            raise SystemExit(f"--arm needs LABEL=DIR, got {entry!r}")
        label, directory = entry.split("=", 1)
        mp4 = Path(directory) / f"{args.prompt}.mp4"
        if not mp4.is_file():
            missing.append(str(mp4))
            continue
        rows.append((f"{label}  [{args.prompt}]", sample_frames(mp4, args.frames)))

    if missing:
        # Named, not skipped: a sheet with a row quietly absent invites the
        # reader to compare two arms believing they are looking at three.
        print("missing: " + "; ".join(missing), file=sys.stderr)
    if not rows:
        raise SystemExit("no arm had this prompt")

    contact_sheet(rows, args.out)
    print(f"{len(rows)} arm(s) x {args.frames} frames -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
