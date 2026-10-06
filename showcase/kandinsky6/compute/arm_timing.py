# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Request times for one or more gate runs, from their manifests.

The write-up quotes a median per arm, and a median read off a log by eye is how
a warm-up creeps into a steady-state number. This reads the manifests, drops the
warm-up (which the generator does not record as a run), and prints median, min
and max with the count, so a spread that invalidates a comparison is visible
next to the number that would have hidden it.

    python arm_timing.py /data/jooman/k6/results/gate-pro/*-setA
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def timings(run_dir: Path) -> dict[str, object] | None:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text())
    seconds = [run["seconds"] for run in manifest.get("runs", []) if "seconds" in run]
    if not seconds:
        return None
    return {
        "arm": manifest.get("arm", run_dir.name),
        "attention_config": Path(str(manifest.get("attention_config", ""))).name,
        "checkpoint": Path(str(manifest.get("checkpoint", ""))).name,
        "offload": manifest.get("offload"),
        "n": len(seconds),
        "median_s": statistics.median(seconds),
        "min_s": min(seconds),
        "max_s": max(seconds),
        "cold_start_s": manifest.get("cold_start_s"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    rows = [row for run_dir in args.run_dirs if (row := timings(run_dir)) is not None]
    if not rows:
        raise SystemExit("no manifest with timed runs among the given directories")
    rows.sort(key=lambda r: r["median_s"])

    width = max(len(str(r["arm"])) for r in rows)
    print(f"{'arm':<{width}}  {'n':>2}  {'median':>8}  {'min':>8}  {'max':>8}  {'spread':>7}  cold start")
    for row in rows:
        spread = (row["max_s"] - row["min_s"]) / row["median_s"] * 100
        cold = f"{row['cold_start_s']:.1f} s" if row["cold_start_s"] is not None else "-"
        print(
            f"{row['arm']:<{width}}  {row['n']:>2}  {row['median_s']:>7.1f}s  {row['min_s']:>7.1f}s  "
            f"{row['max_s']:>7.1f}s  {spread:>6.1f}%  {cold}"
        )
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(rows, indent=2) + "\n")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
