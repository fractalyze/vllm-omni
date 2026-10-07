# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Summarize an Nsight Compute export into one row per kernel launch (the round-6 roofline table).

    python ncu_table.py /path/report.ncu-rep [more.ncu-rep ...]

Per launch: duration (us), SOL SM %, SOL memory %, DRAM %, tensor-pipe active %,
achieved occupancy, grid x block, and the top three warp-stall reasons (warps
stalled per issued instruction, from the raw page). Needs the SpeedOfLight,
Occupancy, LaunchStats and WarpStateStats sections and the
``sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active`` metric.
"""

from __future__ import annotations

import csv
import io
import re
import subprocess
import sys

NCU = "/usr/local/cuda/bin/ncu"
SOL = "GPU Speed Of Light Throughput"
SOL_KEYS = {"Compute (SM) Throughput": "sm", "Memory Throughput": "mem", "DRAM Throughput": "dram"}
STALL = re.compile(r"^smsp__average_warps_issue_stalled_(.+)_per_issue_active\.ratio$")


def _csv(rep: str, page: str) -> list[dict]:
    out = subprocess.run([NCU, "--import", rep, "--csv", "--page", page], capture_output=True, text=True).stdout
    return list(csv.reader(io.StringIO(out))) if page == "raw" else list(csv.DictReader(io.StringIO(out)))


def _num(value: str) -> float:
    return float(value.replace(",", ""))


def kernels(rep: str) -> list[dict]:
    rows: dict[str, dict] = {}
    for r in _csv(rep, "details"):
        k = rows.setdefault(r["ID"], {"name": r["Kernel Name"]})
        sec, metric, unit, value = r["Section Name"], r["Metric Name"], r["Metric Unit"], r["Metric Value"]
        if sec == SOL and metric == "Duration":
            k["us"] = _num(value) * {"ns": 1e-3, "us": 1.0, "ms": 1e3}[unit]
        elif sec == SOL and metric in SOL_KEYS:
            k[SOL_KEYS[metric]] = _num(value)
        elif metric == "Achieved Occupancy":
            k["occ"] = _num(value)
        elif metric in ("Grid Size", "Block Size"):
            k[metric] = value
        elif metric == "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active":
            k["tensor"] = _num(value)
    raw = _csv(rep, "raw")
    if raw:
        header, data = raw[0], raw[2:]  # row 1 holds units
        for line in data:
            rec = dict(zip(header, line))
            k = rows.get(rec.get("ID", ""))
            if k is None:
                continue
            stalls = []
            for col, value in rec.items():
                m = STALL.match(col)
                if m and value:
                    try:
                        stalls.append((m.group(1), _num(value)))
                    except ValueError:
                        pass
            k["stalls"] = sorted(stalls, key=lambda kv: -kv[1])[:3]
    return list(rows.values())


def short(name: str) -> str:
    name = re.sub(r"<.*", "", re.sub(r"^void ", "", name))
    return name.split("::")[-1][:40]


def main() -> None:
    for rep in sys.argv[1:]:
        print(f"## {rep}")
        print("| kernel | us | SM % | mem % | DRAM % | tensor % | occ % | grid x block | top stalls |")
        print("|---|---:|---:|---:|---:|---:|---:|---|---|")
        for k in kernels(rep):
            stalls = ", ".join(f"{n} {v:.1f}" for n, v in k.get("stalls", []))
            cells = [f"{k.get(c, float('nan')):.1f}" for c in ("us", "sm", "mem", "dram", "tensor", "occ")]
            launch = f"{k.get('Grid Size', '')} x {k.get('Block Size', '')}"
            print(f"| {short(k['name'])} | {' | '.join(cells)} | {launch} | {stalls} |")


if __name__ == "__main__":
    main()
