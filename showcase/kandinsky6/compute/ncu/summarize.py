# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""summarize.py <report.ncu-rep>...: one table row per (kernel, grid, block), time-weighted over its launches.

Columns are the fields of the round-6 roofline table: time per launch, SOL SM and memory, tensor-pipe activity,
DRAM and L2 throughput, achieved occupancy, shared-memory bank conflicts and the three largest warp stalls.
"""

import csv
import io
import re
import subprocess
import sys
from collections import defaultdict

NCU = "/usr/local/cuda/bin/ncu"
FIELDS = {
    "t": "gpu__time_duration.sum",
    "sm": "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "mem": "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
    "dram": "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "l2": "lts__t_sectors.avg.pct_of_peak_sustained_elapsed",
    "tc": "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",
    "occ": "sm__warps_active.avg.pct_of_peak_sustained_active",
    "grid": "launch__grid_size",
    "blk": "launch__block_size",
    "bank": "l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum",
}
TO_MS = {"s": 1e3, "ms": 1.0, "us": 1e-3, "ns": 1e-6}
STALL = re.compile(r"^smsp__average_warps_issue_stalled_(\w+)_per_issue_active\.ratio$")


def number(text: str) -> float:
    text = text.replace(",", "")
    return float(text) if re.fullmatch(r"[\d.]+", text) else 0.0


def launches(report: str):
    """Yield (group key, ms, metrics, stalls) for every profiled launch in ``report``."""
    out = subprocess.run([NCU, "--import", report, "--csv", "--page", "raw"], capture_output=True, text=True).stdout
    rows = list(csv.reader(io.StringIO(out)))
    header, units = rows[0], rows[1]
    to_ms = TO_MS[units[header.index(FIELDS["t"])]]
    for row in rows[2:]:
        metrics = {k: number(row[header.index(v)]) for k, v in FIELDS.items() if v in header}
        name = re.sub(r"\(.*", "", row[header.index("Kernel Name")])[:70]
        stalls = [(m.group(1), number(row[i])) for i, h in enumerate(header) if (m := STALL.match(h))]
        key = (name, row[header.index(FIELDS["grid"])], row[header.index(FIELDS["blk"])])
        yield key, metrics["t"] * to_ms, metrics, stalls


def main() -> None:
    groups = defaultdict(list)
    for report in sys.argv[1:]:
        for key, ms, metrics, stalls in launches(report):
            groups[key].append((ms, metrics, stalls))
    print(
        "kernel | grid x block | n | ms/launch | SOL SM % | SOL mem % | tensor % | DRAM % | L2 % | occ % | "
        "bank conflicts/launch | top stalls"
    )
    for (name, grid, block), group in sorted(groups.items(), key=lambda kv: -sum(x[0] for x in kv[1])):
        total = sum(ms for ms, _, _ in group) or 1.0

        def weighted(field: str, group=group, total=total) -> float:
            return sum(ms * metrics.get(field, 0.0) for ms, metrics, _ in group) / total

        stall_sum = defaultdict(float)
        for ms, _, stalls in group:
            for reason, value in stalls:
                stall_sum[reason] += value * ms / total
        top = ", ".join(f"{r} {v:.1f}" for r, v in sorted(stall_sum.items(), key=lambda kv: -kv[1])[:3])
        cols = [f"{weighted(f):.0f}" for f in ("sm", "mem", "tc", "dram", "l2", "occ", "bank")]
        print(
            f"{name} | {grid} x {block} | {len(group)} | {total / len(group):.3f} | " + " | ".join(cols) + f" | {top}"
        )


if __name__ == "__main__":
    main()
