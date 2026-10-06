# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Per-stage timings, read out of a server log.

vLLM-Omni's ``DiffusionPipelineProfilerMixin`` wraps named pipeline methods and
logs one line per call::

    [DiffusionPipelineProfiler] Kandinsky6TI2VAPipeline.vae.decode took 7.421337s

It also *sums* repeated calls of the same name into ``stage_durations``, which
is what the engine reports. Both views matter and they answer different
questions, so this module keeps them apart:

- :func:`parse_stage_log` returns every individual call, in order. The denoise
  loop calls the DiT once per step (twice per step under CFG), so the
  individual calls are the per-step series — which is where a warm-up effect,
  a thermal ramp or a weight-streaming stall shows up as a trend. A single
  summed number hides all three.
- :func:`summarize` aggregates that series per stage.

**A profiled run is not a wall.** The profiler calls
``current_omni_platform.synchronize()` before and after every wrapped call, so
it serializes the pipeline against the host and removes any overlap the real
path has. Stage numbers therefore come from their own runs, and the headline
request time comes from runs with the profiler off. :func:`summarize` carries
``profiled: true`` so a reader cannot mistake one for the other.
"""

from __future__ import annotations

import json
import re
import statistics
from dataclasses import dataclass
from pathlib import Path

PROFILER_LINE = re.compile(
    r"\[DiffusionPipelineProfiler\]\s+(?P<stage>[\w.\[\]]+)\s+took\s+(?P<seconds>[0-9.]+)s"
)

# Allocation failures the K6 recipe saw in VAE decode on an 80 GB H100. On 32
# GB they are the expected first symptom of a decode that no longer fits, and a
# run that only retried its way to success is not the same run as one that did
# not, so the count travels with the stage numbers.
RETRIED_ALLOC = re.compile(r"(?i)(CUDA out of memory|failed to allocate|allocation failure)")


@dataclass
class StageCall:
    """One wrapped call: the stage's name and how long that call took."""

    stage: str
    seconds: float
    line_no: int


def parse_stage_log(log_path: Path) -> list[StageCall]:
    """Every profiler-reported call in ``log_path``, in the order logged."""
    calls: list[StageCall] = []
    with log_path.open(errors="replace") as handle:
        for line_no, line in enumerate(handle, start=1):
            match = PROFILER_LINE.search(line)
            if match:
                calls.append(
                    StageCall(
                        stage=match.group("stage"),
                        seconds=float(match.group("seconds")),
                        line_no=line_no,
                    )
                )
    return calls


def count_retried_allocations(log_path: Path) -> int:
    with log_path.open(errors="replace") as handle:
        return sum(1 for line in handle if RETRIED_ALLOC.search(line))


def _short(stage: str) -> str:
    """``Kandinsky6TI2VAPipeline.vae.decode`` -> ``vae.decode``."""
    return stage.split(".", 1)[1] if "." in stage else stage


def summarize(log_path: Path, *, last_request_only: bool = True) -> dict[str, object]:
    """Aggregate one log's stage calls.

    With ``last_request_only`` (the default) only the calls after the last
    ``<Pipeline>.forward`` boundary are summarized, so a warm-up request in the
    same log does not pollute the breakdown. The pipeline's own ``forward``
    wrapper clears ``stage_durations`` at the start of each request, which is
    the same boundary.
    """
    calls = parse_stage_log(log_path)
    if last_request_only:
        starts = [i for i, c in enumerate(calls) if _short(c.stage) == "forward"]
        # `forward` is logged when it *returns*, so a request's calls are the
        # ones up to and including its own `forward` line: take the last block.
        if len(starts) >= 2:
            calls = calls[starts[-2] + 1 :]

    per_stage: dict[str, list[float]] = {}
    for call in calls:
        per_stage.setdefault(_short(call.stage), []).append(call.seconds)

    stages: dict[str, object] = {}
    for stage, values in per_stage.items():
        stages[stage] = {
            "n_calls": len(values),
            "total_s": sum(values),
            "median_s": statistics.median(values),
            "min_s": min(values),
            "max_s": max(values),
            # The per-call series, so a trend across denoise steps stays
            # visible: a summed total cannot show a ramp.
            "calls_s": values,
        }
    return {
        "profiled": True,
        "stages": stages,
        "retried_allocations": count_retried_allocations(log_path),
        "log": str(log_path),
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Summarize a server log's stage timings.")
    parser.add_argument("log", type=Path)
    parser.add_argument("--all-requests", action="store_true", help="include every request, not just the last")
    args = parser.parse_args()
    print(json.dumps(summarize(args.log, last_request_only=not args.all_requests), indent=2))


if __name__ == "__main__":
    main()
