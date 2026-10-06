# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Ledger rows the world-model vault's ingest can read.

The vault is a strict experiment-record database: ``scripts/ingest_ledger.py``
projects an L0 ledger into L1 trial fields, and never lets a person type a
number a ledger could supply. This module writes that L0 ledger for the
Kandinsky 6 showcase, in the shape of the vault's ``gemma4nv-gate`` adapter —
one JSON object per ABBA run, with a ``metrics`` dict, a ``verdict`` dict and a
``validity`` the adapter maps to a measurement role.

Four fields carry the measurement discipline, and they are the reason a row is
worth more than a number in a terminal:

``validity``
    ``valid`` only when the run's :class:`gpu_guard.GpuGuard` report said
    ``clean``. A run that shared the GPU is written as ``contaminated`` rather
    than dropped: a discarded run that leaves no trace cannot stop someone
    re-deriving the same wrong conclusion from the surviving ones.

``n_pairs`` and the spread
    An ABBA run's unit is the *pair*, not the request. Medians, min and max
    travel together, because a delta inside the control's own spread is null
    and a reader who has only the median cannot tell.

``control`` / ``candidate`` and their commits
    Both arms of the comparison, each with the commit it ran, so the row is
    self-contained once the session is gone.

``profiled``
    True for runs with the stage profiler on. Those runs' wall times are not
    walls (the profiler synchronizes around every stage) and must never be
    promoted into a headline.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import socket
import statistics
import subprocess
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

LEDGER_RELPATH = Path("ledger/evaluations.jsonl")
STUDY = "k6-5090"


def _git_commit(repo: Path) -> str | None:
    out = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    return out.stdout.strip() or None


def _git_dirty(repo: Path) -> bool:
    out = subprocess.run(["git", "-C", str(repo), "status", "--porcelain"], capture_output=True, text=True, check=False)
    return bool(out.stdout.strip())


def run_id(tag: str, host: str | None = None) -> str:
    """``<tag>-<YYYYmmdd-HHMMSS>-<host>-<rand6>``, as the vault's ids read."""
    host = host or socket.gethostname()
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%d-%H%M%S")
    return f"{tag}-{stamp}-{host}-{uuid.uuid4().hex[:6]}"


def spread(values: list[float]) -> dict[str, float | int]:
    """Median with min and max, the showcase's required reporting form."""
    return {
        "n": len(values),
        "median": float(statistics.median(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


@dataclass
class LedgerRow:
    """One run of the harness: one comparison, or one single-arm measurement."""

    run: str
    control: str
    candidate: str
    metrics: dict[str, Any]
    verdict: dict[str, Any]
    validity: str = "valid"
    status: str = "completed"
    profiled: bool = False
    n_pairs: int = 0
    workload: str = "W1"
    host: str = field(default_factory=socket.gethostname)
    time: str = field(default_factory=lambda: dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%MZ"))
    control_commit: str | None = None
    candidate_commit: str | None = None
    harness_commit: str | None = None
    gpu: dict[str, Any] = field(default_factory=dict)
    accuracy: dict[str, Any] = field(default_factory=dict)
    arms: dict[str, Any] = field(default_factory=dict)
    env: dict[str, Any] = field(default_factory=dict)
    output: str | None = None
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"study": STUDY, **{k: v for k, v in self.__dict__.items()}}


def environment_fingerprint(venv_python: Path | None = None) -> dict[str, Any]:
    """Library versions a measurement depends on, recorded with the run.

    The showcase's STATUS.md names the versions once; a ledger row repeats them
    because a row read a month later has no STATUS.md beside it, and a torch or
    vLLM bump between two rows is the first thing to suspect when they
    disagree.
    """
    code = (
        "import json,platform;"
        "d={'python':platform.python_version()};"
        "\nfor m in ('torch','vllm','vllm_omni','diffusers','transformers','flashinfer'):\n"
        "    try:\n"
        "        d[m]=__import__(m).__version__\n"
        "    except Exception as e:\n"
        "        d[m]=None\n"
        "try:\n"
        "    import torch;d['cuda']=torch.version.cuda;d['capability']=list(torch.cuda.get_device_capability())\n"
        "except Exception:\n"
        "    pass\n"
        "print(json.dumps(d))"
    )
    python = str(venv_python) if venv_python else "python3"
    out = subprocess.run([python, "-c", code], capture_output=True, text=True, check=False)
    try:
        versions = json.loads(out.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        versions = {"error": out.stderr.strip()[-500:]}

    driver = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    return {
        "versions": versions,
        "driver": driver or None,
        "hf_home": os.environ.get("HF_HOME"),
    }


class Ledger:
    """Append-only JSONL under ``root``, plus one artifact directory per run."""

    def __init__(self, root: Path, repo: Path | None = None) -> None:
        self.root = Path(root)
        self.repo = Path(repo) if repo else Path(__file__).resolve().parents[3]
        self.path = self.root / LEDGER_RELPATH
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def run_dir(self, run: str) -> Path:
        path = self.root / "runs" / run
        path.mkdir(parents=True, exist_ok=True)
        return path

    def harness_commit(self) -> str | None:
        commit = _git_commit(self.repo)
        # A dirty tree makes the commit a lie about what ran; say so in the id
        # rather than recording a commit that does not contain the code.
        return f"{commit}-dirty" if commit and _git_dirty(self.repo) else commit

    def append(self, row: LedgerRow) -> Path:
        if row.harness_commit is None:
            row.harness_commit = self.harness_commit()
        with self.path.open("a") as handle:
            handle.write(json.dumps(row.as_dict(), sort_keys=False) + "\n")
        return self.path

    def write_report(self, run: str, report: dict[str, Any]) -> Path:
        """The run's full detail, which the row points at but does not inline."""
        path = self.run_dir(run) / "report.json"
        path.write_text(json.dumps(report, indent=2, sort_keys=False))
        return path

    def rows(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]


def validity_from_guard(guard_report: dict[str, Any]) -> str:
    """``valid`` only for a clean GPU; otherwise ``contaminated``."""
    return "valid" if guard_report.get("contamination") == "clean" else "contaminated"
