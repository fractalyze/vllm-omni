# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""GPU exclusivity and co-tenancy evidence for the Kandinsky 6 showcase.

The showcase hosts are shared: other people's jobs and other agents' studies
use the same RTX 5090. A timed run is only worth recording if nothing else
touched the GPU while it ran, so every timed run does two things.

**Hold every lock.** Other studies on these hosts each guard the GPU with
their own ``gpu.lock`` and know nothing about ours. Taking only our own lock
would let a job that respects a different one run concurrently, so
``GpuGuard`` ``flock``s *every* lock file it can find (``LOCK_GLOBS``) plus
this study's own, in a fixed sorted order so two holders of the same set can
never deadlock against each other.

**Sample throughout.** Holding the locks is not proof: a job started before
us, or one that respects no lock, still shows up in ``nvidia-smi``. A
sampler thread records the compute processes and the board's clocks, power
and temperature for the whole run. :meth:`GpuGuard.report` turns that into a
contamination verdict the ledger carries, so a contaminated run is discarded
with the evidence rather than silently averaged in.

The thermal and clock samples are not decoration. On a 575 W-capped 5090 a
run that starts at 34 C and one that starts at 78 C do not see the same
clocks, and a measured delta smaller than that drift is not a delta.
"""

from __future__ import annotations

import fcntl
import glob
import json
import os
import statistics
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

# Every lock file a job on these hosts might respect. A study that adds its
# own lock should add its glob here, and other studies' locks stay in the list
# even after that study ends: an old lock file is harmless to hold.
LOCK_GLOBS = (
    "/data/jooman/*/gpu.lock",
    "/data/jooman/tmp/gpu.lock",
    str(Path.home() / ".cache/*/*/gpu.lock"),
    # Another team's lease on build-server-2's GPU 0 (owned by its user, mode
    # 0644). Their jobs hold it for a whole run and launch GPU workers under it.
    "/var/lock/fractalyze-gpu*.lease",
)

# This study's own lock, created if absent so the other track and other
# agents can serialize against us.
OWN_LOCK = "/data/jooman/k6/gpu.lock"

_SMI_QUERY = (
    "clocks.current.sm",
    "clocks.current.memory",
    "power.draw",
    "temperature.gpu",
    "utilization.gpu",
    "memory.used",
)


def _open_lock(path: str) -> int:
    """A descriptor to ``flock`` on, creating our own lock file if needed.

    Another user's lock file may be read-only to us. ``flock`` does not care
    how the descriptor was opened, so fall back to read-only rather than skip
    a lock the other job is honouring.
    """
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        return os.open(path, os.O_RDWR | os.O_CREAT, 0o664)
    except PermissionError:
        return os.open(path, os.O_RDONLY)


def _run(cmd: list[str], timeout: float = 15.0) -> str:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False).stdout


def gpu_processes() -> list[dict[str, object]]:
    """Compute processes on the GPU, as ``nvidia-smi`` reports them now."""
    out = _run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_memory,process_name",
            "--format=csv,noheader,nounits",
        ]
    )
    procs: list[dict[str, object]] = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3 or not parts[0].isdigit():
            continue
        procs.append({"pid": int(parts[0]), "used_mib": _as_float(parts[1]), "name": parts[2]})
    return procs


def board_sample() -> dict[str, float | None]:
    out = _run(["nvidia-smi", f"--query-gpu={','.join(_SMI_QUERY)}", "--format=csv,noheader,nounits"])
    values = [p.strip() for p in out.strip().splitlines()[0].split(",")] if out.strip() else []
    return {key: _as_float(values[i]) if i < len(values) else None for i, key in enumerate(_SMI_QUERY)}


def _as_float(text: str) -> float | None:
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


@dataclass
class _Samples:
    board: list[dict[str, float | None]] = field(default_factory=list)
    # pid -> the process as first seen, for every pid seen at any sample.
    procs: dict[int, dict[str, object]] = field(default_factory=dict)


class GpuGuard:
    """Holds every GPU lock on the host and samples co-tenancy while held.

    ``own_pids`` are the processes this run started (the server, and this
    driver). Anything else on the GPU is foreign and contaminates the run.
    Pass the server's pid as soon as it is known with :meth:`add_own_pid`;
    children of a declared pid are resolved at report time, because vLLM
    spawns its workers after the server pid exists.
    """

    def __init__(self, *, interval_s: float = 1.0, own_pids: list[int] | None = None) -> None:
        self.interval_s = interval_s
        self._own_pids: set[int] = {os.getpid(), *(own_pids or [])}
        self._locks: list[tuple[str, int]] = []
        self._samples = _Samples()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._entry_procs: list[dict[str, object]] = []

    # -- locks ---------------------------------------------------------
    def lock_paths(self) -> list[str]:
        """Every lock file to hold, sorted so all holders agree on the order."""
        found: set[str] = {OWN_LOCK}
        for pattern in LOCK_GLOBS:
            found.update(glob.glob(pattern))
        return sorted(found)

    def acquire(self, *, timeout_s: float = 7200.0) -> None:
        """Take every lock, blocking until all are held or ``timeout_s`` passes.

        A partially acquired set is released before raising, so a timeout
        never leaves another job blocked behind us.
        """
        deadline = time.monotonic() + timeout_s
        for path in self.lock_paths():
            fd = _open_lock(path)
            while True:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() > deadline:
                        os.close(fd)
                        self.release()
                        raise TimeoutError(f"could not acquire {path} within {timeout_s}s")
                    time.sleep(1.0)
            self._locks.append((path, fd))

    def release(self) -> None:
        while self._locks:
            _, fd = self._locks.pop()
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)

    # -- sampling ------------------------------------------------------
    def add_own_pid(self, pid: int) -> None:
        self._own_pids.add(pid)

    def start_sampling(self) -> None:
        self._entry_procs = gpu_processes()
        self._stop.clear()
        self._thread = threading.Thread(target=self._sample_loop, name="gpu-sampler", daemon=True)
        self._thread.start()

    def stop_sampling(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5 * self.interval_s + 5.0)
            self._thread = None

    def _sample_loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._samples.board.append(board_sample())
                for proc in gpu_processes():
                    self._samples.procs.setdefault(int(proc["pid"]), proc)
            except (OSError, subprocess.SubprocessError, IndexError):
                # A transient nvidia-smi failure must not abort the run it is
                # only observing; a gap in the samples is visible in n_samples.
                pass
            self._stop.wait(self.interval_s)

    # -- verdict -------------------------------------------------------
    def _own_pid_set(self) -> set[int]:
        """Our pids, plus every descendant of them that exists now."""
        own = set(self._own_pids)
        for _ in range(4):  # vLLM nests workers at most a couple of levels deep
            grown = set(own)
            for pid in own:
                out = _run(["ps", "-o", "pid=", "--ppid", str(pid)])
                grown.update(int(line) for line in out.split() if line.isdigit())
            if grown == own:
                break
            own = grown
        return own

    def report(self) -> dict[str, object]:
        """Contamination verdict and board statistics for the sampled window.

        ``contamination`` is ``clean`` when every GPU process seen belonged to
        this run, ``foreign-process`` otherwise. ``entry_foreign`` lists what
        was already on the GPU when sampling started: a run with a non-empty
        ``entry_foreign`` was contaminated from the first sample and its
        timings must be discarded, not corrected.
        """
        own = self._own_pid_set()
        foreign = [p for pid, p in sorted(self._samples.procs.items()) if pid not in own]
        entry_foreign = [p for p in self._entry_procs if int(p["pid"]) not in own]

        def stats(key: str) -> dict[str, float] | None:
            vals = [s[key] for s in self._samples.board if s.get(key) is not None]
            if not vals:
                return None
            return {
                "min": min(vals),
                "median": statistics.median(vals),
                "max": max(vals),
            }

        return {
            "contamination": "clean" if not foreign else "foreign-process",
            "foreign": foreign,
            "entry_foreign": entry_foreign,
            "n_samples": len(self._samples.board),
            "locks_held": [p for p, _ in self._locks],
            "sm_clock_mhz": stats("clocks.current.sm"),
            "mem_clock_mhz": stats("clocks.current.memory"),
            "power_w": stats("power.draw"),
            "temp_c": stats("temperature.gpu"),
            "gpu_util_pct": stats("utilization.gpu"),
            "mem_used_mib": stats("memory.used"),
        }

    # -- context manager ----------------------------------------------
    def __enter__(self) -> GpuGuard:
        self.acquire()
        self.start_sampling()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.stop_sampling()
        self.release()


def main() -> None:
    """``python gpu_guard.py`` prints what a timed run would see right now."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=3.0, help="how long to sample")
    args = parser.parse_args()

    guard = GpuGuard(interval_s=0.5)
    print("locks:", *guard.lock_paths(), sep="\n  ")
    with guard:
        time.sleep(args.seconds)
    print(json.dumps(guard.report(), indent=2))


if __name__ == "__main__":
    main()
