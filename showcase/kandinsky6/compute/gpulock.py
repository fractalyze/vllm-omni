# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Hold every GPU lock on the host, and name the foreign processes on the GPU.

The RTX 5090 of this showcase is shared with other studies and other
people's jobs. Each study keeps its own advisory lock file, so a timed run
has to take *all* of them, not only its own, and must still check
``nvidia-smi``: a job that ignores the locks shows up there and invalidates
the run.

Lock paths are discovered by glob so a study that appears later is picked up
without editing this file, and are taken in sorted order so two holders can
never deadlock against each other.

Usage as a library::

    from gpulock import GpuLocks, foreign_gpu_procs
    with GpuLocks() as locks:
        assert not foreign_gpu_procs(), "foreign process on the GPU"
        ...                                      # timed work

Usage as a command (prints what it would hold, then runs a command under the
locks)::

    python gpulock.py --list
    python gpulock.py -- python block_profile.py ...
"""

from __future__ import annotations

import argparse
import fcntl
import glob
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

# Globs, not a fixed list: a new study drops its own ``gpu.lock`` and is
# honoured without a code change here.
LOCK_GLOBS = (
    "/data/jooman/*/gpu.lock",
    "/data/jooman/*/*/gpu.lock",
    str(Path.home() / ".cache/*/*/gpu.lock"),
)
# This showcase's own lock. Created on first use so the other track's host
# and this one agree on the path even before either has run.
OWN_LOCK = "/data/jooman/k6/gpu.lock"

# `nvidia-smi` lists the X server on a desktop-attached GPU. It holds a few
# MiB and never competes for SMs, so it is not a foreign compute job.
_IGNORED_PROCESS_NAMES = ("/usr/lib/xorg/Xorg", "Xorg", "/usr/bin/X")


@dataclass(frozen=True)
class GpuProc:
    pid: int
    name: str
    used_mib: int

    def __str__(self) -> str:
        return f"pid {self.pid} {self.name} ({self.used_mib} MiB)"


def _own_pids() -> set[int]:
    """This process and its ancestors, which must not count as foreign."""
    pids: set[int] = set()
    pid = os.getpid()
    while pid > 1 and pid not in pids:
        pids.add(pid)
        try:
            stat = Path(f"/proc/{pid}/stat").read_text()
        except OSError:
            break
        # The comm field may contain spaces and parentheses; ppid is the
        # field after the final ')'.
        pid = int(stat[stat.rindex(")") + 2 :].split()[1])
    return pids


def foreign_gpu_procs(include_own_tree: bool = False) -> list[GpuProc]:
    """Compute processes on the GPU that this process did not start.

    ``include_own_tree=True`` reports every compute process, which is what a
    log line wants; the default answers "is the GPU mine right now?".
    """
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:  # pragma: no cover - needs a GPU
        raise RuntimeError(f"nvidia-smi query failed: {exc}") from exc

    mine = set() if include_own_tree else _own_pids()
    procs = []
    for line in out.splitlines():
        if not line.strip():
            continue
        pid_s, name, mem_s = (field.strip() for field in line.split(",", 2))
        pid = int(pid_s)
        if pid in mine or name in _IGNORED_PROCESS_NAMES:
            continue
        procs.append(GpuProc(pid=pid, name=name, used_mib=int(mem_s)))
    return procs


def lock_paths() -> list[str]:
    """Every lock file to hold, sorted. Missing directories are skipped."""
    found = {OWN_LOCK}
    for pattern in LOCK_GLOBS:
        found.update(glob.glob(pattern))
    return sorted(found)


class GpuLocks:
    """Exclusive ``flock`` on every GPU lock file on the host.

    A lock whose parent directory does not exist is skipped rather than
    created: the directory belongs to another study, and inventing it would
    hide a typo in that study's path.
    """

    def __init__(self, paths: list[str] | None = None, blocking: bool = True):
        self.paths = paths if paths is not None else lock_paths()
        self.blocking = blocking
        self.held: list[str] = []
        self._files: list = []

    def __enter__(self) -> GpuLocks:
        mode = fcntl.LOCK_EX if self.blocking else fcntl.LOCK_EX | fcntl.LOCK_NB
        try:
            for path in self.paths:
                if not Path(path).parent.is_dir():
                    continue
                handle = open(path, "a+")  # noqa: SIM115 - held for the block's lifetime
                fcntl.flock(handle.fileno(), mode)
                self._files.append(handle)
                self.held.append(path)
        except BaseException:
            self.__exit__(*sys.exc_info())
            raise
        return self

    def __exit__(self, *exc_info) -> None:
        while self._files:
            handle = self._files.pop()
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            finally:
                handle.close()
        self.held = []


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list", action="store_true", help="print the lock files and the GPU's processes, then exit")
    parser.add_argument("--non-blocking", action="store_true", help="fail instead of waiting for a held lock")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="command to run while the locks are held")
    args = parser.parse_args()

    if args.list:
        for path in lock_paths():
            exists = "exists" if Path(path).exists() else "absent"
            usable = "usable" if Path(path).parent.is_dir() else "no parent dir"
            print(f"{path}\t{exists}\t{usable}")
        for proc in foreign_gpu_procs(include_own_tree=True):
            print(f"gpu\t{proc}")
        return 0

    command = args.command[1:] if args.command and args.command[0] == "--" else args.command
    if not command:
        parser.error("nothing to run: pass a command after `--`, or use --list")

    with GpuLocks(blocking=not args.non_blocking) as locks:
        print(f"holding {len(locks.held)} lock(s)", file=sys.stderr)
        return subprocess.run(command).returncode


if __name__ == "__main__":
    raise SystemExit(main())
