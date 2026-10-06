# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Wait for this shared GPU, then run a command under its locks.

The RTX 5090 of this showcase is shared, and a co-tenant can hold both the
study locks and almost all 32 GB for hours. A measurement cannot be taken in
that state and must not be faked, so this blocks until three things are true
at once and then runs the command:

1. every ``gpu.lock`` on the host is held by this process (``gpulock.py``),
2. ``nvidia-smi`` reports no foreign compute process,
3. at least ``--need-free-gib`` of device memory is free.

The locks are taken **blocking**, which is what makes this fair. An earlier
version took them non-blockingly and went back to sleep on failure; against
a co-tenant that re-takes its lock the moment it releases it, that starves
forever -- observed on this host, where the GPU came free and the lock was
gone again within the 30 s poll. A blocking ``flock`` puts this process in
the kernel's queue instead, so it gets a turn.

Taking a lock does not make the GPU empty: the co-tenant's previous process
may still be tearing down, or a job that ignores the locks may be running.
So once the locks are held, this waits up to ``--hold-wait-min`` for the
memory, and if it does not come, **releases the locks and queues again**
rather than holding them while idle -- otherwise one stuck job would deadlock
the whole host behind this process.

    python run_when_free.py --need-free-gib 12 -- python attn_race.py --json race.json

Exit codes: the command's, or 4 if ``--deadline-min`` passed without a turn
on a free GPU. Every poll logs to stderr, so a background run leaves a
record of how long the GPU was busy and who held it.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gpulock import GpuLocks, foreign_gpu_procs  # noqa: E402


def free_gib() -> float:
    """Free device memory, from ``nvidia-smi`` rather than from torch.

    Deliberately not torch: creating a CUDA context to ask would itself
    reserve memory, and on a full GPU the context creation is what fails.
    """
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return int(out.strip().splitlines()[0]) / 1024.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--need-free-gib", type=float, default=12.0, help="device memory the command needs")
    parser.add_argument("--poll-s", type=float, default=15.0, help="seconds between polls while holding the locks")
    parser.add_argument(
        "--hold-wait-min",
        type=float,
        default=10.0,
        help="how long to wait for free memory while holding the locks before releasing them and queueing again",
    )
    parser.add_argument("--deadline-min", type=float, default=240.0, help="give up after this many minutes")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    command = args.command[1:] if args.command and args.command[0] == "--" else args.command
    if not command:
        parser.error("nothing to run: pass a command after `--`")

    def log(message: str) -> None:
        print(f"{time.strftime('%H:%M:%S')} {message}", file=sys.stderr, flush=True)

    deadline = time.monotonic() + args.deadline_min * 60
    while time.monotonic() < deadline:
        log("queueing for the host's GPU locks")
        with GpuLocks() as locks:  # blocking: this is the fair queue
            log(f"holding {len(locks.held)} lock(s); waiting for the GPU to empty")
            give_up_holding = min(time.monotonic() + args.hold_wait_min * 60, deadline)
            while time.monotonic() < give_up_holding:
                foreign = foreign_gpu_procs()
                available = free_gib()
                if not foreign and available >= args.need_free_gib:
                    log(f"GPU free ({available:.1f} GiB); running the command")
                    # The command would block on the lock copies this
                    # process already holds, so it is told to skip taking
                    # them -- and told who holds them, so the run it records
                    # still names its locks rather than claiming it had none.
                    env = {
                        **os.environ,
                        "K6_LOCKS_HELD_BY": f"run_when_free.py pid {os.getpid()}: " + ", ".join(locks.held),
                    }
                    return subprocess.run([*command, "--no-locks"], env=env).returncode
                held = ", ".join(str(p) for p in foreign) or "none"
                log(f"holding locks, still busy: {available:.1f} GiB free, foreign: {held}")
                time.sleep(args.poll_s)
            log("GPU did not empty within the hold window; releasing the locks and queueing again")

    log(f"gave up after {args.deadline_min:.0f} min: never got a turn on a free GPU")
    return 4


if __name__ == "__main__":
    raise SystemExit(main())
