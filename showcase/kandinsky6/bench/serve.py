# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Launch, wait for, and stop one ``vllm serve`` for a showcase arm.

An *arm* is a named server configuration: a checkpoint, a set of CLI flags and
a set of environment switches. :class:`Arm` is the whole definition, so a
measurement can quote the arm it ran and a reader can rebuild it from the
ledger alone — which is the point, since the ledger is what survives the
session.

Two properties this module exists to guarantee:

**Cold start is measured, not waited out.** :func:`start_server` returns the
time from ``exec`` to the first successful ``/health``, because a change that
makes a request faster by loading more weights up front has moved cost, not
removed it, and the showcase reports cold start beside the request time.

**A stopped server is actually gone.** Weights of this size take tens of
seconds to free. Starting the next arm while the previous one still holds 20
GB gives the next arm a different memory budget and silently changes what it
measures, so :func:`stop_server` waits for the process to exit *and* for the
GPU to drain back to its idle allocation before returning.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from gpu_guard import board_sample

HEALTH_TIMEOUT_S = 1800.0
DRAIN_TIMEOUT_S = 180.0
# Idle board memory is a few MiB; anything under this is "drained". Generous
# enough to tolerate a display server, tight enough to catch a leaked arm.
DRAINED_MIB = 512.0


@dataclass
class Arm:
    """One server configuration, fully described."""

    name: str
    model: str
    """Hub id or local snapshot path passed to ``vllm serve``."""
    cli_args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    port: int = 8091
    notes: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "model": self.model,
            "cli_args": list(self.cli_args),
            "env": dict(self.env),
            "port": self.port,
            "notes": self.notes,
        }


@dataclass
class Server:
    arm: Arm
    process: subprocess.Popen
    log_path: Path
    cold_start_s: float
    base_url: str


def _health_ok(base_url: str) -> bool:
    try:
        with urllib.request.urlopen(f"{base_url}/health", timeout=5.0) as response:
            return 200 <= response.status < 300
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def start_server(
    arm: Arm,
    log_path: Path,
    *,
    venv_bin: Path | None = None,
    timeout_s: float = HEALTH_TIMEOUT_S,
) -> Server:
    """Start ``arm``'s server and block until ``/health`` answers.

    The server runs in its own process group so :func:`stop_server` can signal
    the whole tree: vLLM's engine core and diffusion workers are children, and
    signalling only the parent leaves them holding the GPU.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.setdefault("HF_HOME", "/data/jooman/hf")
    env.update(arm.env)
    if venv_bin is not None:
        env["PATH"] = f"{venv_bin}{os.pathsep}{env.get('PATH', '')}"
        vllm = str(venv_bin / "vllm")
    else:
        vllm = "vllm"

    cmd = [vllm, "serve", arm.model, "--omni", "--port", str(arm.port), *arm.cli_args]
    base_url = f"http://127.0.0.1:{arm.port}"

    t0 = time.perf_counter()
    with log_path.open("wb") as log:
        log.write(f"$ {' '.join(cmd)}\n".encode())
        for key in sorted(arm.env):
            log.write(f"$ env {key}={arm.env[key]}\n".encode())
        log.flush()
        process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)

    deadline = t0 + timeout_s
    while True:
        if _health_ok(base_url):
            break
        if process.poll() is not None:
            tail = log_path.read_text(errors="replace")[-4000:]
            raise RuntimeError(f"{arm.name}: server exited with {process.returncode}\n{tail}")
        if time.perf_counter() > deadline:
            stop_server(Server(arm, process, log_path, 0.0, base_url))
            raise TimeoutError(f"{arm.name}: /health did not answer within {timeout_s}s")
        time.sleep(0.25)

    return Server(arm, process, log_path, time.perf_counter() - t0, base_url)


def stop_server(server: Server, *, timeout_s: float = DRAIN_TIMEOUT_S) -> None:
    """Stop the server and wait for the GPU to drain before returning."""
    process = server.process
    if process.poll() is None:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            process.terminate()
        try:
            process.wait(timeout=timeout_s / 2)
        except subprocess.TimeoutExpired:
            # Our own server, started by us: escalating is in scope.
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                process.kill()
            process.wait(timeout=60.0)

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        used = board_sample().get("memory.used")
        if used is None or used <= DRAINED_MIB:
            return
        time.sleep(1.0)


def server_pids(server: Server) -> list[int]:
    """The server's pid, for :meth:`gpu_guard.GpuGuard.add_own_pid`."""
    return [server.process.pid]


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Start one arm's server and report cold start.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--name", default="manual")
    parser.add_argument("--port", type=int, default=8091)
    parser.add_argument("--log", type=Path, default=Path("server.log"))
    parser.add_argument("--venv-bin", type=Path, default=None)
    parser.add_argument("--keep", action="store_true", help="leave the server running and print its pid")
    parser.add_argument("cli_args", nargs="*", help="extra flags for vllm serve")
    args = parser.parse_args()

    arm = Arm(name=args.name, model=args.model, cli_args=list(args.cli_args), port=args.port)
    server = start_server(arm, args.log, venv_bin=args.venv_bin)
    print(json.dumps({"cold_start_s": server.cold_start_s, "pid": server.process.pid, "url": server.base_url}))
    if not args.keep:
        stop_server(server)


if __name__ == "__main__":
    main()
