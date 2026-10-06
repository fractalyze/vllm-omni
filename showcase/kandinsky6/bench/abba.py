# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""ABBA comparison of two server arms on W1, in one session.

An arm here is a whole server configuration (checkpoint, flags, environment),
so the unit of interleaving is a server *visit*: start the arm, warm up, time
``--repeats`` requests, stop it and wait for the GPU to drain. Visits run in the
order A B B A, so a drift over the session -- the GPU warming toward its power
cap, the page cache filling, a co-tenant arriving -- loads both arms equally
instead of landing on whichever ran second.

Each arm is a JSON file::

    {"name": "sage2",
     "command": ["/data/jooman/k6/vllm-omni/showcase/kandinsky6/serve/run_capped.sh",
                 "/data/jooman/k6/vllm-omni/showcase/kandinsky6/serve/serve_pro_fp8.sh",
                 "--diffusion-attention-config", "@showcase/kandinsky6/compute/arms/sage2.json"],
     "env": {"K6_CKPT": "/data/jooman/k6/ckpt/pro-distill-fp8-min", "K6_MEMMAX": "44G"},
     "port": 8091}

An argument starting with ``@`` is replaced by that file's contents, so an
attention config can be passed by path. The GPU locks are held for the whole
comparison and the board is sampled throughout; the ledger row is
``contaminated`` if anything else touched the GPU.

The reported delta is candidate vs control on the medians of all timed
requests, with each arm's own min-max spread beside it: a delta inside the
control's spread is null.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from drive import RequestFailedError, request_fields, submit_and_fetch  # noqa: E402
from gpu_guard import GpuGuard, board_sample  # noqa: E402
from ledger import Ledger, LedgerRow, environment_fingerprint, run_id, spread, validity_from_guard  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
W1 = {"width": 864, "height": 480, "num_frames": 121, "num_inference_steps": 10, "guidance_scale": 1.0}
DRAINED_MIB = 1024.0


def load_arm(path: Path) -> dict:
    arm = json.loads(path.read_text())
    command = []
    for part in arm["command"]:
        if isinstance(part, str) and part.startswith("@"):
            command.append((REPO / part[1:]).read_text().strip())
        else:
            command.append(part)
    arm["command"] = command
    arm.setdefault("env", {})
    arm.setdefault("port", 8091)
    return arm


def _healthy(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as r:
            return r.status == 200
    except OSError:
        return False


def start(arm: dict, log: Path, timeout_s: float = 1800.0) -> tuple[subprocess.Popen, float]:
    env = {**os.environ, **arm["env"]}
    t0 = time.perf_counter()
    with log.open("wb") as handle:
        proc = subprocess.Popen(arm["command"], stdout=handle, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    while not _healthy(arm["port"]):
        if proc.poll() is not None:
            raise RuntimeError(f"{arm['name']}: server exited ({proc.returncode}); see {log}")
        if time.perf_counter() - t0 > timeout_s:
            stop(proc)
            raise TimeoutError(f"{arm['name']}: not healthy after {timeout_s}s")
        time.sleep(2)
    return proc, time.perf_counter() - t0


def stop(proc: subprocess.Popen) -> None:
    """Stop the arm's whole process group and wait for the GPU to drain.

    Signalling only the launcher would leave the DiffusionWorker serving on the
    port, and the next visit would measure the previous arm.
    """
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=120)
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        proc.wait(timeout=60)
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        used = board_sample().get("memory.used")
        if used is not None and used <= DRAINED_MIB:
            return
        time.sleep(2)


def server_pids(proc: subprocess.Popen) -> list[int]:
    out = subprocess.run(["pgrep", "-g", str(os.getpgid(proc.pid))], capture_output=True, text=True, check=False)
    return [int(p) for p in out.stdout.split()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3, help="timed requests per visit")
    parser.add_argument("--prompts", type=Path, default=Path(__file__).parent / "prompts" / "setA.json")
    parser.add_argument("--prompt-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("/data/jooman/k6/runs"))
    parser.add_argument("--trial", default="", help="vault trial id this comparison decides")
    parser.add_argument("--note", default="")
    args = parser.parse_args()

    arms = {"A": load_arm(args.control), "B": load_arm(args.candidate)}
    prompt = json.loads(args.prompts.read_text())["prompts"][args.prompt_index]
    run = run_id(f"ABBA-{arms['B']['name']}-vs-{arms['A']['name']}")
    ledger = Ledger(args.out)
    run_dir = ledger.run_dir(run)
    guard = GpuGuard(interval_s=1.0)
    walls: dict[str, list[float]] = {"A": [], "B": []}
    visits = []

    with guard:
        for visit_no, key in enumerate("ABBA"):
            arm = arms[key]
            proc, cold_s = start(arm, run_dir / f"server-{visit_no}-{arm['name']}.log")
            for pid in server_pids(proc):
                guard.add_own_pid(pid)
            record = {"visit": visit_no, "arm": arm["name"], "cold_start_s": cold_s, "requests": []}
            try:
                fields = request_fields(prompt["text"], seed=args.seed, **W1)
                for repeat in range(args.repeats + 1):
                    label = "warmup" if repeat == 0 else f"timed{repeat}"
                    mp4 = run_dir / f"v{visit_no}-{arm['name']}-{label}.mp4"
                    try:
                        result = submit_and_fetch(f"http://127.0.0.1:{arm['port']}", fields, mp4)
                    except RequestFailedError as exc:
                        record["requests"].append({"label": label, "failed": str(exc)})
                        continue
                    record["requests"].append({"label": label, **result.as_dict()})
                    if repeat:
                        walls[key].append(result.request_wall_s)
                    print(f"visit {visit_no} {arm['name']} {label}: {result.request_wall_s:.2f}s", flush=True)
            finally:
                stop(proc)
            visits.append(record)

    gpu = guard.report()
    a, b = walls["A"], walls["B"]
    metrics: dict[str, object] = {}
    if a and b:
        metrics = {
            "control_request_wall_s": spread(a),
            "candidate_request_wall_s": spread(b),
            "delta_pct": 100.0 * (statistics.median(b) - statistics.median(a)) / statistics.median(a),
            "control_spread_pct": 100.0 * (max(a) - min(a)) / statistics.median(a),
        }
    row = LedgerRow(
        run=run,
        control=arms["A"]["name"],
        candidate=arms["B"]["name"],
        metrics=metrics,
        verdict={"completed": bool(a and b), "trial": args.trial},
        validity=validity_from_guard(gpu),
        n_pairs=min(len(a), len(b)),
        workload="W1",
        gpu=gpu,
        arms={"A": arms["A"], "B": arms["B"], "prompt_id": prompt["id"], "seed": args.seed, "order": "ABBA"},
        env=environment_fingerprint(Path(sys.executable)),
        output=f"runs/{run}/report.json",
        note=args.note,
    )
    ledger.write_report(run, {"run": run, "visits": visits, "gpu": gpu, "metrics": metrics})
    ledger.append(row)
    print(json.dumps({"run": run, "validity": row.validity, **metrics}, indent=2))


if __name__ == "__main__":
    main()
