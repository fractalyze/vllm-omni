# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Time W1 on a running server and write a ledger row.

W1 is the showcase's headline workload: Pro-distill at 864x480, 121 frames,
24 fps, 10 steps, guidance 1.0, audio on. The measurement protocol is the plan's
and is not optional:

- **One warm-up request, then at least 3 timed repeats.** The first request on a
  fresh server pays one-off costs (lazily built kernels, the first weight
  stream, a cold page cache) that no later request pays, so including it
  measures the wrong thing.
- **Median with min and max.** A median alone cannot show that a 4% difference
  sits inside a 9% spread, and a delta inside the control's own spread is null.
- **The GPU is held and sampled throughout.** Every lock on the host is flocked
  and ``nvidia-smi`` is polled for the whole run, so a row that shared the GPU
  is written as ``contaminated`` rather than averaged into a conclusion.
- **The profiler stays off.** It synchronizes around every stage, so a profiled
  run is not a wall. Stage breakdowns come from ``--breakdown`` runs, which are
  marked ``profiled`` in the ledger and never promoted to a headline.

This assumes a server is already up (``serve.py``, or the showcase's
``serve_fp8.sh``): arms differ in how they are launched, and keeping the launch
outside means a run can be re-timed against a server someone else started.

Example::

    python run_w1.py --arm fp8-layerwise --repeats 3 \\
        --prompts prompts/setA.json --out /data/jooman/k6/runs
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from drive import RequestFailed, request_fields, submit_and_fetch  # noqa: E402
from gpu_guard import GpuGuard  # noqa: E402
from ledger import Ledger, LedgerRow, environment_fingerprint, run_id, spread, validity_from_guard  # noqa: E402
from stages import summarize  # noqa: E402

# W1, exactly as the checkpoint's own model card and config state it.
W1 = {
    "width": 864,
    "height": 480,
    "num_frames": 121,
    "num_inference_steps": 10,
    "guidance_scale": 1.0,
    "fps": 24,
}


def load_prompts(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)["prompts"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, help="name of the server configuration being timed")
    parser.add_argument("--base-url", default="http://127.0.0.1:8091")
    parser.add_argument("--out", type=Path, default=Path("/data/jooman/k6/runs"))
    parser.add_argument("--prompts", type=Path, default=Path(__file__).parent / "prompts" / "setA.json")
    parser.add_argument("--repeats", type=int, default=3, help="timed repeats after the warm-up")
    parser.add_argument("--n-prompts", type=int, default=1, help="how many prompts of the set to time")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--server-log", type=Path, default=Path("/data/jooman/k6/server-fp8.log"))
    parser.add_argument(
        "--breakdown",
        action="store_true",
        help="this server has the stage profiler on: record stage timings and mark the row profiled",
    )
    parser.add_argument("--steps", type=int, default=W1["num_inference_steps"])
    parser.add_argument("--num-frames", type=int, default=W1["num_frames"])
    parser.add_argument("--width", type=int, default=W1["width"])
    parser.add_argument("--height", type=int, default=W1["height"])
    parser.add_argument("--note", default="")
    args = parser.parse_args()

    prompts = load_prompts(args.prompts)[: args.n_prompts]
    geometry = {
        "width": args.width,
        "height": args.height,
        "num_frames": args.num_frames,
        "num_inference_steps": args.steps,
        "guidance_scale": W1["guidance_scale"],
    }
    is_w1 = (
        args.width == W1["width"]
        and args.height == W1["height"]
        and args.num_frames == W1["num_frames"]
        and args.steps == W1["num_inference_steps"]
    )

    run = run_id(f"W1-{args.arm}" if is_w1 else f"SMOKE-{args.arm}")
    ledger = Ledger(args.out)
    run_dir = ledger.run_dir(run)
    guard = GpuGuard(interval_s=1.0)

    records: list[dict] = []
    print(f"run {run}: holding {len(guard.lock_paths())} lock(s)")
    with guard:
        entry = guard.report()
        if entry["entry_foreign"]:
            print(f"WARNING: foreign GPU process already present: {entry['entry_foreign']}", file=sys.stderr)

        for prompt in prompts:
            # Warm-up first: its cost is one-off and is recorded, not timed.
            for repeat in range(args.repeats + 1):
                is_warmup = repeat == 0
                label = "warmup" if is_warmup else f"timed{repeat}"
                out_mp4 = run_dir / f"{prompt['id']}-{label}.mp4"
                fields = request_fields(prompt["text"], seed=args.seed, **geometry)
                started = time.time()
                try:
                    result = submit_and_fetch(args.base_url, fields, out_mp4)
                except RequestFailed as exc:
                    print(f"{prompt['id']} {label}: FAILED {exc}", file=sys.stderr)
                    records.append({"prompt_id": prompt["id"], "label": label, "failed": str(exc)})
                    continue
                print(
                    f"{prompt['id']} {label}: {result.request_wall_s:.2f}s "
                    f"(generate {result.generate_s:.2f}s, download {result.download_s:.3f}s, "
                    f"{result.mp4_bytes / 1e6:.1f} MB)"
                )
                records.append(
                    {
                        "prompt_id": prompt["id"],
                        "label": label,
                        "warmup": is_warmup,
                        "wall_clock": started,
                        **result.as_dict(),
                    }
                )

    gpu = guard.report()
    timed = [r for r in records if not r.get("warmup") and "request_wall_s" in r]
    walls = [r["request_wall_s"] for r in timed]

    stage_report = None
    if args.breakdown and args.server_log.exists():
        stage_report = summarize(args.server_log)

    metrics: dict[str, object] = {
        "n_timed": len(walls),
        "n_failed": sum(1 for r in records if "failed" in r),
    }
    if walls:
        metrics["request_wall_s"] = spread(walls)
        metrics["request_wall_s_spread_pct"] = 100.0 * (max(walls) - min(walls)) / statistics.median(walls)
        metrics["s_per_step"] = statistics.median(walls) / args.steps
    warmup = [r for r in records if r.get("warmup") and "request_wall_s" in r]
    if warmup:
        metrics["warmup_request_wall_s"] = warmup[0]["request_wall_s"]

    row = LedgerRow(
        run=run,
        control=args.arm,
        candidate=args.arm,
        metrics=metrics,
        verdict={"completed": bool(walls), "n_failed": metrics["n_failed"]},
        validity=validity_from_guard(gpu),
        profiled=args.breakdown,
        n_pairs=0,
        workload="W1" if is_w1 else "smoke",
        gpu=gpu,
        arms={"arm": args.arm, "geometry": geometry, "prompt_set": args.prompts.name, "seed": args.seed},
        env=environment_fingerprint(Path(sys.executable)),
        output=f"runs/{run}/report.json",
        note=args.note,
    )
    ledger.write_report(
        run,
        {
            "run": run,
            "arm": args.arm,
            "geometry": geometry,
            "requests": records,
            "gpu": gpu,
            "stages": stage_report,
        },
    )
    path = ledger.append(row)

    print(json.dumps({"run": run, "validity": row.validity, "metrics": metrics}, indent=2))
    print(f"ledger: {path}")
    print(f"artifacts: {run_dir}")


if __name__ == "__main__":
    main()
