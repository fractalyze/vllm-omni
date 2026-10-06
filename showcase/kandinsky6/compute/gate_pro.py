# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Generate one served arm's W1 outputs for a whole prompt set, as MP4s.

The gate's numbers only compare across arms if every arm is scored against
the *same* reference with the *same* scorer. Track M owns the canonical Pro
BF16 references and ``bench/quality.py``, so this produces what that scorer
eats -- one MP4 per prompt, from a real served request -- rather than a second
format that would have to be reconciled later.

It deliberately reuses ``bench/``: ``serve.Arm``/``start_server`` for the
server (which measures cold start instead of sleeping through it, and waits
for the GPU to drain on stop), and ``drive.submit_and_fetch`` for the request
(which clocks from the POST to the content GET and treats a timeout as a
failed measurement rather than a slow one).

    python gate_pro.py --arm arms/tuned.json --name tuned-sage2 \
        --out-dir /data/jooman/k6/results/gate-pro/tuned-sage2

``--arm`` is one of ``compute/arms/*.json``, the same per-role attention
config a server takes; it is passed through as
``--diffusion-attention-config``. Pass ``--arm shipped`` for the platform
default.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from pathlib import Path

BENCH = Path(__file__).resolve().parents[1] / "bench"
sys.path.insert(0, str(BENCH))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from drive import RequestFailedError, request_fields, submit_and_fetch  # noqa: E402
from gpulock import GpuLocks, foreign_gpu_procs  # noqa: E402
from serve import Arm, start_server, stop_server  # noqa: E402

# W1, as PLAN.md fixes it.
W1 = dict(width=864, height=480, num_frames=121, num_inference_steps=10)
DEFAULT_CKPT = "/data/jooman/k6/ckpt/pro-distill-fp8-min"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", required=True, help="an arms/*.json attention config, or 'shipped'")
    parser.add_argument("--name", required=True, help="arm name, used in the ledger and the output dir")
    parser.add_argument("--ckpt", default=DEFAULT_CKPT)
    parser.add_argument("--compile-mode", default=None, help="--diffusion-compile-mode for this arm")
    parser.add_argument("--prompts", type=Path, default=BENCH / "prompts" / "setA.json")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--warmup", action="store_true", default=True, help="one discarded request first")
    parser.add_argument("--no-warmup", dest="warmup", action="store_false")
    parser.add_argument("--no-locks", action="store_true", help="accepted so run_when_free.py can pass it")
    args = parser.parse_args()

    prompt_set = json.loads(args.prompts.read_text())
    prompts = prompt_set["prompts"][: args.limit]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # The same flags Track M's serve_pro_fp8.sh passes, so an arm measured
    # here is comparable with their baseline: anything else would make the
    # attention config only one of several differences.
    cli_args = ["--num-gpus", "1", "--enable-layerwise-offload", "--disable-multithread-weight-load"]
    if args.arm != "shipped":
        cli_args += ["--diffusion-attention-config", Path(args.arm).read_text()]
    if args.compile_mode:
        cli_args += ["--diffusion-compile-mode", args.compile_mode]

    arm = Arm(
        name=args.name,
        model=args.ckpt,
        cli_args=cli_args,
        env={
            "HF_HOME": "/data/jooman/hf",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            # glibc keeps freed blocks below its dynamic mmap threshold, and
            # the layerwise backend frees a DiT's worth of them during hook
            # installation; Track M's serve script pins it for the same
            # reason.
            "MALLOC_MMAP_THRESHOLD_": "131072",
        },
        notes=f"Track C gate generation, arm={args.arm}, compile_mode={args.compile_mode}",
    )

    manifest = {
        "arm": args.name,
        "attention_config": args.arm,
        "compile_mode": args.compile_mode,
        "checkpoint": args.ckpt,
        "geometry": W1,
        "prompt_set": prompt_set.get("set"),
        "prompts_file": str(args.prompts),
        "runs": [],
    }

    with GpuLocks() if not args.no_locks else contextlib.nullcontext():
        foreign = foreign_gpu_procs()
        manifest["foreign_gpu_procs"] = [str(p) for p in foreign]
        if foreign:
            print(f"warning: foreign GPU process(es) present: {foreign}", file=sys.stderr)

        # venv_bin: `vllm` is not on PATH in a bare session, and start_server
        # resolves the executable rather than relying on the shell.
        server = start_server(
            arm, log_path=args.out_dir / "server.log", venv_bin=Path("/data/jooman/k6/venv/bin")
        )
        manifest["cold_start_s"] = round(server.cold_start_s, 3)
        print(f"{args.name}: server ready in {server.cold_start_s:.1f} s", flush=True)
        try:
            if args.warmup:
                # The first request on a fresh server pays one-off costs no
                # later request pays. Discarded, but it must succeed: a
                # warm-up that fails means the arm is broken, not warm.
                fields = request_fields(prompts[0]["prompt"], seed=prompts[0]["seed"], **W1)
                submit_and_fetch(server.base_url, fields, args.out_dir / "warmup.mp4")
                print(f"{args.name}: warm-up ok", flush=True)

            for entry in prompts:
                fields = request_fields(entry["prompt"], seed=entry["seed"], **W1)
                destination = args.out_dir / f"{entry['id']}.mp4"
                started = time.perf_counter()
                try:
                    result = submit_and_fetch(server.base_url, fields, destination)
                except RequestFailedError as exc:
                    print(f"{entry['id']}: FAILED {exc}", file=sys.stderr, flush=True)
                    manifest["runs"].append({"id": entry["id"], "error": str(exc)})
                    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
                    continue
                manifest["runs"].append(
                    {
                        "id": entry["id"],
                        "seed": entry["seed"],
                        "categories": entry.get("categories", []),
                        "seconds": round(time.perf_counter() - started, 3),
                        "mp4": destination.name,
                        "mp4_bytes": result.mp4_bytes,
                    }
                )
                print(f"{entry['id']}: {manifest['runs'][-1]['seconds']:.1f} s -> {destination}", flush=True)
                (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        finally:
            stop_server(server)

    ok = sum(1 for r in manifest["runs"] if "error" not in r)
    print(f"{args.name}: {ok}/{len(prompts)} outputs in {args.out_dir}")
    return 0 if ok == len(prompts) else 1


if __name__ == "__main__":
    raise SystemExit(main())
