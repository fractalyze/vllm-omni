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

# W1, as PLAN.md fixes it and as Track M's BF16 references were generated:
# their ref/setA/manifest.json records this geometry and seed 42 for every
# prompt. Both have to match or the comparison is not same-seed and the LPIPS
# is measuring a different sample, not a different kernel.
W1 = dict(width=864, height=480, num_frames=121, num_inference_steps=10, guidance_scale=1.0)
REFERENCE_SEED = 42
DEFAULT_CKPT = "/data/jooman/k6/ckpt/pro-distill-fp8-min"
BF16_CKPT = "kandinskylab/Kandinsky-6.0-Pro-distill-5s-Diffusers"

# How an arm gets the DiT's weights to the GPU. The two modes are not
# interchangeable: the FP8 DiT is 29 GB and fits in pinned host memory, so
# plain layerwise offload stages it from RAM, while the BF16 DiT is 60.3 GB on
# a 60 GB host and cannot be copied anywhere -- distributed layerwise offload
# without AllGather binds every tensor to the mmapped checkpoint and streams it
# from NVMe through the page cache. ``serve/serve_pro_bf16_ref.sh`` is the
# reference's own script and these are its flags, so an arm asked for
# ``dlo-mmap`` is served the way Track M's references were.
OFFLOAD_FLAGS = {
    "layerwise": ["--enable-layerwise-offload"],
    "dlo-mmap": ["--enable-distributed-layerwise-offload", "--dlo-no-use-allgather"],
}


def serve_flags(
    offload: str,
    attention_config: str | None,
    compile_mode: str | None,
    quantization: str | None = None,
) -> list[str]:
    """The ``vllm serve`` flags for one arm, beyond the model.

    Separated from :func:`main` because this is the part a reader has to trust:
    two arms are only comparable if the flag list differs in exactly the thing
    under test, and that is checkable here without a GPU.

    ``quantization`` is a *load-time* quantization method name, which is a
    different thing from serving one of the pre-quantized checkpoints: it takes
    the exact BF16 checkpoint and quantizes each tensor as it is loaded, so the
    keep list and the scale granularity are chosen here rather than baked into a
    file on disk. ``fp8_per_channel`` is the reason the knob exists -- vLLM
    offers per-output-row FP8 scales online, and the offline converter's own
    docstring says per-row scales are the better recipe but are not loadable
    from a natively-serialized FP8 checkpoint.
    """
    if offload not in OFFLOAD_FLAGS:
        raise ValueError(f"unknown offload mode {offload!r}; expected one of {sorted(OFFLOAD_FLAGS)}")
    flags = ["--num-gpus", "1", *OFFLOAD_FLAGS[offload], "--disable-multithread-weight-load"]
    if attention_config is not None:
        flags += ["--diffusion-attention-config", attention_config]
    if compile_mode:
        flags += ["--diffusion-compile-mode", compile_mode]
    if quantization:
        flags += ["--diffusion-quantization-config", quantization]
    return flags


def _reference_settings(manifest_path: Path, prompts: list[dict]) -> tuple[dict[str, int], dict]:
    """Per-prompt seeds and the geometry, taken from the reference's manifest.

    Read rather than assumed. The gate compares a candidate with a reference
    *on the same sample*: a different seed or a different frame count makes
    LPIPS measure a different video, not a different kernel, and the number
    would look like a quality result. If the manifest is missing, this falls
    back to the declared defaults and says so, because a silent fallback is
    the same trap.
    """
    ids = [entry["id"] for entry in prompts]
    if not manifest_path.is_file():
        print(f"warning: no reference manifest at {manifest_path}; using W1 defaults and seed "
              f"{REFERENCE_SEED} for every prompt", file=sys.stderr)
        return dict.fromkeys(ids, REFERENCE_SEED), dict(W1)

    manifest = json.loads(manifest_path.read_text())
    items = manifest.get("items", {})
    missing = [i for i in ids if i not in items]
    if missing:
        raise SystemExit(f"{manifest_path} has no entry for {missing}; the reference set and the prompt set disagree")

    seeds = {i: int(items[i]["seed"]) for i in ids}
    geometries = {json.dumps(items[i]["geometry"], sort_keys=True) for i in ids}
    if len(geometries) != 1:
        raise SystemExit(f"{manifest_path} mixes geometries across prompts: {sorted(geometries)}")
    geometry = json.loads(next(iter(geometries)))
    # request_fields takes exactly these; anything else in the manifest's
    # geometry is informational and must not be forwarded blindly.
    allowed = {"width", "height", "num_frames", "num_inference_steps", "guidance_scale"}
    unexpected = set(geometry) - allowed
    if unexpected:
        raise SystemExit(f"{manifest_path} geometry has keys this driver does not know: {sorted(unexpected)}")
    return seeds, geometry


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", required=True, help="an arms/*.json attention config, or 'shipped'")
    parser.add_argument("--name", required=True, help="arm name, used in the ledger and the output dir")
    parser.add_argument("--ckpt", default=None, help=f"default: {DEFAULT_CKPT}, or {BF16_CKPT} with --offload dlo-mmap")
    parser.add_argument(
        "--offload",
        choices=sorted(OFFLOAD_FLAGS),
        default="layerwise",
        help="how the DiT reaches the GPU: 'layerwise' stages the FP8 DiT from pinned host RAM, "
        "'dlo-mmap' streams the BF16 DiT from NVMe the way the reference server does",
    )
    parser.add_argument("--compile-mode", default=None, help="--diffusion-compile-mode for this arm")
    parser.add_argument(
        "--quantization",
        default=None,
        help="a load-time quantization method (e.g. fp8_per_channel), or JSON for one with an ignore list",
    )
    parser.add_argument("--prompts", type=Path, default=BENCH / "prompts" / "setA.json")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--warmup", action="store_true", default=True, help="one discarded request first")
    parser.add_argument("--no-warmup", dest="warmup", action="store_false")
    parser.add_argument(
        "--reference-manifest",
        type=Path,
        default=Path("/data/jooman/k6/ref/setA/manifest.json"),
        help="the reference run's manifest, read for its per-prompt seed and geometry so this run "
        "matches it. Pass a missing path to fall back to the defaults",
    )
    parser.add_argument("--no-locks", action="store_true", help="accepted so run_when_free.py can pass it")
    args = parser.parse_args()

    prompt_set = json.loads(args.prompts.read_text())
    prompts = prompt_set["prompts"][: args.limit]
    seeds, geometry = _reference_settings(args.reference_manifest, prompts)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # The same flags Track M's serve scripts pass, so an arm measured here is
    # comparable with their baseline and their reference: anything else would
    # make the attention config only one of several differences.
    checkpoint = args.ckpt or (BF16_CKPT if args.offload == "dlo-mmap" else DEFAULT_CKPT)
    cli_args = serve_flags(
        args.offload,
        None if args.arm == "shipped" else Path(args.arm).read_text(),
        args.compile_mode,
        args.quantization,
    )

    arm = Arm(
        name=args.name,
        model=checkpoint,
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
        notes=f"Track C gate generation, arm={args.arm}, offload={args.offload}, compile_mode={args.compile_mode}, quantization={args.quantization}",
    )

    manifest = {
        "arm": args.name,
        "attention_config": args.arm,
        "compile_mode": args.compile_mode,
        "quantization": args.quantization,
        "checkpoint": checkpoint,
        "offload": args.offload,
        "cli_args": cli_args,
        "geometry": geometry,
        "seeds": seeds,
        "prompt_set": prompt_set.get("set"),
        "prompts_file": str(args.prompts),
        "runs": [],
        "items": {},
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
                first = prompts[0]
                fields = request_fields(first["text"], seed=seeds[first["id"]], **geometry)
                submit_and_fetch(server.base_url, fields, args.out_dir / "warmup.mp4")
                print(f"{args.name}: warm-up ok", flush=True)

            for entry in prompts:
                fields = request_fields(entry["text"], seed=seeds[entry["id"]], **geometry)
                destination = args.out_dir / f"{entry['id']}.mp4"
                started_wall = time.time()
                started = time.perf_counter()
                try:
                    result = submit_and_fetch(server.base_url, fields, destination)
                except RequestFailedError as exc:
                    print(f"{entry['id']}: FAILED {exc}", file=sys.stderr, flush=True)
                    manifest["runs"].append({"id": entry["id"], "error": str(exc)})
                    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
                    continue
                run = {
                    "id": entry["id"],
                    "seed": seeds[entry["id"]],
                    "categories": entry.get("categories", []),
                    "seconds": round(time.perf_counter() - started, 3),
                    "mp4": destination.name,
                    "mp4_bytes": result.mp4_bytes,
                }
                manifest["runs"].append(run)
                # Also in the reference runner's shape, so this directory can
                # *be* the reference for a later arm: _reference_settings reads
                # `items[id].seed` and `.geometry` to make the candidate match
                # the run it is scored against. Set B has no canonical
                # reference, so a BF16 run of it here has to be usable as one.
                manifest["items"][entry["id"]] = {
                    "categories": entry.get("categories", []),
                    "request_wall_s": run["seconds"],
                    "started": started_wall,
                    "seed": seeds[entry["id"]],
                    "geometry": geometry,
                }
                print(f"{entry['id']}: {manifest['runs'][-1]['seconds']:.1f} s -> {destination}", flush=True)
                (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        finally:
            stop_server(server)

    ok = sum(1 for r in manifest["runs"] if "error" not in r)
    print(f"{args.name}: {ok}/{len(prompts)} outputs in {args.out_dir}")
    return 0 if ok == len(prompts) else 1


if __name__ == "__main__":
    raise SystemExit(main())
