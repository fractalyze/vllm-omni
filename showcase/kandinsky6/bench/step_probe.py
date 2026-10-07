# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Drive and score the PiFlow step probe (``VLLM_OMNI_K6_STEP_PROBE``).

A server started with the probe set replays the tail of every request once per
branch in the probe file and saves each branch's decoded frames next to the
request's own (``base``) as ``<out_dir>/<sha1(prompt)[:10]>-s<seed>/<name>.npy``.
Every branch shares the base run's process and kernels, so its LPIPS against
``base`` is the branch's own effect, without the run-to-run floor a second
process adds (0.0272 mean on this pipeline, compiled).

    # Requests for some prompts of either set (the probe file is re-read per request).
    python step_probe.py drive --ids a3-sprint-start b6-... a1-portrait-speech
    # LPIPS of every saved branch against its base, one JSON row per (prompt, branch).
    python step_probe.py score --out-dir /home/jooman/k6/probe/out
    # One branch's frames as <prompt id>.mp4, encoded the way the server encodes
    # (H.264, CRF 18, 24 fps), so the gate scorers take it like any arm's output.
    python step_probe.py encode --out-dir /home/jooman/k6/probe/out --branch reuse-8 --mp4-dir runs/reuse-8
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

BENCH = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCH))

W1 = dict(width=864, height=480, num_frames=121, num_inference_steps=10, guidance_scale=1.0)
SEED = 42
W1_FPS = 24.0


def load_prompts() -> dict[str, str]:
    prompts = {}
    for name in ("setA.json", "setB.json"):
        for entry in json.loads((BENCH / "prompts" / name).read_text())["prompts"]:
            prompts[entry["id"]] = entry["text"]
    return prompts


def probe_dir_name(prompt: str, seed: int) -> str:
    """The directory name the pipeline's probe sink writes for this request."""
    return f"{hashlib.sha1(prompt.encode()).hexdigest()[:10]}-s{seed}"


def drive(args: argparse.Namespace) -> None:
    from drive import request_fields, submit_and_fetch

    prompts = load_prompts()
    args.mp4_dir.mkdir(parents=True, exist_ok=True)
    for prompt_id in args.ids:
        fields = request_fields(prompts[prompt_id], seed=SEED, **W1)
        started = time.perf_counter()
        submit_and_fetch(f"http://127.0.0.1:{args.port}", fields, args.mp4_dir / f"{prompt_id}.mp4")
        print(json.dumps({"id": prompt_id, "request_s": round(time.perf_counter() - started, 2)}), flush=True)


def score(args: argparse.Namespace) -> None:
    import torch
    from quality import _lpips_net

    names = {probe_dir_name(text, SEED): pid for pid, text in load_prompts().items()}
    cache: dict = {}
    net = _lpips_net(cache)
    device = next(net.parameters()).device
    for request_dir in sorted(p for p in args.out_dir.iterdir() if p.is_dir()):
        base_path = request_dir / "base.npy"
        if not base_path.exists():
            continue
        base = torch.from_numpy(np.load(base_path))
        for branch in sorted(request_dir.glob("*.npy")):
            if branch.name == "base.npy":
                continue
            frames = torch.from_numpy(np.load(branch))
            per_frame = []
            with torch.no_grad():
                for start in range(0, len(base), 8):
                    r = base[start : start + 8].to(device).permute(0, 3, 1, 2).float() / 127.5 - 1
                    c = frames[start : start + 8].to(device).permute(0, 3, 1, 2).float() / 127.5 - 1
                    per_frame.extend(net(r, c).flatten().cpu().tolist())
            print(
                json.dumps(
                    {
                        "id": names.get(request_dir.name, request_dir.name),
                        "branch": branch.stem,
                        "lpips_mean": round(statistics.fmean(per_frame), 5),
                        "lpips_max": round(max(per_frame), 5),
                    }
                ),
                flush=True,
            )


def encode(args: argparse.Namespace) -> None:
    from vllm_omni.diffusion.utils.media_utils import mux_video_audio_bytes

    names = {probe_dir_name(text, SEED): pid for pid, text in load_prompts().items()}
    args.mp4_dir.mkdir(parents=True, exist_ok=True)
    for request_dir in sorted(p for p in args.out_dir.iterdir() if p.is_dir()):
        frames_path = request_dir / f"{args.branch}.npy"
        if request_dir.name not in names or not frames_path.exists():
            continue
        target = args.mp4_dir / f"{names[request_dir.name]}.mp4"
        target.write_bytes(mux_video_audio_bytes(np.load(frames_path), fps=W1_FPS))
        print(target, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("drive")
    d.add_argument("--ids", nargs="+", required=True)
    d.add_argument("--port", type=int, default=8094)
    d.add_argument("--mp4-dir", type=Path, default=Path("/home/jooman/k6/probe/mp4"))
    s = sub.add_parser("score")
    s.add_argument("--out-dir", type=Path, required=True)
    e = sub.add_parser("encode")
    e.add_argument("--out-dir", type=Path, required=True)
    e.add_argument("--branch", required=True)
    e.add_argument("--mp4-dir", type=Path, required=True)
    args = parser.parse_args()
    {"drive": drive, "score": score, "encode": encode}[args.cmd](args)


if __name__ == "__main__":
    main()
