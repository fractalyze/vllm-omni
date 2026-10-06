# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Generate one arm's outputs for every prompt in a set, in one process.

The quality gate needs the same prompts and seeds through several attention
arms. Running the offline example once per prompt would reload the pipeline
every time -- about a minute of the roughly two minutes a request takes -- so
this loads once and sweeps the set.

One process per arm, not one process for all arms: an arm is an
``AttentionConfig`` fixed at construction, and rebuilding the pipeline inside
a process to change it would leave the first arm's compiled artifacts and
allocator state behind. Separate processes keep each arm's run independent,
which costs a reload per arm and buys a comparison that is about the arm.

    python gate_run.py --arm shipped --prompts prompts_set_a.json --out-dir runs/a-shipped
    python gate_run.py --arm arms/tuned.json --prompts prompts_set_a.json --out-dir runs/a-tuned

Writes ``<id>.npz`` (frames and audio) per prompt plus a ``manifest.json``
recording the arm, the
geometry, every prompt's seed and wall time, and what ``nvidia-smi`` showed --
so the scorer can refuse to compare two runs that did not share a geometry.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gpulock import foreign_gpu_procs  # noqa: E402

# W1's geometry. Lite is the model (Pro does not fit), but the geometry is
# W1's so the DiT sees the same 50,220 visual tokens, and a quantizing
# attention kernel's error depends on that count.
GEOMETRY = dict(height=480, width=864, num_frames=121, num_inference_steps=10, fps=24)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="kandinskylab/Kandinsky-6.0-Lite-5s-Diffusers")
    parser.add_argument(
        "--arm",
        required=True,
        help="'shipped' for the platform default, or a path to an arms/ attention-config JSON",
    )
    parser.add_argument("--compile-mode", default=None, help="torch.compile mode, or omitted")
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None, help="only the first N prompts")
    parser.add_argument("--no-locks", action="store_true", help="accepted so run_when_free.py can pass it")
    args = parser.parse_args()

    prompt_set = json.loads(args.prompts.read_text())
    prompts = prompt_set["prompts"][: args.limit]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "examples/offline_inference/text_to_video"))
    from text_to_video import build_text_to_video_prompt

    from vllm_omni.entrypoints.omni import Omni
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    attention_config = None if args.arm == "shipped" else json.loads(Path(args.arm).read_text())

    omni = Omni(
        model=args.model,
        model_class_name="Kandinsky6TI2VAPipeline",
        enable_cpu_offload=True,
        diffusion_attention_config=attention_config,
        diffusion_compile_mode=args.compile_mode,
    )

    manifest = {
        "model": args.model,
        "arm": args.arm,
        "compile_mode": args.compile_mode,
        "geometry": GEOMETRY,
        "prompt_set": str(args.prompts.name),
        "foreign_gpu_procs": [str(p) for p in foreign_gpu_procs()],
        "runs": [],
    }

    for entry in prompts:
        sampling = OmniDiffusionSamplingParams(
            height=GEOMETRY["height"],
            width=GEOMETRY["width"],
            num_frames=GEOMETRY["num_frames"],
            num_inference_steps=GEOMETRY["num_inference_steps"],
            seed=entry["seed"],
        )
        started = time.perf_counter()
        # Positional, and through the example's own envelope builder: the
        # request shape is the example's contract, not this script's.
        outputs = omni.generate(build_text_to_video_prompt(entry["text"], None), sampling)
        elapsed = time.perf_counter() - started

        destination = args.out_dir / f"{entry['id']}.npz"
        _save(outputs[0], destination)
        manifest["runs"].append(
            {"id": entry["id"], "seed": entry["seed"], "seconds": round(elapsed, 3), "file": destination.name}
        )
        print(f"{entry['id']}: {elapsed:.1f} s -> {destination}", flush=True)
        (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"wrote {len(manifest['runs'])} output(s) to {args.out_dir}")
    return 0


def _save(output, destination: Path) -> None:
    """Store the request's frames and audio as arrays, not as an MP4.

    The gate scores the model's output, and H.264 is lossy: encoding every arm
    would add codec noise on top of the difference being measured. It would
    add it to each arm about equally, so a ranking would survive -- but the
    absolute LPIPS would no longer be the model's, and the gate's threshold is
    absolute. Frames are kept as uint8 (the pipeline's own output range) and
    audio as float32.
    """
    import numpy as np

    frames, audio, sample_rate = _unwrap(output)
    arrays = {"frames": _as_uint8_frames(frames)}
    if audio is not None:
        arrays["audio"] = np.asarray(audio[0] if isinstance(audio, list) else audio, dtype=np.float32)
        arrays["audio_sample_rate"] = np.asarray(sample_rate or 44100)
    np.savez_compressed(destination, **arrays)


def _unwrap(output):
    """``OmniRequestOutput`` -> ``(frames, audio, sample_rate)``.

    The envelope nests differently depending on how the pipeline returned its
    output, so this follows the same chain the shared example follows rather
    than assuming one shape. The parts that matter: audio arrives either in
    ``multimodal_output`` or inside ``images[0]``, and ``images[0]`` is itself
    either a ``(frames, audio)`` pair, a dict, or the frames.
    """
    audio = None
    sample_rate = None
    frames = output

    if isinstance(frames, list):
        frames = frames[0] if frames else None

    if hasattr(frames, "multimodal_output"):
        multimodal = frames.multimodal_output or {}
        if "audio" in multimodal:
            audio = multimodal["audio"]
            sample_rate = multimodal.get("audio_sample_rate")
        images = getattr(frames, "images", None)
        if not images:
            raise ValueError("the request returned no video frames")
        first = images[0]
        if isinstance(first, tuple) and len(first) == 2:
            frames, audio = first
        elif isinstance(first, dict):
            audio = first.get("audio", audio)
            sample_rate = first.get("audio_sample_rate", sample_rate)
            frames = first.get("frames") or first.get("video")
        else:
            frames = images

    if isinstance(frames, dict):
        audio = frames.get("audio", audio)
        sample_rate = frames.get("audio_sample_rate", sample_rate)
        frames = frames.get("frames") or frames.get("video")
    if isinstance(frames, tuple) and len(frames) == 2:
        frames, audio = frames
    if frames is None:
        raise ValueError("the request returned no video frames")
    return frames, audio, sample_rate


def _as_uint8_frames(frames):
    """``(T, H, W, 3)`` uint8, whatever container the pipeline returned."""
    import numpy as np
    import torch

    if isinstance(frames, list):
        frames = frames[0] if len(frames) == 1 and not isinstance(frames[0], (int, float)) else frames
    if isinstance(frames, torch.Tensor):
        frames = frames.detach().cpu().numpy()
    frames = np.asarray(frames)
    if frames.ndim == 5 and frames.shape[0] == 1:
        frames = frames[0]
    # Channel-first to channel-last if needed.
    if frames.ndim == 4 and frames.shape[1] in (3, 4) and frames.shape[-1] not in (3, 4):
        frames = np.transpose(frames, (0, 2, 3, 1))
    if frames.dtype != np.uint8:
        lo, hi = float(frames.min()), float(frames.max())
        # The pipeline hands back either [0, 1] or [-1, 1] floats.
        frames = (frames + 1.0) / 2.0 if lo < -0.01 else frames
        frames = np.clip(frames * 255.0 if hi <= 1.01 else frames, 0, 255).astype(np.uint8)
    return frames[..., :3]


if __name__ == "__main__":
    raise SystemExit(main())
