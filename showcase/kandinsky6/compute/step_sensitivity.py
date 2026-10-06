# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""How much does an error at denoise step k change the finished video?

The quality gate rejected SageAttention on Kandinsky 6 at set max LPIPS
0.3745 against a 0.25 limit, worst on rendered text. The obvious way to trade
that back is to use the exact kernel where it matters and the fast one where
it does not -- but "where it matters" has to be measured before it can be
used, and the vault's prior is that an error at an early denoise step grows
about 20x by the final latent (`c-qi21sched-fp8-early-weights-dominate`, on a
different model).

So this injects a **fixed, Sage2-sized perturbation at exactly one step** and
scores the finished video against an unperturbed run. One run per step plus a
reference gives the amplification curve directly, and it does so without
needing a per-step attention switch to exist first: the question "is step 0
worth protecting" is answered by perturbing step 0, not by building the
machinery to protect it.

The perturbation is added to the transformer's velocity prediction, scaled to
a chosen fraction of that prediction's own RMS -- 0.039 by default, which is
SageAttention2's measured relative L2 at the attention output on W1 shapes. A
curve measured at Sage2's own error magnitude is the one that predicts what
switching Sage2 off at a step would buy.

    python step_sensitivity.py --prompt-id a3-rendered-text --out-dir runs/sens

Writes one ``.npz`` per step (plus ``reference.npz``) and a ``curve.json``
once scored.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gate_run import GEOMETRY, _save  # noqa: E402

# SageAttention2's measured relative L2 at the attention output, W1 shapes,
# from the backend race. The curve is measured at the error magnitude whose
# placement is actually being decided.
SAGE2_REL_L2 = 0.039


def install_perturbation(step_to_perturb: int | None, magnitude: float, seed: int) -> dict:
    """Perturb the DiT's output on one step. Returns a record of what it did.

    Hooks the transformer rather than the scheduler because that is where a
    kernel error appears: a fast attention kernel perturbs the velocity
    prediction, and the scheduler then integrates it. Perturbing the latent
    after the step would measure something subtly different -- an error the
    scheduler has already scaled.

    The noise is drawn from a fixed generator, so every step's run sees the
    *same* perturbation field and the curve compares placement rather than
    luck.
    """
    import torch

    from vllm_omni.diffusion.models.kandinsky6 import Kandinsky6Transformer3DModel

    state = {"calls": 0, "perturbed_at": None, "relative_rms": None}
    original = Kandinsky6Transformer3DModel.forward

    def forward(self, *args, **kwargs):
        out = original(self, *args, **kwargs)
        index = state["calls"]
        state["calls"] = index + 1
        if step_to_perturb is None or index != step_to_perturb:
            return out

        generator = torch.Generator(device="cpu").manual_seed(seed)

        def perturb(tensor):
            if not isinstance(tensor, torch.Tensor) or not tensor.is_floating_point():
                return tensor
            rms = tensor.float().square().mean().sqrt()
            noise = torch.randn(tensor.shape, generator=generator, dtype=torch.float32).to(tensor.device)
            noise = noise / noise.float().square().mean().sqrt()
            state["relative_rms"] = magnitude
            return (tensor.float() + magnitude * rms * noise).to(tensor.dtype)

        state["perturbed_at"] = index
        if isinstance(out, tuple):
            return tuple(perturb(t) for t in out)
        return perturb(out)

    Kandinsky6Transformer3DModel.forward = forward
    state["restore"] = lambda: setattr(Kandinsky6Transformer3DModel, "forward", original)
    return state


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="kandinskylab/Kandinsky-6.0-Lite-5s-Diffusers")
    parser.add_argument("--prompts", type=Path, default=Path(__file__).resolve().parent / "prompts_set_a.json")
    parser.add_argument(
        "--prompt-id",
        default="a3-rendered-text",
        help="which prompt to measure the curve on. The default is the prompt SageAttention fails worst",
    )
    parser.add_argument("--magnitude", type=float, default=SAGE2_REL_L2)
    parser.add_argument("--steps", type=int, default=None, help="how many steps to probe; default every step")
    parser.add_argument("--noise-seed", type=int, default=7)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--no-locks", action="store_true", help="accepted so run_when_free.py can pass it")
    args = parser.parse_args()

    prompt_set = json.loads(args.prompts.read_text())
    entry = next((p for p in prompt_set["prompts"] if p["id"] == args.prompt_id), None)
    if entry is None:
        parser.error(f"no prompt {args.prompt_id!r} in {args.prompts}")
    total_steps = args.steps or GEOMETRY["num_inference_steps"]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "examples/offline_inference/text_to_video"))
    from text_to_video import build_text_to_video_prompt

    from vllm_omni.entrypoints.omni import Omni
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    omni = Omni(model=args.model, model_class_name="Kandinsky6TI2VAPipeline", enable_cpu_offload=True)
    sampling = OmniDiffusionSamplingParams(
        height=GEOMETRY["height"],
        width=GEOMETRY["width"],
        num_frames=GEOMETRY["num_frames"],
        num_inference_steps=total_steps,
        seed=entry["seed"],
    )
    envelope = build_text_to_video_prompt(entry["text"], None)

    manifest = {
        "model": args.model,
        "prompt_id": args.prompt_id,
        "seed": entry["seed"],
        "geometry": {**GEOMETRY, "num_inference_steps": total_steps},
        "magnitude_relative_rms": args.magnitude,
        "noise_seed": args.noise_seed,
        "runs": [],
    }

    # The reference first: every perturbed run is scored against it, so a
    # failure later still leaves a usable reference on disk.
    for label, step in [("reference", None)] + [(f"step{k:02d}", k) for k in range(total_steps)]:
        state = install_perturbation(step, args.magnitude, args.noise_seed)
        try:
            started = time.perf_counter()
            outputs = omni.generate(envelope, sampling)
            elapsed = time.perf_counter() - started
        finally:
            state["restore"]()

        if step is not None and state["perturbed_at"] != step:
            raise RuntimeError(
                f"{label}: meant to perturb DiT call {step} but the forward ran {state['calls']} times "
                f"and perturbed {state['perturbed_at']}. The call count must equal the step count, or the "
                "curve's x-axis is not the step index."
            )

        destination = args.out_dir / f"{label}.npz"
        _save(outputs[0], destination)
        manifest["runs"].append(
            {"label": label, "step": step, "seconds": round(elapsed, 3), "dit_calls": state["calls"]}
        )
        print(f"{label}: {elapsed:.1f} s, {state['calls']} DiT calls -> {destination}", flush=True)
        (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"wrote {len(manifest['runs'])} run(s) to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
