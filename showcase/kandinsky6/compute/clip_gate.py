# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""G3: does the arm still make the video the prompt asked for?

LPIPS answers a different question from the one a viewer asks. It compares an
arm to a reference frame by frame, so a change that moves every frame slightly
-- a different but equally plausible sample -- scores the same as one that
breaks the video. On this pipeline that matters more than usual: rerunning the
*same* configuration in a fresh process already moves LPIPS, because Inductor
picks kernels by timing.

So G3 asks whether the arm's video still matches its prompt as well as the
reference's does, with CLIP text-image similarity averaged over evenly spaced
frames, and passes an arm whose set score is within ``--tolerance`` (2% by
default) of the reference's. It is a distributional check, not a per-frame one:
it cannot catch a defect that leaves prompt agreement intact, which is why it is
reported *beside* G1 and G2 and never instead of them.

It also writes a contact sheet per arm -- the frames, side by side with the
reference's -- because the last step of a quality gate is a human looking at it.

    python clip_gate.py --arm-dir .../gate-pro/int8-w8a8 --reference-dir /data/jooman/k6/ref/setA

Runs on the CPU by default: it is a few hundred small forward passes, and the
GPU belongs to whatever is being timed.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

BENCH = Path(__file__).resolve().parents[1] / "bench"
sys.path.insert(0, str(BENCH))
sys.path.insert(0, str(Path(__file__).resolve().parent))

CLIP_MODEL = "openai/clip-vit-base-patch32"
DEFAULT_TOLERANCE = 0.02
DEFAULT_FRAMES = 8


def sample_frames(mp4: Path, count: int):
    """``count`` evenly spaced RGB frames, as PIL images.

    Evenly spaced rather than the first N: a video DiT's failures are not
    uniform in time, and the first frames of a 121-frame clip are the ones most
    like the reference whatever the arm did.
    """
    import av
    from PIL import Image

    with av.open(str(mp4)) as container:
        stream = container.streams.video[0]
        total = stream.frames or 0
        decoded = [frame.to_ndarray(format="rgb24") for frame in container.decode(stream)]
    if not decoded:
        raise ValueError(f"{mp4} decoded to no frames")
    total = len(decoded)
    step = max(1, total // count)
    chosen = decoded[:: step][:count]
    return [Image.fromarray(frame) for frame in chosen]


def clip_scores(model, processor, frames, text: str) -> list[float]:
    """Per-frame cosine similarity between the prompt and the frame."""
    import torch

    with torch.no_grad():
        inputs = processor(text=[text], images=frames, return_tensors="pt", padding=True, truncation=True)
        outputs = model(**inputs)
        image = torch.nn.functional.normalize(outputs.image_embeds, dim=-1)
        txt = torch.nn.functional.normalize(outputs.text_embeds, dim=-1)
        return (image @ txt.T).squeeze(-1).tolist()


def g3_verdict(set_arm: float, set_reference: float, tolerance: float = DEFAULT_TOLERANCE) -> dict[str, object]:
    """Whether the arm's prompt agreement is within ``tolerance`` of the reference's.

    Two-sided. An arm that scores *higher* than the reference has not improved
    the model -- CLIP cannot tell a better video from a differently-wrong one --
    so a one-sided test would wave through exactly the drift this check exists
    to catch.
    """
    if set_reference == 0:
        return {"decidable": False, "reason": "the reference scored zero, so a relative test is undefined"}
    relative = (set_arm - set_reference) / set_reference
    return {
        "decidable": True,
        "set_clip_arm": set_arm,
        "set_clip_reference": set_reference,
        "relative_delta": relative,
        "tolerance": tolerance,
        "passes": bool(abs(relative) <= tolerance),
    }


def contact_sheet(rows: list[tuple[str, list]], destination: Path, thumb_width: int = 240) -> None:
    """One row of frames per entry, labelled, as a single image."""
    from PIL import Image, ImageDraw

    if not rows:
        return
    columns = max(len(frames) for _, frames in rows)
    sample = rows[0][1][0]
    thumb_height = round(sample.height * thumb_width / sample.width)
    label_height = 18
    sheet = Image.new(
        "RGB",
        (columns * thumb_width, len(rows) * (thumb_height + label_height)),
        "black",
    )
    draw = ImageDraw.Draw(sheet)
    for row, (label, frames) in enumerate(rows):
        top = row * (thumb_height + label_height)
        draw.text((4, top + 4), label, fill="white")
        for column, frame in enumerate(frames):
            sheet.paste(frame.resize((thumb_width, thumb_height)), (column * thumb_width, top + label_height))
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination, quality=88)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, default=Path("/data/jooman/k6/ref/setA"))
    parser.add_argument("--prompts", type=Path, default=BENCH / "prompts" / "setA.json")
    parser.add_argument("--frames", type=int, default=DEFAULT_FRAMES)
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--contact-sheet", type=Path, default=None)
    args = parser.parse_args()

    import torch
    from transformers import CLIPModel, CLIPProcessor

    torch.set_grad_enabled(False)
    model = CLIPModel.from_pretrained(CLIP_MODEL).eval()
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL)

    prompts = json.loads(args.prompts.read_text())["prompts"]
    rows = []
    sheet_rows = []
    for entry in prompts:
        candidate = args.arm_dir / f"{entry['id']}.mp4"
        reference = args.reference_dir / f"{entry['id']}.mp4"
        if not candidate.is_file() or not reference.is_file():
            continue
        candidate_frames = sample_frames(candidate, args.frames)
        reference_frames = sample_frames(reference, args.frames)
        arm_score = statistics.fmean(clip_scores(model, processor, candidate_frames, entry["text"]))
        reference_score = statistics.fmean(clip_scores(model, processor, reference_frames, entry["text"]))
        rows.append(
            {
                "id": entry["id"],
                "categories": entry.get("categories", []),
                "clip_arm": arm_score,
                "clip_reference": reference_score,
                "relative_delta": (arm_score - reference_score) / reference_score if reference_score else float("inf"),
            }
        )
        print(
            f"  {entry['id']:<22} CLIP arm {arm_score:.4f} reference {reference_score:.4f} "
            f"({rows[-1]['relative_delta'] * 100:+.2f}%)",
            flush=True,
        )
        sheet_rows.append((f"{entry['id']} arm", candidate_frames))
        sheet_rows.append((f"{entry['id']} reference", reference_frames))

    if not rows:
        raise SystemExit(f"no prompt had both an arm and a reference MP4 ({args.arm_dir}, {args.reference_dir})")

    verdict = g3_verdict(
        statistics.fmean(r["clip_arm"] for r in rows),
        statistics.fmean(r["clip_reference"] for r in rows),
        args.tolerance,
    )
    set_arm = verdict["set_clip_arm"]
    set_reference = verdict["set_clip_reference"]
    relative = verdict["relative_delta"]
    passes = verdict["passes"]
    print(
        f"\n  G3 (distributional): set CLIP {set_arm:.4f} against the reference's {set_reference:.4f}, "
        f"{relative * 100:+.2f}% (tolerance +-{args.tolerance * 100:.0f}%) -> "
        + ("PASSES" if passes else "FAILS")
    )

    if args.contact_sheet:
        contact_sheet(sheet_rows, args.contact_sheet)
        print(f"  contact sheet: {args.contact_sheet}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(
                {
                    "arm_dir": str(args.arm_dir),
                    "reference_dir": str(args.reference_dir),
                    "frames_per_prompt": args.frames,
                    "tolerance": args.tolerance,
                    "set_clip_arm": set_arm,
                    "set_clip_reference": set_reference,
                    "relative_delta": relative,
                    "passes_g3": bool(passes),
                    "per_prompt": rows,
                },
                indent=2,
            )
            + "\n"
        )
        print(f"  wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
