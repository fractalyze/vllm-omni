# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Score one arm's outputs against a reference arm's, on the showcase's gate.

``showcase/kandinsky6/PLAN.md`` sets the gate: **mean LPIPS <= 0.05 and max
<= 0.10** over a prompt set for the `approx` tier, plus PSNR and SSIM on the
video and log-mel L1 and SI-SDR on the audio. The reference is the same
checkpoint, same prompts, same seeds, through the arm the model ships with --
which is BF16, so a cuDNN/SDPA arm *is* the BF16 reference.

LPIPS is per frame and the gate wants both its mean and its max over frames,
because a temporal artefact that ruins two frames of 121 is invisible in a
mean. Reported per prompt and then aggregated over the set, so a single
failing prompt is visible rather than averaged away -- which is the whole
point of `c-qwen-image21-fp8-quality-is-prompt-dependent`, where one 8-prompt
set passed at 0.034 and another failed at 0.162.

    python gate_score.py --reference runs/a-shipped --candidate runs/a-tuned --json scores.json
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gpulock import GpuLocks, foreign_gpu_procs  # noqa: E402

# Two bars, deliberately separate.
#
# ADOPTION is the user's gate for this showcase (set 2026-10-06): per prompt
# set, LPIPS mean <= 0.15 and max <= 0.25 against the same-checkpoint BF16
# reference. An arm that clears it may ship.
#
# APPROX is PLAN.md's and the world-model vocabulary's `approx` tier, which is
# a far tighter claim about how close to the reference an arm is. Both are
# reported because they answer different questions: "may we ship this" and
# "what do we call it". An arm can be adoptable and still be `lossy`, and
# saying so is the point -- a showcase that calls a lossy arm near-lossless is
# the failure mode these two numbers exist to prevent.
ADOPTION_MEAN_LPIPS = 0.15
ADOPTION_MAX_LPIPS = 0.25
APPROX_MEAN_LPIPS = 0.05
APPROX_MAX_LPIPS = 0.10

# Kept for callers that imported the old names.
GATE_MEAN_LPIPS = ADOPTION_MEAN_LPIPS
GATE_MAX_LPIPS = ADOPTION_MAX_LPIPS


def tier(set_mean: float, set_max: float, noise_floor_max: float | None = None) -> str:
    """The world-model tier for a set's LPIPS pair.

    ``reorder`` needs a noise floor to mean anything: an arm indistinguishable
    from the reference *at this pipeline's reproducibility* is a reordering,
    not an approximation, and without the floor there is no way to tell that
    from a lucky `approx`. Callers that have not measured the floor get
    `approx` at worst, never `reorder`.
    """
    if set_max == 0.0:
        return "exact"
    if noise_floor_max is not None and set_max <= noise_floor_max:
        return "reorder"
    if set_mean <= APPROX_MEAN_LPIPS and set_max <= APPROX_MAX_LPIPS:
        return "approx"
    return "lossy"


def _load(directory: Path, prompt_id: str) -> dict:
    with np.load(directory / f"{prompt_id}.npz") as data:
        return {key: data[key] for key in data.files}


def _frames_to_lpips_input(frames: np.ndarray, device: torch.device) -> torch.Tensor:
    """``(T, H, W, 3)`` uint8 -> ``(T, 3, H, W)`` in [-1, 1], which is what
    LPIPS expects."""
    tensor = torch.from_numpy(frames).to(device=device, dtype=torch.float32) / 255.0
    return tensor.permute(0, 3, 1, 2) * 2.0 - 1.0


def video_scores(reference: np.ndarray, candidate: np.ndarray, device: torch.device, batch: int = 8) -> dict:
    """Per-frame LPIPS plus PSNR and SSIM over the clip."""
    import lpips
    from torchmetrics.functional import structural_similarity_index_measure

    if reference.shape != candidate.shape:
        raise ValueError(f"frame shape mismatch: {reference.shape} vs {candidate.shape}")

    net = lpips.LPIPS(net="alex").to(device).eval()
    ref = _frames_to_lpips_input(reference, device)
    cand = _frames_to_lpips_input(candidate, device)

    per_frame = []
    with torch.inference_mode():
        for start in range(0, ref.shape[0], batch):
            stop = start + batch
            per_frame.extend(net(ref[start:stop], cand[start:stop]).flatten().tolist())
    per_frame_array = np.asarray(per_frame, dtype=np.float64)

    # PSNR and SSIM on [0, 1]. Both are computed in frame batches: SSIM
    # concatenates five tensors the size of its input, so a whole 121-frame
    # clip at 480x864 asks for 2.9 GB in one allocation and OOMs on a GPU that
    # is doing anything else. Batching makes the scorer cost about a frame's
    # worth of memory, which also means it can run beside other work.
    squared_error = 0.0
    ssim_sum = 0.0
    batches = 0
    with torch.inference_mode():
        for start in range(0, ref.shape[0], batch):
            stop = start + batch
            ref01 = (ref[start:stop] + 1.0) / 2.0
            cand01 = (cand[start:stop] + 1.0) / 2.0
            squared_error += float(torch.sum((ref01 - cand01) ** 2))
            ssim_sum += float(structural_similarity_index_measure(cand01, ref01, data_range=1.0)) * ref01.shape[0]
            batches += ref01.shape[0]
    mse = squared_error / float(ref.numel())
    psnr = float("inf") if mse == 0 else 10.0 * math.log10(1.0 / mse)
    ssim = ssim_sum / max(batches, 1)

    return {
        "lpips_mean": round(float(per_frame_array.mean()), 6),
        "lpips_max": round(float(per_frame_array.max()), 6),
        "lpips_p95": round(float(np.percentile(per_frame_array, 95)), 6),
        "lpips_worst_frame": int(per_frame_array.argmax()),
        "psnr_db": round(psnr, 3),
        "ssim": round(ssim, 6),
        "frames": int(per_frame_array.size),
    }


def audio_scores(reference: np.ndarray, candidate: np.ndarray, sample_rate: int) -> dict:
    """Log-mel L1 and SI-SDR, the gate's two audio numbers.

    SI-SDR is scale-invariant on purpose: a decoder that reproduces the
    waveform at a different gain is not wrong in a way a listener would call
    wrong, and the pipeline peak-normalizes by default.
    """
    ref = np.asarray(reference, dtype=np.float64).reshape(-1)
    cand = np.asarray(candidate, dtype=np.float64).reshape(-1)
    length = min(ref.size, cand.size)
    ref, cand = ref[:length], cand[:length]
    if length == 0:
        return {"note": "no audio in one of the two runs"}

    # SI-SDR: project the candidate onto the reference, then the ratio of
    # that projection's energy to the residual's.
    scale = float(np.dot(cand, ref) / max(np.dot(ref, ref), 1e-12))
    projection = scale * ref
    noise = cand - projection
    signal_energy = max(float(np.dot(projection, projection)), 1e-20)
    noise_energy = max(float(np.dot(noise, noise)), 1e-20)
    si_sdr = 10.0 * math.log10(signal_energy / noise_energy)

    # Log-mel L1 via torchaudio-free mel: a plain magnitude spectrogram with
    # a mel filterbank is enough for an L1 distance and avoids another
    # dependency whose defaults would have to be recorded.
    log_mel_l1 = _log_mel_l1(ref, cand, sample_rate)
    return {
        "si_sdr_db": round(si_sdr, 3),
        "log_mel_l1": round(log_mel_l1, 6),
        "samples": int(length),
        "sample_rate": int(sample_rate),
    }


def _log_mel_l1(reference: np.ndarray, candidate: np.ndarray, sample_rate: int, n_fft: int = 1024) -> float:
    hop = n_fft // 4
    window = np.hanning(n_fft)

    def log_mel(signal: np.ndarray) -> np.ndarray:
        frames = 1 + max(0, (signal.size - n_fft) // hop)
        if frames == 0:
            return np.zeros((1, 1))
        spectra = np.empty((frames, n_fft // 2 + 1))
        for index in range(frames):
            chunk = signal[index * hop : index * hop + n_fft] * window
            spectra[index] = np.abs(np.fft.rfft(chunk))
        mel = spectra @ _mel_filterbank(n_fft, sample_rate)
        return np.log10(mel + 1e-6)

    ref_mel, cand_mel = log_mel(reference), log_mel(candidate)
    frames = min(ref_mel.shape[0], cand_mel.shape[0])
    return float(np.abs(ref_mel[:frames] - cand_mel[:frames]).mean())


def _mel_filterbank(n_fft: int, sample_rate: int, n_mels: int = 64) -> np.ndarray:
    def to_mel(hz):
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def to_hz(mel):
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    edges = to_hz(np.linspace(to_mel(0.0), to_mel(sample_rate / 2), n_mels + 2))
    bins = np.floor((n_fft + 1) * edges / sample_rate).astype(int)
    filters = np.zeros((n_fft // 2 + 1, n_mels))
    for m in range(n_mels):
        left, centre, right = bins[m], bins[m + 1], bins[m + 2]
        for k in range(left, min(centre, filters.shape[0])):
            filters[k, m] = (k - left) / max(centre - left, 1)
        for k in range(centre, min(right, filters.shape[0])):
            filters[k, m] = (right - k) / max(right - centre, 1)
    return filters


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference", type=Path, required=True, help="the reference arm's gate_run.py output dir")
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--no-locks",
        action="store_true",
        help="score without taking the host GPU locks. Only for a CPU run (--device cpu) or when "
        "nothing else is using the GPU: scoring on a shared GPU can perturb a timed run",
    )
    parser.add_argument(
        "--noise-floor-max",
        type=float,
        default=None,
        help="the max LPIPS of a same-arm control run, which is what lets an arm be called `reorder` "
        "rather than a lucky `approx`. Measured on this pipeline as 0.0030",
    )
    args = parser.parse_args()

    # The scorer runs on the GPU, so it takes the host's locks like anything
    # else that does. This is not ceremony: scoring without them once stole
    # 6.4 GB from a timed ABBA comparison that was holding them, which is
    # exactly the contamination the locks exist to prevent.
    with contextlib.ExitStack() as stack:
        if not args.no_locks:
            stack.enter_context(GpuLocks())
            foreign = foreign_gpu_procs()
            if foreign:
                print("foreign process(es) on the GPU; scoring anyway, but this is shared:", file=sys.stderr)
                for proc in foreign:
                    print(f"  {proc}", file=sys.stderr)
        return _score(args)


def _score(args) -> int:
    ref_manifest = json.loads((args.reference / "manifest.json").read_text())
    cand_manifest = json.loads((args.candidate / "manifest.json").read_text())
    if ref_manifest["geometry"] != cand_manifest["geometry"]:
        raise SystemExit("refusing to score: the two runs used different geometries")
    if ref_manifest["model"] != cand_manifest["model"]:
        raise SystemExit("refusing to score: the two runs used different models")

    shared = [r["id"] for r in ref_manifest["runs"]]
    shared = [i for i in shared if i in {r["id"] for r in cand_manifest["runs"]}]
    device = torch.device(args.device)

    rows = []
    for prompt_id in shared:
        reference = _load(args.reference, prompt_id)
        candidate = _load(args.candidate, prompt_id)
        row = {"id": prompt_id, "video": video_scores(reference["frames"], candidate["frames"], device)}
        if "audio" in reference and "audio" in candidate:
            row["audio"] = audio_scores(
                reference["audio"], candidate["audio"], int(reference.get("audio_sample_rate", 44100))
            )
        rows.append(row)
        v = row["video"]
        print(
            f"{prompt_id:<22} LPIPS mean {v['lpips_mean']:.4f} max {v['lpips_max']:.4f} "
            f"(worst frame {v['lpips_worst_frame']})  PSNR {v['psnr_db']:.2f} dB  SSIM {v['ssim']:.4f}"
            + (f"  SI-SDR {row['audio']['si_sdr_db']:.1f} dB" if "audio" in row else ""),
            flush=True,
        )

    means = [r["video"]["lpips_mean"] for r in rows]
    maxes = [r["video"]["lpips_max"] for r in rows]
    set_mean = round(float(np.mean(means)), 6) if means else None
    set_max = round(float(np.max(maxes)), 6) if maxes else None
    verdict = {
        "prompts": len(rows),
        "set_lpips_mean": set_mean,
        "set_lpips_max": set_max,
        "adoption_mean_limit": ADOPTION_MEAN_LPIPS,
        "adoption_max_limit": ADOPTION_MAX_LPIPS,
        "approx_mean_limit": APPROX_MEAN_LPIPS,
        "approx_max_limit": APPROX_MAX_LPIPS,
        "noise_floor_max": args.noise_floor_max,
    }
    verdict["passes_adoption_gate"] = bool(
        means and set_mean <= ADOPTION_MEAN_LPIPS and set_max <= ADOPTION_MAX_LPIPS
    )
    verdict["passes_approx_tier"] = bool(means and set_mean <= APPROX_MEAN_LPIPS and set_max <= APPROX_MAX_LPIPS)
    verdict["tier"] = tier(set_mean, set_max, args.noise_floor_max) if means else "unmeasured"
    # Name the prompts that breach on their own. The gate is a set threshold,
    # but which prompt breaches is the finding -- an arm that fails only on
    # rendered text is a different problem from one that fails everywhere.
    verdict["prompts_over_adoption_max"] = [r["id"] for r in rows if r["video"]["lpips_max"] > ADOPTION_MAX_LPIPS]
    verdict["prompts_over_approx_max"] = [r["id"] for r in rows if r["video"]["lpips_max"] > APPROX_MAX_LPIPS]

    print(
        f"\nset: mean LPIPS {set_mean}, max {set_max}"
        f"\n  adoption gate (mean <= {ADOPTION_MEAN_LPIPS}, max <= {ADOPTION_MAX_LPIPS}): "
        + ("PASSES" if verdict["passes_adoption_gate"] else "FAILS")
        + f"\n  tier: {verdict['tier']}"
        + (f" (noise floor max {args.noise_floor_max})" if args.noise_floor_max is not None else "")
    )
    if verdict["prompts_over_adoption_max"]:
        print("over the adoption max on their own: " + ", ".join(verdict["prompts_over_adoption_max"]))

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(
                {
                    "reference": str(args.reference),
                    "candidate": str(args.candidate),
                    "reference_arm": ref_manifest["arm"],
                    "candidate_arm": cand_manifest["arm"],
                    "candidate_compile_mode": cand_manifest.get("compile_mode"),
                    "geometry": ref_manifest["geometry"],
                    "prompt_set": ref_manifest.get("prompt_set"),
                    "verdict": verdict,
                    "per_prompt": rows,
                },
                indent=2,
            )
            + "\n"
        )
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
