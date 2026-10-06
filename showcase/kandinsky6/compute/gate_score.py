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
import json
import math
from pathlib import Path

import numpy as np
import torch

# The gate of PLAN.md, for the `approx` tier.
GATE_MEAN_LPIPS = 0.05
GATE_MAX_LPIPS = 0.10


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

    # PSNR and SSIM on [0, 1], over the whole clip.
    ref01 = (ref + 1.0) / 2.0
    cand01 = (cand + 1.0) / 2.0
    mse = float(torch.mean((ref01 - cand01) ** 2))
    psnr = float("inf") if mse == 0 else 10.0 * math.log10(1.0 / mse)
    with torch.inference_mode():
        ssim = float(structural_similarity_index_measure(cand01, ref01, data_range=1.0))

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
    args = parser.parse_args()

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
    verdict = {
        "prompts": len(rows),
        "set_lpips_mean": round(float(np.mean(means)), 6) if means else None,
        "set_lpips_max": round(float(np.max(maxes)), 6) if maxes else None,
        "gate_mean_limit": GATE_MEAN_LPIPS,
        "gate_max_limit": GATE_MAX_LPIPS,
    }
    verdict["passes_approx_gate"] = bool(
        means and verdict["set_lpips_mean"] <= GATE_MEAN_LPIPS and verdict["set_lpips_max"] <= GATE_MAX_LPIPS
    )
    # Name the prompts that fail on their own, not just the set aggregate: the
    # gate is a set threshold, but a single bad prompt is the finding.
    verdict["prompts_over_max_limit"] = [
        r["id"] for r in rows if r["video"]["lpips_max"] > GATE_MAX_LPIPS
    ]

    print(
        f"\nset: mean LPIPS {verdict['set_lpips_mean']} (limit {GATE_MEAN_LPIPS}), "
        f"max {verdict['set_lpips_max']} (limit {GATE_MAX_LPIPS}) -> "
        + ("PASSES approx" if verdict["passes_approx_gate"] else "FAILS approx")
    )
    if verdict["prompts_over_max_limit"]:
        print("over the max limit on their own: " + ", ".join(verdict["prompts_over_max_limit"]))

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
