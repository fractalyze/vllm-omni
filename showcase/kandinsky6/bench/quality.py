# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""The quality gate: a candidate MP4 against the BF16 reference MP4.

The plan's gate, in full:

- **Video** LPIPS per frame (mean and max over frames), PSNR, SSIM.
- **Audio** log-mel L1 and SI-SDR.
- **Tier** exact / reorder / approx / lossy, from the world-model vocabulary.
  ``approx`` needs mean LPIPS <= 0.05 **and** max LPIPS <= 0.10 over a prompt
  set.

Three things decide whether the numbers mean anything.

**Max over frames, not just mean.** A 121-frame clip whose face melts for six
frames has an excellent mean. The plan's gate is a conjunction for that reason,
and :func:`score_set` reports the worst frame of the worst prompt so the
failure has an address.

**The encode floor is measured, not assumed.** Both arms' pixels reach us
through an H.264 encode, so a non-zero LPIPS is the floor even for an
arithmetically identical arm. :func:`encode_floor` scores a reference against a
re-encode of itself and gives that floor. A candidate whose LPIPS sits at the
floor is ``exact`` within what this gate can see, and nothing below the floor
is a measurement. Calling a difference "approx" without that number in hand is
an unsupported claim.

**Per-category scores, because the gate is prompt-dependent.** The vault's
``c-qwen-image21-fp8-quality-is-prompt-dependent-2026-09``: one FP8 recipe
passed an 8-prompt set at LPIPS 0.034 and failed another at 0.162, worst on
rendered text and faces. Prompts therefore carry categories and
:func:`score_set` breaks the score down by them, so "passes the gate" cannot
hide a category that does not.

Metrics come from ``lpips`` and ``torchmetrics``, both already in the repo's
dev dependencies; frames and audio are decoded with PyAV.
"""

from __future__ import annotations

import json
import statistics
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

# Gate thresholds from the plan. A tier is a claim about arithmetic; these
# bound what the pixels allow us to claim.
APPROX_MEAN_LPIPS = 0.05
APPROX_MAX_LPIPS = 0.10

MEL_BINS = 128
MEL_FFT = 1024
MEL_HOP = 256


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------- decoding


def read_video(path: Path, *, max_frames: int | None = None) -> np.ndarray:
    """Decode ``path`` to ``(T, H, W, 3)`` uint8 RGB."""
    import av

    frames: list[np.ndarray] = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for frame in container.decode(stream):
            frames.append(frame.to_ndarray(format="rgb24"))
            if max_frames is not None and len(frames) >= max_frames:
                break
    if not frames:
        raise ValueError(f"{path}: no video frames decoded")
    return np.stack(frames)


def read_audio(path: Path) -> tuple[np.ndarray, int]:
    """Decode ``path``'s audio to mono float32 in ``[-1, 1]`` plus its rate.

    Returns an empty array when the container has no audio stream, which is a
    legitimate state (``sample_audio=false``) and not an error.
    """
    import av

    chunks: list[np.ndarray] = []
    rate = 0
    with av.open(str(path)) as container:
        if not container.streams.audio:
            return np.zeros(0, dtype=np.float32), 0
        stream = container.streams.audio[0]
        rate = int(stream.rate)
        resampler = av.audio.resampler.AudioResampler(format="fltp", layout="mono", rate=rate)
        for frame in container.decode(stream):
            for resampled in resampler.resample(frame):
                chunks.append(resampled.to_ndarray().reshape(-1).astype(np.float32))
    if not chunks:
        return np.zeros(0, dtype=np.float32), rate
    return np.concatenate(chunks), rate


# ---------------------------------------------------------------- video


def _lpips_net(cache: dict[str, Any]) -> Any:
    if "lpips" not in cache:
        import lpips as lpips_lib

        cache["lpips"] = lpips_lib.LPIPS(net="alex").to(_device()).eval()
    return cache["lpips"]


def score_video(
    reference: Path,
    candidate: Path,
    *,
    batch: int = 8,
    cache: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """LPIPS (per frame), PSNR and SSIM of ``candidate`` against ``reference``."""
    from torchmetrics.functional.image import (
        peak_signal_noise_ratio,
        structural_similarity_index_measure,
    )

    cache = cache if cache is not None else {}
    ref = read_video(reference)
    cand = read_video(candidate)
    n = min(len(ref), len(cand))
    if ref.shape[1:] != cand.shape[1:]:
        raise ValueError(f"geometry differs: reference {ref.shape[1:]} vs candidate {cand.shape[1:]}")
    ref, cand = ref[:n], cand[:n]

    device = _device()
    net = _lpips_net(cache)
    per_frame: list[float] = []
    psnrs: list[float] = []
    ssims: list[float] = []

    with torch.no_grad():
        for start in range(0, n, batch):
            # (B, 3, H, W) in [0, 1]; LPIPS wants [-1, 1].
            r = torch.from_numpy(ref[start : start + batch]).to(device).permute(0, 3, 1, 2).float() / 255.0
            c = torch.from_numpy(cand[start : start + batch]).to(device).permute(0, 3, 1, 2).float() / 255.0
            per_frame.extend(net(r * 2 - 1, c * 2 - 1).flatten().cpu().tolist())
            for i in range(r.shape[0]):
                psnrs.append(float(peak_signal_noise_ratio(c[i : i + 1], r[i : i + 1], data_range=1.0)))
                ssims.append(float(structural_similarity_index_measure(c[i : i + 1], r[i : i + 1], data_range=1.0)))

    worst = int(np.argmax(per_frame))
    return {
        "n_frames": n,
        "lpips_mean": float(statistics.fmean(per_frame)),
        "lpips_max": float(max(per_frame)),
        "lpips_worst_frame": worst,
        "lpips_per_frame": [float(v) for v in per_frame],
        "psnr_mean": float(statistics.fmean(psnrs)),
        "psnr_min": float(min(psnrs)),
        "ssim_mean": float(statistics.fmean(ssims)),
        "ssim_min": float(min(ssims)),
    }


# ---------------------------------------------------------------- audio


def _log_mel(wave: np.ndarray, rate: int) -> torch.Tensor:
    import torchaudio

    mel = torchaudio.transforms.MelSpectrogram(sample_rate=rate, n_fft=MEL_FFT, hop_length=MEL_HOP, n_mels=MEL_BINS)
    return torch.log(mel(torch.from_numpy(wave).float()) + 1e-6)


def score_audio(reference: Path, candidate: Path) -> dict[str, Any]:
    """Log-mel L1 and SI-SDR of ``candidate``'s audio against ``reference``'s."""
    from torchmetrics.functional.audio import scale_invariant_signal_distortion_ratio

    ref, ref_rate = read_audio(reference)
    cand, cand_rate = read_audio(candidate)
    if ref.size == 0 or cand.size == 0:
        return {"present": False, "reason": "one side has no audio stream"}
    if ref_rate != cand_rate:
        raise ValueError(f"sample rate differs: reference {ref_rate} vs candidate {cand_rate}")

    n = min(ref.size, cand.size)
    ref, cand = ref[:n], cand[:n]
    l1 = float(torch.mean(torch.abs(_log_mel(ref, ref_rate) - _log_mel(cand, cand_rate))))
    si_sdr = float(scale_invariant_signal_distortion_ratio(torch.from_numpy(cand), torch.from_numpy(ref)))
    return {
        "present": True,
        "sample_rate": ref_rate,
        "n_samples": int(n),
        "logmel_l1": l1,
        "si_sdr_db": si_sdr,
    }


# ---------------------------------------------------------------- the gate


def tier_for(mean_lpips: float, max_lpips: float, *, floor_mean: float | None = None) -> str:
    """The world-model tier the *pixels* support.

    This never returns ``exact`` or ``reorder`` on its own: those are claims
    about arithmetic (bit-identical, or identical up to summation order) that
    pixels cannot establish. What it can say is that a candidate sits at the
    encode floor, which is reported as ``at-encode-floor`` for the recorder to
    turn into ``exact``/``reorder`` only with an arithmetic argument.
    """
    if floor_mean is not None and mean_lpips <= floor_mean:
        return "at-encode-floor"
    if mean_lpips <= APPROX_MEAN_LPIPS and max_lpips <= APPROX_MAX_LPIPS:
        return "approx"
    return "lossy"


@dataclass
class PromptScore:
    prompt_id: str
    categories: list[str]
    video: dict[str, Any]
    audio: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        # The per-frame series is kept out of the set-level summary: it is tens
        # of numbers per prompt and belongs in the per-prompt artifact.
        video = {k: v for k, v in self.video.items() if k != "lpips_per_frame"}
        return {
            "prompt_id": self.prompt_id,
            "categories": self.categories,
            "video": video,
            "audio": self.audio,
        }


def score_set(
    pairs: list[tuple[str, list[str], Path, Path]],
    *,
    floor_mean: float | None = None,
) -> dict[str, Any]:
    """Score a whole prompt set.

    ``pairs`` is ``(prompt_id, categories, reference_mp4, candidate_mp4)``. The
    gate is evaluated over the set, as the plan specifies: the mean is the mean
    over prompts of the per-prompt frame means, and the max is the single worst
    frame anywhere in the set.
    """
    cache: dict[str, Any] = {}
    scores: list[PromptScore] = []
    for prompt_id, categories, ref, cand in pairs:
        scores.append(
            PromptScore(
                prompt_id=prompt_id,
                categories=list(categories),
                video=score_video(ref, cand, cache=cache),
                audio=score_audio(ref, cand),
            )
        )

    means = [s.video["lpips_mean"] for s in scores]
    maxes = [s.video["lpips_max"] for s in scores]
    set_mean, set_max = float(statistics.fmean(means)), float(max(maxes))
    worst = scores[int(np.argmax(maxes))]

    by_category: dict[str, dict[str, float]] = {}
    for category in sorted({c for s in scores for c in s.categories}):
        subset = [s for s in scores if category in s.categories]
        by_category[category] = {
            "n_prompts": len(subset),
            "lpips_mean": float(statistics.fmean([s.video["lpips_mean"] for s in subset])),
            "lpips_max": float(max(s.video["lpips_max"] for s in subset)),
        }

    audio = [s.audio for s in scores if s.audio.get("present")]
    return {
        "n_prompts": len(scores),
        "lpips_mean": set_mean,
        "lpips_max": set_max,
        "worst_prompt": worst.prompt_id,
        "worst_prompt_frame": worst.video["lpips_worst_frame"],
        "psnr_mean": float(statistics.fmean([s.video["psnr_mean"] for s in scores])),
        "ssim_mean": float(statistics.fmean([s.video["ssim_mean"] for s in scores])),
        "logmel_l1_mean": float(statistics.fmean([a["logmel_l1"] for a in audio])) if audio else None,
        "si_sdr_db_min": float(min(a["si_sdr_db"] for a in audio)) if audio else None,
        "encode_floor_lpips_mean": floor_mean,
        "tier": tier_for(set_mean, set_max, floor_mean=floor_mean),
        "gate_pass_approx": set_mean <= APPROX_MEAN_LPIPS and set_max <= APPROX_MAX_LPIPS,
        "by_category": by_category,
        "per_prompt": [s.as_dict() for s in scores],
    }


def encode_floor(reference: Path, *, crf: int = 18) -> dict[str, Any]:
    """The gate's noise floor: a reference scored against a re-encode of itself.

    Every number this module produces is a comparison of two H.264 encodes, so
    the floor is what an arithmetically identical arm would score. Report it
    with any claim the gate supports; a difference below it is not a difference.
    """
    with tempfile.TemporaryDirectory() as tmp:
        again = Path(tmp) / "reencoded.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(reference),
                "-c:v",
                "libx264",
                "-crf",
                str(crf),
                "-c:a",
                "copy",
                str(again),
            ],
            check=True,
        )
        video = score_video(reference, again)
        audio = score_audio(reference, again)
    return {
        "lpips_mean": video["lpips_mean"],
        "lpips_max": video["lpips_max"],
        "psnr_mean": video["psnr_mean"],
        "ssim_mean": video["ssim_mean"],
        "audio": audio,
        "crf": crf,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Score a candidate MP4 against a reference MP4.")
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path, nargs="?")
    parser.add_argument("--floor", action="store_true", help="report the encode floor of the reference instead")
    args = parser.parse_args()

    if args.floor:
        print(json.dumps(encode_floor(args.reference), indent=2))
        return
    if args.candidate is None:
        parser.error("a candidate is required unless --floor is given")
    video = score_video(args.reference, args.candidate)
    out = {
        "video": {k: v for k, v in video.items() if k != "lpips_per_frame"},
        "audio": score_audio(args.reference, args.candidate),
        "tier": tier_for(video["lpips_mean"], video["lpips_max"]),
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
