# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Kandinsky 6's video VAE decode at W1, alone: variants vs the served eager decode.

The served decode is fp16 and eager, and it plans its own tiling on every call
from the GPU's free memory (``get_dec_optimal_tiling``): 16-frame temporal
chunks every 8 frames, and full-frame or factorized spatial tiles. At W1 it is
~19 s of a ~190 s request: conv3d (cuDNN, NCHW, so it transposes in and out)
55%, GroupNorm 18%, the causal ``replicate`` pad as its own copy 7%, layout
transposes 7%. Because the plan follows free memory, run the variants in
separate processes or read the timing spread with that in mind. This
times the same decode on W1-shaped latents for each variant and reports its
error against the eager output, so a rewrite is proved at the decode level
before it is served.

Decode time depends on shape, not content, so random latents at W1's latent
shape give honest timings. Errors on random latents bound the kernel-level
difference; the end-to-end quality check is the gate on served outputs.

Usage::

    python vae_decode_bench.py --vae-dir <snapshot>/vae --variants eager,compile --repeats 3
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from safetensors.torch import load_file

from vllm_omni.diffusion.models.kandinsky6 import modeling_kandinsky6_vae as vae_mod
from vllm_omni.diffusion.models.kandinsky6.modeling_kandinsky6_vae import AutoencoderKLHunyuanVideo

# W1: 121 frames at 864x480 -> latents (16, 31, 60, 108).
W1_LATENT = (1, 16, 31, 60, 108)


def load_vae(vae_dir: Path, device: torch.device) -> AutoencoderKLHunyuanVideo:
    config = json.loads((vae_dir / "config.json").read_text())
    vae = AutoencoderKLHunyuanVideo.from_config(config)
    state = load_file(str(vae_dir / "diffusion_pytorch_model.safetensors"))
    missing, _ = vae.load_state_dict(state, strict=False)
    decoder_missing = [k for k in missing if k.startswith(("decoder.", "post_quant_conv."))]
    if decoder_missing:
        raise RuntimeError(f"decoder weights missing from the checkpoint: {decoder_missing[:5]}")
    vae = vae.to(device=device, dtype=torch.float16).eval()
    # As served: decode() replans tiling from free memory on every call.
    vae.use_framewise_decoding = True
    return vae


def apply_variant(vae: AutoencoderKLHunyuanVideo, variant: str) -> None:
    if variant == "eager":
        return
    if variant == "no_empty_cache":
        vae_mod.current_omni_platform.empty_cache = lambda: None
        return
    if variant == "compile":
        vae.decoder = torch.compile(vae.decoder, dynamic=False)
        return
    if variant == "compile_cl3d":
        vae.decoder = vae.decoder.to(memory_format=torch.channels_last_3d)
        vae.decoder = torch.compile(vae.decoder, dynamic=False)
        return
    raise ValueError(f"unknown variant {variant!r}")


def decode(vae: AutoencoderKLHunyuanVideo, z: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return vae.decode(z).sample


def timed(vae: AutoencoderKLHunyuanVideo, z: torch.Tensor, repeats: int) -> tuple[list[float], torch.Tensor]:
    out = decode(vae, z)  # warm-up (and compile)
    torch.accelerator.synchronize()
    walls = []
    for _ in range(repeats):
        torch.accelerator.synchronize()
        start = time.perf_counter()
        out = decode(vae, z)
        torch.accelerator.synchronize()
        walls.append(time.perf_counter() - start)
    return walls, out


def errors(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    r, c = reference.float(), candidate.float()
    return {
        "max_abs": float((r - c).abs().max()),
        "rel_l2": float((r - c).norm() / r.norm()),
        "bit_identical": bool(torch.equal(reference, candidate)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--vae-dir", type=Path, required=True)
    parser.add_argument("--variants", default="eager,no_empty_cache,compile,compile_cl3d")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    device = torch.device("cuda")
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    z = torch.randn(W1_LATENT, generator=generator).to(device=device, dtype=torch.float16)

    results = {}
    reference = None
    original_empty_cache = vae_mod.current_omni_platform.empty_cache
    for variant in args.variants.split(","):
        vae_mod.current_omni_platform.empty_cache = original_empty_cache
        vae = load_vae(args.vae_dir, device)
        apply_variant(vae, variant)
        walls, out = timed(vae, z, args.repeats)
        out = out.cpu()
        if reference is None:
            reference = out
        results[variant] = {
            "median_s": statistics.median(walls),
            "min_s": min(walls),
            "max_s": max(walls),
            "n": len(walls),
            "shape": list(out.shape),
            **errors(reference, out),
            "peak_mem_gib": torch.accelerator.max_memory_allocated() / 2**30,
        }
        print(variant, json.dumps(results[variant]), flush=True)
        del vae
        torch.accelerator.empty_cache()
        torch.accelerator.reset_peak_memory_stats()
    vae_mod.current_omni_platform.empty_cache = original_empty_cache
    if args.out:
        args.out.write_text(json.dumps({"latent_shape": W1_LATENT, "variants": results}, indent=2))


if __name__ == "__main__":
    main()
