# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Offline FP8 quantization of the Kandinsky 6 Pro DiT, to make W1 fit.

Why offline. vLLM-Omni's FP8 path quantizes at *load* time: the linear layers
are created in BF16, filled, and converted by
``process_weights_after_loading``. That needs the whole BF16 DiT resident first
— 58 GB for Pro's 29B parameters — which exceeds this host's ~50 GB of available
RAM, let alone the 32 GB GPU. The load therefore cannot be the thing that
quantizes. Doing it ahead of time, streaming tensor by tensor from the
checkpoint, needs only one tensor in memory at a time and produces a 29 GB
checkpoint that *does* fit host RAM, which is what
``--enable-layerwise-offload`` needs to stream blocks from.

The output is a standard vLLM FP8-serialized checkpoint: ``float8_e4m3fn``
weights, a ``weight_scale`` beside each, and a ``quantization_config`` in
``transformer/config.json`` so the loader builds FP8 linears directly
(``TransformerConfig.from_dict`` reads that key) instead of BF16 ones it would
have to convert.

Scales are **per tensor** by default, because that is what vLLM's native ``fp8``
method reads (it builds a ``PerTensorScaleParameter``, and a per-channel scale
fails its shape assertion on load). ``--scale channel`` keeps each output row's
own range instead of losing it to the matrix's single largest weight, which is
the better recipe but needs a ``compressed-tensors`` checkpoint; see
:func:`quantize_weight`.

Sensitive layers stay BF16 (:func:`keeps_bf16`). Which ones is a quality
question the gate answers, so the list is a flag, not a constant, and the
checkpoint records what was kept in its ``quantization_config.ignored_layers``.

Run it with the vLLM-Omni venv: the ignored-layer list is written in module-path
form via the pipeline's own name mapping, so the tool imports it rather than
keeping a second copy that could drift.

Usage::

    python quantize_dit_fp8.py \
        --src  /data/jooman/hf/hub/models--.../snapshots/<rev> \
        --dst  /data/jooman/k6/ckpt/pro-distill-fp8

The destination is a full model root: the quantized ``transformer/`` plus
symlinks to every other component, so ``vllm serve <dst>`` works unchanged and
no 20 GB text encoder is copied.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

# E4M3's largest finite magnitude. Scaling to this rather than to 240 or 256
# uses the format's full range; the clamp below catches the rounding edge.
FP8_E4M3_MAX = 448.0
FP8_DTYPE = torch.float8_e4m3fn

# Kept in BF16, matched against the checkpoint's own key names. The DiT stores
# its keys at the root (``visual_transformer_blocks.3.…``, ``out_layer.…``); the
# pipeline adds the ``transformer.`` prefix at load time.
#
# Matching is **anchored**, not substring: every transformer block contains a
# ``self_attention.out_layer`` and a ``cross_attention.out_layer``, so a bare
# "out_layer" substring rule would also protect 60 blocks' attention output
# projections — the largest GEMMs in the model — and leave the checkpoint at
# 40 GB instead of 32 GB, defeating the point.
#
# Why these, from the plan's sensitive set for a video DiT:
#  - ``.modulation.`` produces the shift/scale applied to every token, so its
#    error is multiplicative over the residual stream, not additive.
#  - the embeddings and the two output heads: one layer each, and the heads form
#    the n_grid x_0 predictions the policy then integrates ~13 times per step.
#  - the text and audio *branch* blocks: 56 small tensors, little to save, and
#    audio quality is scored on its own metrics.
#  - the first and last visual block: the vault's Qwen-Image 2.1 result is that
#    an early-step error grows about 20x by the final latent; the same argument
#    applies at the ends of a 60-block stack.
KEEP_BF16_ROOT_PREFIXES = (
    "out_layer.",
    "audio_out_layer.",
    "visual_embeddings.",
    "audio_embeddings.",
    "visual_token_type_embeddings",
    "video_text_embeddings.",
    "audio_text_embeddings.",
    "video_pooled_text_embeddings.",
    "audio_pooled_text_embeddings.",
    "video_time_embeddings.",
    "audio_time_embeddings.",
    "video_text_transformer_blocks.",
    "audio_text_transformer_blocks.",
)
# Any module whose name ends in "modulation": the blocks call theirs
# `visual_modulation`, `av_modulation`, `va_modulation` and `text_modulation`, so
# a ".modulation." substring rule silently protected only the two top-level
# output heads and quantized all 60 blocks' modulation after all.
KEEP_BF16_NAME_SUFFIXES = ("modulation",)
KEEP_BF16_BLOCK_PREFIXES = (
    "visual_transformer_blocks.0.",
    "visual_transformer_blocks.59.",
)


# The input and output projections only: every embedding and both output heads.
# What `minimal` keeps.
_MINIMAL_ROOT_PREFIXES = tuple(p for p in KEEP_BF16_ROOT_PREFIXES if not p.endswith("_text_transformer_blocks."))

# Which layers stay BF16, by name. `sensitive` is the plan's full sensitive set
# (above). `minimal` keeps only the embeddings and output heads, which are tiny;
# it exists because the sensitive set costs host RAM the 60 GB host does not
# have: at W1 the Pro DiT is streamed from pinned host memory, and keeping all
# 60 blocks' modulation, both text towers and blocks 0 and 59 in BF16 puts the
# checkpoint at 36.7 GB against 30.4 GB for `minimal` -- the difference between
# a server that loads and one the OOM killer takes. Which one passes the quality
# gate is what the gate is for.
KEEP_PROFILES = ("sensitive", "minimal", "none")


def keeps_bf16(name: str, profile: str = "sensitive") -> bool:
    """True when ``name`` must stay BF16 under ``profile``."""
    if profile == "none":
        return False
    if profile == "minimal":
        return name.startswith(_MINIMAL_ROOT_PREFIXES)
    if profile != "sensitive":
        raise ValueError(f"unknown keep profile {profile!r}; expected one of {KEEP_PROFILES}")
    components = name.split(".")
    return (
        name.startswith(KEEP_BF16_ROOT_PREFIXES)
        or name.startswith(KEEP_BF16_BLOCK_PREFIXES)
        or any(c.endswith(KEEP_BF16_NAME_SUFFIXES) for c in components)
    )


SHARD_BYTES = 4 * 1024**3


def is_quantizable(name: str, tensor_shape: tuple[int, ...], *, keep: str = "sensitive") -> bool:
    """True for a 2-D linear weight that is not on the keep-wide list.

    Only a rank-2 ``*.weight`` is a GEMM operand. Biases, norms, RoPE buffers and
    the token-type embedding are rank <= 1 or not weights, and quantizing them
    would perturb arithmetic that costs nothing to leave alone.
    """
    if not name.endswith(".weight") or len(tensor_shape) != 2:
        return False
    return not keeps_bf16(name, keep)


def quantize_weight(weight: torch.Tensor, *, granularity: str = "tensor") -> tuple[torch.Tensor, torch.Tensor]:
    """``(out, in)`` BF16 -> FP8 E4M3 plus its scale.

    ``granularity``:

    ``tensor``
        One scalar for the whole matrix. This is what vLLM's native ``fp8``
        quantization method reads: it creates a ``PerTensorScaleParameter``, so a
        per-channel scale fails the loader's shape assertion. Default for that
        reason.
    ``channel``
        One scale per output row, which keeps each row's own range instead of
        losing it to the single largest weight in the matrix. Better quality at
        0.004% more storage, but it needs a ``compressed-tensors`` checkpoint
        with ``strategy: channel`` rather than the native ``fp8`` method, so it
        is here for that follow-up and is not yet loadable by this pipeline.

    The scale is ``amax / 448`` computed in FP32, so a BF16 amax cannot round the
    scale up and clip the true maximum. An all-zero tensor or row would divide by
    zero, so its scale is forced to 1.0 and it quantizes to zeros.
    """
    if granularity not in ("tensor", "channel"):
        raise ValueError(f"unknown scale granularity {granularity!r}")
    as_float = weight.to(torch.float32)
    if granularity == "channel":
        amax = as_float.abs().amax(dim=1, keepdim=True)
    else:
        amax = as_float.abs().amax()
    scale = (amax / FP8_E4M3_MAX).clamp(min=torch.finfo(torch.float32).tiny)
    scale = torch.where(amax > 0, scale, torch.ones_like(scale))
    quantized = (as_float / scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(FP8_DTYPE)
    scale = scale.to(torch.float32)
    # A per-tensor scale is written with shape (1,), not as a 0-dim scalar.
    # vLLM's PerTensorScaleParameter is 1-D of size 1; a 0-dim tensor still
    # loads (the loader accepts "1-D param, scalar value") but is not what
    # process_weights_after_loading then reduces over.
    return quantized, scale.reshape(1) if granularity == "tensor" else scale


def module_path(checkpoint_key: str) -> str:
    """A transformer checkpoint key -> the path of the module it loads into.

    Uses the pipeline's own ``_adapt_k6_weight_name``, so the two can never drift:
    whatever the loader renames, this renames identically.
    """
    from vllm_omni.diffusion.models.kandinsky6.pipeline_kandinsky6 import _adapt_k6_weight_name

    return _adapt_k6_weight_name(f"transformer.{checkpoint_key}")[len("transformer.") :]


def find_dit_weights(src_transformer: Path) -> list[Path]:
    """The transformer's safetensors file(s), sharded or single."""
    shards = sorted(src_transformer.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no .safetensors under {src_transformer}")
    return shards


def quantize_checkpoint(
    src: Path,
    dst: Path,
    *,
    keep: str = "sensitive",
    granularity: str = "tensor",
    shard_bytes: int = SHARD_BYTES,
) -> dict[str, object]:
    """Write an FP8 copy of ``src``'s transformer into ``dst``; link the rest."""
    src_transformer = src / "transformer"
    dst_transformer = dst / "transformer"
    dst_transformer.mkdir(parents=True, exist_ok=True)

    with (src_transformer / "config.json").open(encoding="utf-8") as handle:
        config = json.load(handle)

    shard_index: dict[str, str] = {}
    buffer: dict[str, torch.Tensor] = {}
    buffer_bytes = 0
    shard_no = 0
    written: list[Path] = []
    quantized_names: list[str] = []
    kept_names: list[str] = []
    bytes_in = 0
    bytes_out = 0
    started = time.perf_counter()

    def flush() -> None:
        nonlocal buffer, buffer_bytes, shard_no
        if not buffer:
            return
        shard_no += 1
        path = dst_transformer / f"diffusion_pytorch_model-{shard_no:05d}.safetensors"
        save_file(buffer, str(path), metadata={"format": "pt"})
        for key in buffer:
            shard_index[key] = path.name
        written.append(path)
        buffer = {}
        buffer_bytes = 0

    for shard in find_dit_weights(src_transformer):
        with safe_open(str(shard), framework="pt") as reader:
            for name in reader.keys():  # noqa: SIM118 - safetensors reader, not a dict
                tensor = reader.get_tensor(name)
                bytes_in += tensor.numel() * tensor.element_size()
                if is_quantizable(name, tuple(tensor.shape), keep=keep):
                    weight, scale = quantize_weight(tensor, granularity=granularity)
                    buffer[name] = weight
                    buffer[name.removesuffix("weight") + "weight_scale"] = scale
                    buffer_bytes += weight.numel() + scale.numel() * 4
                    bytes_out += weight.numel() + scale.numel() * 4
                    quantized_names.append(name)
                else:
                    kept = tensor.to(torch.bfloat16) if tensor.is_floating_point() else tensor
                    buffer[name] = kept
                    buffer_bytes += kept.numel() * kept.element_size()
                    bytes_out += kept.numel() * kept.element_size()
                    if name.endswith(".weight") and tensor.dim() == 2:
                        kept_names.append(name)
                del tensor
                if buffer_bytes >= shard_bytes:
                    flush()
    flush()

    # The ignored list is what the loader must *not* build as FP8, and vLLM
    # matches it against each layer's **prefix** -- its path in the module tree.
    # Checkpoint key names are not that path: the published bundles name the FFN
    # `feed_forward.net.0.proj`, the text towers' attention `attn` and the time
    # embedding `timestep_embedder.linear_1`, all of which the loader remaps.
    # Writing the raw key names here produced entries that matched nothing, so
    # every layer meant to stay BF16 was built as an FP8 layer whose weight_scale
    # this checkpoint does not carry -- which reads back as whatever was in its
    # uninitialized allocation, and the first forward returns NaN.
    ignored_layers = sorted({module_path(name).removesuffix(".weight") for name in kept_names})
    config["quantization_config"] = {
        "quant_method": "fp8",
        "activation_scheme": "dynamic",
        "weight_block_size": None,
        "ignored_layers": ignored_layers,
    }
    with (dst_transformer / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    index = {
        "metadata": {"total_size": bytes_out},
        "weight_map": shard_index,
    }
    with (dst_transformer / "diffusion_pytorch_model.safetensors.index.json").open("w", encoding="utf-8") as handle:
        json.dump(index, handle, indent=2)

    link_sibling_components(src, dst)

    return {
        "src": str(src),
        "dst": str(dst),
        "shards_written": [p.name for p in written],
        "n_quantized": len(quantized_names),
        "n_kept_2d": len(kept_names),
        "ignored_layers": ignored_layers,
        "bytes_in": bytes_in,
        "bytes_out": bytes_out,
        "scale_granularity": granularity,
        "keep_profile": keep,
        "compression": bytes_in / bytes_out if bytes_out else None,
        "seconds": time.perf_counter() - started,
    }


def link_sibling_components(src: Path, dst: Path) -> list[str]:
    """Symlink every component but ``transformer/`` into ``dst``.

    The text encoder alone is ~16 GB and is not being changed, so copying it
    would waste disk and time. ``model_index.json`` and the scheduler config are
    small and are copied, so the destination stays a valid model root even if the
    source snapshot is garbage-collected.
    """
    linked: list[str] = []
    for entry in sorted(src.iterdir()):
        target = dst / entry.name
        if entry.name == "transformer" or target.exists():
            continue
        if entry.is_dir():
            target.symlink_to(entry.resolve(), target_is_directory=True)
            linked.append(entry.name)
        elif entry.suffix == ".json":
            shutil.copy2(entry.resolve(), target)
            linked.append(entry.name)
    return linked


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True, help="source model root (a Diffusers snapshot)")
    parser.add_argument("--dst", type=Path, required=True, help="destination model root to create")
    parser.add_argument(
        "--keep",
        choices=KEEP_PROFILES,
        default="sensitive",
        help="which layers stay BF16: the plan's sensitive set, only embeddings and heads, or none",
    )
    parser.add_argument(
        "--scale",
        choices=("tensor", "channel"),
        default="tensor",
        help="scale granularity; 'tensor' is what vLLM's native fp8 method loads",
    )
    parser.add_argument("--shard-gib", type=float, default=4.0)
    args = parser.parse_args()

    report = quantize_checkpoint(
        args.src.resolve(),
        args.dst.resolve(),
        keep=args.keep,
        granularity=args.scale,
        shard_bytes=int(args.shard_gib * 1024**3),
    )
    report_path = args.dst / "quantization_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "ignored_layers"}, indent=2))
    print(f"report: {report_path}")
    print(f"free on destination filesystem: {shutil.disk_usage(args.dst).free / 1024**3:.1f} GiB")
    os.sync()


if __name__ == "__main__":
    main()
