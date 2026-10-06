# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Time and profile Kandinsky 6 DiT blocks at Pro shapes on one GPU.

Kandinsky 6.0 Pro does not fit on an RTX 5090 (29B parameters, 60 GB in
BF16), so kernel work on the Pro shape cannot wait for a whole pipeline.
This builds the port's own blocks -- ``Kandinsky6FusedTransformerDecoderBlock``
for the text-to-video-and-audio backbone, ``Kandinsky6TransformerDecoderBlock``
for video only -- at the Pro config's dimensions with random weights, feeds
them the headline workload's token counts, and reports where the time goes.

The numbers are per block. A full Pro forward is 60 visual blocks plus 4
text blocks, so ``--blocks 1`` times 1/60th of a step's visual backbone;
``--repeats`` controls how many timed forwards, never how many blocks.
Random weights change nothing about the time: every shape, launch
configuration and kernel is the checkpoint's. They do make the output
meaningless, so this script never claims a quality number.

Geometry defaults to W1 of ``showcase/kandinsky6/PLAN.md``: 864x480, 121
frames at 24 fps with audio, which the Hunyuan VAE (4x in time, 8x in
space) and the DiT's 1x2x2 patching turn into 31 x 30 x 54 = 50,220 visual
tokens and 218 audio latent frames.

    python block_profile.py --list-shapes
    python block_profile.py --backend CUDNN_ATTN --repeats 5
    python block_profile.py --target attn --backend TORCH_SDPA --json out.json

Timed runs hold every GPU lock on the host and record what ``nvidia-smi``
showed, so a run that shared the GPU can be discarded afterwards.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
from gpulock import GpuLocks, foreign_gpu_procs  # noqa: E402
from kernel_classes import CATEGORIES, classify, split_by_category  # noqa: E402

# Kandinsky 6.0 Pro-distill's DiT config, read from its bundle's
# transformer/config.json (snapshot 7a1a4033). `head_dim` is sum(axes_dims)
# = 128, so the video branch is 32 heads of 128 and the audio branch 16.
#
# The three `True` flags are not the port's defaults, and they change the
# block's structure, not just its numbers: `cross_gates` makes the
# cross-modal modulation emit `2*model_dim + model_dim_a` instead of
# `3*model_dim`, `ca_rope` applies RoPE inside the cross-modal attention,
# and `fix_modulation` feeds each cross-modal modulation its own modality's
# time embedding. A profile taken with the port's defaults would be of a
# block the checkpoint does not have.
PRO = SimpleNamespace(
    model_dim=4096,
    ff_dim=16384,
    time_dim=1024,
    axes_dims=(32, 48, 48),
    num_visual_blocks=60,
    num_text_blocks=4,
    model_dim_a=2048,
    ff_dim_a=7168,
    axes_dims_a=(32, 48, 48),
    in_audio_dim=40,
    ca_rope=True,
    cross_gates=True,
    fix_modulation=True,
    text_token_padding=True,
)
# Kandinsky 6.0 Lite (3.7B), from its bundle's transformer/config.json
# (snapshot 6510114a): the same block structure and the same three flags at
# 44% of the width, half the head dim and about half the depth. It is what
# fits on a 32 GB card end to end, so pipeline-level work runs here while
# kernel work runs at PRO's shapes.
LITE = SimpleNamespace(
    model_dim=1792,
    ff_dim=7168,
    time_dim=512,
    axes_dims=(16, 24, 24),
    num_visual_blocks=32,
    num_text_blocks=2,
    model_dim_a=896,
    ff_dim_a=3584,
    axes_dims_a=(16, 24, 24),
    in_audio_dim=40,
    ca_rope=True,
    cross_gates=True,
    fix_modulation=True,
    text_token_padding=True,
)
CONFIGS = {"pro": PRO, "lite": LITE}


@dataclass
class Shapes:
    """Token counts one forward sees, derived from the request's geometry."""

    height: int
    width: int
    num_frames: int
    fps: float
    text_len: int
    latent_frames: int = 0
    latent_h: int = 0
    latent_w: int = 0
    visual_tokens: int = 0
    audio_len: int = 0

    def __post_init__(self) -> None:
        # Hunyuan VAE: 4x in time (first frame kept), 8x in space; the DiT
        # patches 1x2x2 on top.
        self.latent_frames = (self.num_frames - 1) // 4 + 1
        self.latent_h = self.height // 8 // 2
        self.latent_w = self.width // 8 // 2
        self.visual_tokens = self.latent_frames * self.latent_h * self.latent_w
        # pipeline_kandinsky6.audio_latent_duration, with the audio VAE's
        # 44.1 kHz / 1024 downsample.
        sample_frames = (self.latent_frames - 1) * 4 + 1
        self.audio_len = int(math.ceil(sample_frames / self.fps * 44100 / 1024))


# W1 of PLAN.md. `text_len` is not fixed by the geometry (Qwen2.5-VL-7B
# emits one embedding per prompt token); 256 is a long prompt and is what
# the race uses so the cross-attention cost is not understated.
W1 = dict(height=480, width=864, num_frames=121, fps=24.0, text_len=256)
SMOKE = dict(height=320, width=512, num_frames=25, fps=24.0, text_len=64)
GEOMETRIES = {"w1": W1, "smoke": SMOKE}


@dataclass
class Timing:
    repeats: int
    ms: list[float] = field(default_factory=list)

    @property
    def median(self) -> float:
        return statistics.median(self.ms)

    def summary(self) -> dict:
        return {
            "median_ms": round(self.median, 3),
            "min_ms": round(min(self.ms), 3),
            "max_ms": round(max(self.ms), 3),
            "spread_pct": round(100.0 * (max(self.ms) - min(self.ms)) / self.median, 2),
            "repeats": self.repeats,
            "all_ms": [round(v, 3) for v in self.ms],
        }


@contextlib.contextmanager
def single_process_parallel():
    """A tensor-parallel group of one, for the life of the ``with`` block.

    vLLM's parallel ``Linear`` layers -- which the port uses for every
    projection -- need a model-parallel group, and `initialize_model_parallel`
    in vLLM 0.31.0 reads `get_current_vllm_config()`, so a `VllmConfig` has
    to be current while it runs *and* while layers are built. Holding the
    config open for the whole block is therefore not laziness: a layer
    constructed outside it raises the same assertion.

    Mirrors ``tests/diffusion/models/kandinsky6/test_transformer_kandinsky6.py``,
    plus that config context. Idempotent for the group itself, so nesting is
    harmless.
    """
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed.parallel_state import (
        cleanup_dist_env_and_memory,
        init_distributed_environment,
        initialize_model_parallel,
        model_parallel_is_initialized,
    )

    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29517")
    with set_current_vllm_config(VllmConfig()):
        started_group = not model_parallel_is_initialized()
        if started_group:
            init_distributed_environment(world_size=1, rank=0, local_rank=0, distributed_init_method="env://")
            initialize_model_parallel()
        try:
            yield
        finally:
            if started_group:
                cleanup_dist_env_and_memory()


def _diffusion_config(backend: str | None, attention_config_file: Path | None):
    """A minimal current-diffusion-config carrying one attention selection.

    ``attention_config_file`` is an arm from ``arms/`` -- the same JSON a
    server is given as ``--diffusion-attention-config``, so a block profiled
    here used the per-role selection production would use, masks and all.
    ``backend`` is the blunter form, one backend for every role, which is what
    a single-kernel comparison wants. Neither set leaves the platform default
    in place (CUDNN_ATTN on sm_120).
    """
    import json as json_module

    from vllm_omni.diffusion.data import AttentionConfig, AttentionSpec, build_attention_config

    if attention_config_file and backend:
        raise ValueError("pass --attention-config or --backend, not both: they would disagree per role")
    if attention_config_file:
        attention_config = build_attention_config(json_module.loads(attention_config_file.read_text()))
    elif backend:
        attention_config = AttentionConfig(default=AttentionSpec(backend=backend))
    else:
        attention_config = AttentionConfig()
    return SimpleNamespace(
        diffusion_attention_config=attention_config,
        parallel_config=SimpleNamespace(ring_degree=1, allgather_degree=1),
    )


def build_target(
    target: str,
    cfg,
    backend: str | None,
    device: torch.device,
    dtype: torch.dtype,
    attention_config_file: Path | None = None,
):
    """One block (or one attention layer) at ``cfg``'s dimensions.

    Weights are random: ``torch.nn.Module`` initialization on the meta-free
    path already fills them, except ``Kandinsky6Modulation``, which zeroes
    its output layer by design. Zeroed modulation would make every AdaLN
    scale exactly 1 and every gate exactly 0 -- numerically degenerate and,
    worse, it would let the residual stream stay constant across blocks.
    The time does not depend on the values, but a degenerate stream can
    denormal-stall, so the modulation weights are re-randomized here.
    """
    from vllm_omni.diffusion.config import set_current_diffusion_config
    from vllm_omni.diffusion.models.kandinsky6.kandinsky6_transformer import (
        Kandinsky6Attention,
        Kandinsky6FusedTransformerDecoderBlock,
        Kandinsky6TransformerDecoderBlock,
    )

    head_dim = sum(cfg.axes_dims)
    head_dim_a = sum(cfg.axes_dims_a)
    with set_current_diffusion_config(_diffusion_config(backend, attention_config_file)):
        if target == "fused":
            module = Kandinsky6FusedTransformerDecoderBlock(
                cfg.model_dim,
                cfg.time_dim,
                cfg.ff_dim,
                head_dim,
                cfg.model_dim_a,
                cfg.time_dim,
                cfg.ff_dim_a,
                head_dim_a,
                text_token_padding=cfg.text_token_padding,
                ca_rope=cfg.ca_rope,
                cross_gates=cfg.cross_gates,
                fix_modulation=cfg.fix_modulation,
                prefix="visual_transformer_blocks.0",
            )
        elif target == "decoder":
            module = Kandinsky6TransformerDecoderBlock(
                cfg.model_dim,
                cfg.time_dim,
                cfg.ff_dim,
                head_dim,
                text_token_padding=cfg.text_token_padding,
                self_sequence_parallel=True,
                prefix="visual_transformer_blocks.0",
            )
        elif target == "attn":
            module = Kandinsky6Attention(
                cfg.model_dim,
                head_dim,
                visual=True,
                sequence_parallel=True,
                role="kandinsky6.visual_self",
                role_category="self",
                prefix="visual_transformer_blocks.0.self_attention",
            )
        else:  # pragma: no cover - argparse restricts the choices
            raise ValueError(f"unknown target {target!r}")

    for name, param in module.named_parameters():
        if "modulation" in name:
            torch.nn.init.normal_(param, std=0.02)
    return module.to(device=device, dtype=dtype).eval()


def maybe_compile(module, mode: str | None):
    """``torch.compile`` the block, or return it untouched.

    The fusion hypothesis this switch measures: the port does its AdaLN
    modulation, RoPE and residual gates in fp32 on the whole residual stream
    (``apply_scale_shift_norm``, ``apply_rotary``, ``apply_gate_sum`` all
    upcast). At W1's 50,220 x 4096 one such upcast is a 823 MB fp32 tensor,
    and ``apply_rotary``'s broadcast intermediate -- ``(N, H, D/2, 2, 2)``
    before its ``sum(-1)`` -- is 1.65 GB. Eager mode fuses none of it, so each
    of those is a full round trip to HBM. Inductor should fuse the chain into
    the norm and the gate.

    ``fullgraph=False`` deliberately: the attention backends call into
    extensions Dynamo cannot trace (SageAttention's `_qattn_sm89`, cuDNN's
    fused MHA), so a full graph would either fail or silently fall back to a
    slower traceable path and measure the wrong thing. Graph breaks at the
    attention calls are the intended shape here -- the elementwise chain
    between them is what is being fused.
    """
    if not mode:
        return module
    return torch.compile(module, mode=mode, fullgraph=False, dynamic=False)


def make_inputs(target: str, cfg, shapes: Shapes, device: torch.device, dtype: torch.dtype) -> dict:
    """Activations of the right shape for one forward of ``target``.

    The visual stream is already flattened and batched to ``(1, N, D)``, as
    ``Kandinsky6Transformer3DModel._embed_visual`` hands it to the blocks,
    and the RoPE table is the post-``fractal_flatten`` ``(N, 1, C, 2, 2)``
    fp32 form. Text and audio stay unbatched, which is what the blocks'
    cross-attention expects.
    """
    from vllm_omni.diffusion.models.kandinsky6.kandinsky6_transformer import RoPE1D, RoPE3D

    head_dim_a = sum(cfg.axes_dims_a)
    n_vis = shapes.visual_tokens

    def randn(*shape, d=dtype):
        return torch.randn(*shape, device=device, dtype=d)

    # The RoPE tables are built by the port's own modules, not drawn from
    # `randn`. This is not tidiness. A RoPE table's 2x2 blocks are
    # [[cos, -sin], [sin, cos]], so `apply_rotary` is a rotation and preserves
    # the per-head norm that `query_norm`/`key_norm` just set. Random entries
    # make it an arbitrary linear map that stretches some rows far more than
    # others, and SageAttention's per-block INT8 scale is set by the largest
    # entry in a block -- with a random table the visual self-attention
    # returns NaN at W1, which is how this was found. The dense bf16 backends
    # stay finite, so a random table looks harmless until a quantizing kernel
    # is measured.
    vis_rope_table = RoPE3D(cfg.axes_dims).to(device)(
        shape=(shapes.latent_frames, shapes.latent_h, shapes.latent_w),
        pos=[
            torch.arange(shapes.latent_frames, device=device),
            torch.arange(shapes.latent_h, device=device),
            torch.arange(shapes.latent_w, device=device),
        ],
    )
    # `_embed_visual` flattens (T, H, W, ...) to (N, ...) before the blocks.
    vis_rope = vis_rope_table.flatten(0, 2)
    aud_rope = RoPE1D(head_dim_a, max_pos=max(shapes.audio_len, 1)).to(device)(
        torch.arange(shapes.audio_len, device=device)
    )
    vis = randn(1, n_vis, cfg.model_dim)
    time_v = randn(1, cfg.time_dim)

    if target == "attn":
        return dict(hidden_states=vis, rotary_emb=vis_rope)
    if target == "decoder":
        return dict(
            vis=vis,
            text=randn(shapes.text_len, cfg.model_dim),
            time_embed=time_v,
            rope=vis_rope,
            sparse_params=None,
        )
    return dict(
        vis=vis,
        aud=randn(1, shapes.audio_len, cfg.model_dim_a),
        text_v=randn(shapes.text_len, cfg.model_dim),
        text_a=randn(shapes.text_len, cfg.model_dim_a),
        time_embed=(time_v, randn(1, cfg.time_dim)),
        vis_rope=vis_rope,
        aud_rope=aud_rope,
        sparse_params=None,
    )


def time_forward(module, inputs: dict, repeats: int, warmups: int) -> Timing:
    """Wall time of one forward, from CUDA events, after ``warmups`` runs."""
    timing = Timing(repeats=repeats)
    with torch.inference_mode():
        for _ in range(warmups):
            module(**inputs)
        torch.accelerator.synchronize()
        for _ in range(repeats):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            module(**inputs)
            end.record()
            torch.accelerator.synchronize()
            timing.ms.append(start.elapsed_time(end))
    return timing


def parse_arm_spec(spec: str) -> tuple[str, Path | None, str]:
    """``LABEL=ATTENTION@MODE`` -> ``(label, attention_path_or_None, mode)``.

    ``ATTENTION`` is ``default`` (the platform default, ``None`` here) or a
    path to an arms/ file. The separator is ``@`` and not ``:`` or ``/``
    because the attention value is a path, which contains ``/`` and on some
    hosts a drive-style ``:``.
    """
    label, sep, rest = spec.partition("=")
    if not sep or not label:
        raise ValueError(f"{spec!r} is not LABEL=ATTENTION@MODE: no LABEL= part")
    attention, sep, mode = rest.partition("@")
    if not sep or not attention or not mode:
        raise ValueError(f"{spec!r} is not LABEL=ATTENTION@MODE: no ATTENTION@MODE part")
    return label, (None if attention == "default" else Path(attention)), mode


def _flatten_outputs(out) -> list:
    """A block returns ``(vis, aud)``; an attention layer returns one tensor."""
    if isinstance(out, torch.Tensor):
        return [out]
    return [t for t in out if isinstance(t, torch.Tensor)]


def _parameter_storages(module) -> frozenset[int]:
    """Every parameter's storage pointer, as an order- and name-free set."""
    return frozenset(tensor.data_ptr() for tensor in module.parameters())


def output_deltas(arms: dict, inputs: dict) -> dict:
    """Each arm's output against the first arm's, on the same weights.

    A faster arm is only adoptable if its numbers are close, and for a
    compiled arm "close" is not free: Inductor reorders reductions, and
    ``max-autotune`` swaps cuBLAS GEMMs for Triton templates whose accumulate
    order differs. Both arms here share the *same* module weights only when
    the caller built them that way; when they do not, this comparison is
    meaningless, so it checks and says so rather than reporting a number.

    Reported per output tensor: relative L2 and max absolute difference. The
    control's own RMS is included so a reader can see what the absolute
    numbers are relative to.
    """
    labels = list(arms)
    control_module = arms[labels[0]]
    # Compare storage pointers, not names: `torch.compile` returns an
    # OptimizedModule whose `named_parameters()` prefixes every name with
    # `_orig_mod.`, so a name comparison calls a compiled arm "not the same
    # weights" even when it wraps exactly the control module.
    control_storages = _parameter_storages(control_module)

    with torch.inference_mode():
        reference = [t.float().clone() for t in _flatten_outputs(control_module(**inputs))]

    result: dict[str, dict] = {}
    for label in labels[1:]:
        module = arms[label]
        shared = _parameter_storages(module) == control_storages
        if not shared:
            result[label] = {"comparable": False, "why": "this arm does not share the control's weights"}
            continue
        with torch.inference_mode():
            candidate = _flatten_outputs(module(**inputs))
        rows = []
        for index, (got, want) in enumerate(zip(candidate, reference)):
            diff = got.float() - want
            rows.append({
                "output": index,
                "rel_l2": round(float(torch.linalg.vector_norm(diff) / torch.linalg.vector_norm(want)), 8),
                "max_abs": round(float(diff.abs().max()), 8),
                "control_rms": round(float(want.square().mean().sqrt()), 6),
            })
        result[label] = {"comparable": True, "outputs": rows}
    return result


def compare_arms(arms: dict, inputs: dict, repeats: int, warmups: int, rounds: int) -> dict:
    """Time several modules on the same inputs, in ABBA order, in one process.

    Each round visits the arms forwards then backwards, so a clock or
    co-tenant drift during the run hits every arm about equally instead of
    favouring whichever ran first. Everything the arms could differ by except
    the change itself -- the inputs, the weights' shapes, the session, the
    allocator state -- is held fixed, which a comparison across two processes
    cannot promise.
    """
    labels = list(arms)
    order = []
    for _ in range(rounds):
        order.extend(labels)
        order.extend(reversed(labels))

    samples: dict[str, list[float]] = {label: [] for label in labels}
    for label in order:
        samples[label].extend(time_forward(arms[label], inputs, repeats, warmups).ms)

    result = {}
    for label, values in samples.items():
        median = statistics.median(values)
        result[label] = {
            "median_ms": round(median, 3),
            "min_ms": round(min(values), 3),
            "max_ms": round(max(values), 3),
            "spread_pct": round(100.0 * (max(values) - min(values)) / median, 2),
            "samples": len(values),
        }
    control = result[labels[0]]["median_ms"]
    for label in labels[1:]:
        candidate = result[label]["median_ms"]
        result[label]["delta_pct_vs_" + labels[0]] = round(100.0 * (candidate - control) / control, 2)
        # A delta inside the control's own min-max spread is null, per the
        # study's measurement rule. Say so here rather than leave a reader to
        # compare two columns by eye.
        control_spread = result[labels[0]]["max_ms"] - result[labels[0]]["min_ms"]
        result[label]["inside_control_spread"] = abs(candidate - control) <= control_spread
    return result


def profile_forward(module, inputs: dict, iters: int) -> dict:
    """Device time per kernel category, and the launch gap, over ``iters``.

    The gap is the wall time the GPU spent in no kernel at all: with one
    block in flight and nothing to overlap, it is the launch bubble that
    CUDA graphs or fusion would remove.
    """
    from torch.profiler import ProfilerActivity, profile

    with torch.inference_mode():
        module(**inputs)  # the profiler must not see first-call allocation
        torch.accelerator.synchronize()
        wall_start = time.perf_counter()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for _ in range(iters):
                module(**inputs)
            torch.accelerator.synchronize()
        wall_ms = 1e3 * (time.perf_counter() - wall_start)

    kernels: list[tuple[str, float]] = []
    for event in prof.key_averages():
        # `self_device_time_total` is in microseconds and counts only the
        # event's own device time, so a parent op never double-counts its
        # children. Non-kernel events have none.
        if event.self_device_time_total > 0 and event.device_type.name != "CPU":
            kernels.append((event.key, float(event.self_device_time_total)))
    if not kernels:  # a build whose profiler sees no device events
        raise RuntimeError("the profiler recorded no CUDA kernels; check the torch build")

    totals = split_by_category(kernels)
    device_us = sum(totals.values())
    top = sorted(kernels, key=lambda kv: -kv[1])[:20]
    unnamed = [(n, us) for n, us in kernels if classify(n) == "unclassified"]
    unnamed.sort(key=lambda kv: -kv[1])
    return {
        "iters": iters,
        "device_ms_per_forward": round(device_us / 1e3 / iters, 3),
        "wall_ms_per_forward": round(wall_ms / iters, 3),
        "launch_gap_ms_per_forward": round((wall_ms - device_us / 1e3) / iters, 3),
        "categories_ms_per_forward": {k: round(v / 1e3 / iters, 3) for k, v in totals.items()},
        "categories_share_pct": {k: round(100.0 * v / device_us, 2) for k, v in totals.items()},
        "top_kernels": [{"name": n, "ms_per_forward": round(us / 1e3 / iters, 3)} for n, us in top],
        "unclassified_kernels": [
            {"name": n, "ms_per_forward": round(us / 1e3 / iters, 3)} for n, us in unnamed[:10]
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", choices=sorted(CONFIGS), default="pro", help="which checkpoint's dimensions")
    parser.add_argument("--geometry", choices=sorted(GEOMETRIES), default="w1", help="which request geometry")
    parser.add_argument(
        "--target",
        choices=("fused", "decoder", "attn"),
        default="fused",
        help="fused = the T2VA block (W1's backbone), decoder = video only, attn = visual self-attention alone",
    )
    parser.add_argument(
        "--backend", default=None, help="diffusion attention backend, e.g. CUDNN_ATTN (default: platform)"
    )
    parser.add_argument(
        "--attention-config",
        type=Path,
        default=None,
        help="an arm file from arms/ (the same JSON a server takes as --diffusion-attention-config), "
        "for a per-role selection instead of one backend everywhere",
    )
    parser.add_argument(
        "--compile",
        dest="compile_modes",
        action="append",
        default=None,
        choices=("default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"),
        help="torch.compile the block in this mode; omitted runs eager. Repeatable, which only makes "
        "sense with --compare-compile. Compilation happens inside the warm-ups, so --warmups must be "
        "at least 1 (it is 2 by default)",
    )
    parser.add_argument(
        "--compare-compile",
        action="store_true",
        help="time eager and every --compile mode in ABBA order in one process, on the same inputs, "
        "and report each delta against eager. The honest way to compare them: a ratio taken across "
        "two processes is a ratio across two clock states",
    )
    parser.add_argument(
        "--check-numerics",
        action="store_true",
        help="with --compare-arm, also compare each arm's output tensors against the control's. "
        "Only meaningful where the arms share weights, which they do when they differ only in "
        "compile mode; the check says so rather than reporting a number when they do not",
    )
    parser.add_argument(
        "--compare-arm",
        action="append",
        default=None,
        metavar="LABEL=ATTENTION@MODE",
        help="an arm for a full comparison, repeatable. ATTENTION is an arms/ file or `default` for "
        "the platform default; MODE is `eager` or a torch.compile mode. The first arm given is the "
        "control every other is reported against. Separator is `@` because a path contains `/`. "
        "Use this rather than --compare-compile when the arms differ in more than the compile mode: "
        "vLLM-Omni compiles the DiT's repeated blocks by default (`enforce_eager=False`, "
        "`diffusion_compile_granularity=regional`), so `default@default` -- not `default@eager` -- "
        "is what upstream actually serves, and a baseline has to say which one it is",
    )
    parser.add_argument("--text-len", type=int, default=None, help="override the prompt's embedding count")
    parser.add_argument("--repeats", type=int, default=5, help="timed forwards")
    parser.add_argument("--rounds", type=int, default=2, help="ABBA rounds for --compare-compile")
    parser.add_argument("--warmups", type=int, default=2, help="untimed forwards first")
    parser.add_argument("--profile-iters", type=int, default=3, help="forwards inside the profiler; 0 skips it")
    parser.add_argument("--json", type=Path, default=None, help="write the full result here")
    parser.add_argument("--list-shapes", action="store_true", help="print the derived token counts and exit")
    parser.add_argument(
        "--no-locks",
        action="store_true",
        help="run without the GPU locks and without the foreign-process check. Correctness checks only: "
        "a timing taken without the locks is not a measurement and must never be recorded as one.",
    )
    parser.add_argument("--allow-foreign-gpu", action="store_true", help="run even with another job on the GPU")
    args = parser.parse_args()

    cfg = CONFIGS[args.config]
    geometry = dict(GEOMETRIES[args.geometry])
    if args.text_len is not None:
        geometry["text_len"] = args.text_len
    shapes = Shapes(**geometry)

    if args.list_shapes:
        print(json.dumps({"config": args.config, "geometry": args.geometry, "shapes": asdict(shapes)}, indent=2))
        return 0

    with contextlib.ExitStack() as stack:
        locks = None if args.no_locks else stack.enter_context(GpuLocks())
        # `run_when_free.py` holds the locks on this process's behalf and
        # says so here, so a `--no-locks` run under it is still recorded
        # with the locks it actually ran under.
        locks_held = locks.held if locks else os.environ.get("K6_LOCKS_HELD_BY")
        foreign = foreign_gpu_procs()
        if foreign and not (args.allow_foreign_gpu or args.no_locks):
            print("foreign process(es) on the GPU; not timing:", file=sys.stderr)
            for proc in foreign:
                print(f"  {proc}", file=sys.stderr)
            return 3

        device = torch.device("cuda")
        dtype = torch.bfloat16
        stack.enter_context(single_process_parallel())
        module = maybe_compile(
            build_target(args.target, cfg, args.backend, device, dtype, args.attention_config),
            (args.compile_modes or [None])[0],
        )
        inputs = make_inputs(args.target, cfg, shapes, device, dtype)
        if args.compile_modes and args.warmups < 1:
            parser.error("--compile needs --warmups >= 1, or the first timed forward pays for compilation")

        torch.accelerator.reset_peak_memory_stats()
        if args.compare_arm:
            arms = {}
            modules_by_attention: dict[str, object] = {}
            for spec in args.compare_arm:
                try:
                    label, attention_path, mode = parse_arm_spec(spec)
                except ValueError as exc:
                    parser.error(f"--compare-arm {exc}")
                if attention_path is not None and not attention_path.is_file():
                    parser.error(f"--compare-arm {spec!r}: no such attention config {attention_path}")
                # One module per attention config, shared between that
                # config's eager and compiled arms, so `--check-numerics` can
                # compare their outputs rather than comparing two random
                # initializations.
                key = str(attention_path)
                if key not in modules_by_attention:
                    modules_by_attention[key] = build_target(
                        args.target, cfg, None, device, dtype, attention_path
                    )
                built = modules_by_attention[key]
                arms[label] = built if mode == "eager" else maybe_compile(built, mode)
            numerics = output_deltas(arms, inputs) if args.check_numerics else None
            compare = compare_arms(
                arms, inputs, repeats=args.repeats, warmups=max(args.warmups, 1), rounds=args.rounds
            )
        elif args.compare_compile:
            # A fresh module per arm: torch.compile mutates what it wraps
            # enough that reusing one module across modes would measure the
            # last mode's cached artifacts. The weights are random anyway, so
            # separate modules cost nothing but memory.
            arms = {"eager": build_target(args.target, cfg, args.backend, device, dtype, args.attention_config)}
            for mode in args.compile_modes or ["default"]:
                arms[f"compile={mode}"] = maybe_compile(
                    build_target(args.target, cfg, args.backend, device, dtype, args.attention_config), mode
                )
            numerics = None
            compare = compare_arms(
                arms, inputs, repeats=args.repeats, warmups=max(args.warmups, 1), rounds=args.rounds
            )
        else:
            compare = None
            numerics = None
        timing = time_forward(module, inputs, args.repeats, args.warmups)
        peak_gib = torch.accelerator.max_memory_allocated() / 2**30
        profile_result = profile_forward(module, inputs, args.profile_iters) if args.profile_iters else None

        result = {
            "config": args.config,
            "geometry": args.geometry,
            "target": args.target,
            "backend": args.backend or "platform-default",
            "attention_config_file": str(args.attention_config) if args.attention_config else None,
            "compile_modes": args.compile_modes,
            "shapes": asdict(shapes),
            "dtype": str(dtype),
            "torch": torch.__version__,
            "device": torch.cuda.get_device_name(0),
            "locks_held": locks_held,
            "foreign_gpu_procs": [str(p) for p in foreign],
            "peak_allocated_gib": round(peak_gib, 3),
            "timing": timing.summary(),
            "compare_compile": compare,
            "output_deltas": numerics,
            "profile": profile_result,
        }

    modes = "/compile=" + ",".join(args.compile_modes) if args.compile_modes else ""
    print(f"{args.config}/{args.target}/{args.backend or 'platform-default'}{modes} at {args.geometry}: "
          f"{shapes.visual_tokens} visual tokens, {shapes.audio_len} audio, {shapes.text_len} text")
    s = timing.summary()
    print(f"  one block: {s['median_ms']:.2f} ms median ({s['min_ms']:.2f}-{s['max_ms']:.2f}, "
          f"spread {s['spread_pct']:.2f}%), peak {peak_gib:.2f} GiB allocated")
    if args.target != "attn":
        whole_backbone_s = s["median_ms"] * cfg.num_visual_blocks / 1e3
        print(f"  x{cfg.num_visual_blocks} visual blocks: {whole_backbone_s:.2f} s a forward")
    if compare:
        control = next(iter(compare))
        print(f"  ABBA in one process, against {control}:")
        for label, row in compare.items():
            delta = row.get(f"delta_pct_vs_{control}")
            tail = "" if delta is None else (
                f"  {delta:+.2f}%" + ("  (inside the control's spread: null)" if row["inside_control_spread"] else "")
            )
            print(f"    {label:<22} {row['median_ms']:8.2f} ms  "
                  f"({row['min_ms']:.2f}-{row['max_ms']:.2f}, spread {row['spread_pct']:.2f}%, "
                  f"n={row['samples']}){tail}")
    if numerics:
        print("  output vs the control, same weights:")
        for label, row in numerics.items():
            if not row["comparable"]:
                print(f"    {label:<22} not comparable: {row['why']}")
                continue
            for out in row["outputs"]:
                print(f"    {label:<22} output {out['output']}: rel L2 {out['rel_l2']:.3e}, "
                      f"max abs {out['max_abs']:.3e} (control RMS {out['control_rms']:.4f})")
    if profile_result:
        print(f"  device {profile_result['device_ms_per_forward']:.2f} ms, "
              f"launch gap {profile_result['launch_gap_ms_per_forward']:.2f} ms")
        for category in CATEGORIES:
            ms = profile_result["categories_ms_per_forward"][category]
            if ms:
                print(f"    {category:<13} {ms:8.2f} ms  {profile_result['categories_share_pct'][category]:5.1f}%")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2) + "\n")
        print(f"  wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
