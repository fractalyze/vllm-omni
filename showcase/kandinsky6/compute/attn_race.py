# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Race vLLM-Omni's diffusion attention backends at Kandinsky 6 Pro shapes.

W1 of ``showcase/kandinsky6/PLAN.md`` puts 50,220 visual tokens through 60
blocks of 32 heads of 128, which by shape makes self-attention about two
thirds of the DiT's work. Which attention kernel runs it is therefore the
largest single compute decision on this GPU, and on sm_120 it is open:
FlashAttention-3 is Hopper-only, the platform default here is CUDNN_ATTN,
and the FP8/FP4 attention kernels (SageAttention 2++ and 3) are the ones
the recipe DB credits with 2.3x end-to-end on the nearest analogue.

Each arm runs through ``vllm_omni.diffusion.attention.layer.Attention`` with
one backend pinned per role, so an arm that wins here is selected in
production by the same ``AttentionConfig`` -- no separate code path.

**Accuracy** is measured against attention computed in fp32 on the same
q/k/v, chunked over query blocks so no arm is compared against a reference
that itself had to approximate (a 50,220 x 50,220 fp32 score matrix is
10 TB; see ``reference_attention_fp32``).

**Activations are synthetic, and that is defensible here** -- Pro does not
fit on this GPU, so there is no real Pro forward to capture from. They are
not plain ``randn``: the port pins the distribution a kernel sees.
``Kandinsky6Attention`` applies ``query_norm``/``key_norm`` -- RMSNorm over
``head_dim``, per head, *before* RoPE -- so every q and k row reaching the
kernel has unit RMS by construction, and RoPE is a rotation, which preserves
it. ``make_activations`` reproduces exactly that, with the port's own RoPE
module. ``value`` is the one tensor whose scale is a guess (it is a raw
linear output), and it is drawn at unit RMS.

What this cannot capture is the *structure* of real attention scores: a real
video DiT's q/k are correlated across neighbouring tokens, so the real
softmax is peakier than a random one. A peakier softmax is kinder to a
quantizing kernel, so a SageAttention error measured here is an upper bound
on the real one -- the right direction for a gate, but it means the numbers
rank arms and do not replace the end-to-end quality gate.

    python attn_race.py --roles visual_self --arms CUDNN_ATTN,TORCH_SDPA
    python attn_race.py --all-roles --json race.json

Timed runs hold every GPU lock on the host and refuse to run with a foreign
process on the GPU.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
from block_profile import PRO, W1, Shapes, single_process_parallel  # noqa: E402
from gpulock import GpuLocks, foreign_gpu_procs  # noqa: E402


@dataclass(frozen=True)
class Role:
    """One of Kandinsky 6's attention call sites, at W1's token counts.

    ``name`` is the suffix of the port's role string (the full role is
    ``kandinsky6.<name>``), ``q`` and ``kv`` are sequence lengths, and
    ``heads``/``head_dim`` are that call site's head geometry -- the audio
    branch is 16 heads of 128, not 32.
    """

    name: str
    q: int
    kv: int
    heads: int
    head_dim: int
    calls_per_block: int = 1

    @property
    def role_string(self) -> str:
        return f"kandinsky6.{self.name}"

    @property
    def tflop(self) -> float:
        """FLOPs of one call: QK^T plus AV, both 2 flops per MAC."""
        return 4.0 * self.q * self.kv * self.heads * self.head_dim / 1e12


def w1_roles(shapes: Shapes, cfg=PRO) -> list[Role]:
    """Every attention call in one fused T2VA block at ``shapes``."""
    head_dim = sum(cfg.axes_dims)
    head_dim_a = sum(cfg.axes_dims_a)
    heads_v = cfg.model_dim // head_dim
    heads_a = cfg.model_dim_a // head_dim_a
    return [
        Role("visual_self", shapes.visual_tokens, shapes.visual_tokens, heads_v, head_dim),
        Role("text_cross", shapes.visual_tokens, shapes.text_len, heads_v, head_dim),
        Role("video_audio_cross", shapes.visual_tokens, shapes.audio_len, heads_v, head_dim),
        Role("audio_video_cross", shapes.audio_len, shapes.visual_tokens, heads_a, head_dim_a),
        Role("audio_self", shapes.audio_len, shapes.audio_len, heads_a, head_dim_a),
    ]


# Arms worth racing on sm_120. CUDNN_ATTN is the platform default and so the
# control. FLASH_ATTN resolves to FlashAttention-4 (flash_attn.cute) on
# Blackwell -- plain FA2 wheels only ship Ampere/Ada/Hopper kernels and die
# with "no kernel image" on the first forward, which the platform's own
# `has_flash_attn_4()` check encodes.
DEFAULT_ARMS = ("CUDNN_ATTN", "TORCH_SDPA", "FLASH_ATTN", "FLASHINFER_ATTN", "SAGE_ATTN", "SAGE_ATTN_3")


def make_activations(role: Role, device: torch.device, dtype: torch.dtype, seed: int, value_rms: float = 1.0):
    """q/k/v as the port hands them to the kernel: ``(1, S, H, D)``.

    q and k are RMS-normalized per head and then rotated by a real RoPE
    table, which is exactly what ``Kandinsky6Attention.forward`` does before
    calling the backend. Doing it here rather than drawing from ``randn``
    matters for a quantizing kernel: SageAttention's per-block INT8 scales
    are set by the row norms, so a reference with the wrong row norms would
    flatter or punish it for the wrong reason.
    """
    from vllm_omni.diffusion.models.kandinsky6.kandinsky6_transformer import RoPE1D, apply_rotary

    generator = torch.Generator(device=device).manual_seed(seed)

    def randn(*shape, d=dtype):
        return torch.randn(*shape, device=device, dtype=d, generator=generator)

    # `no_grad`: the RMSNorm module and RoPE table carry parameters, so
    # without it every activation arrives with a grad history that no caller
    # wants and that `reference_attention_fp32` would have to strip.
    with torch.no_grad():
        norm = torch.nn.RMSNorm(role.head_dim, device=device, dtype=dtype)
        query = norm(randn(1, role.q, role.heads, role.head_dim))
        key = norm(randn(1, role.kv, role.heads, role.head_dim))
        value = randn(1, role.kv, role.heads, role.head_dim) * value_rms

    # The port's own RoPE module, so the table is built by the code under
    # test. RoPE1D rather than the visual stream's RoPE3D: both rotate by
    # angles drawn from the same `get_freqs` family, and what a quantizing
    # kernel sees of RoPE is that its input is a rotation of a unit-RMS
    # vector -- which axis the angle came from does not reach the kernel.
        max_pos = max(role.q, role.kv)
        rope = RoPE1D(role.head_dim, max_pos=max_pos).to(device)
        positions = torch.arange(max_pos, device=device)
        table = rope(positions)  # (max_pos, 1, head_dim // 2, 2, 2), fp32
        query = apply_rotary(query, table[: role.q]).to(dtype)
        key = apply_rotary(key, table[: role.kv]).to(dtype)
        del table, rope
    return query, key, value


def reference_attention_fp32(query, key, value, score_budget_gib: float = 0.5) -> torch.Tensor:
    """Softmax attention in fp32, one head at a time, chunked over queries.

    The full score matrix is never materialized: at W1's 50,220 queries and
    keys it would be 10 TB in fp32, and even one head's slice is 10 GB. The
    loop is therefore over heads *and* over query blocks, with the block
    sized so the live ``(q_block, kv)`` fp32 score tile stays under
    ``score_budget_gib`` -- 1,340 queries against 50,220 keys at the 0.5 GiB
    default. Keeping the tile small also matters on this host, where the GPU
    is shared.

    TF32 is disabled explicitly. Leaving it on would give the "fp32"
    reference 10 bits of mantissa, barely better than bf16's 8, and every
    arm's error would be measured against the wrong number.
    """
    tf32_was = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        # (1, S, H, D) -> (H, S, D).
        q = query[0].transpose(0, 1).float()
        k = key[0].transpose(0, 1).float()
        v = value[0].transpose(0, 1).float()
        heads, q_len, head_dim = q.shape
        kv_len = k.shape[1]
        scale = 1.0 / (head_dim**0.5)
        q_block = max(1, min(q_len, int(score_budget_gib * 2**30) // (kv_len * 4)))
        out = torch.empty_like(q)
        for head in range(heads):
            for start in range(0, q_len, q_block):
                stop = min(start + q_block, q_len)
                scores = torch.matmul(q[head, start:stop], k[head].transpose(-2, -1)) * scale
                out[head, start:stop] = torch.matmul(torch.softmax(scores, dim=-1), v[head])
                del scores
        return out.transpose(0, 1).unsqueeze(0)  # back to (1, S, H, D)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf32_was


def accuracy(candidate: torch.Tensor, reference: torch.Tensor) -> dict:
    """Error of one arm's output against the fp32 reference.

    ``rel_l2`` is the number to compare arms on: it is scale-free and, unlike
    a max, does not hinge on one token. ``cosine`` catches an arm that got
    the direction right and the magnitude wrong (a missing or doubled scale).
    """
    cand = candidate.float().flatten()
    ref = reference.float().flatten()
    diff = cand - ref
    return {
        "rel_l2": round(float(torch.linalg.vector_norm(diff) / torch.linalg.vector_norm(ref)), 6),
        "max_abs": round(float(diff.abs().max()), 6),
        "mean_abs": round(float(diff.abs().mean()), 6),
        "cosine": round(float(torch.nn.functional.cosine_similarity(cand, ref, dim=0)), 8),
        "ref_rms": round(float(ref.square().mean().sqrt()), 6),
    }


# SageAttention accuracy variants, as callables taking (q, k, v, scale) in
# NHD layout. The registered SAGE_ATTN backend calls the top-level `sageattn`
# dispatcher, which on sm_120 picks `qk_int8_pv_fp8_cuda` with
# `pv_accum_dtype="fp32+fp16"` and `qk_quant_gran="per_warp"` -- the fastest
# variant, and the one the quality gate rejected as lossy. These are the knobs
# that trade it back:
#
#   qk_quant_gran  per_warp -> per_thread: finer INT8 scales for Q and K, so
#                  one outlier row poisons a smaller group.
#   PV path        fp8 -> fp16 with fp32 accumulation: the value-times-
#                  probability product stops being FP8.
#   smooth_k       subtract K's per-channel mean before quantizing, which is
#                  where most of INT8 attention's error comes from on a
#                  channel with a large offset.
#
# Raced directly rather than through the backend because the backend exposes
# none of them yet: measuring first says whether plumbing them is worth it.
def sage_variants() -> dict:
    from sageattention.core import sageattn_qk_int8_pv_fp8_cuda, sageattn_qk_int8_pv_fp16_cuda

    def fp8(gran):
        def run(q, k, v, scale):
            return sageattn_qk_int8_pv_fp8_cuda(
                q, k, v, tensor_layout="NHD", is_causal=False, qk_quant_gran=gran,
                sm_scale=scale, pv_accum_dtype="fp32+fp16", smooth_k=True,
            )

        return run

    def fp16(gran, smooth_v):
        def run(q, k, v, scale):
            return sageattn_qk_int8_pv_fp16_cuda(
                q, k, v, tensor_layout="NHD", is_causal=False, qk_quant_gran=gran,
                sm_scale=scale, pv_accum_dtype="fp32", smooth_k=True, smooth_v=smooth_v,
            )

        return run

    return {
        "sage_fp8_perwarp": fp8("per_warp"),
        "sage_fp8_perthread": fp8("per_thread"),
        "sage_fp16fp32_perwarp": fp16("per_warp", False),
        "sage_fp16fp32_perthread": fp16("per_thread", False),
        "sage_fp16fp32_perthread_smoothv": fp16("per_thread", True),
    }


def build_attention(role: Role, backend: str):
    """An ``Attention`` layer for ``role`` with ``backend`` pinned."""
    from vllm_omni.diffusion.attention.layer import Attention
    from vllm_omni.diffusion.config import set_current_diffusion_config
    from vllm_omni.diffusion.data import AttentionConfig, AttentionSpec

    config = SimpleNamespace(
        diffusion_attention_config=AttentionConfig(default=AttentionSpec(backend=backend)),
        parallel_config=SimpleNamespace(ring_degree=1, allgather_degree=1),
    )
    with set_current_diffusion_config(config):
        return Attention(
            num_heads=role.heads,
            head_size=role.head_dim,
            causal=False,
            softmax_scale=1.0 / (role.head_dim**0.5),
            num_kv_heads=role.heads,
            prefix=f"race.{role.name}",
            role=role.role_string,
            role_category="self" if role.name.endswith("self") else "cross",
            scatter_idx=2,
            gather_idx=1,
            skip_sequence_parallel=True,
        )


def time_arm(layer, q, k, v, repeats: int, warmups: int) -> dict:
    events = []
    with torch.inference_mode():
        for _ in range(warmups):
            layer(q, k, v, None)
        torch.accelerator.synchronize()
        for _ in range(repeats):
            start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            layer(q, k, v, None)
            stop.record()
            torch.accelerator.synchronize()
            events.append(start.elapsed_time(stop))
    median = statistics.median(events)
    return {
        "median_ms": round(median, 4),
        "min_ms": round(min(events), 4),
        "max_ms": round(max(events), 4),
        "spread_pct": round(100.0 * (max(events) - min(events)) / median, 2),
        "all_ms": [round(v, 4) for v in events],
    }


def race_role(role: Role, arms: list[str], args, device, dtype) -> dict:
    """Every arm on one role, in ABBA order, against one fp32 reference."""
    q, k, v = make_activations(role, device, dtype, seed=args.seed)

    reference = None
    if not args.no_accuracy:
        reference = reference_attention_fp32(q, k, v, score_budget_gib=args.ref_score_budget_gib)

    layers: dict[str, object] = {}
    errors: dict[str, str] = {}
    for arm in arms:
        try:
            layers[arm] = build_attention(role, arm)
        except Exception as exc:  # a backend whose package or arch check fails
            errors[arm] = f"{type(exc).__name__}: {exc}"

    # Accuracy first: one forward per arm, before any timing, so a kernel
    # that is wrong is reported even if it is fast.
    accuracies: dict[str, dict] = {}
    outputs_checked = []
    for arm, layer in list(layers.items()):
        try:
            with torch.inference_mode():
                out = layer(q, k, v, None)
            if out.shape != q.shape:
                raise RuntimeError(f"output shape {tuple(out.shape)} != query shape {tuple(q.shape)}")
            if reference is not None:
                accuracies[arm] = accuracy(out, reference)
            outputs_checked.append(arm)
            del out
        except Exception as exc:
            errors[arm] = f"{type(exc).__name__}: {exc}"
            layers.pop(arm)
    del reference
    torch.accelerator.empty_cache()

    # ABBA: each round runs the arms forwards then backwards, so a drift in
    # clocks or a co-tenant's ramp hits every arm equally rather than
    # favouring whichever ran first.
    order: list[str] = []
    live = list(layers)
    for _ in range(args.rounds):
        order.extend(live)
        order.extend(reversed(live))
    samples: dict[str, list[float]] = {arm: [] for arm in live}
    for arm in order:
        result = time_arm(layers[arm], q, k, v, repeats=args.repeats, warmups=args.warmups)
        samples[arm].extend(result["all_ms"])

    timings = {}
    for arm, values in samples.items():
        median = statistics.median(values)
        timings[arm] = {
            "median_ms": round(median, 4),
            "min_ms": round(min(values), 4),
            "max_ms": round(max(values), 4),
            "spread_pct": round(100.0 * (max(values) - min(values)) / median, 2),
            "samples": len(values),
            "achieved_tflops": round(role.tflop / (median / 1e3), 1),
            "all_ms": [round(value, 4) for value in values],
        }

    del layers, q, k, v
    torch.accelerator.empty_cache()
    return {
        "role": role.name,
        "q": role.q,
        "kv": role.kv,
        "heads": role.heads,
        "head_dim": role.head_dim,
        "tflop_per_call": round(role.tflop, 3),
        "timings": timings,
        "accuracy": accuracies,
        "errors": errors,
    }


def race_sage_variants(role: Role, args, device, dtype) -> dict:
    """Time and score SageAttention's accuracy variants against fp32.

    The dense control is the registered CUDNN_ATTN backend, so the table has
    the same reference point as the backend race above.
    """
    q, k, v = make_activations(role, device, dtype, seed=args.seed)
    reference = None if args.no_accuracy else reference_attention_fp32(
        q, k, v, score_budget_gib=args.ref_score_budget_gib
    )
    scale = 1.0 / (role.head_dim**0.5)

    arms: dict[str, object] = {}
    errors: dict[str, str] = {}
    try:
        control = build_attention(role, "CUDNN_ATTN")
        arms["cudnn_dense"] = lambda qq, kk, vv, _s: control(qq, kk, vv, None)
    except Exception as exc:
        errors["cudnn_dense"] = f"{type(exc).__name__}: {exc}"
    try:
        arms.update(sage_variants())
    except Exception as exc:
        errors["sage_variants"] = f"{type(exc).__name__}: {exc}"

    accuracies: dict[str, dict] = {}
    for label, run in list(arms.items()):
        try:
            with torch.inference_mode():
                out = run(q, k, v, scale)
            if reference is not None:
                accuracies[label] = accuracy(out, reference)
            del out
        except Exception as exc:
            errors[label] = f"{type(exc).__name__}: {exc}"
            arms.pop(label)
    del reference
    torch.accelerator.empty_cache()

    live = list(arms)
    order: list[str] = []
    for _ in range(args.rounds):
        order.extend(live)
        order.extend(reversed(live))

    samples: dict[str, list[float]] = {label: [] for label in live}
    with torch.inference_mode():
        for label in order:
            run = arms[label]
            for _ in range(args.warmups):
                run(q, k, v, scale)
            torch.accelerator.synchronize()
            for _ in range(args.repeats):
                start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                run(q, k, v, scale)
                stop.record()
                torch.accelerator.synchronize()
                samples[label].append(start.elapsed_time(stop))

    timings = {}
    for label, values in samples.items():
        median = statistics.median(values)
        timings[label] = {
            "median_ms": round(median, 4),
            "min_ms": round(min(values), 4),
            "max_ms": round(max(values), 4),
            "spread_pct": round(100.0 * (max(values) - min(values)) / median, 2),
            "samples": len(values),
            "achieved_tflops": round(role.tflop / (median / 1e3), 1),
        }

    del arms, q, k, v
    torch.accelerator.empty_cache()
    return {"role": role.name, "q": role.q, "kv": role.kv, "heads": role.heads, "head_dim": role.head_dim,
            "tflop_per_call": round(role.tflop, 3), "timings": timings, "accuracy": accuracies, "errors": errors}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arms", default=",".join(DEFAULT_ARMS), help="comma-separated backend names")
    parser.add_argument("--roles", default="visual_self", help="comma-separated role names, or 'all'")
    parser.add_argument("--all-roles", action="store_true", help="same as --roles all")
    parser.add_argument("--text-len", type=int, default=W1["text_len"], help="prompt embedding count")
    parser.add_argument("--q-tokens", type=int, default=None, help="override the visual token count (for a quick pass)")
    parser.add_argument("--repeats", type=int, default=3, help="timed calls per arm per visit")
    parser.add_argument("--rounds", type=int, default=2, help="ABBA rounds; each visits every arm twice")
    parser.add_argument("--warmups", type=int, default=2, help="untimed calls before each visit")
    parser.add_argument("--ref-score-budget-gib", type=float, default=0.5,
                        help="cap on the fp32 reference's live score tile; sets its query block")
    parser.add_argument("--no-accuracy", action="store_true", help="skip the fp32 reference (timing only)")
    parser.add_argument("--seed", type=int, default=20261006)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument(
        "--no-locks",
        action="store_true",
        help="run without the GPU locks and without the foreign-process check. Correctness checks only: "
        "a timing taken without the locks is not a measurement and must never be recorded as one.",
    )
    parser.add_argument(
        "--sage-variants",
        action="store_true",
        help="race SageAttention's accuracy variants (INT8 granularity, FP16-vs-FP8 PV, smooth_v) "
        "against CUDNN_ATTN instead of racing the registered backends",
    )
    parser.add_argument("--allow-foreign-gpu", action="store_true")
    args = parser.parse_args()

    shapes = Shapes(**{**W1, "text_len": args.text_len})
    roles = w1_roles(shapes)
    if args.q_tokens:
        roles = [Role(r.name, args.q_tokens if r.q == shapes.visual_tokens else r.q,
                      args.q_tokens if r.kv == shapes.visual_tokens else r.kv,
                      r.heads, r.head_dim) for r in roles]
    wanted = "all" if args.all_roles else args.roles
    if wanted != "all":
        names = [name.strip() for name in wanted.split(",")]
        unknown = set(names) - {r.name for r in roles}
        if unknown:
            parser.error(f"unknown role(s): {sorted(unknown)}; have {[r.name for r in roles]}")
        roles = [r for r in roles if r.name in names]
    arms = [arm.strip().upper() for arm in args.arms.split(",") if arm.strip()]

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

        device, dtype = torch.device("cuda"), torch.bfloat16
        stack.enter_context(single_process_parallel())

        results: list[dict] = []
        payload = {
            "shapes": {"visual_tokens": shapes.visual_tokens, "audio_len": shapes.audio_len,
                       "text_len": shapes.text_len, "q_tokens_override": args.q_tokens},
            "arms_requested": arms,
            "dtype": str(dtype),
            "torch": torch.__version__,
            "device": torch.cuda.get_device_name(0),
            "locks_held": locks_held,
            "foreign_gpu_procs": [str(p) for p in foreign],
            "abba_rounds": args.rounds,
            "repeats_per_visit": args.repeats,
            "roles_requested": [role.name for role in roles],
            "complete": False,
            "roles": results,
        }

        def save() -> None:
            if args.json:
                args.json.parent.mkdir(parents=True, exist_ok=True)
                args.json.write_text(json.dumps(payload, indent=2) + "\n")

        # Written after every role, not once at the end. This GPU is shared
        # and the gap between a co-tenant's jobs may be shorter than the
        # whole race; a run cut off after the first role should still leave
        # the headline role's numbers on disk, with `complete: false` saying
        # it was cut off. Roles are in descending cost order, so the first
        # one done is `visual_self`, which is 98.8% of a block's attention
        # FLOPs.
        save()
        for role in roles:
            if args.sage_variants:
                results.append(race_sage_variants(role, args, device, dtype))
            else:
                results.append(race_role(role, arms, args, device, dtype))
            save()
        payload["complete"] = True
        save()

    for result in results:
        print(f"\n{result['role']}: q={result['q']} kv={result['kv']} "
              f"{result['heads']}x{result['head_dim']}, {result['tflop_per_call']:.2f} TFLOP a call")
        ranked = sorted(result["timings"].items(), key=lambda kv: kv[1]["median_ms"])
        if ranked:
            best = ranked[0][1]["median_ms"]
            print(f"  {'arm':<18} {'median ms':>10} {'min-max':>16} {'spread':>7} {'TFLOP/s':>9} "
                  f"{'vs best':>8}  {'rel L2':>9} {'cosine':>10}")
            for arm, timing in ranked:
                acc = result["accuracy"].get(arm)
                acc_cols = (f"{acc['rel_l2']:>9.6f} {acc['cosine']:>10.7f}" if acc else f"{'-':>9} {'-':>10}")
                print(f"  {arm:<18} {timing['median_ms']:>10.3f} "
                      f"{timing['min_ms']:>7.3f}-{timing['max_ms']:<8.3f} {timing['spread_pct']:>6.2f}% "
                      f"{timing['achieved_tflops']:>9.1f} {timing['median_ms'] / best:>7.3f}x  {acc_cols}")
        for arm, message in result["errors"].items():
            print(f"  {arm:<18} unavailable: {message}")

    if args.json:
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
