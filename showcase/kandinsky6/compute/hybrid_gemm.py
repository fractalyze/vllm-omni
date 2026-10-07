# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""A hybrid FP16-accumulate GEMM for Kandinsky 6's linears on sm_120.

On consumer Blackwell an FP16-input MMA accumulating in FP16 runs faster than
the same MMA accumulating in FP32. Measured on this RTX 5090 at W1's FF1 shape
(50220x4096x16384): FP32 accumulate 227.7 TFLOP/s, FP16 accumulate **351.3**,
so the ceiling is **1.54x** -- not the 2.00x the spec implies, which matters
because it caps what any amount of tuning can return.

A pure FP16 accumulator is not usable: over K=4096 it reaches rel L2 2.4e-3
against FP64, two decimal orders worse than the current path. The hybrid keeps
the fast instruction and bounds the damage by promoting to an FP32 running sum
every ``BLOCK_K`` elements, so the error depends on BLOCK_K rather than on K.

The comparison that decides whether this is safe is **not** hybrid against
exact. It is hybrid-in-FP16 against **what the served path already does**:
BF16 inputs with an FP32 accumulator. BF16 carries 7 mantissa bits and FP16
carries 10, so converting the operands to FP16 *reduces* input quantisation
error while the FP16 accumulation adds some back. ``test_hybrid_gemm.py``
measures both against an FP64 reference on real checkpoint weights and holds the
hybrid to the BF16 path's own error, which is the bar that matters.

The range risk is the other half. FP16 overflows at 65504 where BF16 reaches
~3.4e38, so a layer whose activations or partial sums approach that must not be
converted. ``fp16_safety`` reports the headroom per tensor and the caller
decides; nothing here silently clamps.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

FP16_MAX = 65504.0

# Per-shape configs, from the sweep in `fp16_accum_peak.py` and `tune_hybrid.py`.
# BLOCK_K is the error dial as well as a speed one: it is how many products
# accumulate in FP16 before being promoted, so a smaller BLOCK_K costs speed and
# buys accuracy. 32 is the default because at W1's shapes it measured within 1%
# of the fastest config while scoring 3.3e-4 against 64's 3.9e-4.
_DEFAULT = dict(BLOCK_M=128, BLOCK_N=128, BLOCK_K=32, GROUP_M=8, num_warps=4, num_stages=2)


@triton.jit
def _hybrid_mm(A, B, Bias, C, M, N, K,
               sam, sak, sbk, sbn, scm, scn,
               BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
               GROUP_M: tl.constexpr, HAS_BIAS: tl.constexpr, OUT_FP16: tl.constexpr):
    """C = A @ B (+ Bias), FP16 accumulate per K-block, FP32 across blocks.

    The bias is folded into the epilogue here rather than left to a separate
    kernel. That is the opposite of what PR #38 did for cuBLAS -- there, asking
    the library to fuse a bias cost a worse tile. Here the tile is ours and does
    not change, so the fusion is free and saves a pass over an M*N output.
    """
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    # Grouped rasterisation: consecutive programs share B tiles, which is worth
    # a few percent at these N.
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    a_ptr = A + rm[:, None] * sam + rk[None, :] * sak
    b_ptr = B + rk[:, None] * sbk + rn[None, :] * sbn

    # Promotion is per K-block and that is not a tunable. Decoupling the two --
    # accumulating several blocks in an FP16 partial before promoting, to cut the
    # BLOCK_M x BLOCK_N conversions -- was tried and is 2-5x SLOWER than cuBLAS at
    # every interval above 1 (measured: 0.18-0.45x). The mutable FP16 partial plus
    # a dynamic branch in the loop body defeats Triton's software pipelining and
    # spills, and the conversions it saves were never the bottleneck.
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        k_mask = rk[None, :] + k < K
        x = tl.load(a_ptr, mask=(rm[:, None] < M) & k_mask, other=0.0)
        y = tl.load(b_ptr, mask=(rk[:, None] + k < K) & (rn[None, :] < N), other=0.0)
        acc += tl.dot(x, y, out_dtype=tl.float16).to(tl.float32)
        a_ptr += BLOCK_K * sak
        b_ptr += BLOCK_K * sbk

    if HAS_BIAS:
        acc += tl.load(Bias + rn, mask=rn < N, other=0.0).to(tl.float32)[None, :]

    out = acc.to(tl.float16) if OUT_FP16 else acc.to(tl.bfloat16)
    c_ptr = C + rm[:, None] * scm + rn[None, :] * scn
    tl.store(c_ptr, out, mask=(rm[:, None] < M) & (rn[None, :] < N))


def fp16_safety(t: torch.Tensor) -> dict:
    """How close a tensor is to FP16's limits, and whether it underflows.

    Reported rather than acted on: the decision to route a layer away from this
    kernel belongs to the caller, which knows what the layer is.
    """
    finite = t[torch.isfinite(t)].abs()
    if finite.numel() == 0:
        return {"absmax": 0.0, "headroom": float("inf"), "subnormal_frac": 0.0}
    absmax = finite.max().item()
    nz = finite[finite > 0]
    # FP16's smallest normal is 2^-14; below it precision degrades to subnormal.
    sub = (nz < 2.0**-14).float().mean().item() if nz.numel() else 0.0
    return {"absmax": absmax,
            "headroom": FP16_MAX / absmax if absmax > 0 else float("inf"),
            "subnormal_frac": sub}


# --------------------------------------------------------------------------
# The FP16 operand cache.
# --------------------------------------------------------------------------
#
# The MMA needs FP16 operands and the model holds BF16, so a conversion has to
# happen somewhere. Three places were measured on this GPU at W1's shapes:
#
#   operands pre-converted to FP16        d->d  5487 us   ff1  22935 us
#   BF16 passed straight to the kernel          7372 us        30561 us
#   BF16 + cast in the kernel's registers       5971 us        26977 us
#   BF16 + one host conversion (this)           5902 us        23657 us
#
# **Casting in the kernel is 8-14% WORSE on the FFN shapes**, which is worth
# writing down because it looks like the obvious fix. The grid re-reads each A
# element `ceil(N / BLOCK_N)` times -- 128 times for ff1 -- so an in-register
# cast performs the conversion 128 times where a single streaming pass over
# global memory does it once. Converting once and reusing is right.
#
# What *is* waste is converting the same tensor repeatedly. A fused block feeds
# one visual stream to six different projections (`to_query`, `to_key`,
# `to_value`, the text cross query, the va-cross query, and the av-cross
# key/value), so the identical 411 MB activation was converted six times a
# block. This caches the last conversion.
#
# The key includes `_version`, which PyTorch bumps on any in-place write, so a
# mutated tensor cannot be served a stale copy. One entry only: the copy is the
# size of the activation (411 MB at W1) and this runs on a board with a few GiB
# spare, so holding two would cost more than it saves.
_FP16_CACHE: dict[str, object] = {"key": None, "value": None}

# Below this many elements the conversion is cheap enough that the bookkeeping
# and the retained memory are not worth it.
_CACHE_MIN_ELEMENTS = 1 << 22


def _as_fp16(t: torch.Tensor) -> torch.Tensor:
    """``t`` in FP16, reusing the last conversion when it is the same tensor."""
    if t.dtype == torch.float16:
        return t
    if t.numel() < _CACHE_MIN_ELEMENTS:
        return t.to(torch.float16)
    key = (t.data_ptr(), tuple(t.shape), tuple(t.stride()), t.dtype, t._version)
    if _FP16_CACHE["key"] == key:
        return _FP16_CACHE["value"]
    value = t.to(torch.float16)
    _FP16_CACHE["key"] = key
    _FP16_CACHE["value"] = value
    return value


def clear_fp16_cache() -> None:
    """Drop the retained FP16 copy. For tests and for freeing memory between
    requests; correctness never depends on calling it."""
    _FP16_CACHE["key"] = None
    _FP16_CACHE["value"] = None


def hybrid_matmul(x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor | None = None,
                  *, config: dict | None = None, out_dtype: torch.dtype | None = None) -> torch.Tensor:
    """``x @ w.T (+ bias)`` through the hybrid kernel.

    ``w`` is in ``nn.Linear``'s ``(out_features, in_features)`` layout, so this
    is a drop-in for ``F.linear``. Operands are cast to FP16 if they are not
    already; the caller is expected to have checked range with ``fp16_safety``.
    """
    cfg = {**_DEFAULT, **(config or {})}
    x2 = x.reshape(-1, x.shape[-1])
    M, K = x2.shape
    N = w.shape[0]
    if w.shape[1] != K:
        raise ValueError(f"weight {tuple(w.shape)} does not match input feature size {K}")

    xh = _as_fp16(x2)
    # The kernel reads B as (K, N); `w.t()` is a view, and a non-contiguous B is
    # fine here because the strides are passed explicitly. The weight is not
    # cached here: under distributed layerwise offload a block's weights are
    # re-staged every step, so a cache would never hit. Moving that conversion
    # onto the DLO copy stream is the right fix and belongs in the offload
    # backend, not here.
    wh = (w if w.dtype == torch.float16 else w.to(torch.float16)).t()
    want = out_dtype or x.dtype
    c = torch.empty((M, N), device=x.device,
                    dtype=torch.float16 if want == torch.float16 else torch.bfloat16)
    bias_h = None if bias is None else bias.to(torch.float32)

    grid = (triton.cdiv(M, cfg["BLOCK_M"]) * triton.cdiv(N, cfg["BLOCK_N"]),)
    _hybrid_mm[grid](
        xh, wh, bias_h if bias_h is not None else xh, c, M, N, K,
        xh.stride(0), xh.stride(1), wh.stride(0), wh.stride(1), c.stride(0), c.stride(1),
        BLOCK_M=cfg["BLOCK_M"], BLOCK_N=cfg["BLOCK_N"], BLOCK_K=cfg["BLOCK_K"],
        GROUP_M=cfg["GROUP_M"], HAS_BIAS=bias is not None,
        OUT_FP16=(want == torch.float16),
        num_warps=cfg["num_warps"], num_stages=cfg["num_stages"],
    )
    out = c if want == c.dtype else c.to(want)
    return out.reshape(*x.shape[:-1], N)


def enabled() -> bool:
    """``VLLM_OMNI_K6_HYBRID_GEMM=1`` turns the kernel on. Unset changes nothing."""
    return os.environ.get("VLLM_OMNI_K6_HYBRID_GEMM", "") not in ("", "0", "false", "False")


# --------------------------------------------------------------------------
# The drop-in: what the serving path calls.
# --------------------------------------------------------------------------

# Below this many rows cuBLAS wins and the hybrid is a regression. Measured on
# this GPU at W1's K/N (see the crossover table in measurements.md):
#
#        M     4096x4096   4096x16384   2048x2048
#      218        0.59x        0.57x       0.37x
#      512        0.85x        0.83x       0.57x
#     1024        1.11x        0.99x       0.98x
#     2048        1.03x        1.14x       1.22x
#    50220        1.46x        1.57x       1.17x
#
# 1024 is where it stops losing and 2048 where it reliably wins, so the gate is
# 2048. This keeps W1's audio branch (M=218) and text tower (M=256) on cuBLAS,
# where they belong: the hybrid is 1.7-2.7x SLOWER there, and together they are
# 0.02 s of a 7.98 s/step census, so there is nothing to win and a regression to
# avoid.
MIN_ROWS_FOR_HYBRID = 2048

# How much FP16 headroom a tensor must have before its layer is converted. FP16
# tops out at 65504 and the accumulator is FP16 within a block, so a partial sum
# can reach a multiple of the operand magnitudes; 8x is a deliberately blunt
# margin rather than a derived bound, and a layer that fails it keeps the
# current path instead of getting a scale (scaling is Track M's call, since it
# changes what the checkpoint means).
MIN_FP16_HEADROOM = 8.0

# Activation range is re-checked on a strided sample on every call rather than
# cached per layer, because the residual stream's scale is not constant across
# sampler steps -- an early step and a late step are different distributions, and
# a verdict cached from step 1 would not cover step 10. Every 64th row costs
# ~1.5% of a pass over the activations and catches a gross range violation.
_RANGE_SAMPLE_STRIDE = 64

_warned: set[str] = set()


def _warn_once(key: str, message: str) -> None:
    if key not in _warned:
        _warned.add(key)
        print(f"[hybrid_gemm] {message}", flush=True)


def should_use_hybrid(x: torch.Tensor, w: torch.Tensor, *, name: str = "") -> tuple[bool, str]:
    """Whether this call should take the hybrid kernel, and why not if not.

    Returns a reason rather than just a bool so that a serving run can be
    audited after the fact: "the switch was on" and "the kernel actually ran"
    are different statements, and this study has already been bitten by a flag
    that silently did nothing.
    """
    if x.dtype not in (torch.bfloat16, torch.float16) or w.dtype not in (torch.bfloat16, torch.float16):
        return False, f"dtype {x.dtype}/{w.dtype} not handled"
    rows = x.reshape(-1, x.shape[-1]).shape[0]
    if rows < MIN_ROWS_FOR_HYBRID:
        return False, f"M={rows} below the {MIN_ROWS_FOR_HYBRID}-row crossover"

    w_safe = fp16_safety(w)
    if w_safe["headroom"] < MIN_FP16_HEADROOM:
        return False, (f"weight absmax {w_safe['absmax']:.3g} leaves only "
                       f"{w_safe['headroom']:.1f}x FP16 headroom")
    x2 = x.reshape(-1, x.shape[-1])
    x_safe = fp16_safety(x2[::_RANGE_SAMPLE_STRIDE])
    if x_safe["headroom"] < MIN_FP16_HEADROOM:
        return False, (f"activation absmax {x_safe['absmax']:.3g} leaves only "
                       f"{x_safe['headroom']:.1f}x FP16 headroom")
    return True, "ok"


def hybrid_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None,
                  *, name: str = "") -> torch.Tensor:
    """A drop-in for ``F.linear`` that uses the hybrid kernel where it helps.

    Falls back to ``F.linear`` whenever the hybrid would be a regression or
    unsafe, and says so once per reason. The switch
    (``VLLM_OMNI_K6_HYBRID_GEMM``) is checked here, so a caller can route every
    linear through this unconditionally and an unset environment changes
    nothing.
    """
    if not enabled():
        return torch.nn.functional.linear(x, weight, bias)
    ok, why = should_use_hybrid(x, weight, name=name)
    if not ok:
        _warn_once(f"{name}:{why}", f"{name or 'linear'} on the cuBLAS path: {why}")
        return torch.nn.functional.linear(x, weight, bias)
    return hybrid_matmul(x, weight, bias)
