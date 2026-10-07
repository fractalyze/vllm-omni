# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# ruff: noqa: N803  (Triton kernel arguments follow the Triton convention: A, B, C, M, N, K, BLOCK_*)
"""A hybrid FP16-accumulate GEMM for Kandinsky 6's linears on sm_120.

The serving copy of ``showcase/kandinsky6/compute/hybrid_gemm.py`` (kernel and
``hybrid_matmul``; that file also carries the measurement tooling and the
``hybrid_linear`` drop-in). Operands are cast to FP16 on the host, as there.
Converting them in the kernel instead is bit-identical and saves the copies, but
the conversions sit in the MMA loop: at W1's FF shapes it ran 1.01-1.03x against
the served bias-free cuBLAS GEMM, where the host cast runs 1.12-1.21x. The FP16
copies the allocator keeps are released before the VAE decode instead
(``hybrid_linear.release_hybrid_scratch``).

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
_DEFAULT = dict(BLOCK_M=128, BLOCK_N=128, BLOCK_K=32, GROUP_M=8, num_warps=4, num_stages=4)


@triton.jit
def _hybrid_mm(
    A,
    B,
    Bias,
    C,
    M,
    N,
    K,
    sam,
    sak,
    sbk,
    sbn,
    scm,
    scn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    OUT_FP16: tl.constexpr,
):
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
    return {"absmax": absmax, "headroom": FP16_MAX / absmax if absmax > 0 else float("inf"), "subnormal_frac": sub}


def hybrid_matmul(
    x: torch.Tensor,
    w: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    config: dict | None = None,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
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

    # NOT `_as_fp16` here. This function runs inside the regionally-compiled
    # DiT, and under `dynamic=True` `x2.numel()` and `x2.shape` are SymInts that
    # Dynamo cannot put in a Python dict key: it raises
    # `InternalTorchDynamoError: 'SymNodeVariable' object has no attribute
    # 'value'` on the first request. The operand cache stays in the showcase
    # copy, which is eager measurement code. Repeated conversions of the same
    # activation are therefore still paid on the serving path; removing them
    # needs something Dynamo can trace -- a custom op holding its own cache --
    # not a Python dict.
    xh = x2 if x2.dtype == torch.float16 else x2.to(torch.float16)
    # The kernel reads B as (K, N); `w.t()` is a view, and a non-contiguous B is
    # fine here because the strides are passed explicitly.
    wh = (w if w.dtype == torch.float16 else w.to(torch.float16)).t()
    want = out_dtype or x.dtype
    c = torch.empty((M, N), device=x.device, dtype=torch.float16 if want == torch.float16 else torch.bfloat16)
    bias_h = None if bias is None else bias.to(torch.float32)

    grid = (triton.cdiv(M, cfg["BLOCK_M"]) * triton.cdiv(N, cfg["BLOCK_N"]),)
    _hybrid_mm[grid](
        xh,
        wh,
        bias_h if bias_h is not None else xh,
        c,
        M,
        N,
        K,
        xh.stride(0),
        xh.stride(1),
        wh.stride(0),
        wh.stride(1),
        c.stride(0),
        c.stride(1),
        BLOCK_M=cfg["BLOCK_M"],
        BLOCK_N=cfg["BLOCK_N"],
        BLOCK_K=cfg["BLOCK_K"],
        GROUP_M=cfg["GROUP_M"],
        HAS_BIAS=bias is not None,
        OUT_FP16=(want == torch.float16),
        num_warps=cfg["num_warps"],
        num_stages=cfg["num_stages"],
    )
    out = c if want == c.dtype else c.to(want)
    return out.reshape(*x.shape[:-1], N)


def enabled() -> bool:
    """``VLLM_OMNI_K6_HYBRID_GEMM=1`` turns the kernel on. Unset changes nothing."""
    return os.environ.get("VLLM_OMNI_K6_HYBRID_GEMM", "") not in ("", "0", "false", "False")
