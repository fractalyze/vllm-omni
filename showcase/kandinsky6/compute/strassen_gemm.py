# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# ruff: noqa: N803  (Triton kernels follow the Triton convention: A, B, C, M, N, K, BLOCK_*)
"""R2: one level of Strassen-Winograd over the hybrid FP16-accumulate GEMM.

Strassen computes a 2x2 block product with **seven** half-size multiplications
instead of eight, so one level removes 12.5% of the FLOPs. On Kandinsky 6's FFN
shapes that is 6.74 -> 5.90 TFLOP for ff1.

The FLOPs are not what decides it. One level has to **materialise seven
half-size products** -- 1.75x the output's own size -- and read them back to
assemble four output blocks, and these GEMMs have an arithmetic intensity of
~3076 FLOP/byte, which is to say they are nowhere near memory-bound and have
nothing to gain by trading compute for traffic.

The implementation below is the cheapest arrangement there is, so that the
measurement bounds the idea rather than one coding of it:

* Two arrangements of the ten input additions, because the obvious one is the
  worse one and that is the finding. **Fusing them into the kernel prologue**
  (``fuse_sums=True``) never writes ``A11 + A22`` to memory -- but it **doubles
  the operand loads**, and measured on this GPU a product with one fused sum
  costs 1.30-1.36x a plain half GEMM and one with two costs 1.85-1.98x. That
  turns 7/8 of the FLOPs into 1.38x of the time. **Materialising the sums**
  (``fuse_sums=False``, the default) spends O(MK + KN) of traffic -- small
  beside these shapes' O(MN) output -- and lets each product run as an
  unmodified hybrid GEMM.
* **The assembly is a single fused pass.** One kernel reads all seven products
  at a tile position and writes all four output blocks, which is the information
  -theoretic minimum for the materialising form: seven reads and four writes.
* Products whose A or B side is a single block (M2..M5) take a variant that does
  not load a second tile.

A register-level Strassen, which would avoid the traffic entirely, is not
possible here: one 128x128 FP32 accumulator already needs 128 registers a
thread, and seven of them do not fit in a 255-register file. That is why the
materialising form is the only form.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# (name, a_blocks, a_signs, b_blocks, b_signs) with blocks as (row, col) of the
# 2x2 split. This is classic Strassen; the Winograd variant trades two of the
# eighteen additions for two more, and since the additions on the input side are
# free here it would change nothing measurable.
PRODUCTS = [
    ("M1", [(0, 0), (1, 1)], [1.0, 1.0], [(0, 0), (1, 1)], [1.0, 1.0]),
    ("M2", [(1, 0), (1, 1)], [1.0, 1.0], [(0, 0)], [1.0]),
    ("M3", [(0, 0)], [1.0], [(0, 1), (1, 1)], [1.0, -1.0]),
    ("M4", [(1, 1)], [1.0], [(1, 0), (0, 0)], [1.0, -1.0]),
    ("M5", [(0, 0), (0, 1)], [1.0, 1.0], [(1, 1)], [1.0]),
    ("M6", [(1, 0), (0, 0)], [1.0, -1.0], [(0, 0), (0, 1)], [1.0, 1.0]),
    ("M7", [(0, 1), (1, 1)], [1.0, -1.0], [(1, 0), (1, 1)], [1.0, 1.0]),
]

# C11 = M1 + M4 - M5 + M7 ; C12 = M3 + M5 ; C21 = M2 + M4 ; C22 = M1 - M2 + M3 + M6
COMBINE = {
    (0, 0): {0: 1.0, 3: 1.0, 4: -1.0, 6: 1.0},
    (0, 1): {2: 1.0, 4: 1.0},
    (1, 0): {1: 1.0, 3: 1.0},
    (1, 1): {0: 1.0, 1: -1.0, 2: 1.0, 5: 1.0},
}


@triton.jit
def _strassen_product(A, B, C, M2_, N2_, K2_,
                      a_off1, a_off2, b_off1, b_off2,
                      sa2, sb2, sam, sak, sbk, sbn, scm, scn,
                      BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                      GROUP_M: tl.constexpr, TWO_A: tl.constexpr, TWO_B: tl.constexpr):
    """One Strassen product: (A[o1] + sa2*A[o2]) @ (B[o1] + sb2*B[o2]).

    Accumulation is the hybrid: FP16 inside each K-block, promoted into an FP32
    running sum per block, exactly as ``hybrid_gemm._hybrid_mm`` does. The block
    sums are formed in FP32 and cast to FP16 for the MMA, so the input addition
    itself is not what loses precision.
    """
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M2_, BLOCK_M)
    num_pid_n = tl.cdiv(N2_, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    m_mask = rm[:, None] < M2_
    n_mask = rn[None, :] < N2_

    a1 = A + a_off1 + rm[:, None] * sam + rk[None, :] * sak
    a2 = A + a_off2 + rm[:, None] * sam + rk[None, :] * sak
    b1 = B + b_off1 + rk[:, None] * sbk + rn[None, :] * sbn
    b2 = B + b_off2 + rk[:, None] * sbk + rn[None, :] * sbn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K2_, BLOCK_K):
        k_ok = rk + k < K2_
        x = tl.load(a1, mask=m_mask & k_ok[None, :], other=0.0).to(tl.float32)
        if TWO_A:
            x += sa2 * tl.load(a2, mask=m_mask & k_ok[None, :], other=0.0).to(tl.float32)
        y = tl.load(b1, mask=k_ok[:, None] & n_mask, other=0.0).to(tl.float32)
        if TWO_B:
            y += sb2 * tl.load(b2, mask=k_ok[:, None] & n_mask, other=0.0).to(tl.float32)
        acc += tl.dot(x.to(tl.float16), y.to(tl.float16), out_dtype=tl.float16).to(tl.float32)
        a1 += BLOCK_K * sak
        a2 += BLOCK_K * sak
        b1 += BLOCK_K * sbk
        b2 += BLOCK_K * sbk

    c = C + rm[:, None] * scm + rn[None, :] * scn
    tl.store(c, acc.to(tl.float16), mask=m_mask & n_mask)


@triton.jit
def _combine(P0, P1, P2, P3, P4, P5, P6, C, M2_, N2_, spm, spn, scm, scn,
             BLOCK: tl.constexpr):
    """Assemble the four output blocks from the seven products in one pass.

    Flat 1-D over the product buffers rather than 2-D tiles. The 2-D version
    held seven ``BLOCK_M x BLOCK_N`` FP32 tiles live at once -- 448 KB of
    registers at 128x128 -- and spilled so hard it reached 0.286 TB/s where a
    plain copy on this card does 1.52. Flat indexing keeps five vectors of
    ``BLOCK`` live and the accesses coalesced: a run of ``BLOCK`` consecutive
    product elements is consecutive in each output block too, since the blocks
    share the product's row length.

    The three products that feed two outputs each (M1, M2, M3) are loaded first
    and folded into all four accumulators, so nothing has to be re-read.
    """
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    total = M2_ * N2_
    mask = idx < total
    row = idx // N2_
    col = idx % N2_
    off = row * spm + col * spn

    m1 = tl.load(P0 + off, mask=mask, other=0.0).to(tl.float32)
    m2 = tl.load(P1 + off, mask=mask, other=0.0).to(tl.float32)
    m3 = tl.load(P2 + off, mask=mask, other=0.0).to(tl.float32)
    c11 = m1
    c12 = m3
    c21 = m2
    c22 = m1 - m2 + m3
    m4 = tl.load(P3 + off, mask=mask, other=0.0).to(tl.float32)
    c11 += m4
    c21 += m4
    m5 = tl.load(P4 + off, mask=mask, other=0.0).to(tl.float32)
    c11 -= m5
    c12 += m5
    c22 += tl.load(P5 + off, mask=mask, other=0.0).to(tl.float32)
    c11 += tl.load(P6 + off, mask=mask, other=0.0).to(tl.float32)

    c_off = row * scm + col * scn
    tl.store(C + c_off, c11.to(tl.float16), mask=mask)
    tl.store(C + c_off + N2_ * scn, c12.to(tl.float16), mask=mask)
    tl.store(C + c_off + M2_ * scm, c21.to(tl.float16), mask=mask)
    tl.store(C + c_off + M2_ * scm + N2_ * scn, c22.to(tl.float16), mask=mask)


_CFG = dict(BLOCK_M=128, BLOCK_N=128, BLOCK_K=32, GROUP_M=8, num_warps=4, num_stages=2)


def strassen_matmul(x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor | None = None,
                    *, config: dict | None = None, out: torch.Tensor | None = None,
                    scratch: list[torch.Tensor] | None = None,
                    fuse_sums: bool = False) -> torch.Tensor:
    """``x @ w.T (+ bias)`` with one level of Strassen over the hybrid kernel.

    Requires M, K and N all even, which W1's FFN shapes are (50220, 4096, 16384).
    ``scratch`` lets a caller hand in the seven product buffers so a benchmark
    does not measure allocation; they are 1.75x the output's size in total.

    ``fuse_sums=False`` (the default) materialises the ten input combinations and
    runs seven unmodified hybrid GEMMs: **0.946-0.962x** the single hybrid GEMM
    on the FFN shapes. ``fuse_sums=True`` folds them into the kernel prologue
    instead and is much worse, **0.663-0.738x**, because it doubles the operand
    loads. Neither wins, so nothing here is on a serving path; both are kept
    because the gap between them is the result.
    """
    if not fuse_sums:
        return _strassen_materialised(x, w, bias, config=config, out=out, scratch=scratch)
    cfg = {**_CFG, **(config or {})}
    x2 = x.reshape(-1, x.shape[-1])
    M, K = x2.shape
    N = w.shape[0]
    if M % 2 or K % 2 or N % 2:
        raise ValueError(f"one-level Strassen needs even M, K, N; got {M}, {K}, {N}")

    xh = x2 if x2.dtype == torch.float16 else x2.to(torch.float16)
    wh = (w if w.dtype == torch.float16 else w.to(torch.float16)).t()  # (K, N) view
    M2, K2, N2 = M // 2, K // 2, N // 2

    ps = scratch if scratch is not None else [
        torch.empty((M2, N2), device=x.device, dtype=torch.float16) for _ in range(7)]

    sam, sak = xh.stride(0), xh.stride(1)
    sbk, sbn = wh.stride(0), wh.stride(1)
    grid = (triton.cdiv(M2, cfg["BLOCK_M"]) * triton.cdiv(N2, cfg["BLOCK_N"]),)
    for i, (_name, a_blocks, a_signs, b_blocks, b_signs) in enumerate(PRODUCTS):
        a_off = [r * M2 * sam + c * K2 * sak for r, c in a_blocks]
        b_off = [r * K2 * sbk + c * N2 * sbn for r, c in b_blocks]
        two_a, two_b = len(a_blocks) == 2, len(b_blocks) == 2
        _strassen_product[grid](
            xh, wh, ps[i], M2, N2, K2,
            a_off[0], a_off[1] if two_a else a_off[0],
            b_off[0], b_off[1] if two_b else b_off[0],
            a_signs[1] if two_a else 0.0, b_signs[1] if two_b else 0.0,
            sam, sak, sbk, sbn, ps[i].stride(0), ps[i].stride(1),
            BLOCK_M=cfg["BLOCK_M"], BLOCK_N=cfg["BLOCK_N"], BLOCK_K=cfg["BLOCK_K"],
            GROUP_M=cfg["GROUP_M"], TWO_A=two_a, TWO_B=two_b,
            num_warps=cfg["num_warps"], num_stages=cfg["num_stages"])

    c = out if out is not None else torch.empty((M, N), device=x.device, dtype=torch.float16)
    cblock = 1024
    cgrid = (triton.cdiv(M2 * N2, cblock),)
    _combine[cgrid](*ps, c, M2, N2, ps[0].stride(0), ps[0].stride(1), c.stride(0), c.stride(1),
                    BLOCK=cblock, num_warps=4)
    result = c if bias is None else c + bias.to(c.dtype)
    return result.reshape(*x.shape[:-1], N).to(x.dtype)


def _strassen_materialised(x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor | None,
                           *, config: dict | None, out: torch.Tensor | None,
                           scratch: list[torch.Tensor] | None) -> torch.Tensor:
    """Strassen with the input combinations written to memory first.

    The faster of the two arrangements and still a loss. For ff1
    (50220x4096x16384): seven half GEMMs 19.98 ms + input sums 1.31 ms + assembly
    2.90 ms = 24.20 ms, against one hybrid GEMM's 23.27 ms. The seven products do
    come in under the whole GEMM -- 0.800x, better than the 0.875 the FLOP count
    promises, because the half shape runs at a higher rate -- and then the sums
    and the assembly spend the saving and a little more.
    """
    from hybrid_gemm import hybrid_matmul  # noqa: PLC0415  (measurement-only path)

    x2 = x.reshape(-1, x.shape[-1])
    M, K = x2.shape
    N = w.shape[0]
    if M % 2 or K % 2 or N % 2:
        raise ValueError(f"one-level Strassen needs even M, K, N; got {M}, {K}, {N}")
    M2, K2, N2 = M // 2, K // 2, N // 2
    xh = x2 if x2.dtype == torch.float16 else x2.to(torch.float16)
    wh = w if w.dtype == torch.float16 else w.to(torch.float16)

    a = {(i, j): xh[i * M2:(i + 1) * M2, j * K2:(j + 1) * K2].contiguous()
         for i in (0, 1) for j in (0, 1)}
    # The GEMM is x @ w.T, so B = w.T and B[i][j] is w[j-block rows, i-block cols].
    b = {(i, j): wh[j * N2:(j + 1) * N2, i * K2:(i + 1) * K2].contiguous()
         for i in (0, 1) for j in (0, 1)}

    m = [
        hybrid_matmul(a[(0, 0)] + a[(1, 1)], b[(0, 0)] + b[(1, 1)], out_dtype=torch.float16),
        hybrid_matmul(a[(1, 0)] + a[(1, 1)], b[(0, 0)], out_dtype=torch.float16),
        hybrid_matmul(a[(0, 0)], b[(0, 1)] - b[(1, 1)], out_dtype=torch.float16),
        hybrid_matmul(a[(1, 1)], b[(1, 0)] - b[(0, 0)], out_dtype=torch.float16),
        hybrid_matmul(a[(0, 0)] + a[(0, 1)], b[(1, 1)], out_dtype=torch.float16),
        hybrid_matmul(a[(1, 0)] - a[(0, 0)], b[(0, 0)] + b[(0, 1)], out_dtype=torch.float16),
        hybrid_matmul(a[(0, 1)] - a[(1, 1)], b[(1, 0)] + b[(1, 1)], out_dtype=torch.float16),
    ]
    c = out if out is not None else torch.empty((M, N), device=x.device, dtype=torch.float16)
    cblock = 1024
    _combine[(triton.cdiv(M2 * N2, cblock),)](
        *m, c, M2, N2, m[0].stride(0), m[0].stride(1), c.stride(0), c.stride(1),
        BLOCK=cblock, num_warps=4)
    result = c if bias is None else c + bias.to(c.dtype)
    return result.reshape(*x.shape[:-1], N).to(x.dtype)
