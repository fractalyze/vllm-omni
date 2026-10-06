# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Sort CUDA kernel names into the categories a DiT step is split by.

A profile of one Kandinsky 6 denoise step answers one question: how much of
the step is attention, how much is the block GEMMs, and how much is the
elementwise work around them (AdaLN modulation, RoPE, norms, residual
gates). The split has to come from kernel names, because the port's forward
is plain eager PyTorch with no profiler annotations, and adding them would
change what is measured.

Name matching is ordered and the first rule wins, so a narrow rule must
precede a broad one: an attention kernel whose name happens to contain
``gemm`` has to be claimed by ATTENTION before GEMM sees it. Anything no
rule claims lands in ``unclassified`` and is reported by name, never folded
into a category — an unnamed kernel is a gap in this table, not elementwise
work.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

# Categories, in report order. ``unclassified`` is deliberately last and is
# not a bucket anyone may add to by hand.
CATEGORIES = ("attention", "gemm", "elementwise", "norm", "copy", "unclassified")

# (category, compiled pattern). Order matters; see the module docstring.
_RULES: list[tuple[str, re.Pattern[str]]] = [
    # --- attention -----------------------------------------------------
    # FlashAttention 2/3/4 (including the Blackwell CuTe build), cuDNN
    # fused MHA, the SDPA mem-efficient and flash paths, FlashInfer's
    # ragged prefill, SageAttention's INT8/FP8 kernels, and the
    # flex_attention template the NABLA path lowers to.
    ("attention", re.compile(r"flash_fwd|flash_?atten|fmha|mha_fwd", re.I)),
    ("attention", re.compile(r"cudnn.*(sdpa|attn|mha)|(sdpa|attn|mha).*cudnn", re.I)),
    ("attention", re.compile(r"efficient_attention|attention_kernel|scaled_dot_product", re.I)),
    ("attention", re.compile(r"BatchPrefill|batch_prefill|ragged.*prefill", re.I)),
    # SageAttention. The CUDA kernels the sm_120 path actually emits are
    # `qk_int_sv_f8_attn_kernel` (no digit after `int`), plus two prologues --
    # `QuantInt8Kernel` and `MeanScaleKernel` -- that quantize Q and K per
    # block. An earlier version of this table had only the `qk_int8_sv` name
    # from SageAttention's other variants and its Triton kernels'
    # `quant_per_block`, so 61.43 ms of a 260 ms block (23.5%) landed in
    # `unclassified` and attention read as 0.7%. The `\d*` and the two
    # prologue names are the fix; the names are kept rather than loosened to
    # `qk_` so an unrelated kernel cannot drift into the attention share.
    ("attention", re.compile(r"qk_int\d*_sv|qk_int\d*_pv|sageattn|sage_attn", re.I)),
    ("attention", re.compile(r"QuantInt8Kernel|MeanScaleKernel|quant_per_block|quant_per_thread|sub_mean")),
    ("attention", re.compile(r"flex_attention|triton_tem_fused_.*attention", re.I)),
    # --- GEMM ----------------------------------------------------------
    # cuBLASLt's Blackwell kernels are nvjet_*; cuBLAS classic kernels keep
    # the sm??_*_gemm / *_tn_*_kernel shapes; CUTLASS emits cutlass_* or
    # a cutlass::device_kernel wrapper.
    ("gemm", re.compile(r"nvjet|cutlass|^gemm|_gemm|gemm_|cublas|sgemm|hgemm|bgemm|xmma", re.I)),
    ("gemm", re.compile(r"bmm|batched_gemm|addmm|matmul", re.I)),
    ("gemm", re.compile(r"device_kernel|tensorop", re.I)),
    # --- norm ----------------------------------------------------------
    # LayerNorm and RMSNorm, split out of elementwise because the port
    # calls them on fp32 upcasts of the residual stream and their traffic
    # is a fusion target of its own.
    ("norm", re.compile(r"layer_norm|layernorm|rms_norm|rmsnorm|GroupNorm|group_norm", re.I)),
    ("norm", re.compile(r"vectorized_layer_norm|RowwiseMoments|welford", re.I)),
    # --- copy / layout -------------------------------------------------
    # Transposes, contiguous() and the device-to-device copies the
    # fractal flatten / unflatten and the (B,S,H,D) reshapes leave behind.
    ("copy", re.compile(r"CatArrayBatchedCopy|copy_device_to_device|direct_copy|vectorized_copy", re.I)),
    ("copy", re.compile(r"transpose|permute|contiguous|cat_|concat", re.I)),
    # --- elementwise ---------------------------------------------------
    # The AdaLN shift/scale/gate chain, the RoPE complex multiply and its
    # sum, GELU/SiLU, and the fp32 casts around all of them.
    ("elementwise", re.compile(r"elementwise_kernel|vectorized_elementwise|unrolled_elementwise", re.I)),
    ("elementwise", re.compile(r"reduce_kernel|ReduceOp|sum_functor|gelu|silu|sigmoid|tanh", re.I)),
    ("elementwise", re.compile(r"CUDAFunctor|BinaryFunctor|UnaryFunctor|fill_|_cast|convert", re.I)),
    ("elementwise", re.compile(r"triton_poi_fused|triton_red_fused|triton_per_fused", re.I)),
]


def classify(kernel_name: str) -> str:
    """The category of one CUDA kernel, or ``"unclassified"``."""
    for category, pattern in _RULES:
        if pattern.search(kernel_name):
            return category
    return "unclassified"


def split_by_category(kernels: Iterable[tuple[str, float]]) -> dict[str, float]:
    """Total device time per category, for ``(kernel_name, us)`` pairs.

    Every category in ``CATEGORIES`` is present, so a caller can report a
    zero share without special-casing an absent key.
    """
    totals = dict.fromkeys(CATEGORIES, 0.0)
    for name, us in kernels:
        totals[classify(name)] += us
    return totals
