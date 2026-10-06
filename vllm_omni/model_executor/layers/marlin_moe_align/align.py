# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# Copyright 2026 Fractalyze Inc. All rights reserved.
"""Deterministic Marlin MoE route alignment from vLLM PR #48032.

Provenance: vllm-project/vllm PR #48032 @ a718a4b1 (Apache-2.0, Copyright
contributors to the vLLM project).
- Verbatim from vllm/model_executor/layers/fused_moe/moe_align_block_size.py:
  RADIX_SORT_MIN_ROUTED_ENTRIES, MoEAlignRadixScratch, _allocate_outputs,
  moe_align_block_size_stable_small and moe_align_block_size_radix, with the
  `torch.ops._moe_C` / `vllm._custom_ops` calls pointed at the extension
  (_ext.py).
- Rewritten: `moe_align_block_size` here is the PR's selection inside
  fused_marlin_moe (radix path from 257 routed entries, stable one-block
  kernel below), shaped as a drop-in for the `moe_align_block_size` name
  that vLLM 0.30.0's marlin_moe module calls; and the scratch is one per
  process instead of one per MarlinExperts (see _radix_scratch).

Both paths emit every expert's routes in flattened-route order, so identical
`topk_ids` give identical `sorted_token_ids`, and Marlin's split-K sums the
same rows in the same blocks on every call.
"""

from dataclasses import dataclass, field

import torch
from vllm.triton_utils import triton
from vllm.utils.math_utils import round_up

from vllm_omni.model_executor.layers.marlin_moe_align import _ext

# The stable one-block decode path is explicitly bounded to 256 routes.
# Every larger input uses the near-linear radix path.
RADIX_SORT_MIN_ROUTED_ENTRIES = 257


@dataclass
class MoEAlignRadixScratch:
    max_num_tokens: int
    topk: int
    num_experts: int
    device: torch.device
    max_numel: int = field(init=False)
    sort_workspace: torch.Tensor = field(init=False)
    sorted_expert_ids: torch.Tensor = field(init=False)
    compact_sorted_token_ids: torch.Tensor = field(init=False)
    token_indices: torch.Tensor = field(init=False)
    topk_ids_for_sort: torch.Tensor = field(init=False)
    padded_expert_offsets: torch.Tensor = field(init=False)
    unpadded_expert_offsets: torch.Tensor = field(init=False)

    def __post_init__(self) -> None:
        self.max_numel = self.max_num_tokens * self.topk
        self.sorted_expert_ids = torch.empty(self.max_numel, dtype=torch.int32, device=self.device)
        self.compact_sorted_token_ids = torch.empty_like(self.sorted_expert_ids)
        self.token_indices = torch.arange(self.max_numel, dtype=torch.int32, device=self.device)
        self.topk_ids_for_sort = torch.empty_like(self.sorted_expert_ids)
        self.padded_expert_offsets = torch.empty(self.num_experts + 1, dtype=torch.int32, device=self.device)
        self.unpadded_expert_offsets = torch.empty(self.num_experts + 1, dtype=torch.int64, device=self.device)
        workspace_size = _ext.load().moe_permute_sort_workspace_size(self.max_numel, self.num_experts)
        self.sort_workspace = torch.empty(workspace_size, dtype=torch.int8, device=self.device)

    def validate(self, topk_ids: torch.Tensor, num_experts: int) -> None:
        assert topk_ids.device == self.token_indices.device
        assert topk_ids.size(1) == self.topk
        assert topk_ids.numel() <= self.max_numel
        assert num_experts == self.num_experts


def _allocate_outputs(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    pad_sorted_ids: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)
    if pad_sorted_ids:
        max_num_tokens_padded = round_up(max_num_tokens_padded, block_size)
    if topk_ids.numel() < num_experts:
        max_num_tokens_padded = min(topk_ids.numel() * block_size, max_num_tokens_padded)
    sorted_ids = torch.empty((max_num_tokens_padded,), dtype=torch.int32, device=topk_ids.device)
    max_num_m_blocks = triton.cdiv(max_num_tokens_padded, block_size)
    expert_ids = torch.empty((max_num_m_blocks,), dtype=torch.int32, device=topk_ids.device)
    num_tokens_post_pad = torch.empty((1), dtype=torch.int32, device=topk_ids.device)
    return sorted_ids, expert_ids, num_tokens_post_pad


def moe_align_block_size_stable_small(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    expert_map: torch.Tensor | None = None,
    pad_sorted_ids: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Deterministically align at most 256 flattened routed entries."""
    if topk_ids.numel() >= RADIX_SORT_MIN_ROUTED_ENTRIES:
        raise ValueError(f"small stable alignment supports at most {RADIX_SORT_MIN_ROUTED_ENTRIES - 1} routed entries")
    sorted_ids, expert_ids, num_tokens_post_pad = _allocate_outputs(topk_ids, block_size, num_experts, pad_sorted_ids)
    _ext.load().moe_align_block_size_stable_small(
        topk_ids,
        num_experts,
        block_size,
        sorted_ids,
        expert_ids,
        num_tokens_post_pad,
        expert_map,
    )
    return sorted_ids, expert_ids, num_tokens_post_pad


def moe_align_block_size_radix(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    scratch: MoEAlignRadixScratch,
    expert_map: torch.Tensor | None = None,
    pad_sorted_ids: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    scratch.validate(topk_ids, num_experts)
    sorted_ids, expert_ids, num_tokens_post_pad = _allocate_outputs(topk_ids, block_size, num_experts, pad_sorted_ids)
    _ext.load().moe_align_block_size_radix(
        topk_ids,
        num_experts,
        block_size,
        sorted_ids,
        expert_ids,
        num_tokens_post_pad,
        scratch.sort_workspace,
        scratch.sorted_expert_ids,
        scratch.compact_sorted_token_ids,
        scratch.token_indices,
        scratch.topk_ids_for_sort,
        scratch.padded_expert_offsets,
        scratch.unpadded_expert_offsets,
        expert_map,
    )
    return sorted_ids, expert_ids, num_tokens_post_pad


# The PR gives each MarlinExperts its own scratch, sized to the layer's
# max_num_tokens. Here one scratch per (device, topk, num_experts) serves
# every layer of the process: the scratch only carries temporaries inside one
# alignment, whose kernels finish in stream order before the layer's GEMM, and
# a process runs its MoE layers one after another on one stream (CUDA graph
# replays included), so no two alignments overlap. Sharing saves one set of
# buffers per layer (48 in the thinker). The buffer grows only outside CUDA
# graph capture: vLLM's profile run aligns max_num_batched_tokens before it
# captures, so captured graphs keep pointing at the final buffer; a replaced
# scratch is kept alive all the same, in case a graph still points at it.
_RADIX_SCRATCH: dict[tuple[torch.device, int, int], MoEAlignRadixScratch] = {}
_RETIRED_SCRATCH: list[MoEAlignRadixScratch] = []


def _radix_scratch(topk_ids: torch.Tensor, num_experts: int) -> MoEAlignRadixScratch:
    key = (topk_ids.device, topk_ids.size(1), num_experts)
    scratch = _RADIX_SCRATCH.get(key)
    if scratch is None or topk_ids.numel() > scratch.max_numel:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "Marlin MoE alignment scratch must grow outside CUDA graph capture; "
                "align the largest batch (vLLM's profile run) before capturing"
            )
        if scratch is not None:
            _RETIRED_SCRATCH.append(scratch)
        scratch = MoEAlignRadixScratch(
            max_num_tokens=topk_ids.size(0),
            topk=topk_ids.size(1),
            num_experts=num_experts,
            device=topk_ids.device,
        )
        _RADIX_SCRATCH[key] = scratch
    return scratch


def moe_align_block_size(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    expert_map: torch.Tensor | None = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """PR #48032's alignment choice in fused_marlin_moe.

    vLLM 0.30.0's fused_marlin_moe calls
    `moe_align_block_size(topk_ids, block_size_m, global_num_experts,
    expert_map, ignore_invalid_experts=True)`; both paths drop invalid expert
    ids as `ignore_invalid_experts=True` does, so the keyword is accepted and
    ignored, as the PR's call site does.
    """
    if topk_ids.numel() >= RADIX_SORT_MIN_ROUTED_ENTRIES:
        return moe_align_block_size_radix(
            topk_ids, block_size, num_experts, _radix_scratch(topk_ids, num_experts), expert_map
        )
    return moe_align_block_size_stable_small(topk_ids, block_size, num_experts, expert_map)
