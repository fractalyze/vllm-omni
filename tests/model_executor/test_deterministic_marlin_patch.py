# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The VLLM_OMNI_DETERMINISTIC_MARLIN patch: Marlin's MoE alignment comes out
with each expert's rows in row order, whatever order the atomics left them
in, and decode-sized alignments are left as they are.
"""

import pytest
import torch
from vllm.model_executor.layers.fused_moe.experts import marlin_moe

import vllm_omni.patch as patch_module

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _alignment(topk_ids: torch.Tensor, num_experts: int, block_size: int, generator: torch.Generator):
    """(sorted_ids, expert_ids, num_tokens_post_padded) as moe_align_block_size
    lays them out, with each segment's rows in a random order, and the row-
    ordered sorted_ids the patch must produce. The buffers run one block past
    the padded total, as vLLM sizes them, with unset expert ids there."""
    rows = topk_ids.flatten()
    num_rows = rows.numel()
    shuffled, ordered, expert_ids = [], [], []
    for expert in range(num_experts):
        segment = torch.nonzero(rows == expert).flatten()
        if segment.numel() == 0:
            continue
        padded = -(-segment.numel() // block_size) * block_size
        padding = [num_rows] * (padded - segment.numel())
        permuted = segment[torch.randperm(segment.numel(), generator=generator)]
        shuffled += permuted.tolist() + padding
        ordered += segment.tolist() + padding
        expert_ids += [expert] * (padded // block_size)
    total = len(shuffled)
    tail = [num_rows] * block_size
    return (
        torch.tensor(shuffled + tail, dtype=torch.int32),
        torch.tensor(expert_ids + [-7], dtype=torch.int32),
        torch.tensor([total], dtype=torch.int32),
        torch.tensor(ordered + tail, dtype=torch.int32),
    )


@pytest.mark.parametrize("num_tokens,block_size", [(32, 8), (64, 16), (9, 8)])
def test_segments_come_out_in_row_order(num_tokens, block_size):
    generator = torch.Generator().manual_seed(num_tokens)
    num_experts, topk = 16, 4
    topk_ids = torch.stack([torch.randperm(num_experts, generator=generator)[:topk] for _ in range(num_tokens)])
    sorted_ids, expert_ids, post_padded, expected = _alignment(topk_ids, num_experts, block_size, generator)
    assert not torch.equal(sorted_ids, expected)

    result = patch_module._sorted_within_experts(sorted_ids, expert_ids, post_padded, block_size, topk_ids.numel())

    assert result.dtype == sorted_ids.dtype
    assert torch.equal(result, expected)


class _ShuffledAlign:
    """Marlin's moe_align_block_size returning a shuffled alignment; keeps the
    shuffled and the row-ordered ids of its last call."""

    def __init__(self):
        self.generator = torch.Generator().manual_seed(0)
        self.shuffled = self.expected = None

    def __call__(self, topk_ids, block_size, num_experts, *args, **kwargs):
        sorted_ids, expert_ids, post_padded, self.expected = _alignment(
            topk_ids, num_experts, block_size, self.generator
        )
        self.shuffled = sorted_ids
        return sorted_ids, expert_ids, post_padded


@pytest.fixture
def fake_align(monkeypatch):
    align = _ShuffledAlign()
    monkeypatch.setattr(marlin_moe, "moe_align_block_size", align)
    return align


def test_switch_off_leaves_marlin_alone(monkeypatch, fake_align):
    monkeypatch.delenv("VLLM_OMNI_DETERMINISTIC_MARLIN", raising=False)
    patch_module._patch_deterministic_marlin_moe()
    assert marlin_moe.moe_align_block_size is fake_align


def test_switch_on_sorts_prefill_alignments(monkeypatch, fake_align):
    monkeypatch.setenv("VLLM_OMNI_DETERMINISTIC_MARLIN", "1")
    patch_module._patch_deterministic_marlin_moe()
    topk_ids = torch.stack([torch.randperm(8)[:2] for _ in range(32)])

    sorted_ids, _, _ = marlin_moe.moe_align_block_size(topk_ids, 8, 8, None, ignore_invalid_experts=True)

    assert torch.equal(sorted_ids, fake_align.expected)


def test_switch_on_skips_decode_sized_alignments(monkeypatch, fake_align):
    monkeypatch.setenv("VLLM_OMNI_DETERMINISTIC_MARLIN", "1")
    patch_module._patch_deterministic_marlin_moe()
    topk_ids = torch.stack([torch.randperm(8)[:2] for _ in range(8)])

    sorted_ids, _, _ = marlin_moe.moe_align_block_size(topk_ids, 8, 8)

    # Eight tokens fit one block per expert: the atomics' order is returned.
    assert sorted_ids is fake_align.shuffled


def test_patch_installs_once(monkeypatch, fake_align):
    monkeypatch.setenv("VLLM_OMNI_DETERMINISTIC_MARLIN", "1")
    patch_module._patch_deterministic_marlin_moe()
    installed = marlin_moe.moe_align_block_size
    patch_module._patch_deterministic_marlin_moe()
    assert marlin_moe.moe_align_block_size is installed
