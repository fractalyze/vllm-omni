# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The deterministic Marlin MoE patch: Marlin's MoE alignment runs vLLM PR
#48032's kernels, which emit each expert's rows in row order on every call,
and importing vllm_omni installs them without building them.

The alignment cases use the Qwen3-Omni thinker's routing (128 experts, top 8)
at the block_size_m vLLM 0.30.0's fused_marlin_moe picks for each token count;
32 tokens (256 routes) take the stable one-block kernel, 64 and 2048 the
radix path. The kernels are JIT-built, so those cases need a CUDA GPU and the
CUDA toolkit wheel beside torch.
"""

import subprocess
import sys

import pytest
import torch
from vllm.model_executor.layers.fused_moe.experts import marlin_moe

import vllm_omni.patch as patch_module
from vllm_omni.model_executor.layers.marlin_moe_align import align
from vllm_omni.platforms import current_omni_platform

NUM_EXPERTS, TOPK = 128, 8
# (tokens, block_size_m) as fused_marlin_moe chooses them for 128 experts, top 8.
CASES = [(32, 8), (64, 8), (2048, 64)]

requires_cuda = pytest.mark.skipif(not current_omni_platform.is_cuda(), reason="the alignment kernels are CUDA kernels")


def _topk_ids(num_tokens: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(num_tokens)
    scores = torch.rand(num_tokens, NUM_EXPERTS, generator=generator)
    return scores.topk(TOPK, dim=1).indices.to(torch.int32).cuda()


def _row_ordered(topk_ids: torch.Tensor, block_size: int) -> tuple[torch.Tensor, torch.Tensor, int]:
    """The canonical alignment: experts in order, each expert's rows in row
    order, padded with the route count to a whole block. Returns the active
    sorted ids, the active blocks' expert ids, and the padded total."""
    rows = topk_ids.flatten().cpu()
    num_rows = rows.numel()
    sorted_ids, expert_ids = [], []
    for expert in range(NUM_EXPERTS):
        segment = torch.nonzero(rows == expert).flatten().tolist()
        if not segment:
            continue
        padded = -(-len(segment) // block_size) * block_size
        sorted_ids += segment + [num_rows] * (padded - len(segment))
        expert_ids += [expert] * (padded // block_size)
    return torch.tensor(sorted_ids, dtype=torch.int32), torch.tensor(expert_ids, dtype=torch.int32), len(sorted_ids)


def _active(topk_ids: torch.Tensor, block_size: int) -> tuple[torch.Tensor, torch.Tensor, int]:
    sorted_ids, expert_ids, post_padded = align.moe_align_block_size(
        topk_ids, block_size, NUM_EXPERTS, None, ignore_invalid_experts=True
    )
    total = int(post_padded.item())
    return sorted_ids[:total].cpu(), expert_ids[: total // block_size].cpu(), total


@pytest.mark.local_model
@pytest.mark.cuda
@requires_cuda
@pytest.mark.parametrize("num_tokens,block_size", CASES)
def test_alignment_is_row_ordered(num_tokens, block_size):
    topk_ids = _topk_ids(num_tokens)
    sorted_ids, expert_ids, total = _active(topk_ids, block_size)
    expected_ids, expected_experts, expected_total = _row_ordered(topk_ids, block_size)

    assert total == expected_total
    assert torch.equal(sorted_ids, expected_ids)
    assert torch.equal(expert_ids, expected_experts)


@pytest.mark.local_model
@pytest.mark.cuda
@requires_cuda
@pytest.mark.parametrize("num_tokens,block_size", CASES)
def test_repeated_alignments_agree(num_tokens, block_size):
    topk_ids = _topk_ids(num_tokens)
    prefixes = {tuple(_active(topk_ids, block_size)[0].tolist()) for _ in range(10)}
    assert len(prefixes) == 1


@pytest.mark.core_model
@pytest.mark.cpu
def test_patch_installs_the_pr_alignment_without_building_it():
    """Importing vllm_omni points Marlin at the PR's alignment and compiles
    nothing: a server that never runs Marlin never builds the kernels."""
    script = """
from torch.utils import cpp_extension

def refuse(*args, **kwargs):
    raise AssertionError("the alignment kernels were built at import")

cpp_extension.load = refuse
import vllm_omni  # noqa: F401
from vllm.model_executor.layers.fused_moe.experts import marlin_moe
from vllm_omni.model_executor.layers.marlin_moe_align import _ext, align

assert marlin_moe.moe_align_block_size is align.moe_align_block_size
assert _ext.load.cache_info().currsize == 0
"""
    subprocess.run([sys.executable, "-c", script], check=True)


@pytest.mark.core_model
@pytest.mark.cpu
def test_patch_installs_once():
    patch_module._patch_deterministic_marlin_moe()
    installed = marlin_moe.moe_align_block_size
    patch_module._patch_deterministic_marlin_moe()
    assert marlin_moe.moe_align_block_size is installed is align.moe_align_block_size
