# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""The exact-attention schedule picks the right attention for each call.

VLLM_OMNI_K6_EXACT_ATTN_BLOCKS pins the first and last n visual blocks to exact
attention; VLLM_OMNI_K6_EXACT_ATTN_STEPS makes whole sampler steps exact via
set_exact_attention_step. With neither set, nothing changes: no twin is built.
"""

from __future__ import annotations

import pytest
import torch

from tests.diffusion.models.kandinsky6.test_quantized_layer_prefixes import TINY_CONFIG, _init_single_rank

pytestmark = pytest.mark.filterwarnings("ignore")


def _build(monkeypatch, *, steps: str = "0", blocks: str = "0", num_blocks: int = 4):
    monkeypatch.setenv("VLLM_OMNI_K6_EXACT_ATTN_STEPS", steps)
    monkeypatch.setenv("VLLM_OMNI_K6_EXACT_ATTN_BLOCKS", blocks)
    from diffusers.models.modeling_utils import no_init_weights

    from vllm_omni.diffusion.models.kandinsky6.kandinsky6_transformer import Kandinsky6Transformer3DModel

    config = dict(TINY_CONFIG, num_visual_blocks=num_blocks)
    with torch.device("meta"), no_init_weights():
        return Kandinsky6Transformer3DModel.from_diffusers_config(config)


def _visual_self(dit, index: int):
    return dit.visual_transformer_blocks[index].video_dec_block.self_attention


@pytest.fixture
def tp_group():
    class _Test:
        cleanups = []

        def addCleanup(self, fn):  # noqa: N802 - absltest-compatible shim
            self.cleanups.append(fn)

    holder = _Test()
    _init_single_rank(holder)
    yield
    for fn in holder.cleanups:
        fn()


def test_disabled_builds_no_twin(monkeypatch, tp_group) -> None:
    dit = _build(monkeypatch)
    for index in range(4):
        attention = _visual_self(dit, index)
        assert attention.attn_exact is None
        assert attention._attention_for_call() is attention.attn


def test_edge_blocks_are_pinned_exact(monkeypatch, tp_group) -> None:
    dit = _build(monkeypatch, blocks="1")
    pinned = [_visual_self(dit, i).always_exact for i in range(4)]
    assert pinned == [True, False, False, True]
    assert _visual_self(dit, 0)._attention_for_call() is _visual_self(dit, 0).attn_exact
    assert _visual_self(dit, 1)._attention_for_call() is _visual_self(dit, 1).attn


def test_step_flag_switches_the_middle_blocks(monkeypatch, tp_group) -> None:
    from vllm_omni.diffusion.models.kandinsky6.kandinsky6_transformer import set_exact_attention_step

    dit = _build(monkeypatch, steps="2")
    middle = _visual_self(dit, 1)
    assert middle._attention_for_call() is middle.attn
    set_exact_attention_step(dit, True)
    assert middle._attention_for_call() is middle.attn_exact
    set_exact_attention_step(dit, False)
    assert middle._attention_for_call() is middle.attn
    # Only the visual self-attention gets a twin.
    assert dit.visual_transformer_blocks[1].video_dec_block.cross_attention.attn_exact is None
