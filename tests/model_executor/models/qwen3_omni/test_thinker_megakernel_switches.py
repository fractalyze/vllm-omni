# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The VLLM_OMNI_THINKER_MEGAKERNEL switches, and the thinker's forward with
the megakernel off or declining a step: both must leave the stock forward
untouched. CPU-only; the kernels themselves are in test_thinker_megakernel.py.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

import vllm_omni.model_executor.models.qwen3_omni.megakernel.thinker as megakernel_thinker
import vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_moe_thinker as thinker_module
from vllm_omni import envs
from vllm_omni.model_executor.models.qwen3_omni.megakernel.thinker import ThinkerMegakernel
from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_moe_thinker import (
    Qwen3OmniMoeThinkerForConditionalGeneration,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_SWITCHES = (
    "VLLM_OMNI_THINKER_MEGAKERNEL",
    "VLLM_OMNI_THINKER_MEGAKERNEL_CTAS",
    "VLLM_OMNI_THINKER_MEGAKERNEL_PREFILL",
    "VLLM_OMNI_THINKER_MEGAKERNEL_PREFILL_CTAS",
)


def test_switches_default_off(monkeypatch):
    for name in _SWITCHES:
        monkeypatch.delenv(name, raising=False)
    assert envs.VLLM_OMNI_THINKER_MEGAKERNEL is False
    assert envs.VLLM_OMNI_THINKER_MEGAKERNEL_PREFILL is False
    assert envs.VLLM_OMNI_THINKER_MEGAKERNEL_CTAS is None
    assert envs.VLLM_OMNI_THINKER_MEGAKERNEL_PREFILL_CTAS is None


@pytest.mark.parametrize("value,expected", [("1", True), ("0", False), ("true", False)])
def test_switch_is_on_only_at_one(monkeypatch, value, expected):
    monkeypatch.setenv("VLLM_OMNI_THINKER_MEGAKERNEL", value)
    monkeypatch.setenv("VLLM_OMNI_THINKER_MEGAKERNEL_PREFILL", value)
    assert envs.VLLM_OMNI_THINKER_MEGAKERNEL is expected
    assert envs.VLLM_OMNI_THINKER_MEGAKERNEL_PREFILL is expected


@pytest.mark.parametrize("value,expected", [("64", 64), ("0", None)])
def test_cta_caps(monkeypatch, value, expected):
    monkeypatch.setenv("VLLM_OMNI_THINKER_MEGAKERNEL_CTAS", value)
    monkeypatch.setenv("VLLM_OMNI_THINKER_MEGAKERNEL_PREFILL_CTAS", value)
    assert envs.VLLM_OMNI_THINKER_MEGAKERNEL_CTAS == expected
    assert envs.VLLM_OMNI_THINKER_MEGAKERNEL_PREFILL_CTAS == expected


class _StockModel:
    def __init__(self):
        self.calls = 0

    def __call__(self, input_ids, positions, intermediate_tensors, inputs_embeds=None, **kwargs):
        self.calls += 1
        return inputs_embeds * 2, None


def _thinker() -> Qwen3OmniMoeThinkerForConditionalGeneration:
    thinker = object.__new__(Qwen3OmniMoeThinkerForConditionalGeneration)
    nn.Module.__init__(thinker)
    thinker.use_deepstack = False
    thinker.language_model = SimpleNamespace(model=_StockModel())
    return thinker


@pytest.fixture(autouse=True)
def first_rank(monkeypatch):
    monkeypatch.setattr(thinker_module, "get_pp_group", lambda: SimpleNamespace(is_first_rank=True))


def test_switch_off_keeps_the_stock_forward():
    thinker = _thinker()
    assert thinker.megakernel is None
    embeds = torch.ones(1, 4)
    out = thinker.forward(None, torch.zeros(3, 1, dtype=torch.long), inputs_embeds=embeds)
    torch.testing.assert_close(out, embeds * 2)
    assert thinker.language_model.model.calls == 1


@pytest.mark.parametrize("tokens", [1, 8])
def test_steps_without_kv_caches_keep_the_stock_forward(monkeypatch, tokens):
    """The profile run has no attention metadata, so no step reaches a kernel
    and none is built."""
    monkeypatch.setattr(megakernel_thinker, "_attn_metadata", lambda: None)
    thinker = _thinker()
    thinker.megakernel = ThinkerMegakernel(decode_ctas=None, prefill=True, prefill_ctas=None)
    embeds = torch.ones(tokens, 4)
    out = thinker.forward(None, torch.zeros(3, tokens, dtype=torch.long), inputs_embeds=embeds)
    torch.testing.assert_close(out, embeds * 2)
    assert thinker.language_model.model.calls == 1
