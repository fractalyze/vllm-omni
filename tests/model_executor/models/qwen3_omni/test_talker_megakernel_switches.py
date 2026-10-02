# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The VLLM_OMNI_TALKER_MEGAKERNEL and VLLM_OMNI_CODE_PREDICTOR_MEGAKERNEL
switches, and the talker's and code predictor's forwards with the
megakernels off or declining a step: both must leave the stock forward
untouched. CPU-only; the kernels themselves are in test_talker_megakernel.py.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

import vllm_omni.model_executor.models.qwen3_omni.megakernel.talker as megakernel_talker
from vllm_omni import envs
from vllm_omni.model_executor.models.common.qwen3_code_predictor import CodePredictorWrapper
from vllm_omni.model_executor.models.qwen3_omni.megakernel.talker import TalkerMegakernel
from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_moe_code_predictor_mtp import (
    Qwen3OmniMoeTalkerCodePredictor,
)
from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_moe_talker import (
    Qwen3OmniMoeTalkerForConditionalGeneration,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_SWITCHES = (
    "VLLM_OMNI_TALKER_MEGAKERNEL",
    "VLLM_OMNI_TALKER_MEGAKERNEL_CTAS",
    "VLLM_OMNI_CODE_PREDICTOR_MEGAKERNEL",
    "VLLM_OMNI_CODE_PREDICTOR_MEGAKERNEL_CTAS",
)


def test_switches_default_off(monkeypatch):
    for name in _SWITCHES:
        monkeypatch.delenv(name, raising=False)
    assert envs.VLLM_OMNI_TALKER_MEGAKERNEL is False
    assert envs.VLLM_OMNI_CODE_PREDICTOR_MEGAKERNEL is False
    assert envs.VLLM_OMNI_TALKER_MEGAKERNEL_CTAS == 96
    assert envs.VLLM_OMNI_CODE_PREDICTOR_MEGAKERNEL_CTAS is None


@pytest.mark.parametrize("value,expected", [("1", True), ("0", False), ("true", False)])
def test_switch_is_on_only_at_one(monkeypatch, value, expected):
    monkeypatch.setenv("VLLM_OMNI_TALKER_MEGAKERNEL", value)
    monkeypatch.setenv("VLLM_OMNI_CODE_PREDICTOR_MEGAKERNEL", value)
    assert envs.VLLM_OMNI_TALKER_MEGAKERNEL is expected
    assert envs.VLLM_OMNI_CODE_PREDICTOR_MEGAKERNEL is expected


@pytest.mark.parametrize("value,talker,code_predictor", [("64", 64, 64), ("0", 96, None)])
def test_cta_caps(monkeypatch, value, talker, code_predictor):
    monkeypatch.setenv("VLLM_OMNI_TALKER_MEGAKERNEL_CTAS", value)
    monkeypatch.setenv("VLLM_OMNI_CODE_PREDICTOR_MEGAKERNEL_CTAS", value)
    assert envs.VLLM_OMNI_TALKER_MEGAKERNEL_CTAS == talker
    assert envs.VLLM_OMNI_CODE_PREDICTOR_MEGAKERNEL_CTAS == code_predictor


class _StockModel:
    def __init__(self):
        self.calls = 0

    def __call__(self, input_ids, positions, intermediate_tensors, inputs_embeds=None, **kwargs):
        self.calls += 1
        return inputs_embeds * 2, None


def _talker() -> Qwen3OmniMoeTalkerForConditionalGeneration:
    talker = object.__new__(Qwen3OmniMoeTalkerForConditionalGeneration)
    nn.Module.__init__(talker)
    talker.language_model = SimpleNamespace(model=_StockModel())
    return talker


def test_talker_switch_off_keeps_the_stock_forward():
    talker = _talker()
    assert talker.megakernel is None
    embeds = torch.ones(1, 4)
    out = talker.forward(None, torch.zeros(3, 1, dtype=torch.long), inputs_embeds=embeds)
    torch.testing.assert_close(out, embeds * 2)
    assert talker.language_model.model.calls == 1


def test_talker_steps_without_kv_caches_keep_the_stock_forward(monkeypatch):
    """The profile run has no attention metadata, so no step reaches the
    kernel and none is built."""
    monkeypatch.setattr(megakernel_talker, "_attn_metadata", lambda: None)
    talker = _talker()
    talker.megakernel = TalkerMegakernel(ctas=96)
    embeds = torch.ones(1, 4)
    out = talker.forward(None, torch.zeros(3, 1, dtype=torch.long), inputs_embeds=embeds)
    torch.testing.assert_close(out, embeds * 2)
    assert talker.language_model.model.calls == 1
    assert talker.megakernel._decoder is None


def test_talker_steps_with_extra_arguments_keep_the_stock_forward(monkeypatch):
    monkeypatch.setattr(megakernel_talker, "_attn_metadata", lambda: pytest.fail("the kernel path was consulted"))
    talker = _talker()
    talker.megakernel = TalkerMegakernel(ctas=96)
    talker.forward(None, torch.zeros(3, 1, dtype=torch.long), inputs_embeds=torch.ones(1, 4), extra=True)
    assert talker.language_model.model.calls == 1


def test_code_predictor_switch_off_keeps_the_stock_forward(monkeypatch):
    calls = []

    def stock_forward(self, *args):
        calls.append(args)
        return "stock"

    monkeypatch.setattr(CodePredictorWrapper, "forward", stock_forward)
    wrapper = object.__new__(Qwen3OmniMoeTalkerCodePredictor)
    nn.Module.__init__(wrapper)
    assert wrapper.megakernel is None
    args = (torch.zeros(1), torch.zeros(1, 4), torch.zeros(1, 1, 4))
    assert wrapper.forward(*args, sample_uniforms=torch.ones(1)) == "stock"
    assert len(calls) == 1 and calls[0][:3] == args and torch.equal(calls[0][-1], torch.ones(1))


def test_code_predictor_switch_off_loads_without_the_kernel(monkeypatch):
    monkeypatch.setattr(CodePredictorWrapper, "load_weights", lambda self, weights: {"loaded"})
    wrapper = object.__new__(Qwen3OmniMoeTalkerCodePredictor)
    nn.Module.__init__(wrapper)
    assert wrapper.load_weights([]) == {"loaded"}
    assert wrapper.megakernel is None
