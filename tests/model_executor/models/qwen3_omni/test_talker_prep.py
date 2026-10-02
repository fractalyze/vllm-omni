# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""VLLM_OMNI_TALKER_PREP: the talker's prefill input for a text prompt, cut
on the host, is the GPU path's input exactly; other prompts keep the GPU
path. CPU-only, on a small stand-in talker.
"""

import pytest
import torch

from tests.model_executor.models.qwen3_omni.talker_stand_in import (
    ASSISTANT,
    AUDIO,
    IM_START,
    SYSTEM,
    THINKER_HIDDEN,
    USER,
    talker_model,
)
from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni import chat_segments

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

# system part, user part, then the assistant header and the first reply token.
_TEXT_PROMPT = [IM_START, SYSTEM, 11, 12, IM_START, USER, 21, 22, 23, IM_START, ASSISTANT, 198]
_FIRST_TOKEN = 31


def _inputs(prompt, result_ids):
    torch.manual_seed(1)
    rows = len(result_ids)
    return (
        torch.randn(rows, THINKER_HIDDEN).to(torch.bfloat16),
        torch.randn(rows, THINKER_HIDDEN).to(torch.bfloat16),
        torch.tensor([prompt]),
        torch.tensor(result_ids),
        4300,
        torch.randn(1, 1, THINKER_HIDDEN).to(torch.bfloat16),
        torch.randn(1, 1, THINKER_HIDDEN).to(torch.bfloat16),
        torch.randn(1, 1, THINKER_HIDDEN).to(torch.bfloat16),
    )


def _gpu_path(prompt, result_ids):
    embed, hidden, input_ids, result, speaker, bos, eos, pad = _inputs(prompt, result_ids)
    return talker_model(host_prep=False)._thinker_to_talker_prefill(
        embed, hidden, None, input_ids, result, speaker, bos, eos, pad
    )


def _host_path(prompt, result_ids):
    return talker_model(host_prep=True)._text_prompt_to_talker_prefill(*_inputs(prompt, result_ids))


def _assert_same(a, b):
    for x, y in zip(a, b, strict=True):
        assert x.dtype == y.dtype
        assert torch.equal(x, y)


def test_chat_segments():
    assert chat_segments(_TEXT_PROMPT, 13, IM_START) == [(0, 4, SYSTEM), (4, 9, USER), (9, 13, ASSISTANT)]


@pytest.mark.parametrize("reply", [[_FIRST_TOKEN], [_FIRST_TOKEN, 32, 33]])
def test_text_prompt_input_matches_the_gpu_path(reply):
    result_ids = _TEXT_PROMPT + reply
    host = _host_path(_TEXT_PROMPT, result_ids)
    assert host is not None
    _assert_same(host, _gpu_path(_TEXT_PROMPT, result_ids))


def test_the_switch_routes_text_prompts_to_the_host_path():
    embed, hidden, input_ids, result, speaker, bos, eos, pad = _inputs(_TEXT_PROMPT, _TEXT_PROMPT + [_FIRST_TOKEN])
    model = talker_model(host_prep=True)
    model._get_talker_user_parts = None  # the GPU path's helper must not run
    model._thinker_to_talker_prefill(embed, hidden, None, input_ids, result, speaker, bos, eos, pad)


def test_multimodal_prompts_take_the_gpu_path():
    prompt = [IM_START, USER, AUDIO, 21, IM_START, ASSISTANT, 198]
    assert _host_path(prompt, prompt + [_FIRST_TOKEN]) is None
