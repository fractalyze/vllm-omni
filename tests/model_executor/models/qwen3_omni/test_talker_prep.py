# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""VLLM_OMNI_TALKER_PREP: the talker's prefill input for a text prompt, cut
on the host, is the GPU path's input exactly; other prompts keep the GPU
path. CPU-only, on a small stand-in talker.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni import (
    Qwen3OmniMoeForConditionalGeneration,
    chat_segments,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_THINKER_HIDDEN = 8
_TALKER_HIDDEN = 6
_IM_START, _SYSTEM, _USER, _ASSISTANT = 151644, 8948, 872, 77091
_AUDIO, _IMAGE, _VIDEO = 151675, 151655, 151656


class _Talker(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.text_projection = nn.Linear(_THINKER_HIDDEN, _TALKER_HIDDEN).to(torch.bfloat16)
        self.hidden_projection = nn.Linear(_THINKER_HIDDEN, _TALKER_HIDDEN).to(torch.bfloat16)
        self.codec_embedding = nn.Embedding(5000, _TALKER_HIDDEN).to(torch.bfloat16)

    def embed_input_ids(self, ids):
        return self.codec_embedding(ids)


def _model(host_prep: bool) -> Qwen3OmniMoeForConditionalGeneration:
    model = object.__new__(Qwen3OmniMoeForConditionalGeneration)
    nn.Module.__init__(model)
    model.talker = _Talker()
    model.thinker_config = SimpleNamespace(audio_token_id=_AUDIO, image_token_id=_IMAGE, video_token_id=_VIDEO)
    model.config = SimpleNamespace(
        im_start_token_id=_IM_START,
        system_token_id=_SYSTEM,
        user_token_id=_USER,
        assistant_token_id=_ASSISTANT,
        tts_pad_token_id=151671,
        talker_config=SimpleNamespace(
            text_config=SimpleNamespace(hidden_size=_TALKER_HIDDEN),
            codec_nothink_id=4203,
            codec_think_bos_id=4204,
            codec_think_eos_id=4205,
            codec_pad_id=4196,
            codec_bos_id=4197,
        ),
    )
    model._host_talker_prep = host_prep
    return model


# system part, user part, then the assistant header and the first reply token.
_TEXT_PROMPT = [_IM_START, _SYSTEM, 11, 12, _IM_START, _USER, 21, 22, 23, _IM_START, _ASSISTANT, 198]
_FIRST_TOKEN = 31


def _inputs(prompt, result_ids):
    torch.manual_seed(1)
    rows = len(result_ids)
    return (
        torch.randn(rows, _THINKER_HIDDEN).to(torch.bfloat16),
        torch.randn(rows, _THINKER_HIDDEN).to(torch.bfloat16),
        torch.tensor([prompt]),
        torch.tensor(result_ids),
        4300,
        torch.randn(1, 1, _THINKER_HIDDEN).to(torch.bfloat16),
        torch.randn(1, 1, _THINKER_HIDDEN).to(torch.bfloat16),
        torch.randn(1, 1, _THINKER_HIDDEN).to(torch.bfloat16),
    )


def _gpu_path(prompt, result_ids):
    embed, hidden, input_ids, result, speaker, bos, eos, pad = _inputs(prompt, result_ids)
    return _model(False)._thinker_to_talker_prefill(embed, hidden, None, input_ids, result, speaker, bos, eos, pad)


def _host_path(prompt, result_ids):
    return _model(True)._text_prompt_to_talker_prefill(*_inputs(prompt, result_ids))


def _assert_same(a, b):
    for x, y in zip(a, b, strict=True):
        assert x.dtype == y.dtype
        assert torch.equal(x, y)


def test_chat_segments():
    assert chat_segments(_TEXT_PROMPT, 13, _IM_START) == [(0, 4, _SYSTEM), (4, 9, _USER), (9, 13, _ASSISTANT)]


@pytest.mark.parametrize("reply", [[_FIRST_TOKEN], [_FIRST_TOKEN, 32, 33]])
def test_text_prompt_input_matches_the_gpu_path(reply):
    result_ids = _TEXT_PROMPT + reply
    host = _host_path(_TEXT_PROMPT, result_ids)
    assert host is not None
    _assert_same(host, _gpu_path(_TEXT_PROMPT, result_ids))


def test_the_switch_routes_text_prompts_to_the_host_path():
    embed, hidden, input_ids, result, speaker, bos, eos, pad = _inputs(_TEXT_PROMPT, _TEXT_PROMPT + [_FIRST_TOKEN])
    model = _model(True)
    model._get_talker_user_parts = None  # the GPU path's helper must not run
    model._thinker_to_talker_prefill(embed, hidden, None, input_ids, result, speaker, bos, eos, pad)


def test_multimodal_prompts_take_the_gpu_path():
    prompt = [_IM_START, _USER, _AUDIO, 21, _IM_START, _ASSISTANT, 198]
    assert _host_path(prompt, prompt + [_FIRST_TOKEN]) is None
