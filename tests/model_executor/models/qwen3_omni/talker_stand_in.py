# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""A small CPU stand-in for Qwen3-Omni's talker stage: the model with a
talker of real (tiny) projections and codec embedding, for tests of the
talker's prefill input."""

from types import SimpleNamespace

import torch
from torch import nn

from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni import Qwen3OmniMoeForConditionalGeneration

THINKER_HIDDEN = 8
TALKER_HIDDEN = 6
IM_START, SYSTEM, USER, ASSISTANT = 151644, 8948, 872, 77091
AUDIO, IMAGE, VIDEO = 151675, 151655, 151656


class _Talker(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.text_projection = nn.Linear(THINKER_HIDDEN, TALKER_HIDDEN).to(torch.bfloat16)
        self.hidden_projection = nn.Linear(THINKER_HIDDEN, TALKER_HIDDEN).to(torch.bfloat16)
        self.codec_embedding = nn.Embedding(5000, TALKER_HIDDEN).to(torch.bfloat16)

    def embed_input_ids(self, ids):
        return self.codec_embedding(ids)


def talker_model(*, host_prep: bool = False, preprefill: bool = False) -> Qwen3OmniMoeForConditionalGeneration:
    model = object.__new__(Qwen3OmniMoeForConditionalGeneration)
    nn.Module.__init__(model)
    model.talker = _Talker()
    model.thinker_config = SimpleNamespace(audio_token_id=AUDIO, image_token_id=IMAGE, video_token_id=VIDEO)
    model.config = SimpleNamespace(
        im_start_token_id=IM_START,
        system_token_id=SYSTEM,
        user_token_id=USER,
        assistant_token_id=ASSISTANT,
        tts_pad_token_id=151671,
        talker_config=SimpleNamespace(
            text_config=SimpleNamespace(hidden_size=TALKER_HIDDEN),
            codec_nothink_id=4203,
            codec_think_bos_id=4204,
            codec_think_eos_id=4205,
            codec_pad_id=4196,
            codec_bos_id=4197,
        ),
    )
    model._host_talker_prep = host_prep
    model._talker_preprefill = preprefill
    model.tts_text_spk_token_ids = {"ethan": 4300}
    model.default_tts_text_spk_type = "ethan"
    return model
