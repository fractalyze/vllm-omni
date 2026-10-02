# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The VLLM_OMNI_CODE2WAV_STREAM_GRAPHS and VLLM_OMNI_CODE2WAV_COMPILE
switches: which graphs Qwen3-Omni's code2wav asks its CUDA graph wrapper
for. CPU-only; the wrapper is a stand-in that records its arguments.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

import vllm_omni.model_executor.models.qwen3_tts.cuda_graph_decoder_wrapper as wrapper_module
from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_code2wav import (
    Qwen3OmniMoeCode2Wav,
    streaming_capture_sizes,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize(
    "chunk,left,expected",
    [
        # The served config: 25-frame chunks with 25 frames of left context.
        (25, 25, [2, 4, 8, 16, 25, 32, 50]),
        (4, 0, [2, 4]),
        # No chunking config: powers of two up to 64.
        (0, 0, [2, 4, 8, 16, 32, 64]),
    ],
)
def test_streaming_capture_sizes(chunk, left, expected):
    assert streaming_capture_sizes(chunk, left) == expected


class _RecordingWrapper:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.capture_sizes = kwargs.get("capture_sizes")

    def warmup(self, device, **kwargs):
        self.warmup_kwargs = kwargs


@pytest.fixture
def code2wav(monkeypatch):
    monkeypatch.setattr(wrapper_module, "CUDAGraphDecoderWrapper", _RecordingWrapper)
    model = object.__new__(Qwen3OmniMoeCode2Wav)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(num_quantizers=16)
    return model


def _enable(code2wav) -> _RecordingWrapper:
    code2wav.enable_cudagraph(device=torch.device("cuda"), codec_chunk_frames=25, codec_left_context_frames=25)
    return code2wav._cudagraph_wrapper


def test_switches_off_keep_the_wrapper_defaults(monkeypatch, code2wav):
    monkeypatch.delenv("VLLM_OMNI_CODE2WAV_STREAM_GRAPHS", raising=False)
    monkeypatch.delenv("VLLM_OMNI_CODE2WAV_COMPILE", raising=False)
    wrapper = _enable(code2wav)
    assert wrapper.kwargs["capture_sizes"] is None
    assert wrapper.kwargs["compile_shapes"] is None
    assert wrapper.warmup_kwargs == {
        "dtype": torch.long,
        "codec_chunk_frames": 25,
        "codec_left_context_frames": 25,
    }


def test_stream_graphs_capture_streaming_sizes_only(monkeypatch, code2wav):
    monkeypatch.setenv("VLLM_OMNI_CODE2WAV_STREAM_GRAPHS", "1")
    assert _enable(code2wav).kwargs["capture_sizes"] == [2, 4, 8, 16, 25, 32, 50]


def test_compile_adds_the_first_chunk_shape(monkeypatch, code2wav):
    monkeypatch.setenv("VLLM_OMNI_CODE2WAV_COMPILE", "1")
    assert _enable(code2wav).kwargs["compile_shapes"] == [(1, 2)]
