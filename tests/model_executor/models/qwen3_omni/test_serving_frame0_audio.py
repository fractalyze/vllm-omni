# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""VLLM_OMNI_FRAME0_AUDIO: the talker's chunk 0 reaches the API ahead of
code2wav's, and the API passes whichever arrives first and drops the other.
CPU-only; test_serving_frame0_audio_ipc.py shares weights on a GPU.
"""

import asyncio
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_omni.entrypoints.async_omni_base import AsyncOmniBase
from vllm_omni.model_executor.models.qwen3_omni.serving import frame0_audio
from vllm_omni.model_executor.models.qwen3_omni.serving.frame0 import TalkerFrame0
from vllm_omni.model_executor.models.qwen3_omni.serving.frame0_audio import (
    Frame0AudioReceiver,
    Frame0AudioSender,
    match_request,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_run_dir_follows_the_switch(monkeypatch, tmp_path):
    monkeypatch.setenv("VLLM_OMNI_QWEN3_OMNI_RUN_DIR", str(tmp_path / "run"))
    assert frame0_audio.run_dir() == str(tmp_path / "run")
    assert frame0_audio.socket_address() == f"ipc://{tmp_path}/run/qwen3_omni_frame0_audio.ipc"


def test_handles_encode_to_plain_values_and_back():
    shape = (torch.Tensor, torch.Size([3, 4]), (4, 1), 0, torch.storage.TypedStorage, torch.bfloat16)
    args = (*shape, 0, b"h", 24, 0, False, b"r", 7, b"e", True)

    values = frame0_audio._encode_handle(args)

    assert values[:4] == [[3, 4], [4, 1], 0, "bfloat16"]
    assert frame0_audio._decode_handle(values) == args


def test_only_plain_tensors_are_shared():
    args = (torch.nn.Parameter, torch.Size([1]), (1,), 0, torch.storage.TypedStorage, torch.float32)
    with pytest.raises(TypeError):
        frame0_audio._encode_handle(args)


@pytest.mark.parametrize(
    "talker_id,expected",
    [
        ("chatcmpl-1", "chatcmpl-1"),
        ("chatcmpl-1-abc", "chatcmpl-1"),
        ("chatcmpl-2", None),
    ],
)
def test_match_request(talker_id, expected):
    assert match_request({"chatcmpl-1": None, "chatcmpl-10": None}, talker_id) == expected


def test_switch_adds_the_sender_to_frame0s_listeners(monkeypatch):
    monkeypatch.delenv("VLLM_OMNI_FRAME0_AUDIO", raising=False)
    assert TalkerFrame0.with_listeners().listeners == []
    monkeypatch.setenv("VLLM_OMNI_FRAME0_AUDIO", "1")
    (listener,) = TalkerFrame0.with_listeners().listeners
    assert isinstance(listener, Frame0AudioSender)


def test_sender_waits_for_code2wavs_export(monkeypatch):
    monkeypatch.setattr(frame0_audio, "map_code2wav", lambda model: None)
    sender = Frame0AudioSender()
    sent = []
    monkeypatch.setattr(sender, "_send", lambda req_id, pcm: sent.append(req_id))
    sender(SimpleNamespace(model=object()), "r", torch.zeros(1, 16, dtype=torch.long))
    assert sent == []


def test_a_sender_failure_leaves_chunk0_to_code2wav(monkeypatch):
    def broken(model):
        raise RuntimeError("no IPC")

    monkeypatch.setattr(frame0_audio, "map_code2wav", broken)
    Frame0AudioSender()(SimpleNamespace(model=object()), "r", torch.zeros(1, 16, dtype=torch.long))


def _states():
    return {"chatcmpl-1": SimpleNamespace(queue=asyncio.Queue())}


def _stage_msg(finished=False, audio=True):
    output = SimpleNamespace(outputs=[SimpleNamespace(multimodal_output={"audio": 1} if audio else None)])
    return SimpleNamespace(request_id="chatcmpl-1", stage_id=2, finished=finished, engine_outputs=output)


def test_the_talkers_chunk0_goes_first_and_code2wavs_is_dropped():
    states = _states()
    receiver = Frame0AudioReceiver(states)

    assert receiver.deliver("chatcmpl-1-x", np.ones(4, dtype=np.float32))

    message = states["chatcmpl-1"].queue.get_nowait()
    assert message.stage_id == 2 and not message.finished
    audio = message.engine_outputs.outputs[0].multimodal_output["audio"]
    assert audio.shape == (1, 4)
    assert not receiver.keep_stage_output(_stage_msg())
    # Chunk 1 onwards passes.
    assert receiver.keep_stage_output(_stage_msg())


def test_code2wavs_chunk0_first_drops_the_talkers():
    states = _states()
    receiver = Frame0AudioReceiver(states)
    assert receiver.keep_stage_output(_stage_msg())
    assert not receiver.deliver("chatcmpl-1", np.ones(4, dtype=np.float32))
    assert states["chatcmpl-1"].queue.empty()


def test_a_one_chunk_reply_keeps_only_its_finish():
    receiver = Frame0AudioReceiver(_states())
    receiver.deliver("chatcmpl-1", np.ones(4, dtype=np.float32))
    last = _stage_msg(finished=True)
    assert receiver.keep_stage_output(last)
    assert last.engine_outputs.outputs[0].multimodal_output is None


@pytest.mark.parametrize("stage_id,kept", [(2, False), (1, True)])
def test_the_api_skips_code2wavs_dropped_chunk(monkeypatch, stage_id, kept):
    receiver = Frame0AudioReceiver(_states())
    receiver.deliver("chatcmpl-1", np.ones(4, dtype=np.float32))
    state = object()
    monkeypatch.setattr(
        "vllm_omni.entrypoints.omni_base.OmniBase._handle_output_message",
        lambda self, msg: (False, msg.request_id, stage_id, state),
    )
    api = object.__new__(AsyncOmniBase)
    api._frame0_audio = receiver

    result = api._handle_output_message(_stage_msg())

    assert result == ((False, "chatcmpl-1", stage_id, state) if kept else (True, None, None, None))
