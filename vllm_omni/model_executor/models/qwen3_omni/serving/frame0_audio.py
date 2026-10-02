# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# Copyright 2026 Fractalyze Inc. All rights reserved.
"""Ships Qwen3-Omni's first audio chunk from the talker's process
(VLLM_OMNI_FRAME0_AUDIO, with VLLM_OMNI_FRAME0).

With frame 0 shipped at the talker's prefill step, the codes still go to
code2wav's stage, which decodes them once its scheduler takes the chunk, and
the audio goes back to the API through the orchestrator: a stage hop and an
orchestrator round trip around a small decode. With the switch on:

- code2wav's stage, once its weights are loaded, moves them to memory CUDA
  IPC can share (the stages run with expandable segments, which it cannot)
  and writes their handles to the run directory. The stages have no memory
  for a second copy: the thinker's and the talker's KV caches fill theirs;
- the talker, at its first frame 0, maps those weights into a code2wav of its
  own with a CUDA graph for one frame (padded to the 2-frame bucket, as
  code2wav's own graphs are), so its decode replays the arithmetic code2wav's
  stage runs for that chunk. Each frame 0 is decoded there and pushed to the
  API process over a ZMQ socket in the run directory;
- the API puts each pushed chunk on its request's queue as code2wav's output,
  and of that and code2wav's own chunk 0 drops whichever comes second.
  Code2wav decodes each chunk statelessly from its codes and left context, so
  chunk 1 follows either one sample for sample.

The run directory is VLLM_OMNI_QWEN3_OMNI_RUN_DIR, or one per user under the
system temp directory; servers sharing a host need one each.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from collections.abc import Iterator
from typing import Any

import msgspec
import numpy as np
import torch
from torch import nn
from vllm.logger import init_logger

from vllm_omni import envs
from vllm_omni.platforms import current_omni_platform

logger = init_logger(__name__)

CODE2WAV_STAGE_ID = 2

# Which chunk 0 reached a request first.
_FROM_TALKER = "talker"
_FROM_CODE2WAV = "code2wav"


def run_dir() -> str:
    """The directory the stages and the API of one server share."""
    path = envs.VLLM_OMNI_QWEN3_OMNI_RUN_DIR or os.path.join(tempfile.gettempdir(), f"vllm-omni-{os.getuid()}")
    os.makedirs(path, exist_ok=True)
    return path


def socket_address() -> str:
    """The socket the talker pushes chunk 0 to and the API pulls from."""
    return f"ipc://{run_dir()}/qwen3_omni_frame0_audio.ipc"


def _handles_path() -> str:
    return os.path.join(run_dir(), "qwen3_omni_code2wav_ipc.msgpack")


def _encode_handle(args: tuple) -> list:
    """reduce_tensor's arguments for a CUDA tensor as plain values; the
    tensor and storage classes are fixed, the dtype goes by name."""
    tensor_cls, size, stride, offset, storage_cls, dtype, *rest = args
    if tensor_cls is not torch.Tensor or storage_cls is not torch.storage.TypedStorage:
        raise TypeError(f"cannot share a {tensor_cls.__name__} on a {storage_cls.__name__}")
    return [list(size), list(stride), offset, str(dtype).removeprefix("torch."), *rest]


def _decode_handle(values: list) -> tuple:
    """rebuild_cuda_tensor's arguments from _encode_handle's values."""
    size, stride, offset, dtype, *rest = values
    return (
        torch.Tensor,
        torch.Size(size),
        tuple(stride),
        offset,
        torch.storage.TypedStorage,
        getattr(torch, dtype),
        *rest,
    )


# ---------------------------------------------------------- code2wav stage


def _tensors(module: nn.Module) -> Iterator[tuple[nn.Module, str, str, str, torch.Tensor]]:
    """(owner, kind, leaf name, full name, tensor) of every parameter and
    buffer, persistent or not."""
    for prefix, owner in module.named_modules():
        for kind in ("_parameters", "_buffers"):
            for leaf, tensor in getattr(owner, kind).items():
                if tensor is not None:
                    yield owner, kind, leaf, f"{prefix}.{leaf}" if prefix else leaf, tensor


def share_tensors(module: nn.Module) -> dict[str, Any]:
    """Moves `module`'s CUDA tensors to memory CUDA IPC can share; their
    encoded handles by name."""
    from torch.multiprocessing.reductions import reduce_tensor

    expandable = "expandable_segments:True" in os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
    handles = {}
    if expandable:
        torch.cuda.memory._set_allocator_settings("expandable_segments:False")
    try:
        for owner, kind, leaf, name, tensor in list(_tensors(module)):
            if tensor.device.type != "cuda":
                continue
            moved = tensor.detach().clone()
            if kind == "_parameters":
                owner._parameters[leaf] = nn.Parameter(moved, requires_grad=False)
            else:
                owner._buffers[leaf] = moved
            handles[name] = _encode_handle(reduce_tensor(moved)[1])
    finally:
        if expandable:
            torch.cuda.memory._set_allocator_settings("expandable_segments:True")
    # The originals' memory goes back to the device, so the stage's footprint
    # stays what vLLM profiled.
    current_omni_platform.empty_cache()
    return handles


def export_code2wav(code2wav: nn.Module) -> None:
    """Shares code2wav's tensors and writes their handles to the run
    directory, replacing any a previous server left."""
    handles = share_tensors(code2wav)
    tmp = _handles_path() + ".tmp"
    with open(tmp, "wb") as f:
        f.write(msgspec.msgpack.encode(handles))
    os.replace(tmp, _handles_path())
    logger.info("VLLM_OMNI_FRAME0_AUDIO: code2wav's %d tensors shared with the talker", len(handles))


# ------------------------------------------------------------- talker stage


def map_tensors(module: nn.Module, handles: dict[str, Any]) -> torch.device | None:
    """Puts the tensors `handles` (export_code2wav's, by name) share into
    `module` (built on the meta device); their device."""
    from torch.multiprocessing.reductions import rebuild_cuda_tensor

    device = None
    for owner, kind, leaf, name, _ in list(_tensors(module)):
        if name not in handles:
            raise RuntimeError(f"no shared tensor for {name}")
        tensor = rebuild_cuda_tensor(*_decode_handle(handles[name]))
        device = tensor.device
        if kind == "_parameters":
            owner._parameters[leaf] = nn.Parameter(tensor, requires_grad=False)
        else:
            owner._buffers[leaf] = tensor
    return device


def map_code2wav(model) -> nn.Module | None:
    """Qwen3-Omni's code2wav on code2wav stage's own tensors, mapped over
    CUDA IPC, with a CUDA graph for a one-frame chunk; None until the stage
    has exported them. `model` is the talker stage's Qwen3-Omni model."""
    from vllm.model_executor.models.utils import init_vllm_registered_model
    from vllm.utils.torch_utils import set_default_torch_dtype

    from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_code2wav import (
        FIRST_CHUNK_FRAMES,
        Qwen3OmniMoeCode2Wav,
    )
    from vllm_omni.model_executor.models.qwen3_tts.cuda_graph_decoder_wrapper import CUDAGraphDecoderWrapper

    if not os.path.exists(_handles_path()):
        return None
    with open(_handles_path(), "rb") as f:
        handles = msgspec.msgpack.decode(f.read())
    config = model.code2wav_config
    vllm_config = model.vllm_config.with_hf_config(config, architectures=["Qwen3OmniMoeCode2Wav"])
    with set_default_torch_dtype(vllm_config.model_config.dtype), torch.device("meta"):
        code2wav = init_vllm_registered_model(
            vllm_config=vllm_config, prefix="code2wav", hf_config=config, architectures=["Qwen3OmniMoeCode2Wav"]
        )
    device = map_tensors(code2wav, handles)
    code2wav.precompute_snake_caches()
    code2wav.eval()
    wrapper = CUDAGraphDecoderWrapper(
        decoder=code2wav,
        capture_sizes=[FIRST_CHUNK_FRAMES],
        compile_shapes=Qwen3OmniMoeCode2Wav.first_chunk_compile_shapes(),
        num_quantizers=config.num_quantizers,
    )
    wrapper.warmup(device)
    code2wav._cudagraph_wrapper = wrapper
    code2wav._cudagraph_enabled = True
    return code2wav


def decode_frame0(code2wav, codes: torch.Tensor) -> np.ndarray:
    """Chunk 0's samples (float32) for frame 0's codes [1, quantizers], as
    code2wav's stage decodes a first chunk: no left context."""
    quantizers = code2wav.config.num_quantizers
    window = codes.reshape(1, quantizers, 1).to(torch.long)
    with torch.inference_mode():
        wav = code2wav.chunked_decode_streaming(window, [0], [quantizers])[0]
    return wav.reshape(-1).float().cpu().numpy()


class Frame0AudioSender:
    """The talker's frame-0 listener: decodes each frame 0 on code2wav's
    weights and pushes the samples to the API."""

    def __init__(self) -> None:
        self._code2wav: nn.Module | None = None
        self._push = None

    def _send(self, req_id: str, pcm: np.ndarray) -> None:
        import zmq

        if self._push is None:
            self._push = zmq.Context.instance().socket(zmq.PUSH)
            self._push.connect(socket_address())
        self._push.send_multipart([req_id.encode(), pcm.tobytes()], flags=zmq.NOBLOCK)

    def __call__(self, runner, req_id: str, codes: torch.Tensor) -> None:
        # A failure costs only the head start: code2wav's own chunk 0 follows.
        try:
            if self._code2wav is None:
                model = runner.model.unwrap() if hasattr(runner.model, "unwrap") else runner.model
                self._code2wav = map_code2wav(model)
                if self._code2wav is None:
                    return
                logger.info("VLLM_OMNI_FRAME0_AUDIO: the talker decodes chunk 0 on code2wav's weights")
            self._send(req_id, decode_frame0(self._code2wav, codes))
        except Exception:
            logger.exception("VLLM_OMNI_FRAME0_AUDIO: chunk 0 for %s failed", req_id)


# -------------------------------------------------------------- API process


def match_request(request_ids, talker_req_id: str) -> str | None:
    """The API's id for the talker's request: the same, or the longest that
    one of the two extends by a "-" suffix, as stages suffix the API's id."""
    if talker_req_id in request_ids:
        return talker_req_id
    found = [r for r in request_ids if talker_req_id.startswith(r + "-") or r.startswith(talker_req_id + "-")]
    return max(found, key=len) if found else None


def audio_message(req_id: str, pcm: np.ndarray):
    """An OutputMessage as code2wav's stage sends a chunk: `pcm` under the
    audio key of its first completion output."""
    from vllm.outputs import RequestOutput

    from vllm_omni.engine.messages import OutputMessage
    from vllm_omni.outputs.mm_outputs import MultimodalCompletionOutput, MultimodalPayload

    payload = MultimodalPayload.from_dict({"audio": torch.from_numpy(pcm.copy()).view(1, -1)})
    completion = MultimodalCompletionOutput(
        multimodal_output=payload, index=0, text="", token_ids=[], cumulative_logprob=None, logprobs=None
    )
    output = RequestOutput(
        request_id=req_id,
        prompt=None,
        prompt_token_ids=[],
        prompt_logprobs=None,
        outputs=[completion],
        finished=False,
    )
    return OutputMessage(request_id=req_id, stage_id=CODE2WAV_STAGE_ID, engine_outputs=output, finished=False)


class Frame0AudioReceiver:
    """The API's side: delivers the talker's chunk 0 and drops code2wav's
    when the talker's came first, or the talker's when code2wav's did."""

    stage_id = CODE2WAV_STAGE_ID

    def __init__(self, request_states: dict[str, Any]) -> None:
        self._request_states = request_states
        # API request id → which chunk 0 reached it first.
        self._first: dict[str, str] = {}
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.get_running_loop().create_task(self._listen())

    def deliver(self, talker_req_id: str, pcm: np.ndarray) -> bool:
        """Puts the talker's chunk 0 on its request's queue unless
        code2wav's came first. True if it went."""
        req_id = match_request(self._request_states, talker_req_id)
        if req_id is None or req_id in self._first:
            return False
        self._first[req_id] = _FROM_TALKER
        self._request_states[req_id].queue.put_nowait(audio_message(req_id, pcm))
        return True

    def keep_stage_output(self, msg) -> bool:
        """Whether code2wav's output `msg` goes on: its chunk 0 is dropped
        after the talker's, and a request's last message keeps only its
        finish."""
        first = self._first.get(msg.request_id)
        if msg.finished:
            self._first.pop(msg.request_id, None)
        elif first is None:
            self._first[msg.request_id] = _FROM_CODE2WAV
        if first != _FROM_TALKER:
            return True
        if not msg.finished:
            self._first[msg.request_id] = _FROM_CODE2WAV
            return False
        outputs = getattr(msg.engine_outputs, "outputs", None)
        if outputs:
            outputs[0].multimodal_output = None
        return True

    async def _listen(self) -> None:
        import zmq
        import zmq.asyncio

        pull = zmq.asyncio.Context.instance().socket(zmq.PULL)
        try:
            pull.bind(socket_address())
        except zmq.ZMQError:
            # A Unix socket path past ~107 bytes, or a second server on the
            # same run directory. Code2wav's own chunk 0 still serves.
            logger.exception("VLLM_OMNI_FRAME0_AUDIO: cannot listen on %s", socket_address())
            return
        logged = False
        while True:
            talker_req_id, pcm = await pull.recv_multipart()
            sent = self.deliver(talker_req_id.decode(), np.frombuffer(pcm, dtype=np.float32))
            if not logged:
                logged = True
                logger.info(
                    "VLLM_OMNI_FRAME0_AUDIO: first chunk 0 from the talker (%s) %s",
                    talker_req_id.decode(),
                    "delivered" if sent else "dropped",
                )
