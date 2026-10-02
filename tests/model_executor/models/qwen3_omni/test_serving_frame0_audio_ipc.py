# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""VLLM_OMNI_FRAME0_AUDIO's weight sharing: one process exports a module's
CUDA tensors and another maps them into a copy built on the meta device,
reading the same values without a second copy.
"""

import multiprocessing

import pytest
import torch
from torch import nn

from vllm_omni.model_executor.models.qwen3_omni.serving import frame0_audio

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA IPC needs a GPU"),
]


def _module() -> nn.Module:
    module = nn.Sequential(nn.Linear(4, 3), nn.LayerNorm(3))
    module.register_buffer("scale", torch.arange(3.0))
    return module


def _expected() -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    return {name: tensor.detach().clone() for name, tensor in _module().state_dict().items()}


def _export(run_dir: str, exported, release) -> None:
    import os

    os.environ["VLLM_OMNI_QWEN3_OMNI_RUN_DIR"] = run_dir
    torch.manual_seed(0)
    module = _module().to(torch.bfloat16).cuda()
    frame0_audio.export_code2wav(module)
    exported.set()
    # The exporter owns the memory: it must outlive the mapping.
    release.wait(60)


def test_a_mapped_copy_reads_the_exported_tensors(monkeypatch, tmp_path):
    monkeypatch.setenv("VLLM_OMNI_QWEN3_OMNI_RUN_DIR", str(tmp_path))
    spawn = multiprocessing.get_context("spawn")
    exported, release = spawn.Event(), spawn.Event()
    exporter = spawn.Process(target=_export, args=(str(tmp_path), exported, release))
    exporter.start()
    try:
        assert exported.wait(120)
        import msgspec

        handles = msgspec.msgpack.decode((tmp_path / "qwen3_omni_code2wav_ipc.msgpack").read_bytes())
        with torch.device("meta"):
            mapped = _module().to(torch.bfloat16)

        device = frame0_audio.map_tensors(mapped, handles)

        assert device.type == "cuda"
        for name, tensor in mapped.state_dict().items():
            assert torch.equal(tensor.cpu(), _expected()[name].to(torch.bfloat16)), name
    finally:
        release.set()
        exporter.join(60)
