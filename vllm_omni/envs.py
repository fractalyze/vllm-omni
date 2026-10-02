# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Environment switches read by vLLM-Omni's model executors.

Each entry is read when its attribute is accessed (`envs.NAME`); a consumer
reads a switch once, when it is constructed, and keeps the value.
"""

import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    VLLM_OMNI_THINKER_MEGAKERNEL: bool = False
    VLLM_OMNI_THINKER_MEGAKERNEL_CTAS: int | None = None
    VLLM_OMNI_THINKER_MEGAKERNEL_PREFILL: bool = False
    VLLM_OMNI_THINKER_MEGAKERNEL_PREFILL_CTAS: int | None = None
    VLLM_OMNI_TALKER_MEGAKERNEL: bool = False
    VLLM_OMNI_TALKER_MEGAKERNEL_CTAS: int = 96
    VLLM_OMNI_CODE_PREDICTOR_MEGAKERNEL: bool = False
    VLLM_OMNI_CODE_PREDICTOR_MEGAKERNEL_CTAS: int | None = None
    VLLM_OMNI_DETERMINISTIC_MARLIN: bool = False
    VLLM_OMNI_CODE2WAV_STREAM_GRAPHS: bool = False
    VLLM_OMNI_CODE2WAV_COMPILE: bool = False
    VLLM_OMNI_FRAME0: bool = False


def _ctas(name: str) -> int | None:
    """A CTA cap: unset or 0 means one CTA per SM."""
    return int(os.environ.get(name, "0")) or None


environment_variables: dict[str, Callable[[], Any]] = {
    # "1" runs Qwen3-Omni's thinker decode steps (one token, one request) on
    # the decode megakernel, one launch a step
    # (model_executor/models/qwen3_omni/megakernel/). Needs the thinker's MoE
    # on moe_backend triton and an sm_120a GPU.
    "VLLM_OMNI_THINKER_MEGAKERNEL": lambda: os.environ.get("VLLM_OMNI_THINKER_MEGAKERNEL", "0") == "1",
    # How many CTAs (SMs) a decode step takes; the rest stay free for the
    # stages that share the GPU. Unset or 0: every SM.
    "VLLM_OMNI_THINKER_MEGAKERNEL_CTAS": lambda: _ctas("VLLM_OMNI_THINKER_MEGAKERNEL_CTAS"),
    # With VLLM_OMNI_THINKER_MEGAKERNEL=1, "1" also runs one request's prompt
    # chunks of 2 to 64 tokens on the prefill megakernel, one launch a chunk.
    "VLLM_OMNI_THINKER_MEGAKERNEL_PREFILL": lambda: os.environ.get("VLLM_OMNI_THINKER_MEGAKERNEL_PREFILL", "0") == "1",
    # How many CTAs (SMs) a prefill chunk takes. Unset or 0: every SM.
    "VLLM_OMNI_THINKER_MEGAKERNEL_PREFILL_CTAS": lambda: _ctas("VLLM_OMNI_THINKER_MEGAKERNEL_PREFILL_CTAS"),
    # "1" runs Qwen3-Omni's talker decode steps (one token, one request) on the
    # talker megakernel, one launch a step. Needs an sm_120a GPU.
    "VLLM_OMNI_TALKER_MEGAKERNEL": lambda: os.environ.get("VLLM_OMNI_TALKER_MEGAKERNEL", "0") == "1",
    # How many CTAs (SMs) a talker step takes. Unset or 0: 96, which leaves
    # the thinker's decode its SMs when the stages share the GPU under MPS.
    "VLLM_OMNI_TALKER_MEGAKERNEL_CTAS": lambda: _ctas("VLLM_OMNI_TALKER_MEGAKERNEL_CTAS") or 96,
    # "1" runs Qwen3-Omni's code predictor (codes 1 to 15 of each audio frame)
    # on the code-predictor megakernel, one launch a call. Needs an sm_120a GPU.
    "VLLM_OMNI_CODE_PREDICTOR_MEGAKERNEL": lambda: os.environ.get("VLLM_OMNI_CODE_PREDICTOR_MEGAKERNEL", "0") == "1",
    # How many CTAs (SMs) a code-predictor launch takes; the rest stay free
    # for code2wav. Unset or 0: every SM.
    "VLLM_OMNI_CODE_PREDICTOR_MEGAKERNEL_CTAS": lambda: _ctas("VLLM_OMNI_CODE_PREDICTOR_MEGAKERNEL_CTAS"),
    # "1" makes vLLM's Marlin MoE give the same bits across identical
    # requests: each expert's rows are sorted by row before the prefill's
    # grouped GEMM (vllm_omni/patch.py). Read once, when vllm_omni is imported.
    "VLLM_OMNI_DETERMINISTIC_MARLIN": lambda: os.environ.get("VLLM_OMNI_DETERMINISTIC_MARLIN", "0") == "1",
    # "1" captures Qwen3-Omni code2wav's CUDA graphs only at the frame counts
    # a streaming (async_chunk) decode uses, instead of every size up to a
    # non-streaming decode's. Needs the stage's enforce_eager off.
    "VLLM_OMNI_CODE2WAV_STREAM_GRAPHS": lambda: os.environ.get("VLLM_OMNI_CODE2WAV_STREAM_GRAPHS", "0") == "1",
    # "1" decodes Qwen3-Omni code2wav's one-frame first chunk on torch.compile
    # (with a CUDA graph) instead of a plain CUDA graph. Inductor's kernels
    # differ from eager ones, so the first chunk's samples change slightly.
    "VLLM_OMNI_CODE2WAV_COMPILE": lambda: os.environ.get("VLLM_OMNI_CODE2WAV_COMPILE", "0") == "1",
    # "1" ships each Qwen3-Omni talker request's first audio frame with its
    # prefill step instead of after its first decode step
    # (model_executor/models/qwen3_omni/serving/frame0.py).
    "VLLM_OMNI_FRAME0": lambda: os.environ.get("VLLM_OMNI_FRAME0", "0") == "1",
}


def __getattr__(name: str):
    if name in environment_variables:
        return environment_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(environment_variables.keys())
