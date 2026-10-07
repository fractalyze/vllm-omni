# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# Copyright 2026 Fractalyze Inc. All rights reserved.
"""JIT-builds the PR #48032 alignment extension (csrc/) against the running torch.

Same mechanism as the Qwen3-Omni megakernels: `torch.utils.cpp_extension`
with the CUDA toolkit torch was built against (`megakernel/_ext.py`).
"""

import functools
from pathlib import Path
from types import ModuleType

import torch

_SOURCE = Path(__file__).parent / "csrc" / "moe_align_stable.cu"
_CACHE_DIR = Path.home() / ".cache" / "vllm_omni" / "marlin_moe_align"


def _build_dir() -> Path:
    """One build directory per torch build and GPU architecture: ninja does
    not rebuild when either changes."""
    major, minor = torch.cuda.get_device_capability()
    return _CACHE_DIR / f"ext-{torch.__version__}-sm{major}{minor}"


@functools.cache
def load() -> ModuleType:
    from torch.utils import cpp_extension

    # Imported here: the patch installs this module while vllm_omni is still
    # importing, before the model registry behind that package can load.
    from vllm_omni.model_executor.models.qwen3_omni.megakernel._ext import _cuda_home, _link_dir

    cuda_home = _cuda_home()
    cpp_extension.CUDA_HOME = str(cuda_home)
    build_dir = _build_dir()
    build_dir.mkdir(parents=True, exist_ok=True)
    return cpp_extension.load(
        name="vllm_omni_marlin_moe_align",
        sources=[str(_SOURCE)],
        extra_cflags=["-O3"],
        # The kernels include CUB, whose CCCL headers refuse an nvcc newer
        # than the CUDA runtime headers beside it. The torch 2.13 + cu132
        # venv pairs nvcc 13.4 (nvidia-cuda-nvcc) with 13.2 runtime headers;
        # the kernels use nothing past 13.2, so the check is waived.
        extra_cuda_cflags=["-O3", "-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK"],
        extra_ldflags=[f"-L{_link_dir(cuda_home, build_dir)}"],
        build_directory=str(build_dir),
    )
