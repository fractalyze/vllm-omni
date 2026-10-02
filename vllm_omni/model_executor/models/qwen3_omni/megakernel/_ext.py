# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# Copyright 2026 Fractalyze Inc. All rights reserved.
"""JIT-builds the thinker megakernel extension (csrc/) against the running torch.

The extension must share a process with the torch that loads it, so it is
compiled with the CUDA toolkit that torch was built against. When `CUDA_HOME`
is unset, that toolkit is the `nvidia-cuda-nvcc` wheel installed beside torch
(`site-packages/nvidia/cu<major>`), not whatever nvcc the host has on its path.
"""

import functools
import os
from pathlib import Path
from types import ModuleType

_CSRC = Path(__file__).parent / "csrc"
_SOURCES = (
    "thinker_ops.cpp",
    "thinker_attention.cu",
    "thinker_decode.cu",
    "thinker_moe.cu",
    "thinker_prefill.cu",
)
# The kernels use sm_120a instructions (RTX 5090).
_GENCODE = "-gencode=arch=compute_120a,code=sm_120a"
_CACHE_DIR = Path.home() / ".cache" / "vllm_omni" / "qwen3_omni_megakernel"


def _cuda_home() -> Path:
    if "CUDA_HOME" in os.environ:
        return Path(os.environ["CUDA_HOME"])
    import nvidia
    import torch

    major = torch.version.cuda.split(".")[0]
    for root in nvidia.__path__:
        home = Path(root) / f"cu{major}"
        if (home / "bin" / "nvcc").exists():
            return home
    raise RuntimeError(f"no CUDA {major} nvcc found; install nvidia-cuda-nvcc for CUDA {major} or set CUDA_HOME")


def _link_dir(cuda_home: Path, build_dir: Path) -> Path:
    """Returns a directory holding the unversioned libcudart.so the linker needs.

    The toolkit wheels ship only libcudart.so.<major>, and the extension build
    links with -lcudart.
    """
    lib = cuda_home / "lib"
    if (lib / "libcudart.so").exists():
        return lib
    link_dir = build_dir / "lib"
    link_dir.mkdir(parents=True, exist_ok=True)
    versioned = sorted(lib.glob("libcudart.so.*"))[0]
    link = link_dir / "libcudart.so"
    if not link.exists():
        link.symlink_to(versioned)
    return link_dir


def _build_dir() -> Path:
    """One build directory per torch build: an extension built against another
    torch in the same directory would load, and fail on its ABI."""
    import torch

    return _CACHE_DIR / f"ext-{torch.__version__}"


@functools.cache
def load() -> ModuleType:
    from torch.utils import cpp_extension

    cuda_home = _cuda_home()
    # cpp_extension resolves its toolkit once, at import, and vLLM imports it
    # long before this runs; point it at the toolkit matching torch.
    cpp_extension.CUDA_HOME = str(cuda_home)
    build_dir = _build_dir()
    build_dir.mkdir(parents=True, exist_ok=True)
    return cpp_extension.load(
        name="vllm_omni_qwen3_omni_megakernel",
        sources=[str(_CSRC / name) for name in _SOURCES],
        extra_cflags=["-O3"],
        # -lineinfo only maps SASS to source lines for ncu; codegen is unchanged.
        extra_cuda_cflags=["-O3", "-lineinfo", _GENCODE],
        extra_ldflags=[f"-L{_link_dir(cuda_home, build_dir)}"],
        build_directory=str(build_dir),
    )
