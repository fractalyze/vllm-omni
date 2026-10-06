# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Rank-local mmap staging accepts a dtype-casting transform, not a reshaping one.

A checkpoint may store some tensors wider than the runtime dtype: the published
Kandinsky 6 bundles keep 767 norms and embeddings (131 MB) in FP32 while the
module is built in BF16. The ordinary loader casts those on copy. Under rank-local
mmap a parameter is staged from its file-backed view, so the cast has to happen
in the staging transform -- and the staging slot has to be keyed by the dtype the
transform produces, or every packed source fails the layout check. That is what
let the 60.3 GB BF16 Pro DiT serve on a 59 GB host as the quality reference.
"""

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.model_loader.checkpoint_adapters.direct_mmap import _Kandinsky6DirectMmapAdapter
from vllm_omni.diffusion.offloader.distributed_layerwise_backend import DistributedLayerwiseOffloadHook

Hook = DistributedLayerwiseOffloadHook


def test_cast_transform_keys_the_slot_by_runtime_dtype() -> None:
    wide = nn.Parameter(torch.arange(6, dtype=torch.float32).reshape(2, 3), requires_grad=False)
    narrow = nn.Parameter(torch.ones(4, dtype=torch.bfloat16), requires_grad=False)
    cast = {id(wide): lambda t: t.to(torch.bfloat16)}

    sources, metadata = Hook._collect_mmap_sources({"wide": wide, "narrow": narrow}, {}, cast)

    assert set(metadata) == {torch.bfloat16}
    names = [m["name"] for m in metadata[torch.bfloat16]]
    assert names == ["wide", "narrow"]
    for source_info, meta in zip(sources[torch.bfloat16], metadata[torch.bfloat16], strict=True):
        staged = Hook._resolve_mmap_source(source_info, meta, torch.bfloat16)
        assert staged.dtype == torch.bfloat16
    staged_wide = Hook._resolve_mmap_source(sources[torch.bfloat16][0], metadata[torch.bfloat16][0], torch.bfloat16)
    torch.testing.assert_close(staged_wide, wide.to(torch.bfloat16))


def test_reshaping_transform_is_rejected() -> None:
    weight = nn.Parameter(torch.zeros(2, 3), requires_grad=False)
    with pytest.raises(ValueError, match="shape"):
        Hook._collect_mmap_sources({"w": weight}, {}, {id(weight): lambda t: t.reshape(3, 2)})


def test_kandinsky6_policy_casts_to_the_planned_dtype() -> None:
    planned = torch.empty(4, dtype=torch.bfloat16, device="meta")
    policy = _Kandinsky6DirectMmapAdapter().policy_for("transformer.x.weight", planned)
    assert policy.allow_custom_loader
    # The dtype is captured at planning time: the file-backed view that later
    # replaces the parameter carries the checkpoint's FP32.
    assert policy.transform(torch.zeros(4, dtype=torch.float32)).dtype == torch.bfloat16
