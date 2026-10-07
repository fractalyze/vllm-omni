# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Staging a streamed weight in another dtype of the same width (``dlo_stage_weight_dtype``)."""

from __future__ import annotations

import torch
from absl.testing import absltest

from vllm_omni.diffusion.offloader.distributed_layerwise_backend import (
    STAGE_WEIGHT_DTYPE_ATTR,
    apply_stage_casts,
    stage_dtype_for,
)


def _block() -> torch.nn.Module:
    block = torch.nn.Module()
    block.attn = torch.nn.Module()
    block.attn.proj = torch.nn.Linear(4, 4, bias=True)
    block.norm = torch.nn.LayerNorm(4)
    return block


class StageDtypeForTest(absltest.TestCase):
    def test_unflagged_keeps_its_dtype(self) -> None:
        self.assertIsNone(stage_dtype_for(_block(), "attn.proj.weight", torch.bfloat16))

    def test_flagged_weight_only(self) -> None:
        block = _block()
        setattr(block.attn.proj, STAGE_WEIGHT_DTYPE_ATTR, torch.float16)
        self.assertEqual(stage_dtype_for(block, "attn.proj.weight", torch.bfloat16), torch.float16)
        self.assertIsNone(stage_dtype_for(block, "attn.proj.bias", torch.bfloat16))
        self.assertIsNone(stage_dtype_for(block, "norm.weight", torch.bfloat16))

    def test_same_dtype_is_a_no_op(self) -> None:
        block = _block()
        setattr(block.attn.proj, STAGE_WEIGHT_DTYPE_ATTR, torch.float16)
        self.assertIsNone(stage_dtype_for(block, "attn.proj.weight", torch.float16))

    def test_width_change_is_refused(self) -> None:
        block = _block()
        setattr(block.attn.proj, STAGE_WEIGHT_DTYPE_ATTR, torch.float32)
        with self.assertRaises(ValueError):
            stage_dtype_for(block, "attn.proj.weight", torch.bfloat16)


class ApplyStageCastsTest(absltest.TestCase):
    def test_converts_only_the_listed_slices_in_place(self) -> None:
        values = (torch.randn(64) * 3).to(torch.bfloat16)
        buffer = values.clone()
        apply_stage_casts({torch.bfloat16: buffer}, ((torch.bfloat16, 16, 32, torch.float16),))
        self.assertTrue(torch.equal(buffer[16:48].view(torch.float16), values[16:48].to(torch.float16)))
        self.assertTrue(torch.equal(buffer[:16], values[:16]))
        self.assertTrue(torch.equal(buffer[48:], values[48:]))


if __name__ == "__main__":
    absltest.main()
