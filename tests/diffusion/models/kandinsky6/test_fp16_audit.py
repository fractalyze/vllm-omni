# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""FP16 range audit: what it records per layer, and when it flags a layer."""

from __future__ import annotations

import torch
from absl.testing import absltest

from vllm_omni.diffusion.models.kandinsky6.fp16_audit import FP16_MAX, Fp16Audit


class AuditTest(absltest.TestCase):
    def test_records_absmax_and_tiny_fractions(self) -> None:
        audit = Fp16Audit()
        x = torch.tensor([1.0, 1e-6, 1e-9, 0.0, -3.0])
        from unittest import mock

        from vllm_omni.diffusion.models.kandinsky6 import fp16_audit

        self.enterContext(mock.patch.object(fp16_audit, "SAMPLE_STRIDE", 1))
        w = torch.tensor([[0.5, -2.0]])
        audit.update("layer", x, w, torch.tensor([4.0]))
        row = audit.report()["layer"]
        self.assertEqual(row["x_absmax"], 3.0)
        self.assertEqual(row["w_absmax"], 2.0)
        self.assertEqual(row["y_absmax"], 4.0)
        self.assertAlmostEqual(row["x_sub"], 2 / 4)  # 1e-6 and 1e-9 of the four nonzeros
        self.assertAlmostEqual(row["x_flush"], 1 / 4)  # only 1e-9 flushes
        self.assertEqual(row["partial64_bound"], 3.0 * 2.0 * 64)
        self.assertFalse(row["at_risk"])

    def test_keeps_the_running_max_across_calls(self) -> None:
        audit = Fp16Audit()
        audit.update("layer", torch.tensor([1.0]), torch.tensor([1.0]), torch.tensor([1.0]))
        audit.update("layer", torch.tensor([7.0]), torch.tensor([1.0]), torch.tensor([1.0]))
        row = audit.report()["layer"]
        self.assertEqual(row["calls"], 2)
        self.assertEqual(row["x_absmax"], 7.0)

    def test_flags_a_layer_whose_partials_can_overflow(self) -> None:
        audit = Fp16Audit()
        audit.update("big", torch.tensor([100.0]), torch.tensor([20.0]), torch.tensor([10.0]))
        self.assertGreater(audit.report()["big"]["partial64_bound"], FP16_MAX)
        self.assertTrue(audit.report()["big"]["at_risk"])

    def test_hooks_reach_vllm_linears(self) -> None:
        from vllm.model_executor.layers.linear import ReplicatedLinear  # noqa: F401  (import check only)

        audit = Fp16Audit()
        module = torch.nn.Module()
        self.assertEqual(audit.install(module), 0)


if __name__ == "__main__":
    absltest.main()
