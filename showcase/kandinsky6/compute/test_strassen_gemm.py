# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""One-level Strassen over the hybrid GEMM: correctness and the error budget.

R2 was rejected on speed (0.946-0.962x the hybrid kernel, against a gate of >5%
faster), so none of this is on a serving path. The tests exist because the
*measurement* is the deliverable: a Strassen that silently computed something
else would have produced a timing number for the wrong arithmetic, and the error
figure reported to the other tracks has to be trustworthy.
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

import torch
from absl.testing import absltest, parameterized

sys.path.insert(0, str(Path(__file__).resolve().parent))

_CKPT = sorted(glob.glob(
    "/data/jooman/hf/hub/models--kandinskylab--Kandinsky-6.0-Pro-distill-5s-Diffusers"
    "/snapshots/*/transformer/*.safetensors"))


def _rel_l2(got: torch.Tensor, ref: torch.Tensor) -> float:
    return ((got.double() - ref).norm() / ref.norm()).item()


def _load(name: str) -> torch.Tensor | None:
    from safetensors import safe_open

    for shard in _CKPT:
        with safe_open(shard, framework="pt") as f:
            if name in f.keys():
                return f.get_tensor(name)
    return None


class StrassenCorrectnessTest(parameterized.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if not torch.cuda.is_available():
            raise absltest.SkipTest("needs a GPU")

    @parameterized.named_parameters(
        ("materialised", False),
        ("prologue_fused", True),
    )
    def test_both_arrangements_agree_with_a_plain_matmul(self, fuse_sums):
        """Strassen's seven products must reassemble into the actual product.

        The failure this guards is specific and quiet: a wrong sign in one of
        the eighteen additions still returns a correctly-shaped, finite,
        plausible-looking matrix. It would have been timed happily.
        """
        from strassen_gemm import strassen_matmul

        torch.manual_seed(0)
        for M, K, N in ((256, 128, 192), (512, 256, 384), (1024, 512, 256)):
            x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
            w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.05
            ref = x.double() @ w.double().t()
            got = strassen_matmul(x, w, fuse_sums=fuse_sums)
            self.assertEqual(got.shape, (M, N))
            rel = _rel_l2(got, ref)
            self.assertLess(rel, 5e-3, f"{M}x{K}x{N} fuse_sums={fuse_sums}: rel L2 {rel:.3e}")

    def test_the_two_arrangements_match_each_other(self):
        """They are the same arithmetic in a different order, so they should
        land within rounding of one another. If they diverge, one of them has a
        sign or an index wrong and the other is the reference."""
        from strassen_gemm import strassen_matmul

        torch.manual_seed(0)
        x = torch.randn(512, 256, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(384, 256, device="cuda", dtype=torch.bfloat16) * 0.05
        a = strassen_matmul(x, w, fuse_sums=False)
        b = strassen_matmul(x, w, fuse_sums=True)
        torch.testing.assert_close(a, b, rtol=5e-3, atol=5e-3)

    def test_error_stays_within_the_hybrid_path_s_own(self):
        """The point of the error measurement reported to the other tracks.

        Strassen is famously less stable than the plain algorithm, and the brief
        expected 2-3x the hybrid's error. On real checkpoint weights it is
        **1.05x** -- because at BF16 input precision the error floor is set by
        input quantisation, not by the reduction order. A regression here would
        mean that reasoning has stopped holding.
        """
        from hybrid_gemm import hybrid_matmul
        from strassen_gemm import strassen_matmul

        if not _CKPT:
            raise absltest.SkipTest("needs the W1 checkpoint for real weights")
        w = _load("visual_transformer_blocks.0.video_dec_block.feed_forward.net.0.proj.weight")
        self.assertIsNotNone(w)
        w = w.cuda()
        torch.manual_seed(0)
        x = torch.randn(1024, w.shape[1], device="cuda", dtype=torch.float32)
        ref = x.double() @ w.double().t()
        xb, wb = x.to(torch.bfloat16), w.to(torch.bfloat16)

        e_hyb = _rel_l2(hybrid_matmul(xb, wb), ref)
        e_str = _rel_l2(strassen_matmul(xb, wb), ref)
        print(f"\n  real weights: hybrid {e_hyb:.3e}  strassen {e_str:.3e}  "
              f"ratio {e_str / e_hyb:.2f}x")
        self.assertLess(e_str, 1.5 * e_hyb,
                        f"strassen {e_str:.3e} is more than 1.5x the hybrid's {e_hyb:.3e}; "
                        f"the input-quantisation argument no longer holds")

    def test_odd_dimensions_are_refused_rather_than_truncated(self):
        """W1's shapes are all even, but a silently dropped row would be a
        wrong answer rather than an error, so the check is explicit."""
        from strassen_gemm import strassen_matmul

        w = torch.randn(192, 128, device="cuda", dtype=torch.bfloat16) * 0.05
        for bad in ((255, 128), (256, 127)):
            x = torch.randn(*bad, device="cuda", dtype=torch.bfloat16)
            wb = w if bad[1] % 2 == 0 else torch.randn(192, bad[1], device="cuda",
                                                       dtype=torch.bfloat16) * 0.05
            with self.assertRaises(ValueError):
                strassen_matmul(x, wb)

    def test_bias_is_applied_once(self):
        from strassen_gemm import strassen_matmul

        torch.manual_seed(0)
        x = torch.randn(256, 128, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(192, 128, device="cuda", dtype=torch.bfloat16) * 0.05
        b = torch.randn(192, device="cuda", dtype=torch.bfloat16)
        torch.testing.assert_close(strassen_matmul(x, w, b), strassen_matmul(x, w) + b,
                                   rtol=2e-2, atol=2e-2)
        self.assertFalse(torch.allclose(strassen_matmul(x, w, b), strassen_matmul(x, w),
                                        rtol=1e-3, atol=1e-3))

    def test_three_dimensional_input_keeps_its_shape(self):
        from strassen_gemm import strassen_matmul

        torch.manual_seed(0)
        w = torch.randn(192, 128, device="cuda", dtype=torch.bfloat16) * 0.05
        x = torch.randn(2, 256, 128, device="cuda", dtype=torch.bfloat16)
        got = strassen_matmul(x, w)
        self.assertEqual(got.shape, (2, 256, 192))
        self.assertEqual(got.dtype, x.dtype)


class StrassenCombineTest(absltest.TestCase):
    """The assembly kernel, which was the first implementation's real bug."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if not torch.cuda.is_available():
            raise absltest.SkipTest("needs a GPU")

    def test_the_assembly_reaches_copy_bandwidth(self):
        """The first combine held seven FP32 tiles live per program -- 448 KB of
        registers at 128x128 -- and spilled to 0.286 TB/s where a copy does
        1.52. The flat version reaches ~1.55. Guarding it matters because a slow
        assembly would be charged to Strassen rather than to the kernel.
        """
        import statistics

        import triton

        import strassen_gemm as S

        M2, N2 = 4096, 4096
        ps = [torch.randn((M2, N2), device="cuda", dtype=torch.float16) for _ in range(7)]
        c = torch.empty((M2 * 2, N2 * 2), device="cuda", dtype=torch.float16)
        grid = (triton.cdiv(M2 * N2, 1024),)

        def run():
            S._combine[grid](*ps, c, M2, N2, ps[0].stride(0), ps[0].stride(1),
                             c.stride(0), c.stride(1), BLOCK=1024, num_warps=4)

        for _ in range(3):
            run()
        torch.cuda.synchronize()
        samples = []
        for _ in range(10):
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            run()
            e1.record()
            torch.cuda.synchronize()
            samples.append(e0.elapsed_time(e1) / 1e3)
        moved = (7 + 4) * M2 * N2 * 2
        tbs = moved / statistics.median(samples) / 1e12
        print(f"\n  assembly bandwidth {tbs:.3f} TB/s")
        self.assertGreater(tbs, 0.9, f"assembly at {tbs:.3f} TB/s suggests register spilling")


if __name__ == "__main__":
    absltest.main()
