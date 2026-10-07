# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Exactness of the hybrid FP16-accumulate GEMM, against FP64 and against the
path it would replace.

The bar is deliberately not "close to exact". The served path today is BF16
inputs with an FP32 accumulator, which is itself inexact, so the question is
whether the hybrid adds error *beyond what the model already tolerates*. BF16
has 7 explicit mantissa bits and FP16 has 10, so moving the operands to FP16
reduces input quantisation error while the FP16 accumulation adds some back, and
only a measurement says which wins.

Real checkpoint weights are used where available (W1's own
``visual_transformer_blocks.0``) because a weight matrix's singular values
decide how much a reduction-order change shows up, and `randn` has none of that
structure. Tests that need the checkpoint skip without it rather than silently
testing something else.
"""

from __future__ import annotations

import glob
import os
import sys
from pathlib import Path

import torch
from absl.testing import absltest, parameterized

sys.path.insert(0, str(Path(__file__).resolve().parent))

_CKPT = sorted(glob.glob(
    "/data/jooman/hf/hub/models--kandinskylab--Kandinsky-6.0-Pro-distill-5s-Diffusers"
    "/snapshots/*/transformer/*.safetensors"))

# Real layers worth testing: the two FFN shapes and an attention projection,
# which between them cover every K and N the visual stream uses at W1.
_LAYERS = [
    ("ff1", "visual_transformer_blocks.0.video_dec_block.feed_forward.net.0.proj.weight"),
    ("ff2", "visual_transformer_blocks.0.video_dec_block.feed_forward.net.2.weight"),
    ("to_query", "visual_transformer_blocks.0.video_dec_block.self_attention.to_query.weight"),
]


def _cuda() -> bool:
    return torch.cuda.is_available()


def _load(name: str) -> torch.Tensor | None:
    from safetensors import safe_open

    for shard in _CKPT:
        with safe_open(shard, framework="pt") as f:
            if name in f.keys():
                return f.get_tensor(name)
    return None


def _rel_l2(got: torch.Tensor, ref: torch.Tensor) -> float:
    return ((got.double() - ref).norm() / ref.norm()).item()


class HybridGemmNumericsTest(parameterized.TestCase):
    """Error against FP64, held to the BF16 path's own error."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if not _cuda():
            raise absltest.SkipTest("needs a GPU")
        if not _CKPT:
            raise absltest.SkipTest("needs the W1 checkpoint for real weights")

    @parameterized.named_parameters(*[(n, n, k) for n, k in _LAYERS])
    def test_hybrid_is_not_worse_than_the_bf16_path_it_replaces(self, _name, key):
        """The hybrid must not cost more accuracy than BF16+FP32-accumulate does.

        Both are scored against an FP64 reference on the same inputs. `M` is a
        slice rather than W1's full 50,220 rows because the error is a per-row
        property and an FP64 matmul at the full size does not fit.
        """
        from hybrid_gemm import hybrid_matmul

        w = _load(key)
        self.assertIsNotNone(w, f"{key} not in the checkpoint")
        w = w.cuda()
        N, K = w.shape
        M = 1024
        torch.manual_seed(0)
        # Unit-variance activations: the residual stream is RMS-normed before
        # every projection in this model, so this is the regime that matters.
        x = torch.randn(M, K, device="cuda", dtype=torch.float32)

        ref = x.double() @ w.double().t()
        bf16 = (x.to(torch.bfloat16) @ w.to(torch.bfloat16).t())
        hyb = hybrid_matmul(x.to(torch.bfloat16), w.to(torch.bfloat16))

        e_bf16 = _rel_l2(bf16, ref)
        e_hyb = _rel_l2(hyb, ref)
        print(f"\n  {_name}: K={K} N={N}  bf16+fp32acc {e_bf16:.3e}   hybrid {e_hyb:.3e}   "
              f"ratio {e_hyb / e_bf16:.2f}x")
        # The hybrid may be better (FP16 inputs quantise less than BF16) or
        # slightly worse (FP16 accumulation within a block). A 2x allowance on
        # the path's existing error is the claim; anything beyond that is a
        # different trade and should fail here rather than be discovered in a
        # gate run.
        self.assertLess(e_hyb, 2.0 * e_bf16,
                        f"hybrid error {e_hyb:.3e} exceeds twice the BF16 path's {e_bf16:.3e}")

    def test_smaller_block_k_lowers_the_error(self):
        """BLOCK_K is the accuracy dial, and the direction must hold.

        The whole design rests on error depending on BLOCK_K rather than on K.
        If a larger BLOCK_K did not cost accuracy, the promotion would be doing
        nothing and the kernel would be a plain FP16-accumulate GEMM.
        """
        from hybrid_gemm import hybrid_matmul

        w = _load(_LAYERS[0][1])
        self.assertIsNotNone(w)
        w = w.cuda()
        N, K = w.shape
        torch.manual_seed(0)
        x = torch.randn(1024, K, device="cuda", dtype=torch.float32)
        ref = x.double() @ w.double().t()

        # 16/32/64 rather than up to 128: at BLOCK_M=BLOCK_N=128 and 4 stages a
        # BLOCK_K of 128 needs 192 KB of shared memory against this GPU's 99 KB,
        # so it is not a configuration the tuner can emit either.
        errs = {}
        for bk in (16, 32, 64):
            got = hybrid_matmul(x.to(torch.bfloat16), w.to(torch.bfloat16),
                                config={"BLOCK_K": bk})
            errs[bk] = _rel_l2(got, ref)
        print(f"\n  BLOCK_K -> rel L2: " + "  ".join(f"{k}:{v:.3e}" for k, v in errs.items()))
        self.assertLess(errs[16], errs[64],
                        f"BLOCK_K is supposed to bound the error but 16 ({errs[16]:.3e}) "
                        f"is not better than 64 ({errs[64]:.3e})")

    def test_the_promotion_is_actually_happening(self):
        """The test above is insensitive, so this one carries the invariant.

        With BF16 inputs the error floor is set by input quantisation (~2.3e-3
        on real weights) and both accumulate modes sit on it, so
        ``test_smaller_block_k_lowers_the_error`` only moves by fractions of a
        percent and would still pass if the FP32 promotion were removed
        entirely. Feeding the kernel FP16 inputs drops the input floor by an
        order of magnitude and makes the accumulator visible: promoted, the
        error is a few e-4; unpromoted over K=4096 it is a few e-3.

        This is what fails if someone "simplifies" the inner loop to
        ``acc += tl.dot(x, y, out_dtype=tl.float16)`` without the ``.to(float32)``.
        """
        from hybrid_gemm import hybrid_matmul

        w = _load(_LAYERS[0][1])
        self.assertIsNotNone(w)
        w = w.cuda().to(torch.float16)
        N, K = w.shape
        torch.manual_seed(0)
        x = torch.randn(1024, K, device="cuda", dtype=torch.float16)
        ref = x.double() @ w.double().t()

        hybrid = _rel_l2(hybrid_matmul(x, w, out_dtype=torch.float16), ref)
        # The unpromoted comparison: one `tl.dot` over the whole K in FP16. The
        # kernel cannot be asked for that, so it is approximated by the largest
        # BLOCK_K that fits, which is where the error trend points.
        coarse = _rel_l2(hybrid_matmul(x, w, config={"BLOCK_K": 64}, out_dtype=torch.float16), ref)
        print(f"\n  fp16 inputs: hybrid(BK=32) {hybrid:.3e}   BK=64 {coarse:.3e}")
        self.assertLess(hybrid, 1e-3,
                        f"with FP16 inputs a promoting kernel should reach a few e-4, got {hybrid:.3e}; "
                        f"this is what a dropped `.to(tl.float32)` looks like")

    def test_bias_in_the_epilogue_matches_adding_it_afterwards(self):
        """The fused bias must be arithmetically the bias.

        Guards the cheap failure: an epilogue that drops the bias still returns
        correctly-shaped output. (PR #38 removed cuBLAS's fused bias because it
        cost a worse tile; this kernel's tile is ours and does not change, so
        fusing is free here. The two facts are not in tension.)
        """
        from hybrid_gemm import hybrid_matmul

        torch.manual_seed(0)
        M, K, N = 512, 1024, 768
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02
        b = torch.randn(N, device="cuda", dtype=torch.bfloat16)

        fused = hybrid_matmul(x, w, b)
        separate = hybrid_matmul(x, w) + b
        torch.testing.assert_close(fused, separate, rtol=2e-2, atol=2e-2)
        # And the bias is doing something, so the above is not two zeros agreeing.
        self.assertFalse(torch.allclose(fused, hybrid_matmul(x, w), rtol=1e-3, atol=1e-3))

    def test_shapes_and_dtypes_match_f_linear(self):
        """A drop-in must be drop-in: same output shape and dtype as F.linear,
        including for a 3-D input, which is how the blocks hold the stream."""
        from hybrid_gemm import hybrid_matmul

        torch.manual_seed(0)
        K, N = 512, 256
        w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02
        for shape in ((777, K), (1, 777, K), (2, 3, K)):
            x = torch.randn(*shape, device="cuda", dtype=torch.bfloat16)
            got = hybrid_matmul(x, w)
            want = torch.nn.functional.linear(x, w)
            self.assertEqual(got.shape, want.shape, f"shape mismatch for input {shape}")
            self.assertEqual(got.dtype, want.dtype, f"dtype mismatch for input {shape}")

    def test_a_non_multiple_of_block_m_is_not_truncated(self):
        """W1's 50,220 rows are not a multiple of any BLOCK_M used here, so the
        masked tail is on the serving path, not an edge case."""
        from hybrid_gemm import hybrid_matmul

        torch.manual_seed(0)
        K, N = 1024, 512
        w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02
        for M in (128 * 3 + 1, 50220 % 1024 + 7):
            x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
            got = hybrid_matmul(x, w)
            want = torch.nn.functional.linear(x, w)
            self.assertEqual(got.shape, want.shape)
            # The tail rows must be real values, not left as uninitialised memory.
            self.assertTrue(torch.isfinite(got).all(), f"non-finite output at M={M}")
            rel = (got.float() - want.float()).norm() / want.float().norm()
            self.assertLess(rel.item(), 5e-2, f"tail rows wrong at M={M}: rel {rel:.2e}")


class Fp16SafetyTest(absltest.TestCase):
    """The range report, which is what keeps a layer off this kernel."""

    def test_headroom_and_subnormals_are_reported(self):
        from hybrid_gemm import FP16_MAX, fp16_safety

        t = torch.tensor([0.0, 1.0, 100.0, 2.0**-20])
        got = fp16_safety(t)
        self.assertAlmostEqual(got["absmax"], 100.0, places=5)
        self.assertAlmostEqual(got["headroom"], FP16_MAX / 100.0, places=2)
        self.assertGreater(got["subnormal_frac"], 0.0, "2^-20 is subnormal in FP16")

    def test_a_tensor_over_fp16_max_reports_headroom_below_one(self):
        """This is the signal a caller routes a layer away on."""
        from hybrid_gemm import fp16_safety

        got = fp16_safety(torch.tensor([1.0, 1e5]))
        self.assertLess(got["headroom"], 1.0)

    def test_non_finite_values_do_not_poison_the_report(self):
        from hybrid_gemm import fp16_safety

        got = fp16_safety(torch.tensor([1.0, float("inf"), float("nan"), 3.0]))
        self.assertAlmostEqual(got["absmax"], 3.0, places=5)


class HybridSwitchTest(absltest.TestCase):
    def test_unset_is_off(self):
        from hybrid_gemm import enabled

        for value in ("", "0", "false", "False"):
            with absltest.mock.patch.dict(os.environ, {"VLLM_OMNI_K6_HYBRID_GEMM": value}):
                self.assertFalse(enabled(), f"{value!r} must not enable the kernel")

    def test_one_is_on(self):
        from hybrid_gemm import enabled

        with absltest.mock.patch.dict(os.environ, {"VLLM_OMNI_K6_HYBRID_GEMM": "1"}):
            self.assertTrue(enabled())


if __name__ == "__main__":
    absltest.main()


class HybridRoutingTest(absltest.TestCase):
    """`hybrid_linear`'s fallback policy — the part the serving path depends on.

    Each of these is a way the kernel could silently become a regression or a
    wrong answer in production, which is why they are tested at the routing
    level and not only at the kernel level.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if not _cuda():
            raise absltest.SkipTest("needs a GPU")

    def test_the_switch_being_off_is_exactly_f_linear(self):
        """Unset must be bit-identical to today, or the flag is not a flag."""
        from hybrid_gemm import hybrid_linear

        torch.manual_seed(0)
        x = torch.randn(4096, 512, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(256, 512, device="cuda", dtype=torch.bfloat16) * 0.02
        b = torch.randn(256, device="cuda", dtype=torch.bfloat16)
        with absltest.mock.patch.dict(os.environ, {"VLLM_OMNI_K6_HYBRID_GEMM": "0"}):
            got = hybrid_linear(x, w, b)
        torch.testing.assert_close(got, torch.nn.functional.linear(x, w, b), rtol=0, atol=0)

    def test_small_m_falls_back_so_the_audio_branch_does_not_regress(self):
        """W1's audio branch is M=218, where the hybrid is 1.7-2.7x slower.

        Routing it to the hybrid would be a regression dressed as an
        optimisation, so the crossover gate must hold.
        """
        from hybrid_gemm import MIN_ROWS_FOR_HYBRID, should_use_hybrid

        w = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16) * 0.02
        x_small = torch.randn(218, 2048, device="cuda", dtype=torch.bfloat16)
        ok, why = should_use_hybrid(x_small, w)
        self.assertFalse(ok)
        self.assertIn("crossover", why)

        x_big = torch.randn(MIN_ROWS_FOR_HYBRID, 2048, device="cuda", dtype=torch.bfloat16)
        ok, _ = should_use_hybrid(x_big, w)
        self.assertTrue(ok, "at the threshold the hybrid should be taken")

    def test_an_out_of_range_weight_keeps_the_current_path(self):
        """FP16 overflows at 65504 where BF16 reaches ~3.4e38, so a large-weight
        layer must not be converted. This is the failure that would produce inf
        rather than a slightly different number."""
        from hybrid_gemm import should_use_hybrid

        x = torch.randn(4096, 512, device="cuda", dtype=torch.bfloat16)
        w_hot = torch.full((256, 512), 3.0e4, device="cuda", dtype=torch.bfloat16)
        ok, why = should_use_hybrid(x, w_hot)
        self.assertFalse(ok)
        self.assertIn("weight absmax", why)

    def test_out_of_range_activations_keep_the_current_path(self):
        from hybrid_gemm import should_use_hybrid

        w = torch.randn(256, 512, device="cuda", dtype=torch.bfloat16) * 0.02
        x_hot = torch.randn(4096, 512, device="cuda", dtype=torch.bfloat16)
        x_hot[0] = 5.0e4
        ok, why = should_use_hybrid(x_hot, w)
        self.assertFalse(ok)
        self.assertIn("activation absmax", why)

    def test_the_activation_check_samples_rather_than_scanning(self):
        """The range check runs on every call, so it must not cost a full pass.

        It strides the rows; this pins that it does, because a check that reads
        all 411 MB of a W1 activation on every call would eat the speedup it is
        protecting.
        """
        from hybrid_gemm import _RANGE_SAMPLE_STRIDE

        self.assertGreater(_RANGE_SAMPLE_STRIDE, 1,
                           "a stride of 1 means the check reads the whole activation")

    def test_routing_on_a_real_w1_shape_takes_the_hybrid(self):
        """The shapes this exists for must actually be routed to it."""
        from hybrid_gemm import should_use_hybrid

        # 8192 rather than W1's 50,220: same verdict, a quarter of the memory.
        x = torch.randn(8192, 4096, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16) * 0.02
        ok, why = should_use_hybrid(x, w)
        self.assertTrue(ok, f"a W1 visual projection should take the hybrid, got: {why}")
