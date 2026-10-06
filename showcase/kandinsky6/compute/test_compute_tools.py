# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Tests for the Track C compute tools. No GPU, no checkpoint.

These guard the two pieces of logic that a wrong answer would silently
corrupt every later measurement with: the token counts derived from a
request's geometry, and the kernel-name table that the attention / GEMM /
elementwise split is computed from.

    /data/jooman/k6/venv/bin/python -m pytest showcase/kandinsky6/compute/test_compute_tools.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from absl.testing import absltest, parameterized

sys.path.insert(0, str(Path(__file__).resolve().parent))

from block_profile import LITE, PRO, Shapes  # noqa: E402
from kernel_classes import CATEGORIES, classify, split_by_category  # noqa: E402


class ShapesTest(absltest.TestCase):
    def test_w1_token_counts_match_the_plan(self):
        """PLAN.md's W1: 31 x 30 x 54 = 50,220 visual tokens, 218 audio."""
        shapes = Shapes(height=480, width=864, num_frames=121, fps=24.0, text_len=256)
        self.assertEqual((shapes.latent_frames, shapes.latent_h, shapes.latent_w), (31, 30, 54))
        self.assertEqual(shapes.visual_tokens, 50220)
        self.assertEqual(shapes.audio_len, 218)

    def test_audio_length_matches_the_pipeline_helper(self):
        """The same arithmetic as pipeline_kandinsky6.audio_latent_duration."""
        import math

        for num_frames in (25, 49, 121, 125):
            shapes = Shapes(height=320, width=512, num_frames=num_frames, fps=24.0, text_len=8)
            sample_frames = (shapes.latent_frames - 1) * 4 + 1
            expected = int(math.ceil(sample_frames / 24.0 * 44100 / 1024))
            self.assertEqual(shapes.audio_len, expected, msg=f"num_frames={num_frames}")

    def test_single_frame_request_keeps_one_latent_frame(self):
        shapes = Shapes(height=320, width=512, num_frames=1, fps=24.0, text_len=8)
        self.assertEqual(shapes.latent_frames, 1)
        self.assertEqual(shapes.visual_tokens, 20 * 32)


class ConfigTest(parameterized.TestCase):
    @parameterized.named_parameters(("pro", PRO), ("lite", LITE))
    def test_model_dim_is_a_multiple_of_head_dim(self, cfg):
        """An architectural constraint of DiffusionTransformer3D, not of the port."""
        self.assertEqual(cfg.model_dim % sum(cfg.axes_dims), 0)
        self.assertEqual(cfg.model_dim_a % sum(cfg.axes_dims_a), 0)

    def test_pro_is_the_shape_the_plan_names(self):
        self.assertEqual(PRO.model_dim, 4096)
        self.assertEqual(sum(PRO.axes_dims), 128)
        self.assertEqual(PRO.model_dim // sum(PRO.axes_dims), 32)  # 32 heads of 128
        self.assertEqual(PRO.ff_dim, 16384)
        self.assertEqual(PRO.num_visual_blocks, 60)


class KernelClassesTest(parameterized.TestCase):
    @parameterized.named_parameters(
        # Names as they appear in a torch profiler table on sm_120.
        ("fa2", "void flash_fwd_kernel<Flash_fwd_kernel_traits<128, 128, 128, 4, ...>>", "attention"),
        ("fa4_cute", "flash::FlashAttentionForwardSm100", "attention"),
        ("cudnn", "cudnn_generated_fort_native_sdpa_sm90_flash_fprop", "attention"),
        ("sdpa_efficient", "fmha_cutlassF_bf16_aligned_64x128_rf_sm80", "attention"),
        ("flashinfer", "BatchPrefillWithRaggedKVCacheKernel", "attention"),
        ("sage2", "qk_int8_sv_f8_accum_f32_attn_inst_buf", "attention"),
        # The names the sm_120 path actually emits, from
        # results/k6c-p01b-block-profile-w1-tuned.json. `qk_int_sv` has no
        # digit after `int`; missing it put 23.5% of a block in `unclassified`.
        ("sage2_sm120", "void qk_int_sv_f8_attn_kernel<128u, 64u, 32u, 64u, 128u, (DataType)1>", "attention"),
        ("sage2_quant_prologue", "void QuantInt8Kernel<128u, 32u, 1u, false, false, __nv_bfloat16>", "attention"),
        ("sage2_mean_scale", "void MeanScaleKernel<64u, false, __nv_bfloat16>", "attention"),
        ("flex", "triton_tem_fused_flex_attention_0", "attention"),
        ("cublaslt", "nvjet_tst_192x128_64x4_1x2_h_bz_coopA_NTT", "gemm"),
        ("cutlass", "cutlass::device_kernel<cutlass_80_tensorop_bf16_s16816gemm>", "gemm"),
        ("bmm", "void at::native::bmm_out_cuda_impl", "gemm"),
        ("layernorm", "void at::native::vectorized_layer_norm_kernel<float, float>", "norm"),
        ("rmsnorm", "void at::native::rms_norm_kernel", "norm"),
        ("elementwise", "void at::native::vectorized_elementwise_kernel<4, CUDAFunctor_add>", "elementwise"),
        ("gelu", "void at::native::GeluCUDAKernelImpl", "elementwise"),
        ("reduce", "void at::native::reduce_kernel<512, 1, ReduceOp>", "elementwise"),
        ("copy", "void at::native::direct_copy_kernel_cuda", "copy"),
        ("cat", "void CatArrayBatchedCopy<float, unsigned int, 4, 128>", "copy"),
    )
    def test_known_kernel_names(self, name, expected):
        self.assertEqual(classify(name), expected)

    def test_an_unknown_name_is_not_silently_bucketed(self):
        """A kernel no rule claims must be visible, not folded into a share."""
        self.assertEqual(classify("some_kernel_nobody_has_seen_yet"), "unclassified")

    def test_attention_wins_over_gemm_when_a_name_matches_both(self):
        """Ordered rules: an attention kernel whose name contains `gemm`
        must not be counted as a block GEMM, or the headline split moves."""
        self.assertEqual(classify("flash_fwd_gemm_kernel_sm120"), "attention")

    def test_split_returns_every_category(self):
        totals = split_by_category([("flash_fwd_kernel", 100.0), ("nvjet_tst_192x128", 50.0)])
        self.assertEqual(set(totals), set(CATEGORIES))
        self.assertEqual(totals["attention"], 100.0)
        self.assertEqual(totals["gemm"], 50.0)
        self.assertEqual(totals["elementwise"], 0.0)

    def test_split_sums_to_the_input_total(self):
        kernels = [("flash_fwd", 1.0), ("nvjet", 2.0), ("mystery", 4.0), ("layer_norm", 8.0)]
        self.assertEqual(sum(split_by_category(kernels).values()), 15.0)


if __name__ == "__main__":
    absltest.main()


class ActivationsTest(absltest.TestCase):
    """The synthetic activations must match what the port feeds a kernel.

    These run on the CPU: the distribution is the claim, and it does not
    depend on the device.
    """

    def _role(self, q=64, kv=64, heads=2, head_dim=128):
        from attn_race import Role

        return Role("visual_self", q, kv, heads, head_dim)

    def test_query_and_key_rows_have_unit_rms(self):
        """`Kandinsky6Attention` RMS-normalizes per head before RoPE, and
        RoPE is a rotation, so every row the kernel sees has RMS 1. A
        SageAttention error measured against rows of the wrong norm would
        be measuring the wrong kernel."""
        import torch
        from attn_race import make_activations

        role = self._role()
        q, k, _ = make_activations(role, torch.device("cpu"), torch.float32, seed=7)
        for name, tensor in (("query", q), ("key", k)):
            rms = tensor.square().mean(dim=-1).sqrt()
            self.assertAlmostEqual(float(rms.mean()), 1.0, delta=0.02, msg=name)
            self.assertAlmostEqual(float(rms.max()), 1.0, delta=0.05, msg=name)

    def test_shapes_are_the_backend_contract(self):
        """Backends require exactly (B, S, H, D)."""
        import torch
        from attn_race import make_activations

        role = self._role(q=48, kv=96, heads=4, head_dim=128)
        q, k, v = make_activations(role, torch.device("cpu"), torch.float32, seed=7)
        self.assertEqual(tuple(q.shape), (1, 48, 4, 128))
        self.assertEqual(tuple(k.shape), (1, 96, 4, 128))
        self.assertEqual(tuple(v.shape), (1, 96, 4, 128))

    def test_the_seed_fixes_the_activations(self):
        """Every arm must see the same q/k/v, or the race is meaningless."""
        import torch
        from attn_race import make_activations

        role = self._role()
        first = make_activations(role, torch.device("cpu"), torch.float32, seed=11)
        second = make_activations(role, torch.device("cpu"), torch.float32, seed=11)
        for a, b in zip(first, second):
            torch.testing.assert_close(a, b, rtol=0, atol=0)


class ReferenceAttentionTest(absltest.TestCase):
    def test_matches_an_unchunked_reference(self):
        """The chunked, per-head fp32 reference must equal the obvious
        one-shot computation. A wrong reference silently rescores every arm."""
        import torch
        from attn_race import reference_attention_fp32

        torch.manual_seed(3)
        q = torch.randn(1, 130, 3, 32, dtype=torch.float32)
        k = torch.randn(1, 177, 3, 32, dtype=torch.float32)
        v = torch.randn(1, 177, 3, 32, dtype=torch.float32)

        scale = 1.0 / 32**0.5
        qh, kh, vh = (t[0].transpose(0, 1) for t in (q, k, v))
        expected = torch.matmul(
            torch.softmax(torch.matmul(qh, kh.transpose(-2, -1)) * scale, dim=-1), vh
        ).transpose(0, 1).unsqueeze(0)

        # A budget that forces several query blocks per head, so the chunk
        # boundaries are actually exercised.
        got = reference_attention_fp32(q, k, v, score_budget_gib=177 * 4 * 16 / 2**30)
        torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-5)

    def test_restores_the_tf32_setting(self):
        import torch
        from attn_race import reference_attention_fp32

        was = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            q = torch.randn(1, 8, 1, 16)
            reference_attention_fp32(q, q.clone(), q.clone())
            self.assertTrue(torch.backends.cuda.matmul.allow_tf32)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = was


class AccuracyTest(absltest.TestCase):
    def test_identical_output_scores_zero_error(self):
        import torch
        from attn_race import accuracy

        reference = torch.randn(1, 16, 2, 32)
        scored = accuracy(reference.clone(), reference)
        self.assertEqual(scored["rel_l2"], 0.0)
        self.assertEqual(scored["max_abs"], 0.0)
        self.assertAlmostEqual(scored["cosine"], 1.0, places=6)

    def test_a_missing_scale_shows_in_rel_l2_but_not_cosine(self):
        """The two numbers answer different questions: a kernel that dropped
        its softmax scale is a large rel_l2 at cosine 1."""
        import torch
        from attn_race import accuracy

        reference = torch.randn(1, 16, 2, 32)
        scored = accuracy(reference * 2.0, reference)
        self.assertAlmostEqual(scored["rel_l2"], 1.0, places=5)
        self.assertAlmostEqual(scored["cosine"], 1.0, places=6)


class AttentionArmConfigTest(parameterized.TestCase):
    """The arm files in `arms/` must be valid `--diffusion-attention-config`
    values that resolve the role they claim to.

    This is the whole adoption path for a race winner: no code changes, just
    one of these files. A typo in a role string would silently fall through
    to the platform default and the "winner" would never run, so the role
    names are asserted against the ones `kandinsky6_transformer.py` passes.
    """

    # The five role strings in kandinsky6_transformer.py, with the
    # role_category each call site passes alongside.
    K6_ROLES = (
        ("kandinsky6.visual_self", "self"),
        ("kandinsky6.text_self", "self"),
        ("kandinsky6.audio_self", "self"),
        ("kandinsky6.text_cross", "cross"),
        ("kandinsky6.video_audio_cross", "cross"),
        ("kandinsky6.audio_video_cross", "cross"),
    )

    def _arms_dir(self):
        return Path(__file__).resolve().parent / "arms"

    def test_the_role_strings_are_the_ones_the_port_passes(self):
        """Read them out of the model source rather than trusting this list."""
        import re

        source = (
            Path(__file__).resolve().parents[3]
            / "vllm_omni/diffusion/models/kandinsky6/kandinsky6_transformer.py"
        ).read_text()
        # Every `kandinsky6.*` string literal, not just the ones directly
        # after `role=`: one call site picks its role with a conditional
        # (`role="kandinsky6.visual_self" if ... else "kandinsky6.audio_self"`),
        # so the second branch has no `role=` in front of it.
        in_source = set(re.findall(r'"(kandinsky6\.[a-z_]+)"', source))
        self.assertEqual(in_source, {role for role, _ in self.K6_ROLES})

    # What each arm file is expected to resolve, role -> backend. A role left
    # out must resolve to None (platform default). `control.json` says "auto",
    # which AttentionConfig normalizes to "no override", so it expects nothing.
    ARMS = {
        "control.json": {},
        "sage2.json": {"kandinsky6.visual_self": "SAGE_ATTN"},
        "sage3.json": {"kandinsky6.visual_self": "SAGE_ATTN_3"},
        "flash.json": {"kandinsky6.visual_self": "FLASH_ATTN"},
        "tuned.json": {
            "kandinsky6.visual_self": "SAGE_ATTN",
            "kandinsky6.video_audio_cross": "SAGE_ATTN",
            "kandinsky6.audio_video_cross": "TORCH_SDPA",
            "kandinsky6.audio_self": "TORCH_SDPA",
        },
    }

    # Roles that receive a padding mask, so a mask-rejecting backend must
    # never be pinned to them. test_role_masks.py establishes the list by
    # running a forward; this is the consequence for the config files.
    MASKED_ROLES = frozenset({"kandinsky6.text_self", "kandinsky6.text_cross"})
    MASK_REJECTING_BACKENDS = frozenset({"SAGE_ATTN", "SAGE_ATTN_3"})

    @parameterized.named_parameters(
        ("control", "control.json"),
        ("sage2", "sage2.json"),
        ("sage3", "sage3.json"),
        ("flash", "flash.json"),
        ("tuned", "tuned.json"),
    )
    def test_arm_resolves_exactly_the_roles_it_claims(self, filename):
        import json

        from vllm_omni.diffusion.data import build_attention_config

        expected = self.ARMS[filename]
        config = build_attention_config(json.loads((self._arms_dir() / filename).read_text()))
        for role, category in self.K6_ROLES:
            spec, _ = config.resolve_with_source(role=role, role_category=category)
            resolved = spec.backend if spec else None
            self.assertEqual(resolved, expected.get(role), msg=role)

    @parameterized.named_parameters(
        ("control", "control.json"),
        ("sage2", "sage2.json"),
        ("sage3", "sage3.json"),
        ("flash", "flash.json"),
        ("tuned", "tuned.json"),
    )
    def test_no_mask_rejecting_backend_on_a_masked_role(self, filename):
        """SageAttention raises on attn_mask, and the two text roles get one.
        Pinning one there would serve fine until the first padded prompt."""
        for role, backend in self.ARMS[filename].items():
            if role in self.MASKED_ROLES:
                self.assertNotIn(backend, self.MASK_REJECTING_BACKENDS, msg=f"{filename}: {role}")

    @parameterized.named_parameters(
        ("sage2", "sage2.json"),
        ("sage3", "sage3.json"),
        ("flash", "flash.json"),
        ("tuned", "tuned.json"),
    )
    def test_the_named_backends_exist_in_the_registry(self, filename):
        """A backend name that is not a registry member would only fail at
        serve time, after the weights are loaded."""
        from vllm_omni.diffusion.attention.backends.registry import DiffusionAttentionBackendEnum

        for backend in self.ARMS[filename].values():
            self.assertIn(backend, DiffusionAttentionBackendEnum.__members__)

    def test_every_arm_file_on_disk_is_covered(self):
        """A new arm added without a row in ARMS would go untested."""
        on_disk = {path.name for path in self._arms_dir().glob("*.json")}
        self.assertEqual(on_disk, set(self.ARMS))


class ArmSpecTest(parameterized.TestCase):
    """`--compare-arm LABEL=ATTENTION@MODE`. A spec that parses wrongly would
    silently measure a different arm than the label claims, which is worse
    than an error."""

    @parameterized.named_parameters(
        ("platform_eager", "shipped-eager=default@eager", "shipped-eager", None, "eager"),
        ("platform_compiled", "shipped=default@default", "shipped", None, "default"),
        ("file_compiled", "tuned=arms/tuned.json@max-autotune", "tuned", "arms/tuned.json", "max-autotune"),
        # A path with a drive-style colon and slashes must survive, which is
        # why the separator is `@` rather than `:` or `/`.
        ("absolute_path", "t=/a/b/c.json@default", "t", "/a/b/c.json", "default"),
    )
    def test_valid_specs(self, spec, label, attention, mode):
        from block_profile import parse_arm_spec

        got_label, got_attention, got_mode = parse_arm_spec(spec)
        self.assertEqual(got_label, label)
        self.assertEqual(None if got_attention is None else str(got_attention), attention)
        self.assertEqual(got_mode, mode)

    @parameterized.named_parameters(
        ("no_label", "default@eager"),
        ("no_mode", "x=default"),
        ("empty_label", "=default@eager"),
        ("empty_mode", "x=default@"),
        ("empty_attention", "x=@eager"),
    )
    def test_invalid_specs_raise(self, spec):
        from block_profile import parse_arm_spec

        with self.assertRaises(ValueError):
            parse_arm_spec(spec)


class FlattenOutputsTest(absltest.TestCase):
    def test_a_single_tensor_and_a_tuple_both_flatten(self):
        import torch
        from block_profile import _flatten_outputs

        one = torch.zeros(2)
        self.assertEqual(len(_flatten_outputs(one)), 1)
        self.assertEqual(len(_flatten_outputs((one, one.clone()))), 2)

    def test_a_none_in_the_tuple_is_dropped(self):
        """The fused block returns (vis, aud) and aud is None for T2V."""
        import torch
        from block_profile import _flatten_outputs

        self.assertEqual(len(_flatten_outputs((torch.zeros(2), None))), 1)


class ParameterStorageTest(absltest.TestCase):
    """`output_deltas` only reports a number when two arms share weights, and
    it must recognize a compiled module as sharing them. torch.compile returns
    an OptimizedModule whose parameter *names* are all prefixed `_orig_mod.`,
    so a name-based comparison silently answers "not comparable" for every
    compiled arm -- which it did, and which made the check useless exactly
    where it was needed."""

    def test_a_compiled_module_shares_its_own_storages(self):
        import torch
        from block_profile import _parameter_storages

        module = torch.nn.Linear(4, 4)
        compiled = torch.compile(module)
        self.assertEqual(_parameter_storages(compiled), _parameter_storages(module))

    def test_two_separate_modules_do_not(self):
        import torch
        from block_profile import _parameter_storages

        self.assertNotEqual(
            _parameter_storages(torch.nn.Linear(4, 4)),
            _parameter_storages(torch.nn.Linear(4, 4)),
        )


class RopeTableTest(absltest.TestCase):
    """`make_inputs` must build RoPE tables with the port's own modules.

    A RoPE table's 2x2 blocks are [[cos, -sin], [sin, cos]], so `apply_rotary`
    is a rotation and preserves the per-head norm that query_norm/key_norm
    just set. An earlier version drew the table from `randn`, which makes it
    an arbitrary linear map: the dense bf16 backends stayed finite, so it
    looked harmless, but SageAttention's per-block INT8 scale is set by the
    largest entry in a block and the visual self-attention returned NaN at W1.
    """

    def _tables(self, cfg_name="pro"):
        import torch
        from block_profile import CONFIGS, W1, Shapes, make_inputs

        inputs = make_inputs("fused", CONFIGS[cfg_name], Shapes(**W1), torch.device("cpu"), torch.bfloat16)
        return inputs["vis_rope"], inputs["aud_rope"]

    def test_every_block_is_a_rotation(self):
        import torch

        for name, table in zip(("vis_rope", "aud_rope"), self._tables()):
            blocks = table.reshape(-1, 2, 2)
            dets = blocks[:, 0, 0] * blocks[:, 1, 1] - blocks[:, 0, 1] * blocks[:, 1, 0]
            torch.testing.assert_close(dets, torch.ones_like(dets), rtol=0, atol=1e-5, msg=f"{name}: det != 1")
            norms = blocks[:, 0, :].norm(dim=-1)
            torch.testing.assert_close(norms, torch.ones_like(norms), rtol=0, atol=1e-5, msg=f"{name}: row norm")

    def test_the_tables_have_the_shape_apply_rotary_broadcasts_against(self):
        from block_profile import PRO, W1, Shapes

        shapes = Shapes(**W1)
        vis_rope, aud_rope = self._tables()
        self.assertEqual(tuple(vis_rope.shape), (shapes.visual_tokens, 1, sum(PRO.axes_dims) // 2, 2, 2))
        self.assertEqual(tuple(aud_rope.shape), (shapes.audio_len, 1, sum(PRO.axes_dims_a) // 2, 2, 2))

    def test_the_tables_are_fp32(self):
        """The port's apply_rotary upcasts to fp32; a bf16 table would quantize
        the angles before the rotation."""
        import torch

        for table in self._tables():
            self.assertEqual(table.dtype, torch.float32)

    def test_lite_shapes_work_too(self):
        """Lite's head_dim is 64, not 128, so the table halves with it."""
        import torch
        from block_profile import LITE, W1, Shapes, make_inputs

        inputs = make_inputs("fused", LITE, Shapes(**W1), torch.device("cpu"), torch.bfloat16)
        self.assertEqual(inputs["vis_rope"].shape[-3], sum(LITE.axes_dims) // 2)
        self.assertTrue(torch.isfinite(inputs["vis_rope"]).all())


class OutputDeltaControlTest(absltest.TestCase):
    """A non-finite control must be named as such, not reported as NaN deltas.

    This is the lesson of the RoPE bug: the check did run, and it did return
    NaN for every arm, and a NaN rel_l2 reads as "could not tell" when it
    should read as "the baseline is broken".
    """

    def test_a_nan_control_is_reported_as_a_broken_control(self):
        import torch
        from block_profile import output_deltas

        class Nan(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(1))

            def forward(self, x):
                return x * float("nan")

        control = Nan()
        result = output_deltas({"control": control, "other": control}, {"x": torch.ones(4)})
        self.assertFalse(result["control_is_finite"])
        self.assertIn("not finite", result["why"])

    def test_a_finite_control_reports_per_arm_rows(self):
        import torch
        from block_profile import output_deltas

        class Scale(torch.nn.Module):
            def __init__(self, factor):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(1))
                self.factor = factor

            def forward(self, x):
                return x * self.factor

        control = Scale(1.0)
        result = output_deltas({"control": control, "same": control}, {"x": torch.ones(4)})
        self.assertTrue(result["control_is_finite"])
        self.assertTrue(result["same"]["comparable"])
        self.assertEqual(result["same"]["outputs"][0]["rel_l2"], 0.0)
