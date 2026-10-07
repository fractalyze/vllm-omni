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
        # The exact-attention schedule's fallback role. A visual self-attention
        # call routed through it resolves to the platform default, because no
        # arm config names it -- that is how `VLLM_OMNI_K6_EXACT_ATTN_STEPS` and
        # `_BLOCKS` make individual calls exact without a second backend
        # setting. It belongs here because the port really does pass it.
        ("kandinsky6.visual_self_exact", "self"),
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
        # The two cheap audio roles only. Dense bf16 either way, so this arm
        # is the one expected to sit at the gate's noise floor -- `tuned.json`
        # does not (set mean LPIPS 0.118 against a 0.05 limit).
        "lossless.json": {
            "kandinsky6.audio_video_cross": "TORCH_SDPA",
            "kandinsky6.audio_self": "TORCH_SDPA",
        },
        # `tuned.json` without Sage on video_audio_cross. That call is 0.19 ms
        # a block, so dropping it costs almost no speed and removes one of the
        # two quantized paths -- the cheapest thing to try against the gate's
        # max limit.
        "sage2-visual-only.json": {
            "kandinsky6.visual_self": "SAGE_ATTN",
            "kandinsky6.audio_video_cross": "TORCH_SDPA",
            "kandinsky6.audio_self": "TORCH_SDPA",
        },
        # The band dial: the same Sage2 kernel on the same roles, restricted to a
        # range of visual blocks so the ends of the stack keep exact attention.
        # Only the range differs between the three, which is the point -- the
        # band is the free parameter once the weights are BF16 and the kernel is
        # chosen.
        "sage2-wide.json": {
            "kandinsky6.visual_self": "SAGE_ATTN",
            "kandinsky6.audio_video_cross": "TORCH_SDPA",
            "kandinsky6.audio_self": "TORCH_SDPA",
        },
        "sage2-mid.json": {
            "kandinsky6.visual_self": "SAGE_ATTN",
            "kandinsky6.audio_video_cross": "TORCH_SDPA",
            "kandinsky6.audio_self": "TORCH_SDPA",
        },
        "sage2-narrow.json": {
            "kandinsky6.visual_self": "SAGE_ATTN",
            "kandinsky6.audio_video_cross": "TORCH_SDPA",
            "kandinsky6.audio_self": "TORCH_SDPA",
        },
        # The band dial's zero point: the same three roles with no `layers`
        # range at all, so Sage2 runs on every visual block. It exists so that a
        # band comparison changes exactly one key. It is the fastest arm measured
        # that still passes G2 and G3, but it is 4-of-9 over the user's own max
        # against the shipped arm's 1-of-9, so it is not the headline -- see the
        # L7 section of measurements.md.
        "sage2-edge0.json": {
            "kandinsky6.visual_self": "SAGE_ATTN",
            "kandinsky6.audio_video_cross": "TORCH_SDPA",
            "kandinsky6.audio_self": "TORCH_SDPA",
        },
        # The H5 lossy fast mode's attention: SageAttention3 on the visual
        # self-attention, same two audio roles on SDPA. Judged by G3 and contact
        # sheets only and never presented as passing G1, so it is listed here
        # for coverage rather than as a gate candidate.
        "h5-sage3.json": {
            "kandinsky6.visual_self": "SAGE_ATTN_3",
            "kandinsky6.audio_video_cross": "TORCH_SDPA",
            "kandinsky6.audio_self": "TORCH_SDPA",
        },
        # The same roles, with Sage's accuracy knobs on: FP16 PV with FP32
        # accumulation and per-thread INT8 granularity.
        "sage2-accurate.json": {
            "kandinsky6.visual_self": "SAGE_ATTN",
            "kandinsky6.audio_video_cross": "TORCH_SDPA",
            "kandinsky6.audio_self": "TORCH_SDPA",
        },
        # tuned.json with Sage3 (FP4) on the dominant call. Its kernel is
        # 1.27x Sage2's, so this is the fastest arm worth gating -- the
        # objective is the fastest config that passes, and on Pro the first
        # prompt left error budget to spend.
        "sage3-tuned.json": {
            "kandinsky6.visual_self": "SAGE_ATTN_3",
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
        ("lossless", "lossless.json"),
        ("sage2_visual_only", "sage2-visual-only.json"),
        ("sage2_accurate", "sage2-accurate.json"),
        ("sage3_tuned", "sage3-tuned.json"),
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
        ("lossless", "lossless.json"),
        ("sage2_visual_only", "sage2-visual-only.json"),
        ("sage2_accurate", "sage2-accurate.json"),
        ("sage3_tuned", "sage3-tuned.json"),
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
        ("lossless", "lossless.json"),
        ("sage2_visual_only", "sage2-visual-only.json"),
        ("sage2_accurate", "sage2-accurate.json"),
        ("sage3_tuned", "sage3-tuned.json"),
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


class NablaGeometryTest(parameterized.TestCase):
    """The port's NABLA path needs both patched latent dims divisible by 8.

    `fractal_flatten(..., block_mask=True)` patches the grid in 8x8 tiles, so
    a grid whose H or W is not a multiple of 8 fails the reshape. That rules
    out W1 (31 x 30 x 54) and the 512x320 smoke (7 x 20 x 32), which is why
    `--geometry nabla-ok` exists. Asserted here so a backlog item that says
    "NABLA at W1" is contradicted by the test suite rather than by an hour of
    GPU time.
    """

    @parameterized.named_parameters(
        ("w1", "w1", False),
        ("smoke", "smoke", False),
        ("nabla_ok", "nabla-ok", True),
    )
    def test_sparse_params_accepts_only_a_divisible_grid(self, geometry, expected_ok):
        import torch
        from block_profile import GEOMETRIES, PRO, Shapes, sparse_params

        shapes = Shapes(**GEOMETRIES[geometry])
        if expected_ok:
            params = sparse_params(PRO, shapes, torch.device("cpu"), threshold=0.9)
            blocks = shapes.latent_frames * (shapes.latent_h // 8) * (shapes.latent_w // 8)
            # The prior must be block-granular, or it will not broadcast
            # against nabla_block_mask's (B, h, S/64, S/64) scores.
            self.assertEqual(tuple(params["sta_mask"].shape), (blocks, blocks))
            self.assertEqual(blocks, shapes.visual_tokens // 64)
            self.assertTrue(params["to_fractal"])
        else:
            with self.assertRaisesRegex(ValueError, "divisible by 8"):
                sparse_params(PRO, shapes, torch.device("cpu"), threshold=0.9)

    def test_the_fractal_reorder_itself_rejects_w1(self):
        """The constraint is the port's, not this harness's: show it failing
        in `fractal_flatten` directly."""
        import torch

        from vllm_omni.diffusion.models.kandinsky6.kandinsky6_transformer import fractal_flatten

        frames, height, width = 31, 30, 54  # W1's patched latent grid
        x = torch.zeros(frames, height, width, 8)
        rope = torch.zeros(frames, height, width, 1, 4, 2, 2)
        with self.assertRaises(RuntimeError):
            fractal_flatten(x, rope, (frames, height, width), block_mask=True)


class GateTierTest(parameterized.TestCase):
    """The two bars are separate and both are reported.

    The adoption gate (mean <= 0.15, max <= 0.25) says whether an arm may
    ship; the `approx` tier (0.05 / 0.10) says what to call it. An arm can
    clear the first and still be `lossy`, and a showcase that collapses the
    two is how a lossy arm gets published as near-lossless.
    """

    @parameterized.named_parameters(
        # (set_mean, set_max, noise_floor_max, expected tier)
        ("identical", 0.0, 0.0, None, "exact"),
        ("inside_the_noise_floor", 0.001, 0.0025, 0.0030, "reorder"),
        ("above_the_floor_but_tight", 0.01, 0.02, 0.0030, "approx"),
        ("no_floor_measured_never_reorder", 0.001, 0.0025, None, "approx"),
        ("sage2_on_set_a", 0.1178, 0.3745, 0.0030, "lossy"),
        ("mean_ok_max_not", 0.02, 0.2, 0.0030, "lossy"),
    )
    def test_tier(self, set_mean, set_max, floor, expected):
        from gate_score import tier

        self.assertEqual(tier(set_mean, set_max, floor), expected)

    def test_sage2_on_set_a_passes_the_mean_and_fails_the_max(self):
        """The measured numbers, as a regression on the limits themselves: if
        someone widens a limit, this says which conclusion changes."""
        from gate_score import ADOPTION_MAX_LPIPS, ADOPTION_MEAN_LPIPS

        set_mean, set_max = 0.1178, 0.3745
        self.assertLessEqual(set_mean, ADOPTION_MEAN_LPIPS)
        self.assertGreater(set_max, ADOPTION_MAX_LPIPS)

    def test_the_adoption_bar_is_looser_than_the_approx_tier(self):
        from gate_score import (
            ADOPTION_MAX_LPIPS,
            ADOPTION_MEAN_LPIPS,
            APPROX_MAX_LPIPS,
            APPROX_MEAN_LPIPS,
        )

        self.assertGreater(ADOPTION_MEAN_LPIPS, APPROX_MEAN_LPIPS)
        self.assertGreater(ADOPTION_MAX_LPIPS, APPROX_MAX_LPIPS)


class SageAccuracySpecTest(absltest.TestCase):
    """`arms/sage2-accurate.json` must actually select the accurate variant.

    The arm differs from `sage2-visual-only.json` only in a `quant` block, so
    a resolver that silently ignored it would leave two files that look
    different and behave identically -- and the gate numbers would be
    attributed to a knob that never took effect.
    """

    def _quant_of(self, filename, role="kandinsky6.visual_self"):
        import json

        from vllm_omni.diffusion.data import build_attention_config

        path = Path(__file__).resolve().parent / "arms" / filename
        config = build_attention_config(json.loads(path.read_text()))
        spec, _ = config.resolve_with_source(role=role, role_category="self")
        return spec.backend_kwargs()

    def test_the_accurate_arm_selects_the_fp16_fp32accum_per_thread_variant(self):
        from vllm_omni.diffusion.attention.backends.sage_attn import _resolve_variant

        variant = _resolve_variant(self._quant_of("sage2-accurate.json"))
        self.assertIsNotNone(variant)
        self.assertEqual(variant["name"], "qk_int8_pv_fp16_fp32accum_per_thread")

    def test_the_plain_arm_keeps_the_dispatcher(self):
        """No quant spec means the top-level dispatcher, which is what every
        release before this knob used."""
        from vllm_omni.diffusion.attention.backends.sage_attn import _resolve_variant

        self.assertIsNone(_resolve_variant(self._quant_of("sage2-visual-only.json")))
        self.assertIsNone(_resolve_variant(self._quant_of("tuned.json")))


class ServeFlagsTest(parameterized.TestCase):
    """The flag list is the arm's definition, so it is tested without a GPU.

    Two arms are only comparable if their server flags differ in exactly the
    thing under test. These assert the two offload modes are the ones the
    showcase's own serve scripts use, and that asking for an unknown one fails
    loudly rather than silently serving the default -- which would produce a
    plausible number for the wrong configuration.
    """

    def test_the_bf16_mode_matches_the_reference_serve_script(self):
        from gate_pro import serve_flags

        script = (Path(__file__).resolve().parents[1] / "serve" / "serve_pro_bf16_ref.sh").read_text()
        flags = serve_flags("dlo-mmap", None, None)
        for flag in ("--enable-distributed-layerwise-offload", "--dlo-no-use-allgather",
                     "--disable-multithread-weight-load"):
            self.assertIn(flag, flags)
            self.assertIn(flag, script)
        self.assertNotIn("--enable-layerwise-offload", flags)

    def test_the_fp8_mode_matches_the_baseline_serve_script(self):
        from gate_pro import serve_flags

        script = (Path(__file__).resolve().parents[1] / "serve" / "serve_pro_fp8.sh").read_text()
        flags = serve_flags("layerwise", None, None)
        self.assertIn("--enable-layerwise-offload", flags)
        self.assertIn("--enable-layerwise-offload", script)
        self.assertNotIn("--enable-distributed-layerwise-offload", flags)

    @parameterized.parameters("layerwise", "dlo-mmap")
    def test_the_attention_config_is_the_only_other_difference(self, offload):
        from gate_pro import serve_flags

        shipped = serve_flags(offload, None, None)
        armed = serve_flags(offload, '{"per_role": {}}', None)
        self.assertEqual(armed[: len(shipped)], shipped)
        self.assertEqual(armed[len(shipped):], ["--diffusion-attention-config", '{"per_role": {}}'])

    def test_an_unknown_offload_mode_is_refused(self):
        from gate_pro import serve_flags

        with self.assertRaisesRegex(ValueError, "unknown offload mode"):
            serve_flags("resident", None, None)


class ReferenceManifestRoundTripTest(absltest.TestCase):
    """A gate run's own manifest has to be readable as a reference.

    Set B has no canonical BF16 reference, so one has to be generated here and
    then scored against -- which only works if the directory a gate run writes
    is the shape ``_reference_settings`` reads. This asserts the round trip
    rather than trusting that two dict literals in the same file agree.
    """

    def test_a_gate_manifest_is_a_reference_manifest(self):
        import json
        import tempfile

        from gate_pro import _reference_settings

        prompts = [
            {"id": "b1-newsreader", "categories": ["face", "speech"]},
            {"id": "b2-market-haggle", "categories": ["speech"]},
        ]
        geometry = {"width": 864, "height": 480, "num_frames": 121,
                    "num_inference_steps": 10, "guidance_scale": 1.0}
        manifest = {
            "arm": "bf16-stream-control",
            "items": {
                entry["id"]: {
                    "categories": entry["categories"],
                    "request_wall_s": 175.0,
                    "started": 1791300000.0,
                    "seed": 42,
                    "geometry": geometry,
                }
                for entry in prompts
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(manifest))
            seeds, read_geometry = _reference_settings(path, prompts)

        self.assertEqual(seeds, {"b1-newsreader": 42, "b2-market-haggle": 42})
        self.assertEqual(read_geometry, geometry)


class LoadTimeQuantizationFlagTest(absltest.TestCase):
    """Load-time quantization is a flag on the exact checkpoint, not a new one.

    The offline converter writes per-*tensor* scales because vLLM's serialized
    fp8 method cannot read per-row ones. vLLM's online methods can, so the
    better recipe is reachable without a converter at all -- provided the flag
    is passed and provided it is the only difference from the arm it is
    compared with.
    """

    def test_the_method_name_reaches_the_server(self):
        from gate_pro import serve_flags

        flags = serve_flags("dlo-mmap", None, None, "fp8_per_channel")
        self.assertEqual(flags[-2:], ["--diffusion-quantization-config", "fp8_per_channel"])

    def test_vllm_knows_the_method_and_it_is_per_output_row(self):
        """Guards the name against a vLLM bump, and the recipe against a typo:
        ``fp8_per_tensor`` is also a valid name and is the thing being replaced."""
        from vllm.config.quantization import resolve_quantization_config

        args = resolve_quantization_config("fp8_per_channel", None)
        group_shape = args.linear.weight.scale.group_shape
        self.assertEqual((group_shape.row, group_shape.col), (-1, 1))
        self.assertIsNone(args.linear.activation, "weight-only: activations stay BF16")

    def test_no_quantization_leaves_the_flag_list_alone(self):
        from gate_pro import serve_flags

        self.assertEqual(serve_flags("layerwise", None, None), serve_flags("layerwise", None, None, None))


class OffloadPlacementTest(absltest.TestCase):
    """Resident blocks and text-encoder offload, validated against vLLM-Omni's own parser.

    This arm is weight-traffic-bound -- 60.3 GB re-read per step -- so where a
    tensor lives is a performance lever, and neither of these two placements has
    a CLI flag. The emitted config is checked by the parser that will consume it
    rather than against a literal, because the schema lives in another repo path
    and a silently rejected config would serve the default arm under the
    candidate's name.
    """

    def _parsed(self, flags):
        import json

        from vllm_omni.diffusion.offloader.config import parse_diffusion_offload_config

        self.assertIn("--diffusion-offload-config", flags)
        payload = flags[flags.index("--diffusion-offload-config") + 1]
        return parse_diffusion_offload_config(json.loads(payload))

    def test_the_plain_arms_keep_the_legacy_flags(self):
        """No extras means byte-identical flags to the reference serve script;
        the public config is only reached when something needs it."""
        from gate_pro import serve_flags

        for offload in ("layerwise", "dlo-mmap"):
            flags = serve_flags(offload, None, None)
            self.assertNotIn("--diffusion-offload-config", flags)

    def test_resident_blocks_resolve_to_the_streaming_backend(self):
        from vllm_omni.diffusion.offloader.config import DLOTransfer, OffloadStrategy, _public_strategy
        from gate_pro import serve_flags

        flags = serve_flags("dlo-mmap", None, None, None, resident_layers=5)
        parsed = self._parsed(flags)
        self.assertEqual(parsed.layer_options["dit"].resident_layers, 5)
        self.assertEqual(parsed.layer_options["dit"].weight_transfer, DLOTransfer.RANK_LOCAL)
        self.assertEqual(_public_strategy(parsed), OffloadStrategy.DISTRIBUTED_LAYER_WISE)
        self.assertIn("--enable-distributed-layerwise-offload", flags)

    def test_the_allgather_flag_is_not_passed_beside_the_config(self):
        """`_validate_legacy_layer_options` rejects that pair outright, so the
        server would refuse to start."""
        from gate_pro import serve_flags

        flags = serve_flags("dlo-mmap", None, None, None, resident_layers=5, offload_text_encoder=True)
        self.assertNotIn("--dlo-no-use-allgather", flags)

    def test_offloading_the_text_encoder_selects_it(self):
        from gate_pro import serve_flags

        parsed = self._parsed(serve_flags("dlo-mmap", None, None, None, offload_text_encoder=True))
        self.assertEqual(parsed.components, frozenset({"dit", "text_encoder"}))

    def test_a_streamed_arm_without_resident_blocks_still_streams(self):
        """Rank-local with no resident block resolves to plain layer-wise, which
        stages from pinned host memory -- a different arm. The backend flag is
        what keeps it the streamed one."""
        from gate_pro import serve_flags

        flags = serve_flags("dlo-mmap", None, None, None, offload_text_encoder=True)
        self.assertIn("--enable-distributed-layerwise-offload", flags)


class WorkingGateTest(parameterized.TestCase):
    """G2, the floor-relative gate, including its refusal to guess a floor.

    The literal gate is absolute, and on this pipeline a rerun of the same
    configuration in a fresh process already moves LPIPS because Inductor picks
    kernels by timing. An absolute bar below that movement is a bar on the noise,
    not on the arm. These tests pin the arithmetic and, more importantly, that a
    missing floor is reported as undecided rather than treated as zero -- which
    would restate the literal gate under a second name and look like a second
    opinion.
    """

    def test_an_arm_at_the_floor_passes(self):
        from gate_pro_score import g2_verdict

        verdict = g2_verdict(0.196, 0.269, 0.196, 0.269)
        self.assertTrue(verdict["passes"])
        self.assertAlmostEqual(verdict["mean_over_floor"], 1.0)

    def test_an_arm_inside_the_slack_passes_and_outside_fails(self):
        from gate_pro_score import g2_verdict

        self.assertTrue(g2_verdict(0.24, 0.33, 0.196, 0.269)["passes"])
        self.assertFalse(g2_verdict(0.26, 0.33, 0.196, 0.269)["passes"])

    def test_either_half_can_fail_it(self):
        from gate_pro_score import g2_verdict

        self.assertFalse(g2_verdict(0.20, 0.40, 0.196, 0.269)["passes"], "max over the limit")
        self.assertFalse(g2_verdict(0.30, 0.28, 0.196, 0.269)["passes"], "mean over the limit")

    @parameterized.parameters((None, 0.269), (0.196, None), (None, None))
    def test_a_missing_floor_is_undecided_not_a_pass(self, floor_mean, floor_max):
        from gate_pro_score import g2_verdict

        verdict = g2_verdict(0.01, 0.02, floor_mean, floor_max)
        self.assertFalse(verdict["decidable"])
        self.assertNotIn("passes", verdict)


class DistributionalGateTest(absltest.TestCase):
    """G3's verdict: two-sided, and undefined rather than passing at zero.

    CLIP cannot tell a better video from a differently-wrong one, so an arm
    whose prompt agreement rises is drifting just as much as one whose
    agreement falls. A one-sided test would pass exactly the case this check
    exists to catch.
    """

    def test_a_small_move_either_way_passes(self):
        from clip_gate import g3_verdict

        self.assertTrue(g3_verdict(0.3110, 0.3099)["passes"])
        self.assertTrue(g3_verdict(0.3099 * 0.99, 0.3099)["passes"])

    def test_a_rise_past_the_tolerance_fails_like_a_fall(self):
        from clip_gate import g3_verdict

        self.assertFalse(g3_verdict(0.3099 * 1.05, 0.3099)["passes"])
        self.assertFalse(g3_verdict(0.3099 * 0.95, 0.3099)["passes"])

    def test_a_zero_reference_is_undecided(self):
        from clip_gate import g3_verdict

        self.assertFalse(g3_verdict(0.3, 0.0)["decidable"])


class BandedArmTest(parameterized.TestCase):
    """The three banded arms differ only in their layer range.

    They exist to measure one dial, so anything else differing between them
    would make the curve measure two things at once.
    """

    BANDS = {"sage2-wide.json": "3:57", "sage2-mid.json": "6:54", "sage2-narrow.json": "12:48"}
    # The dial's zero point carries no range at all, so it cannot be expressed
    # as a band string. It is tested separately below.
    NO_BAND = "sage2-edge0.json"

    def _arm(self, filename):
        import json

        return json.loads((Path(__file__).resolve().parent / "arms" / filename).read_text())

    @parameterized.parameters(*sorted(BANDS))
    def test_the_range_is_the_only_difference_from_the_others(self, filename):
        import copy

        reference = copy.deepcopy(self._arm("sage2-mid.json"))
        candidate = copy.deepcopy(self._arm(filename))
        for arm in (reference, candidate):
            arm["per_role"]["kandinsky6"]["visual_self"].pop("layers")
        self.assertEqual(candidate, reference)

    @parameterized.parameters(*sorted(BANDS.items()))
    def test_each_band_is_the_range_it_is_named_for(self, filename, layers):
        self.assertEqual(self._arm(filename)["per_role"]["kandinsky6"]["visual_self"]["layers"], layers)

    @parameterized.parameters(*sorted(BANDS.items()))
    def test_the_range_is_symmetric_about_a_sixty_block_stack(self, filename, layers):
        """Kandinsky 6 Pro has 60 visual blocks and the schedule protects both
        ends, so an asymmetric band would be measuring two changes."""
        start, stop = (int(part) for part in layers.split(":"))
        self.assertEqual(start, 60 - stop, f"{filename}: {start} exact at the front, {60 - stop} at the back")

    def test_the_zero_band_arm_differs_only_by_the_missing_range(self):
        """`sage2-edge0.json` is the band dial at zero exact blocks.

        It must be `sage2-mid.json` with the `layers` key removed and nothing
        else, because the L7 comparison it was built for reads the difference
        between the two arms as the band's entire contribution -- a 26% quality
        change and an 11% speed change. L7's conclusion, that the band is not
        redundant with an exact first step, is only sound if exactly one thing
        differs.
        """
        import copy

        reference = copy.deepcopy(self._arm("sage2-mid.json"))
        candidate = copy.deepcopy(self._arm(self.NO_BAND))

        visual = reference["per_role"]["kandinsky6"]["visual_self"]
        self.assertIn("layers", visual, "sage2-mid must carry a range or this comparison means nothing")
        del visual["layers"]
        self.assertNotIn(
            "layers", candidate["per_role"]["kandinsky6"]["visual_self"],
            "the zero-band arm must not restrict the range at all",
        )
        self.assertEqual(candidate, reference)


class CompileDynamicFlagTest(parameterized.TestCase):
    """Static-shape compilation, which this workload can use and does not by default.

    The platform compiles the DiT with `dynamic=True`, which is right for a
    server that sees many geometries. These arms only ever run W1, so
    specialising on 50,220 tokens is available for free -- except that changing
    compilation changes which kernels run, and on this pipeline that is worth
    LPIPS 0.1455 against a differently-compiled reference. Hence the flag is
    opt-in and absent by default.
    """

    def test_absent_by_default_so_the_platform_decides(self):
        from gate_pro import serve_flags

        self.assertNotIn("--diffusion-compile-dynamic", serve_flags("dlo-mmap", None, None))

    def test_turning_it_off_uses_the_negative_flag(self):
        """vLLM renders a boolean field as a flag pair. Passing a value --
        `--diffusion-compile-dynamic false` -- is rejected at argument parsing
        with "unrecognized arguments: false", which is how this was found: two
        arms died at start-up before the server ever loaded a weight."""
        from gate_pro import serve_flags

        flags = serve_flags("dlo-mmap", None, None, compile_dynamic=False)
        self.assertIn("--no-diffusion-compile-dynamic", flags)
        self.assertNotIn("--diffusion-compile-dynamic", flags)
        self.assertNotIn("false", flags)

    def test_asking_for_the_default_emits_nothing(self):
        """Dynamic is already the platform default, so a flag for it would be a
        no-op that still changes the recorded command."""
        from gate_pro import serve_flags

        self.assertEqual(serve_flags("dlo-mmap", None, None, compile_dynamic=True),
                         serve_flags("dlo-mmap", None, None))

    def test_vllm_omni_still_defaults_to_dynamic(self):
        """If upstream ever flips this, the arm stops being a change and the
        comparison it is in becomes a null result that looks like a win."""
        from vllm_omni.diffusion.data import OmniDiffusionConfig

        self.assertIs(OmniDiffusionConfig.__dataclass_fields__["diffusion_compile_dynamic"].default, True)
