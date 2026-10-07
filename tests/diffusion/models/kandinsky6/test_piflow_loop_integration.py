# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""``piflow_denoise_loop`` against a real (small, random) Kandinsky 6 DiT.

:mod:`test_scheduling_piflow` checks the pi-Flow *math* against a naive oracle.
This checks the *integration*: that the loop calls the DiT with the arguments it
expects, splits a grid head of ``out_visual_dim = in_visual_dim * n_grid`` the
right way round, keeps the packed ``(sum_T, H, W, C)`` layout intact, and
returns latents that are finite and actually moved.

It uses a deliberately tiny DiT with random weights — two visual blocks,
``model_dim`` 64 — so it runs in seconds. It needs a GPU: the diffusion attention
backends (cuDNN / FlashAttention) have no CPU kernel. Random weights cannot tell us
the video looks right, but they do catch every structural failure, which is
where a port of this shape goes wrong: a transposed grid axis, a per-request
scalar not expanded per token, an argument the DiT names differently, or latents
that come back all zeros.

The all-zeros assertion is not decoration. A pipeline that silently returns a
black video is indistinguishable from a working one at the API level, and that
is exactly the failure this port hit on first run.
"""

from __future__ import annotations

import json
import os
import tempfile
from unittest import mock

import torch
from absl.testing import absltest

_MASTER_PORT = "29577"


def _init_single_rank(test: absltest.TestCase) -> None:
    """A one-rank TP group for vLLM's parallel layers, torn down after the test.

    The group is process-global: leaving it up makes the next test module that
    initializes its own fail with "tensor model parallel group is already
    initialized", so this mirrors the repo's fixture convention and cleans up.
    """
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import init_distributed_environment, initialize_model_parallel
    from vllm.distributed.parallel_state import cleanup_dist_env_and_memory, model_parallel_is_initialized

    if model_parallel_is_initialized():
        return
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", _MASTER_PORT)
    with set_current_vllm_config(VllmConfig()):
        init_distributed_environment(
            world_size=1,
            rank=0,
            distributed_init_method=f"tcp://127.0.0.1:{_MASTER_PORT}",
            local_rank=0,
            backend="gloo",
        )
        initialize_model_parallel(1, 1)
    test.addCleanup(cleanup_dist_env_and_memory)


# A Pro-shaped config shrunk until it runs on a laptop. The ratios that matter
# are kept: a grid head (out = in * n_grid) for both modalities, a separate audio
# branch, and head_dim = sum(axes_dims).
N_GRID = 4
IN_VISUAL_DIM = 4
IN_AUDIO_DIM = 6
TINY_CONFIG = {
    "model_dim": 64,
    "time_dim": 32,
    "ff_dim": 128,
    "num_text_blocks": 1,
    "num_visual_blocks": 2,
    "patch_size": [1, 2, 2],
    "axes_dims": [8, 12, 12],
    "in_visual_dim": IN_VISUAL_DIM,
    "out_visual_dim": IN_VISUAL_DIM * N_GRID,
    "in_text_dim": 16,
    "in_text_dim2": 8,
    "visual_cond": True,
    "is_multimodal": True,
    "in_audio_dim": IN_AUDIO_DIM,
    "out_audio_dim": IN_AUDIO_DIM * N_GRID,
    "model_dim_a": 32,
    "time_dim_a": 32,
    "ff_dim_a": 64,
    "axes_dims_a": [8, 12, 12],
    "audio_freqs_scaling": 0.144,
    "text_token_padding": False,
    "ca_rope": True,
    "cross_gates": True,
    "fix_modulation": True,
    "visual_token_type_num_embeddings": 2,
}

LATENT_FRAMES = 3
LATENT_H = 4
LATENT_W = 6
AUDIO_LEN = 12
TEXT_LEN = 5


def _initialize_randomly(module: torch.nn.Module, *, std: float = 0.02) -> None:
    """Give every parameter a finite value, as a checkpoint load would."""
    with torch.no_grad():
        for name, param in module.named_parameters():
            if name.endswith(".bias"):
                param.zero_()
            elif "norm" in name.rsplit(".", 2)[0].split(".")[-1].lower():
                # A norm's affine weight multiplies its normalized input, so it
                # starts at one; a normal(0, 0.02) here would scale activations
                # to nothing.
                param.fill_(1.0)
            else:
                param.normal_(0.0, std)
        for name, buffer in module.named_buffers():
            if not torch.isfinite(buffer).all():
                raise AssertionError(f"buffer {name} is not finite after construction")


class PiflowLoopIntegrationTest(absltest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        _init_single_rank(self)
        torch.manual_seed(0)

        from vllm_omni.diffusion.models.kandinsky6.kandinsky6_transformer import (
            Kandinsky6Transformer3DModel,
        )

        if not torch.cuda.is_available():
            self.skipTest("the diffusion attention backends have no CPU kernel")
        self.device = torch.device("cuda")
        # BF16, not FP32: the diffusion attention backends accept only half
        # precision, so an FP32 model never reaches the attention kernel.
        self.dtype = torch.bfloat16
        self.dit = Kandinsky6Transformer3DModel.from_diffusers_config(dict(TINY_CONFIG)).to(
            device=self.device, dtype=self.dtype
        )
        # vLLM's parallel linear layers allocate their parameters with
        # torch.empty and never initialize them: a real model is always filled
        # from a checkpoint right after construction. Without a checkpoint the
        # weights hold whatever was in that memory, and the first LayerNorm turns
        # it into NaN, so every parameter has to be given a value here. The output
        # heads additionally start deliberately zeroed, which would make the DiT
        # predict exactly zero -- the degenerate case this test exists to reject.
        _initialize_randomly(self.dit)
        self.dit.eval()

    def _bundle_and_conditioning(self):
        from vllm_omni.diffusion.models.kandinsky6.pipeline_kandinsky6 import (
            LatentBundle,
            compute_rope1d,
            compute_visual_rope,
        )

        video = torch.randn(LATENT_FRAMES, LATENT_H, LATENT_W, IN_VISUAL_DIM, device=self.device, dtype=self.dtype)
        audio = torch.randn(AUDIO_LEN, IN_AUDIO_DIM, device=self.device, dtype=self.dtype)
        bundle = LatentBundle(
            video=video,
            audio=audio,
            video_cu_seqlens=torch.tensor([0, LATENT_FRAMES], dtype=torch.int32, device=self.device),
            audio_cu_seqlens=torch.tensor([0, AUDIO_LEN], dtype=torch.int32, device=self.device),
        )
        text_embeds = {
            "text_embeds": torch.randn(TEXT_LEN, TINY_CONFIG["in_text_dim"], device=self.device, dtype=self.dtype),
            "pooled_embed": torch.randn(1, TINY_CONFIG["in_text_dim2"], device=self.device, dtype=self.dtype),
        }
        visual_rope = compute_visual_rope(
            self.dit.visual_rope_embeddings,
            (LATENT_FRAMES, LATENT_H // 2, LATENT_W // 2),
            (1.0, 1.0, 1.0),
        )
        audio_rope = compute_rope1d(self.dit.audio_rope_embeddings, AUDIO_LEN)
        text_rope = [
            compute_rope1d(self.dit.video_text_rope_embeddings, TEXT_LEN),
            compute_rope1d(self.dit.audio_text_rope_embeddings, TEXT_LEN),
        ]
        return bundle, text_embeds, visual_rope, audio_rope, text_rope

    def test_loop_moves_latents_and_stays_finite(self) -> None:
        from vllm_omni.diffusion.models.kandinsky6.pipeline_kandinsky6 import piflow_denoise_loop
        from vllm_omni.diffusion.models.kandinsky6.scheduling_kandinsky6_piflow import (
            KandinskyPiflowScheduler,
        )

        bundle, text_embeds, visual_rope, audio_rope, text_rope = self._bundle_and_conditioning()
        before_video = bundle.video.clone()
        before_audio = bundle.audio.clone()
        scheduler = KandinskyPiflowScheduler(shift=5.0, n_grid=N_GRID, num_policy_substeps=16)

        with torch.no_grad():
            out = piflow_denoise_loop(
                bundle=bundle,
                dit=self.dit,
                text_embeds=text_embeds,
                visual_rope=visual_rope,
                audio_rope=audio_rope,
                text_rope=text_rope,
                num_steps=4,
                scheduler=scheduler,
            )

        self.assertEqual(tuple(out.video.shape), (LATENT_FRAMES, LATENT_H, LATENT_W, IN_VISUAL_DIM))
        self.assertEqual(tuple(out.audio.shape), (AUDIO_LEN, IN_AUDIO_DIM))
        # The FP32 schedule tensors must not promote the latents: the VAE and the
        # rest of the pipeline work in the model dtype.
        self.assertEqual(out.video.dtype, self.dtype)
        self.assertEqual(out.audio.dtype, self.dtype)
        self.assertTrue(torch.isfinite(out.video).all(), "video latents must stay finite")
        self.assertTrue(torch.isfinite(out.audio).all(), "audio latents must stay finite")
        # The degenerate failures a black video comes from: all zeros, or
        # unchanged noise because the policy was never applied.
        self.assertGreater(float(out.video.abs().max()), 0.0, "video latents collapsed to zero")
        self.assertGreater(float(out.audio.abs().max()), 0.0, "audio latents collapsed to zero")
        self.assertFalse(torch.allclose(out.video, before_video), "video latents did not move")
        self.assertFalse(torch.allclose(out.audio, before_audio), "audio latents did not move")

    def test_rejects_cfg(self) -> None:
        """Distilled PiFlow has no CFG branch, so guidance must be refused."""
        from vllm_omni.diffusion.models.kandinsky6.pipeline_kandinsky6 import piflow_denoise_loop
        from vllm_omni.diffusion.models.kandinsky6.scheduling_kandinsky6_piflow import (
            KandinskyPiflowScheduler,
        )

        bundle, text_embeds, visual_rope, audio_rope, text_rope = self._bundle_and_conditioning()
        with self.assertRaisesRegex(ValueError, "guidance"):
            piflow_denoise_loop(
                bundle=bundle,
                dit=self.dit,
                text_embeds=text_embeds,
                visual_rope=visual_rope,
                audio_rope=audio_rope,
                text_rope=text_rope,
                num_steps=2,
                scheduler=KandinskyPiflowScheduler(shift=5.0, n_grid=N_GRID),
                guidance_weight=5.0,
            )


class PiflowStepCacheIntegrationTest(PiflowLoopIntegrationTest):
    """Late-step caching (``VLLM_OMNI_K6_PIFLOW_CACHE_STEPS``) and the step probe.

    Inherits the tiny DiT; the inherited tests run again here, which is cheap.
    """

    def _run(self, num_steps: int = 4, **kwargs):
        from vllm_omni.diffusion.models.kandinsky6.pipeline_kandinsky6 import piflow_denoise_loop
        from vllm_omni.diffusion.models.kandinsky6.scheduling_kandinsky6_piflow import (
            KandinskyPiflowScheduler,
        )

        torch.manual_seed(0)
        bundle, text_embeds, visual_rope, audio_rope, text_rope = self._bundle_and_conditioning()
        calls = []
        handle = self.dit.register_forward_hook(lambda *_: calls.append(1))
        try:
            with torch.no_grad():
                out = piflow_denoise_loop(
                    bundle=bundle,
                    dit=self.dit,
                    text_embeds=text_embeds,
                    visual_rope=visual_rope,
                    audio_rope=audio_rope,
                    text_rope=text_rope,
                    num_steps=num_steps,
                    scheduler=KandinskyPiflowScheduler(shift=5.0, n_grid=N_GRID, num_policy_substeps=16),
                    **kwargs,
                )
        finally:
            handle.remove()
        return out, len(calls)

    def test_cached_steps_skip_the_dit(self) -> None:
        base, base_calls = self._run()
        self.assertEqual(base_calls, 4)
        for mode in ("reuse", "extrapolate"):
            with mock.patch.dict(
                os.environ,
                {"VLLM_OMNI_K6_PIFLOW_CACHE_STEPS": "3,4", "VLLM_OMNI_K6_PIFLOW_CACHE_MODE": mode},
            ):
                cached, calls = self._run()
            self.assertEqual(calls, 2, mode)
            self.assertTrue(torch.isfinite(cached.video).all(), mode)
            # How close a cached step lands is a property of trained weights (the
            # step probe measures it); random ones only show the path was taken.
            self.assertFalse(torch.equal(cached.video, base.video), mode)

    def test_probe_branches_reproduce_the_switch(self) -> None:
        """A probe's cache branch is the same computation as the serving switch,
        and the probe leaves the request's own result untouched."""
        base, _ = self._run()
        with mock.patch.dict(os.environ, {"VLLM_OMNI_K6_PIFLOW_CACHE_STEPS": "3,4"}):
            switched, _ = self._run()

        sunk = {}
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "probe.json")
            with open(path, "w") as f:
                json.dump(
                    {
                        "out_dir": tmp,
                        "branches": [
                            {"name": "reuse-3-4", "cache_steps": [3, 4]},
                            {"name": "perturb-4", "perturb_step": 4, "eps": 0.01},
                        ],
                    },
                    f,
                )
            with mock.patch.dict(os.environ, {"VLLM_OMNI_K6_STEP_PROBE": path}):
                probed, calls = self._run(step_probe_sink=lambda name, bundle: sunk.update({name: bundle}))

        # 4 base calls, none for the cache branch (its steps 3-4 are both cached),
        # one for the perturbed step 4.
        self.assertEqual(calls, 5)
        torch.testing.assert_close(probed.video, base.video, rtol=0, atol=0)
        torch.testing.assert_close(sunk["reuse-3-4"].video, switched.video, rtol=0, atol=0)
        torch.testing.assert_close(sunk["reuse-3-4"].audio, switched.audio, rtol=0, atol=0)
        self.assertFalse(torch.equal(sunk["perturb-4"].video, base.video))

    def test_step_one_cannot_be_cached(self) -> None:
        from vllm_omni.diffusion.models.kandinsky6.pipeline_kandinsky6 import piflow_cache_steps

        with mock.patch.dict(os.environ, {"VLLM_OMNI_K6_PIFLOW_CACHE_STEPS": "1,10"}):
            with self.assertRaisesRegex(ValueError, "step 1"):
                piflow_cache_steps()
        with mock.patch.dict(os.environ, {"VLLM_OMNI_K6_PIFLOW_CACHE_STEPS": " 9, 10"}):
            self.assertEqual(piflow_cache_steps(), frozenset({9, 10}))


if __name__ == "__main__":
    absltest.main()
