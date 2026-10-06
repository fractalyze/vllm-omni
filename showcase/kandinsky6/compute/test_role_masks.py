# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Which Kandinsky 6 attention roles receive an attention mask. CPU only.

This is the constraint the per-role arms in `arms/` are built on.
`SageAttentionImpl.forward_cuda` raises outright on a mask
(`"SAGE_ATTN does not support attn_mask"`), so a mask-rejecting backend may
only be pinned to a call site that never sees one. Reading the forward and
counting `attn_mask=` arguments is not enough: the mask reaches an attention
layer through two names (`attn_mask` and a trailing `attention_mask` kwarg),
and whether the fused block forwards it depends on `text_token_padding`,
`cross_gates` and `ca_rope`. So this runs a tiny T2VA model with a real
padding mask and records what each role was actually handed.

It runs on the CPU with TORCH_SDPA and needs no checkpoint; the answer is a
property of the module graph, not of the device.

    /data/jooman/k6/venv/bin/python -m pytest showcase/kandinsky6/compute/test_role_masks.py -q
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import torch
from absl.testing import absltest

# A tiny T2VA config carrying the four flags both real bundles set. The video
# and audio head dims are deliberately equal (sum(axes_dims) ==
# sum(axes_dims_a) == 12): with `ca_rope` the cross-modal attention applies one
# modality's RoPE table to the other's queries, so unequal head dims raise. The
# Pro and Lite bundles both satisfy this -- Pro at 128 and 128, Lite at 64 and
# 64 -- but a config invented for a test does not unless it is made to.
TINY_T2VA = dict(
    in_visual_dim=4,
    out_visual_dim=4,
    in_text_dim=8,
    in_text_dim2=6,
    time_dim=16,
    patch_size=(1, 2, 2),
    model_dim=24,
    ff_dim=32,
    num_text_blocks=1,
    num_visual_blocks=2,
    axes_dims=(4, 4, 4),
    visual_cond=False,
    is_multimodal=True,
    in_audio_dim=6,
    model_dim_a=12,
    ff_dim_a=16,
    axes_dims_a=(4, 4, 4),
    attention_engine="sdpa",
    ca_rope=True,
    cross_gates=True,
    fix_modulation=True,
    text_token_padding=True,
)

# The two roles whose backend must accept a mask, and the four that are free.
MASKED_ROLES = {"kandinsky6.text_self", "kandinsky6.text_cross"}
MASK_FREE_ROLES = {
    "kandinsky6.visual_self",
    "kandinsky6.audio_self",
    "kandinsky6.video_audio_cross",
    "kandinsky6.audio_video_cross",
}


class RoleMaskTest(absltest.TestCase):
    def setUp(self):
        super().setUp()
        # Same shim as tests/diffusion/models/kandinsky6: the parallel Linear
        # layers dispatch their GEMM through a CUDA-only helper otherwise.
        import vllm.model_executor.layers.linear as linear_mod
        from vllm.config import VllmConfig, set_current_vllm_config
        from vllm.distributed.parallel_state import (
            cleanup_dist_env_and_memory,
            init_distributed_environment,
            initialize_model_parallel,
            model_parallel_is_initialized,
        )
        from vllm.model_executor.layers.utils import default_unquantized_gemm

        self.enterContext(
            absltest.mock.patch.object(
                linear_mod, "dispatch_unquantized_gemm", lambda *a, **k: default_unquantized_gemm
            )
        )
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29534")
        self.enterContext(set_current_vllm_config(VllmConfig()))
        if not model_parallel_is_initialized():
            init_distributed_environment(world_size=1, rank=0, local_rank=0, distributed_init_method="env://")
            initialize_model_parallel()
            self.addCleanup(cleanup_dist_env_and_memory)

    def _roles_and_masks(self) -> dict[str, set[bool]]:
        """Run one forward with a padding mask; report mask presence per role."""
        from vllm_omni.diffusion.config import set_current_diffusion_config
        from vllm_omni.diffusion.data import AttentionConfig, AttentionSpec
        from vllm_omni.diffusion.models.kandinsky6 import Kandinsky6Transformer3DModel
        from vllm_omni.diffusion.models.kandinsky6.kandinsky6_transformer import Kandinsky6Attention

        diffusion_config = SimpleNamespace(
            diffusion_attention_config=AttentionConfig(default=AttentionSpec(backend="TORCH_SDPA")),
            parallel_config=SimpleNamespace(ring_degree=1, allgather_degree=1),
        )
        with set_current_diffusion_config(diffusion_config):
            model = Kandinsky6Transformer3DModel(**TINY_T2VA).eval()

        seen: dict[str, set[bool]] = {}
        original = Kandinsky6Attention.forward

        def probe(self, hidden_states, encoder_hidden_states=None, attn_mask=None, **kwargs):
            mask = attn_mask if attn_mask is not None else kwargs.get("attention_mask")
            seen.setdefault(self.attn.role, set()).add(mask is not None)
            return original(self, hidden_states, encoder_hidden_states, attn_mask, **kwargs)

        self.enterContext(absltest.mock.patch.object(Kandinsky6Attention, "forward", probe))

        frames, height, width, audio_len, text_len = 2, 4, 4, 7, 5
        visual_rope = model.visual_rope_embeddings(
            shape=(frames, height // 2, width // 2),
            pos=[torch.arange(frames), torch.arange(height // 2), torch.arange(width // 2)],
        )
        with torch.no_grad():
            model(
                x_video=torch.randn(frames, height, width, 4),
                x_audio=torch.randn(audio_len, 6),
                text_embed=[torch.randn(text_len, 8)] * 2,
                pooled_text_embed=[torch.randn(1, 6)] * 2,
                time=[torch.tensor([500.0])] * 2,
                visual_rope=visual_rope,
                audio_rope=model.audio_rope_embeddings(torch.arange(audio_len)),
                text_rope=[
                    model.video_text_rope_embeddings(torch.arange(text_len)),
                    model.audio_text_rope_embeddings(torch.arange(text_len)),
                ],
                # One padded position, which is what text_token_padding means.
                attention_mask=torch.tensor([1, 1, 1, 1, 0], dtype=torch.bool),
            )
        return seen

    def test_every_role_is_exercised(self):
        """A role that the forward never reaches would make the other two
        assertions vacuously true."""
        seen = self._roles_and_masks()
        self.assertEqual(set(seen), MASKED_ROLES | MASK_FREE_ROLES)

    def test_only_the_text_roles_receive_a_mask(self):
        """So `arms/*.json` may pin a mask-rejecting backend to the other four,
        and must not pin one to these."""
        seen = self._roles_and_masks()
        for role in MASKED_ROLES:
            self.assertIn(True, seen[role], msg=f"{role} was expected to receive the padding mask")

    def test_the_mask_free_roles_never_receive_one(self):
        seen = self._roles_and_masks()
        for role in MASK_FREE_ROLES:
            self.assertEqual(
                seen[role],
                {False},
                msg=f"{role} saw a mask; a mask-rejecting backend must not be pinned to it",
            )


if __name__ == "__main__":
    absltest.main()
