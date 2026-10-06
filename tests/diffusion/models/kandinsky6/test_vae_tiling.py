# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Tiled VAE decode must return the full frame for any latent size.

The tile loops used to start at ``range(0, size - tile + 1, stride)``, which
never decodes the tail when ``(size - tile)`` is not a multiple of the stride.
At W1 (864x480) the latent is 60x108 with 32-row tiles every 24 rows, so tiles
started at rows 0 and 24 only and the served video came out 448 rows tall,
silently cropped. Upstream diffusers iterates ``range(0, size, stride)`` and lets
the last tile be partial; so does the port now.

Channels are shrunk so this runs on CPU; tiling geometry does not depend on them.
"""

from __future__ import annotations

import torch
from absl.testing import parameterized

from vllm_omni.diffusion.models.kandinsky6.modeling_kandinsky6_vae import AutoencoderKLHunyuanVideo

TINY_VAE_CONFIG = {
    "in_channels": 3,
    "out_channels": 3,
    "latent_channels": 4,
    "block_out_channels": [8, 8, 8, 8],
    "layers_per_block": 1,
    "norm_num_groups": 4,
    "act_fn": "silu",
    "mid_block_add_attention": False,
    "spatial_compression_ratio": 8,
    "temporal_compression_ratio": 4,
    "down_block_types": ["HunyuanVideoDownBlock3D"] * 4,
    "up_block_types": ["HunyuanVideoUpBlock3D"] * 4,
}


class TiledDecodeShapeTest(parameterized.TestCase):
    @parameterized.named_parameters(
        ("w1_864x480", 60, 108),  # the shape that came out 448 rows tall
        ("smoke_512x320", 40, 64),
        ("aligned", 56, 104),  # (size - tile) a multiple of the stride
    )
    def test_tiled_decode_returns_the_full_frame(self, latent_h: int, latent_w: int) -> None:
        torch.manual_seed(0)
        vae = AutoencoderKLHunyuanVideo.from_config(TINY_VAE_CONFIG).eval()
        vae.use_tiling = True
        vae.use_framewise_decoding = False
        z = torch.randn(1, TINY_VAE_CONFIG["latent_channels"], 1, latent_h, latent_w)
        with torch.no_grad():
            out = vae.tiled_decode(z, return_dict=False)[0]
        self.assertEqual(tuple(out.shape[-2:]), (latent_h * 8, latent_w * 8))
        # The tail must be decoded, not left as allocation contents or zeros.
        self.assertTrue(bool(torch.isfinite(out).all()))
        self.assertGreater(float(out[..., -8:, -8:].abs().sum()), 0.0)

    def test_tile_split_covers_the_latent(self) -> None:
        vae = AutoencoderKLHunyuanVideo.from_config(TINY_VAE_CONFIG)
        z = torch.zeros(1, 4, 1, 60, 108)
        tasks, spec = vae._decode_tile_split(z)
        rows, cols = spec.grid_shape
        tile_h, tile_w, stride_h, stride_w = vae._decode_tile_params()
        self.assertGreaterEqual((rows - 1) * stride_h + tile_h, 60)
        self.assertGreaterEqual((cols - 1) * stride_w + tile_w, 108)


if __name__ == "__main__":
    parameterized.absltest.main()
