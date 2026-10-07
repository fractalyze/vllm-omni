# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The census must list every GEMM the block actually issues.

This exists because it did not. `gemm_census.py` was written by reading
`Kandinsky6FusedTransformerDecoderBlock.forward` and it missed three full-size
projections at W1's 50,220 tokens -- `va_cross_attention.out_layer` and both
halves of the decoder block's *text* cross-attention. The missing 1.40 s/step
showed up as a gap between the census and the profiled GEMM bucket, and that gap
was investigated as if it were a kernel problem.

A census that misses a GEMM inflates exactly the quantity it was built to
measure, and inflates it in the direction that looks like a discovery. So the
invariant worth testing is not any particular number: it is that the row list is
derived from the same module tree the block builds.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from absl.testing import absltest

sys.path.insert(0, str(Path(__file__).resolve().parent))


class CensusCoversTheBlockTest(absltest.TestCase):
    def setUp(self):
        super().setUp()
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
        os.environ.setdefault("MASTER_PORT", "29536")
        self.enterContext(set_current_vllm_config(VllmConfig()))
        if not model_parallel_is_initialized():
            init_distributed_environment(world_size=1, rank=0, local_rank=0, distributed_init_method="env://")
            initialize_model_parallel()
            self.addCleanup(cleanup_dist_env_and_memory)

    def _tiny_fused_block(self):
        """One fused block at small dimensions. Dimensions do not matter here --
        only which projections exist, which is what the census enumerates."""
        from types import SimpleNamespace

        from vllm_omni.diffusion.config import set_current_diffusion_config
        from vllm_omni.diffusion.data import AttentionConfig, AttentionSpec
        from vllm_omni.diffusion.models.kandinsky6.kandinsky6_transformer import (
            Kandinsky6FusedTransformerDecoderBlock,
        )

        config = SimpleNamespace(
            diffusion_attention_config=AttentionConfig(default=AttentionSpec(backend="TORCH_SDPA")),
            parallel_config=SimpleNamespace(ring_degree=1, allgather_degree=1),
        )
        with set_current_diffusion_config(config):
            return Kandinsky6FusedTransformerDecoderBlock(
                24, 16, 32, 12, 24, 16, 32, 12, text_token_padding=False,
                ca_rope=True, cross_gates=True, fix_modulation=True, prefix="visual_transformer_blocks.0",
            )

    def test_every_visual_stream_projection_is_in_the_census(self):
        """The census's visual-side GEMM count must equal the block's.

        The visual stream is the one that matters: at W1 it carries 50,220 of the
        50,694 tokens in a block, so a missed projection there is seconds a step
        while a missed audio one is milliseconds. Counted as *GEMMs*, not rows --
        the census fuses QKV into one row standing for three calls.
        """
        from vllm.model_executor.layers.linear import LinearBase

        from gemm_census import W1_VISUAL_TOKENS, block_gemms

        block = self._tiny_fused_block()
        model_dim = 24

        # Every linear in the block whose input is the visual stream. Identified
        # structurally: the visual branch is `video_dec_block` plus the two
        # projections of each cross-attention that face the visual side.
        visual_calls = []
        for name, mod in block.named_modules():
            if not isinstance(mod, LinearBase):
                continue
            if name.startswith("video_dec_block") and "modulation" not in name:
                # A cross-attention's key/value side reads the *other* stream.
                # `video_dec_block.cross_attention` attends to the 256 text
                # tokens, so its to_key/to_value are text-side while its
                # to_query and out_layer carry the visual stream. `gemm_insitu.py`
                # confirms it: 12 calls run at large M and these two are not
                # among them.
                if name.endswith(("cross_attention.to_key", "cross_attention.to_value")):
                    continue
                visual_calls.append(name)
            elif name.startswith("va_cross_attention") and name.endswith(("to_query", "out_layer")):
                visual_calls.append(name)
            elif name.startswith("av_cross_attention") and name.endswith(("to_key", "to_value")):
                visual_calls.append(name)

        census_calls = sum(
            # A row's N/model_dim is how many d-wide projections it stands for:
            # QKV at 3*d is three calls, av_cross's fused KV at 2*d_a is two.
            max(1, round(g["n"] / model_dim)) if g["name"] in {"visual.qkv", "cross.kv_from_visual"} else 1
            for g in block_gemms(d=model_dim, ff=32, d_a=24, ff_a=32)
            if g["m"] == W1_VISUAL_TOKENS
        )

        self.assertEqual(
            census_calls, len(visual_calls),
            f"the census accounts for {census_calls} visual-stream GEMMs but the block issues "
            f"{len(visual_calls)}: {sorted(visual_calls)}. A census that misses a GEMM inflates "
            f"the gap it was built to measure -- add the row rather than adjusting the gap.",
        )

    def test_the_census_marks_bias_the_way_the_block_does(self):
        """`bias` on each row must match the block, because the bias is not free:
        on sm_120 a fused bias costs ~21% of a large GEMM (see PR #38)."""
        from vllm.model_executor.layers.linear import LinearBase

        from gemm_census import block_gemms

        block = self._tiny_fused_block()
        has_bias = {n: getattr(m, "bias", None) is not None
                    for n, m in block.named_modules() if isinstance(m, LinearBase)}

        # The attention projections carry a bias; every FFN does not. If that
        # ever stops being true the census's bias column is silently wrong.
        attn = [n for n in has_bias if "attention" in n and "modulation" not in n]
        ffn = [n for n in has_bias if "feed_forward" in n]
        self.assertTrue(attn and ffn)
        self.assertTrue(all(has_bias[n] for n in attn), f"expected a bias on every attention projection: {attn}")
        self.assertFalse(any(has_bias[n] for n in ffn), f"expected no bias on any FFN projection: {ffn}")

        rows = {g["name"]: g["bias"] for g in block_gemms()}
        for name in ("visual.qkv", "visual.attn_out", "visual.text_cross_q", "visual.text_cross_out",
                     "cross.q_from_visual", "cross.out_from_visual", "cross.kv_from_visual"):
            self.assertTrue(rows[name], f"{name} is an attention projection and must be marked bias=True")
        for name in ("visual.ff1", "visual.ff2", "audio.ff1", "audio.ff2", "text.ff1"):
            self.assertFalse(rows[name], f"{name} is an FFN and must be marked bias=False")


if __name__ == "__main__":
    absltest.main()
