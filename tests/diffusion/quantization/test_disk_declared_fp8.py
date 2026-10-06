# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""A checkpoint that declares ``quant_method: fp8`` is a serialized FP8 checkpoint.

A ``quantization_config`` inside ``transformer/config.json`` describes weights
already on disk. vLLM-Omni's factory maps the name ``fp8`` to the *online*
``DiffusionFp8Config``, and resolving a disk-declared config through it broke two
things at once: the model was built with BF16 parameters that cannot take the
checkpoint's ``weight_scale``, and the loader, seeing an online config, loaded on
the GPU and then moved the whole pipeline to the host -- on Kandinsky 6 Pro that
put the 16.6 GB text encoder back into host RAM and the worker into the OOM
killer. The loader's offline check now also reads ``is_checkpoint_fp8_serialized``.
"""

from absl.testing import absltest

from vllm_omni.diffusion.data import TransformerConfig

FP8_DECLARED = {
    "in_visual_dim": 16,
    "quantization_config": {
        "quant_method": "fp8",
        "activation_scheme": "dynamic",
        "weight_block_size": None,
        "ignored_layers": ["out_layer.out_layer"],
    },
}


class DiskDeclaredFp8Test(absltest.TestCase):
    def test_resolves_to_serialized_config(self) -> None:
        config = TransformerConfig.from_dict(FP8_DECLARED)
        self.assertEqual(config.quant_method, "fp8")
        self.assertTrue(config.quant_config.is_checkpoint_fp8_serialized)

    def test_keeps_ignored_layers(self) -> None:
        config = TransformerConfig.from_dict(FP8_DECLARED)
        self.assertEqual(config.quant_config.ignored_layers, ["out_layer.out_layer"])

    def test_no_declared_config_stays_unquantized(self) -> None:
        config = TransformerConfig.from_dict({"in_visual_dim": 16})
        self.assertIsNone(config.quant_config)


if __name__ == "__main__":
    absltest.main()
