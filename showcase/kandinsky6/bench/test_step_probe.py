# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""step_probe.py finds what the pipeline's probe sink wrote, and encodes it."""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np
from absl.testing import absltest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import step_probe  # noqa: E402


class StepProbeTest(absltest.TestCase):
    def test_directory_name_matches_the_pipeline_sink(self) -> None:
        """The scorer maps directories back to prompt ids by recomputing the sink's name."""
        from vllm_omni.diffusion.models.kandinsky6.pipeline_kandinsky6 import Kandinsky6TI2VAPipeline

        with tempfile.TemporaryDirectory() as tmp:
            probe = Path(tmp) / "probe.json"
            probe.write_text(f'{{"out_dir": "{tmp}/out", "branches": []}}')
            pipeline = object.__new__(Kandinsky6TI2VAPipeline)
            with mock.patch.dict("os.environ", {"VLLM_OMNI_K6_STEP_PROBE": str(probe)}):
                sink = Kandinsky6TI2VAPipeline._step_probe_sink(pipeline, "a prompt", 42)
            sink("base", frames=np.zeros((2, 4, 4, 3), dtype=np.uint8))
            written = [p.name for p in (Path(tmp) / "out").iterdir()]
        self.assertEqual(written, [step_probe.probe_dir_name("a prompt", 42)])

    def test_encode_writes_one_mp4_per_known_prompt(self) -> None:
        prompts = step_probe.load_prompts()
        prompt_id = "a3-sprint-start"
        with tempfile.TemporaryDirectory() as tmp:
            request_dir = Path(tmp) / "out" / step_probe.probe_dir_name(prompts[prompt_id], step_probe.SEED)
            request_dir.mkdir(parents=True)
            frames = np.random.default_rng(0).integers(0, 255, (8, 64, 96, 3), dtype=np.uint8)
            np.save(request_dir / "reuse-8.npy", frames)
            (Path(tmp) / "out" / "unknown-s42").mkdir()
            args = argparse.Namespace(out_dir=Path(tmp) / "out", branch="reuse-8", mp4_dir=Path(tmp) / "mp4")
            step_probe.encode(args)
            self.assertEqual([p.name for p in (Path(tmp) / "mp4").iterdir()], [f"{prompt_id}.mp4"])


if __name__ == "__main__":
    absltest.main()
