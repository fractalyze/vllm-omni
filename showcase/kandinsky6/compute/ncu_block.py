# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""One compiled Kandinsky 6 fused block at W1, for Nsight Compute.

Profiling a whole request is not possible: ncu replays every kernel 8-40x. This
builds the single unit the brief asks for -- one fused block at W1's token counts
with the served attention arm and the hybrid GEMM installed -- warms it so
Triton/Inductor JIT is out of the way, then opens a bounded profiler region
around exactly one forward.

Run under:
    ncu --target-processes all --profile-from-start off --clock-control none \
        --section SpeedOfLight ... -c 40 --export out --force-overwrite \
        python ncu_block.py

`--clock-control none` matters on this card: the default locks base clock, and
every number this study has is at the real 575 W-capped clocks.

The attention config is `sage2-edge0.json`, not `sage2-mid.json`, deliberately.
A block built at prefix `visual_transformer_blocks.0` is *outside* sage2-mid's
`6:54` band and would resolve to the platform default, so profiling it would
measure cuDNN rather than the SageAttention2 kernel the served arm runs on 48 of
its 60 blocks.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "bench"))
sys.path.insert(0, "/data/jooman/k6/vllm-omni")

LARGE_ONLY = (r"modulation|time_embeddings|audio_dec_block|text_transformer_blocks"
              r"|av_cross_attention\.(to_query|out_layer)"
              r"|va_cross_attention\.to_(key|value)"
              r"|video_dec_block\.cross_attention\.to_(key|value)")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--geometry", default="w1")
    p.add_argument("--arm", default=str(HERE / "arms" / "sage2-edge0.json"))
    p.add_argument("--hybrid", action="store_true", default=True)
    p.add_argument("--no-hybrid", dest="hybrid", action="store_false")
    p.add_argument("--warmups", type=int, default=3)
    args = p.parse_args()

    import torch

    import block_profile as bp
    from gpulock import GpuLocks, foreign_gpu_procs
    from vllm_omni.diffusion.models.kandinsky6.hybrid_linear import install_hybrid

    cfg = bp.CONFIGS["pro"]
    shapes = bp.Shapes(**bp.GEOMETRIES[args.geometry])
    dev, dt = torch.device("cuda"), torch.bfloat16

    with GpuLocks():
        foreign = foreign_gpu_procs()
        print(f"foreign GPU procs before: {foreign or 'none'}", flush=True)
        with bp.single_process_parallel():
            module = bp.build_target("fused", cfg, None, dev, dt,
                                     attention_config_file=Path(args.arm))
            n = 0
            if args.hybrid:
                n, kept = install_hybrid(module, exclude=LARGE_ONLY)
            print(f"hybrid linears wrapped: {n}", flush=True)
            inputs = bp.make_inputs("fused", cfg, shapes, dev, dt)
            module = bp.maybe_compile(module, "default")

            with torch.no_grad():
                for _ in range(args.warmups):
                    module(**inputs)
                torch.cuda.synchronize()
                print("warm; opening the profiler region", flush=True)
                torch.cuda.profiler.start()
                module(**inputs)
                torch.cuda.synchronize()
                torch.cuda.profiler.stop()
            print("region closed", flush=True)
        print(f"foreign GPU procs after: {foreign_gpu_procs() or 'none'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
