# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""R3: what a block-Hadamard rotation does to NVFP4 and INT8 quantization error, on real K6 tensors.

Input: ``<name>.call<i>.pt`` files, each a dict of real activation rows ``x`` (a
subsample of one linear's input on one sampler step) and that linear's BF16
weight ``w`` (on call 0), captured from a served W1 request. For every layer and
rotation block ``b`` (0 = none) it reports the relative L2 error of the layer's
output against the exact product, in FP64:

  nvfp4     x and W both NVFP4 (the H5 fast mode's GEMM)
  int8-w    W alone INT8 per output row, x exact (the weight-only checkpoint)

Rotating x and W by the same orthogonal matrix leaves ``x W^T`` unchanged, so
any difference is the quantizer's.

    python rotation_error.py --dump-dir /home/jooman/k6/acts --json out.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

from vllm_omni.diffusion.models.kandinsky6.step_precision import nvfp4_linear, rotate_blocks

BLOCKS = (0, 16, 32, 64, 128)


def int8_rows(w: torch.Tensor) -> torch.Tensor:
    """Fake-quantize ``w`` to INT8 with one FP32 scale per output row, as the weight-only checkpoint does."""
    scale = w.float().abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / 127.0
    return ((w.float() / scale).round().clamp(-127, 127) * scale).to(w.dtype)


def rel(out: torch.Tensor, ref: torch.Tensor) -> float:
    return float((out.double() - ref).norm() / ref.norm())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dump-dir", type=Path, required=True)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    weights = {p.name.split(".call")[0]: torch.load(p)["w"] for p in args.dump_dir.glob("*.call0.pt")}
    rows = []
    for path in sorted(args.dump_dir.glob("*.pt")):
        name, call = path.name.split(".call")[0], int(path.name.split(".call")[1].split(".")[0])
        x = torch.load(path)["x"].cuda()
        w = weights[name].cuda()
        ref = x.double() @ w.double().T
        row = {"layer": name, "call": call, "m": x.shape[0], "k": x.shape[1], "n": w.shape[0]}
        for b in BLOCKS:
            if b and x.shape[1] % b:
                continue
            row[f"nvfp4_b{b}"] = rel(nvfp4_linear(x, w, None, b), ref)
            xr, wr = (rotate_blocks(x, b), rotate_blocks(w, b)) if b else (x, w)
            row[f"int8w_b{b}"] = rel(xr.double() @ int8_rows(wr).double().T, ref)
        rows.append(row)
        print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)

    summary = {}
    for kind in ("nvfp4", "int8w"):
        for b in BLOCKS:
            vals = [r[f"{kind}_b{b}"] for r in rows if f"{kind}_b{b}" in r]
            if vals:
                summary[f"{kind}_b{b}"] = {"mean": statistics.fmean(vals), "max": max(vals), "n": len(vals)}
    for key, v in summary.items():
        print(f"{key:12s} mean {v['mean']:.4f}  max {v['max']:.4f}  (n={v['n']})")
    if args.json:
        args.json.write_text(json.dumps({"rows": rows, "summary": summary}, indent=1))


if __name__ == "__main__":
    main()
