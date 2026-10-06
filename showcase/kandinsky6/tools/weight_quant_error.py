# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""How much does a 1-byte weight format cost the DiT, and does granularity help?

The gate says the FP8 stack spends the whole error budget (LPIPS 0.262 against
BF16, limit 0.15) and both checkpoints on these hosts carry one scale per
tensor. Whether that granularity is the reason is a question about the weights
alone, so it does not need a GPU, a server or a video: quantize each tensor both
ways and compare the error it introduces.

Reported per tensor as relative error ``||W - dequant(quant(W))||_F / ||W||_F``,
which is the quantity a GEMM's output error is proportional to for a
well-conditioned activation distribution. Four combinations are reported, FP8
E4M3 and INT8 each at per-tensor and per-output-row granularity, because the two
formats answer the granularity question differently and that is the finding:
E4M3 carries its own exponent, so a scale only slides the matrix along the
exponent ladder and the error stays at the mantissa's step, while INT8 is fixed
point and the scale *is* the step.

    python weight_quant_error.py --limit 48

Streams one tensor at a time from the safetensors file, so it needs no more
memory than the largest tensor.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch
from safetensors import safe_open

FP8_E4M3_MAX = 448.0
FP8_DTYPE = torch.float8_e4m3fn
INT8_MAX = 127.0

# The formats worth comparing, as (name, representable maximum, rounding dtype).
# FP8 E4M3 carries its own 4-bit exponent, so a scale only moves the whole
# matrix up or down its exponent ladder and the error stays at the 3-bit
# mantissa's step. INT8 is fixed point: there the scale *is* the step, so a
# per-row scale buys real precision. That difference is the point of the tool.
FORMATS = {"fp8": (FP8_E4M3_MAX, FP8_DTYPE), "int8": (INT8_MAX, torch.int8)}


def relative_error(weight: torch.Tensor, *, granularity: str, fmt: str = "fp8") -> float:
    """Relative Frobenius error of one round trip through ``fmt``."""
    maximum, dtype = FORMATS[fmt]
    as_float = weight.to(torch.float32)
    amax = as_float.abs().amax(dim=1, keepdim=True) if granularity == "channel" else as_float.abs().amax()
    scale = (amax / maximum).clamp(min=torch.finfo(torch.float32).tiny)
    scale = torch.where(amax > 0, scale, torch.ones_like(scale))
    scaled = (as_float / scale).clamp(-maximum, maximum)
    # INT8 rounds to nearest; a cast to int8 truncates, which would report an
    # error twice the real one and make the format look worse than it is.
    quantized = torch.round(scaled).to(dtype) if dtype is torch.int8 else scaled.to(dtype)
    round_trip = quantized.to(torch.float32) * scale
    denominator = as_float.norm()
    if denominator == 0:
        return 0.0
    return float((as_float - round_trip).norm() / denominator)


def outlier_ratio(weight: torch.Tensor) -> float:
    """The matrix's amax divided by the median row's amax.

    This is the mechanism in one number: a per-tensor scale is set by the single
    largest weight, so a typical row is quantized with a step this many times
    coarser than its own range would need.
    """
    as_float = weight.to(torch.float32)
    row_amax = as_float.abs().amax(dim=1)
    median = float(row_amax.median())
    return float(row_amax.max()) / median if median > 0 else float("inf")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "/data/jooman/hf/hub/models--kandinskylab--Kandinsky-6.0-Pro-distill-5s-Diffusers/snapshots"
        ),
        help="a model root, or the snapshots directory of one",
    )
    parser.add_argument("--limit", type=int, default=48, help="how many 2-D weights to sample")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    candidates = sorted(args.checkpoint.glob("**/transformer/*.safetensors"))
    if not candidates:
        raise SystemExit(f"no transformer safetensors under {args.checkpoint}")
    rows = []
    with safe_open(str(candidates[0]), framework="pt") as handle:
        names = [name for name in handle.keys() if name.endswith(".weight")]
        # Spread the sample over the stack rather than taking the first N keys,
        # which would all come from one block and say nothing about the ends.
        step = max(1, len(names) // args.limit)
        for name in names[::step][: args.limit]:
            tensor = handle.get_tensor(name)
            if tensor.ndim != 2:
                continue
            per_tensor = relative_error(tensor, granularity="tensor")
            per_channel = relative_error(tensor, granularity="channel")
            int8_per_channel = relative_error(tensor, granularity="channel", fmt="int8")
            int8_per_tensor = relative_error(tensor, granularity="tensor", fmt="int8")
            rows.append(
                {
                    "name": name,
                    "shape": list(tensor.shape),
                    "rel_err_per_tensor": per_tensor,
                    "rel_err_per_channel": per_channel,
                    "rel_err_int8_per_channel": int8_per_channel,
                    "rel_err_int8_per_tensor": int8_per_tensor,
                    "ratio": per_tensor / per_channel if per_channel > 0 else float("inf"),
                    "fp8_over_int8_per_channel": per_channel / int8_per_channel if int8_per_channel > 0 else float("inf"),
                    "outlier_ratio": outlier_ratio(tensor),
                }
            )
            print(
                f"{name.split('.', 2)[-1][:48]:48s} {str(tuple(tensor.shape)):>17s}  "
                f"fp8 {per_tensor:.5f}/{per_channel:.5f}  int8 {int8_per_tensor:.5f}/{int8_per_channel:.5f}"
                f"  fp8row/int8row x{rows[-1]['fp8_over_int8_per_channel']:.2f}"
                f"  amax/med-row {rows[-1]['outlier_ratio']:.1f}",
                flush=True,
            )

    if not rows:
        raise SystemExit("no 2-D weights sampled")
    summary = {
        "sampled": len(rows),
        "median_rel_err_per_tensor": statistics.median(r["rel_err_per_tensor"] for r in rows),
        "median_rel_err_per_channel": statistics.median(r["rel_err_per_channel"] for r in rows),
        "median_ratio": statistics.median(r["ratio"] for r in rows),
        "median_rel_err_int8_per_channel": statistics.median(r["rel_err_int8_per_channel"] for r in rows),
        "median_rel_err_int8_per_tensor": statistics.median(r["rel_err_int8_per_tensor"] for r in rows),
        "median_fp8_over_int8_per_channel": statistics.median(r["fp8_over_int8_per_channel"] for r in rows),
        "max_ratio": max(r["ratio"] for r in rows),
        "median_outlier_ratio": statistics.median(r["outlier_ratio"] for r in rows),
        "worst": max(rows, key=lambda r: r["ratio"])["name"],
    }
    print(
        f"\n{len(rows)} tensors, median relative error:\n"
        f"  FP8 E4M3  per-tensor {summary['median_rel_err_per_tensor']:.5f}   "
        f"per-row {summary['median_rel_err_per_channel']:.5f}   "
        f"(per-tensor costs {summary['median_ratio']:.2f}x, worst {summary['max_ratio']:.2f}x)\n"
        f"  INT8      per-tensor {summary['median_rel_err_int8_per_tensor']:.5f}   "
        f"per-row {summary['median_rel_err_int8_per_channel']:.5f}\n"
        f"  FP8 per-row / INT8 per-row: {summary['median_fp8_over_int8_per_channel']:.2f}x"
    )
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({"summary": summary, "tensors": rows}, indent=2) + "\n")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
