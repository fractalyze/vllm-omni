# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Score a served arm's whole prompt set against the canonical BF16 reference.

A thin aggregator over Track M's ``bench/quality.py``: it finds the matching
reference and candidate MP4 for every prompt, calls their ``score_set`` so
both tracks' numbers come from one implementation, and then applies the
**user's adoption gate** -- per prompt set, LPIPS mean <= 0.15 and max <= 0.25
-- on top of the tier their scorer already reports.

Two bars, deliberately: the gate says whether an arm may ship, the tier says
what to call it. An arm can clear the gate and still be ``lossy``.

    python gate_pro_score.py --arm-dir /data/jooman/k6/results/gate-pro/tuned \
        --reference-dir /data/jooman/k6/ref/setA --json scores/pro-a-tuned.json

Scoring runs on the GPU, so it takes the host locks unless ``--no-locks`` is
passed -- a scorer that ignores them can steal memory from a timed run, which
has happened.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path

BENCH = Path(__file__).resolve().parents[1] / "bench"
sys.path.insert(0, str(BENCH))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gpulock import GpuLocks  # noqa: E402
from quality import score_set  # noqa: E402

ADOPTION_MEAN_LPIPS = 0.15
ADOPTION_MAX_LPIPS = 0.25

# G2, the coordinator's working gate (2026-10-07 00:25), pending the user.
# Rationale, which belongs beside the constant: the literal gate above is
# absolute, and on this pipeline *rerunning the same configuration in a fresh
# process* already moves LPIPS, because Inductor picks kernels by timing. A bar
# an arm can only clear by being bit-exact is not a bar on the arm, it is a bar
# on the noise. G2 therefore measures an arm against the pipeline's own
# numerical floor -- BF16 compiled against BF16 eager at the same seed -- and
# allows a quarter more than that floor.
G2_FLOOR_SLACK = 1.25


def g2_verdict(
    set_mean: float,
    set_max: float,
    floor_mean: float | None,
    floor_max: float | None,
    *,
    slack: float = G2_FLOOR_SLACK,
) -> dict[str, object]:
    """The floor-relative verdict, or an explicit refusal to guess one.

    Returns ``decidable: False`` when the floor is unknown rather than assuming
    a floor of zero, which would silently restate the literal gate under a
    second name.
    """
    if floor_mean is None or floor_max is None:
        return {"decidable": False, "reason": "no measured floor supplied (--g2-floor-mean/--g2-floor-max)"}
    limit_mean = slack * floor_mean
    limit_max = slack * floor_max
    return {
        "decidable": True,
        "floor_mean": floor_mean,
        "floor_max": floor_max,
        "slack": slack,
        "limit_mean": limit_mean,
        "limit_max": limit_max,
        "passes": bool(set_mean <= limit_mean and set_max <= limit_max),
        "mean_over_floor": set_mean / floor_mean if floor_mean > 0 else float("inf"),
        "max_over_floor": set_max / floor_max if floor_max > 0 else float("inf"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm-dir", type=Path, required=True, help="a gate_pro.py output directory")
    parser.add_argument("--reference-dir", type=Path, default=Path("/data/jooman/k6/ref/setA"))
    parser.add_argument("--prompts", type=Path, default=BENCH / "prompts" / "setA.json")
    parser.add_argument("--floor-mean", type=float, default=None, help="the encode floor's mean LPIPS, if measured")
    parser.add_argument(
        "--g2-floor-mean",
        type=float,
        default=None,
        help="the numerical floor's set mean LPIPS (BF16 compiled vs BF16 eager), for the G2 working gate",
    )
    parser.add_argument(
        "--g2-floor-max",
        type=float,
        default=None,
        help="the numerical floor's set max LPIPS, for the G2 working gate",
    )
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--no-locks", action="store_true")
    args = parser.parse_args()

    prompts = {p["id"]: p for p in json.loads(args.prompts.read_text())["prompts"]}
    arm_manifest = json.loads((args.arm_dir / "manifest.json").read_text())

    pairs = []
    missing = []
    for run in arm_manifest["runs"]:
        if "error" in run:
            missing.append(f"{run['id']} (the arm's request failed)")
            continue
        prompt_id = run["id"]
        reference = args.reference_dir / f"{prompt_id}.mp4"
        candidate = args.arm_dir / run["mp4"]
        if not reference.is_file():
            missing.append(f"{prompt_id} (no reference at {reference})")
            continue
        pairs.append((prompt_id, prompts.get(prompt_id, {}).get("categories", []), reference, candidate))

    if missing:
        # Named rather than silently dropped: a set scored over 7 of 9 prompts
        # is not the same gate, and the two that are missing are exactly the
        # ones a reader would want to know about.
        print("not scored: " + "; ".join(missing), file=sys.stderr)
    if not pairs:
        raise SystemExit(f"nothing to score in {args.arm_dir}")

    with contextlib.nullcontext() if args.no_locks else GpuLocks():
        result = score_set(pairs, floor_mean=args.floor_mean)

    # score_set returns a flat dict: lpips_mean/lpips_max over the set,
    # per_prompt rows, by_category rollups and its own tier.
    set_mean = result["lpips_mean"]
    set_max = result["lpips_max"]
    passes = set_mean <= ADOPTION_MEAN_LPIPS and set_max <= ADOPTION_MAX_LPIPS
    over = [p["prompt_id"] for p in result["per_prompt"] if p["video"]["lpips_max"] > ADOPTION_MAX_LPIPS]

    verdict = {
        "arm": arm_manifest.get("arm"),
        "attention_config": arm_manifest.get("attention_config"),
        "compile_mode": arm_manifest.get("compile_mode"),
        "checkpoint": arm_manifest.get("checkpoint"),
        "reference": str(args.reference_dir),
        "prompts_scored": len(pairs),
        "prompts_not_scored": missing,
        "set_lpips_mean": set_mean,
        "set_lpips_max": set_max,
        "adoption_mean_limit": ADOPTION_MEAN_LPIPS,
        "adoption_max_limit": ADOPTION_MAX_LPIPS,
        "passes_adoption_gate": bool(passes),
        "prompts_over_adoption_max": over,
        "tier": result.get("tier"),
        "g2_working_gate": g2_verdict(set_mean, set_max, args.g2_floor_mean, args.g2_floor_max),
        "passes_approx_tier": result.get("gate_pass_approx"),
        "request_seconds_median": _median_seconds(arm_manifest),
    }

    print(f"\narm {verdict['arm']}: {len(pairs)} prompt(s) against {args.reference_dir.name}")
    for prompt in result["per_prompt"]:
        video = prompt["video"]
        flag = "  <-- over the max" if video["lpips_max"] > ADOPTION_MAX_LPIPS else ""
        print(
            f"  {prompt['prompt_id']:<22} {','.join(prompt['categories']):<24} "
            f"LPIPS mean {video['lpips_mean']:.4f} max {video['lpips_max']:.4f} "
            f"PSNR {video['psnr_mean']:.1f} dB SSIM {video['ssim_mean']:.3f}{flag}"
        )
    for name, stats in (result.get("by_category") or {}).items():
        print(f"  category {name:<14} mean {stats.get('lpips_mean', float('nan')):.4f} "
              f"max {stats.get('lpips_max', float('nan')):.4f}")
    g2 = verdict["g2_working_gate"]
    if g2["decidable"]:
        print(
            f"\n  G2 (floor-relative, coordinator-chosen pending the user): floor mean "
            f"{g2['floor_mean']:.4f} max {g2['floor_max']:.4f}, limits {g2['limit_mean']:.4f}/"
            f"{g2['limit_max']:.4f} at {g2['slack']}x -> "
            + ("PASSES" if g2["passes"] else "FAILS")
            + f" (this arm is {g2['mean_over_floor']:.2f}x the floor's mean, "
            f"{g2['max_over_floor']:.2f}x its max)"
        )
    else:
        print(f"\n  G2 (floor-relative): not decided -- {g2['reason']}")
    print(
        f"\n  worst prompt {result.get('worst_prompt')} frame {result.get('worst_prompt_frame')}"
        f"\n  G1 set mean {set_mean:.4f} (limit {ADOPTION_MEAN_LPIPS}), max {set_max:.4f} "
        f"(limit {ADOPTION_MAX_LPIPS}) -> " + ("PASSES the adoption gate" if passes else "FAILS the adoption gate")
        + f"; tier {verdict['tier']}"
    )
    if over:
        print("  over the max on their own: " + ", ".join(over))

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({"verdict": verdict, "detail": result}, indent=2) + "\n")
        print(f"  wrote {args.json}")
    return 0


def _median_seconds(manifest: dict) -> float | None:
    times = sorted(r["seconds"] for r in manifest["runs"] if "seconds" in r)
    if not times:
        return None
    middle = len(times) // 2
    return round(times[middle] if len(times) % 2 else (times[middle - 1] + times[middle]) / 2, 3)


if __name__ == "__main__":
    raise SystemExit(main())
