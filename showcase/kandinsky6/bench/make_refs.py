# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Generate quality-gate MP4s for a prompt set from a running server.

Used twice: against the BF16 reference server (``serve/serve_pro_bf16_ref.sh``)
to build the reference set, and against a candidate arm to produce what the gate
scores. Same prompts, same seed, same W1 geometry, one MP4 per prompt named by
prompt id, so ``quality.score_set`` can pair the two directories file by file.

Existing outputs are skipped, so an interrupted reference run resumes instead of
regenerating an hour of work. A ``manifest.json`` beside the MP4s records the
server it came from and each request's wall time.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from drive import request_fields, submit_and_fetch  # noqa: E402

W1 = {"width": 864, "height": 480, "num_frames": 121, "num_inference_steps": 10, "guidance_scale": 1.0}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8094")
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label", required=True, help="what produced these (arm name, 'bf16-reference')")
    parser.add_argument("--limit", type=int, default=0, help="only the first N prompts (0 = all)")
    parser.add_argument("--only", default="", help="comma-separated prompt ids to generate, in set order")
    args = parser.parse_args()

    prompts = json.loads(args.prompts.read_text())["prompts"]
    if args.only:
        wanted = set(args.only.split(","))
        prompts = [p for p in prompts if p["id"] in wanted]
    if args.limit:
        prompts = prompts[: args.limit]
    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"label": args.label, "items": {}}

    for prompt in prompts:
        mp4 = args.out / f"{prompt['id']}.mp4"
        if mp4.exists() and prompt["id"] in manifest["items"]:
            print(f"{prompt['id']}: exists, skipped", flush=True)
            continue
        started = time.time()
        result = submit_and_fetch(args.base_url, request_fields(prompt["text"], seed=args.seed, **W1), mp4)
        manifest["items"][prompt["id"]] = {
            "categories": prompt["categories"],
            "request_wall_s": result.request_wall_s,
            "started": started,
            "seed": args.seed,
            "geometry": W1,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2))
        print(f"{prompt['id']}: {result.request_wall_s:.1f}s -> {mp4}", flush=True)


if __name__ == "__main__":
    main()
