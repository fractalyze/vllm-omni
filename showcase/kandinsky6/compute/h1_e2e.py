# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Is the H1 arm bytes-bound? Base against base+hybrid, ABBA, with a busy/idle split.

H1 cuts 23-31 s of GEMM time out of a request and Track M's first end-to-end
reading moved by about 1 s. Either the saving is not real end to end or it is
being absorbed, and those have opposite next steps: a kernel problem versus a
bytes problem.

The discriminator is **where the GPU's time goes**, not how long the request
takes. This arm streams the 56 GiB DiT from NVMe every step through distributed
layerwise offload, so each step is some compute and some waiting on the next
block to arrive. If the hybrid removes compute and the request does not move,
**GPU idle must grow by about what compute lost** -- and then the lever is bytes
per step, not instructions.

Utilisation is sampled at 10 Hz for the whole request rather than inferred from
a profiler inside the server: the question is a coarse one (did tens of seconds
move from busy to idle) and a sampler cannot perturb the thing it measures the
way an in-process profiler can.

ABBA, one warm-up and two timed requests per visit, because this host's
request times drift with page-cache state and a single A-then-B comparison has
already been wrong once in this study.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
# `bench/serve.py` imports its siblings by bare name (`gpu_guard`), so the
# package directory has to be importable as well as the package.
sys.path.insert(0, str(HERE.parent / "bench"))

from bench import drive, serve  # noqa: E402
from gpulock import GpuLocks, foreign_gpu_procs  # noqa: E402

VENV_BIN = Path("/data/jooman/k6/venv/bin")
BASE_ENV = {
    "HF_HOME": "/data/jooman/hf",
    "TORCHINDUCTOR_CACHE_DIR": "/data/jooman/k6/inductor-cache",
    "VLLM_OMNI_K6_EXACT_ATTN_STEPS": "1",
    "VLLM_OMNI_K6_PIFLOW_CACHE_STEPS": "8",
    "VLLM_OMNI_K6_PIFLOW_CACHE_MODE": "reuse",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
}
W1 = dict(width=864, height=480, num_frames=121, num_inference_steps=10, guidance_scale=1.0, seed=42)


class Sampler:
    """GPU utilisation and power at 10 Hz, in a thread, with timestamps."""

    def __init__(self) -> None:
        self.rows: list[tuple[float, float, float]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        def loop():
            while not self._stop.is_set():
                try:
                    out = subprocess.run(
                        ["nvidia-smi", "--query-gpu=utilization.gpu,power.draw",
                         "--format=csv,noheader,nounits"],
                        capture_output=True, text=True, timeout=5).stdout.strip()
                    u, p = (float(v) for v in out.split(","))
                    self.rows.append((time.time(), u, p))
                except Exception:  # noqa: BLE001
                    pass
                time.sleep(0.1)

        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def window(self, t0: float, t1: float) -> dict:
        """Busy fraction and seconds over [t0, t1].

        "Busy" is utilisation >= 50%: the GPU reports an instantaneous sampling
        of SM occupancy, and on this arm the pattern is a square wave between
        ~100 (a block computing) and ~0 (waiting on the next block's H2D), so
        the threshold sits in an empty part of the distribution rather than
        slicing through a mode. The `u_hist` field below is reported so that
        assumption is checkable rather than asserted.
        """
        rows = [r for r in self.rows if t0 <= r[0] <= t1]
        if not rows:
            return {}
        span = t1 - t0
        busy = [r for r in rows if r[1] >= 50]
        frac = len(busy) / len(rows)
        hist = {}
        for lo in (0, 10, 25, 50, 75, 90):
            hi = {0: 10, 10: 25, 25: 50, 50: 75, 75: 90, 90: 101}[lo]
            hist[f"{lo}-{hi}"] = sum(1 for r in rows if lo <= r[1] < hi)
        return {"samples": len(rows), "span_s": span,
                "busy_frac": frac, "busy_s": frac * span, "idle_s": (1 - frac) * span,
                "power_w_median_busy": statistics.median([r[2] for r in busy]) if busy else 0.0,
                "u_hist": hist}


# Exactly the 12 large-M linears a fused block issues stay on the hybrid; the 20
# small-M ones (audio branch at M=218, text tower at M=256, and the key/value
# sides of the cross-attentions that read the small stream) go back to cuBLAS,
# where they are 1.7-2.7x faster. Validated against the module names
# `gemm_insitu.py` recorded: 12/12 kept, 0/20 leaked.
LARGE_M_ONLY = (r"modulation|time_embeddings|audio_dec_block|text_transformer_blocks"
                r"|av_cross_attention\.(to_query|out_layer)"
                r"|va_cross_attention\.to_(key|value)"
                r"|video_dec_block\.cross_attention\.to_(key|value)")


def arm_for(name: str, hybrid: bool, port: int) -> serve.Arm:
    env = dict(BASE_ENV)
    if hybrid:
        env["VLLM_OMNI_K6_HYBRID_GEMM"] = "1"
        if "large" in name:
            env["VLLM_OMNI_K6_HYBRID_GEMM_EXCLUDE"] = LARGE_M_ONLY
    return serve.Arm(
        name=name,
        model="kandinskylab/Kandinsky-6.0-Pro-distill-5s-Diffusers",
        cli_args=["--num-gpus", "1",
                  "--enable-distributed-layerwise-offload", "--dlo-no-use-allgather",
                  "--disable-multithread-weight-load",
                  "--diffusion-attention-config",
                  (HERE / "arms" / "sage2-mid.json").read_text()],
        env=env, port=port,
        notes="final stack" + (" + hybrid FP16-accumulate GEMM" if hybrid else ""),
    )


def visit(label: str, hybrid: bool, out_dir: Path, prompt: str, timed: int) -> dict:
    """One server lifetime: warm-up, then `timed` measured requests."""
    out_dir.mkdir(parents=True, exist_ok=True)
    log = out_dir / "server.log"
    arm = arm_for(label, hybrid, port=8097)
    srv = serve.start_server(arm, log, venv_bin=VENV_BIN)
    try:
        text = log.read_text(errors="replace")
        # The proof the kernel is installed, printed by pipeline_kandinsky6 at
        # construction. Captured verbatim rather than paraphrased.
        proof = [ln.strip() for ln in text.splitlines() if "hybrid FP16-accumulate GEMM" in ln]
        sampler = Sampler()
        sampler.start()
        fields = drive.request_fields(prompt, **W1)
        drive.submit_and_fetch(srv.base_url, fields, out_dir / "warmup.mp4")
        runs = []
        for i in range(timed):
            t0 = time.time()
            r = drive.submit_and_fetch(srv.base_url, fields, out_dir / f"r{i}.mp4")
            t1 = time.time()
            runs.append({"seconds": r.request_wall_s, "generate_s": r.generate_s,
                         "window": sampler.window(t0, t1)})
            print(f"  {label} run {i}: {r.request_wall_s:.1f} s  "
                  f"busy {runs[-1]['window'].get('busy_s', 0):.1f} s / "
                  f"idle {runs[-1]['window'].get('idle_s', 0):.1f} s", flush=True)
        sampler.stop()
        if not proof:
            proof = [ln.strip() for ln in log.read_text(errors="replace").splitlines()
                     if "hybrid FP16-accumulate GEMM" in ln]
        return {"label": label, "hybrid": hybrid, "cold_start_s": srv.cold_start_s,
                "install_log": proof, "runs": runs}
    finally:
        serve.stop_server(srv)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--timed", type=int, default=2)
    p.add_argument("--out", type=Path, default=Path("/data/jooman/k6/results/h1-e2e"))
    p.add_argument("--json", type=Path, default=Path("/data/jooman/k6/results/h1-e2e.json"))
    p.add_argument("--plan", choices=("abba", "large-only"), default="abba",
                   help="'large-only' re-runs the hybrid with only the 12 large-M linears wrapped")
    args = p.parse_args()

    prompts = json.loads(Path("/data/jooman/k6/prompts/setA.json").read_text())["prompts"]
    prompt = next(x for x in prompts if x["id"] == "a1-portrait-speech")["text"]

    PLAN = ([("A-base", False), ("B-hybrid", True), ("B-hybrid", True), ("A-base", False)]
            if args.plan == "abba" else
            [("C-hybrid-large-only", True), ("C-hybrid-large-only", True)])
    report = {"order": args.plan, "workload": W1, "visits": []}
    with GpuLocks():
        foreign = foreign_gpu_procs()
        report["foreign_gpu_procs"] = [str(x) for x in foreign]
        if foreign:
            print(f"warning: foreign GPU process(es): {foreign}", file=sys.stderr)
        for i, (label, hybrid) in enumerate(
                PLAN):
            name = f"{label}-{i}"
            print(f"\n=== visit {i}: {name}", flush=True)
            report["visits"].append(visit(name, hybrid, args.out / name, prompt, args.timed))
            args.json.write_text(json.dumps(report, indent=2) + "\n")

    def pool(hyb: bool, key: str) -> list[float]:
        return [r["window"].get(key, 0.0) if key != "seconds" else r["seconds"]
                for v in report["visits"] if v["hybrid"] is hyb for r in v["runs"]]

    print("\n=== ABBA result ===")
    for key, unit in (("seconds", "request"), ("busy_s", "GPU busy"), ("idle_s", "GPU idle")):
        a, b = pool(False, key), pool(True, key)
        if not a or not b:
            continue
        ma, mb = statistics.median(a), statistics.median(b)
        print(f"  {unit:9s}: base {ma:7.1f} s   hybrid {mb:7.1f} s   {mb - ma:+7.1f} s")
        report.setdefault("summary", {})[key] = {"base": ma, "hybrid": mb, "delta": mb - ma}
    args.json.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
