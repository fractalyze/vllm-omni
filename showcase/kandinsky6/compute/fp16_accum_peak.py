# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The two tensor-core ceilings on sm_120, measured rather than quoted.

`gemm_census.py` found Kandinsky 6's large GEMMs at 98-99% of a measured
221.4 TFLOP/s and concluded no kernel could win more than 1-2%. **That ceiling
was the FP32-accumulate one.** On consumer Blackwell, FP16-input MMA with an
FP16 accumulator is documented to run at twice the rate of the same MMA with an
FP32 accumulator, so there is a second ceiling the census never probed, and the
"no kernel to write" conclusion was conditional on an instruction choice rather
than on the hardware.

This measures both, at the same clocks and power limit, plus the hybrid that is
actually usable: FP16 accumulate inside each K-block, promoted to an FP32
running sum every `BK` elements. The point is to size H2's remaining headroom
before tuning -- a 1.35x kernel against a 2.0x ceiling has room, a 1.35x kernel
against a 1.4x ceiling does not.

Peaks are read while the GPU is busy: sampling clocks after
`torch.cuda.synchronize()` reads an idle board and overstates the ceiling, which
cost this study one meaningless 230 TFLOP/s figure already.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import triton  # noqa: E402
import triton.language as tl  # noqa: E402

from gpulock import GpuLocks, foreign_gpu_procs  # noqa: E402


@triton.jit
def _mm(A, B, C, M, N, K, sam, sak, sbk, sbn, scm, scn,
        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
        MODE: tl.constexpr, GM: tl.constexpr):
    """One GEMM in three accumulate modes.

    MODE 0  FP32 accumulator throughout -- what cuBLAS and the census measured.
    MODE 1  FP16 accumulator throughout -- the ceiling probe. Not a usable
            kernel: a K=4096 reduction in FP16 loses far too much, and it exists
            only to measure how fast the hardware can go.
    MODE 2  the hybrid -- FP16 accumulate within each BK block, promoted into an
            FP32 running sum. Error is bounded by BK rather than by K, which is
            what makes it usable.
    """
    pid = tl.program_id(0)
    npm, npn = tl.cdiv(M, BM), tl.cdiv(N, BN)
    g = pid // (GM * npn)
    fm = g * GM
    gs = min(npm - fm, GM)
    pm = fm + (pid % gs)
    pn = (pid % (GM * npn)) // gs
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    a = A + rm[:, None] * sam + rk[None, :] * sak
    b = B + rk[:, None] * sbk + rn[None, :] * sbn

    if MODE == 1:
        acc16 = tl.zeros((BM, BN), dtype=tl.float16)
        for _ in range(0, K, BK):
            x = tl.load(a, mask=rm[:, None] < M, other=0.0)
            y = tl.load(b)
            acc16 += tl.dot(x, y, out_dtype=tl.float16)
            a += BK * sak
            b += BK * sbk
        acc = acc16.to(tl.float32)
    else:
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for _ in range(0, K, BK):
            x = tl.load(a, mask=rm[:, None] < M, other=0.0)
            y = tl.load(b)
            if MODE == 2:
                acc += tl.dot(x, y, out_dtype=tl.float16).to(tl.float32)
            else:
                acc = tl.dot(x, y, acc)
            a += BK * sak
            b += BK * sbk

    c = C + rm[:, None] * scm + rn[None, :] * scn
    tl.store(c, acc.to(tl.float16), mask=rm[:, None] < M)


def board_under_load(run, seconds: float = 2.0) -> dict:
    """Clocks and power sampled WHILE the GPU is working."""
    import subprocess
    import threading
    import time

    import torch

    samples: list[dict] = []
    stop = threading.Event()

    def sample():
        while not stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu",
                     "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5).stdout.strip()
                sm, mem, pw, t, u = (float(v) for v in out.split(","))
                samples.append({"sm_mhz": sm, "mem_mhz": mem, "power_w": pw, "temp_c": t, "util_pct": u})
            except Exception:  # noqa: BLE001
                pass
            time.sleep(0.1)

    th = threading.Thread(target=sample, daemon=True)
    th.start()
    t_end = time.time() + seconds
    while time.time() < t_end:
        run()
    torch.cuda.synchronize()
    stop.set()
    th.join(timeout=2)
    busy = [s for s in samples if s["util_pct"] >= 50] or samples
    if not busy:
        return {}
    return {k: round(statistics.median(s[k] for s in busy), 2) for k in busy[0]}


def time_kernel(run, *, warmup: int = 5, iters: int = 20) -> float:
    import torch

    for _ in range(warmup):
        run()
    torch.cuda.synchronize()
    s = []
    for _ in range(iters):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        run()
        e1.record()
        torch.cuda.synchronize()
        s.append(e0.elapsed_time(e1) / 1e3)
    return statistics.median(s)


MODES = {0: "fp32 acc", 1: "fp16 acc (ceiling probe)", 2: "hybrid (fp16 acc, fp32 promote per BK)"}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--m", type=int, default=50220)
    p.add_argument("--k", type=int, default=4096)
    p.add_argument("--n", type=int, default=16384)
    p.add_argument("--json", type=Path, default=None)
    p.add_argument("--no-locks", action="store_true")
    args = p.parse_args()

    import torch

    M, K, N = args.m, args.k, args.n
    flop = 2 * M * K * N
    report: dict = {"shape": [M, K, N], "torch": torch.__version__,
                    "device": torch.cuda.get_device_name(0), "rows": []}

    with contextlib.nullcontext() if args.no_locks else GpuLocks():
        foreign = foreign_gpu_procs()
        report["foreign_gpu_procs"] = [str(x) for x in foreign]
        if foreign:
            print(f"warning: foreign GPU process(es): {foreign}", file=sys.stderr)

        a = torch.randn(M, K, device="cuda", dtype=torch.float16)
        b = (torch.randn(K, N, device="cuda", dtype=torch.float16) * 0.02)
        c = torch.empty(M, N, device="cuda", dtype=torch.float16)
        # fp64 reference on a slice: the whole M would not fit, and the error is
        # a per-row property so a slice measures it.
        rows = 4096
        ref = a[:rows].double() @ b.double()

        s = time_kernel(lambda: a @ b)
        board = board_under_load(lambda: a @ b)
        print(f"cuBLAS fp16 in / fp32 acc : {flop / s / 1e12:7.1f} TF   "
              f"sm {board.get('sm_mhz')} MHz, {board.get('power_w')} W")
        report["cublas"] = {"tflops": flop / s / 1e12, "board": board}

        best: dict = {}
        for mode in (0, 1, 2):
            for BM, BN, BK, w, st in ((128, 256, 64, 8, 3), (128, 128, 64, 4, 4),
                                      (256, 128, 64, 8, 3), (128, 128, 32, 4, 4)):
                def run(BM=BM, BN=BN, BK=BK, w=w, st=st, mode=mode):
                    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
                    _mm[grid](a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                              c.stride(0), c.stride(1), BM=BM, BN=BN, BK=BK, MODE=mode, GM=8,
                              num_warps=w, num_stages=st)
                try:
                    t = time_kernel(run)
                    err = ((c[:rows].double() - ref).norm() / ref.norm()).item()
                    tf = flop / t / 1e12
                    row = {"mode": mode, "mode_name": MODES[mode], "BM": BM, "BN": BN, "BK": BK,
                           "warps": w, "stages": st, "tflops": tf, "rel_l2": err}
                    report["rows"].append(row)
                    print(f"  mode {mode} {MODES[mode]:40s} {BM}x{BN}x{BK} w{w} s{st}: "
                          f"{tf:7.1f} TF  relL2 {err:.2e}")
                    if tf > best.get(mode, {"tflops": 0})["tflops"]:
                        best[mode] = row
                except Exception as exc:  # noqa: BLE001
                    print(f"  mode {mode} {BM}x{BN}x{BK} w{w} s{st}: FAIL {str(exc)[:70]}")

        print()
        for mode in sorted(best):
            r = best[mode]
            print(f"best mode {mode} ({MODES[mode]:40s}): {r['tflops']:7.1f} TF  relL2 {r['rel_l2']:.2e}")
        if 0 in best and 1 in best:
            print(f"\nFP16-accumulate ceiling is {best[1]['tflops'] / best[0]['tflops']:.2f}x the "
                  f"FP32-accumulate one (spec claim: 2.00x)")
        if 0 in best and 2 in best:
            print(f"the usable hybrid is {best[2]['tflops'] / best[0]['tflops']:.2f}x, "
                  f"so H2 headroom to the ceiling is "
                  f"{best[1]['tflops'] / best[2]['tflops']:.2f}x")
        report["best"] = best

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + "\n")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
