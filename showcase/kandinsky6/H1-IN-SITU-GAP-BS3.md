# Where the hybrid GEMM's "missing gain" went: nowhere — the yardstick was wrong

Answer to "find where the missing ~60% of the hybrid gain goes in situ".
Measured on build-server-3, 2026-10-07 19:50–20:13 KST, GPU locks held, no
foreign GPU process. Delivered through git because bs3 cannot ssh bs2.

**All three hypotheses are refuted, and there is no large gap left to explain.**
The in-situ kernel is within 3% of the microbench on every shape. The "~60%"
came from my own withdrawn GEMM estimate, not from the kernel.

## The premise, corrected first

The 60% figure traces to a −23…−31 s/request GEMM saving I reported in round 4
and withdrew: it came from an isolated GEMM *sequence* benchmark that read
154.5 ms a block where the port's own fused block spends **124.7 ms** on those
same GEMMs. With the module's baseline:

| | |
|---|---:|
| module GEMM work, base → hybrid | 124.70 → 89.58 ms = **1.392×** |
| microbench, same shapes | 1.42–1.51× |
| module block total | 377.52 → 351.46 ms = **−14.1 s** a request |
| end-to-end ABBA, GPU busy | **−10.3 s** |
| **unexplained** | **3.8 s, 27%** |

3.8 s of a 150 s request, measured by 10 Hz utilisation sampling, is close to
that measurement's own resolution.

## (1) Power and clock: refuted — the ratio is clock-invariant

60 s sustained against a 3 s burst, clocks and power sampled throughout:

| shape | burst 3 s | sustained 60 s | clock (burst → sustained) | power |
|---|---:|---:|---|---:|
| d→d | 1.626× | **1.633×** | 2227 → 2145 MHz | 458 → 538 W |
| ff1 | 1.644× | **1.639×** | 2137 → 2115 MHz | 574 → 575 W |
| ff2 | 1.528× | **1.528×** | 2040 → 2235 MHz | 574 → 575 W |

The board does sag — d→d loses 82 MHz and gains 80 W between burst and
sustained — but **both arms sag together**, so the ratio does not move. Both are
tensor-core-bound at whatever clock they get. **A sustained-versus-burst
argument cannot explain a ratio difference** when the thing being compared is
two kernels on the same silicon at the same moment.

## (2) In-situ kernel duration: refuted — it matches the microbench

`nsys` trace of a real served request (25 s window mid-denoise, hybrid arm with
the large-M gate, 728 linears wrapped). `_hybrid_mm`: **1304 instances**,
grouped by launch grid, which separates the shapes:

| grid | count | in situ median | microbench | ratio |
|---:|---:|---:|---:|---:|
| 12576 (d→d, and ff2 shares this grid) | 978 | **5644 us** | 5669 us | **0.996×** |
| 6288 (d→d at half N) | 216 | **2920 us** | 3037 us | **0.961×** |
| 50304 (ff1) | 108 | **24059 us** | 23408 us | **1.028×** |

**Within 3% on every shape, and two of the three are faster in situ.** The
kernel does exactly what the benchmark says it does inside a served request.

This also settles (1) more strongly than a clock log would: if the served run
were throttled relative to the microbench, these durations would be longer. They
are not. (The `dmon` capture I took coincided with the failed request described
below and has no usable data; these durations replace it.)

## (3) L2 / H2D interference: refuted twice

**Emulated**, with a background H2D loop from pinned memory sustained at
**57.6 GB/s** — PCIe 5 ×16 saturation, harder than the offload path pushes:

| shape | clean | with H2D at 57.6 GB/s |
|---|---:|---:|
| ff1 | 1.639× | **1.635×** |
| ff2 | 1.528× | **1.532×** |

(d→d read 1.921× under H2D, which I do not believe and do not quote: the arms
were timed sequentially there and the copy loop was still ramping.)

**And observed** in the same trace: 124 pinned→device copies, **106.23 GiB** in
the 25 s window, 54.4 GB/s while copying but only **2.00 s of copy time** —
8% of the window. The weight stream is well overlapped and nowhere near a wall,
which is the same conclusion the end-to-end busy/idle split reached from the
other direction (idle 5.2 s in a 160 s request).

## So what is the answer

**There is no missing 60%.** The kernel delivers 1.39–1.40× in the module and
its in-situ durations match the microbench within 3%. What looked like a large
shortfall was a baseline I measured badly and have withdrawn, and what remains
(3.8 s of a 150 s request) sits inside the resolution of the 10 Hz sampling that
measured it.

**The honest figure for the write-up is the one from the module: the hybrid is
worth about −14 s of GEMM time a request, arriving as about −10 s of request
time.** Anything larger was mine and is wrong.

**What would still be worth measuring**, if someone wants the last few percent:
per-block `_hybrid_mm` totals from this trace come to ~99 ms against the
synthetic block's 89.6 ms, a ~10% difference that my shape bucketing cannot
resolve because d→d and ff2 share a launch grid. Separating them needs the
kernel to carry distinguishable launch parameters, which is a one-line change to
the benchmark harness rather than to the kernel.

## A regression this hunt found, and it was mine

The first `nsys` run did not produce data: the served request failed on its
first step with

```
torch._dynamo.exc.InternalTorchDynamoError:
    AttributeError: 'SymNodeVariable' object has no attribute 'value'
```

**PR #65's FP16 operand cache — mine, merged earlier today — had broken the
served hybrid arm.** It runs inside the regionally-compiled DiT, where
`dynamic=True` makes the input's `numel()` and `shape` SymInts, and I keyed a
Python dict on them. Every test passed because every test of mine was eager.

Reverted from the serving path in **PR #67**, which also adds
`HybridUnderDynamicCompileTest` — it compiles `hybrid_matmul` and the wrapped
`HybridFp16LinearMethod.apply` with `dynamic=True` at two different M and
asserts bit-identity with eager. Verified to have teeth: reintroducing the bug
fails both cases.

**The lesson is not about Dynamo.** It is that a change to the serving path is
untested until something compiles it, and that I shipped one without ever
running a served request.
