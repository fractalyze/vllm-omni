# X1 on build-server-3: the final stack's quality rise is #38, not the cache

Delivered through git because bs3 cannot reach bs2 — the tailnet policy refuses
`ssh jooman@build-server-2`. Also in bs3's `/data/jooman/k6/HANDOFF.md`.

Measured 2026-10-07 12:53–13:37 KST, Track C, per `/data/jooman/k6/ATTRIBUTION.md`.

## The number

**X1 = head WITHOUT the cache**, set A, scored against bs3's existing same-host
compiled reference (`results/gate-pro/bf16-stream-control-full`, generated ~05:30
on **pre-#38** code):

| | set mean | set max | G2 | over the 0.25 max |
|---|---:|---:|---|---:|
| **X1 (head, no cache)** | **0.1384** | **0.4680** | passes — 0.95× floor mean, 1.13× its max | 2 (a3, a8) |

The brief's predictions were **H38 ≈ 0.139** and **Hc ≈ 0.112**.
**0.1384 — H38 is confirmed and Hc is excluded.**

## The split

| run | code | cache | set mean | set max |
|---|---|---|---:|---:|
| X0 | pre-#38 | off | 0.1117 | 0.3446 |
| **X1** | **head** | **off** | **0.1384** | **0.4680** |
| X3 | head | step 8 | 0.1393 | 0.4675 |

| transition | what it isolates | Δ mean | Δ max |
|---|---|---:|---:|
| X0 → X1 | #38 + #40, against a pre-#38 reference | **+0.0267 (+23.9%)** | **+0.1234 (+35.8%)** |
| X1 → X3 | **the cache, isolated** | **+0.0009 (+0.7%)** | **−0.0005 (−0.1%)** |

**96.7% of the mean rise is #38/#40; 3.3% is the cache.** The cache's +0.7%
reproduces bs1's independent screen almost exactly (0.1142 → 0.1152, +0.9%).

Per prompt, X1 against X3, shows the same thing — every prompt within ~1%:
a1 0.0481/0.0834 vs 0.0464/0.0822 · a2 0.1721/0.2260 vs 0.1736/0.2279 ·
a3 0.2788/0.4680 vs 0.2776/0.4675 · a4 0.0794/0.1030 vs 0.0814/0.1046 ·
a5 0.0570/0.0676 vs 0.0638/0.0758 · a6 0.0884/0.1652 vs 0.0898/0.1672 ·
a7 0.1130/0.1496 vs 0.1138/0.1500 · a8 0.2710/0.4172 vs 0.2696/0.4124 ·
a9 0.1376/0.2386 vs 0.1374/0.2381.

## What this means, stated carefully

**This is a reference mismatch, not a quality regression.** #38 changes the
attention projections in *every* configuration, the reference included: the bias
is added after the GEMM instead of inside its epilogue, which is a
reassociation. Scoring a #38 arm against a pre-#38 reference measures that
reassociation plus the arm. It does not say the output got worse — only that it
moved relative to a control built on different code.

**So the 0.1393 in the final-stack row should not be read as the stack's
quality.** Track M's run — regenerating the reference on head and re-scoring X3
against it — is the one that gives the stack's real G1, and H38 predicts
**0.112–0.115** there. If it lands in that range, the final stack's quality is
indistinguishable from the shipped arm's and it is simply 12.9% faster.

**And the late-step cache is close to free in quality**: +0.7% of the set mean,
nothing on the max, confirmed on two hosts independently.

## Caveat on this run's timings — do not quote them

Request times were 183.6–310.3 s, far above the ~173 s this configuration should
take, because **other users' jobs loaded the host throughout**: a root `cc1plus`
compile and `falco` near 100% CPU, three users' `dart:firmware-n` processes, load
average peaking at **43**, 2 GB of free RAM and `kswapd` active. No foreign *GPU*
process at any point, so the run is not disqualified.

**The quality numbers are unaffected and that is checkable rather than assumed:**
there was **no autotuning and no compile activity** in the run, and **zero files
were written to the pinned Inductor cache after 13:00**, so X1 executed the same
cached kernels as X3 and as the reference. Host load changes timing, not
arithmetic. The timings above are discarded; the LPIPS numbers stand.
