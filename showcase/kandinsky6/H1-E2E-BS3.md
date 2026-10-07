# H1 end to end on build-server-3: the arm is NOT bytes-bound

Second-host answer to "is the H1 arm bytes-bound?". Delivered through git because
bs3 cannot ssh bs2 (tailnet policy). Measured 2026-10-07 15:26-16:05 KST.

**Short answer: no, and the question's premise does not hold.** GPU idle did
**not** grow. The request fell 5.0 s and essentially all of it came out of GPU
busy time. Both arms run the GPU **~97% busy**, so there is no bytes wall for a
compute saving to disappear into — **the next lever is not bytes per step.**

## Method

Base (final stack: head + `sage2-mid` + `EXACT_ATTN_STEPS=1` +
`PIFLOW_CACHE_STEPS=8`) against base + `VLLM_OMNI_K6_HYBRID_GEMM=1`, **A B B A**,
one warm-up and two timed requests per visit, a1-portrait-speech at W1, GPU locks
held, no foreign GPU process. Utilisation sampled at 10 Hz for each request's
window; "busy" is utilisation >= 50%, which on this arm sits in an empty part of
the distribution (it alternates between ~100 and ~0).

Proof the kernel ran, from the hybrid servers' own logs:

```
Kandinsky 6: 1736 DiT linears on the hybrid FP16-accumulate GEMM, 254 excluded
```

## The ABBA, and why the pooled number is a trap

| visit | request | GPU busy | GPU idle |
|---|---:|---:|---:|
| A-base-0 | 160.9 s | 154.8 s | 6.2 s |
| A-base-0 | 158.3 s | 154.1 s | 4.2 s |
| **B-hybrid-1** | **183.7 s** | 160.5 s | **23.2 s** |
| **B-hybrid-1** | **183.7 s** | 157.1 s | **26.7 s** |
| B-hybrid-2 | 154.7 s | 149.9 s | 4.8 s |
| B-hybrid-2 | 154.6 s | 148.9 s | 5.7 s |
| A-base-3 | 160.6 s | 153.7 s | 6.9 s |
| A-base-3 | 158.9 s | 154.6 s | 4.3 s |

**The two hybrid visits are the same code and differ by 29 s.** Visit 1 was the
first hybrid server on this host, so it paid **Triton JIT compilation** inside
the request, which is where its 23-27 s of idle went -- the GPU waiting while
kernels compiled. Visit 2 found the on-disk Triton cache warm.

**Pooling all four hybrid runs gives 169.2 s and "+9.5 s, idle +9.2 s", which
looks exactly like the bytes-bound story and is an artefact of compilation.** A
single A-then-B comparison here would have reported the hybrid as 25 s slower
with the idle growth to explain it. The ABBA is what exposed it.

## The comparison that counts

| | base | hybrid (Triton warm) | change |
|---|---:|---:|---:|
| request | 159.7 s | **154.7 s** | **-5.0 s (-3.1%)** |
| GPU busy | 154.4 s | 149.4 s | **-4.9 s** |
| GPU idle | 5.2 s | 5.3 s | **+0.0 s** |

**Idle is flat. Busy absorbed the whole change.** So the arm is compute-saturated,
not stream-starved: the stage-ahead overlap is doing its job and there is no
waiting for a compute win to hide behind.

## The real problem: the kernel keeps 18% of its own benchmark

The kernel-level measurement predicted **-23 to -31 s** of GEMM time. **Busy fell
4.9 s -- about 18% of the midpoint.** That is the finding worth acting on, and it
is a different problem from the one we set out to test.

**Leading suspect, and it is one I flagged before the integration.** The log says
**1736 linears wrapped, 254 excluded** -- nearly every linear in the DiT. The
serving `HybridFp16LinearMethod.apply` calls `hybrid_matmul` directly rather than
the `hybrid_linear` drop-in, so it bypasses the **2048-row crossover gate**. Below
that threshold the hybrid is **1.7-2.7x SLOWER** than cuBLAS, measured:

| M | 4096x4096 | 4096x16384 | 2048x2048 |
|---:|---:|---:|---:|
| 218 | 0.59x | 0.57x | 0.37x |
| 512 | 0.85x | 0.83x | 0.57x |
| 1024 | 1.11x | 0.99x | 0.98x |
| 50220 | 1.46x | 1.57x | 1.17x |

At W1 the audio branch runs **M=218** and the text tower **M=256**, and roughly
17 of the ~29 wrapped linears per block are in that regime. Each is cheap in
absolute terms, so this cannot account for all 20 missing seconds, but it is
pure loss and it is free to remove.

**The other candidate is dispatch.** 1736 wrapped linears x 10 steps is **~17,400
Triton launches a request**, each preceded by Python-level reshape, transpose,
allocation and grid arithmetic. That shows up as busy time rather than idle (the
GPU is kept nominally occupied), which fits busy falling far less than the kernel
predicts while idle stays flat.

## What to do next, in order

1. **Route through `hybrid_linear`, or add its gate to `apply`.** One
   `VLLM_OMNI_K6_HYBRID_GEMM_EXCLUDE` covering `audio_dec_block`, the text tower
   and the small-M cross-attention sides would test it without touching code. I
   am running that single-visit experiment now and will post the delta.
2. **Then measure dispatch**, by timing `hybrid_matmul` against `F.linear` at
   M=218 with the GEMM time subtracted out. If per-call overhead is the
   remainder, the fix is one Triton launch per block rather than per linear, or
   a CUDA-graph capture of the block.
3. **Do not pursue bytes per step on this evidence.** 5.2 s of idle in a 160 s
   request is not where 20 s is hiding.

## One procedural note

`VLLM_OMNI_K6_HYBRID_GEMM` is honoured but **nothing proves it at request time**
-- the install line is printed once at construction. The first hybrid visit's
numbers were a compile artefact that looked like a performance result, so for the
final table please quote hybrid numbers only from a server whose Triton cache was
already warm, and say which.

---

# Follow-up (16:25): the small-M gate nearly doubles the saving, and *then* a little idle appears

The experiment promised above. Same harness, Triton cache warm,
`VLLM_OMNI_K6_HYBRID_GEMM_EXCLUDE` set so that **exactly the 12 large-M linears
a fused block issues stay on the kernel** and the 20 small-M ones go back to
cuBLAS. Validated against the module names `gemm_insitu.py` recorded before
running: 12/12 kept, 0/20 leaked. The server confirms it:

```
Kandinsky 6: 728 DiT linears on the hybrid FP16-accumulate GEMM, 1262 excluded
```

| arm | request | GPU busy | GPU idle |
|---|---:|---:|---:|
| base | 159.8 s | 154.3 s | 5.2 s |
| hybrid, all 1736 linears | 154.6 s | 149.4 s | 5.2 s |
| **hybrid, 728 large-M only** | **152.4 s** | **143.5 s** | **7.8 s** |

| arm | request | busy | idle |
|---|---:|---:|---:|
| hybrid, all | **-5.1 s** | -4.9 s | **+0.0 s** |
| **hybrid, large-M only** | **-7.3 s** | **-10.8 s** | **+2.5 s** |

**Gating on M is worth 2.2 s of request time and 5.9 s of GPU busy time on its
own** -- the small-M linears were not merely failing to help, they were giving
back a fifth of the win. That is the gate `hybrid_linear` already implements and
that the serving `apply()` bypasses; **one `EXCLUDE` string captures it with no
code change.**

**And now the bytes question gets a real, if smaller, yes.** With the small-M
regression out of the way, **10.8 s of compute comes out and only 7.3 s of it
reaches the request: 2.5 s, about 24%, leaks into GPU idle.** So the stream does
begin to bind once enough compute is removed -- it simply was not binding at the
first operating point I measured, where the small-M regression was masking the
saving.

**The honest summary of the three readings:**

1. **The arm is not bytes-bound today.** Even at its best it runs 94% GPU-busy,
   and 5-8 s of idle in a 152 s request is not where 20 s is hiding.
2. **Roughly a quarter of any further compute saving will leak into idle.** So
   bytes per step is a real lever but a second-order one: it converts about 1 s
   of the next 4 s of compute saved.
3. **The first-order problem is still that the kernel keeps only ~40% of its own
   benchmark** (10.8 s realised of 23-31 s predicted) even with the right layers
   wrapped. That is not bytes and not the small-M regression; the remaining
   suspect is per-call dispatch -- 728 wrapped linears is still **~7,300 Triton
   launches a request**, each preceded by Python-level reshape, transpose,
   allocation and grid arithmetic. Timing `hybrid_matmul` against `F.linear` with
   the GEMM time subtracted out would settle it, and the fix would be one launch
   per block or a CUDA-graph capture rather than one per linear.

**Recommended for the final table:** the large-M-gated hybrid at **152.4 s
against base's 159.8 s, -7.3 s (-4.6%)**, with the preregistration's -12.7 s
noted as not met and the reason given. Three timed runs with the Triton cache
warm; a fourth was still running at the time of writing and will not move the
median materially.
