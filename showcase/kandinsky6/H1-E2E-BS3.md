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
