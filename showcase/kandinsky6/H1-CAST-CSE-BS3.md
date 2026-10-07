# Inductor already shares the q/k/v cast: there is nothing to CSE

Answer to the last R1 lead — "q, k and v each cast the same LN output to FP16
separately, and Inductor does not CSE the cast, so each attention input is cast
3 times per block". **It does CSE it.** No code change: the data says the premise
does not hold.

Measured on build-server-3, 2026-10-07 20:52–20:56 KST. Delivered through git
because bs3 cannot ssh bs2.

## 1. A shared cast produces identical codegen

Three `hybrid_matmul` calls on one BF16 tensor, against the same three with the
cast hoisted by hand, both compiled `dynamic=True, fullgraph=True`:

| | `to_copy` kernels | `tl.store(out_ptr…)` |
|---|---:|---:|
| three separate casts | 2 | 4 |
| one shared cast | **2** | **4** |

Identical, and both compile under `fullgraph=True`. Hoisting changes nothing
because Inductor had already hoisted it.

## 2. In the real compiled block, 24 of 29 casts are already fused away

The port's own `Kandinsky6FusedTransformerDecoderBlock`, hybrid installed on the
12 large-M linears, compiled as the model runner does, emits **29 kernel
definitions containing a cast**. **24 are fused into their producer** — the layer
norm, the RMS norm, the GELU or an attention epilogue, e.g.
`triton_red_fused__to_copy_add_mul_native_layer_norm_split_t_view_1` and
`triton_poi_fused__to_copy_gelu_t_view_12`. Only **5** are standalone.

So the dominant pattern is already the one the lead asked for: the norm that
produces an attention input writes FP16 directly and no separate pass exists.

## 3. The repeated launches are different tensors, not one cast twice

This settles it. Reading the generated `call()` body for every standalone cast
that launches more than once:

| kernel | launches | input buffers |
|---|---:|---|
| `triton_poi_fused__to_copy_t_view_0` | 2 | `arg6_1`, `buf15` |
| `triton_poi_fused__to_copy_permute_t_view_5` | 2 | `buf29`, `buf55` |
| `triton_poi_fused__to_copy_t_view_1` | 2 | `arg7_1`, `arg15_1` |

**Every repeat reads a different buffer.** No tensor is cast twice, so a
hand-hoisted cast has nothing to collapse. q, k and v already share one cast;
the second launch is the other attention's input.

## 4. What the casts actually cost, from the served trace

nsys trace of a real served request (25.81 s of GPU kernel time, ~109 blocks in
the window):

| kernel | GrdX | launches a block | ms a block | what it is |
|---|---:|---:|---:|---|
| `to_copy_t_view_0` | 401760 | 2.00 | **2.898** | activation-sized, 205.7 M elements |
| `to_copy_permute_t_view_4` | 401760 | 2.00 | **1.622** | activation-sized, permuted layout |
| `to_copy_t_view_1` | 32768 | 8.00 | 0.300 | one 4096x4096 **weight** per d→d linear |
| `to_copy_t_view_18` | 8192 | 2.00 | 0.026 | small |

**Standalone activation casts: 4.52 ms a block ≈ 2.44 s a request.** Every
`to_copy` kernel together is 1895 ms of the 25.81 s window — **7.34% of GPU
time**, an upper bound of 9.42 s a request — but 24 of 29 are fused into real
work and cannot be removed without removing the work.

The weight casts are **8 launches a block and only 0.30 ms** — one per d→d
weight, 16.8 M elements each. Cheap, and consistent with #64's FP16 weight
staging measuring no change.

## 5. So, and what is actually left

**No change shipped.** The mechanism is already implemented by the compiler, and
I am not putting a speculative model change on the serving path for a hypothesis
the generated code refutes — particularly not hours after PR #65, where I shipped
exactly that kind of change on a plausible-sounding theory and broke the arm.

**What is genuinely still on the table:** the two activation-sized standalone
casts are two *different* tensors, so removing them means making each one's
**producer** emit FP16 directly — which is what Inductor already managed for the
other 24. Those two are where its fusion did not reach. That is an Inductor
fusion question, not a CSE and not a model change, and it is worth about
**2.4 s a request** — the largest single item I know of still in the hybrid arm.

## Reproduce

- (1) and (3): `TORCH_LOGS=output_code`, compile three `hybrid_matmul` calls on
  one tensor versus the same with the cast hoisted, then grep the generated
  `call()` body for each cast kernel's `.run(` call sites and compare the first
  argument.
- (2): `block_profile.build_target("fused", ...)` plus
  `hybrid_linear.install_hybrid`, `torch.compile(dynamic=True)`, and count
  kernel definitions matching `to_copy`.
- (4): `nsys stats --report cuda_gpu_trace` on a served request's trace, grouped
  by kernel name and `GrdX`.

---

# Follow-up (21:07): neither standalone cast can be removed by an exact change

Timeboxed attempt on the ~2.4 s named above. **Both producers identified; neither
can emit FP16, and the permute copy is structurally required.** No change
shipped.

## `to_copy_t_view_0` — 2.898 ms a block (~1.57 s a request)

Two call sites, two different inputs, and neither has a producer Inductor could
fuse a cast into:

| input | what it is | can the producer emit FP16? |
|---|---|---|
| `arg6_1` | a **graph input** — the block's incoming hidden states, BF16 from outside the compiled region | **No.** There is no producer in the graph. Casting in the caller moves the same work outside the region. |
| `buf15 = buf14[0]`, `(1, 32, s23, 128)`, carrying an `aten::_scaled_dot_product_*` alignment assert | the **extern attention kernel's output** | **No.** cuDNN / SageAttention fix their output dtype, and Inductor cannot fuse into an extern call. |

## `to_copy_permute_t_view_4` — 1.622 ms a block (~0.88 s a request)

This one casts **and** permutes, because the attention output is `(1, H, S, D)`
and `out_layer` consumes `(S, H*D)`. The suggestion was to hand the kernel the
permuted view instead and let its strided loads absorb the layout. **It cannot,
for a structural reason:**

```
permuted view contiguous: False
reshape to (M, K) without a copy: not possible
```

After `permute(0, 2, 1, 3)` the H and D axes are **not adjacent in memory**, so
there is no pair of strides `(sam, sak)` that describes `(M, H*D)` — and
`_hybrid_mm` takes exactly two strides for A. The copy is not an oversight; it is
what makes the operand addressable by a 2-D-strided kernel.

Measured at W1's self-attention shape (M=50,220, H=32, D=128, N=4096):

| | time |
|---|---:|
| permute + copy + flatten alone | 575 us |
| the GEMM on the contiguous copy | 5523 us |
| both together | 5707 us |

So the copy is ~10% on top of that GEMM, and it buys addressability.

## What would actually remove it, and why I did not do it

A **3-D-A variant of the kernel**: take `(M, H, D)` with three strides and
reconstruct `k = h*D + d` inside the loop. The loads would plausibly coalesce
well — D=128 BF16 is 256 contiguous bytes — so this is not obviously a loss. But
it is a **new kernel**, not an exact change, and it is bounded by the 0.88 s a
request that this one cast costs. Against a remaining budget of half an hour and
having broken the arm once today by shipping a serving-path change on a
plausible theory (#65), the trade is not worth it.

**Recommendation: close this lever.** The ~2.4 s is real but 1.57 s of it has no
fusable producer at all, and the other 0.88 s needs a new kernel variant. If
someone does want it, the 3-D-A kernel is the only route and it should be
microbenched against `575 us + 5523 us` before any integration.
