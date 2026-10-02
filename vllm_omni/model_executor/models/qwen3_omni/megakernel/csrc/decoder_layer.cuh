// Copyright 2026 Fractalyze Inc. All rights reserved.
//
// One decoder layer, as the megakernels run it: five phases separated by grid
// barriers, on every CTA of a persistent launch. The layer is templated on the
// decoder's DecoderDims (layer.h), so S2 Pro and other models share it.
//
//   1. RMSNorm → QKV GEMV. Every CTA normalises the whole residual itself,
//      which is cheaper than a barrier and a broadcast, then computes its
//      share of the q, k and v rows.
//   2. Attention. Query head h and sequence chunk s form work item
//      h × S + s, one per CTA, for S chunks per head. An item applies QK-norm
//      (when the layer has one) and RoPE to its head, attends over its chunk,
//      and writes a partial (max, sum, unnormalised output). The first chunk
//      of the first query head of each KV group writes this step's key and
//      value to the cache.
//   3. Merge → O GEMV. Every CTA merges all partials into the attention
//      output, then adds its share of O rows into the residual.
//   4. RMSNorm → gate-up GEMV. Gate and up rows are interleaved in pairs and
//      a CTA owns whole pairs, so SiLU(gate) · up happens in the epilogue.
//   5. Down GEMV, added into the residual.
//
// Weights never depend on activations, so right after each GEMV a CTA starts
// pulling the first `prefetch_bytes` of its slice of the next GEMV into L2:
// the tail of the phase, the barrier and the next phase's ramp-up run with
// HBM already streaming. It is not issued any earlier: one GEMV streams more
// than L2 holds, so lines fetched before it would be evicted by it unread.
// The caller names the GEMV that follows the layer.
//
// The residual stream is fp32 throughout; GEMV inputs are rounded to bf16 at
// the points Fish's bf16 model rounds them. Data another CTA wrote in this
// launch is read with __ldcg, from L2, never from a possibly stale L1 line.

#ifndef S2MK_CSRC_DECODER_LAYER_CUH_
#define S2MK_CSRC_DECODER_LAYER_CUH_

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

#include "gemv_core.cuh"
#include "kv_cache.h"
#include "layer.h"

namespace s2mk {

// Positions a warp loads before scoring any of them.
constexpr int kPosUnroll = 8;
// Candidates each warp keeps when the sampler selects its top-k.
constexpr int kSamplerTopK = 30;

static_assert(kHeadDim == 32 * 4, "attention gives each lane 4 dims");

template <typename D>
struct CheckDims {
  static_assert(D::kDim % kThreads == 0,
                "RmsNorm gives each thread whole dims");
  static_assert(D::kQDim % kThreads == 0);
  static_assert(D::kQHeads % D::kKvHeads == 0);
  static_assert(D::kFfn % kVecElems == 0);
  static_assert(D::kFfn >= D::kDim && D::kFfn >= D::kQDim,
                "Shared::xs holds every GEMV input");
  static constexpr bool kOk = true;
};

template <typename D>
struct Shared {
  static_assert(CheckDims<D>::kOk);
  // The current GEMV's input, as bf16.
  uint4 xs[D::kFfn / kVecElems];
  float ys[kMaxRowsPerCta];
  // GemvRowsFixedOrder's per-warp partials, for a decoder that sums in order.
  float partials[D::kFixedOrderGemv ? kWarps * kMaxRowsPerCta : 1];
  float red[kWarps];
  // A layer input no other CTA wrote: every CTA computes it whole.
  float x0[D::kDim];
  // Attention: this step's q, k and v for the item's head, then each warp's
  // running max, sum and output.
  float q[kHeadDim];
  float k[kHeadDim];
  float v[kHeadDim];
  float warp_m[kWarps];
  float warp_l[kWarps];
  union {
    float warp_o[kWarps][kHeadDim];
    // The sampler: each warp's top-k keys.
    uint64_t top_keys[kWarps][kSamplerTopK];
  };
  // Attention merge: each item's max, then its weight in its head's output;
  // and each item's sum.
  float merge_weight[kMaxCtas];
  float merge_sum[kMaxCtas];
};

// The grid-wide buffers a layer passes between its phases, sized by the
// decoder's DecoderDims D.
struct LayerBuffers {
  float* residual;  // [D::kDim]: the layer's input in, its output out
  float* qkv;  // [D::kQkvRows]
  float* partial_ml;  // [attention items, 2]: running max and sum
  float* partial_o;  // [attention items, kHeadDim]: unnormalised output
  __nv_bfloat16* act;  // [D::kFfn]: SiLU(gate) · up
  float eps;
  int prefetch_bytes;
};

// Where a layer's attention reads and writes this step.
struct AttentionStep {
  KvLayout kv;
  int layer;  // index into kv
  int pos;
  const __nv_bfloat16* rope;  // [positions, kHeadDim / 2, (cos, sin)]
  int splits;  // sequence chunks per query head
};

// A GEMV's weight and split, for an L2 prefetch of this CTA's slice: n units
// of `rows_per_unit` rows of k columns.
struct GemvSlice {
  const __nv_bfloat16* w;
  int k;
  int n;
  int rows_per_unit;
};

__device__ __forceinline__ void Prefetch(const GemvSlice& slice, int bytes) {
  PrefetchSliceL2(slice.w, slice.k, slice.n, slice.rows_per_unit, bytes);
}

template <typename D>
__device__ __forceinline__ GemvSlice QkvSlice(const LayerWeights& w) {
  return {w.wqkv, D::kDim, D::kQkvRows, 1};
}

template <typename D>
__device__ __forceinline__ GemvSlice OSlice(const LayerWeights& w) {
  return {w.wo, D::kQDim, D::kDim, 1};
}

__device__ __forceinline__ float RoundBf16(float x) {
  return __bfloat162float(__float2bfloat16(x));
}

// The decoder's GEMV, sh.ys = W[begin, begin + rows) · sh.xs.
template <typename D>
__device__ __forceinline__ void LayerGemv(const __nv_bfloat16* w, int k,
                                          int begin, int rows, Shared<D>& sh) {
  if constexpr (D::kFixedOrderGemv) {
    GemvRowsFixedOrder(w, k, begin, rows, sh.xs, sh.ys, sh.partials, rows, 0);
  } else {
    GemvRows(w, k, begin, rows, sh.xs, sh.ys);
  }
}

// The gate-up GEMV of this CTA's `pairs` gate-up pairs from pair `begin`:
// sh.ys holds pair i's gate at 2i and its up at 2i + 1, or, for a stacked
// w13 (D::kStackedGateUp), every gate and then every up.
template <typename D>
__device__ __forceinline__ void GateUpGemv(const __nv_bfloat16* w13,
                                           int begin, int pairs,
                                           Shared<D>& sh) {
  if constexpr (D::kStackedGateUp) {
    static_assert(D::kFixedOrderGemv, "only the fixed-order GEMV skips rows");
    GemvRowsFixedOrder(w13, D::kDim, begin, 2 * pairs, sh.xs, sh.ys,
                       sh.partials, pairs, D::kFfn - pairs);
  } else {
    LayerGemv(w13, D::kDim, 2 * begin, 2 * pairs, sh);
  }
}

// Pair i's gate and up after GateUpGemv.
template <typename D>
__device__ __forceinline__ float2 GateUp(const Shared<D>& sh, int pairs,
                                         int i) {
  if constexpr (D::kStackedGateUp) {
    return make_float2(sh.ys[i], sh.ys[pairs + i]);
  } else {
    return make_float2(sh.ys[2 * i], sh.ys[2 * i + 1]);
  }
}

// Starts fetching this CTA's share of the gate-up GEMV into L2.
template <typename D>
__device__ __forceinline__ void PrefetchGateUp(const __nv_bfloat16* w13,
                                               int bytes) {
  if constexpr (D::kStackedGateUp) {
    const int half = bytes / 32 * 16;
    PrefetchSliceL2(w13, D::kDim, D::kFfn, 1, half);
    PrefetchSliceL2(w13 + int64_t{D::kFfn} * D::kDim, D::kDim, D::kFfn, 1,
                    half);
  } else {
    PrefetchSliceL2(w13, D::kDim, D::kFfn, 2, bytes);
  }
}

// The sum of `v` over the CTA, in a fixed order so it is deterministic.
__device__ inline float BlockSum(float v, float* red) {
  v = WarpSum(v);
  if (threadIdx.x % 32 == 0) red[threadIdx.x / 32] = v;
  __syncthreads();
  float sum = 0.f;
#pragma unroll
  for (int w = 0; w < kWarps; ++w) sum += red[w];
  __syncthreads();
  return sum;
}

// out = RMSNorm(x) · weight in bf16, and a copy into `copy` when non-null.
// x is `local` when non-null (this CTA's shared memory), else `global`, which
// other CTAs wrote.
template <typename D>
__device__ inline void RmsNorm(const float* global, const float* local,
                               const __nv_bfloat16* weight, float eps,
                               __nv_bfloat16* out, __nv_bfloat16* copy,
                               float* red) {
  constexpr int kDimPerThread = D::kDim / kThreads;
  float v[kDimPerThread];
  float squares = 0.f;
#pragma unroll
  for (int j = 0; j < kDimPerThread; ++j) {
    const int i = threadIdx.x + j * kThreads;
    v[j] = local != nullptr ? local[i] : __ldcg(global + i);
    squares += v[j] * v[j];
  }
  const float inv = rsqrtf(BlockSum(squares, red) / D::kDim + eps);
#pragma unroll
  for (int j = 0; j < kDimPerThread; ++j) {
    const int i = threadIdx.x + j * kThreads;
    const __nv_bfloat16 y =
        __float2bfloat16(v[j] * inv * __bfloat162float(weight[i]));
    out[i] = y;
    if (copy != nullptr) copy[i] = y;
  }
}

// This step's q (fp32) and k, v (bf16-rounded, as the cache holds them) for
// query head h, after QK-norm (when the layer has one) and RoPE.
template <typename D>
__device__ inline void LoadHead(const LayerBuffers& b, const AttentionStep& a,
                                const LayerWeights& w, int h, Shared<D>& sh) {
  static_assert(!D::kRotateHalf, "split attention rotates pairs (2i, 2i + 1)");
  const int t = threadIdx.x;
  const int g = h / D::kGqa;
  if (t < kHeadDim) {
    sh.q[t] = __ldcg(b.qkv + h * kHeadDim + t);
  } else if (t < 2 * kHeadDim) {
    sh.k[t - kHeadDim] =
        __ldcg(b.qkv + D::kQDim + g * kHeadDim + (t - kHeadDim));
  } else if (t < 3 * kHeadDim) {
    sh.v[t - 2 * kHeadDim] = RoundBf16(__ldcg(
        b.qkv + D::kQDim + (D::kKvHeads + g) * kHeadDim + (t - 2 * kHeadDim)));
  }
  const bool qk_norm = w.q_norm != nullptr;
  if (qk_norm) {
    // Warps 0-3 hold q, warps 4-7 hold k.
    float square = 0.f;
    if (t < kHeadDim) square = sh.q[t] * sh.q[t];
    if (kHeadDim <= t && t < 2 * kHeadDim) {
      square = sh.k[t - kHeadDim] * sh.k[t - kHeadDim];
    }
    square = WarpSum(square);
    if (t % 32 == 0) sh.red[t / 32] = square;
  }
  __syncthreads();

  // Threads 0-63 rotate q's pairs (2i, 2i + 1), threads 64-127 k's.
  constexpr int kPairs = kHeadDim / 2;
  if (t < 2 * kPairs) {
    const bool is_q = t < kPairs;
    const int i = t % kPairs;
    float* x = is_q ? sh.q : sh.k;
    float x0 = x[2 * i];
    float x1 = x[2 * i + 1];
    if (qk_norm) {
      const float* red = sh.red + (is_q ? 0 : 4);
      const float inv =
          rsqrtf((red[0] + red[1] + red[2] + red[3]) / kHeadDim + b.eps);
      const __nv_bfloat16* norm = is_q ? w.q_norm : w.k_norm;
      x0 = x0 * inv * __bfloat162float(norm[2 * i]);
      x1 = x1 * inv * __bfloat162float(norm[2 * i + 1]);
    }
    const __nv_bfloat16* cis = a.rope + (int64_t{a.pos} * kPairs + i) * 2;
    const float c = __bfloat162float(cis[0]);
    const float s = __bfloat162float(cis[1]);
    float r0 = x0 * c - x1 * s;
    float r1 = x1 * c + x0 * s;
    if (!is_q) {
      r0 = RoundBf16(r0);
      r1 = RoundBf16(r1);
    }
    x[2 * i] = r0;
    x[2 * i + 1] = r1;
  }
  __syncthreads();
}

template <typename D>
__device__ inline void Attention(const LayerBuffers& b, const AttentionStep& a,
                                 const LayerWeights& w, Shared<D>& sh) {
  const int splits = a.splits;
  const int item = blockIdx.x;
  if (item >= D::kQHeads * splits) return;
  const int h = item / splits;
  const int s = item % splits;
  const int g = h / D::kGqa;
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const KvLayout& kv = a.kv;
  const int layer = a.layer;
  const int pos = a.pos;

  LoadHead(b, a, w, h, sh);
  if (h % D::kGqa == 0 && s == 0 && threadIdx.x < kHeadDim) {
    kv.Key(layer, g, pos)[threadIdx.x] = __float2bfloat16(sh.k[threadIdx.x]);
    kv.Value(layer, g, pos)[threadIdx.x] = __float2bfloat16(sh.v[threadIdx.x]);
  }

  // The step attends to positions [0, pos]; chunk s covers
  // [s × len / S, (s + 1) × len / S). Position pos comes from shared memory,
  // the rest from the cache.
  const int len = pos + 1;
  const int begin = s * len / splits;
  const int end = (s + 1) * len / splits;
  const float scale = rsqrtf(static_cast<float>(kHeadDim));
  float q[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) q[j] = sh.q[lane * 4 + j];

  float m = -INFINITY;
  float l = 0.f;
  float o[4] = {0.f, 0.f, 0.f, 0.f};
  for (int base = begin + warp * kPosUnroll; base < end;
       base += kWarps * kPosUnroll) {
    uint2 kr[kPosUnroll];
    uint2 vr[kPosUnroll];
#pragma unroll
    for (int u = 0; u < kPosUnroll; ++u) {
      const int t = base + u;
      if (t < end && t != pos) {
        kr[u] = __ldcg(reinterpret_cast<const uint2*>(kv.Key(layer, g, t)) +
                       lane);
        vr[u] = __ldcg(
            reinterpret_cast<const uint2*>(kv.Value(layer, g, t)) + lane);
      }
    }
#pragma unroll
    for (int u = 0; u < kPosUnroll; ++u) {
      const int t = base + u;
      if (t >= end) break;
      float k[4];
      float v[4];
      if (t == pos) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          k[j] = sh.k[lane * 4 + j];
          v[j] = sh.v[lane * 4 + j];
        }
      } else {
        const __nv_bfloat162* k2 =
            reinterpret_cast<const __nv_bfloat162*>(&kr[u]);
        const __nv_bfloat162* v2 =
            reinterpret_cast<const __nv_bfloat162*>(&vr[u]);
#pragma unroll
        for (int j = 0; j < 2; ++j) {
          const float2 kf = __bfloat1622float2(k2[j]);
          const float2 vf = __bfloat1622float2(v2[j]);
          k[2 * j] = kf.x;
          k[2 * j + 1] = kf.y;
          v[2 * j] = vf.x;
          v[2 * j + 1] = vf.y;
        }
      }
      float dot = 0.f;
#pragma unroll
      for (int j = 0; j < 4; ++j) dot = fmaf(q[j], k[j], dot);
      const float score = WarpSum(dot) * scale;
      const float m_new = fmaxf(m, score);
      const float correction = expf(m - m_new);
      const float weight = expf(score - m_new);
      l = l * correction + weight;
#pragma unroll
      for (int j = 0; j < 4; ++j) o[j] = fmaf(o[j], correction, weight * v[j]);
      m = m_new;
    }
  }

  if (lane == 0) {
    sh.warp_m[warp] = m;
    sh.warp_l[warp] = l;
  }
#pragma unroll
  for (int j = 0; j < 4; ++j) sh.warp_o[warp][lane * 4 + j] = o[j];
  __syncthreads();
  if (threadIdx.x < kHeadDim) {
    // A warp that saw no position has l = 0 and m = -inf; skip it.
    float m_all = -INFINITY;
    for (int wi = 0; wi < kWarps; ++wi) {
      if (sh.warp_l[wi] > 0.f) m_all = fmaxf(m_all, sh.warp_m[wi]);
    }
    float l_all = 0.f;
    float o_all = 0.f;
    for (int wi = 0; wi < kWarps; ++wi) {
      if (sh.warp_l[wi] == 0.f) continue;
      const float f = expf(sh.warp_m[wi] - m_all);
      l_all += sh.warp_l[wi] * f;
      o_all += sh.warp_o[wi][threadIdx.x] * f;
    }
    b.partial_o[item * kHeadDim + threadIdx.x] = o_all;
    if (threadIdx.x == 0) {
      b.partial_ml[item * 2] = m_all;
      b.partial_ml[item * 2 + 1] = l_all;
    }
  }
}

// Merges every item's partial into the attention output, as bf16 in `out`.
template <typename D>
__device__ inline void MergeAttention(const LayerBuffers& b, int splits,
                                      Shared<D>& sh, __nv_bfloat16* out) {
  // Every item's (max, sum) into shared memory in one round of loads, then
  // thread h weighs head h's chunks: exp(m_s - m) / l for the head's overall
  // max m and sum l. A chunk that saw no position (l_s = 0) weighs 0.
  const int items = D::kQHeads * splits;
  float* m = sh.merge_weight;
  float* l = sh.merge_sum;
  for (int i = threadIdx.x; i < items; i += kThreads) {
    const float2 ml = __ldcg(reinterpret_cast<const float2*>(b.partial_ml) + i);
    m[i] = ml.x;
    l[i] = ml.y;
  }
  __syncthreads();
  if (threadIdx.x < D::kQHeads) {
    const int first = threadIdx.x * splits;
    float m_all = -INFINITY;
    for (int s = first; s < first + splits; ++s) {
      if (l[s] > 0.f) m_all = fmaxf(m_all, m[s]);
    }
    float l_all = 0.f;
    for (int s = first; s < first + splits; ++s) {
      m[s] = l[s] > 0.f ? expf(m[s] - m_all) : 0.f;
      l_all += l[s] * m[s];
    }
    for (int s = first; s < first + splits; ++s) m[s] /= l_all;
  }
  __syncthreads();

  // Each thread merges four adjacent dims of one head, one 16-byte load per
  // chunk, all independent.
  const float4* partial_o = reinterpret_cast<const float4*>(b.partial_o);
  for (int i = threadIdx.x; i < D::kQDim / 4; i += kThreads) {
    const int first = i * 4 / kHeadDim * splits;
    const int d4 = i % (kHeadDim / 4);
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
    for (int s = 0; s < splits; ++s) {
      const float f = sh.merge_weight[first + s];
      const float4 o = __ldcg(partial_o + (first + s) * (kHeadDim / 4) + d4);
      acc.x = fmaf(o.x, f, acc.x);
      acc.y = fmaf(o.y, f, acc.y);
      acc.z = fmaf(o.z, f, acc.z);
      acc.w = fmaf(o.w, f, acc.w);
    }
    __nv_bfloat162* out2 = reinterpret_cast<__nv_bfloat162*>(out + i * 4);
    out2[0] = __floats2bfloat162_rn(acc.x, acc.y);
    out2[1] = __floats2bfloat162_rn(acc.z, acc.w);
  }
}

// Short attention: lane j of a warp holds dims 4j to 4j + 3 of a head.

// RoPE at this step's position on this lane's dims. Pairs (2i, 2i + 1) sit in
// one lane. Rotate-half (D::kRotateHalf) pairs dim d with d + kHeadDim / 2,
// which sits in lane j ^ 16: the warp swaps halves, so every lane of the warp
// must call it.
template <typename D>
__device__ __forceinline__ float4 RotateLane(const AttentionStep& a, int lane,
                                             float4 x) {
  if constexpr (D::kRotateHalf) {
    static_assert(kHeadDim == 32 * 4, "lane j ^ 16 holds dim d ± 64");
    float4 other;
    other.x = __shfl_xor_sync(0xffffffffu, x.x, 16);
    other.y = __shfl_xor_sync(0xffffffffu, x.y, 16);
    other.z = __shfl_xor_sync(0xffffffffu, x.z, 16);
    other.w = __shfl_xor_sync(0xffffffffu, x.w, 16);
    // Dim d turns at frequency d mod kHeadDim / 2: x cos − x' sin in the
    // first half, x cos + x' sin in the second.
    const __nv_bfloat162* cis = reinterpret_cast<const __nv_bfloat162*>(
        a.rope + (int64_t{a.pos} * (kHeadDim / 2) + 4 * (lane % 16)) * 2);
    const float2 cs0 = __bfloat1622float2(cis[0]);
    const float2 cs1 = __bfloat1622float2(cis[1]);
    const float2 cs2 = __bfloat1622float2(cis[2]);
    const float2 cs3 = __bfloat1622float2(cis[3]);
    const float sign = lane < 16 ? -1.f : 1.f;
    return make_float4(fmaf(sign * other.x, cs0.y, x.x * cs0.x),
                       fmaf(sign * other.y, cs1.y, x.y * cs1.x),
                       fmaf(sign * other.z, cs2.y, x.z * cs2.x),
                       fmaf(sign * other.w, cs3.y, x.w * cs3.x));
  } else {
    const __nv_bfloat162* cis = reinterpret_cast<const __nv_bfloat162*>(
        a.rope + (int64_t{a.pos} * (kHeadDim / 2) + 2 * lane) * 2);
    const float2 cs0 = __bfloat1622float2(cis[0]);
    const float2 cs1 = __bfloat1622float2(cis[1]);
    return make_float4(x.x * cs0.x - x.y * cs0.y, x.y * cs0.x + x.x * cs0.y,
                       x.z * cs1.x - x.w * cs1.y, x.w * cs1.x + x.z * cs1.y);
  }
}

__device__ __forceinline__ float4 RoundBf16(float4 x) {
  return make_float4(RoundBf16(x.x), RoundBf16(x.y), RoundBf16(x.z),
                     RoundBf16(x.w));
}

__device__ __forceinline__ float4 Bf16x4(uint2 x) {
  const __nv_bfloat162* x2 = reinterpret_cast<const __nv_bfloat162*>(&x);
  const float2 lo = __bfloat1622float2(x2[0]);
  const float2 hi = __bfloat1622float2(x2[1]);
  return make_float4(lo.x, lo.y, hi.x, hi.y);
}

__device__ __forceinline__ void StoreBf16x4(__nv_bfloat16* out, float4 x) {
  __nv_bfloat162* out2 = reinterpret_cast<__nv_bfloat162*>(out);
  out2[0] = __floats2bfloat162_rn(x.x, x.y);
  out2[1] = __floats2bfloat162_rn(x.z, x.w);
}

__device__ __forceinline__ float Dot(float4 a, float4 b) {
  float dot = a.x * b.x;
  dot = fmaf(a.y, b.y, dot);
  dot = fmaf(a.z, b.z, dot);
  return fmaf(a.w, b.w, dot);
}

// QK-norm of one head on this lane's dims: RMSNorm over the head's kHeadDim
// dims, times `norm`. A null `norm` means the layer has none.
__device__ __forceinline__ float4 QkNormLane(float4 x,
                                             const __nv_bfloat16* norm,
                                             float eps, int lane) {
  if (norm == nullptr) return x;
  const float inv = rsqrtf(WarpSum(Dot(x, x)) / kHeadDim + eps);
  const float4 w =
      Bf16x4(*reinterpret_cast<const uint2*>(norm + 4 * lane));
  return make_float4(x.x * inv * w.x, x.y * inv * w.y, x.z * inv * w.z,
                     x.w * inv * w.w);
}

// This lane's dims of KV group g's key (after QK-norm and RoPE) and value this
// step, bf16-rounded as the cache holds them.
template <typename D>
__device__ __forceinline__ void LoadKvLane(const float* qkv_rows,
                                           const AttentionStep& a,
                                           const __nv_bfloat16* k_norm,
                                           float eps, int g, int lane,
                                           float4& k, float4& v) {
  const float4* qkv = reinterpret_cast<const float4*>(qkv_rows);
  constexpr int kHeadVecs = kHeadDim / 4;
  k = RoundBf16(RotateLane<D>(
      a, lane,
      QkNormLane(__ldcg(qkv + (D::kQHeads + g) * kHeadVecs + lane), k_norm,
                 eps, lane)));
  v = RoundBf16(
      __ldcg(qkv + (D::kQHeads + D::kKvHeads + g) * kHeadVecs + lane));
}

__device__ __forceinline__ void StoreKvLane(const AttentionStep& a, int g,
                                            int lane, float4 k, float4 v) {
  StoreBf16x4(a.kv.Key(a.layer, g, a.pos) + 4 * lane, k);
  StoreBf16x4(a.kv.Value(a.layer, g, a.pos) + 4 * lane, v);
}

// Writes this step's key and value of every KV group, without attending:
// CTA 0's warp g writes group g.
template <typename D>
__device__ inline void WriteKv(const LayerBuffers& b, const AttentionStep& a,
                               const LayerWeights& w) {
  static_assert(D::kKvHeads <= kWarps);
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  if (blockIdx.x != 0 || warp >= D::kKvHeads) return;
  float4 k;
  float4 v;
  LoadKvLane<D>(b.qkv, a, w.k_norm, b.eps, warp, lane, k, v);
  StoreKvLane(a, warp, lane, k, v);
}

// S2 Pro's Fast AR context, and the default longest context ShortAttention
// takes: it holds every position's value and scores in registers, so the
// bound is a register budget.
constexpr int kShortMaxPositions = 10;

// Attention over a context of at most kMaxPositions, which every CTA computes
// for every query head itself, straight into the O GEMV's input `out` as
// bf16: no partials, and no barrier between attending and merging. Warp w
// takes heads kHeadsPerWarp × w onward, which share a KV group. The context
// is short enough to score every position before weighing any, so no
// position waits on the one before it. This step's q, k and v come from
// `qkv_rows` ([D::kQkvRows]), and CTA 0 also writes the key and value to the
// cache.
template <typename D, int kMaxPositions = kShortMaxPositions>
__device__ inline void ShortAttention(const float* qkv_rows,
                                      const AttentionStep& a,
                                      const LayerWeights& w, float eps,
                                      __nv_bfloat16* out) {
  constexpr int kHeadsPerWarp = D::kQHeads / kWarps;
  static_assert(D::kQHeads == kHeadsPerWarp * kWarps &&
                    (kHeadsPerWarp == 1 || kHeadsPerWarp == 2),
                "every warp takes one or two whole heads");
  static_assert(D::kGqa % kHeadsPerWarp == 0,
                "a warp's heads share one KV group");
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const KvLayout& kv = a.kv;
  const int pos = a.pos;
  const int h = kHeadsPerWarp * warp;
  const int g = h / D::kGqa;

  const float4* qkv = reinterpret_cast<const float4*>(qkv_rows);
  constexpr int kHeadVecs = kHeadDim / 4;
  float4 q[kHeadsPerWarp];
#pragma unroll
  for (int i = 0; i < kHeadsPerWarp; ++i) {
    q[i] = RotateLane<D>(
        a, lane,
        QkNormLane(__ldcg(qkv + (h + i) * kHeadVecs + lane), w.q_norm, eps,
                   lane));
  }
  float4 k_pos;
  float4 v_pos;
  LoadKvLane<D>(qkv_rows, a, w.k_norm, eps, g, lane, k_pos, v_pos);
  if (blockIdx.x == 0 && h % D::kGqa == 0) {
    StoreKvLane(a, g, lane, k_pos, v_pos);
  }

  // Position t's key, value and each head's score; positions past `pos`
  // score -inf and weigh 0.
  float4 values[kMaxPositions];
  float s[kHeadsPerWarp][kMaxPositions];
  const float scale = rsqrtf(static_cast<float>(kHeadDim));
#pragma unroll
  for (int t = 0; t < kMaxPositions; ++t) {
    float4 k = k_pos;
    values[t] = v_pos;
    if (t < pos) {
      k = Bf16x4(
          __ldcg(reinterpret_cast<const uint2*>(kv.Key(a.layer, g, t)) + lane));
      values[t] = Bf16x4(__ldcg(
          reinterpret_cast<const uint2*>(kv.Value(a.layer, g, t)) + lane));
    }
#pragma unroll
    for (int i = 0; i < kHeadsPerWarp; ++i) s[i][t] = Dot(q[i], k);
  }
  float m[kHeadsPerWarp];
#pragma unroll
  for (int i = 0; i < kHeadsPerWarp; ++i) m[i] = -INFINITY;
#pragma unroll
  for (int t = 0; t < kMaxPositions; ++t) {
    const bool live = t <= pos;
#pragma unroll
    for (int i = 0; i < kHeadsPerWarp; ++i) {
      s[i][t] = live ? WarpSum(s[i][t]) * scale : -INFINITY;
      m[i] = fmaxf(m[i], s[i][t]);
    }
  }
  float l[kHeadsPerWarp];
  float4 o[kHeadsPerWarp];
#pragma unroll
  for (int i = 0; i < kHeadsPerWarp; ++i) {
    l[i] = 0.f;
    o[i] = make_float4(0.f, 0.f, 0.f, 0.f);
  }
#pragma unroll
  for (int t = 0; t < kMaxPositions; ++t) {
    const float4 v = values[t];
#pragma unroll
    for (int i = 0; i < kHeadsPerWarp; ++i) {
      const float weight = expf(s[i][t] - m[i]);
      l[i] += weight;
      o[i] = make_float4(fmaf(weight, v.x, o[i].x), fmaf(weight, v.y, o[i].y),
                         fmaf(weight, v.z, o[i].z), fmaf(weight, v.w, o[i].w));
    }
  }
#pragma unroll
  for (int i = 0; i < kHeadsPerWarp; ++i) {
    const float inv = 1.f / l[i];
    StoreBf16x4(out + (h + i) * kHeadDim + 4 * lane,
                make_float4(o[i].x * inv, o[i].y * inv, o[i].z * inv,
                            o[i].w * inv));
  }
  __syncthreads();
}

// ShortAttention for a context too long to hold every position's key and
// value in registers at once, as the Qwen3-Omni code predictor's 16 are: the
// positions go in chunks of kChunk, merged by a running max and sum, and a
// chunk that starts past `pos` is skipped. A chunk's cached keys and values
// are loaded together, one L2 round trip a chunk, and the first chunk's are
// loaded before anything this step computes: after the cache write the
// compiler may not hoist them. S2 Pro's Fast AR keeps ShortAttention, whose
// register budget the decode megakernel is tuned to.
template <typename D, int kMaxPositions, int kChunk>
__device__ inline void ChunkedShortAttention(const float* qkv_rows,
                                             const AttentionStep& a,
                                             const LayerWeights& w, float eps,
                                             __nv_bfloat16* out) {
  constexpr int kHeadsPerWarp = D::kQHeads / kWarps;
  static_assert(D::kQHeads == kHeadsPerWarp * kWarps &&
                    (kHeadsPerWarp == 1 || kHeadsPerWarp == 2),
                "every warp takes one or two whole heads");
  static_assert(D::kGqa % kHeadsPerWarp == 0,
                "a warp's heads share one KV group");
  static_assert(kMaxPositions % kChunk == 0 && kChunk < kMaxPositions);
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int pos = a.pos;
  const int h = kHeadsPerWarp * warp;
  const int g = h / D::kGqa;

  // A group's positions are kHeadDim apart, so one base pointer each with
  // constant offsets addresses a whole chunk.
  constexpr int kPosStride = kHeadDim / 4;
  const uint2* k_base =
      reinterpret_cast<const uint2*>(a.kv.Key(a.layer, g, 0)) + lane;
  const uint2* v_base =
      reinterpret_cast<const uint2*>(a.kv.Value(a.layer, g, 0)) + lane;
  uint2 k_cached[kChunk];
  uint2 v_cached[kChunk];
  auto load_chunk = [&](int first) {
#pragma unroll
    for (int j = 0; j < kChunk; ++j) {
      if (first + j < pos) {
        k_cached[j] = __ldcg(k_base + (first + j) * kPosStride);
        v_cached[j] = __ldcg(v_base + (first + j) * kPosStride);
      }
    }
  };
  load_chunk(0);

  const float4* qkv = reinterpret_cast<const float4*>(qkv_rows);
  constexpr int kHeadVecs = kHeadDim / 4;
  float4 q[kHeadsPerWarp];
#pragma unroll
  for (int i = 0; i < kHeadsPerWarp; ++i) {
    q[i] = RotateLane<D>(
        a, lane,
        QkNormLane(__ldcg(qkv + (h + i) * kHeadVecs + lane), w.q_norm, eps,
                   lane));
  }
  float4 k_pos;
  float4 v_pos;
  LoadKvLane<D>(qkv_rows, a, w.k_norm, eps, g, lane, k_pos, v_pos);
  if (blockIdx.x == 0 && h % D::kGqa == 0) {
    StoreKvLane(a, g, lane, k_pos, v_pos);
  }

  const float scale = rsqrtf(static_cast<float>(kHeadDim));
  float m[kHeadsPerWarp];
  float l[kHeadsPerWarp];
  float4 o[kHeadsPerWarp];
#pragma unroll
  for (int i = 0; i < kHeadsPerWarp; ++i) {
    m[i] = -INFINITY;
    l[i] = 0.f;
    o[i] = make_float4(0.f, 0.f, 0.f, 0.f);
  }
  // One chunk: score, merge into the running max and sum, weigh.
  auto attend_chunk = [&](int first) {
    // Each head's score of position first + j; positions past `pos` score
    // -inf and weigh 0.
    float s[kHeadsPerWarp][kChunk];
#pragma unroll
    for (int j = 0; j < kChunk; ++j) {
      const float4 k = first + j < pos ? Bf16x4(k_cached[j]) : k_pos;
#pragma unroll
      for (int i = 0; i < kHeadsPerWarp; ++i) s[i][j] = Dot(q[i], k);
    }
    float m_new[kHeadsPerWarp];
#pragma unroll
    for (int i = 0; i < kHeadsPerWarp; ++i) m_new[i] = m[i];
#pragma unroll
    for (int j = 0; j < kChunk; ++j) {
      const bool live = first + j <= pos;
#pragma unroll
      for (int i = 0; i < kHeadsPerWarp; ++i) {
        s[i][j] = live ? WarpSum(s[i][j]) * scale : -INFINITY;
        m_new[i] = fmaxf(m_new[i], s[i][j]);
      }
    }
#pragma unroll
    for (int i = 0; i < kHeadsPerWarp; ++i) {
      const float correction = expf(m[i] - m_new[i]);
      l[i] *= correction;
      o[i] = make_float4(o[i].x * correction, o[i].y * correction,
                         o[i].z * correction, o[i].w * correction);
      m[i] = m_new[i];
    }
#pragma unroll
    for (int j = 0; j < kChunk; ++j) {
      const float4 v = first + j < pos ? Bf16x4(v_cached[j]) : v_pos;
#pragma unroll
      for (int i = 0; i < kHeadsPerWarp; ++i) {
        const float weight = expf(s[i][j] - m[i]);
        l[i] += weight;
        o[i] = make_float4(
            fmaf(weight, v.x, o[i].x), fmaf(weight, v.y, o[i].y),
            fmaf(weight, v.z, o[i].z), fmaf(weight, v.w, o[i].w));
      }
    }
  };
#pragma unroll 1
  for (int first = 0; first < kMaxPositions && first <= pos;
       first += kChunk) {
    if (first > 0) load_chunk(first);
    attend_chunk(first);
  }
#pragma unroll
  for (int i = 0; i < kHeadsPerWarp; ++i) {
    const float inv = 1.f / l[i];
    StoreBf16x4(out + (h + i) * kHeadDim + 4 * lane,
                make_float4(o[i].x * inv, o[i].y * inv, o[i].z * inv,
                            o[i].w * inv));
  }
  __syncthreads();
}

__device__ __forceinline__ float Silu(float x) { return x / (1.f + expf(-x)); }

__device__ __forceinline__ int64_t GlobalTimer() {
  int64_t t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

__device__ __forceinline__ int SmId() {
  int id;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(id));
  return id;
}

// Phase 1's arithmetic: this CTA's share [*begin, *begin + *rows) of the q, k
// and v rows of RMSNorm(x) · wqkv, into sh.ys. x is `local` when non-null,
// else `global`, as RmsNorm takes them. The decode's per-code QKV table is
// built through this too, so its rows match the layer's bit for bit when the
// grid is the same size.
template <typename D>
__device__ inline void QkvRows(const LayerWeights& w, const float* global,
                               const float* local, float eps, Shared<D>& sh,
                               int* begin, int* rows) {
  RmsNorm<D>(global, local, w.attention_norm, eps,
             reinterpret_cast<__nv_bfloat16*>(sh.xs), nullptr, sh.red);
  *begin = RowBegin(D::kQkvRows, blockIdx.x);
  *rows = RowBegin(D::kQkvRows, blockIdx.x + 1) - *begin;
  LayerGemv(w.wqkv, D::kDim, *begin, *rows, sh);
}

// Makes x0, which every CTA holds whole, the next layer's input: copies this
// CTA's O rows of it into `residual`, and into `dump` when non-null.
template <typename D>
__device__ void PublishInput(const float* x0, float* residual, float* dump) {
  __syncthreads();
  const int begin = RowBegin(D::kDim, blockIdx.x);
  const int end = RowBegin(D::kDim, blockIdx.x + 1);
  for (int i = begin + threadIdx.x; i < end; i += kThreads) {
    residual[i] = x0[i];
    if (dump != nullptr) dump[i] = x0[i];
  }
}

// How a layer attends. kSplit gives each (query head, sequence chunk) item
// to one CTA and merges the items' partials after a barrier; kShort has every
// CTA attend over the whole short context itself (ShortAttention), which
// saves that barrier.
enum class AttentionKind { kSplit, kShort };

// Runs one decoder layer on every CTA, calling `sync()` for each of its grid
// barriers: five with kSplit attention, four with kShort. The input is
// `b.residual`, or `local` when non-null, in which case every CTA holds the
// whole input in shared memory and has written its own O rows of it to
// `b.residual`. The output is `b.residual`, and each CTA's rows of it also go
// to `dump` when non-null. `next` is the GEMV after the layer, whose slice
// each CTA prefetches.
//
// A kShort layer may take its q, k and v precomputed in `folded_qkv`
// ([D::kQkvRows], read-only). It then skips phase 1 and its barrier: every
// CTA reads the whole of it, and the phase before prefetches the O GEMV.
// kShortPositions is the short attention's longest context; a kShortChunk
// below it selects ChunkedShortAttention with chunks of that many positions.
template <typename D, AttentionKind kAttention = AttentionKind::kSplit,
          int kShortPositions = kShortMaxPositions,
          int kShortChunk = kShortPositions, typename Sync>
__device__ void DecoderLayer(const LayerWeights& w, const LayerBuffers& b,
                             const AttentionStep& a, const float* local,
                             const GemvSlice& next, float* dump, Sync& sync,
                             Shared<D>& sh, const float* folded_qkv = nullptr) {
  __nv_bfloat16* xs = reinterpret_cast<__nv_bfloat16*>(sh.xs);
  const int cta = blockIdx.x;

  const float* qkv = folded_qkv;
  if (qkv == nullptr) {
    int begin;
    int rows;
    QkvRows(w, b.residual, local, b.eps, sh, &begin, &rows);
    PrefetchSliceL2(w.wo, D::kQDim, D::kDim, 1, b.prefetch_bytes);
    for (int i = threadIdx.x; i < rows; i += kThreads) {
      b.qkv[begin + i] = sh.ys[i];
    }
    sync();
    qkv = b.qkv;
  }

  if constexpr (kAttention == AttentionKind::kSplit) {
    Attention(b, a, w, sh);
    sync();
    MergeAttention(b, a.splits, sh, xs);
  } else {
    if constexpr (kShortChunk < kShortPositions) {
      ChunkedShortAttention<D, kShortPositions, kShortChunk>(qkv, a, w, b.eps,
                                                             xs);
    } else {
      ShortAttention<D, kShortPositions>(qkv, a, w, b.eps, xs);
    }
  }
  {
    const int begin = RowBegin(D::kDim, cta);
    const int rows = RowBegin(D::kDim, cta + 1) - begin;
    LayerGemv(w.wo, D::kQDim, begin, rows, sh);
    PrefetchGateUp<D>(w.w13, b.prefetch_bytes);
    for (int i = threadIdx.x; i < rows; i += kThreads) {
      b.residual[begin + i] = __ldcg(b.residual + begin + i) + sh.ys[i];
    }
  }
  sync();

  RmsNorm<D>(b.residual, nullptr, w.ffn_norm, b.eps, xs, nullptr, sh.red);
  {
    const int begin = RowBegin(D::kFfn, cta);
    const int pairs = RowBegin(D::kFfn, cta + 1) - begin;
    GateUpGemv(w.w13, begin, pairs, sh);
    PrefetchSliceL2(w.w2, D::kFfn, D::kDim, 1, b.prefetch_bytes);
    for (int i = threadIdx.x; i < pairs; i += kThreads) {
      const float2 gate_up = GateUp(sh, pairs, i);
      b.act[begin + i] = __float2bfloat16(Silu(gate_up.x) * gate_up.y);
    }
  }
  sync();

  for (int i = threadIdx.x; i < D::kFfn / kVecElems; i += kThreads) {
    sh.xs[i] = __ldcg(reinterpret_cast<const uint4*>(b.act) + i);
  }
  {
    const int begin = RowBegin(D::kDim, cta);
    const int rows = RowBegin(D::kDim, cta + 1) - begin;
    LayerGemv(w.w2, D::kFfn, begin, rows, sh);
    Prefetch(next, b.prefetch_bytes);
    for (int i = threadIdx.x; i < rows; i += kThreads) {
      const float r = __ldcg(b.residual + begin + i) + sh.ys[i];
      b.residual[begin + i] = r;
      if (dump != nullptr) dump[begin + i] = r;
    }
  }
  sync();
}

// The first two phases of a short-attention layer whose output nothing
// reads: its QKV GEMV and this step's key and value, so later positions can
// attend to them. Calls `sync()` twice: the second keeps the next GEMV from
// overwriting qkv before CTA 0 has read it.
template <typename D, typename Sync>
__device__ void DecoderLayerKvOnly(const LayerWeights& w, const LayerBuffers& b,
                                   const AttentionStep& a, const float* local,
                                   const GemvSlice& next, Sync& sync,
                                   Shared<D>& sh) {
  int begin;
  int rows;
  QkvRows(w, b.residual, local, b.eps, sh, &begin, &rows);
  Prefetch(next, b.prefetch_bytes);
  for (int i = threadIdx.x; i < rows; i += kThreads) {
    b.qkv[begin + i] = sh.ys[i];
  }
  sync();
  WriteKv<D>(b, a, w);
  sync();
}

}  // namespace s2mk

#endif  // S2MK_CSRC_DECODER_LAYER_CUH_
