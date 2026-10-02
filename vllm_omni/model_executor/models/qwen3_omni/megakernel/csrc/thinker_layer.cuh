// Copyright 2026 Fractalyze Inc. All rights reserved.
//
// The thinker's two blocks as device functions every CTA of a persistent
// launch calls, so a launch can run one block (thinker_attention.cu,
// thinker_moe.cu) or every layer of a decode step (thinker_decode.cu).
//
// Attention (thinker_attention.h), three phases:
//   1. Every CTA normalizes the residual itself and computes its share of the
//      q, k and v rows. Barrier.
//   2. Query head h and chunk s form item h × S + s, one per CTA. An item
//      applies QK-norm and M-RoPE to its head's q and its KV group's k, then
//      attends over its chunk of positions [0, pos] and writes a partial
//      (max, sum, unnormalized output). The first chunk of the first head of
//      each KV group also writes this step's key and value to the cache.
//      Barrier.
//   3. Every CTA merges all partials into the attention output and computes
//      its share of the O rows, added to the residual.
//
// MoE (thinker_moe.h), three phases:
//   1. Every CTA normalizes the residual itself; CTA c computes router rows
//      [RowBegin(128, c), RowBegin(128, c + 1)). Barrier.
//   2. Every CTA picks the same top 8 from the 128 bf16-rounded logits (the
//      larger logit first, the lower expert on a tie) and computes their
//      renormalized softmax weights. It then computes its share of the 8 ×
//      768 (gate, up) row pairs and writes SiLU(gate) × up. Barrier.
//   3. Every CTA stages the 8 experts' activations and computes its share of
//      the 2048 output rows, each the rank-ordered weighted sum of the 8
//      experts' down rows, added to the residual.
//
// Each block's last phase writes the residual rows RowBegin(kThinkerDim, cta)
// from the same rows of residual_in, so the two may be one buffer. Data
// another CTA wrote in the launch is read with __ldcg. The attention loop and
// merge follow decoder_layer.cuh's split attention, with rotate-half M-RoPE
// and vLLM's paged cache, and stay apart so S2 Pro's decode is untouched.

#ifndef S2MK_CSRC_THINKER_LAYER_CUH_
#define S2MK_CSRC_THINKER_LAYER_CUH_

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

#include "decoder_layer.cuh"
#include "gemv_core.cuh"
#include "int4_gemv_core.cuh"
#include "thinker_attention.h"
#include "thinker_moe.h"

namespace s2mk {
namespace thinker {

constexpr int kDim = kThinkerDim;
constexpr int kQHeads = kThinkerQHeads;
constexpr int kKvHeads = kThinkerKvHeads;
constexpr int kGqa = kQHeads / kKvHeads;
constexpr int kHead = kThinkerHeadDim;
constexpr int kHalf = kHead / 2;
constexpr int kQDim = kThinkerQDim;
constexpr int kQkvRows = kThinkerQkvRows;
constexpr int kExperts = kThinkerExperts;
constexpr int kTopK = kThinkerTopK;
constexpr int kFfn = kThinkerExpertFfn;
constexpr int kPairs = kTopK * kFfn;
// A row of packed words, in 16-byte vectors, at each width.
constexpr int kDimVecs = kDim / kInt4Group;
constexpr int kFfnVecs = kFfn / kInt4Group;

static_assert(kHead == 32 * 4, "attention gives each lane 4 dims");
static_assert(3 * kHead <= kThreads, "LoadHead gives q, k, v a thread a dim");

struct AttentionShared {
  uint4 xs[kQDim / kVecElems];  // the GEMV input: h, then attention's output
  float ys[kMaxRowsPerCta];
  float red[kWarps];
  float q[kHead];
  float k[kHead];
  float v[kHead];
  float warp_m[kWarps];
  float warp_l[kWarps];
  float warp_o[kWarps][kHead];
  float merge_weight[kMaxCtas];
  float merge_sum[kMaxCtas];
};

struct MoeShared {
  uint4 h[kDim / kVecElems];  // the normalized residual, bf16
  uint4 act[kTopK * kFfn / kVecElems];  // SiLU(gate) × up per rank, bf16
  float ys[2 * kMaxRowsPerCta];
  float red[kWarps];
  int experts[kTopK];
  float weights[kTopK];
};

union Shared {
  AttentionShared attention;
  MoeShared moe;
};

// What a block takes per step, apart from its params: a decode step reads
// each layer's params in place from global memory and passes these beside
// them, so no params struct is copied to the stack.
struct BlockStep {
  // Attention only: the token attends to cache positions [0, pos]; its key
  // and value go to `slot` (block × block_size + offset), or nowhere when
  // negative; and it turns at M-RoPE positions (temporal, height, width).
  int pos;
  int64_t slot;
  int64_t positions[3];
  const float* residual_in;
  float* residual;
};

// The M-RoPE position of `axis`, without indexing `positions` at run time,
// which would put the step in local memory.
__device__ __forceinline__ int64_t Position(const BlockStep& step, int axis) {
  return axis == 0 ? step.positions[0]
                   : (axis == 1 ? step.positions[1] : step.positions[2]);
}

__device__ __forceinline__ void PrefetchL2(const void* ptr, int64_t bytes) {
  asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(ptr),
               "r"(static_cast<uint32_t>(bytes))
               : "memory");
}

// The int4 counterpart of PrefetchSliceL2 (gemv_core.cuh): two ranges, words
// and scales. Every thinker width keeps both 16-byte multiples, which the
// bulk prefetch requires. One thread calls it.
__device__ __forceinline__ void PrefetchInt4Rows(const int32_t* packed,
                                                 const __nv_bfloat16* scales,
                                                 int k, int64_t begin,
                                                 int rows) {
  PrefetchL2(packed + begin * (k / 8), int64_t{rows} * (k / 2));
  PrefetchL2(scales + begin * (k / kInt4Group),
             int64_t{rows} * (k / kInt4Group) * sizeof(__nv_bfloat16));
}

// ---------------------------------------------------------------- attention

// KV group g's key or value (`base` is kv.key or kv.value) at cache slot
// `slot` (block × block_size + offset).
__device__ __forceinline__ __nv_bfloat16* SlotAt(const PagedKv& kv,
                                                 __nv_bfloat16* base, int g,
                                                 int64_t slot) {
  return base + slot / kv.block_size * kv.block_stride +
         slot % kv.block_size * kv.slot_stride + g * kv.head_stride;
}

// The slot of the sequence's position t, through its block table.
__device__ __forceinline__ int64_t SlotOf(const PagedKv& kv, int t) {
  return int64_t{kv.block_table[t / kv.block_size]} * kv.block_size +
         t % kv.block_size;
}

// KV group g's key or value at the sequence's position t.
__device__ __forceinline__ __nv_bfloat16* KvAt(const PagedKv& kv,
                                               __nv_bfloat16* base, int g,
                                               int t) {
  return SlotAt(kv, base, g, SlotOf(kv, t));
}

// The position frequency i turns at, of the step's (temporal, height, width).
__device__ __forceinline__ int MropeAxis(int i) {
  if (i < kThinkerMropeInterleaved && i % 3 != 0) return i % 3;
  return 0;
}

// This step's q (fp32) for query head h and k, v (bf16-rounded, as the cache
// holds them) for its KV group, after QK-norm and rotate-half M-RoPE, for a
// model of kQH query and kKvH KV heads of kHead dims. P has the fields of
// ThinkerAttentionParams these read.
template <int kQH, int kKvH, typename P>
__device__ inline void LoadHead(const P& p, const BlockStep& step, int h,
                                AttentionShared& sh) {
  constexpr int kGroup = kQH / kKvH;
  constexpr int kQ = kQH * kHead;
  const int t = threadIdx.x;
  const int g = h / kGroup;
  if (t < kHead) {
    sh.q[t] = __ldcg(p.qkv + h * kHead + t);
  } else if (t < 2 * kHead) {
    sh.k[t - kHead] = __ldcg(p.qkv + kQ + g * kHead + (t - kHead));
  } else if (t < 3 * kHead) {
    sh.v[t - 2 * kHead] = RoundBf16(
        __ldcg(p.qkv + kQ + (kKvH + g) * kHead + (t - 2 * kHead)));
  }
  __syncthreads();
  // Warps 0-3 hold q's squares, warps 4-7 k's.
  float square = 0.f;
  if (t < kHead) square = sh.q[t] * sh.q[t];
  if (kHead <= t && t < 2 * kHead) square = sh.k[t - kHead] * sh.k[t - kHead];
  square = WarpSum(square);
  if (t % 32 == 0) sh.red[t / 32] = square;
  __syncthreads();

  // Threads 0-63 rotate q's pairs (i, i + 64), threads 64-127 k's.
  if (t < 2 * kHalf) {
    const bool is_q = t < kHalf;
    const int i = t % kHalf;
    float* x = is_q ? sh.q : sh.k;
    const float* red = sh.red + (is_q ? 0 : 4);
    const float inv =
        rsqrtf((red[0] + red[1] + red[2] + red[3]) / kHead + p.eps);
    const __nv_bfloat16* norm = is_q ? p.q_norm : p.k_norm;
    const float x0 = x[i] * inv * __bfloat162float(norm[i]);
    const float x1 = x[i + kHalf] * inv * __bfloat162float(norm[i + kHalf]);
    const __nv_bfloat16* cs =
        p.cos_sin + Position(step, MropeAxis(i)) * kHead;
    const float c = __bfloat162float(cs[i]);
    const float s = __bfloat162float(cs[kHalf + i]);
    float r0 = x0 * c - x1 * s;
    float r1 = x1 * c + x0 * s;
    if (!is_q) {
      r0 = RoundBf16(r0);
      r1 = RoundBf16(r1);
    }
    x[i] = r0;
    x[i + kHalf] = r1;
  }
  __syncthreads();
}

// Item h × splits + s attends over its chunk and writes its partial.
template <int kQH, int kKvH, typename P>
__device__ inline void Attend(const P& p, const BlockStep& step,
                              AttentionShared& sh) {
  const int item = blockIdx.x;
  if (item >= kQH * p.splits) return;
  const int h = item / p.splits;
  const int s = item % p.splits;
  const int g = h / (kQH / kKvH);
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const PagedKv& kv = p.kv;
  const int pos = step.pos;

  LoadHead<kQH, kKvH>(p, step, h, sh);
  if (h % (kQH / kKvH) == 0 && s == 0 && threadIdx.x < kHead && step.slot >= 0) {
    SlotAt(kv, kv.key, g, step.slot)[threadIdx.x] =
        __float2bfloat16(sh.k[threadIdx.x]);
    SlotAt(kv, kv.value, g, step.slot)[threadIdx.x] =
        __float2bfloat16(sh.v[threadIdx.x]);
  }

  // Chunk s covers [s × len / S, (s + 1) × len / S) of the len = pos + 1
  // positions. Position pos comes from shared memory, the rest from the cache.
  const int len = pos + 1;
  const int begin = s * len / p.splits;
  const int end = (s + 1) * len / p.splits;
  const float scale = rsqrtf(static_cast<float>(kHead));
  const float4 q = make_float4(sh.q[lane * 4], sh.q[lane * 4 + 1],
                               sh.q[lane * 4 + 2], sh.q[lane * 4 + 3]);

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
        kr[u] = __ldcg(reinterpret_cast<const uint2*>(KvAt(kv, kv.key, g, t)) +
                       lane);
        vr[u] = __ldcg(
            reinterpret_cast<const uint2*>(KvAt(kv, kv.value, g, t)) + lane);
      }
    }
#pragma unroll
    for (int u = 0; u < kPosUnroll; ++u) {
      const int t = base + u;
      if (t >= end) break;
      float4 k;
      float4 v;
      if (t == pos) {
        k = make_float4(sh.k[lane * 4], sh.k[lane * 4 + 1], sh.k[lane * 4 + 2],
                        sh.k[lane * 4 + 3]);
        v = make_float4(sh.v[lane * 4], sh.v[lane * 4 + 1], sh.v[lane * 4 + 2],
                        sh.v[lane * 4 + 3]);
      } else {
        k = Bf16x4(kr[u]);
        v = Bf16x4(vr[u]);
      }
      const float score = WarpSum(Dot(q, k)) * scale;
      const float m_new = fmaxf(m, score);
      const float correction = expf(m - m_new);
      const float weight = expf(score - m_new);
      l = l * correction + weight;
      o[0] = fmaf(o[0], correction, weight * v.x);
      o[1] = fmaf(o[1], correction, weight * v.y);
      o[2] = fmaf(o[2], correction, weight * v.z);
      o[3] = fmaf(o[3], correction, weight * v.w);
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
  if (threadIdx.x < kHead) {
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
    p.partial_o[item * kHead + threadIdx.x] = o_all;
    if (threadIdx.x == 0) {
      p.partial_ml[item * 2] = m_all;
      p.partial_ml[item * 2 + 1] = l_all;
    }
  }
}

// Merges every item's partial into the attention output, as bf16 in `out`.
template <int kQH, typename P>
__device__ inline void Merge(const P& p, AttentionShared& sh,
                             __nv_bfloat16* out) {
  const int splits = p.splits;
  const int items = kQH * splits;
  float* m = sh.merge_weight;
  float* l = sh.merge_sum;
  for (int i = threadIdx.x; i < items; i += kThreads) {
    const float2 ml = __ldcg(reinterpret_cast<const float2*>(p.partial_ml) + i);
    m[i] = ml.x;
    l[i] = ml.y;
  }
  __syncthreads();
  // Thread h weighs head h's chunks: exp(m_s - m) / l for the head's overall
  // max m and sum l. A chunk that saw no position (l_s = 0) weighs 0.
  if (threadIdx.x < kQH) {
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
  const float4* partial_o = reinterpret_cast<const float4*>(p.partial_o);
  for (int i = threadIdx.x; i < kQH * kHead / 4; i += kThreads) {
    const int first = i * 4 / kHead * splits;
    const int d4 = i % (kHead / 4);
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
    for (int s = 0; s < splits; ++s) {
      const float f = m[first + s];
      const float4 o = __ldcg(partial_o + (first + s) * (kHead / 4) + d4);
      acc.x = fmaf(o.x, f, acc.x);
      acc.y = fmaf(o.y, f, acc.y);
      acc.z = fmaf(o.z, f, acc.z);
      acc.w = fmaf(o.w, f, acc.w);
    }
    StoreBf16x4(out + i * 4, acc);
  }
}

// The attention block on every CTA, calling sync() for its two barriers.
template <typename Sync>
__device__ void AttentionBlock(const ThinkerAttentionParams& p,
                               const BlockStep& step, Sync& sync,
                               AttentionShared& sh) {
  const int cta = blockIdx.x;
  __nv_bfloat16* xs = reinterpret_cast<__nv_bfloat16*>(sh.xs);

  // 1. RMSNorm, then this CTA's q, k and v rows.
  RmsNorm<ThinkerNormDims>(step.residual_in, nullptr, p.norm, p.eps, xs,
                           nullptr, sh.red);
  {
    const int begin = RowBegin(kQkvRows, cta);
    const int rows = RowBegin(kQkvRows, cta + 1) - begin;
    Int4GemvRows<1, 32, kDim / (32 * kInt4Group)>(
        reinterpret_cast<const uint4*>(p.wqkv_packed), p.wqkv_scales, begin,
        rows, sh.xs, sh.ys);
    for (int i = threadIdx.x; i < rows; i += kThreads) {
      p.qkv[begin + i] = sh.ys[i];
    }
  }
  sync();

  // 2. This CTA's (head, chunk) item.
  Attend<kQHeads, kKvHeads>(p, step, sh);
  sync();

  // 3. The merged attention output, then this CTA's O rows.
  Merge<kQHeads>(p, sh, xs);
  {
    const int begin = RowBegin(kDim, cta);
    const int rows = RowBegin(kDim, cta + 1) - begin;
    Int4GemvRows<1, 32, kQDim / (32 * kInt4Group)>(
        reinterpret_cast<const uint4*>(p.wo_packed), p.wo_scales, begin, rows,
        sh.xs, sh.ys);
    for (int i = threadIdx.x; i < rows; i += kThreads) {
      step.residual[begin + i] = __ldcg(step.residual_in + begin + i) + sh.ys[i];
    }
  }
}

// ---------------------------------------------------------------------- MoE

// A key whose unsigned order is the value descending, then the index
// ascending.
__device__ __forceinline__ uint64_t TopKey(float value, int index) {
  uint32_t u = __float_as_uint(value);
  u = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
  return (uint64_t{u} << 32) | (0xFFFFFFFFu - static_cast<uint32_t>(index));
}

__device__ __forceinline__ uint64_t WarpMaxKey(uint64_t v) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    const uint64_t other = __shfl_xor_sync(0xffffffffu, v, offset);
    v = other > v ? other : v;
  }
  return v;
}

// The calling warp picks the top kK of the kE logits, in rank order, and
// their weights: softmax over the logits, renormalized over the top kK.
// Lane 0 writes experts[kK] and weights[kK].
template <int kE = kExperts, int kK = kTopK>
__device__ inline void RouteWarp(const float* logits, int* experts,
                                 float* weights) {
  const int lane = threadIdx.x % 32;
  constexpr int kPerLane = kE / 32;
  uint64_t keys[kPerLane];
#pragma unroll
  for (int j = 0; j < kPerLane; ++j) {
    const int e = lane + 32 * j;
    keys[j] = TopKey(RoundBf16(__ldcg(logits + e)), e);
  }
  float top_value = 0.f;
  for (int r = 0; r < kK; ++r) {
    uint64_t best = 0;
#pragma unroll
    for (int j = 0; j < kPerLane; ++j) best = keys[j] > best ? keys[j] : best;
    const uint64_t top = WarpMaxKey(best);
#pragma unroll
    for (int j = 0; j < kPerLane; ++j) {
      if (keys[j] == top) keys[j] = 0;
    }
    if (lane == 0) {
      const int e = static_cast<int>(0xFFFFFFFFu - static_cast<uint32_t>(top));
      experts[r] = e;
      // The logit, recovered from the key's order bits.
      uint32_t u = static_cast<uint32_t>(top >> 32);
      u = (u & 0x80000000u) ? (u & 0x7FFFFFFFu) : ~u;
      const float v = __uint_as_float(u);
      if (r == 0) top_value = v;
      weights[r] = expf(v - top_value);
    }
  }
  if (lane == 0) {
    float total = 0.f;
    for (int r = 0; r < kK; ++r) total += weights[r];
    for (int r = 0; r < kK; ++r) weights[r] /= total;
  }
}

// Warp 0 routes the token into sh.experts and sh.weights (RouteWarp).
__device__ inline void Route(const float* logits, MoeShared& sh) {
  if (threadIdx.x >= 32) return;
  RouteWarp(logits, sh.experts, sh.weights);
}

// The MoE block on every CTA, calling sync() for its two barriers.
template <typename Sync>
__device__ void MoeBlock(const ThinkerMoeParams& p, const BlockStep& step,
                         Sync& sync, MoeShared& sh) {
  const int cta = blockIdx.x;
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;

  // 1. RMSNorm, then this CTA's router rows.
  RmsNorm<ThinkerNormDims>(step.residual_in, nullptr, p.norm, p.eps,
                           reinterpret_cast<__nv_bfloat16*>(sh.h), nullptr,
                           sh.red);
  __syncthreads();
  {
    const int begin = RowBegin(kExperts, cta);
    const int end = RowBegin(kExperts, cta + 1);
    for (int e = begin + warp; e < end; e += kWarps) {
      const uint4* w =
          reinterpret_cast<const uint4*>(p.router + int64_t{e} * kDim);
      float acc = 0.f;
      for (int i = lane; i < kDim / kVecElems; i += 32) {
        acc += Dot8(LoadStream(w + i), sh.h[i]);
      }
      acc = WarpSum(acc);
      if (lane == 0) p.router_logits[e] = acc;
    }
  }
  sync();

  // 2. Routing, then this CTA's (gate, up) pairs.
  Route(p.router_logits, sh);
  __syncthreads();
  {
    const int begin = RowBegin(kPairs, cta);
    const int pairs = RowBegin(kPairs, cta + 1) - begin;
    const uint4* w13 = reinterpret_cast<const uint4*>(p.w13_packed);
    auto map = [&](int r) {
      const int pair = begin + r / 2;
      const int e = sh.experts[pair / kFfn];
      const int64_t row =
          int64_t{e} * 2 * kFfn + (r % 2) * kFfn + pair % kFfn;
      return Int4Row{w13 + row * kDimVecs, p.w13_scales + row * kDimVecs,
                     sh.h};
    };
    Int4GemvMappedRows<32, 2>(map, 2 * pairs, sh.ys);
    // The down rows are unknown until routing, so this is their earliest
    // prefetch: one thread per chosen expert.
    if (threadIdx.x < kTopK) {
      const int64_t row = int64_t{sh.experts[threadIdx.x]} * kDim +
                          RowBegin(kDim, cta);
      PrefetchInt4Rows(p.w2_packed, p.w2_scales, kFfn, row,
                       RowBegin(kDim, cta + 1) - RowBegin(kDim, cta));
    }
    for (int i = threadIdx.x; i < pairs; i += kThreads) {
      const float gate = sh.ys[2 * i];
      const float up = sh.ys[2 * i + 1];
      p.act[begin + i] = __float2bfloat16(Silu(gate) * up);
    }
    if (cta == 0 && threadIdx.x < kTopK) {
      p.experts[threadIdx.x] = sh.experts[threadIdx.x];
      p.weights[threadIdx.x] = sh.weights[threadIdx.x];
    }
  }
  sync();

  // 3. The down rows of every chosen expert, summed in rank order.
  for (int i = threadIdx.x; i < kTopK * kFfn / kVecElems; i += kThreads) {
    sh.act[i] = __ldcg(reinterpret_cast<const uint4*>(p.act) + i);
  }
  {
    const int begin = RowBegin(kDim, cta);
    const int rows = RowBegin(kDim, cta + 1) - begin;
    const uint4* w2 = reinterpret_cast<const uint4*>(p.w2_packed);
    auto map = [&](int r) {
      const int rank = r / rows;
      const int64_t row = int64_t{sh.experts[rank]} * kDim + begin + r % rows;
      return Int4Row{w2 + row * kFfnVecs, p.w2_scales + row * kFfnVecs,
                     sh.act + rank * (kFfn / kVecElems)};
    };
    Int4GemvMappedRows<8, 3>(map, kTopK * rows, sh.ys);
    for (int i = threadIdx.x; i < rows; i += kThreads) {
      float moe = 0.f;
      for (int rank = 0; rank < kTopK; ++rank) {
        moe = fmaf(sh.weights[rank], sh.ys[rank * rows + i], moe);
      }
      step.residual[begin + i] = __ldcg(step.residual_in + begin + i) + moe;
    }
  }
}

}  // namespace thinker
}  // namespace s2mk

#endif  // S2MK_CSRC_THINKER_LAYER_CUH_
