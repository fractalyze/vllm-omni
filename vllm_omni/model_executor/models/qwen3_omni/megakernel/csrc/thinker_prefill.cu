// Copyright 2026 Fractalyze Inc. All rights reserved.
//
// One prefill chunk of Qwen3-Omni's thinker (thinker_prefill.h) as a
// persistent launch, nine phases a layer with a grid barrier after each:
//
//   P1. Per token (a CTA each): the last layer's MoE rows summed in rank
//       order into the residual, then RMSNorm into h.
//   P2. q, k, v rows for every token, a dense int4 GEMM on tensor cores.
//   P3. A warp per (KV head, token): the key's QK-norm and M-RoPE; key and
//       value to the chunk's kv buffer and the cache.
//   P4. Attention, a warp per (head, token): q's QK-norm and M-RoPE, then
//       causal online softmax over the cache before the chunk and the
//       chunk's own keys.
//   P5. O rows, a dense GEMM, added to the residual.
//   P6. Per token: RMSNorm into h.
//   P7. Router logits.
//   P8. Every CTA routes every token (the decode's top 8) and lists each
//       expert's (token, rank) slots; then a CTA per (expert, 16-row
//       block) unit of gate and up rows for the expert's tokens, SiLU'd.
//   P9. A CTA per (expert, 64-row block) unit of down rows, one partial
//       per slot.
//
// After the last layer, P1's sum and the final norm. Every sum has one
// order, so the result is deterministic.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <type_traits>

#include "barrier.cuh"
#include "decoder_layer.cuh"
#include "gemv_core.cuh"
#include "int4_mma_core.cuh"
#include "thinker_layer.cuh"
#include "thinker_prefill.h"

namespace s2mk {
namespace {

using thinker::kDim;
using thinker::kExperts;
using thinker::kFfn;
using thinker::kGqa;
using thinker::kHead;
using thinker::kHalf;
using thinker::kKvHeads;
using thinker::kQDim;
using thinker::kQHeads;
using thinker::kQkvRows;
using thinker::kTopK;

constexpr int kMaxTokens = kThinkerPrefillMaxTokens;
constexpr int kMaxSlots = kMaxTokens * kTopK;
constexpr int kBlockRows = 16;
// Row stride of the warps' C fragments in shared memory: off by 8 banks a row.
constexpr int kPartStride = kMaxTokens + 8;
// A dense GEMM's per-warp partials: kWarps slices of 16 rows × 64 tokens.
constexpr int kDenseSmemFloats = kWarps * kBlockRows * kPartStride;

struct RoutingShared {
  int experts[kMaxTokens][kTopK];
  float weights[kMaxTokens][kTopK];
  int count[kExperts];
  int offset[kExperts + 1];
  int used[kExperts];
  int fill[kExperts];  // the next free entry of each expert's list
  int num_used;
  int tokens[kMaxSlots];  // each expert's tokens, experts in order
  int slots[kMaxSlots];  // their (token × kTopK + rank)
};

__device__ __forceinline__ int GlobalWarp() {
  return blockIdx.x * kWarps + threadIdx.x / 32;
}

__device__ __forceinline__ int GridWarps() { return gridDim.x * kWarps; }

// ---------------------------------------------------------------- P1, P6

// Per token, a CTA each: with `moe_layer` ≥ 0, the residual gains that
// layer's MoE rows (Σ over ranks of weight × partial, in rank order); the
// residual then goes to `dump` when set, and RMSNorm × norm to `out`.
__device__ void TokenNorm(const ThinkerPrefillParams& p, int moe_layer,
                          const __nv_bfloat16* norm, __nv_bfloat16* out,
                          float* dump, float* red) {
  constexpr int kPerThread = kDim / kThreads;
  for (int t = blockIdx.x; t < p.tokens; t += gridDim.x) {
    float* row = p.residual + int64_t{t} * kDim;
    float v[kPerThread];
    float squares = 0.f;
#pragma unroll
    for (int j = 0; j < kPerThread; ++j) {
      const int i = threadIdx.x + j * kThreads;
      float x = __ldcg(row + i);
      if (moe_layer >= 0) {
        const float* w = p.weights + (int64_t{moe_layer} * p.tokens + t) * kTopK;
        float moe = 0.f;
        for (int r = 0; r < kTopK; ++r) {
          moe = fmaf(__ldcg(w + r),
                     __ldcg(p.partial + (int64_t{t} * kTopK + r) * kDim + i),
                     moe);
        }
        x += moe;
        row[i] = x;
      }
      if (dump != nullptr) dump[int64_t{t} * kDim + i] = x;
      v[j] = x;
      squares += x * x;
    }
    const float inv = rsqrtf(BlockSum(squares, red) / kDim + p.eps);
#pragma unroll
    for (int j = 0; j < kPerThread; ++j) {
      const int i = threadIdx.x + j * kThreads;
      out[int64_t{t} * kDim + i] =
          __float2bfloat16(v[j] * inv * __bfloat162float(norm[i]));
    }
  }
}

// ---------------------------------------------------------------- P2, P5

// The warp's C fragments d[m][n] to part[m × kBlockRows + row][token], or
// added to what is there with kAdd.
template <bool kAdd, int kMTiles, int kNTiles>
__device__ __forceinline__ void StoreFragments(
    const float (&d)[kMTiles][kNTiles][4], float* part) {
  const int lane = threadIdx.x % 32;
  const int g = lane / 4;
  const int t = lane % 4;
#pragma unroll
  for (int m = 0; m < kMTiles; ++m) {
#pragma unroll
    for (int n = 0; n < kNTiles; ++n) {
      float* r = part + (m * kBlockRows + g) * kPartStride + n * kMmaTokens +
                 2 * t;
      float2* lo = reinterpret_cast<float2*>(r);
      float2* hi = reinterpret_cast<float2*>(r + 8 * kPartStride);
      if constexpr (kAdd) {
        *lo = make_float2(lo->x + d[m][n][0], lo->y + d[m][n][1]);
        *hi = make_float2(hi->x + d[m][n][2], hi->y + d[m][n][3]);
      } else {
        *lo = make_float2(d[m][n][0], d[m][n][1]);
        *hi = make_float2(d[m][n][2], d[m][n][3]);
      }
    }
  }
}


// y[t][row] for this CTA's rows RowBegin(n) of W · x[t]^T and every token,
// handed to store(row, t, y). The CTA's 16-row blocks split their groups
// between warps; the slices meet in shared memory, summed in slice order.
// The first of CTA cta's rows of a dense phase of n rows: whole 16-row
// tiles, so no mma row goes unused (O's 2048 rows on 170 CTAs were 12 a
// CTA); CTAs past the last tile take none.
__device__ __forceinline__ int DenseBegin(int n, int cta) {
  return static_cast<int>(int64_t{cta} * (n / kBlockRows) / gridDim.x) * kBlockRows;
}

template <int kNTiles, typename Store>
__device__ void DenseRows(const int32_t* packed, const __nv_bfloat16* scales,
                          int k, int n, const __nv_bfloat16* x, int tokens,
                          float* partials, const Store& store) {
  const int warp = threadIdx.x / 32;
  const int begin = DenseBegin(n, blockIdx.x);
  const int rows = DenseBegin(n, blockIdx.x + 1) - begin;
  if (rows == 0) return;
  const int groups = k / kInt4Group;
  const int blocks = (rows + kBlockRows - 1) / kBlockRows;
  const int slices = blocks >= kWarps ? 1 : kWarps / blocks;
  for (int unit = warp; unit < blocks * slices; unit += kWarps) {
    const int block = unit % blocks;
    const int slice = unit / blocks;
    float d[kNTiles][4];
#pragma unroll
    for (int i = 0; i < kNTiles; ++i) d[i][0] = d[i][1] = d[i][2] = d[i][3] = 0.f;
    const int row0 = block * kBlockRows;
    Int4MmaRows<kNTiles>(reinterpret_cast<const uint4*>(packed), scales,
                         groups, begin + row0, min(kBlockRows, rows - row0), x,
                         k, nullptr, tokens, slice * groups / slices,
                         (slice + 1) * groups / slices, d);
    StoreFragments<false, 1, kNTiles>(
        reinterpret_cast<const float(&)[1][kNTiles][4]>(d),
        partials + (slice * blocks + block) * kBlockRows * kPartStride);
  }
  __syncthreads();
  // Consecutive threads take consecutive rows: whole lines of y.
  for (int i = threadIdx.x; i < rows * tokens; i += kThreads) {
    const int j = i / rows;
    const int row = i % rows;
    const int block = row / kBlockRows;
    float sum = 0.f;
    for (int slice = 0; slice < slices; ++slice) {
      sum += partials[((slice * blocks + block) * kBlockRows +
                       row % kBlockRows) * kPartStride + j];
    }
    store(begin + row, j, sum);
  }
  __syncthreads();
}

template <typename Store>
__device__ void Dense(const int32_t* packed, const __nv_bfloat16* scales,
                      int k, int n, const __nv_bfloat16* x, int tokens,
                      float* partials, const Store& store) {
  if (tokens <= kMmaTokens) {
    DenseRows<1>(packed, scales, k, n, x, tokens, partials, store);
  } else if (tokens <= 2 * kMmaTokens) {
    DenseRows<2>(packed, scales, k, n, x, tokens, partials, store);
  } else if (tokens <= 4 * kMmaTokens) {
    DenseRows<4>(packed, scales, k, n, x, tokens, partials, store);
  } else {
    DenseRows<8>(packed, scales, k, n, x, tokens, partials, store);
  }
}

// ------------------------------------------------------------------ P3, P4

__device__ __forceinline__ float4 LoadF4(const float* p) {
  return __ldcg(reinterpret_cast<const float4*>(p));
}

// A head's dims 4 lane .. 4 lane + 3 after QK-norm and rotate-half M-RoPE at
// token `token`'s positions. Every lane of the warp calls it.
__device__ __forceinline__ float4 NormRope(const ThinkerPrefillParams& p,
                                           const ThinkerAttentionParams& a,
                                           float4 x, const __nv_bfloat16* norm,
                                           int token, int lane) {
  const float inv = rsqrtf(WarpSum(Dot(x, x)) / kHead + a.eps);
  const float4 w = Bf16x4(*reinterpret_cast<const uint2*>(norm + 4 * lane));
  float y[4] = {x.x * inv * w.x, x.y * inv * w.y, x.z * inv * w.z,
                x.w * inv * w.w};
  float other[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) other[j] = __shfl_xor_sync(0xffffffffu, y[j], 16);
  float out[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int f = (4 * lane + j) % kHalf;
    const int64_t pos =
        p.positions[thinker::MropeAxis(f) * p.positions_stride + token];
    const __nv_bfloat16* cs = a.cos_sin + pos * kHead;
    const float c = __bfloat162float(cs[f]);
    const float s = __bfloat162float(cs[kHalf + f]);
    out[j] = lane < 16 ? fmaf(-other[j], s, y[j] * c) : fmaf(other[j], s, y[j] * c);
  }
  return make_float4(out[0], out[1], out[2], out[3]);
}

// Token i's key (after QK-norm and M-RoPE) or value of KV head g in the
// chunk's kv buffer, dims 4 lane .. 4 lane + 3.
__device__ __forceinline__ __nv_bfloat16* ChunkKvAt(
    const ThinkerPrefillParams& p, int i, int g, bool value, int lane) {
  return p.kv + ((int64_t{i} * kKvHeads + g) * 2 + value) * kHead + 4 * lane;
}

// Every key of the chunk is rotated once here, not once per query that
// reads it.
__device__ void ChunkKv(const ThinkerPrefillParams& p,
                        const ThinkerAttentionParams& a) {
  const int lane = threadIdx.x % 32;
  for (int item = GlobalWarp(); item < p.tokens * kKvHeads;
       item += GridWarps()) {
    const int i = item / kKvHeads;
    const int g = item % kKvHeads;
    const float* qkv_i = p.qkv + int64_t{i} * kQkvRows + kQDim;
    const float4 k = NormRope(p, a, LoadF4(qkv_i + g * kHead + 4 * lane),
                              a.k_norm, i, lane);
    const float4 v = LoadF4(qkv_i + (kKvHeads + g) * kHead + 4 * lane);
    StoreBf16x4(ChunkKvAt(p, i, g, false, lane), k);
    StoreBf16x4(ChunkKvAt(p, i, g, true, lane), v);
    const int64_t slot = p.slot_mapping[i];
    if (slot >= 0) {
      StoreBf16x4(thinker::SlotAt(a.kv, a.kv.key, g, slot) + 4 * lane, k);
      StoreBf16x4(thinker::SlotAt(a.kv, a.kv.value, g, slot) + 4 * lane, v);
    }
  }
}

// One online-softmax step of a warp's (m, l, o) for a key and value.
// Keys a warp scores together: their warp sums interleave, so a key's
// shuffle chain does not wait for the key before.
constexpr int kKeysAtOnce = 4;

// One online-softmax step of a warp's (m, l, o) over up to kKeysAtOnce keys
// and values; keys past `count` are left out.
__device__ __forceinline__ void Attend(float4 q, const float4 (&k)[kKeysAtOnce],
                                       const float4 (&v)[kKeysAtOnce], int count,
                                       float scale, float& m, float& l, float4& o) {
  float score[kKeysAtOnce];
#pragma unroll
  for (int u = 0; u < kKeysAtOnce; ++u) score[u] = Dot(q, k[u]);
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
#pragma unroll
    for (int u = 0; u < kKeysAtOnce; ++u) {
      score[u] += __shfl_xor_sync(0xffffffffu, score[u], offset);
    }
  }
  float m_new = m;
#pragma unroll
  for (int u = 0; u < kKeysAtOnce; ++u) {
    score[u] = u < count ? score[u] * scale : -INFINITY;
    m_new = fmaxf(m_new, score[u]);
  }
  const float correction = expf(m - m_new);
  l *= correction;
  o = make_float4(o.x * correction, o.y * correction, o.z * correction,
                  o.w * correction);
#pragma unroll
  for (int u = 0; u < kKeysAtOnce; ++u) {
    const float weight = expf(score[u] - m_new);
    l += weight;
    o = make_float4(fmaf(weight, v[u].x, o.x), fmaf(weight, v[u].y, o.y),
                    fmaf(weight, v[u].z, o.z), fmaf(weight, v[u].w, o.w));
  }
  m = m_new;
}

__device__ __forceinline__ float4 LoadBf16x4(const __nv_bfloat16* p) {
  return Bf16x4(__ldcg(reinterpret_cast<const uint2*>(p)));
}

// Attends over keys 0 .. count - 1, kKeysAtOnce at a time; key(j, value)
// loads key or value j's dims 4 lane .. 4 lane + 3.
template <typename KeyAt>
__device__ __forceinline__ void AttendAll(float4 q, int count, float scale,
                                          const KeyAt& key, float& m, float& l,
                                          float4& o) {
  for (int j = 0; j < count; j += kKeysAtOnce) {
    float4 k[kKeysAtOnce];
    float4 v[kKeysAtOnce];
#pragma unroll
    for (int u = 0; u < kKeysAtOnce; ++u) {
      const bool in = j + u < count;
      k[u] = in ? key(j + u, false) : make_float4(0.f, 0.f, 0.f, 0.f);
      v[u] = in ? key(j + u, true) : make_float4(0.f, 0.f, 0.f, 0.f);
    }
    Attend(q, k, v, count - j, scale, m, l, o);
  }
}

// A CTA's warps take kWarps consecutive (token, head) items, one token's
// heads, which share kStagedGroups KV groups.
constexpr int kStagedGroups = kWarps / kGqa;
static_assert(kQHeads % kWarps == 0, "a CTA's items are one token's heads");

// Each CTA stages its token's chunk keys and values for its heads' KV groups
// in shared memory, so a warp's keys come from there instead of one L2
// round trip a batch. Keys cached before the chunk stay in L2.
__device__ void PrefillAttention(const ThinkerPrefillParams& p,
                                 const ThinkerAttentionParams& a, int p0,
                                 float* smem) {
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const float scale = rsqrtf(static_cast<float>(kHead));
  const int items = p.tokens * kQHeads;
  auto* staged = reinterpret_cast<__nv_bfloat16*>(smem);
  for (int base = blockIdx.x * kWarps; base < items; base += GridWarps()) {
    const int i = base / kQHeads;
    const int g0 = base % kQHeads / kGqa;
    // [key j][group][key, value][kHead] for j = 0 .. i.
    constexpr int kVecsPerRow = kHead * 2 / 16;
    const int vecs = (i + 1) * kStagedGroups * 2 * kVecsPerRow;
    for (int n = threadIdx.x; n < vecs; n += kThreads) {
      const int row = n / kVecsPerRow;
      const int key = row / (kStagedGroups * 2);
      const int group = row / 2 % kStagedGroups;
      const bool value = row % 2 != 0;
      reinterpret_cast<uint4*>(staged)[n] = __ldcg(
          reinterpret_cast<const uint4*>(ChunkKvAt(p, key, g0 + group, value, 0)) +
          n % kVecsPerRow);
    }
    __syncthreads();
    const int h = base % kQHeads + warp;
    const int g = h / kGqa;
    const float4 q =
        NormRope(p, a, LoadF4(p.qkv + int64_t{i} * kQkvRows + h * kHead + 4 * lane),
                 a.q_norm, i, lane);
    float m = -INFINITY;
    float l = 0.f;
    float4 o = make_float4(0.f, 0.f, 0.f, 0.f);
    AttendAll(q, p0, scale, [&](int j, bool value) {
      return LoadBf16x4(thinker::KvAt(a.kv, value ? a.kv.value : a.kv.key, g, j) +
                        4 * lane);
    }, m, l, o);
    AttendAll(q, i + 1, scale, [&](int j, bool value) {
      return Bf16x4(*reinterpret_cast<const uint2*>(
          staged + ((j * kStagedGroups + (g - g0)) * 2 + value) * kHead + 4 * lane));
    }, m, l, o);
    const float inv = 1.f / l;
    StoreBf16x4(p.attn + int64_t{i} * kQDim + h * kHead + 4 * lane,
                make_float4(o.x * inv, o.y * inv, o.z * inv, o.w * inv));
    __syncthreads();
  }
}

// ---------------------------------------------------------------------- P7

__device__ void Router(const ThinkerPrefillParams& p,
                       const ThinkerMoeParams& moe) {
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  for (int e = RowBegin(kExperts, blockIdx.x);
       e < RowBegin(kExperts, blockIdx.x + 1); ++e) {
    const uint4* w = reinterpret_cast<const uint4*>(moe.router + int64_t{e} * kDim);
    for (int t = warp; t < p.tokens; t += kWarps) {
      const uint4* x = reinterpret_cast<const uint4*>(p.h + int64_t{t} * kDim);
      float acc = 0.f;
      for (int v = lane; v < kDim / kVecElems; v += 32) {
        acc += Dot8(__ldg(w + v), __ldcg(x + v));
      }
      acc = WarpSum(acc);
      if (lane == 0) p.router_logits[t * kExperts + e] = acc;
    }
  }
}

// ---------------------------------------------------------------------- P8

// The MoE phases run CTA-wide units, k split between the warps and the
// slices summed in shared memory in warp order. Routing skews (one expert
// may take most of the chunk), and with a unit a warp, the busiest expert's
// warps were the phase's critical path.
//
// Slots for a gate-up unit's gate and up slices: warps w and w + 8 share
// slot w, as every warp's at 64 tokens would overflow shared memory.
constexpr int kGateUpSlots = 8;
constexpr int kMoeSmemFloats = kGateUpSlots * 2 * kBlockRows * kPartStride;
constexpr int kSmemFloats =
    kMoeSmemFloats > kDenseSmemFloats ? kMoeSmemFloats : kDenseSmemFloats;
static_assert(kMaxTokens * kStagedGroups * 2 * kHead * 2 <=
                  kSmemFloats * static_cast<int>(sizeof(float)),
              "a token's staged keys and values fit the dense phases' memory");

// Groups a warp streams per batch, for a slice of `groups`: fewer when its
// row and token tiles need the registers.
constexpr int MoeDepth(int m_tiles, int n_tiles, int groups) {
  const int depth =
      m_tiles * n_tiles <= 4 ? 16 : m_tiles * n_tiles <= 8 ? 8 : 4;
  const int need = (groups + 3) / 4 * 4;
  return depth < need ? depth : need;
}

// Every CTA routes every token and lists each expert's slots; CTA 0 records
// the layer's routing. A token's mma column never mixes with another's, so
// the lists' order, which the atomics leave open, changes no result.
__device__ void RouteTokens(const ThinkerPrefillParams& p, int layer,
                            RoutingShared& rs) {
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  for (int t = warp; t < p.tokens; t += kWarps) {
    thinker::RouteWarp(p.router_logits + t * kExperts, rs.experts[t],
                       rs.weights[t]);
  }
  if (threadIdx.x < kExperts) rs.count[threadIdx.x] = 0;
  __syncthreads();
  const int slots = p.tokens * kTopK;
  for (int s = threadIdx.x; s < slots; s += kThreads) {
    atomicAdd(&rs.count[rs.experts[s / kTopK][s % kTopK]], 1);
  }
  __syncthreads();
  if (warp == 0) {
    constexpr int kPerLane = kExperts / 32;
    int count[kPerLane];
    int sum = 0;
    int used = 0;
#pragma unroll
    for (int j = 0; j < kPerLane; ++j) {
      count[j] = rs.count[kPerLane * lane + j];
      sum += count[j];
      used += count[j] > 0;
    }
    int sum_before = sum;
    int used_before = used;
#pragma unroll
    for (int d = 1; d < 32; d *= 2) {
      const int s = __shfl_up_sync(0xffffffffu, sum_before, d);
      const int u = __shfl_up_sync(0xffffffffu, used_before, d);
      if (lane >= d) {
        sum_before += s;
        used_before += u;
      }
    }
    sum_before -= sum;
    used_before -= used;
#pragma unroll
    for (int j = 0; j < kPerLane; ++j) {
      const int e = kPerLane * lane + j;
      rs.offset[e] = sum_before;
      rs.fill[e] = sum_before;
      sum_before += count[j];
      if (count[j] > 0) rs.used[used_before++] = e;
    }
    if (lane == 31) {
      rs.offset[kExperts] = sum_before;
      rs.num_used = used_before;
    }
  }
  __syncthreads();
  for (int s = threadIdx.x; s < slots; s += kThreads) {
    const int at = atomicAdd(&rs.fill[rs.experts[s / kTopK][s % kTopK]], 1);
    rs.tokens[at] = s / kTopK;
    rs.slots[at] = s;
  }
  if (blockIdx.x == 0) {
    for (int s = threadIdx.x; s < slots; s += kThreads) {
      const int64_t out = (int64_t{layer} * p.tokens) * kTopK + s;
      p.experts[out] = rs.experts[s / kTopK][s % kTopK];
      p.weights[out] = rs.weights[s / kTopK][s % kTopK];
    }
  }
  __syncthreads();
}

constexpr int kGateUpBlocks = kFfn / kBlockRows;

// Gate and up rows block × 16 .. + 16 of expert e for its `count` tokens,
// SiLU'd into act; each warp takes a sixteenth of k.
template <int kNTiles>
__device__ void GateUpUnit(const ThinkerPrefillParams& p,
                           const ThinkerMoeParams& moe, int e, int block,
                           const int* tokens, const int* slots, int count,
                           float* parts) {
  const int warp = threadIdx.x / 32;
  constexpr int kGroups = kDim / kInt4Group;
  constexpr int kSlice = kGroups / kWarps;
  // One pass streams gate and up together, sharing each token fragment.
  float d[2][kNTiles][4];
#pragma unroll
  for (int m = 0; m < 2; ++m) {
#pragma unroll
    for (int i = 0; i < kNTiles; ++i) {
      d[m][i][0] = d[m][i][1] = d[m][i][2] = d[m][i][3] = 0.f;
    }
  }
  Int4MmaTiles<2, kNTiles, MoeDepth(2, kNTiles, kSlice)>(
      reinterpret_cast<const uint4*>(moe.w13_packed), moe.w13_scales, kGroups,
      e * 2 * kFfn + block * kBlockRows, kFfn, kBlockRows, p.h, kDim, tokens,
      count, warp * kSlice, (warp + 1) * kSlice, d);
  float* slot = parts + warp % kGateUpSlots * 2 * kBlockRows * kPartStride;
  if (warp < kGateUpSlots) StoreFragments<false, 2, kNTiles>(d, slot);
  __syncthreads();
  if (warp >= kGateUpSlots) StoreFragments<true, 2, kNTiles>(d, slot);
  __syncthreads();
  for (int i = threadIdx.x; i < kBlockRows * count; i += kThreads) {
    const int j = i / kBlockRows;
    const int row = i % kBlockRows;
    float gate = 0.f;
    float up = 0.f;
    for (int w = 0; w < kGateUpSlots; ++w) {
      const float* part = parts + w * 2 * kBlockRows * kPartStride;
      gate += part[row * kPartStride + j];
      up += part[(kBlockRows + row) * kPartStride + j];
    }
    p.act[int64_t{slots[j]} * kFfn + block * kBlockRows + row] =
        __float2bfloat16(Silu(gate) * up);
  }
  __syncthreads();
}

// ---------------------------------------------------------------------- P9

// A down unit's warps: kDownTiles row tiles × kDownSlices slices of k.
constexpr int kDownTiles = 4;
constexpr int kDownSlices = kWarps / kDownTiles;
constexpr int kDownUnitRows = kDownTiles * kBlockRows;
constexpr int kDownBlocks = kDim / kDownUnitRows;
static_assert(kWarps * kBlockRows * kPartStride <= kMoeSmemFloats,
              "a down unit's slices fit the gate-up slots");

// Down rows block × 64 .. + 64 of expert e, one partial per slot.
template <int kNTiles>
__device__ void DownUnit(const ThinkerPrefillParams& p,
                         const ThinkerMoeParams& moe, int e, int block,
                         const int* slots, int count, float* parts) {
  const int warp = threadIdx.x / 32;
  const int tile = warp % kDownTiles;
  const int slice = warp / kDownTiles;
  constexpr int kGroups = kFfn / kInt4Group;
  constexpr int kSlice = kGroups / kDownSlices;
  float d[1][kNTiles][4];
#pragma unroll
  for (int i = 0; i < kNTiles; ++i) {
    d[0][i][0] = d[0][i][1] = d[0][i][2] = d[0][i][3] = 0.f;
  }
  Int4MmaTiles<1, kNTiles, MoeDepth(1, kNTiles, kSlice)>(
      reinterpret_cast<const uint4*>(moe.w2_packed), moe.w2_scales, kGroups,
      e * kDim + block * kDownUnitRows + tile * kBlockRows, 0, kBlockRows,
      p.act, kFfn, slots, count, slice * kSlice, (slice + 1) * kSlice, d);
  StoreFragments<false, 1, kNTiles>(
      d, parts + (slice * kDownTiles + tile) * kBlockRows * kPartStride);
  __syncthreads();
  for (int i = threadIdx.x; i < kDownUnitRows * count; i += kThreads) {
    const int j = i / kDownUnitRows;
    const int row = i % kDownUnitRows;
    float sum = 0.f;
    for (int s = 0; s < kDownSlices; ++s) {
      sum += parts[((s * kDownTiles + row / kBlockRows) * kBlockRows +
                    row % kBlockRows) * kPartStride + j];
    }
    p.partial[int64_t{slots[j]} * kDim + block * kDownUnitRows + row] = sum;
  }
  __syncthreads();
}

// Starts unit u's rows toward L2 while the CTA computes the unit before:
// between a unit's loads its warps reduce in shared memory, and HBM would
// sit idle. Gate-up units are 16 gate rows and the 16 up rows kFfn after;
// down units are kDownUnitRows rows. Called by one thread.
__device__ void PrefetchUnit(const ThinkerMoeParams& moe, const RoutingShared& rs,
                             int u, bool gate_up) {
  if (gate_up) {
    if (u >= rs.num_used * kGateUpBlocks) return;
    const int64_t row = int64_t{rs.used[u / kGateUpBlocks]} * 2 * kFfn +
                        u % kGateUpBlocks * kBlockRows;
    thinker::PrefetchInt4Rows(moe.w13_packed, moe.w13_scales, kDim, row, kBlockRows);
    thinker::PrefetchInt4Rows(moe.w13_packed, moe.w13_scales, kDim, row + kFfn,
                              kBlockRows);
  } else {
    if (u >= rs.num_used * kDownBlocks) return;
    const int64_t row = int64_t{rs.used[u / kDownBlocks]} * kDim +
                        u % kDownBlocks * kDownUnitRows;
    thinker::PrefetchInt4Rows(moe.w2_packed, moe.w2_scales, kFfn, row, kDownUnitRows);
  }
}

// Runs unit(kNTiles) with n-tiles enough for `count` tokens.
template <typename Unit>
__device__ __forceinline__ void ForTiles(int count, const Unit& unit) {
  if (count <= kMmaTokens) {
    unit(std::integral_constant<int, 1>{});
  } else if (count <= 2 * kMmaTokens) {
    unit(std::integral_constant<int, 2>{});
  } else if (count <= 4 * kMmaTokens) {
    unit(std::integral_constant<int, 4>{});
  } else {
    unit(std::integral_constant<int, 8>{});
  }
}

// This CTA's rows of a dense phase's int4 weights, n rows of k, into L2.
__device__ __forceinline__ void PrefetchDense(const int32_t* packed,
                                              const __nv_bfloat16* scales,
                                              int k, int n) {
  const int begin = DenseBegin(n, blockIdx.x);
  const int rows = DenseBegin(n, blockIdx.x + 1) - begin;
  if (rows > 0) thinker::PrefetchInt4Rows(packed, scales, k, begin, rows);
}

// At the barrier after phase `index`, the CTA's first thread starts this
// CTA's slice of a later dense phase toward L2, as the decode does: the
// dense phases read little each, so their HBM latency, not bandwidth, sets
// their pace. Called by one thread.
__device__ void PrefetchAhead(const ThinkerPrefillParams& p, int index) {
  const int layer = index / kThinkerPrefillBarriersPerLayer;
  switch (index % kThinkerPrefillBarriersPerLayer) {
    case 1: {  // after q, k, v: this layer's O rows, two phases on
      const ThinkerAttentionParams& a = p.attention[layer];
      PrefetchDense(a.wo_packed, a.wo_scales, kQDim, kDim);
      break;
    }
    case 4: {  // after O: the router rows, two phases on
      const int begin = RowBegin(kExperts, blockIdx.x);
      thinker::PrefetchL2(p.moe[layer].router + int64_t{begin} * kDim,
                 int64_t{RowBegin(kExperts, blockIdx.x + 1) - begin} * kDim *
                     sizeof(__nv_bfloat16));
      break;
    }
    case 8:  // after the down rows: the next layer's q, k, v
      if (layer + 1 < p.num_layers) {
        const ThinkerAttentionParams& a = p.attention[layer + 1];
        PrefetchDense(a.wqkv_packed, a.wqkv_scales, kDim, kQkvRows);
      }
      break;
    default:
      break;
  }
}

__global__ void __launch_bounds__(kThreads, 1)
    ThinkerPrefillKernel(const __grid_constant__ ThinkerPrefillParams p) {
  extern __shared__ float partials[];
  __shared__ RoutingShared rs;
  __shared__ float red[kWarps];
  GridBarrier barrier(p.sync, p.error, p.timeout_ns, 0);
  int64_t* stamps =
      p.profile == nullptr
          ? nullptr
          : p.profile + int64_t{blockIdx.x} * 2 *
                            kThinkerPrefillBarriersPerLayer * p.num_layers;
  // The arrival stamp waits for the whole CTA to finish the phase.
  auto sync = [&] {
    if (threadIdx.x == 0) PrefetchAhead(p, barrier.index());
    if (stamps != nullptr) {
      __syncthreads();
      if (threadIdx.x == 0) stamps[2 * barrier.index()] = GlobalTimer();
    }
    barrier.Sync();
    if (stamps != nullptr && threadIdx.x == 0) {
      stamps[2 * (barrier.index() - 1) + 1] = GlobalTimer();
    }
  };
  const int p0 = p.seq_len[0] - p.tokens;
  const int tokens = p.tokens;
  if (threadIdx.x == 0) {
    PrefetchDense(p.attention[0].wqkv_packed, p.attention[0].wqkv_scales, kDim,
                  kQkvRows);
  }

  for (int layer = 0; layer < p.num_layers; ++layer) {
    const ThinkerAttentionParams& a = p.attention[layer];
    const ThinkerMoeParams& moe = p.moe[layer];
    TokenNorm(p, layer - 1, a.norm, p.h,
              layer == p.hidden_layer ? p.hidden : nullptr, red);
    sync();
    Dense(a.wqkv_packed, a.wqkv_scales, kDim, kQkvRows, p.h, tokens, partials,
          [&](int row, int t, float y) { p.qkv[int64_t{t} * kQkvRows + row] = y; });
    sync();
    ChunkKv(p, a);
    sync();
    PrefillAttention(p, a, p0, partials);
    sync();
    Dense(a.wo_packed, a.wo_scales, kQDim, kDim, p.attn, tokens, partials,
          [&](int row, int t, float y) {
            float* r = p.residual + int64_t{t} * kDim + row;
            *r = __ldcg(r) + y;
          });
    sync();
    TokenNorm(p, -1, moe.norm, p.h, nullptr, red);
    sync();
    Router(p, moe);
    sync();
    RouteTokens(p, layer, rs);
    for (int u = blockIdx.x; u < rs.num_used * kGateUpBlocks; u += gridDim.x) {
      if (threadIdx.x == 0) PrefetchUnit(moe, rs, u + gridDim.x, true);
      const int e = rs.used[u / kGateUpBlocks];
      const int at = rs.offset[e];
      ForTiles(rs.count[e], [&](auto tiles) {
        GateUpUnit<decltype(tiles)::value>(p, moe, e, u % kGateUpBlocks,
                                           rs.tokens + at, rs.slots + at,
                                           rs.count[e], partials);
      });
    }
    sync();
    if (threadIdx.x == 0) PrefetchUnit(moe, rs, blockIdx.x, false);
    for (int u = blockIdx.x; u < rs.num_used * kDownBlocks; u += gridDim.x) {
      if (threadIdx.x == 0) PrefetchUnit(moe, rs, u + gridDim.x, false);
      const int e = rs.used[u / kDownBlocks];
      const int at = rs.offset[e];
      ForTiles(rs.count[e], [&](auto tiles) {
        DownUnit<decltype(tiles)::value>(p, moe, e, u % kDownBlocks,
                                         rs.slots + at, rs.count[e], partials);
      });
    }
    sync();
  }
  TokenNorm(p, p.num_layers - 1, p.final_norm, p.final_hidden,
            p.hidden_layer == p.num_layers ? p.hidden : nullptr, red);
}

}  // namespace

cudaError_t LaunchThinkerPrefill(const ThinkerPrefillParams& params,
                                 int num_ctas, cudaStream_t stream) {
  cudaError_t err =
      cudaMemsetAsync(params.sync, 0, kSyncWords * sizeof(unsigned), stream);
  if (err != cudaSuccess) return err;
  constexpr int kSmem = kSmemFloats * sizeof(float);
  err = cudaFuncSetAttribute(ThinkerPrefillKernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem);
  if (err != cudaSuccess) return err;
  ThinkerPrefillKernel<<<num_ctas, kThreads, kSmem, stream>>>(params);
  return cudaGetLastError();
}

}  // namespace s2mk
