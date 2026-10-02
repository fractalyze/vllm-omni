// Copyright 2026 Fractalyze Inc. All rights reserved.
//
// The int4 skinny-GEMV loop for Qwen3-Omni's thinker: one CTA computes
// ys[m][i] = W[row_begin + i] · xs[m] for kM activation rows, over a run of
// W's rows, where W is compressed-tensors W4A16 as vLLM loads it:
//
//   - packed: int32 [n, k / 8], value j of a row in bits 4 (j % 8) of word
//     j / 8, so a row's k values are contiguous;
//   - scales: bf16 [n, k / 32], symmetric: w = (q − 8) × scale for the group
//     of 32 values a scale covers.
//
// A 16-byte vector is exactly one group, so each vector costs one scale.
// kLanes lanes share a row, each taking kGroupsPerLane of its groups, and a
// warp covers 32 / kLanes rows at once. A row's partial sums meet by a
// butterfly over its lanes in a fixed order, so the result does not change
// from run to run.

#ifndef S2MK_CSRC_INT4_GEMV_CORE_CUH_
#define S2MK_CSRC_INT4_GEMV_CORE_CUH_

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "gemv_core.cuh"

namespace s2mk {

// Values a scale covers, and a 16-byte vector holds.
constexpr int kInt4Group = 32;
// Row blocks a warp loads before consuming any. Sixteen warps already keep
// enough loads in flight: at 4, the thinker decode step spills, and a spill
// store waits for the load it saves, serializing them.
constexpr int kInt4RowUnroll = 1;

// q − 8 for the eight 4-bit values of `word`, exactly: a float with exponent
// 2^23 holds q in its low mantissa bits.
__device__ __forceinline__ void Int4Values(uint32_t word, float (&out)[8]) {
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    out[j] = __uint_as_float(0x4B000000u | ((word >> (4 * j)) & 0xFu)) -
             8388616.f;
  }
}

// Σ over the group's 32 values of (q − 8) × x, for x at `x` (32 bf16 in
// shared memory).
__device__ __forceinline__ float Int4GroupDot(uint4 w, const uint4* x) {
  const uint32_t words[4] = {w.x, w.y, w.z, w.w};
  float acc = 0.f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    float q[8];
    Int4Values(words[i], q);
    const __nv_bfloat162* x2 = reinterpret_cast<const __nv_bfloat162*>(&x[i]);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float2 xf = __bfloat1622float2(x2[j]);
      acc = fmaf(q[2 * j], xf.x, acc);
      acc = fmaf(q[2 * j + 1], xf.y, acc);
    }
  }
  return acc;
}

// ys[m × rows + i] = W[row_begin + i] · xs[m] in fp32 for i in [0, rows) and
// m in [0, kM), k = 32 × kLanes × kGroupsPerLane. `packed` and `scales` are
// the whole matrix; `xs` is kM rows of k bf16 in shared memory, as uint4.
// Called by every thread of the CTA. The caller writes xs before the call
// and must not still be reading ys; the call's barriers make those writes
// visible and leave ys complete on return.
template <int kM, int kLanes, int kGroupsPerLane>
__device__ __noinline__ void Int4GemvRows(const uint4* packed,
                                          const __nv_bfloat16* scales,
                                          int row_begin, int rows,
                                          const uint4* xs, float* ys) {
  static_assert(32 % kLanes == 0);
  constexpr int kRowsPerStep = 32 / kLanes;
  constexpr int kGroups = kLanes * kGroupsPerLane;
  constexpr int kVecsPerXRow = kGroups * (kInt4Group / kVecElems);
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int sub = lane % kLanes;
  const int slot = lane / kLanes;
  __syncthreads();

  // Warp w takes rows [w × rows / kWarps, (w + 1) × rows / kWarps).
  const int warp_begin = warp * rows / kWarps;
  const int warp_end = (warp + 1) * rows / kWarps;
  for (int base = warp_begin; base < warp_end;
       base += kRowsPerStep * kInt4RowUnroll) {
    uint4 w[kInt4RowUnroll][kGroupsPerLane];
    __nv_bfloat16 s[kInt4RowUnroll][kGroupsPerLane];
#pragma unroll
    for (int u = 0; u < kInt4RowUnroll; ++u) {
      const int row = base + u * kRowsPerStep + slot;
      if (row < warp_end) {
        const int64_t r = row_begin + row;
#pragma unroll
        for (int g = 0; g < kGroupsPerLane; ++g) {
          const int group = sub + g * kLanes;
          w[u][g] = LoadStream(packed + r * kGroups + group);
          s[u][g] = scales[r * kGroups + group];
        }
      }
    }
#pragma unroll
    for (int u = 0; u < kInt4RowUnroll; ++u) {
      const int row = base + u * kRowsPerStep + slot;
      float acc[kM];
#pragma unroll
      for (int m = 0; m < kM; ++m) acc[m] = 0.f;
      if (row < warp_end) {
#pragma unroll
        for (int g = 0; g < kGroupsPerLane; ++g) {
          const int x_vec = (sub + g * kLanes) * (kInt4Group / kVecElems);
#pragma unroll
          for (int m = 0; m < kM; ++m) {
            acc[m] = fmaf(__bfloat162float(s[u][g]),
                          Int4GroupDot(w[u][g], xs + m * kVecsPerXRow + x_vec),
                          acc[m]);
          }
        }
      }
      // Every lane joins the butterfly, rows past the warp's end with zeros.
#pragma unroll
      for (int m = 0; m < kM; ++m) {
#pragma unroll
        for (int offset = kLanes / 2; offset > 0; offset >>= 1) {
          acc[m] += __shfl_xor_sync(0xffffffffu, acc[m], offset);
        }
        if (sub == 0 && row < warp_end) ys[m * rows + row] = acc[m];
      }
    }
  }
  __syncthreads();
}

// One row of an int4 GEMV whose rows come from different matrices: its
// packed words, its scales and the activations it multiplies.
struct Int4Row {
  const uint4* packed;  // the row's k / 32 vectors
  const __nv_bfloat16* scales;  // the row's k / 32 scales
  const uint4* x;  // k bf16 in shared memory
};

// ys[i] = map(i).packed · map(i).x for i in [0, rows), with a single
// activation row per output row: Int4GemvRows for rows that each name their
// own matrix and input, as an MoE's rows across its experts do. `map` is
// called as map(i) and returns an Int4Row. Called by every thread of the
// CTA, with the same barriers as Int4GemvRows.
template <int kLanes, int kGroupsPerLane, typename Map>
__device__ __forceinline__ void Int4GemvMappedRows(const Map& map, int rows,
                                                   float* ys) {
  static_assert(32 % kLanes == 0);
  constexpr int kRowsPerStep = 32 / kLanes;
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int sub = lane % kLanes;
  const int slot = lane / kLanes;
  __syncthreads();

  const int warp_begin = warp * rows / kWarps;
  const int warp_end = (warp + 1) * rows / kWarps;
  for (int base = warp_begin; base < warp_end;
       base += kRowsPerStep * kInt4RowUnroll) {
    uint4 w[kInt4RowUnroll][kGroupsPerLane];
    __nv_bfloat16 s[kInt4RowUnroll][kGroupsPerLane];
    const uint4* x[kInt4RowUnroll];
#pragma unroll
    for (int u = 0; u < kInt4RowUnroll; ++u) {
      const int row = base + u * kRowsPerStep + slot;
      x[u] = nullptr;
      if (row < warp_end) {
        const Int4Row r = map(row);
        x[u] = r.x;
#pragma unroll
        for (int g = 0; g < kGroupsPerLane; ++g) {
          const int group = sub + g * kLanes;
          w[u][g] = LoadStream(r.packed + group);
          s[u][g] = r.scales[group];
        }
      }
    }
#pragma unroll
    for (int u = 0; u < kInt4RowUnroll; ++u) {
      const int row = base + u * kRowsPerStep + slot;
      float acc = 0.f;
      if (row < warp_end) {
#pragma unroll
        for (int g = 0; g < kGroupsPerLane; ++g) {
          const int x_vec = (sub + g * kLanes) * (kInt4Group / kVecElems);
          acc = fmaf(__bfloat162float(s[u][g]),
                     Int4GroupDot(w[u][g], x[u] + x_vec), acc);
        }
      }
#pragma unroll
      for (int offset = kLanes / 2; offset > 0; offset >>= 1) {
        acc += __shfl_xor_sync(0xffffffffu, acc, offset);
      }
      if (sub == 0 && row < warp_end) ys[row] = acc;
    }
  }
  __syncthreads();
}

}  // namespace s2mk

#endif  // S2MK_CSRC_INT4_GEMV_CORE_CUH_
