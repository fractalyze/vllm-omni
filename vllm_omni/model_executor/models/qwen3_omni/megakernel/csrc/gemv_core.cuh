// Copyright 2026 Fractalyze Inc. All rights reserved.
//
// The batch-1 GEMV loop every s2mk kernel streams its weights through: one
// CTA computes y = W x for a contiguous run of W's rows, with x already in
// shared memory.
//
// Work split, for a CTA owning `rows` rows of k columns:
//
//   - The rows are one contiguous slice of the row-major weight, cut into
//     512-byte chunks (32 lanes × 16 bytes). Each warp streams an equal,
//     contiguous run of chunks. A run may start or end mid-row, so a warp
//     flushes its partial dot product into a shared-memory row accumulator
//     whenever it leaves a row.
//
// A row's accumulator receives a third partial only when two warp-run
// boundaries fall inside that one row, i.e. when runs are at least two chunks
// shorter than a row. No S2 Pro shape does that at the RTX 5090's 170 CTAs,
// and two fp32 adds onto zero commute, so there the result is run-to-run
// deterministic. Another CTA count shortens the runs and can break that.

#ifndef S2MK_CSRC_GEMV_CORE_CUH_
#define S2MK_CSRC_GEMV_CORE_CUH_

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace s2mk {

constexpr int kThreads = 512;
constexpr int kWarps = kThreads / 32;
// bf16 elements per 16-byte vector and per warp-wide chunk.
constexpr int kVecElems = 8;
constexpr int kChunkElems = 32 * kVecElems;
// 16-byte loads each lane issues before consuming any of them: the kernel's
// memory-level-parallelism knob.
constexpr int kUnroll = 8;

// Streaming load: the weights are read once per step, so keep them out of L1.
__device__ __forceinline__ uint4 LoadStream(const uint4* ptr) {
  uint4 v;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(ptr));
  return v;
}

__device__ __forceinline__ float Dot8(uint4 w, uint4 x) {
  const __nv_bfloat162* w2 = reinterpret_cast<const __nv_bfloat162*>(&w);
  const __nv_bfloat162* x2 = reinterpret_cast<const __nv_bfloat162*>(&x);
  float acc = 0.f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    float2 wf = __bfloat1622float2(w2[i]);
    float2 xf = __bfloat1622float2(x2[i]);
    acc = fmaf(wf.x, xf.x, acc);
    acc = fmaf(wf.y, xf.y, acc);
  }
  return acc;
}

__device__ __forceinline__ float WarpSum(float v) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    v += __shfl_xor_sync(0xffffffffu, v, offset);
  }
  return v;
}

// Sets ys[i] = W[row_begin + i] · x in fp32 for i in [0, rows), where W is a
// row-major bf16 matrix of k columns and xs holds x in shared memory.
//
// Called by every thread of the CTA. The caller writes xs before the call and
// must not still be reading ys from a previous call; the call's barriers make
// those writes visible and leave ys complete on return.
//
// Kept out of line: inlined into the Slow AR megakernel's five call sites, it
// competes with their live state for registers and spills.
inline __device__ __noinline__ void GemvRows(const __nv_bfloat16* w, int k,
                                         int row_begin, int rows,
                                         const uint4* xs, float* ys) {
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int vecs_per_row = k / kVecElems;
  const int chunks_per_row = k / kChunkElems;

  for (int i = threadIdx.x; i < rows; i += kThreads) ys[i] = 0.f;
  __syncthreads();

  const int chunks = rows * chunks_per_row;
  const int chunk_begin = warp * chunks / kWarps;
  const int chunk_end = (warp + 1) * chunks / kWarps;
  // Chunk c of this CTA starts at vector c * 32 of its slice.
  const uint4* wv_base = reinterpret_cast<const uint4*>(w) +
                         int64_t{row_begin} * vecs_per_row + lane;

  float acc = 0.f;
  for (int base = chunk_begin; base < chunk_end; base += kUnroll) {
    // Issue every load of the batch before the first use.
    uint4 wv[kUnroll];
#pragma unroll
    for (int u = 0; u < kUnroll; ++u) {
      if (base + u < chunk_end) wv[u] = LoadStream(wv_base + (base + u) * 32);
    }
#pragma unroll
    for (int u = 0; u < kUnroll; ++u) {
      const int c = base + u;
      if (c >= chunk_end) break;
      const int row = c / chunks_per_row;
      const int col = c - row * chunks_per_row;
      acc += Dot8(wv[u], xs[col * 32 + lane]);
      if (col == chunks_per_row - 1 || c == chunk_end - 1) {
        const float sum = WarpSum(acc);
        if (lane == 0) atomicAdd(&ys[row], sum);
        acc = 0.f;
      }
    }
  }
  __syncthreads();
}

// GemvRows with a result that does not depend on how warp runs fall across
// rows: a warp's run visits each row in one contiguous stretch, so it writes
// that row's partial once into `partials` ([kWarps × rows], shared memory),
// and each row sums its warps' partials in warp order. It costs the partials'
// shared memory and one pass over them; GemvRows stays the faster choice
// wherever a row takes at most two partials.
//
// The rows may come in two runs: the first `split` from row_begin, the rest
// `skip` rows further on, as a stacked gate and up weight holds a CTA's
// share of each. ys keeps them in that order.
inline __device__ __noinline__ void GemvRowsFixedOrder(
    const __nv_bfloat16* w, int k, int row_begin, int rows, const uint4* xs,
    float* ys, float* partials, int split, int skip) {
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int vecs_per_row = k / kVecElems;
  const int chunks_per_row = k / kChunkElems;

  for (int i = threadIdx.x; i < kWarps * rows; i += kThreads) partials[i] = 0.f;
  __syncthreads();

  const int chunks = rows * chunks_per_row;
  const int chunk_begin = warp * chunks / kWarps;
  const int chunk_end = (warp + 1) * chunks / kWarps;
  const int split_chunk = split * chunks_per_row;
  const uint4* wv_base = reinterpret_cast<const uint4*>(w) +
                         int64_t{row_begin} * vecs_per_row + lane;
  const int64_t skip_vecs = int64_t{skip} * vecs_per_row;

  float acc = 0.f;
  for (int base = chunk_begin; base < chunk_end; base += kUnroll) {
    uint4 wv[kUnroll];
#pragma unroll
    for (int u = 0; u < kUnroll; ++u) {
      const int c = base + u;
      if (c < chunk_end) {
        const int64_t jump = c >= split_chunk ? skip_vecs : 0;
        wv[u] = LoadStream(wv_base + c * 32 + jump);
      }
    }
#pragma unroll
    for (int u = 0; u < kUnroll; ++u) {
      const int c = base + u;
      if (c >= chunk_end) break;
      const int row = c / chunks_per_row;
      const int col = c - row * chunks_per_row;
      acc += Dot8(wv[u], xs[col * 32 + lane]);
      if (col == chunks_per_row - 1 || c == chunk_end - 1) {
        const float sum = WarpSum(acc);
        if (lane == 0) partials[warp * rows + row] = sum;
        acc = 0.f;
      }
    }
  }
  __syncthreads();
  for (int i = threadIdx.x; i < rows; i += kThreads) {
    float sum = 0.f;
    for (int w_ = 0; w_ < kWarps; ++w_) sum += partials[w_ * rows + i];
    ys[i] = sum;
  }
  __syncthreads();
}

// CTA c's share [RowBegin(c), RowBegin(c + 1)) of n rows split over the grid.
__device__ __forceinline__ int RowBegin(int n, int cta) {
  return static_cast<int>(int64_t{cta} * n / gridDim.x);
}

// Starts fetching the first `max_bytes` of this CTA's slice of a GEMV from HBM
// into L2, ahead of the GemvRows call that reads it. The GEMV has n units of
// `rows_per_unit` rows of a row-major bf16 matrix of k columns, split over the
// grid as RowBegin splits them. Nothing waits for the fetch: a load that finds
// the bytes in L2 hits there, and one that runs ahead of it goes to HBM.
//
// Called by every thread; thread 0 issues it. `max_bytes` is a multiple of 16.
// Kept out of line, like GemvRows: inlined, the slice arithmetic makes the
// Slow AR megakernel spill.
inline __device__ __noinline__ void PrefetchSliceL2(const __nv_bfloat16* w,
                                                  int k, int n,
                                                  int rows_per_unit,
                                                  int max_bytes) {
  if (threadIdx.x != 0 || max_bytes == 0) return;
  const int64_t row_bytes = int64_t{k} * sizeof(__nv_bfloat16);
  const int begin = rows_per_unit * RowBegin(n, blockIdx.x);
  const int end = rows_per_unit * RowBegin(n, blockIdx.x + 1);
  const int64_t bytes = min(int64_t{end - begin} * row_bytes,
                            int64_t{max_bytes});
  asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;"
               ::"l"(w + int64_t{begin} * k),
                 "r"(static_cast<uint32_t>(bytes))
               : "memory");
}

}  // namespace s2mk

#endif  // S2MK_CSRC_GEMV_CORE_CUH_
