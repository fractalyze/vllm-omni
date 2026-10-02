// Copyright 2026 Fractalyze Inc. All rights reserved.
//
// One decode step of Qwen3-Omni's thinker (thinker_decode.h) as a persistent
// launch of one CTA per SM: every layer's attention and MoE blocks
// (thinker_layer.cuh) with a grid barrier after each phase, then the final
// norm and the LM head, whose rows each CTA splits into runs that fit its
// shared memory.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "barrier.cuh"
#include "decoder_layer.cuh"
#include "gemv_core.cuh"
#include "thinker_decode.h"
#include "thinker_layer.cuh"

namespace s2mk {
namespace {

// The LM head's runs: a CTA's rows in the fewest runs of at most
// kMaxRowsPerCta. A run of at least kWarps rows gives each warp a row's
// worth of chunks or more, so no row takes more than two warps' partials and
// GemvRows' sum is order-independent.
__device__ void LmHead(const ThinkerDecodeParams& p, thinker::Shared& sh) {
  float* ys = sh.attention.ys;
  const int begin = RowBegin(p.vocab, blockIdx.x);
  const int rows = RowBegin(p.vocab, blockIdx.x + 1) - begin;
  const int runs = (rows + kMaxRowsPerCta - 1) / kMaxRowsPerCta;
  for (int r = 0; r < runs; ++r) {
    const int run_begin = begin + r * rows / runs;
    const int run_rows = begin + (r + 1) * rows / runs - run_begin;
    GemvRows(p.lm_head, kThinkerDim, run_begin, run_rows, sh.attention.xs, ys);
    for (int i = threadIdx.x; i < run_rows; i += kThreads) {
      p.logits[run_begin + i] = ys[i];
    }
  }
}

// On arriving at barrier `index`, starts fetching into L2 this CTA's slice of
// a GEMV two phases on, so HBM streams it through the next phase and both
// barriers: the O rows during attention, the router during O, and the next
// layer's q, k and v rows (or the LM head's first rows) during the down
// projection. The experts' rows are named only by routing; the MoE block
// prefetches the down rows itself.
__device__ void PrefetchAhead(const ThinkerDecodeParams& p, int index) {
  if (threadIdx.x != 0) return;
  const int layer = index / kThinkerBarriersPerLayer;
  const int cta = blockIdx.x;
  switch (index % kThinkerBarriersPerLayer) {
    case 0: {  // after QKV
      const int begin = RowBegin(kThinkerDim, cta);
      thinker::PrefetchInt4Rows(p.attention[layer].wo_packed,
                                p.attention[layer].wo_scales, kThinkerQDim,
                                begin, RowBegin(kThinkerDim, cta + 1) - begin);
      break;
    }
    case 1:  // after attention
      PrefetchSliceL2(p.moe[layer].router, kThinkerDim, kThinkerExperts, 1,
                      kThinkerExperts * kThinkerDim * 2);
      break;
    case 4: {  // after gate-up
      if (layer + 1 < p.num_layers) {
        const int begin = RowBegin(kThinkerQkvRows, cta);
        thinker::PrefetchInt4Rows(
            p.attention[layer + 1].wqkv_packed,
            p.attention[layer + 1].wqkv_scales, kThinkerDim, begin,
            RowBegin(kThinkerQkvRows, cta + 1) - begin);
      } else if (p.lm_head != nullptr) {
        PrefetchSliceL2(p.lm_head, kThinkerDim, p.vocab, 1, 64 << 10);
      }
      break;
    }
    default:
      break;
  }
}

// Copies this CTA's rows of the residual into hidden row `row`.
__device__ void DumpHidden(const ThinkerDecodeParams& p, int row) {
  if (p.hidden == nullptr) return;
  const int begin = RowBegin(kThinkerDim, blockIdx.x);
  const int end = RowBegin(kThinkerDim, blockIdx.x + 1);
  for (int i = begin + threadIdx.x; i < end; i += kThreads) {
    p.hidden[int64_t{row} * kThinkerDim + i] = __ldcg(p.residual + i);
  }
}

__global__ void __launch_bounds__(kThreads, 1)
    ThinkerDecodeKernel(const __grid_constant__ ThinkerDecodeParams p) {
  __shared__ thinker::Shared sh;
  GridBarrier barrier(p.sync, p.error, p.timeout_ns, 0);
  int64_t* stamps =
      p.profile == nullptr
          ? nullptr
          : p.profile + int64_t{blockIdx.x} * 2 * kThinkerBarriersPerLayer *
                            p.num_layers;
  // The arrival stamp waits for the whole CTA to finish the phase.
  auto sync = [&] {
    if (p.prefetch) PrefetchAhead(p, barrier.index());
    if (stamps != nullptr) {
      __syncthreads();
      if (threadIdx.x == 0) stamps[2 * barrier.index()] = GlobalTimer();
    }
    barrier.Sync();
    if (stamps != nullptr && threadIdx.x == 0) {
      stamps[2 * (barrier.index() - 1) + 1] = GlobalTimer();
    }
  };

  const thinker::BlockStep step{
      p.seq_len[0] - 1,
      p.slot_mapping[0],
      {p.positions[0], p.positions[p.positions_stride],
       p.positions[2 * p.positions_stride]},
      p.residual,
      p.residual};
  for (int layer = 0; layer < p.num_layers; ++layer) {
    DumpHidden(p, layer);
    thinker::AttentionBlock(p.attention[layer], step, sync, sh.attention);
    sync();
    thinker::MoeBlock(p.moe[layer], step, sync, sh.moe);
    sync();
  }
  DumpHidden(p, p.num_layers);

  RmsNorm<ThinkerNormDims>(p.residual, nullptr, p.final_norm, p.eps,
                           reinterpret_cast<__nv_bfloat16*>(sh.attention.xs),
                           blockIdx.x == 0 ? p.final_hidden : nullptr,
                           sh.attention.red);
  if (p.lm_head != nullptr) LmHead(p, sh);
}

}  // namespace

cudaError_t LaunchThinkerDecode(const ThinkerDecodeParams& params, int num_ctas,
                                cudaStream_t stream) {
  cudaError_t err =
      cudaMemsetAsync(params.sync, 0, kSyncWords * sizeof(unsigned), stream);
  if (err != cudaSuccess) return err;
  ThinkerDecodeKernel<<<num_ctas, kThreads, 0, stream>>>(params);
  return cudaGetLastError();
}

}  // namespace s2mk
