// Copyright 2026 Fractalyze Inc. All rights reserved.
//
// The thinker's attention block for one token (thinker_attention.h) as its own
// persistent launch of one CTA per SM; thinker_layer.cuh has its phases.

#include <cuda_runtime.h>

#include "barrier.cuh"
#include "thinker_attention.h"
#include "thinker_layer.cuh"

namespace s2mk {
namespace {

__global__ void __launch_bounds__(kThreads, 1)
    ThinkerAttentionKernel(const __grid_constant__ ThinkerAttentionParams p) {
  __shared__ thinker::AttentionShared sh;
  GridBarrier barrier(p.sync, p.error, p.timeout_ns, 0);
  auto sync = [&] { barrier.Sync(); };
  const thinker::BlockStep step{
      p.pos,
      thinker::SlotOf(p.kv, p.pos),
      {p.positions[0], p.positions[1], p.positions[2]},
      p.residual_in,
      p.residual};
  thinker::AttentionBlock(p, step, sync, sh);
}

}  // namespace

cudaError_t LaunchThinkerAttention(const ThinkerAttentionParams& params,
                                   int num_ctas, cudaStream_t stream) {
  cudaError_t err =
      cudaMemsetAsync(params.sync, 0, kSyncWords * sizeof(unsigned), stream);
  if (err != cudaSuccess) return err;
  ThinkerAttentionKernel<<<num_ctas, kThreads, 0, stream>>>(params);
  return cudaGetLastError();
}

}  // namespace s2mk
