// Copyright 2026 Fractalyze Inc. All rights reserved.
//
// The thinker's MoE block for one token (thinker_moe.h) as its own persistent
// launch of one CTA per SM; thinker_layer.cuh has its phases.

#include <cuda_runtime.h>

#include "barrier.cuh"
#include "thinker_layer.cuh"
#include "thinker_moe.h"

namespace s2mk {
namespace {

__global__ void __launch_bounds__(kThreads, 1)
    ThinkerMoeKernel(const __grid_constant__ ThinkerMoeParams p) {
  __shared__ thinker::MoeShared sh;
  GridBarrier barrier(p.sync, p.error, p.timeout_ns, 0);
  auto sync = [&] { barrier.Sync(); };
  const thinker::BlockStep step{0, -1, {0, 0, 0}, p.residual_in, p.residual};
  thinker::MoeBlock(p, step, sync, sh);
}

}  // namespace

cudaError_t LaunchThinkerMoe(const ThinkerMoeParams& params, int num_ctas,
                             cudaStream_t stream) {
  cudaError_t err =
      cudaMemsetAsync(params.sync, 0, kSyncWords * sizeof(unsigned), stream);
  if (err != cudaSuccess) return err;
  ThinkerMoeKernel<<<num_ctas, kThreads, 0, stream>>>(params);
  return cudaGetLastError();
}

}  // namespace s2mk
