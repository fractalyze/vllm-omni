// Copyright 2026 Fractalyze Inc. All rights reserved.

#ifndef S2MK_CSRC_THINKER_MOE_H_
#define S2MK_CSRC_THINKER_MOE_H_

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "barrier.h"
#include "thinker_dims.h"

namespace s2mk {

// Qwen3-Omni's thinker: its MoE block's widths.
constexpr int kThinkerExperts = 128;
constexpr int kThinkerTopK = 8;
constexpr int kThinkerExpertFfn = 768;

// One token through the thinker's MoE block in one launch:
// residual = residual_in + Σ_k weight_k × down_{e_k}(SiLU(gate_{e_k}(h)) ×
// up_{e_k}(h)), h = RMSNorm(residual_in) × norm, over the top 8 of the 128
// router logits, renormalized. The experts are compressed-tensors W4A16
// (int4_gemv_core.cuh); norm and router are bf16. Three phases and two grid
// barriers: the router, then each expert's gate and up rows, then each
// output row's 8 experts summed in rank order, so the result is
// deterministic.
struct ThinkerMoeParams {
  const float* residual_in;  // [kThinkerDim]
  const __nv_bfloat16* norm;  // [kThinkerDim]
  const __nv_bfloat16* router;  // [kThinkerExperts, kThinkerDim]
  // [kThinkerExperts, 2 × kThinkerExpertFfn, kThinkerDim / 8]: gate rows,
  // then up rows; and their scales, [..., kThinkerDim / 32].
  const int32_t* w13_packed;
  const __nv_bfloat16* w13_scales;
  // [kThinkerExperts, kThinkerDim, kThinkerExpertFfn / 8] and its scales.
  const int32_t* w2_packed;
  const __nv_bfloat16* w2_scales;
  float eps;
  int64_t timeout_ns;

  // Workspace.
  float* router_logits;  // [kThinkerExperts]
  __nv_bfloat16* act;  // [kThinkerTopK, kThinkerExpertFfn]

  // Outputs.
  float* residual;  // [kThinkerDim]
  int32_t* experts;  // [kThinkerTopK]: the chosen experts, in rank order
  float* weights;  // [kThinkerTopK]: their renormalized weights
  unsigned* sync;  // kSyncWords words
  ErrorRecord* error;  // device view of the host-mapped record
};

// A plain launch of one CTA per SM (see LaunchCodePredictor on MPS).
cudaError_t LaunchThinkerMoe(const ThinkerMoeParams& params, int num_ctas,
                             cudaStream_t stream);

}  // namespace s2mk

#endif  // S2MK_CSRC_THINKER_MOE_H_
