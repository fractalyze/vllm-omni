// Copyright 2026 Fractalyze Inc. All rights reserved.

#ifndef S2MK_CSRC_THINKER_DECODE_H_
#define S2MK_CSRC_THINKER_DECODE_H_

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "barrier.h"
#include "thinker_attention.h"
#include "thinker_moe.h"

namespace s2mk {

// One decode step of Qwen3-Omni's thinker in one launch: every layer's
// attention and MoE blocks (thinker_layer.cuh), the final norm and the bf16
// LM head. Six grid barriers a layer, one after each block phase.
//
// Each layer's blocks come as their own params, built once for the layer's
// weights, cache and workspace; the step supplies what changes per step and
// points every block's residual at `residual`. The step's token is read from
// device memory, as vLLM's runner keeps it, so a CUDA graph can replay the
// launch.
struct ThinkerDecodeParams {
  const ThinkerAttentionParams* attention;  // [num_layers], on the device
  const ThinkerMoeParams* moe;  // [num_layers], on the device
  int num_layers;
  // The token sits at position seq_len[0] − 1 and its keys and values go to
  // cache slot slot_mapping[0]; a negative slot, as in vLLM's padded capture
  // runs, writes nothing.
  const int32_t* seq_len;
  const int64_t* slot_mapping;
  // M-RoPE position of axis a (temporal, height, width) at
  // positions[a × positions_stride].
  const int64_t* positions;
  int64_t positions_stride;
  const __nv_bfloat16* final_norm;  // [kThinkerDim]
  // [vocab, kThinkerDim]; null skips the LM head.
  const __nv_bfloat16* lm_head;
  int vocab;
  // Nonzero starts each CTA's slice of a GEMV two phases ahead into L2
  // (PrefetchAhead in thinker_decode.cu); zero is for measuring its gain.
  int prefetch;
  float eps;
  int64_t timeout_ns;

  // The step's input embedding in; the last layer's output out.
  float* residual;  // [kThinkerDim]
  float* logits;  // [vocab]
  // [kThinkerDim]: the final norm's output, as the LM head reads it; null
  // skips it.
  __nv_bfloat16* final_hidden;
  // [num_layers + 1, kThinkerDim]: the residual entering each layer and the
  // last layer's output; null skips it.
  float* hidden;
  // [num_ctas, barriers, 2]: each CTA's globaltimer on arriving at and
  // leaving every grid barrier; null skips it.
  int64_t* profile;
  unsigned* sync;  // kSyncWords words
  ErrorRecord* error;  // device view of the host-mapped record
};

// Grid barriers a layer: one after each of its blocks' six phases.
constexpr int kThinkerBarriersPerLayer = 6;

// A plain launch of one CTA per SM (see LaunchCodePredictor on MPS).
cudaError_t LaunchThinkerDecode(const ThinkerDecodeParams& params, int num_ctas,
                                cudaStream_t stream);

}  // namespace s2mk

#endif  // S2MK_CSRC_THINKER_DECODE_H_
