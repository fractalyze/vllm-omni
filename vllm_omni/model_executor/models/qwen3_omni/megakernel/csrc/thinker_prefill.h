// Copyright 2026 Fractalyze Inc. All rights reserved.

#ifndef S2MK_CSRC_THINKER_PREFILL_H_
#define S2MK_CSRC_THINKER_PREFILL_H_

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "barrier.h"
#include "thinker_attention.h"
#include "thinker_moe.h"

namespace s2mk {

// Tokens a prefill launch takes at most: 8 n-tiles of mma.m16n8k16.
constexpr int kThinkerPrefillMaxTokens = 64;
// Grid barriers a prefill layer takes, one after each of its phases.
constexpr int kThinkerPrefillBarriersPerLayer = 9;

// One prefill chunk of Qwen3-Omni's thinker in one launch: `tokens` prompt
// tokens through every layer and the final norm, on tensor cores
// (int4_mma_core.cuh). The chunk's tokens sit at cache positions
// seq_len[0] − tokens on, attend causally to the cache before them and to
// each other, and write their keys and values to slot_mapping.
//
// Each layer's weights, cache and norms come from the decode's block params
// (thinker_attention.h, thinker_moe.h); their workspace fields are unused.
// Everything per step is read from device memory, as vLLM keeps it.
struct ThinkerPrefillParams {
  const ThinkerAttentionParams* attention;  // [num_layers], on the device
  const ThinkerMoeParams* moe;  // [num_layers], on the device
  int num_layers;
  int tokens;  // ≤ kThinkerPrefillMaxTokens
  const int32_t* seq_len;  // [1]: the cache's length after this chunk
  const int64_t* slot_mapping;  // [tokens]; negative writes nothing
  // M-RoPE position of token i on axis a at positions[a × stride + i].
  const int64_t* positions;
  int64_t positions_stride;
  const __nv_bfloat16* final_norm;  // [kThinkerDim]
  float eps;
  int64_t timeout_ns;

  // The chunk's input embeddings in, as fp32; each layer's residual after.
  float* residual;  // [tokens, kThinkerDim]
  // Workspace.
  __nv_bfloat16* h;  // [tokens, kThinkerDim]: a norm's output
  float* qkv;  // [tokens, kThinkerQkvRows]
  __nv_bfloat16* attn;  // [tokens, kThinkerQDim]
  // [tokens, kThinkerKvHeads, 2, kThinkerHeadDim]: the chunk's keys after
  // QK-norm and M-RoPE, then its values, as the cache holds them.
  __nv_bfloat16* kv;
  float* router_logits;  // [tokens, kThinkerExperts]
  __nv_bfloat16* act;  // [tokens × kThinkerTopK, kThinkerExpertFfn]
  float* partial;  // [tokens, kThinkerTopK, kThinkerDim]: down rows per rank

  // Outputs.
  // [num_layers, tokens, kThinkerTopK]: each layer's experts and weights.
  int32_t* experts;
  float* weights;
  __nv_bfloat16* final_hidden;  // [tokens, kThinkerDim]: the final norm's output
  // [tokens, kThinkerDim]: the residual entering layer hidden_layer (the
  // talker's accept layer); null skips it.
  float* hidden;
  int hidden_layer;
  // [num_ctas, barriers, 2]: each CTA's globaltimer on arriving at and
  // leaving every grid barrier; null skips it.
  int64_t* profile;
  unsigned* sync;  // kSyncWords words
  ErrorRecord* error;  // device view of the host-mapped record
};

// A plain launch of `num_ctas` CTAs, one per SM at most.
cudaError_t LaunchThinkerPrefill(const ThinkerPrefillParams& params,
                                 int num_ctas, cudaStream_t stream);

}  // namespace s2mk

#endif  // S2MK_CSRC_THINKER_PREFILL_H_
