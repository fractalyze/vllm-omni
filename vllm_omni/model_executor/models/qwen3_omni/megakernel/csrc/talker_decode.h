// Copyright 2026 Fractalyze Inc. All rights reserved.
//
// One decode step of Qwen3-Omni's talker, the codec-token model vLLM serves
// as Qwen3MoeForCausalLM: 20 layers of 1024 dims, attention of 16 query and 2
// KV heads, and an MoE of 128 bf16 experts, top 6, beside a shared expert its
// own sigmoid gate scales. The codec head and sampling stay with the caller.

#ifndef S2MK_CSRC_TALKER_DECODE_H_
#define S2MK_CSRC_TALKER_DECODE_H_

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "barrier.h"
#include "thinker_attention.h"

namespace s2mk {

constexpr int kTalkerDim = 1024;
constexpr int kTalkerQHeads = 16;
constexpr int kTalkerKvHeads = 2;
constexpr int kTalkerHeadDim = 128;
constexpr int kTalkerQDim = kTalkerQHeads * kTalkerHeadDim;
constexpr int kTalkerQkvRows = kTalkerQDim + 2 * kTalkerKvHeads * kTalkerHeadDim;
constexpr int kTalkerExperts = 128;
constexpr int kTalkerTopK = 6;
constexpr int kTalkerExpertFfn = 384;
constexpr int kTalkerSharedFfn = 768;
constexpr int kTalkerBarriersPerLayer = 6;

struct TalkerNormDims {
  static constexpr int kDim = kTalkerDim;
};

// One layer's weights, bf16 and row-major as vLLM holds them, and its cache.
struct TalkerLayerParams {
  const __nv_bfloat16* norm;  // [kTalkerDim]: input_layernorm
  const __nv_bfloat16* wqkv;  // [kTalkerQkvRows, kTalkerDim]: q, k, v rows
  const __nv_bfloat16* q_norm;  // [kTalkerHeadDim]
  const __nv_bfloat16* k_norm;  // [kTalkerHeadDim]
  const __nv_bfloat16* wo;  // [kTalkerDim, kTalkerQDim]
  PagedKv kv;
  const __nv_bfloat16* moe_norm;  // [kTalkerDim]: post_attention_layernorm
  const __nv_bfloat16* router;  // [kTalkerExperts, kTalkerDim]
  // [kTalkerExperts, 2 × kTalkerExpertFfn, kTalkerDim]: each expert's gate
  // rows, then its up rows.
  const __nv_bfloat16* w13;
  const __nv_bfloat16* w2;  // [kTalkerExperts, kTalkerDim, kTalkerExpertFfn]
  // [2 × kTalkerSharedFfn, kTalkerDim]: the shared expert's gate rows, then
  // its up rows.
  const __nv_bfloat16* shared_w13;
  const __nv_bfloat16* shared_w2;  // [kTalkerDim, kTalkerSharedFfn]
  const __nv_bfloat16* shared_gate;  // [kTalkerDim]
};

struct TalkerDecodeParams {
  const TalkerLayerParams* layers;  // [num_layers], on the device
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
  // vLLM's cos_sin_cache: [positions, kTalkerHeadDim], each position's 64
  // cosines, then its 64 sines.
  const __nv_bfloat16* cos_sin;
  const __nv_bfloat16* final_norm;  // [kTalkerDim]
  int splits;  // attention chunks per query head
  float eps;
  int64_t timeout_ns;

  // The step's input embedding in, the last layer's output out.
  float* residual;  // [kTalkerDim]
  __nv_bfloat16* final_hidden;  // [kTalkerDim]: the final norm's output

  // Workspace.
  float* qkv;  // [kTalkerQkvRows]
  float* partial_ml;  // [kTalkerQHeads × splits, 2]
  float* partial_o;  // [kTalkerQHeads × splits, kTalkerHeadDim]
  float* router_logits;  // [kTalkerExperts + 1]: then the shared gate's
  __nv_bfloat16* act;  // [kTalkerTopK × kTalkerExpertFfn + kTalkerSharedFfn]

  // [num_layers + 1, kTalkerDim]: the residual entering each layer and the
  // last layer's output; null skips it.
  float* hidden;
  // [num_ctas, num_layers × kTalkerBarriersPerLayer, 2]: each CTA's
  // globaltimer on arriving at and leaving every grid barrier; null skips it.
  int64_t* profile;
  unsigned* sync;  // kSyncWords words
  ErrorRecord* error;  // device view of the host-mapped record
};

cudaError_t LaunchTalkerDecode(const TalkerDecodeParams& params, int num_ctas,
                               cudaStream_t stream);

}  // namespace s2mk

#endif  // S2MK_CSRC_TALKER_DECODE_H_
