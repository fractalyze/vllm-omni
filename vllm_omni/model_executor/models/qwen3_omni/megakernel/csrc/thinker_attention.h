// Copyright 2026 Fractalyze Inc. All rights reserved.

#ifndef S2MK_CSRC_THINKER_ATTENTION_H_
#define S2MK_CSRC_THINKER_ATTENTION_H_

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "barrier.h"
#include "thinker_dims.h"

namespace s2mk {

// Qwen3-Omni's thinker: its attention block's widths.
constexpr int kThinkerQHeads = 32;
constexpr int kThinkerKvHeads = 4;
constexpr int kThinkerHeadDim = 128;
constexpr int kThinkerQDim = kThinkerQHeads * kThinkerHeadDim;
// vLLM's fused qkv_proj: q rows, then k rows, then v rows.
constexpr int kThinkerQkvRows =
    kThinkerQDim + 2 * kThinkerKvHeads * kThinkerHeadDim;
// Interleaved M-RoPE: frequency i < 3 × 20 with i mod 3 = 1 turns at the
// height position, with i mod 3 = 2 at the width position, and every other
// frequency at the temporal position (vLLM's apply_interleaved_rope).
constexpr int kThinkerMropeInterleaved = 60;

// vLLM's paged KV cache for one layer, read and written in place. Position t
// of KV head g lives in block block_table[t / block_size], at slot
// t mod block_size; strides are in elements and a head's kThinkerHeadDim
// values are contiguous. Keys are stored after QK-norm and RoPE.
struct PagedKv {
  __nv_bfloat16* key;
  __nv_bfloat16* value;
  const int32_t* block_table;
  int64_t block_stride;
  int64_t slot_stride;
  int64_t head_stride;
  int block_size;
};

// One token through the thinker's attention block in one launch:
// residual = residual_in + o_proj(attention(q, k, v)) over h =
// RMSNorm(residual_in) × norm, with QK-norm and interleaved M-RoPE
// (rotate-half) on q and k. The token sits at cache position `pos` and
// attends to positions [0, pos]. qkv_proj and o_proj are compressed-tensors
// W4A16 (int4_gemv_core.cuh). Three phases and two grid barriers: the QKV
// GEMV, then (query head, chunk) items that attend over a chunk each, then
// every CTA merges the items and computes its O rows. No atomics, so the
// result is deterministic.
struct ThinkerAttentionParams {
  const float* residual_in;  // [kThinkerDim]
  const __nv_bfloat16* norm;  // [kThinkerDim]: input_layernorm
  const int32_t* wqkv_packed;  // [kThinkerQkvRows, kThinkerDim / 8]
  const __nv_bfloat16* wqkv_scales;  // [kThinkerQkvRows, kThinkerDim / 32]
  const __nv_bfloat16* q_norm;  // [kThinkerHeadDim]
  const __nv_bfloat16* k_norm;  // [kThinkerHeadDim]
  const int32_t* wo_packed;  // [kThinkerDim, kThinkerQDim / 8]
  const __nv_bfloat16* wo_scales;  // [kThinkerDim, kThinkerQDim / 32]
  // vLLM's cos_sin_cache: [positions, kThinkerHeadDim], each position's 64
  // cosines, then its 64 sines.
  const __nv_bfloat16* cos_sin;
  const int32_t* positions;  // [3]: temporal, height, width
  PagedKv kv;
  int pos;
  int splits;  // chunks per query head
  float eps;
  int64_t timeout_ns;

  // Workspace.
  float* qkv;  // [kThinkerQkvRows]
  float* partial_ml;  // [kThinkerQHeads × splits, 2]: running max and sum
  float* partial_o;  // [kThinkerQHeads × splits, kThinkerHeadDim]

  // Outputs.
  float* residual;  // [kThinkerDim]
  unsigned* sync;  // kSyncWords words
  ErrorRecord* error;  // device view of the host-mapped record
};

// A plain launch of one CTA per SM (see LaunchCodePredictor on MPS).
cudaError_t LaunchThinkerAttention(const ThinkerAttentionParams& params,
                                   int num_ctas, cudaStream_t stream);

}  // namespace s2mk

#endif  // S2MK_CSRC_THINKER_ATTENTION_H_
