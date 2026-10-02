// Copyright 2026 Fractalyze Inc. All rights reserved.

#ifndef S2MK_CSRC_QWEN3OMNI_CP_H_
#define S2MK_CSRC_QWEN3OMNI_CP_H_

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "barrier.h"
#include "kv_cache.h"
#include "layer.h"

namespace s2mk {

// Qwen3-Omni's code predictor: the dense decoder its talker runs once per
// audio frame to predict codebooks 1-15 from the talker's hidden state and
// codebook 0. Its rows are 1,024 wide, so over the RTX 5090's 170 CTAs a
// GEMV row is split across more than two warps and needs the fixed-order sum.
// The layer reads the checkpoint's layout: rotate-half RoPE and stacked gate
// and up rows.
using CpDims = DecoderDims<1024, 16, 8, 3072, /*kFixedOrderGemv_=*/true,
                           /*kRotateHalf_=*/true, /*kStackedGateUp_=*/true>;

constexpr int kCpCodeGroups = 16;
// Passes that end in a codebook head: codes 1..15.
constexpr int kCpHeads = kCpCodeGroups - 1;
constexpr int kCpVocab = 2048;
// Position 0 is the talker's hidden state and position p ≥ 1 the embedding of
// code p − 1; the last position that feeds a head is kCpHeads.
constexpr int kCpPositions = kCpHeads + 1;
// The most decoder layers a launch takes.
constexpr int kCpMaxLayers = 8;
// The largest top-k the code sampler takes.
constexpr int kCpMaxTopK = 50;

// The phases a profiled launch times: a layer's four, the KV-only last layer
// of position 0 (its QKV GEMV, then its key-value write) and the head.
enum class CpPhase { kQkv, kAttentionO, kGateUp, kDown, kKvOnlyQkv, kKvWrite,
                     kHead, kCount };
constexpr int kCpPhases = static_cast<int>(CpPhase::kCount);

// A profiled launch writes one row of int64 words per CTA: the kProfileHeader
// words, then the clock64 cycles thread 0 spent waiting at grid barriers and
// in the sampler, the barriers it passed, then for each CpPhase the cycles
// from leaving the previous barrier (or the sampler) to arriving at the
// phase's barrier, and the cycles waiting there.
constexpr int kCpProfileWords = kProfileHeader + 3 + 2 * kCpPhases;

// `num_frames` frames of the code predictor in one launch, each a pass over
// positions 0..15 of a 16-entry KV cache, the incremental equivalent of
// vLLM-Omni's re-prefill: position p ≥ 1 ends in head p − 1, whose logits
// the code sampler draws code p from with that pass's recorded uniforms.
// Frames are independent: frame f's inputs are the talker's.
struct CodePredictorParams {
  const LayerWeights* layers;  // [num_layers], on the device
  int num_layers;
  const __nv_bfloat16* final_norm;  // [kDim]
  const __nv_bfloat16* heads;  // [kCpHeads, kCpVocab, kDim]
  const __nv_bfloat16* embeddings;  // [kCpHeads, kCpVocab, kDim]
  // [kCpPositions, kHeadDim / 2, (cos, sin)]: frequency i turns dims i and
  // i + kHeadDim / 2.
  const __nv_bfloat16* rope;
  KvLayout kv;  // max_seq = kCpPositions
  float eps;
  int top_k;  // in [1, kCpMaxTopK]
  float top_p;

  int num_frames;
  const __nv_bfloat16* talker_hidden;  // [num_frames, kDim]
  const __nv_bfloat16* code0_embed;  // [num_frames, kDim]
  const float* uniforms;  // [num_frames, kCpHeads, kCpVocab]
  // [num_frames, kCpHeads]: the recorded codes 1..15 fed forward, or null to
  // feed the kernel's own.
  const int64_t* forced_codes;
  // Bytes of each CTA's next GEMV slice to prefetch into L2; a multiple of 16.
  int prefetch_bytes;
  int64_t timeout_ns;

  // Workspace.
  float* residual;  // [kDim]
  float* qkv;  // [kQkvRows]
  __nv_bfloat16* act;  // [kFfn]

  // Outputs.
  float* logits;  // [num_frames, kCpHeads, kCpVocab]
  int64_t* codes;  // [num_frames, kCpHeads]: the drawn codes
  // [num_frames, kCpPositions, num_layers + 1, kDim]: each position's decoder
  // input, then each layer's output; or null.
  float* dump;
  int64_t* profile;  // [num_ctas, kCpProfileWords] or null
  unsigned* sync;  // kSyncWords words
  ErrorRecord* error;  // device view of the host-mapped record
};

// Launches the code predictor on `num_ctas` CTAs, one per SM. Its grid
// barriers need every CTA resident at once. A cooperative launch has the
// driver guarantee that, which under CUDA MPS means waiting until no other
// client's work is on any SM: sharing the GPU with vLLM-Omni's other stages,
// that hung the stages. A plain launch places CTAs as SMs free up; one CTA
// per SM still makes them all resident, and a CTA that never is trips the
// barrier watchdog instead of hanging.
cudaError_t LaunchCodePredictor(const CodePredictorParams& params,
                                int num_ctas, bool cooperative,
                                cudaStream_t stream);

// Draws one code per case with the code predictor's sampler, one CTA per
// case: `logits` and `uniforms` are [cases, kCpVocab].
cudaError_t LaunchCodeSamplerProbe(const float* logits, const float* uniforms,
                                   int top_k, float top_p, int64_t* codes,
                                   int cases, cudaStream_t stream);

}  // namespace s2mk

#endif  // S2MK_CSRC_QWEN3OMNI_CP_H_
