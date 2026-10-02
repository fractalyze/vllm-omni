// Copyright 2026 Fractalyze Inc. All rights reserved.

#ifndef S2MK_CSRC_THINKER_DIMS_H_
#define S2MK_CSRC_THINKER_DIMS_H_

namespace s2mk {

// Qwen3-Omni's thinker: its residual width, shared by its attention and MoE
// blocks.
constexpr int kThinkerDim = 2048;

// The DecoderDims subset RmsNorm (decoder_layer.cuh) reads.
struct ThinkerNormDims {
  static constexpr int kDim = kThinkerDim;
};

}  // namespace s2mk

#endif  // S2MK_CSRC_THINKER_DIMS_H_
