// Copyright 2026 Fractalyze Inc. All rights reserved.

#ifndef S2MK_CSRC_LAYER_H_
#define S2MK_CSRC_LAYER_H_

#include <cuda_bf16.h>

#include "kv_cache.h"

namespace s2mk {

// A decoder's dimensions. decoder_layer.cuh is written against these, so one
// layer implementation serves every model whose heads are kHeadDim wide.
//
// kFixedOrderGemv sums each GEMV row's per-warp partials in warp order
// (GemvRowsFixedOrder) instead of adding them atomically (GemvRows). A
// decoder needs it when its rows are split across more than two warps, as a
// narrow decoder's are over the RTX 5090's 170 CTAs: atomic adds of three or
// more partials land in any order, so the result would change run to run.
//
// The last two name the checkpoint's layout, which the layer then reads in
// place. kRotateHalf: RoPE pairs dim i with i + kHeadDim / 2, where it
// otherwise pairs 2i with 2i + 1. kStackedGateUp: w13 holds every gate row,
// then every up row, where it otherwise interleaves them in pairs.
template <int kDim_, int kQHeads_, int kKvHeads_, int kFfn_,
          bool kFixedOrderGemv_ = false, bool kRotateHalf_ = false,
          bool kStackedGateUp_ = false>
struct DecoderDims {
  static constexpr int kDim = kDim_;
  static constexpr int kQHeads = kQHeads_;
  static constexpr int kKvHeads = kKvHeads_;
  static constexpr int kFfn = kFfn_;
  static constexpr bool kFixedOrderGemv = kFixedOrderGemv_;
  static constexpr bool kRotateHalf = kRotateHalf_;
  static constexpr bool kStackedGateUp = kStackedGateUp_;
  static constexpr int kQDim = kQHeads * kHeadDim;
  static constexpr int kQkvRows = (kQHeads + 2 * kKvHeads) * kHeadDim;
  static constexpr int kGqa = kQHeads / kKvHeads;
};

// S2 Pro's decoder, shared by the Slow AR and the Fast AR; s2mk/shapes.py
// holds the same numbers.
using S2ProDims = DecoderDims<2560, 32, 8, 9728>;

// S2 Pro's dimensions under their own names, for the S2 Pro kernels and ops.
constexpr int kDim = S2ProDims::kDim;
constexpr int kQHeads = S2ProDims::kQHeads;
constexpr int kKvHeads = S2ProDims::kKvHeads;
constexpr int kQDim = S2ProDims::kQDim;
constexpr int kQkvRows = S2ProDims::kQkvRows;
constexpr int kFfn = S2ProDims::kFfn;

// The most CTAs a launch may use; attention has at most one item per CTA.
constexpr int kMaxCtas = 256;
// The most GEMV rows one CTA may own in any phase.
constexpr int kMaxRowsPerCta = 256;

// Grid barriers per decoder layer; decoder_layer.cuh lists them. A layer
// with short attention has no barrier between attending and merging.
constexpr int kBarriersPerLayer = 5;
constexpr int kShortBarriersPerLayer = kBarriersPerLayer - 1;

// A profiled launch's row of int64 words per CTA starts with kProfileHeader
// words: %globaltimer at start and end, clock64 at start and end, then the
// CTA's %smid. The globaltimer pair converts the CTA's clock64 differences to
// nanoseconds.
constexpr int kProfileHeader = 5;

// One layer's weights, shaped by its DecoderDims D. The layout matches the
// int64 rows s2mk/slow_ar.py writes.
struct LayerWeights {
  const __nv_bfloat16* wqkv;  // [D::kQkvRows, D::kDim]: q heads, then k, then v
  const __nv_bfloat16* wo;  // [D::kDim, D::kQDim]
  const __nv_bfloat16* w13;  // [2 * D::kFfn, D::kDim]: gate and up rows
  const __nv_bfloat16* w2;  // [D::kDim, D::kFfn]
  const __nv_bfloat16* attention_norm;  // [D::kDim]
  const __nv_bfloat16* ffn_norm;  // [D::kDim]
  const __nv_bfloat16* q_norm;  // [kHeadDim], or null: no QK-norm
  const __nv_bfloat16* k_norm;  // [kHeadDim], or null with q_norm
};

}  // namespace s2mk

#endif  // S2MK_CSRC_LAYER_H_
