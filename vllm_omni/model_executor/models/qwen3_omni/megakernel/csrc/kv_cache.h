// Copyright 2026 Fractalyze Inc. All rights reserved.

#ifndef S2MK_CSRC_KV_CACHE_H_
#define S2MK_CSRC_KV_CACHE_H_

#include <cuda_bf16.h>

#include <cstdint>

#ifdef __CUDACC__
#define S2MK_HOST_DEVICE __host__ __device__
#else
#define S2MK_HOST_DEVICE
#endif

namespace s2mk {

constexpr int kHeadDim = 128;

// Where the Slow AR's keys and values live. Every cache read and write goes
// through Key() and Value(), so changing the layout touches only this struct
// and the Python that fills it (s2mk/slow_ar.py).
//
// The layout is Fish's own cache: one [1, kv_heads, max_seq, kHeadDim] bf16
// tensor per layer for keys and one for values, keys stored after QK-norm and
// RoPE.
struct KvLayout {
  __nv_bfloat16* const* k;  // [num_layers] per-layer key bases, on the device
  __nv_bfloat16* const* v;  // [num_layers] per-layer value bases
  int64_t max_seq;

  S2MK_HOST_DEVICE __nv_bfloat16* Key(int layer, int kv_head, int pos) const {
    return k[layer] + Offset(kv_head, pos);
  }

  S2MK_HOST_DEVICE __nv_bfloat16* Value(int layer, int kv_head,
                                        int pos) const {
    return v[layer] + Offset(kv_head, pos);
  }

 private:
  S2MK_HOST_DEVICE int64_t Offset(int kv_head, int pos) const {
    return (kv_head * max_seq + pos) * kHeadDim;
  }
};

}  // namespace s2mk

#undef S2MK_HOST_DEVICE

#endif  // S2MK_CSRC_KV_CACHE_H_
