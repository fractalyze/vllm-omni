// Copyright 2026 Fractalyze Inc. All rights reserved.
//
// Qwen3-Omni's code-predictor sampler on one CTA, as vLLM-Omni's
// CodePredictorWrapper runs it in "stored" mode
// (vllm_omni/model_executor/models/common/qwen3_code_predictor.py): the
// logits rounded to bf16, top-k, top-p over the softmax of the kept logits,
// then a Gumbel-max draw with production's uniforms, argmax of
// logit − log(−log u) over the codebook. There is no temperature in this mode.
//
// The code predictor runs it redundantly on every CTA: the logits and the
// uniforms are the same everywhere, so every CTA draws the same code without
// a barrier or a broadcast.
//
// Top-k keeps exactly k logits, breaking exact ties by the lower codebook
// index, where production keeps every logit equal to the k-th. The budget rule
// excuses a code only a tie decides.
//
// Top-k is a radix select: a logit rounded to bf16 has 16 order bits, so with
// its 11-bit index below them every key is distinct, and three 9-bit passes of
// a 512-bin histogram (one bin per thread) find the k-th largest exactly. The
// k keys at or above it are ranked by pairwise comparison, so their order,
// and every sum over them, is the same on every CTA.

#ifndef S2MK_CSRC_CODE_SAMPLER_CUH_
#define S2MK_CSRC_CODE_SAMPLER_CUH_

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

#include "decoder_layer.cuh"
#include "gemv_core.cuh"

namespace s2mk {

constexpr int kCodeIndexBits = 11;
constexpr int kCodeKeyBits = 16 + kCodeIndexBits;
constexpr int kDigitBits = 9;
constexpr int kDigits = 1 << kDigitBits;
static_assert(kDigits == kThreads && kCodeKeyBits % kDigitBits == 0);

// A key whose unsigned order is the value descending, then the index
// ascending, for a value already rounded to bf16.
__device__ __forceinline__ uint32_t CodeKey(float bf16_value, int index) {
  uint32_t u = __float_as_uint(bf16_value) >> 16;
  u = (u & 0x8000u) ? (~u & 0xFFFFu) : (u | 0x8000u);
  return (u << kCodeIndexBits) | ((1u << kCodeIndexBits) - 1 - index);
}

__device__ __forceinline__ float CodeKeyValue(uint32_t key) {
  uint32_t u = key >> kCodeIndexBits;
  u = (u & 0x8000u) ? (u & 0x7FFFu) : (~u & 0xFFFFu);
  return __uint_as_float(u << 16);
}

__device__ __forceinline__ int CodeKeyIndex(uint32_t key) {
  constexpr uint32_t kMask = (1u << kCodeIndexBits) - 1;
  return static_cast<int>(kMask - (key & kMask));
}

template <int kTopK>
struct CodeSamplerScratch {
  unsigned histogram[kDigits];
  unsigned warp_total[kWarps];
  uint32_t kth;  // the k-th largest key, resolved kDigitBits at a time
  unsigned remaining;  // its rank among the keys sharing the resolved bits
  unsigned count;
  uint32_t candidates[kTopK];  // the top-k keys, in no order
  uint32_t ranked[kTopK];  // the top-k keys, in rank order
  int code;
};

// The code production draws from `logits` ([kVocab], which other CTAs may
// have written) with `uniforms` ([kVocab], the Gumbel noise's uniforms),
// keeping the top min(top_k, kTopK) logits and then the ranks whose preceding
// probability mass is below `top_p`. Called by every thread of the CTA; every
// thread gets the code. Kept out of line, like SampleToken.
template <int kVocab, int kTopK>
inline __device__ __noinline__ int SampleCode(const float* logits,
                                              const float* uniforms, int top_k,
                                              float top_p,
                                              CodeSamplerScratch<kTopK>& s) {
  static_assert(kVocab % kThreads == 0 && kVocab <= (1 << kCodeIndexBits));
  static_assert(kTopK <= 64, "warp 0 holds two ranks per lane");
  constexpr int kPerThread = kVocab / kThreads;
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int k = min(top_k, kTopK);

  uint32_t keys[kPerThread];
#pragma unroll
  for (int j = 0; j < kPerThread; ++j) {
    const int i = threadIdx.x + j * kThreads;
    keys[j] = CodeKey(RoundBf16(__ldcg(logits + i)), i);
  }
  if (threadIdx.x == 0) {
    s.kth = 0;
    s.remaining = k;
    s.count = 0;
  }

  // Thread t owns digit kDigits − 1 − t, so an inclusive scan over threads
  // counts the keys at or above each digit.
  const unsigned digit = kDigits - 1 - threadIdx.x;
#pragma unroll
  for (int shift = kCodeKeyBits - kDigitBits; shift >= 0;
       shift -= kDigitBits) {
    s.histogram[threadIdx.x] = 0;
    __syncthreads();
    const uint32_t resolved = s.kth >> (shift + kDigitBits);
#pragma unroll
    for (int j = 0; j < kPerThread; ++j) {
      if ((keys[j] >> (shift + kDigitBits)) == resolved) {
        atomicAdd(&s.histogram[(keys[j] >> shift) & (kDigits - 1)], 1u);
      }
    }
    __syncthreads();
    const unsigned here = s.histogram[digit];
    unsigned at_or_above = here;
#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
      const unsigned other = __shfl_up_sync(0xffffffffu, at_or_above, offset);
      if (lane >= offset) at_or_above += other;
    }
    if (lane == 31) s.warp_total[warp] = at_or_above;
    __syncthreads();
    for (int w = 0; w < warp; ++w) at_or_above += s.warp_total[w];
    const unsigned above = at_or_above - here;
    const unsigned remaining = s.remaining;
    __syncthreads();
    if (above < remaining && remaining <= at_or_above) {
      s.kth |= digit << shift;
      s.remaining = remaining - above;
    }
    __syncthreads();
  }

  const uint32_t kth = s.kth;
#pragma unroll
  for (int j = 0; j < kPerThread; ++j) {
    if (keys[j] >= kth) s.candidates[atomicAdd(&s.count, 1u)] = keys[j];
  }
  __syncthreads();
  if (threadIdx.x < k) {
    const uint32_t mine = s.candidates[threadIdx.x];
    int rank = 0;
    for (int j = 0; j < k; ++j) rank += s.candidates[j] > mine;
    s.ranked[rank] = mine;
  }
  __syncthreads();

  if (warp == 0) {
    // Lane l holds ranks l and l + 32.
    float value[2];
    int index[2];
    bool live[2];
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int rank = lane + 32 * h;
      live[h] = rank < k;
      value[h] = live[h] ? CodeKeyValue(s.ranked[rank]) : -INFINITY;
      index[h] = live[h] ? CodeKeyIndex(s.ranked[rank]) : 0;
    }
    const float top = __shfl_sync(0xffffffffu, value[0], 0);
    float probability[2];
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      probability[h] = live[h] ? expf(value[h] - top) : 0.f;
    }
    const float total = WarpSum(probability[0] + probability[1]);
    probability[0] /= total;
    probability[1] /= total;

    // Inclusive cumulative probability in rank order: ranks 0-31, then 32-63.
    float cumulative[2] = {probability[0], probability[1]};
#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const float other = __shfl_up_sync(0xffffffffu, cumulative[h], offset);
        if (lane >= offset) cumulative[h] += other;
      }
    }
    cumulative[1] += __shfl_sync(0xffffffffu, cumulative[0], 31);

    // Production drops a rank once the probability before it reaches top_p,
    // then draws argmax(logit − log(−log u)); the first maximum by codebook
    // index wins, as torch.argmax picks it.
    float score = -INFINITY;
    int winner = kVocab;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      if (!live[h] || cumulative[h] - probability[h] >= top_p) continue;
      const float u = __ldcg(uniforms + index[h]);
      const float candidate = value[h] - logf(-logf(u));
      if (candidate > score || (candidate == score && index[h] < winner)) {
        score = candidate;
        winner = index[h];
      }
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      const float other_score = __shfl_xor_sync(0xffffffffu, score, offset);
      const int other = __shfl_xor_sync(0xffffffffu, winner, offset);
      if (other_score > score || (other_score == score && other < winner)) {
        score = other_score;
        winner = other;
      }
    }
    if (lane == 0) s.code = winner;
  }
  __syncthreads();
  const int code = s.code;
  __syncthreads();
  return code;
}

}  // namespace s2mk

#endif  // S2MK_CSRC_CODE_SAMPLER_CUH_
