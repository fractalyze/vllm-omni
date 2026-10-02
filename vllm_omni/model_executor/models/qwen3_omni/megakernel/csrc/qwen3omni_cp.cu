// Copyright 2026 Fractalyze Inc. All rights reserved.
//
// The Qwen3-Omni code-predictor megakernel: N frames of the code predictor in
// one persistent launch of one CTA per SM. A frame runs positions 0..15 of a
// 16-entry KV cache:
//
//   1. the position's input, which every CTA holds whole: the talker's hidden
//      state at position 0, codebook 0's embedding at position 1, and the
//      embedding of code p − 1 (from table p − 2) at position p ≥ 2;
//   2. the decoder layers (decoder_layer.cuh) with short attention: every
//      CTA attends every head over the ≤ 16 positions, saving a barrier a
//      layer;
//   3. at position p ≥ 1, the final norm and head p − 1, then one barrier;
//   4. the code sampler (code_sampler.cuh), run redundantly on every CTA.
//
// Position 0 feeds no head, so the release build runs only the QKV GEMV and
// the key-value write of its last layer. The dump build runs the whole
// layer, so its output can be checked.
//
// Each GEMV prefetches the first slice of the GEMV after it into L2: a
// layer's last GEMV → the next layer or the head → the next position's first
// layer.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "barrier.cuh"
#include "code_sampler.cuh"
#include "decoder_layer.cuh"
#include "gemv_core.cuh"
#include "qwen3omni_cp.h"

namespace s2mk {
namespace {

using CpSampler = CodeSamplerScratch<kCpMaxTopK>;
// Positions a warp loads per L2 round trip in the short attention, bounded
// by the registers their keys and values hold.
constexpr int kAttentionChunk = 4;

__device__ __forceinline__ void LoadRow(const __nv_bfloat16* row, float* x0) {
  for (int i = threadIdx.x; i < CpDims::kDim; i += kThreads) {
    x0[i] = __bfloat162float(row[i]);
  }
}

__device__ __forceinline__ const __nv_bfloat16* TableRow(
    const __nv_bfloat16* tables, int table, int row) {
  return tables + (int64_t{table} * kCpVocab + row) * CpDims::kDim;
}

template <bool kDump, bool kProfile>
__global__ void __launch_bounds__(kThreads, 1)
    CodePredictorKernel(const __grid_constant__ CodePredictorParams p) {
  __shared__ Shared<CpDims> sh;
  __shared__ CpSampler sampler;
  // The layers' weight pointers, which the layer reads in its latency-bound
  // attention: from shared memory rather than one global load each.
  __shared__ LayerWeights layers[kCpMaxLayers];
  __nv_bfloat16* xs = reinterpret_cast<__nv_bfloat16*>(sh.xs);
  GridBarrier barrier(p.sync, p.error, p.timeout_ns, 0);
  const int cta = blockIdx.x;
  constexpr int kDim = CpDims::kDim;

  const LayerBuffers buffers{p.residual, p.qkv, nullptr, nullptr,
                             p.act,      p.eps, p.prefetch_bytes};
  // Thread 0's clock64 cycles waiting at barriers and in the sampler, and
  // working and waiting per phase; `phase` is the next barrier's phase.
  int64_t waited = 0;
  int64_t sampling = 0;
  int64_t phase_work[kCpPhases] = {};
  int64_t phase_wait[kCpPhases] = {};
  int phase = 0;
  int64_t left = 0;
  if constexpr (kProfile) {
    if (threadIdx.x == 0) {
      int64_t* row = p.profile + int64_t{cta} * kCpProfileWords;
      row[0] = GlobalTimer();
      row[2] = clock64();
      row[4] = SmId();
    }
    left = clock64();
  }
  // The grid barrier, its wait counted in a profiled build from the moment the
  // whole CTA has finished the phase.
  auto sync = [&] {
    int64_t start = 0;
    if constexpr (kProfile) {
      __syncthreads();
      start = clock64();
    }
    barrier.Sync();
    if constexpr (kProfile) {
      const int64_t now = clock64();
      waited += now - start;
      phase_work[phase] += start - left;
      phase_wait[phase] += now - start;
      left = now;
      ++phase;
    }
  };
  auto begin_phases = [&](CpPhase first_phase) {
    if constexpr (kProfile) phase = static_cast<int>(first_phase);
  };
  for (int i = threadIdx.x; i < p.num_layers; i += kThreads) {
    layers[i] = p.layers[i];
  }
  __syncthreads();
  const GemvSlice first = QkvSlice<CpDims>(layers[0]);
  const int last = p.num_layers - 1;

  for (int frame = 0; frame < p.num_frames; ++frame) {
    barrier.set_step(frame);
    LoadRow(p.talker_hidden + int64_t{frame} * kDim, sh.x0);
    for (int pos = 0; pos < kCpPositions; ++pos) {
      const GemvSlice head{
          p.heads + int64_t{max(pos - 1, 0)} * kCpVocab * kDim, kDim, kCpVocab,
          1};
      float* dump = kDump ? p.dump + (int64_t{frame} * kCpPositions + pos) *
                                         (p.num_layers + 1) * kDim
                          : nullptr;
      PublishInput<CpDims>(sh.x0, p.residual, dump);
      for (int layer = 0; layer <= last; ++layer) {
        const AttentionStep attention{p.kv, layer, pos, p.rope, 1};
        const float* local = layer == 0 ? sh.x0 : nullptr;
        float* layer_dump = kDump ? dump + (layer + 1) * kDim : nullptr;
        const bool kv_only = layer == last && pos == 0 && !kDump;
        begin_phases(kv_only ? CpPhase::kKvOnlyQkv : CpPhase::kQkv);
        if (layer < last) {
          DecoderLayer<CpDims, AttentionKind::kShort, kCpPositions,
                       kAttentionChunk>(
              layers[layer], buffers, attention, local,
              QkvSlice<CpDims>(layers[layer + 1]), layer_dump, sync, sh);
        } else if (pos > 0 || kDump) {
          DecoderLayer<CpDims, AttentionKind::kShort, kCpPositions,
                       kAttentionChunk>(
              layers[layer], buffers, attention, local,
              pos > 0 ? head : first, layer_dump, sync, sh);
        } else {
          DecoderLayerKvOnly<CpDims>(layers[layer], buffers, attention, local,
                                     first, sync, sh);
        }
      }
      if (pos == 0) {
        LoadRow(p.code0_embed + int64_t{frame} * kDim, sh.x0);
        continue;
      }

      // Head pos − 1 and the draw of code pos.
      const int pass = pos - 1;
      float* logits =
          p.logits + (int64_t{frame} * kCpHeads + pass) * kCpVocab;
      begin_phases(CpPhase::kHead);
      RmsNorm<CpDims>(p.residual, nullptr, p.final_norm, p.eps, xs, nullptr,
                      sh.red);
      {
        const int begin = RowBegin(kCpVocab, cta);
        const int rows = RowBegin(kCpVocab, cta + 1) - begin;
        LayerGemv(head.w, kDim, begin, rows, sh);
        Prefetch(first, p.prefetch_bytes);
        for (int i = threadIdx.x; i < rows; i += kThreads) {
          logits[begin + i] = sh.ys[i];
        }
      }
      sync();
      const int64_t out = int64_t{frame} * kCpHeads + pass;
      const int64_t sample_start = kProfile ? clock64() : 0;
      const int code = SampleCode<kCpVocab>(
          logits, p.uniforms + out * kCpVocab, p.top_k, p.top_p, sampler);
      if constexpr (kProfile) {
        left = clock64();
        sampling += left - sample_start;
      }
      if (cta == 0 && threadIdx.x == 0) p.codes[out] = code;
      if (pos < kCpHeads) {
        const int fed = p.forced_codes != nullptr
                            ? static_cast<int>(p.forced_codes[out])
                            : code;
        LoadRow(TableRow(p.embeddings, pass, fed), sh.x0);
      }
    }
  }

  if constexpr (kProfile) {
    __syncthreads();
    if (threadIdx.x == 0) {
      int64_t* row = p.profile + int64_t{cta} * kCpProfileWords;
      row[3] = clock64();
      row[1] = GlobalTimer();
      row[kProfileHeader] = waited;
      row[kProfileHeader + 1] = sampling;
      row[kProfileHeader + 2] = barrier.index();
      for (int i = 0; i < kCpPhases; ++i) {
        row[kProfileHeader + 3 + i] = phase_work[i];
        row[kProfileHeader + 3 + kCpPhases + i] = phase_wait[i];
      }
    }
  }
}

template <bool kDump, bool kProfile>
const void* Kernel() {
  return reinterpret_cast<const void*>(CodePredictorKernel<kDump, kProfile>);
}

__global__ void __launch_bounds__(kThreads, 1)
    CodeSamplerProbeKernel(const float* logits, const float* uniforms,
                           int top_k, float top_p, int64_t* codes) {
  __shared__ CpSampler sampler;
  const int64_t c = blockIdx.x;
  const int code = SampleCode<kCpVocab>(logits + c * kCpVocab,
                                        uniforms + c * kCpVocab, top_k, top_p,
                                        sampler);
  if (threadIdx.x == 0) codes[c] = code;
}

}  // namespace

cudaError_t LaunchCodePredictor(const CodePredictorParams& params,
                                int num_ctas, bool cooperative,
                                cudaStream_t stream) {
  cudaError_t err =
      cudaMemsetAsync(params.sync, 0, kSyncWords * sizeof(unsigned), stream);
  if (err != cudaSuccess) return err;
  const bool profile = params.profile != nullptr;
  const void* kernel =
      params.dump != nullptr
          ? (profile ? Kernel<true, true>() : Kernel<true, false>())
          : (profile ? Kernel<false, true>() : Kernel<false, false>());
  cudaLaunchAttribute attr = {};
  attr.id = cudaLaunchAttributeCooperative;
  attr.val.cooperative = 1;
  cudaLaunchConfig_t config = {};
  config.gridDim = dim3(num_ctas);
  config.blockDim = dim3(kThreads);
  config.stream = stream;
  config.attrs = &attr;
  config.numAttrs = cooperative ? 1 : 0;
  void* args[] = {const_cast<CodePredictorParams*>(&params)};
  return cudaLaunchKernelExC(&config, kernel, args);
}

cudaError_t LaunchCodeSamplerProbe(const float* logits, const float* uniforms,
                                   int top_k, float top_p, int64_t* codes,
                                   int cases, cudaStream_t stream) {
  CodeSamplerProbeKernel<<<cases, kThreads, 0, stream>>>(logits, uniforms,
                                                         top_k, top_p, codes);
  return cudaGetLastError();
}

}  // namespace s2mk
