// Copyright 2026 Fractalyze Inc. All rights reserved.
//
// One decode step of Qwen3-Omni's talker (talker_decode.h) as a persistent
// launch, a grid barrier after each of a layer's six phases:
//
//   1. Every CTA normalizes the residual itself and computes its share of the
//      q, k and v rows.
//   2. Query head h and chunk s form item h × S + s, one per CTA: QK-norm,
//      M-RoPE, the cache write and attention over the chunk, as the thinker's
//      (thinker_layer.cuh).
//   3. Every CTA merges the partials and computes its share of the O rows,
//      added to the residual.
//   4. Every CTA normalizes the residual and computes its share of the
//      router logits, the shared expert's gate logit and its SiLU(gate) × up.
//   5. Every CTA picks the top 6 experts and computes its share of their
//      SiLU(gate) × up.
//   6. Every CTA computes its share of the output rows: the six experts' down
//      rows, weighted, and the shared expert's, scaled by the sigmoid of its
//      gate, added to the residual.
//
// Every dot product is one warp's: its lanes stride the row and WarpSum adds
// them, so each row sums in one order at any CTA count.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "barrier.cuh"
#include "decoder_layer.cuh"
#include "gemv_core.cuh"
#include "talker_decode.h"
#include "thinker_layer.cuh"

namespace s2mk {
namespace {

constexpr int kDim = kTalkerDim;
constexpr int kExperts = kTalkerExperts;
constexpr int kTopK = kTalkerTopK;
constexpr int kFfn = kTalkerExpertFfn;
constexpr int kShared = kTalkerSharedFfn;
constexpr int kPairs = kTopK * kFfn;
// Rows, in 16-byte vectors, at each width.
constexpr int kDimVecs = kDim / kVecElems;
constexpr int kQVecs = kTalkerQDim / kVecElems;
constexpr int kFfnVecs = kFfn / kVecElems;
constexpr int kSharedVecs = kShared / kVecElems;
// Phase 4's units: the router's rows, the shared gate's row, then the shared
// expert's (gate, up) pairs.
constexpr int kSharedGateUnit = kExperts;
constexpr int kMoeUnits = kExperts + 1 + kShared;

static_assert(kTalkerHeadDim == thinker::kHead, "the thinker's attention dims");
static_assert(kTalkerQDim <= thinker::kQDim, "AttentionShared::xs holds O's input");

struct MoeShared {
  uint4 h[kDimVecs];  // the normalized residual, bf16
  uint4 act[(kPairs + kShared) / kVecElems];  // SiLU(gate) × up, bf16
  int experts[kTopK];
  float weights[kTopK];
};

struct Shared {
  thinker::AttentionShared attention;
  MoeShared moe;
};

// The attention fields thinker::Attend and thinker::Merge read, for a layer.
struct AttentionView {
  const float* qkv;
  const __nv_bfloat16* q_norm;
  const __nv_bfloat16* k_norm;
  const __nv_bfloat16* cos_sin;
  PagedKv kv;
  float eps;
  int splits;
  float* partial_ml;
  float* partial_o;
};

struct RowRef {
  const uint4* w;
  const uint4* x;  // shared memory
};

// ys[r] = map(r).w · map(r).x over kVecs 16-byte vectors, for r in [0, rows):
// warp w takes rows [w × kAtOnce, (w + 1) × kAtOnce), then the next kWarps ×
// kAtOnce, issuing a group's loads before its first use. Every thread calls
// it; ys is complete on return.
template <int kVecs, int kAtOnce, typename Map>
__device__ void WarpRows(int rows, const Map& map, float* ys) {
  constexpr int kPerLane = (kVecs + 31) / 32;
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  for (int base = warp * kAtOnce; base < rows; base += kWarps * kAtOnce) {
    uint4 w[kAtOnce][kPerLane];
#pragma unroll
    for (int a = 0; a < kAtOnce; ++a) {
      if (base + a >= rows) break;
      const uint4* row = map(base + a).w;
#pragma unroll
      for (int j = 0; j < kPerLane; ++j) {
        if (lane + 32 * j < kVecs) w[a][j] = LoadStream(row + lane + 32 * j);
      }
    }
#pragma unroll
    for (int a = 0; a < kAtOnce; ++a) {
      if (base + a >= rows) break;
      const uint4* x = map(base + a).x;
      float acc = 0.f;
#pragma unroll
      for (int j = 0; j < kPerLane; ++j) {
        if (lane + 32 * j < kVecs) acc += Dot8(w[a][j], x[lane + 32 * j]);
      }
      acc = WarpSum(acc);
      if (lane == 0) ys[base + a] = acc;
    }
  }
  __syncthreads();
}

// Starts fetching into L2 `rows` rows of a row-major bf16 weight of `vecs`
// 16-byte vectors from row `begin`. One thread calls it.
__device__ __forceinline__ void PrefetchRows(const __nv_bfloat16* w, int vecs,
                                             int64_t begin, int rows) {
  if (rows > 0) {
    thinker::PrefetchL2(w + begin * vecs * kVecElems,
                        int64_t{rows} * vecs * sizeof(uint4));
  }
}

// This CTA's phase-4 units as row ranges: [unit_begin, unit_begin + singles)
// of the router (the last may be the shared gate's row) and `pairs` shared
// (gate, up) pairs from `first_pair`.
struct MoeUnits {
  int unit_begin;
  int singles;
  int first_pair;
  int pairs;
};

__device__ __forceinline__ MoeUnits UnitsOf(int cta) {
  const int unit_begin = RowBegin(kMoeUnits, cta);
  const int unit_end = RowBegin(kMoeUnits, cta + 1);
  const int pair_start = max(unit_begin, kSharedGateUnit + 1);
  return {unit_begin, max(0, min(unit_end, kSharedGateUnit + 1) - unit_begin),
          pair_start - (kSharedGateUnit + 1), max(0, unit_end - pair_start)};
}

// On arriving at barrier `index`, starts fetching into L2 this CTA's rows of
// the phase after the next, so HBM streams them through a phase and two
// barriers: O's during attention, phase 4's during O, and the next layer's
// q, k and v during the down rows. The experts' rows are named only by
// routing; phase 5 prefetches the down rows itself.
__device__ void PrefetchAhead(const TalkerDecodeParams& p, int index) {
  if (threadIdx.x != 0) return;
  const int layer = index / kTalkerBarriersPerLayer;
  const int cta = blockIdx.x;
  const TalkerLayerParams& w = p.layers[layer];
  switch (index % kTalkerBarriersPerLayer) {
    case 0: {  // after QKV
      const int begin = RowBegin(kDim, cta);
      PrefetchRows(w.wo, kQVecs, begin, RowBegin(kDim, cta + 1) - begin);
      break;
    }
    case 1: {  // after attention
      const MoeUnits u = UnitsOf(cta);
      const int routers = min(u.singles, kExperts - u.unit_begin);
      if (routers > 0) PrefetchRows(w.router, kDimVecs, u.unit_begin, routers);
      PrefetchRows(w.shared_w13, kDimVecs, u.first_pair, u.pairs);
      PrefetchRows(w.shared_w13, kDimVecs, kShared + u.first_pair, u.pairs);
      break;
    }
    case 4: {  // after gate-up
      if (layer + 1 < p.num_layers) {
        const int begin = RowBegin(kTalkerQkvRows, cta);
        PrefetchRows(p.layers[layer + 1].wqkv, kDimVecs, begin,
                     RowBegin(kTalkerQkvRows, cta + 1) - begin);
      }
      break;
    }
    default:
      break;
  }
}

// Copies this CTA's rows of the residual into hidden row `row`.
__device__ void DumpHidden(const TalkerDecodeParams& p, int row) {
  if (p.hidden == nullptr) return;
  const int begin = RowBegin(kDim, blockIdx.x);
  const int end = RowBegin(kDim, blockIdx.x + 1);
  for (int i = begin + threadIdx.x; i < end; i += kThreads) {
    p.hidden[int64_t{row} * kDim + i] = __ldcg(p.residual + i);
  }
}

template <typename Sync>
__device__ void Attention(const TalkerDecodeParams& p,
                          const TalkerLayerParams& w,
                          const thinker::BlockStep& step, Sync& sync,
                          Shared& sh) {
  const int cta = blockIdx.x;
  thinker::AttentionShared& as = sh.attention;
  __nv_bfloat16* xs = reinterpret_cast<__nv_bfloat16*>(as.xs);
  const AttentionView view{p.qkv,     w.q_norm, w.k_norm,     p.cos_sin,
                           w.kv,      p.eps,    p.splits,     p.partial_ml,
                           p.partial_o};

  // 1. RMSNorm, then this CTA's q, k and v rows.
  RmsNorm<TalkerNormDims>(p.residual, nullptr, w.norm, p.eps, xs, nullptr,
                          as.red);
  __syncthreads();
  {
    const int begin = RowBegin(kTalkerQkvRows, cta);
    const int rows = RowBegin(kTalkerQkvRows, cta + 1) - begin;
    const uint4* wqkv = reinterpret_cast<const uint4*>(w.wqkv);
    WarpRows<kDimVecs, 4>(
        rows,
        [&](int r) {
          return RowRef{wqkv + int64_t{begin + r} * kDimVecs, as.xs};
        },
        as.ys);
    for (int i = threadIdx.x; i < rows; i += kThreads) {
      p.qkv[begin + i] = as.ys[i];
    }
  }
  sync();

  // 2. This CTA's (head, chunk) item.
  thinker::Attend<kTalkerQHeads, kTalkerKvHeads>(view, step, as);
  sync();

  // 3. The merged attention output, then this CTA's O rows.
  thinker::Merge<kTalkerQHeads>(view, as, xs);
  __syncthreads();
  {
    const int begin = RowBegin(kDim, cta);
    const int rows = RowBegin(kDim, cta + 1) - begin;
    const uint4* wo = reinterpret_cast<const uint4*>(w.wo);
    WarpRows<kQVecs, 2>(
        rows,
        [&](int r) {
          return RowRef{wo + int64_t{begin + r} * kQVecs, as.xs};
        },
        as.ys);
    for (int i = threadIdx.x; i < rows; i += kThreads) {
      p.residual[begin + i] = __ldcg(p.residual + begin + i) + as.ys[i];
    }
  }
}

template <typename Sync>
__device__ void Moe(const TalkerDecodeParams& p, const TalkerLayerParams& w,
                    Sync& sync, Shared& sh) {
  const int cta = blockIdx.x;
  MoeShared& ms = sh.moe;
  float* ys = sh.attention.ys;
  __nv_bfloat16* act = reinterpret_cast<__nv_bfloat16*>(ms.act);

  // 4. RMSNorm, then this CTA's units: router rows and the shared gate's row,
  //    one row each, then shared (gate, up) pairs, two rows each.
  RmsNorm<TalkerNormDims>(p.residual, nullptr, w.moe_norm, p.eps,
                          reinterpret_cast<__nv_bfloat16*>(ms.h), nullptr,
                          sh.attention.red);
  __syncthreads();
  {
    const MoeUnits units = UnitsOf(cta);
    const int unit_begin = units.unit_begin;
    const int singles = units.singles;
    const int pairs = units.pairs;
    const int first_pair = units.first_pair;
    const uint4* router = reinterpret_cast<const uint4*>(w.router);
    const uint4* gate = reinterpret_cast<const uint4*>(w.shared_gate);
    const uint4* w13 = reinterpret_cast<const uint4*>(w.shared_w13);
    WarpRows<kDimVecs, 4>(
        singles + 2 * pairs,
        [&](int r) {
          if (r < singles) {
            const int unit = unit_begin + r;
            return RowRef{unit == kSharedGateUnit
                              ? gate
                              : router + int64_t{unit} * kDimVecs,
                          ms.h};
          }
          const int q = r - singles;
          const int64_t row = (q % 2) * kShared + first_pair + q / 2;
          return RowRef{w13 + row * kDimVecs, ms.h};
        },
        ys);
    for (int i = threadIdx.x; i < singles; i += kThreads) {
      p.router_logits[unit_begin + i] = ys[i];
    }
    for (int i = threadIdx.x; i < pairs; i += kThreads) {
      const float g = ys[singles + 2 * i];
      const float u = ys[singles + 2 * i + 1];
      p.act[kPairs + first_pair + i] = __float2bfloat16(Silu(g) * u);
    }
  }
  sync();

  // 5. Routing, then this CTA's (gate, up) pairs of the chosen experts.
  if (threadIdx.x < 32) {
    thinker::RouteWarp<kExperts, kTopK>(p.router_logits, ms.experts,
                                        ms.weights);
  }
  __syncthreads();
  {
    const int begin = RowBegin(kPairs, cta);
    const int pairs = RowBegin(kPairs, cta + 1) - begin;
    const uint4* w13 = reinterpret_cast<const uint4*>(w.w13);
    WarpRows<kDimVecs, 4>(
        2 * pairs,
        [&](int r) {
          const int pair = begin + r / 2;
          const int64_t row = int64_t{ms.experts[pair / kFfn]} * 2 * kFfn +
                              (r % 2) * kFfn + pair % kFfn;
          return RowRef{w13 + row * kDimVecs, ms.h};
        },
        ys);
    // The down rows are unknown until routing, so this is their earliest
    // prefetch: one thread per chosen expert, and one for the shared one.
    const int out_begin = RowBegin(kDim, cta);
    const int out_rows = RowBegin(kDim, cta + 1) - out_begin;
    if (threadIdx.x < kTopK) {
      PrefetchRows(w.w2, kFfnVecs,
                   int64_t{ms.experts[threadIdx.x]} * kDim + out_begin, out_rows);
    } else if (threadIdx.x == kTopK) {
      PrefetchRows(w.shared_w2, kSharedVecs, out_begin, out_rows);
    }
    for (int i = threadIdx.x; i < pairs; i += kThreads) {
      p.act[begin + i] = __float2bfloat16(Silu(ys[2 * i]) * ys[2 * i + 1]);
    }
  }
  sync();

  // 6. This CTA's output rows: the chosen experts' down rows in rank order,
  //    then the shared expert's.
  for (int i = threadIdx.x; i < (kPairs + kShared) / kVecElems; i += kThreads) {
    ms.act[i] = __ldcg(reinterpret_cast<const uint4*>(p.act) + i);
  }
  __syncthreads();
  {
    const int begin = RowBegin(kDim, cta);
    const int rows = RowBegin(kDim, cta + 1) - begin;
    const uint4* w2 = reinterpret_cast<const uint4*>(w.w2);
    const uint4* shared_w2 = reinterpret_cast<const uint4*>(w.shared_w2);
    WarpRows<kFfnVecs, 8>(
        kTopK * rows,
        [&](int r) {
          const int rank = r / rows;
          const int64_t row =
              int64_t{ms.experts[rank]} * kDim + begin + r % rows;
          return RowRef{w2 + row * kFfnVecs, ms.act + rank * kFfnVecs};
        },
        ys);
    WarpRows<kSharedVecs, 4>(
        rows,
        [&](int r) {
          return RowRef{shared_w2 + int64_t{begin + r} * kSharedVecs,
                        ms.act + kTopK * kFfnVecs};
        },
        ys + kTopK * rows);
    const float shared_scale =
        1.f / (1.f + expf(-RoundBf16(__ldcg(p.router_logits + kSharedGateUnit))));
    for (int i = threadIdx.x; i < rows; i += kThreads) {
      float moe = 0.f;
      for (int rank = 0; rank < kTopK; ++rank) {
        moe = fmaf(ms.weights[rank], ys[rank * rows + i], moe);
      }
      moe = fmaf(shared_scale, ys[kTopK * rows + i], moe);
      p.residual[begin + i] = __ldcg(p.residual + begin + i) + moe;
    }
  }
}

__global__ void __launch_bounds__(kThreads, 1)
    TalkerDecodeKernel(const __grid_constant__ TalkerDecodeParams p) {
  __shared__ Shared sh;
  GridBarrier barrier(p.sync, p.error, p.timeout_ns, 0);
  int64_t* stamps =
      p.profile == nullptr
          ? nullptr
          : p.profile + int64_t{blockIdx.x} * 2 * kTalkerBarriersPerLayer *
                            p.num_layers;
  // The arrival stamp waits for the whole CTA to finish the phase.
  auto sync = [&] {
    PrefetchAhead(p, barrier.index());
    if (stamps != nullptr) {
      __syncthreads();
      if (threadIdx.x == 0) stamps[2 * barrier.index()] = GlobalTimer();
    }
    barrier.Sync();
    if (stamps != nullptr && threadIdx.x == 0) {
      stamps[2 * (barrier.index() - 1) + 1] = GlobalTimer();
    }
  };

  const thinker::BlockStep step{
      p.seq_len[0] - 1,
      p.slot_mapping[0],
      {p.positions[0], p.positions[p.positions_stride],
       p.positions[2 * p.positions_stride]},
      p.residual,
      p.residual};
  for (int layer = 0; layer < p.num_layers; ++layer) {
    const TalkerLayerParams& w = p.layers[layer];
    DumpHidden(p, layer);
    Attention(p, w, step, sync, sh);
    sync();
    Moe(p, w, sync, sh);
    sync();
  }
  DumpHidden(p, p.num_layers);
  RmsNorm<TalkerNormDims>(p.residual, nullptr, p.final_norm, p.eps,
                          reinterpret_cast<__nv_bfloat16*>(sh.attention.xs),
                          blockIdx.x == 0 ? p.final_hidden : nullptr,
                          sh.attention.red);
}

}  // namespace

cudaError_t LaunchTalkerDecode(const TalkerDecodeParams& params, int num_ctas,
                               cudaStream_t stream) {
  cudaError_t err =
      cudaMemsetAsync(params.sync, 0, kSyncWords * sizeof(unsigned), stream);
  if (err != cudaSuccess) return err;
  TalkerDecodeKernel<<<num_ctas, kThreads, 0, stream>>>(params);
  return cudaGetLastError();
}

}  // namespace s2mk
