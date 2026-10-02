// Copyright 2026 Fractalyze Inc. All rights reserved.
//
// The Python bindings of Qwen3-Omni's megakernels: the thinker, talker and
// code-predictor slice of decode-mk's s2mk/csrc/ops.cpp.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <algorithm>
#include <cstring>
#include <optional>

#include "barrier.h"
#include "layer.h"
#include "qwen3omni_cp.h"
#include "talker_decode.h"
#include "thinker_attention.h"
#include "thinker_decode.h"
#include "thinker_moe.h"
#include "thinker_prefill.h"

namespace s2mk {
namespace {

void CheckOk(cudaError_t err, const char* what) {
  TORCH_CHECK(err == cudaSuccess, what, " failed: ", cudaGetErrorString(err));
}

// The data pointer of a contiguous CUDA tensor of `dtype`.
template <typename T>
T* Ptr(const torch::Tensor& t, torch::ScalarType dtype, const char* name) {
  TORCH_CHECK(t.is_cuda(), name, " must live on the GPU");
  TORCH_CHECK(t.scalar_type() == dtype, name, " must be ", dtype);
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
  return reinterpret_cast<T*>(t.data_ptr());
}

// The device view of an ErrorRecord held in a pinned int32 CPU tensor.
ErrorRecord* MappedRecord(const torch::Tensor& error) {
  TORCH_CHECK(error.is_pinned() && error.scalar_type() == torch::kInt32 &&
                  error.numel() * sizeof(int32_t) == sizeof(ErrorRecord),
              "error must be a pinned int32 tensor of ",
              sizeof(ErrorRecord) / sizeof(int32_t), " elements");
  void* device = nullptr;
  CheckOk(cudaHostGetDevicePointer(&device, error.data_ptr(), 0),
          "mapping the error record");
  return static_cast<ErrorRecord*>(device);
}

// Checks a [num_layers, 8] table of LayerWeights pointers.
const LayerWeights* LayerTable(const torch::Tensor& layers, const char* name) {
  static_assert(sizeof(LayerWeights) == 8 * sizeof(int64_t));
  TORCH_CHECK(layers.dim() == 2 && layers.size(1) == 8 && layers.size(0) > 0,
              name, " must be [num_layers, 8]");
  return Ptr<const LayerWeights>(layers, torch::kInt64, name);
}

void CheckNumel(const torch::Tensor& t, int64_t numel, const char* name) {
  TORCH_CHECK(t.numel() == numel, name, " must hold ", numel,
              " elements, got ", t.numel());
}

void CheckPrefetch(int64_t prefetch_bytes) {
  TORCH_CHECK(0 <= prefetch_bytes && prefetch_bytes <= (1 << 30) &&
                  prefetch_bytes % 16 == 0,
              "prefetch_bytes must be a multiple of 16 in [0, 2^30], got ",
              prefetch_bytes);
}

void RunCodePredictor(
    const torch::Tensor& layers, const torch::Tensor& final_norm,
    const torch::Tensor& heads, const torch::Tensor& embeddings,
    const torch::Tensor& rope, const torch::Tensor& k_caches,
    const torch::Tensor& v_caches, double eps, int64_t top_k, double top_p,
    const torch::Tensor& talker_hidden, const torch::Tensor& code0_embed,
    const torch::Tensor& uniforms,
    const std::optional<torch::Tensor>& forced_codes, int64_t prefetch_bytes,
    int64_t timeout_ns, const torch::Tensor& residual, const torch::Tensor& qkv,
    const torch::Tensor& act, const torch::Tensor& logits,
    const torch::Tensor& codes, const std::optional<torch::Tensor>& dump,
    const std::optional<torch::Tensor>& profile, const torch::Tensor& sync,
    const torch::Tensor& error, int64_t num_ctas, bool cooperative) {
  using D = CpDims;
  const LayerWeights* layer_table = LayerTable(layers, "layers");
  const int num_layers = static_cast<int>(layers.size(0));
  const int64_t frames = talker_hidden.numel() / D::kDim;
  TORCH_CHECK(num_layers <= kCpMaxLayers, "at most ", kCpMaxLayers,
              " layers, got ", num_layers);
  TORCH_CHECK(frames > 0, "no frames");
  const int max_rows = std::max({D::kQkvRows, D::kDim, 2 * D::kFfn, kCpVocab});
  TORCH_CHECK(0 < num_ctas && num_ctas <= kMaxCtas &&
                  (max_rows + num_ctas - 1) / num_ctas <= kMaxRowsPerCta,
              num_ctas, " CTAs leave more than ", kMaxRowsPerCta,
              " rows per CTA");
  TORCH_CHECK(0 < top_k && top_k <= kCpMaxTopK, "top_k must be in [1, ",
              kCpMaxTopK, "], got ", top_k);
  CheckPrefetch(prefetch_bytes);
  CheckNumel(final_norm, D::kDim, "final_norm");
  CheckNumel(heads, int64_t{kCpHeads} * kCpVocab * D::kDim, "heads");
  CheckNumel(embeddings, int64_t{kCpHeads} * kCpVocab * D::kDim, "embeddings");
  CheckNumel(rope, int64_t{kCpPositions} * kHeadDim, "rope");
  TORCH_CHECK(k_caches.numel() == num_layers && v_caches.numel() == num_layers,
              "one key and one value cache per layer");
  CheckNumel(talker_hidden, frames * D::kDim, "talker_hidden");
  CheckNumel(code0_embed, frames * D::kDim, "code0_embed");
  CheckNumel(uniforms, frames * kCpHeads * kCpVocab, "uniforms");
  CheckNumel(residual, D::kDim, "residual");
  CheckNumel(qkv, D::kQkvRows, "qkv");
  CheckNumel(act, D::kFfn, "act");
  CheckNumel(logits, frames * kCpHeads * kCpVocab, "logits");
  CheckNumel(codes, frames * kCpHeads, "codes");
  if (forced_codes.has_value()) {
    CheckNumel(*forced_codes, frames * kCpHeads, "forced_codes");
  }
  if (dump.has_value()) {
    CheckNumel(*dump, frames * kCpPositions * (num_layers + 1) * D::kDim,
               "dump");
  }
  if (profile.has_value()) {
    CheckNumel(*profile, num_ctas * kCpProfileWords, "profile");
  }

  const auto kBf16 = torch::kBFloat16;
  const auto kF32 = torch::kFloat32;
  const auto kI64 = torch::kInt64;
  CodePredictorParams p{};
  p.layers = layer_table;
  p.num_layers = num_layers;
  p.final_norm = Ptr<const __nv_bfloat16>(final_norm, kBf16, "final_norm");
  p.heads = Ptr<const __nv_bfloat16>(heads, kBf16, "heads");
  p.embeddings = Ptr<const __nv_bfloat16>(embeddings, kBf16, "embeddings");
  p.rope = Ptr<const __nv_bfloat16>(rope, kBf16, "rope");
  p.kv.k = Ptr<__nv_bfloat16* const>(k_caches, kI64, "k_caches");
  p.kv.v = Ptr<__nv_bfloat16* const>(v_caches, kI64, "v_caches");
  p.kv.max_seq = kCpPositions;
  p.eps = static_cast<float>(eps);
  p.top_k = static_cast<int>(top_k);
  p.top_p = static_cast<float>(top_p);
  p.num_frames = static_cast<int>(frames);
  p.talker_hidden =
      Ptr<const __nv_bfloat16>(talker_hidden, kBf16, "talker_hidden");
  p.code0_embed = Ptr<const __nv_bfloat16>(code0_embed, kBf16, "code0_embed");
  p.uniforms = Ptr<const float>(uniforms, kF32, "uniforms");
  p.forced_codes = forced_codes.has_value()
                       ? Ptr<const int64_t>(*forced_codes, kI64, "forced_codes")
                       : nullptr;
  p.prefetch_bytes = static_cast<int>(prefetch_bytes);
  p.timeout_ns = timeout_ns;
  p.residual = Ptr<float>(residual, kF32, "residual");
  p.qkv = Ptr<float>(qkv, kF32, "qkv");
  p.act = Ptr<__nv_bfloat16>(act, kBf16, "act");
  p.logits = Ptr<float>(logits, kF32, "logits");
  p.codes = Ptr<int64_t>(codes, kI64, "codes");
  p.dump = dump.has_value() ? Ptr<float>(*dump, kF32, "dump") : nullptr;
  p.profile =
      profile.has_value() ? Ptr<int64_t>(*profile, kI64, "profile") : nullptr;
  p.sync = Ptr<unsigned>(sync, torch::kInt32, "sync");
  p.error = MappedRecord(error);

  const c10::cuda::CUDAGuard guard(residual.device());
  CheckOk(LaunchCodePredictor(p, static_cast<int>(num_ctas), cooperative,
                              at::cuda::getCurrentCUDAStream()),
          "code predictor launch");
}

// A params struct as a CPU uint8 tensor, and back: the thinker's blocks are
// built once per layer and launched, alone or stacked into a decode step,
// from these bytes.
template <typename T>
torch::Tensor StructBytes(const T& value) {
  torch::Tensor bytes = torch::empty({static_cast<int64_t>(sizeof(T))},
                                     torch::dtype(torch::kUInt8));
  std::memcpy(bytes.data_ptr(), &value, sizeof(T));
  return bytes;
}

template <typename T>
T StructFrom(const torch::Tensor& bytes, const char* name) {
  TORCH_CHECK(bytes.device().is_cpu() && bytes.scalar_type() == torch::kUInt8 &&
                  bytes.is_contiguous() && bytes.numel() == sizeof(T),
              name, " must be ", sizeof(T), " contiguous CPU bytes");
  T value;
  std::memcpy(&value, bytes.data_ptr(), sizeof(T));
  return value;
}

torch::Tensor ThinkerMoeParamsOf(
    const torch::Tensor& residual_in, const torch::Tensor& norm,
    const torch::Tensor& router, const torch::Tensor& w13_packed,
    const torch::Tensor& w13_scales, const torch::Tensor& w2_packed,
    const torch::Tensor& w2_scales, double eps, int64_t timeout_ns,
    const torch::Tensor& router_logits, const torch::Tensor& act,
    const torch::Tensor& residual, const torch::Tensor& experts,
    const torch::Tensor& weights, const torch::Tensor& sync,
    const torch::Tensor& error, int64_t num_ctas) {
  constexpr int64_t kD = kThinkerDim, kE = kThinkerExperts,
                    kF = kThinkerExpertFfn, kK = kThinkerTopK;
  // A CTA's gate-up and down rows must fit Shared's 2 × kMaxRowsPerCta ys.
  const int64_t max_rows = std::max((2 * kK * kF + num_ctas - 1) / num_ctas,
                                    kK * ((kD + num_ctas - 1) / num_ctas));
  TORCH_CHECK(0 < num_ctas && num_ctas <= kMaxCtas &&
                  max_rows <= 2 * kMaxRowsPerCta,
              "num_ctas ", num_ctas, " gives a CTA too many rows");
  CheckNumel(residual_in, kD, "residual_in");
  CheckNumel(norm, kD, "norm");
  CheckNumel(router, kE * kD, "router");
  CheckNumel(w13_packed, kE * 2 * kF * kD / 8, "w13_packed");
  CheckNumel(w13_scales, kE * 2 * kF * kD / 32, "w13_scales");
  CheckNumel(w2_packed, kE * kD * kF / 8, "w2_packed");
  CheckNumel(w2_scales, kE * kD * kF / 32, "w2_scales");
  CheckNumel(router_logits, kE, "router_logits");
  CheckNumel(act, kK * kF, "act");
  CheckNumel(residual, kD, "residual");
  CheckNumel(experts, kK, "experts");
  CheckNumel(weights, kK, "weights");
  const auto kBf16 = torch::kBFloat16;
  ThinkerMoeParams p{};
  p.residual_in = Ptr<const float>(residual_in, torch::kFloat32, "residual_in");
  p.norm = Ptr<const __nv_bfloat16>(norm, kBf16, "norm");
  p.router = Ptr<const __nv_bfloat16>(router, kBf16, "router");
  p.w13_packed = Ptr<const int32_t>(w13_packed, torch::kInt32, "w13_packed");
  p.w13_scales = Ptr<const __nv_bfloat16>(w13_scales, kBf16, "w13_scales");
  p.w2_packed = Ptr<const int32_t>(w2_packed, torch::kInt32, "w2_packed");
  p.w2_scales = Ptr<const __nv_bfloat16>(w2_scales, kBf16, "w2_scales");
  p.eps = static_cast<float>(eps);
  p.timeout_ns = timeout_ns;
  p.router_logits = Ptr<float>(router_logits, torch::kFloat32, "router_logits");
  p.act = Ptr<__nv_bfloat16>(act, kBf16, "act");
  p.residual = Ptr<float>(residual, torch::kFloat32, "residual");
  p.experts = Ptr<int32_t>(experts, torch::kInt32, "experts");
  p.weights = Ptr<float>(weights, torch::kFloat32, "weights");
  p.sync = Ptr<unsigned>(sync, torch::kInt32, "sync");
  p.error = MappedRecord(error);
  return StructBytes(p);
}

void RunThinkerMoe(const torch::Tensor& params, int64_t num_ctas) {
  CheckOk(LaunchThinkerMoe(StructFrom<ThinkerMoeParams>(params, "params"),
                           static_cast<int>(num_ctas),
                           at::cuda::getCurrentCUDAStream()),
          "thinker MoE launch");
}

// A layer's paged cache from vLLM's key and value views, [blocks,
// block_size, kv_heads, head_dim] each, in any strides that keep a head's
// values contiguous.
PagedKv PagedKvOf(const torch::Tensor& key_cache,
                  const torch::Tensor& value_cache,
                  const torch::Tensor& block_table,
                  int64_t kv_heads = kThinkerKvHeads) {
  for (const torch::Tensor* t : {&key_cache, &value_cache}) {
    TORCH_CHECK(t->is_cuda() && t->scalar_type() == torch::kBFloat16 &&
                    t->dim() == 4,
                "the KV cache must be 4-D bf16 on the GPU");
    TORCH_CHECK(t->size(2) == kv_heads &&
                    t->size(3) == kThinkerHeadDim && t->stride(3) == 1,
                "the KV cache must be [blocks, block_size, ", kv_heads,
                ", ", kThinkerHeadDim, "] with each head contiguous");
  }
  TORCH_CHECK(key_cache.sizes() == value_cache.sizes() &&
                  key_cache.strides() == value_cache.strides(),
              "key and value caches must share a layout");
  PagedKv kv{};
  kv.key = reinterpret_cast<__nv_bfloat16*>(key_cache.data_ptr());
  kv.value = reinterpret_cast<__nv_bfloat16*>(value_cache.data_ptr());
  kv.block_table = Ptr<const int32_t>(block_table, torch::kInt32, "block_table");
  kv.block_stride = key_cache.stride(0);
  kv.slot_stride = key_cache.stride(1);
  kv.head_stride = key_cache.stride(2);
  kv.block_size = static_cast<int>(key_cache.size(1));
  return kv;
}

torch::Tensor ThinkerAttentionParamsOf(
    const torch::Tensor& residual_in, const torch::Tensor& norm,
    const torch::Tensor& wqkv_packed, const torch::Tensor& wqkv_scales,
    const torch::Tensor& q_norm, const torch::Tensor& k_norm,
    const torch::Tensor& wo_packed, const torch::Tensor& wo_scales,
    const torch::Tensor& cos_sin, const torch::Tensor& positions,
    const torch::Tensor& key_cache, const torch::Tensor& value_cache,
    const torch::Tensor& block_table, int64_t pos, int64_t splits, double eps,
    int64_t timeout_ns, const torch::Tensor& qkv,
    const torch::Tensor& partial_ml, const torch::Tensor& partial_o,
    const torch::Tensor& residual, const torch::Tensor& sync,
    const torch::Tensor& error, int64_t num_ctas) {
  constexpr int64_t kD = kThinkerDim, kQkv = kThinkerQkvRows,
                    kQ = kThinkerQDim, kHd = kThinkerHeadDim;
  const int64_t items = kThinkerQHeads * splits;
  TORCH_CHECK(0 < num_ctas && num_ctas <= kMaxCtas &&
                  (kQkv + num_ctas - 1) / num_ctas <= kMaxRowsPerCta,
              "num_ctas ", num_ctas, " gives a CTA too many rows");
  TORCH_CHECK(splits > 0 && items <= num_ctas, "splits ", splits, " needs ",
              items, " CTAs, have ", num_ctas);
  CheckNumel(residual_in, kD, "residual_in");
  CheckNumel(norm, kD, "norm");
  CheckNumel(wqkv_packed, kQkv * kD / 8, "wqkv_packed");
  CheckNumel(wqkv_scales, kQkv * kD / 32, "wqkv_scales");
  CheckNumel(q_norm, kHd, "q_norm");
  CheckNumel(k_norm, kHd, "k_norm");
  CheckNumel(wo_packed, kD * kQ / 8, "wo_packed");
  CheckNumel(wo_scales, kD * kQ / 32, "wo_scales");
  TORCH_CHECK(cos_sin.dim() == 2 && cos_sin.size(1) == kHd,
              "cos_sin must be [positions, ", kHd, "]");
  CheckNumel(positions, 3, "positions");
  CheckNumel(qkv, kQkv, "qkv");
  CheckNumel(partial_ml, items * 2, "partial_ml");
  CheckNumel(partial_o, items * kHd, "partial_o");
  CheckNumel(residual, kD, "residual");
  const PagedKv kv = PagedKvOf(key_cache, value_cache, block_table);
  TORCH_CHECK(0 <= pos && pos < block_table.numel() * kv.block_size,
              "pos ", pos, " lies past the block table");
  const auto kBf16 = torch::kBFloat16;
  ThinkerAttentionParams p{};
  p.residual_in = Ptr<const float>(residual_in, torch::kFloat32, "residual_in");
  p.norm = Ptr<const __nv_bfloat16>(norm, kBf16, "norm");
  p.wqkv_packed = Ptr<const int32_t>(wqkv_packed, torch::kInt32, "wqkv_packed");
  p.wqkv_scales = Ptr<const __nv_bfloat16>(wqkv_scales, kBf16, "wqkv_scales");
  p.q_norm = Ptr<const __nv_bfloat16>(q_norm, kBf16, "q_norm");
  p.k_norm = Ptr<const __nv_bfloat16>(k_norm, kBf16, "k_norm");
  p.wo_packed = Ptr<const int32_t>(wo_packed, torch::kInt32, "wo_packed");
  p.wo_scales = Ptr<const __nv_bfloat16>(wo_scales, kBf16, "wo_scales");
  p.cos_sin = Ptr<const __nv_bfloat16>(cos_sin, kBf16, "cos_sin");
  p.positions = Ptr<const int32_t>(positions, torch::kInt32, "positions");
  p.kv = kv;
  p.pos = static_cast<int>(pos);
  p.splits = static_cast<int>(splits);
  p.eps = static_cast<float>(eps);
  p.timeout_ns = timeout_ns;
  p.qkv = Ptr<float>(qkv, torch::kFloat32, "qkv");
  p.partial_ml = Ptr<float>(partial_ml, torch::kFloat32, "partial_ml");
  p.partial_o = Ptr<float>(partial_o, torch::kFloat32, "partial_o");
  p.residual = Ptr<float>(residual, torch::kFloat32, "residual");
  p.sync = Ptr<unsigned>(sync, torch::kInt32, "sync");
  p.error = MappedRecord(error);
  return StructBytes(p);
}

void RunThinkerAttention(const torch::Tensor& params, int64_t num_ctas) {
  CheckOk(LaunchThinkerAttention(
              StructFrom<ThinkerAttentionParams>(params, "params"),
              static_cast<int>(num_ctas), at::cuda::getCurrentCUDAStream()),
          "thinker attention launch");
}

// Every layer's block params stacked on the device, [num_layers, bytes].
template <typename T>
const T* StackedParams(const torch::Tensor& stacked, const char* name) {
  TORCH_CHECK(stacked.is_cuda() && stacked.scalar_type() == torch::kUInt8 &&
                  stacked.is_contiguous() && stacked.dim() == 2 &&
                  stacked.size(1) == sizeof(T),
              name, " must be [layers, ", sizeof(T), "] uint8 on the GPU");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(stacked.data_ptr()) % alignof(T) == 0,
              name, " is misaligned");
  return reinterpret_cast<const T*>(stacked.data_ptr());
}

void RunThinkerDecode(
    const torch::Tensor& attention, const torch::Tensor& moe,
    const torch::Tensor& seq_len, const torch::Tensor& slot_mapping,
    const torch::Tensor& positions, const torch::Tensor& final_norm,
    const std::optional<torch::Tensor>& lm_head, double eps, bool prefetch,
    int64_t timeout_ns, const torch::Tensor& residual,
    const std::optional<torch::Tensor>& logits,
    const std::optional<torch::Tensor>& final_hidden,
    const std::optional<torch::Tensor>& hidden,
    const std::optional<torch::Tensor>& profile, const torch::Tensor& sync,
    const torch::Tensor& error, int64_t num_ctas) {
  constexpr int64_t kD = kThinkerDim;
  TORCH_CHECK(attention.size(0) == moe.size(0),
              "attention and MoE params must cover the same layers");
  TORCH_CHECK(0 < num_ctas && num_ctas <= kMaxCtas, "num_ctas ", num_ctas,
              " out of range");
  const int64_t num_layers = attention.size(0);
  TORCH_CHECK(seq_len.is_cuda() && seq_len.scalar_type() == torch::kInt32 &&
                  seq_len.numel() >= 1,
              "seq_len must hold an int32 on the GPU");
  TORCH_CHECK(slot_mapping.is_cuda() &&
                  slot_mapping.scalar_type() == torch::kInt64 &&
                  slot_mapping.numel() >= 1,
              "slot_mapping must hold an int64 on the GPU");
  TORCH_CHECK(positions.is_cuda() && positions.scalar_type() == torch::kInt64 &&
                  positions.dim() == 2 && positions.size(0) == 3 &&
                  positions.size(1) >= 1,
              "positions must be int64 [3, tokens] on the GPU");
  CheckNumel(final_norm, kD, "final_norm");
  CheckNumel(residual, kD, "residual");
  ThinkerDecodeParams p{};
  p.attention = StackedParams<ThinkerAttentionParams>(attention, "attention");
  p.moe = StackedParams<ThinkerMoeParams>(moe, "moe");
  p.num_layers = static_cast<int>(num_layers);
  p.seq_len = reinterpret_cast<const int32_t*>(seq_len.data_ptr());
  p.slot_mapping = reinterpret_cast<const int64_t*>(slot_mapping.data_ptr());
  p.positions = reinterpret_cast<const int64_t*>(positions.data_ptr());
  p.positions_stride = positions.stride(0);
  p.final_norm =
      Ptr<const __nv_bfloat16>(final_norm, torch::kBFloat16, "final_norm");
  TORCH_CHECK(lm_head.has_value() == logits.has_value(),
              "lm_head and logits come together");
  if (lm_head.has_value()) {
    TORCH_CHECK(lm_head->dim() == 2 && lm_head->size(1) == kD,
                "lm_head must be [vocab, ", kD, "]");
    CheckNumel(*logits, lm_head->size(0), "logits");
    p.lm_head =
        Ptr<const __nv_bfloat16>(*lm_head, torch::kBFloat16, "lm_head");
    p.vocab = static_cast<int>(lm_head->size(0));
    p.logits = Ptr<float>(*logits, torch::kFloat32, "logits");
  }
  if (final_hidden.has_value()) {
    CheckNumel(*final_hidden, kD, "final_hidden");
    p.final_hidden =
        Ptr<__nv_bfloat16>(*final_hidden, torch::kBFloat16, "final_hidden");
  }
  p.prefetch = prefetch ? 1 : 0;
  p.eps = static_cast<float>(eps);
  p.timeout_ns = timeout_ns;
  p.residual = Ptr<float>(residual, torch::kFloat32, "residual");
  if (hidden.has_value()) {
    CheckNumel(*hidden, (num_layers + 1) * kD, "hidden");
    p.hidden = Ptr<float>(*hidden, torch::kFloat32, "hidden");
  }
  if (profile.has_value()) {
    CheckNumel(*profile, num_ctas * num_layers * kThinkerBarriersPerLayer * 2,
               "profile");
    p.profile = Ptr<int64_t>(*profile, torch::kInt64, "profile");
  }
  p.sync = Ptr<unsigned>(sync, torch::kInt32, "sync");
  p.error = MappedRecord(error);
  const c10::cuda::CUDAGuard guard(residual.device());
  CheckOk(LaunchThinkerDecode(p, static_cast<int>(num_ctas),
                              at::cuda::getCurrentCUDAStream()),
          "thinker decode launch");
}

torch::Tensor TalkerLayerParamsOf(
    const torch::Tensor& norm, const torch::Tensor& wqkv,
    const torch::Tensor& q_norm, const torch::Tensor& k_norm,
    const torch::Tensor& wo, const torch::Tensor& key_cache,
    const torch::Tensor& value_cache, const torch::Tensor& block_table,
    const torch::Tensor& moe_norm, const torch::Tensor& router,
    const torch::Tensor& w13, const torch::Tensor& w2,
    const torch::Tensor& shared_w13, const torch::Tensor& shared_w2,
    const torch::Tensor& shared_gate) {
  constexpr int64_t kD = kTalkerDim, kE = kTalkerExperts;
  const auto kBf16 = torch::kBFloat16;
  CheckNumel(norm, kD, "norm");
  CheckNumel(wqkv, int64_t{kTalkerQkvRows} * kD, "wqkv");
  CheckNumel(q_norm, kTalkerHeadDim, "q_norm");
  CheckNumel(k_norm, kTalkerHeadDim, "k_norm");
  CheckNumel(wo, kD * kTalkerQDim, "wo");
  CheckNumel(moe_norm, kD, "moe_norm");
  CheckNumel(router, kE * kD, "router");
  CheckNumel(w13, kE * 2 * kTalkerExpertFfn * kD, "w13");
  CheckNumel(w2, kE * kD * kTalkerExpertFfn, "w2");
  CheckNumel(shared_w13, 2 * kTalkerSharedFfn * kD, "shared_w13");
  CheckNumel(shared_w2, kD * kTalkerSharedFfn, "shared_w2");
  CheckNumel(shared_gate, kD, "shared_gate");
  TalkerLayerParams p{};
  p.norm = Ptr<const __nv_bfloat16>(norm, kBf16, "norm");
  p.wqkv = Ptr<const __nv_bfloat16>(wqkv, kBf16, "wqkv");
  p.q_norm = Ptr<const __nv_bfloat16>(q_norm, kBf16, "q_norm");
  p.k_norm = Ptr<const __nv_bfloat16>(k_norm, kBf16, "k_norm");
  p.wo = Ptr<const __nv_bfloat16>(wo, kBf16, "wo");
  p.kv = PagedKvOf(key_cache, value_cache, block_table, kTalkerKvHeads);
  p.moe_norm = Ptr<const __nv_bfloat16>(moe_norm, kBf16, "moe_norm");
  p.router = Ptr<const __nv_bfloat16>(router, kBf16, "router");
  p.w13 = Ptr<const __nv_bfloat16>(w13, kBf16, "w13");
  p.w2 = Ptr<const __nv_bfloat16>(w2, kBf16, "w2");
  p.shared_w13 = Ptr<const __nv_bfloat16>(shared_w13, kBf16, "shared_w13");
  p.shared_w2 = Ptr<const __nv_bfloat16>(shared_w2, kBf16, "shared_w2");
  p.shared_gate = Ptr<const __nv_bfloat16>(shared_gate, kBf16, "shared_gate");
  return StructBytes(p);
}

void RunTalkerDecode(
    const torch::Tensor& layers, const torch::Tensor& seq_len,
    const torch::Tensor& slot_mapping, const torch::Tensor& positions,
    const torch::Tensor& cos_sin, const torch::Tensor& final_norm,
    int64_t splits, double eps, int64_t timeout_ns,
    const torch::Tensor& residual, const torch::Tensor& final_hidden,
    const torch::Tensor& qkv, const torch::Tensor& partial_ml,
    const torch::Tensor& partial_o, const torch::Tensor& router_logits,
    const torch::Tensor& act, const std::optional<torch::Tensor>& hidden,
    const std::optional<torch::Tensor>& profile, const torch::Tensor& sync,
    const torch::Tensor& error, int64_t num_ctas) {
  constexpr int64_t kD = kTalkerDim;
  const int64_t items = kTalkerQHeads * splits;
  TORCH_CHECK(0 < num_ctas && num_ctas <= kMaxCtas &&
                  (kTalkerQkvRows + num_ctas - 1) / num_ctas <= kMaxRowsPerCta &&
                  2 * ((kTalkerTopK * kTalkerExpertFfn + num_ctas - 1) / num_ctas) <=
                      kMaxRowsPerCta &&
                  (kTalkerTopK + 1) * ((kD + num_ctas - 1) / num_ctas) <=
                      kMaxRowsPerCta,
              "num_ctas ", num_ctas, " gives a CTA too many rows");
  TORCH_CHECK(splits > 0 && items <= num_ctas, "splits ", splits, " needs ",
              items, " CTAs, have ", num_ctas);
  TORCH_CHECK(seq_len.is_cuda() && seq_len.scalar_type() == torch::kInt32 &&
                  seq_len.numel() >= 1,
              "seq_len must hold an int32 on the GPU");
  TORCH_CHECK(slot_mapping.is_cuda() &&
                  slot_mapping.scalar_type() == torch::kInt64 &&
                  slot_mapping.numel() >= 1,
              "slot_mapping must hold an int64 on the GPU");
  TORCH_CHECK(positions.is_cuda() && positions.scalar_type() == torch::kInt64 &&
                  positions.dim() == 2 && positions.size(0) == 3 &&
                  positions.size(1) >= 1,
              "positions must be int64 [3, tokens] on the GPU");
  TORCH_CHECK(cos_sin.dim() == 2 && cos_sin.size(1) == kTalkerHeadDim,
              "cos_sin must be [positions, ", kTalkerHeadDim, "]");
  CheckNumel(final_norm, kD, "final_norm");
  CheckNumel(residual, kD, "residual");
  CheckNumel(final_hidden, kD, "final_hidden");
  CheckNumel(qkv, kTalkerQkvRows, "qkv");
  CheckNumel(partial_ml, items * 2, "partial_ml");
  CheckNumel(partial_o, items * kTalkerHeadDim, "partial_o");
  CheckNumel(router_logits, kTalkerExperts + 1, "router_logits");
  CheckNumel(act, kTalkerTopK * kTalkerExpertFfn + kTalkerSharedFfn, "act");
  TalkerDecodeParams p{};
  p.layers = StackedParams<TalkerLayerParams>(layers, "layers");
  p.num_layers = static_cast<int>(layers.size(0));
  p.seq_len = reinterpret_cast<const int32_t*>(seq_len.data_ptr());
  p.slot_mapping = reinterpret_cast<const int64_t*>(slot_mapping.data_ptr());
  p.positions = reinterpret_cast<const int64_t*>(positions.data_ptr());
  p.positions_stride = positions.stride(0);
  const auto kBf16 = torch::kBFloat16;
  p.cos_sin = Ptr<const __nv_bfloat16>(cos_sin, kBf16, "cos_sin");
  p.final_norm = Ptr<const __nv_bfloat16>(final_norm, kBf16, "final_norm");
  p.splits = static_cast<int>(splits);
  p.eps = static_cast<float>(eps);
  p.timeout_ns = timeout_ns;
  p.residual = Ptr<float>(residual, torch::kFloat32, "residual");
  p.final_hidden = Ptr<__nv_bfloat16>(final_hidden, kBf16, "final_hidden");
  p.qkv = Ptr<float>(qkv, torch::kFloat32, "qkv");
  p.partial_ml = Ptr<float>(partial_ml, torch::kFloat32, "partial_ml");
  p.partial_o = Ptr<float>(partial_o, torch::kFloat32, "partial_o");
  p.router_logits = Ptr<float>(router_logits, torch::kFloat32, "router_logits");
  p.act = Ptr<__nv_bfloat16>(act, kBf16, "act");
  if (hidden.has_value()) {
    CheckNumel(*hidden, (p.num_layers + 1) * kD, "hidden");
    p.hidden = Ptr<float>(*hidden, torch::kFloat32, "hidden");
  }
  if (profile.has_value()) {
    CheckNumel(*profile, num_ctas * p.num_layers * kTalkerBarriersPerLayer * 2,
               "profile");
    p.profile = Ptr<int64_t>(*profile, torch::kInt64, "profile");
  }
  p.sync = Ptr<unsigned>(sync, torch::kInt32, "sync");
  p.error = MappedRecord(error);
  const c10::cuda::CUDAGuard guard(residual.device());
  CheckOk(LaunchTalkerDecode(p, static_cast<int>(num_ctas),
                             at::cuda::getCurrentCUDAStream()),
          "talker decode launch");
}

void RunThinkerPrefill(
    const torch::Tensor& attention, const torch::Tensor& moe, int64_t tokens,
    const torch::Tensor& seq_len, const torch::Tensor& slot_mapping,
    const torch::Tensor& positions, const torch::Tensor& final_norm, double eps,
    int64_t timeout_ns, const torch::Tensor& residual, const torch::Tensor& h,
    const torch::Tensor& qkv, const torch::Tensor& attn,
    const torch::Tensor& kv, const torch::Tensor& router_logits, const torch::Tensor& act,
    const torch::Tensor& partial, const torch::Tensor& experts,
    const torch::Tensor& weights, const torch::Tensor& final_hidden,
    const std::optional<torch::Tensor>& hidden, int64_t hidden_layer,
    const std::optional<torch::Tensor>& profile, const torch::Tensor& sync,
    const torch::Tensor& error, int64_t num_ctas) {
  constexpr int64_t kD = kThinkerDim, kK = kThinkerTopK;
  TORCH_CHECK(attention.size(0) == moe.size(0),
              "attention and MoE params must cover the same layers");
  const int64_t layers = attention.size(0);
  TORCH_CHECK(0 < tokens && tokens <= kThinkerPrefillMaxTokens, "tokens ",
              tokens, " out of [1, ", kThinkerPrefillMaxTokens, "]");
  TORCH_CHECK(0 < num_ctas && num_ctas <= kMaxCtas &&
                  (kThinkerQkvRows + num_ctas - 1) / num_ctas <= 256,
              "num_ctas ", num_ctas, " gives a CTA more than 256 rows");
  TORCH_CHECK(seq_len.is_cuda() && seq_len.scalar_type() == torch::kInt32,
              "seq_len must be int32 on the GPU");
  TORCH_CHECK(slot_mapping.is_cuda() &&
                  slot_mapping.scalar_type() == torch::kInt64 &&
                  slot_mapping.numel() >= tokens,
              "slot_mapping must hold tokens int64 on the GPU");
  TORCH_CHECK(positions.is_cuda() && positions.scalar_type() == torch::kInt64 &&
                  positions.dim() == 2 && positions.size(0) == 3 &&
                  positions.size(1) >= tokens && positions.stride(1) == 1,
              "positions must be int64 [3, tokens] on the GPU");
  CheckNumel(final_norm, kD, "final_norm");
  TORCH_CHECK(residual.numel() >= tokens * kD && h.numel() >= tokens * kD &&
                  qkv.numel() >= tokens * kThinkerQkvRows &&
                  attn.numel() >= tokens * kThinkerQDim &&
                  kv.numel() >= tokens * 2 * kThinkerKvHeads * kThinkerHeadDim &&
                  router_logits.numel() >= tokens * kThinkerExperts &&
                  act.numel() >= tokens * kK * kThinkerExpertFfn &&
                  partial.numel() >= tokens * kK * kD &&
                  experts.numel() >= layers * tokens * kK &&
                  weights.numel() >= layers * tokens * kK &&
                  final_hidden.numel() >= tokens * kD,
              "a prefill buffer is too small for ", tokens, " tokens");
  ThinkerPrefillParams p{};
  p.attention = StackedParams<ThinkerAttentionParams>(attention, "attention");
  p.moe = StackedParams<ThinkerMoeParams>(moe, "moe");
  p.num_layers = static_cast<int>(layers);
  p.tokens = static_cast<int>(tokens);
  p.seq_len = reinterpret_cast<const int32_t*>(seq_len.data_ptr());
  p.slot_mapping = reinterpret_cast<const int64_t*>(slot_mapping.data_ptr());
  p.positions = reinterpret_cast<const int64_t*>(positions.data_ptr());
  p.positions_stride = positions.stride(0);
  p.final_norm =
      Ptr<const __nv_bfloat16>(final_norm, torch::kBFloat16, "final_norm");
  p.eps = static_cast<float>(eps);
  p.timeout_ns = timeout_ns;
  p.residual = Ptr<float>(residual, torch::kFloat32, "residual");
  p.h = Ptr<__nv_bfloat16>(h, torch::kBFloat16, "h");
  p.qkv = Ptr<float>(qkv, torch::kFloat32, "qkv");
  p.attn = Ptr<__nv_bfloat16>(attn, torch::kBFloat16, "attn");
  p.kv = Ptr<__nv_bfloat16>(kv, torch::kBFloat16, "kv");
  p.router_logits = Ptr<float>(router_logits, torch::kFloat32, "router_logits");
  p.act = Ptr<__nv_bfloat16>(act, torch::kBFloat16, "act");
  p.partial = Ptr<float>(partial, torch::kFloat32, "partial");
  p.experts = Ptr<int32_t>(experts, torch::kInt32, "experts");
  p.weights = Ptr<float>(weights, torch::kFloat32, "weights");
  p.final_hidden =
      Ptr<__nv_bfloat16>(final_hidden, torch::kBFloat16, "final_hidden");
  if (hidden.has_value()) {
    TORCH_CHECK(hidden->numel() >= tokens * kD, "hidden is too small");
    p.hidden = Ptr<float>(*hidden, torch::kFloat32, "hidden");
  }
  p.hidden_layer = static_cast<int>(hidden_layer);
  if (profile.has_value()) {
    CheckNumel(*profile,
               num_ctas * layers * kThinkerPrefillBarriersPerLayer * 2,
               "profile");
    p.profile = Ptr<int64_t>(*profile, torch::kInt64, "profile");
  }
  p.sync = Ptr<unsigned>(sync, torch::kInt32, "sync");
  p.error = MappedRecord(error);
  const c10::cuda::CUDAGuard guard(residual.device());
  CheckOk(LaunchThinkerPrefill(p, static_cast<int>(num_ctas),
                               at::cuda::getCurrentCUDAStream()),
          "thinker prefill launch");
}

}  // namespace
}  // namespace s2mk

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  namespace py = pybind11;
  m.def("thinker_moe_params", &s2mk::ThinkerMoeParamsOf,
        "The thinker MoE block's launch params, as CPU bytes.",
        py::arg("residual_in"), py::arg("norm"), py::arg("router"),
        py::arg("w13_packed"), py::arg("w13_scales"), py::arg("w2_packed"),
        py::arg("w2_scales"), py::arg("eps"), py::arg("timeout_ns"),
        py::arg("router_logits"), py::arg("act"), py::arg("residual"),
        py::arg("experts"), py::arg("weights"), py::arg("sync"),
        py::arg("error"), py::arg("num_ctas"));
  m.def("thinker_attention_params", &s2mk::ThinkerAttentionParamsOf,
        "The thinker attention block's launch params, as CPU bytes.",
        py::arg("residual_in"), py::arg("norm"), py::arg("wqkv_packed"),
        py::arg("wqkv_scales"), py::arg("q_norm"), py::arg("k_norm"),
        py::arg("wo_packed"), py::arg("wo_scales"), py::arg("cos_sin"),
        py::arg("positions"), py::arg("key_cache"), py::arg("value_cache"),
        py::arg("block_table"), py::arg("pos"), py::arg("splits"),
        py::arg("eps"), py::arg("timeout_ns"), py::arg("qkv"),
        py::arg("partial_ml"), py::arg("partial_o"), py::arg("residual"),
        py::arg("sync"), py::arg("error"), py::arg("num_ctas"));
  m.def("run_thinker_moe", &s2mk::RunThinkerMoe,
        "One token through Qwen3-Omni's thinker MoE block, one launch.",
        py::arg("params"), py::arg("num_ctas"));
  m.def("run_thinker_attention", &s2mk::RunThinkerAttention,
        "One token through Qwen3-Omni's thinker attention block, one launch.",
        py::arg("params"), py::arg("num_ctas"));
  m.def("run_thinker_decode", &s2mk::RunThinkerDecode,
        "One decode step of Qwen3-Omni's thinker, one launch.",
        py::arg("attention"), py::arg("moe"), py::arg("seq_len"),
        py::arg("slot_mapping"), py::arg("positions"), py::arg("final_norm"),
        py::arg("lm_head"), py::arg("eps"), py::arg("prefetch"),
        py::arg("timeout_ns"), py::arg("residual"), py::arg("logits"),
        py::arg("final_hidden"), py::arg("hidden"), py::arg("profile"),
        py::arg("sync"), py::arg("error"), py::arg("num_ctas"));
  m.def("run_thinker_prefill", &s2mk::RunThinkerPrefill,
        "One prefill chunk of Qwen3-Omni's thinker, one launch.",
        py::arg("attention"), py::arg("moe"), py::arg("tokens"),
        py::arg("seq_len"), py::arg("slot_mapping"), py::arg("positions"),
        py::arg("final_norm"), py::arg("eps"), py::arg("timeout_ns"),
        py::arg("residual"), py::arg("h"), py::arg("qkv"), py::arg("attn"),
        py::arg("kv"), py::arg("router_logits"), py::arg("act"),
        py::arg("partial"), py::arg("experts"), py::arg("weights"),
        py::arg("final_hidden"), py::arg("hidden"), py::arg("hidden_layer"),
        py::arg("profile"),
        py::arg("sync"), py::arg("error"), py::arg("num_ctas"));
  m.def("run_code_predictor", &s2mk::RunCodePredictor,
        "Runs N frames of Qwen3-Omni's code predictor in one launch.",
        py::arg("layers"), py::arg("final_norm"), py::arg("heads"),
        py::arg("embeddings"), py::arg("rope"), py::arg("k_caches"),
        py::arg("v_caches"), py::arg("eps"), py::arg("top_k"),
        py::arg("top_p"), py::arg("talker_hidden"), py::arg("code0_embed"),
        py::arg("uniforms"), py::arg("forced_codes"),
        py::arg("prefetch_bytes"), py::arg("timeout_ns"), py::arg("residual"),
        py::arg("qkv"), py::arg("act"), py::arg("logits"), py::arg("codes"),
        py::arg("dump"), py::arg("profile"), py::arg("sync"), py::arg("error"),
        py::arg("num_ctas"), py::arg("cooperative"));
  m.def("talker_layer_params", &s2mk::TalkerLayerParamsOf,
        "A talker layer's weights and cache for the decode kernel, as CPU bytes.",
        py::arg("norm"), py::arg("wqkv"), py::arg("q_norm"), py::arg("k_norm"),
        py::arg("wo"), py::arg("key_cache"), py::arg("value_cache"),
        py::arg("block_table"), py::arg("moe_norm"), py::arg("router"),
        py::arg("w13"), py::arg("w2"), py::arg("shared_w13"),
        py::arg("shared_w2"), py::arg("shared_gate"));
  m.def("run_talker_decode", &s2mk::RunTalkerDecode,
        "One decode step of Qwen3-Omni's talker, one launch.",
        py::arg("layers"), py::arg("seq_len"), py::arg("slot_mapping"),
        py::arg("positions"), py::arg("cos_sin"), py::arg("final_norm"),
        py::arg("splits"), py::arg("eps"), py::arg("timeout_ns"),
        py::arg("residual"), py::arg("final_hidden"), py::arg("qkv"),
        py::arg("partial_ml"), py::arg("partial_o"), py::arg("router_logits"),
        py::arg("act"), py::arg("hidden"), py::arg("profile"), py::arg("sync"),
        py::arg("error"), py::arg("num_ctas"));
  m.attr("ERROR_RECORD_WORDS") =
      py::int_(sizeof(s2mk::ErrorRecord) / sizeof(int32_t));
  m.attr("SYNC_WORDS") = py::int_(s2mk::kSyncWords);
}
