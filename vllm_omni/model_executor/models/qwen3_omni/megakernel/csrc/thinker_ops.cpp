// Copyright 2026 Fractalyze Inc. All rights reserved.
//
// The Python bindings of the thinker megakernels: the thinker slice of
// decode-mk's s2mk/csrc/ops.cpp.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cstring>
#include <optional>

#include "barrier.h"
#include "layer.h"
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

void CheckNumel(const torch::Tensor& t, int64_t numel, const char* name) {
  TORCH_CHECK(t.numel() == numel, name, " must hold ", numel,
              " elements, got ", t.numel());
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
  m.attr("ERROR_RECORD_WORDS") =
      py::int_(sizeof(s2mk::ErrorRecord) / sizeof(int32_t));
  m.attr("SYNC_WORDS") = py::int_(s2mk::kSyncWords);
}
