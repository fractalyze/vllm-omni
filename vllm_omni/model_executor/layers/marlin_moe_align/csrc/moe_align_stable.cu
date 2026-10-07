// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
// Copyright 2026 Fractalyze Inc. All rights reserved.
//
// Provenance: vllm-project/vllm PR #48032 @ a718a4b1 (Apache-2.0, Copyright
// contributors to the vLLM project), "Make Marlin MoE route alignment
// deterministic".
// - Verbatim: the four kernels from
//   csrc/libtorch_stable/moe/moe_align_sum_kernels.cu (lines 402-569:
//   stable_small_route_align_kernel, prepare_radix_sort_keys_kernel,
//   copy_compact_tokens_to_padded_kernel,
//   build_padded_metadata_from_offsets_kernel), and CubKeyValueSorter plus
//   computeExpertOffsetsAndInverse from
//   csrc/libtorch_stable/moe/permute_unpermute_kernels/
//   moe_permute_unpermute_kernel.{h,cu} (lines 8-119 of the .cu).
// - Rewritten from torch::stable to ATen: the host entry points
//   moe_permute_sort_workspace_size, moe_align_block_size_stable_small (the
//   PR's moe_align_block_size_impl with stable_token_order) and
//   moe_align_block_size_radix, with the PR's checks and launch shapes, and
//   pybind bindings in place of its STABLE_TORCH_LIBRARY registration.
//
// Removed once the pinned vLLM ships #48032; vllm_omni/patch.py has the
// switch.

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

// From moe_align_sum_kernels.cu line 21.
#define CEILDIV(x, y) (((x) + (y) - 1) / (y))

// ---- verbatim: moe_permute_unpermute_kernel.h (CubKeyValueSorter decl) ----
class CubKeyValueSorter {
 public:
  CubKeyValueSorter();

  CubKeyValueSorter(int const num_experts);

  void updateNumExperts(int const num_experts);

  static size_t getWorkspaceSize(size_t const num_key_value_pairs,
                                 int const num_experts);

  void run(void* workspace, size_t const workspace_size, int const* keys_in,
           int* keys_out, int const* values_in, int* values_out,
           size_t const num_key_value_pairs, cudaStream_t stream);

 private:
  static int expertsToBits(int experts);
  int num_experts_;
  int num_bits_;
};
// ---- verbatim: moe_permute_unpermute_kernel.cu (lines 8-119) ----

// CubKeyValueSorter definition begin
CubKeyValueSorter::CubKeyValueSorter()
    : num_experts_(0), num_bits_(sizeof(int) * 8) {}

int CubKeyValueSorter::expertsToBits(int num_experts) {
  // Max value we represent is V = num_experts + (num_experts - 1) = 2 *
  // num_experts - 1 The maximum number of bits is therefore floor(log2(V)) + 1
  return static_cast<int>(log2(2 * num_experts - 1)) + 1;
}

CubKeyValueSorter::CubKeyValueSorter(int const num_experts)
    : num_experts_(num_experts), num_bits_(expertsToBits(num_experts)) {}

void CubKeyValueSorter::updateNumExperts(int const num_experts) {
  num_experts_ = num_experts;
  num_bits_ = expertsToBits(num_experts);
}

size_t CubKeyValueSorter::getWorkspaceSize(size_t const num_key_value_pairs,
                                           int const num_experts) {
  int num_bits = expertsToBits(num_experts);
  size_t required_storage = 0;
  int* null_int = nullptr;
  cub::DeviceRadixSort::SortPairs(nullptr, required_storage, null_int, null_int,
                                  null_int, null_int, num_key_value_pairs, 0,
                                  num_bits);

  //   when num_key_value_pairs, num_experts, num_bits, required_storage = 64,
  //   4, 3, 0 The required_storage seems to vary between 0 and 1 for the same
  //   inputs
  if (required_storage == 0) {
    required_storage = 1;
  }
  return required_storage;
}

void CubKeyValueSorter::run(void* workspace, size_t const workspace_size,
                            int const* keys_in, int* keys_out,
                            int const* values_in, int* values_out,
                            size_t const num_key_value_pairs,
                            cudaStream_t stream) {
  size_t expected_ws_size = getWorkspaceSize(num_key_value_pairs, num_experts_);
  size_t actual_ws_size = workspace_size;

  STD_TORCH_CHECK(
      expected_ws_size <= workspace_size,
      "[CubKeyValueSorter::run] The allocated workspace is too small "
      "to run this problem.");
  cub::DeviceRadixSort::SortPairs(workspace, actual_ws_size, keys_in, keys_out,
                                  values_in, values_out, num_key_value_pairs, 0,
                                  num_bits_, stream);
}
// CubKeyValueSorter definition end

static inline size_t pad_to_multiple_of_16(size_t const& input) {
  static constexpr int ALIGNMENT = 16;
  return ALIGNMENT * ((input + ALIGNMENT - 1) / ALIGNMENT);
}
template <class T>
__device__ inline int64_t findTotalEltsLessThanTarget(T const* sorted_indices,
                                                      int64_t const arr_length,
                                                      T const target) {
  int64_t low = 0, high = arr_length - 1, target_location = -1;
  while (low <= high) {
    int64_t mid = (low + high) / 2;

    if (sorted_indices[mid] >= target) {
      high = mid - 1;
    } else {
      low = mid + 1;
      target_location = mid;
    }
  }
  return target_location + 1;
}

// Computes expert offsets (ending with the valid row count) and optional
// inverse indices.
__global__ void computeExpertOffsetsAndInverseKernel(
    int const* sorted_experts, int64_t const sorted_experts_len,
    int const num_experts, int64_t* expert_first_token_offset,
    int const* sorted_rows, int* inverse) {
  int const index = blockIdx.x * blockDim.x + threadIdx.x;

  // Note that expert goes [0, num_experts] (inclusive) because we want a count
  // for the total number of active tokens at the end of the scan.
  if (index < num_experts + 1) {
    expert_first_token_offset[index] =
        findTotalEltsLessThanTarget(sorted_experts, sorted_experts_len, index);
  }
  if (inverse != nullptr && index < sorted_experts_len) {
    inverse[sorted_rows[index]] = index;
  }
}

void computeExpertOffsetsAndInverse(int const* sorted_indices,
                                    int const total_indices,
                                    int const num_experts,
                                    int64_t* expert_first_token_offset,
                                    int const* sorted_rows, int* inverse,
                                    cudaStream_t stream) {
  int const num_entries = inverse != nullptr
                              ? std::max(num_experts + 1, total_indices)
                              : num_experts + 1;
  int const threads = inverse != nullptr ? 256 : std::min(1024, num_entries);
  int const blocks = (num_entries + threads - 1) / threads;

  computeExpertOffsetsAndInverseKernel<<<blocks, threads, 0, stream>>>(
      sorted_indices, total_indices, num_experts, expert_first_token_offset,
      sorted_rows, inverse);
}
// ---- verbatim: moe_align_sum_kernels.cu (lines 402-569) ----
namespace vllm { namespace moe {
// Decode-sized inputs fit in one block. Shared-memory counts and offsets build
// the padded expert ranges; comparing only earlier routes gives a stable rank.
// The O(num_routes^2 + num_experts) work is explicitly capped by max_routes.
template <typename scalar_t, int32_t max_routes>
__global__ void stable_small_route_align_kernel(
    const scalar_t* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_token_ids, int32_t* __restrict__ expert_ids,
    int32_t* __restrict__ total_tokens_post_pad,
    const int32_t* __restrict__ expert_map, int32_t num_experts,
    int32_t block_size, int32_t numel, int32_t max_num_tokens_padded,
    int32_t max_num_m_blocks, bool has_expert_map) {
  __shared__ int32_t expert_counts[1024];
  __shared__ int32_t expert_offsets[1024];
  __shared__ int32_t route_experts[max_routes];
  using BlockScan = cub::BlockScan<int32_t, 1024>;
  __shared__ typename BlockScan::TempStorage scan_storage;

  const int32_t tid = threadIdx.x;
  for (int32_t index = tid; index < max_num_tokens_padded;
       index += blockDim.x) {
    sorted_token_ids[index] = numel;
  }
  if (tid < num_experts) {
    expert_counts[tid] = 0;
  }
  __syncthreads();

  if (tid < numel) {
    const int32_t global_expert_id = static_cast<int32_t>(topk_ids[tid]);
    int32_t output_expert_id = -1;
    if (global_expert_id >= 0 && global_expert_id < num_experts) {
      output_expert_id =
          has_expert_map ? expert_map[global_expert_id] : global_expert_id;
    }
    route_experts[tid] = output_expert_id;
    if (output_expert_id >= 0) {
      atomicAdd(&expert_counts[output_expert_id], 1);
    }
  }
  __syncthreads();

  int32_t padded_count = 0;
  if (tid < num_experts) {
    padded_count = CEILDIV(expert_counts[tid], block_size) * block_size;
  }
  int32_t padded_offset;
  BlockScan(scan_storage).ExclusiveSum(padded_count, padded_offset);
  if (tid <= num_experts) {
    expert_offsets[tid] = padded_offset;
  }
  if (tid == num_experts) {
    total_tokens_post_pad[0] = padded_offset;
  }
  __syncthreads();

  if (tid < num_experts) {
    for (int32_t index = expert_offsets[tid]; index < expert_offsets[tid + 1];
         index += block_size) {
      expert_ids[index / block_size] = tid;
    }
  }
  const int32_t first_inactive_block = expert_offsets[num_experts] / block_size;
  for (int32_t index = first_inactive_block + tid; index < max_num_m_blocks;
       index += blockDim.x) {
    expert_ids[index] = -1;
  }

  if (tid < numel) {
    const int32_t output_expert_id = route_experts[tid];
    if (output_expert_id >= 0) {
      int32_t rank = 0;
      for (int32_t previous = 0; previous < tid; ++previous) {
        rank += route_experts[previous] == output_expert_id;
      }
      sorted_token_ids[expert_offsets[output_expert_id] + rank] = tid;
    }
  }
}

template <typename scalar_t>
__global__ void prepare_radix_sort_keys_kernel(
    const scalar_t* __restrict__ topk_ids, int32_t* __restrict__ keys,
    const int32_t* __restrict__ expert_map, size_t numel, int32_t num_experts,
    bool has_expert_map) {
  for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < numel;
       index += blockDim.x * gridDim.x) {
    const int32_t expert_id = static_cast<int32_t>(topk_ids[index]);
    if (expert_id < 0 || expert_id >= num_experts) {
      keys[index] = 2 * num_experts - 1;
    } else if (has_expert_map) {
      const int32_t local_expert_id = expert_map[expert_id];
      keys[index] =
          local_expert_id < 0 ? num_experts + expert_id : local_expert_id;
    } else {
      keys[index] = expert_id;
    }
  }
}

__global__ void copy_compact_tokens_to_padded_kernel(
    const int32_t* __restrict__ compact_sorted_token_ids,
    int32_t* __restrict__ sorted_token_ids,
    const int32_t* __restrict__ padded_expert_offsets,
    const int64_t* __restrict__ unpadded_expert_offsets, int32_t num_experts) {
  const int32_t expert_id = blockIdx.x;
  if (expert_id >= num_experts) {
    return;
  }

  const int64_t compact_start = unpadded_expert_offsets[expert_id];
  const int64_t compact_end = unpadded_expert_offsets[expert_id + 1];
  const int32_t padded_start = padded_expert_offsets[expert_id];
  for (int64_t index = compact_start + threadIdx.x; index < compact_end;
       index += blockDim.x) {
    sorted_token_ids[padded_start + index - compact_start] =
        compact_sorted_token_ids[index];
  }
}

__global__ void build_padded_metadata_from_offsets_kernel(
    int32_t* __restrict__ sorted_token_ids, int32_t* __restrict__ expert_ids,
    int32_t* __restrict__ total_tokens_post_pad,
    int32_t* __restrict__ padded_expert_offsets,
    const int64_t* __restrict__ unpadded_expert_offsets, int32_t num_experts,
    int32_t block_size, int32_t numel, int32_t max_num_tokens_padded,
    int32_t max_num_m_blocks) {
  if (blockIdx.x == 1) {
    for (int32_t index = threadIdx.x; index < max_num_tokens_padded;
         index += blockDim.x) {
      sorted_token_ids[index] = numel;
    }
    return;
  }

  using BlockScan = cub::BlockScan<int32_t, 1024>;
  __shared__ typename BlockScan::TempStorage scan_storage;

  const int32_t expert_id = threadIdx.x;
  int32_t padded_count = 0;
  if (expert_id < num_experts) {
    const int64_t count = unpadded_expert_offsets[expert_id + 1] -
                          unpadded_expert_offsets[expert_id];
    padded_count = CEILDIV(count, block_size) * block_size;
  }

  int32_t padded_offset;
  BlockScan(scan_storage).ExclusiveSum(padded_count, padded_offset);
  if (expert_id <= num_experts) {
    padded_expert_offsets[expert_id] = padded_offset;
  }
  if (expert_id == num_experts) {
    total_tokens_post_pad[0] = padded_offset;
  }
  __syncthreads();

  if (expert_id < num_experts) {
    for (int32_t index = padded_expert_offsets[expert_id];
         index < padded_expert_offsets[expert_id + 1]; index += block_size) {
      expert_ids[index / block_size] = expert_id;
    }
  }
  const int32_t fill_start =
      padded_expert_offsets[num_experts] / block_size + threadIdx.x;
  for (int32_t index = fill_start; index < max_num_m_blocks;
       index += blockDim.x) {
    expert_ids[index] = -1;
  }
}
} }  // namespace vllm::moe

// ---- rewritten: the PR's torch::stable host code in ATen ----
#define DISPATCH_IDS(TYPE, NAME, ...)                                 \
  AT_DISPATCH_SWITCH(TYPE, NAME,                                      \
                     AT_DISPATCH_CASE_INTEGRAL_TYPES(__VA_ARGS__)      \
                         AT_DISPATCH_CASE(at::ScalarType::UInt32, __VA_ARGS__))

int64_t moe_permute_sort_workspace_size(int64_t num_expanded_rows,
                                        int64_t n_expert) {
  return static_cast<int64_t>(
      CubKeyValueSorter::getWorkspaceSize(num_expanded_rows, n_expert));
}

void moe_align_block_size_stable_small(
    at::Tensor topk_ids, int64_t num_experts, int64_t block_size,
    at::Tensor sorted_token_ids, at::Tensor experts_ids,
    at::Tensor num_tokens_post_pad, std::optional<at::Tensor> maybe_expert_map) {
  const at::cuda::OptionalCUDAGuard device_guard(topk_ids.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  constexpr int WARP_SIZE = 32;
  int64_t padded_num_experts =
      ((num_experts + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
  STD_TORCH_CHECK(padded_num_experts < 1024,
                  "padded_num_experts must be less than 1024");
  STD_TORCH_CHECK(topk_ids.numel() <= 256,
                  "stable alignment supports at most 256 routed entries; "
                  "use moe_align_block_size_radix for larger inputs");
  bool has_expert_map = maybe_expert_map.has_value();
  at::Tensor expert_map = has_expert_map
                              ? maybe_expert_map.value()
                              : at::empty({0}, topk_ids.options().dtype(at::kInt));
  DISPATCH_IDS(topk_ids.scalar_type(), "moe_align_block_size_kernel", [&] {
    constexpr int32_t stable_small_route_limit = 256;
    vllm::moe::stable_small_route_align_kernel<
        scalar_t, stable_small_route_limit><<<1, 1024, 0, stream>>>(
        reinterpret_cast<const scalar_t*>(topk_ids.const_data_ptr()),
        reinterpret_cast<int32_t*>(sorted_token_ids.mutable_data_ptr()),
        reinterpret_cast<int32_t*>(experts_ids.mutable_data_ptr()),
        reinterpret_cast<int32_t*>(num_tokens_post_pad.mutable_data_ptr()),
        reinterpret_cast<const int32_t*>(expert_map.const_data_ptr()),
        num_experts, block_size, topk_ids.numel(), sorted_token_ids.size(0),
        experts_ids.size(0), has_expert_map);
  });
}

void moe_align_block_size_radix(
    at::Tensor topk_ids, int64_t num_experts, int64_t block_size,
    at::Tensor sorted_token_ids, at::Tensor experts_ids,
    at::Tensor num_tokens_post_pad, at::Tensor sort_workspace,
    at::Tensor sorted_expert_ids, at::Tensor compact_sorted_token_ids,
    at::Tensor token_indices, at::Tensor topk_ids_for_sort,
    at::Tensor padded_expert_offsets, at::Tensor unpadded_expert_offsets,
    std::optional<at::Tensor> maybe_expert_map) {
  const at::cuda::OptionalCUDAGuard device_guard(topk_ids.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int64_t numel = topk_ids.numel();

  STD_TORCH_CHECK(num_experts > 0 && num_experts < 1024,
                  "num_experts must be in [1, 1024)");
  STD_TORCH_CHECK(block_size > 0, "block_size must be positive");
  STD_TORCH_CHECK(numel > 0, "topk_ids must not be empty");
  STD_TORCH_CHECK(numel <= std::numeric_limits<int32_t>::max(),
                  "topk_ids contains too many entries for int32 token indices");
  const int64_t max_padding = num_experts * (block_size - 1);
  STD_TORCH_CHECK(numel <= std::numeric_limits<int32_t>::max() - max_padding,
                  "padded token count exceeds the int32 alignment ABI");
  STD_TORCH_CHECK(topk_ids.is_contiguous(), "topk_ids must be contiguous");

  auto check_scratch = [&](const at::Tensor& tensor, at::ScalarType dtype,
                           const char* name) {
    STD_TORCH_CHECK(tensor.device() == topk_ids.device(), name,
                    " must be on the same device as topk_ids");
    STD_TORCH_CHECK(tensor.scalar_type() == dtype, name,
                    " has an unexpected dtype");
    STD_TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  };
  check_scratch(sort_workspace, at::kChar, "sort_workspace");
  check_scratch(sorted_expert_ids, at::kInt, "sorted_expert_ids");
  check_scratch(compact_sorted_token_ids, at::kInt, "compact_sorted_token_ids");
  check_scratch(token_indices, at::kInt, "token_indices");
  check_scratch(topk_ids_for_sort, at::kInt, "topk_ids_for_sort");
  check_scratch(padded_expert_offsets, at::kInt, "padded_expert_offsets");
  check_scratch(unpadded_expert_offsets, at::kLong, "unpadded_expert_offsets");
  STD_TORCH_CHECK(sorted_expert_ids.numel() >= numel,
                  "sorted_expert_ids is too small");
  STD_TORCH_CHECK(compact_sorted_token_ids.numel() >= numel,
                  "compact_sorted_token_ids is too small");
  STD_TORCH_CHECK(token_indices.numel() >= numel, "token_indices is too small");
  STD_TORCH_CHECK(topk_ids_for_sort.numel() >= numel,
                  "topk_ids_for_sort is too small");
  STD_TORCH_CHECK(padded_expert_offsets.numel() >= num_experts + 1,
                  "padded_expert_offsets is too small");
  STD_TORCH_CHECK(unpadded_expert_offsets.numel() >= num_experts + 1,
                  "unpadded_expert_offsets is too small");

  bool has_expert_map = maybe_expert_map.has_value();
  if (has_expert_map) {
    check_scratch(maybe_expert_map.value(), at::kInt, "expert_map");
    STD_TORCH_CHECK(maybe_expert_map.value().numel() >= num_experts,
                    "expert_map is too small");
  }
  const int32_t* expert_map_ptr =
      has_expert_map ? reinterpret_cast<const int32_t*>(
                           maybe_expert_map.value().const_data_ptr())
                     : nullptr;

  DISPATCH_IDS(topk_ids.scalar_type(), "moe_align_block_size_radix", [&] {
    constexpr int32_t key_threads = 256;
    // Conservative portable CUDA grid cap. The key-prep kernel uses a
    // grid-stride loop, so larger inputs are still processed correctly.
    constexpr int32_t max_key_grid_blocks = 65535;
    const int32_t key_blocks =
        std::min<int64_t>(CEILDIV(numel, key_threads), max_key_grid_blocks);
    vllm::moe::prepare_radix_sort_keys_kernel<scalar_t>
        <<<key_blocks, key_threads, 0, stream>>>(
            reinterpret_cast<const scalar_t*>(topk_ids.const_data_ptr()),
            reinterpret_cast<int32_t*>(topk_ids_for_sort.mutable_data_ptr()),
            expert_map_ptr, numel, num_experts, has_expert_map);
  });

  // CUB radix sort is stable. Monotonic input values therefore preserve
  // flattened route order within equal expert keys without a composite key.
  CubKeyValueSorter sorter(num_experts);
  sorter.run(
      sort_workspace.mutable_data_ptr(), sort_workspace.numel(),
      reinterpret_cast<const int32_t*>(topk_ids_for_sort.const_data_ptr()),
      reinterpret_cast<int32_t*>(sorted_expert_ids.mutable_data_ptr()),
      reinterpret_cast<const int32_t*>(token_indices.const_data_ptr()),
      reinterpret_cast<int32_t*>(compact_sorted_token_ids.mutable_data_ptr()),
      numel, stream);

  computeExpertOffsetsAndInverse(
      reinterpret_cast<const int32_t*>(sorted_expert_ids.const_data_ptr()),
      numel, num_experts,
      reinterpret_cast<int64_t*>(unpadded_expert_offsets.mutable_data_ptr()),
      /*sorted_rows=*/nullptr, /*inverse=*/nullptr, stream);

  vllm::moe::build_padded_metadata_from_offsets_kernel<<<2, 1024, 0, stream>>>(
      reinterpret_cast<int32_t*>(sorted_token_ids.mutable_data_ptr()),
      reinterpret_cast<int32_t*>(experts_ids.mutable_data_ptr()),
      reinterpret_cast<int32_t*>(num_tokens_post_pad.mutable_data_ptr()),
      reinterpret_cast<int32_t*>(padded_expert_offsets.mutable_data_ptr()),
      reinterpret_cast<const int64_t*>(unpadded_expert_offsets.const_data_ptr()),
      num_experts, block_size, numel, sorted_token_ids.size(0),
      experts_ids.size(0));

  vllm::moe::
      copy_compact_tokens_to_padded_kernel<<<num_experts, 256, 0, stream>>>(
          reinterpret_cast<const int32_t*>(
              compact_sorted_token_ids.const_data_ptr()),
          reinterpret_cast<int32_t*>(sorted_token_ids.mutable_data_ptr()),
          reinterpret_cast<const int32_t*>(padded_expert_offsets.const_data_ptr()),
          reinterpret_cast<const int64_t*>(
              unpadded_expert_offsets.const_data_ptr()),
          num_experts);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("moe_permute_sort_workspace_size", &moe_permute_sort_workspace_size);
  m.def("moe_align_block_size_stable_small", &moe_align_block_size_stable_small);
  m.def("moe_align_block_size_radix", &moe_align_block_size_radix);
}
