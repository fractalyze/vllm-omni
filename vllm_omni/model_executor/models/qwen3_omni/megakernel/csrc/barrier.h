// Copyright 2026 Fractalyze Inc. All rights reserved.

#ifndef S2MK_CSRC_BARRIER_H_
#define S2MK_CSRC_BARRIER_H_

#include <cuda_runtime.h>

#include <cstdint>

namespace s2mk {

// Words of zeroed device memory a launch's barriers use: the arrival counter
// and the watchdog's claim on the error record.
constexpr int kSyncWords = 2;

constexpr int32_t kErrorNone = 0;
constexpr int32_t kErrorBarrierTimeout = 1;

// Where a hung launch stopped, in host-mapped memory so it survives the trap.
// The layout matches the int32 tensor s2mk/barrier.py reads.
struct ErrorRecord {
  int32_t status;
  int32_t cta;  // the CTA whose watchdog fired
  int32_t barrier;  // launch-wide barrier index
  int32_t step;
  int32_t arrived;  // CTAs that had arrived at that barrier
  int32_t expected;
};

// The words each CTA of the barrier probe writes a round: one per thread.
constexpr int kProbeWordsPerCta = 512;

// Launches the barrier probe the tests run: `rounds` barriers on `num_ctas`
// CTAs, each round checking that every thread's write of the round is
// visible after the barrier and counting violations into `mismatches`. CTA
// `skip_cta` returns instead of arriving at barrier `skip_barrier`, which
// must trip the watchdog; -1 skips nothing.
struct BarrierProbeArgs {
  int rounds;
  int skip_cta;
  int skip_barrier;
  int step;
  int64_t timeout_ns;
  unsigned* sync;  // kSyncWords words
  ErrorRecord* error;  // device view of the host-mapped record
  int* slots;  // 2 × num_ctas × kProbeWordsPerCta
  unsigned* mismatches;
};

cudaError_t LaunchBarrierProbe(const BarrierProbeArgs& args, int num_ctas,
                               cudaStream_t stream);

// The device pointer of a host-mapped allocation.
cudaError_t MappedDevicePointer(void* host, void** device);

}  // namespace s2mk

#endif  // S2MK_CSRC_BARRIER_H_
