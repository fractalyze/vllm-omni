// Copyright 2026 Fractalyze Inc. All rights reserved.
//
// A grid barrier on one monotonic global counter. Barrier b of a launch is
// passed once the counter reaches (b + 1) × gridDim.x: each CTA arrives with
// a release add and waits with acquire loads, so the counter is zeroed once
// per launch and never reset between barriers. Every CTA must be resident at
// once, which the launcher guarantees with a cooperative launch.
//
// One thread arrives for the whole CTA, with no fence of its own. The
// __syncthreads() before the arrival orders every thread's stores before it,
// and a release is cumulative: it publishes whatever the releasing thread has
// observed, which includes those stores. The acquire load and the
// __syncthreads() after it hand them to every thread of every waiting CTA.
//
// Every wait carries a watchdog. A CTA that waits longer than the timeout
// claims the error record, writes where it stopped, and traps, so a missing
// arrival reports itself instead of wedging the GPU.

#ifndef S2MK_CSRC_BARRIER_CUH_
#define S2MK_CSRC_BARRIER_CUH_

#include <cuda_runtime.h>

#include <cstdint>

#include "barrier.h"

namespace s2mk {

class GridBarrier {
 public:
  // `sync` is kSyncWords zeroed words of device memory; `error` is host-mapped.
  __device__ GridBarrier(unsigned* sync, ErrorRecord* error, int64_t timeout_ns,
                         int step)
      : sync_(sync), error_(error), timeout_ns_(timeout_ns), step_(step) {}

  // Called by every thread of every CTA.
  __device__ void Sync() {
    __syncthreads();
    if (threadIdx.x == 0) {
      Arrive();
      Wait((index_ + 1) * gridDim.x);
    }
    ++index_;
    __syncthreads();
  }

  // The launch-wide index of the next barrier.
  __device__ int index() const { return index_; }

  // The decode step a watchdog report names from now on.
  __device__ void set_step(int step) { step_ = step; }

 private:
  __device__ void Arrive() {
    asm volatile("red.release.gpu.global.add.u32 [%0], 1;" ::"l"(sync_)
                 : "memory");
  }

  __device__ unsigned Arrivals() const {
    unsigned v;
    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];"
                 : "=r"(v)
                 : "l"(sync_)
                 : "memory");
    return v;
  }

  __device__ static int64_t Now() {
    int64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
  }

  __device__ void Wait(unsigned target) {
    const int64_t start = Now();
    unsigned arrived;
    while ((arrived = Arrivals()) < target) {
      if (Now() - start > timeout_ns_) Report(arrived, target);
    }
  }

  // One CTA writes the record; the rest trap without touching it.
  __device__ void Report(unsigned arrived, unsigned target) {
    if (atomicCAS(sync_ + 1, 0u, 1u) == 0u) {
      volatile ErrorRecord* rec = error_;
      rec->cta = static_cast<int32_t>(blockIdx.x);
      rec->barrier = index_;
      rec->step = step_;
      rec->arrived = static_cast<int32_t>(arrived - (target - gridDim.x));
      rec->expected = static_cast<int32_t>(gridDim.x);
      __threadfence_system();
      rec->status = kErrorBarrierTimeout;
      __threadfence_system();
    }
    __trap();
  }

  unsigned* sync_;
  ErrorRecord* error_;
  int64_t timeout_ns_;
  int step_;
  int index_ = 0;
};

}  // namespace s2mk

#endif  // S2MK_CSRC_BARRIER_CUH_
